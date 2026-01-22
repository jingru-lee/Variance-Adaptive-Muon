"""
Here is an original implementation of Muon and Distributed Muon. 
_Note_ that 'class Muon'  is an old version which does not include weight decay for matrix params
the newer version with weight decay is in the 'class DistributedMuon'.
Source: https://github.com/KellerJordan/modded-nanogpt
Source: https://github.com/toothacher17/Megatron-LM/tree/moonshot/distributedmuon-impl
"""

import math
import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from .schedule import cos_inf_schedule, wsd_schedule


# copy from https://github.com/KellerJordan/Muon/tree/master
# @torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def normalize_range(range: Tuple[int, int], start):
    return (range[0] - start, range[1] - start)


class MuonDistMeta:
    # Index of the buffer this parameter belongs to
    buffer_idx: int = 0
    # Index of the bucket this parameter belongs to
    bucket_idx: int = 0
    # Parameter shape after tensor parallelism
    shape: torch.Size = None
    # Parameter location in global buffer
    global_range: Tuple[int, int] = None
    tp_split_dim: int = -1
    # Parameter location in global buffer (current data parallel slice)
    local_range: Tuple[int, int] = None

    def __init__(
        self,
        buffer_idx: int,
        bucket_idx: int,
        shape: torch.Size,
        global_range: Tuple[int, int],
        tp_split_dim: int,
    ):
        self.buffer_idx = buffer_idx
        self.bucket_idx = bucket_idx
        self.shape = shape
        self.global_range = global_range
        self.tp_split_dim = tp_split_dim

    def set_local_buffer_range(self, local_buffer_range: Tuple[int, int]):
        start = max(self.global_range[0], local_buffer_range[0])
        end = min(self.global_range[1], local_buffer_range[1])
        self.local_range = (
            (start, end)
            if start < end
            else (local_buffer_range[0], local_buffer_range[0])
        )


# # adjust LR based on: https://github.com/MoonshotAI/Moonlight
# def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
#     A, B = param_shape[:2]
#     adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
#     adjusted_lr = lr * adjusted_ratio
#     return adjusted_lr


class DistributedMuon_NSR(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of . (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        param_groups,
        lr=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        alpha=1000.0,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            alpha=alpha,
        )

        super().__init__(param_groups, defaults)
        self.distributed_mode = False
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for group in self.param_groups:
            for p in group["params"]:
                # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
                if p.ndim >= 2 and p.size(0) < 10000:
                    self.state[p]["use_muon"] = True
                else:
                    self.state[p]["use_muon"] = False

    def enable_distributed_mode(
        self,
        global_buffer_sizes,
        dist_group,
        tp_group,
        dist_metas: Dict[torch.nn.Parameter, MuonDistMeta],
    ):
        """
        Enable distributed mode for the optimizer.
        
        Args:
            global_buffer_size: Global buffer size for gradient aggregation
            dist_group: Optimizer sharding group for data parallelism
            tp_group: Tensor parallelism group for parameter sharding
            dist_metas: Distribution metadata for all parameters
        """

        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.tp_group = tp_group
        self.dist_metas = dist_metas

        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)

        # Calculate local buffer range for each rank
        self.local_buffer_sizes = []
        self.local_buffer_ranges = []
        for bucket_sizes in global_buffer_sizes:
            local_bucket_sizes = []
            local_bucket_ranges = []
            for global_bucket_size, bucket_offset in bucket_sizes:
                assert global_bucket_size % world_size == 0
                local_buffer_size = global_bucket_size // world_size
                local_buffer_start = local_buffer_size * rank + bucket_offset
                local_buffer_range = (
                    local_buffer_start,
                    local_buffer_start + local_buffer_size,
                )
                local_bucket_sizes.append(local_buffer_size)
                local_bucket_ranges.append(local_buffer_range)

            self.local_buffer_sizes.append(local_bucket_sizes)
            self.local_buffer_ranges.append(local_bucket_ranges)

        # Calculate local range for each parameter
        for dist_meta in dist_metas.values():
            local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                dist_meta.bucket_idx
            ]
            dist_meta.set_local_buffer_range(local_buffer_range)

        self.distributed_mode = True

    def adjust_lr_for_muon(self, lr, matched_adamw_rms, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self):
        dtype = torch.bfloat16
        device = torch.cuda.current_device()

        # ns_inputs: M_hat (denoised Nesterov momentum) for orthogonalization
        ns_inputs = {}

        ########################################
        #   Update M / V, then compute         #
        #   M_tilde_snr and M_hat              #
        ########################################
        for group in self.param_groups:
            momentum = group["momentum"]
            alpha = group["alpha"]
            eps = group["adamw_eps"]
            params = group["params"]

            one_minus_momentum = 1 - momentum
            nesterov_factor = momentum / one_minus_momentum
            variance_factor = momentum * one_minus_momentum

            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                # In non-distributed mode, gradient must be 2D, consistent with original implementation
                assert self.distributed_mode or g.dim() == 2

                state = self.state[p]
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                    state["variance_buffer"] = torch.zeros_like(g)
                    state["muon_step"] = 0  # Step counter for bias correction

                M = state["muon_buffer"]
                V = state["variance_buffer"]
                
                # Update step counter
                state["muon_step"] += 1
                step = state["muon_step"]

                # 1. Update M_t and V_t
                # V_t = beta * V_{t-1} + beta * (1 - beta) * (M_{t-1} - G_t)^2
                V.mul_(momentum).addcmul_(M - g, M - g, value=variance_factor)
                # M_t = beta * M_{t-1} + (1 - beta) * G_t
                M.mul_(momentum).add_(g, alpha=one_minus_momentum)

                # 2. Compute correct bias correction factors
                # First moment M: update weight is (1-beta), bias correction factor is (1 - beta^t)
                # Second moment V: update weight is beta*(1-beta), bias correction factor is beta*(1 - beta^t)
                bias_correction_M = 1 - momentum ** step
                bias_correction_V = 1 - momentum ** step
                
                # 3. Apply bias correction
                M_corrected = M / bias_correction_M
                V_corrected = V / bias_correction_V

                # 4. Compute denoised momentum for Muon: M_hat = (beta/(1-beta)) * M_corrected + G_t
                M_hat = M_corrected * nesterov_factor + g
                
                # 5. Adam-style preconditioning: M_tilde_snr = M_hat / sqrt(M_hat^2 + alpha * V_corrected) (element-wise)
                M_tilde_snr = M_hat / (M_hat.square() + alpha * V_corrected).sqrt().add_(eps)

                # Store M_tilde_snr for subsequent orthogonalization
                ns_inputs[p] = M_tilde_snr.to(dtype)

        ########################################
        #   Distributed: Redistribute M_hat    #
        #   across data parallel dimension     #
        ########################################
        if self.distributed_mode:
            # Build local/global buffers for ns_input
            def build_buffers():
                local_bufs = [
                    [
                        torch.empty((local_buffer_size), device=device, dtype=dtype)
                        for local_buffer_size in local_bucket_sizes
                    ]
                    for local_bucket_sizes in self.local_buffer_sizes
                ]
                global_bufs = [
                    [
                        torch.empty((global_buffer_size), device=device, dtype=dtype)
                        for (global_buffer_size, bucket_offset) in global_bucket_sizes
                    ]
                    for global_bucket_sizes in self.global_buffer_sizes
                ]
                return local_bufs, global_bufs

            ns_local_bufs, ns_global_bufs = build_buffers()

            # Flatten each tensor and fill into corresponding local buffer
            def fill_local_buffers(tensor_dict, local_bufs):
                for param, t in tensor_dict.items():
                    dist_meta = self.dist_metas[param]
                    local_buf = local_bufs[dist_meta.buffer_idx][dist_meta.bucket_idx]
                    local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                        dist_meta.bucket_idx
                    ]
                    local_range = normalize_range(
                        dist_meta.local_range, local_buffer_range[0]
                    )
                    local_buf[local_range[0] : local_range[1]].copy_(t.view(-1))

            fill_local_buffers(ns_inputs, ns_local_bufs)

            # all_gather_into_tensor: Aggregate across data parallel dimension
            def all_gather_dp(global_bufs, local_bufs):
                for gbufs, lbufs in zip(global_bufs, local_bufs):
                    for gb, lb in zip(gbufs, lbufs):
                        dist.all_gather_into_tensor(
                            gb,
                            lb,
                            group=self.dist_group,
                        )

            all_gather_dp(ns_global_bufs, ns_local_bufs)

            # Restore tensors with global view after tensor parallelism from global buffer
            def restore_global(tensor_dict, global_bufs):
                for p in tensor_dict.keys():
                    dist_meta = self.dist_metas[p]
                    gbuf = global_bufs[dist_meta.buffer_idx][dist_meta.bucket_idx]
                    global_range = dist_meta.global_range
                    offset = self.global_buffer_sizes[dist_meta.buffer_idx][
                        dist_meta.bucket_idx
                    ][1]
                    tensor_dict[p] = gbuf[
                        global_range[0] - offset : global_range[1] - offset
                    ].view(dist_meta.shape)

            restore_global(ns_inputs, ns_global_bufs)

            # Tensor parallelism related information
            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        #######################################
        #   Compute O_t and apply updates     #
        #######################################
        for group in self.param_groups:
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            params = group["params"]

            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue

                ns_input = ns_inputs[p]  # M_tilde_snr in post-DP, pre-TP view

                tp_split_dim = -1
                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim

                # If using TP, gather M_tilde_snr across TP dimension to get full TP view
                if tp_split_dim != -1:
                    # Perform all_gather + concatenation on ns_input
                    def tp_gather_cat(x):
                        shards = [torch.empty_like(x) for _ in range(tp_world_size)]
                        dist.all_gather(shards, x, self.tp_group)
                        return torch.cat(shards, dim=tp_split_dim)

                    ns_input = tp_gather_cat(ns_input)

                # 4. O_t = NS5(M_tilde_snr)
                O_full = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                # If using TP, chunk O_full back to current TP rank
                if tp_split_dim != -1:
                    O_tp_local = O_full.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]
                else:
                    O_tp_local = O_full

                # If distributed_mode=True, slice according to DP local_range
                if self.distributed_mode:
                    local_range_in_global_range = normalize_range(
                        dist_meta.local_range, dist_meta.global_range[0]
                    )
                    start, end = local_range_in_global_range

                    def dp_slice(x):
                        # Same as original implementation: flatten, slice, reshape to muon_buffer shape
                        flat = x.reshape(-1)
                        sliced = flat[start:end]
                        return sliced.view_as(self.state[p]["muon_buffer"])

                    O_local = dp_slice(O_tp_local)
                else:
                    # In non-distributed mode, shape matches original 2D parameter
                    O_local = O_tp_local

                # Adjust learning rate using Frobenius norm of local O
                adjusted_lr = self.adjust_lr_for_muon(lr, 0.2, O_local.shape)

                # Weight decay: W_{t+1} = W_t - eta * lambda * W_t - eta_hat * O_t
                p.data.mul_(1 - lr * weight_decay)
                # Apply update (only update the slice corresponding to current rank)
                p.data.add_(O_local, alpha=-adjusted_lr)

        ############################
        #   AdamW fallback section #
        ############################
        for group in self.param_groups:
            # Initialize step counter
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            step = group["step"]
            params = group["params"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:
                if self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                state = self.state[p]

                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                buf1 = state["adamw_exp_avg"]
                buf2 = state["adamw_exp_avg_sq"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)