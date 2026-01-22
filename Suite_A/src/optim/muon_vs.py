"""
Distributed Muon Optimizer with Variance-Stabilized Preprocessing (Muon_VS)

This implementation provides a distributed variant of the Muon optimizer that incorporates
variance-stabilized preprocessing inspired by variance reduction techniques in momentum-based
optimization. The key distinction from standard Muon is the integration of a second-moment
estimate (variance) into the preprocessing step before orthogonalization.

Reference implementations:
- Muon: https://github.com/KellerJordan/modded-nanogpt
- Distributed Muon: https://github.com/toothacher17/Megatron-LM/tree/moonshot/distributedmuon-impl
"""

import math
import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from .schedule import cos_inf_schedule, wsd_schedule


def zeropower_via_newtonschulz5(G, steps):
    """
    Compute the zeroth power (orthogonalization) of matrix G using Newton-Schulz iteration.
    
    This function applies a quintic (5th-order) Newton-Schulz iteration to compute the
    orthogonal component of G. The iteration coefficients are optimized to maximize the
    convergence slope at the origin, allowing stable computation in lower precision (bfloat16).
    
    Mathematically, this approximates the singular value decomposition U from G ≈ UΣV^T,
    computing U that satisfies U^T U ≈ I (orthogonal constraint).
    
    Args:
        G (torch.Tensor): A 2D matrix of shape (m, n) to be orthogonalized.
        steps (int): Number of Newton-Schulz iterations to perform.
    
    Returns:
        torch.Tensor: The orthogonalized matrix of the same shape as G.
    
    Note:
        The output approximates UV^T rather than exact UV^T, with singular values
        approximately uniformly distributed in [0.5, 1.5], which empirically shows
        minimal performance degradation compared to exact SVD.
    """
    assert len(G.shape) == 2, "Input tensor must be 2-dimensional"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Normalize spectral norm to be at most 1 for numerical stability
    X = X / (X.norm() + 1e-7)
    
    # Perform quintic Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def normalize_range(range: Tuple[int, int], start):
    """
    Adjust range indices relative to a new starting position.
    
    Args:
        range (Tuple[int, int]): Original range [start, end).
        start (int): New reference starting position.
    
    Returns:
        Tuple[int, int]: Adjusted range relative to the new start position.
    """
    return (range[0] - start, range[1] - start)


class MuonDistMeta:
    """
    Metadata structure for distributed parameter management in Muon optimizer.
    
    This class tracks the distributed layout of parameters across multiple processes
    and devices, including buffer assignment, spatial partitioning dimensions, and
    range information for both global and local perspectives.
    
    Attributes:
        buffer_idx (int): Index of the global buffer this parameter belongs to.
        bucket_idx (int): Index of the bucket within the buffer.
        shape (torch.Size): Parameter shape after tensor parallelism (TP) partitioning.
        global_range (Tuple[int, int]): [start, end) indices in the global flattened buffer.
        tp_split_dim (int): Tensor parallelism split dimension (-1 indicates no TP).
        local_range (Tuple[int, int]): [start, end) indices in the local buffer partition.
    """

    buffer_idx: int = 0
    bucket_idx: int = 0
    shape: torch.Size = None
    global_range: Tuple[int, int] = None
    tp_split_dim: int = -1
    local_range: Tuple[int, int] = None

    def __init__(
        self,
        buffer_idx: int,
        bucket_idx: int,
        shape: torch.Size,
        global_range: Tuple[int, int],
        tp_split_dim: int,
    ):
        """Initialize metadata for a distributed parameter."""
        self.buffer_idx = buffer_idx
        self.bucket_idx = bucket_idx
        self.shape = shape
        self.global_range = global_range
        self.tp_split_dim = tp_split_dim

    def set_local_buffer_range(self, local_buffer_range: Tuple[int, int]):
        """
        Set the local buffer range based on data parallelism partitioning.
        
        Args:
            local_buffer_range (Tuple[int, int]): The data parallel slice boundaries.
        """
        start = max(self.global_range[0], local_buffer_range[0])
        end = min(self.global_range[1], local_buffer_range[1])
        self.local_range = (
            (start, end)
            if start < end
            else (local_buffer_range[0], local_buffer_range[0])
        )


class DistributedMuon_VS(torch.optim.Optimizer):
    """
    Distributed Muon optimizer with Variance-Stabilized preprocessing.
    
    This optimizer combines momentum-based gradient updates with orthogonalization via
    Newton-Schulz iteration, enhanced with variance stabilization. The key innovation is
    preprocessing the momentum estimate with a variance term before orthogonalization,
    which stabilizes the optimization trajectory.
    
    The optimizer implements a hybrid two-group strategy:
    - Group 1: 2D parameters (≥2D with size[0]<10000) → Full-matrix Muon with variance preprocessing
    - Group 2: Embeddings, 1D parameters, and lm_head → AdamW fallback
    
    Key Features:
        - Variance-stabilized preprocessing: M_tilde = M_hat / sqrt(M_hat^2 + α*V)
        - Full-matrix orthogonalization via Newton-Schulz iteration
        - Supports distributed training (data parallelism + tensor parallelism)
        - Adaptive learning rate scaling based on parameter dimensions
        - Mixed precision support (bfloat16 for numerical stability)
    
    Arguments:
        param_groups (list): Parameter groups for optimization.
        lr (float): Base learning rate for full-matrix Muon updates. Default: 0.02
        weight_decay (float): L2 regularization coefficient. Default: 0.1
        matched_adamw_rms (float): Target RMS scaling factor for AdamW matching. Default: 0.2
        momentum (float): Momentum coefficient for first-order exponential moving average. Default: 0.95
        nesterov (bool): Enable Nesterov-style momentum acceleration. Default: True
        ns_steps (int): Number of Newton-Schulz iterations for orthogonalization. Default: 5
        adamw_betas (tuple): Exponential decay rates (β1, β2) for AdamW. Default: (0.95, 0.95)
        adamw_eps (float): Numerical stability term for AdamW. Default: 1e-8
    
    Example:
        >>> optimizer = DistributedMuon_VS(
        ...     param_groups=model.parameters(),
        ...     lr=0.02,
        ...     momentum=0.95,
        ...     ns_steps=5
        ... )
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()
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
    ):
        """
        Initialize the Distributed Muon_VS optimizer.
        
        Args:
            param_groups: List of parameter group dictionaries.
            lr: Learning rate for Muon updates.
            weight_decay: Coefficient for L2 weight decay regularization.
            matched_adamw_rms: Target RMS scaling for learning rate adjustment.
            momentum: Momentum coefficient in [0.9, 0.99].
            nesterov: Whether to apply Nesterov acceleration.
            ns_steps: Newton-Schulz iteration count (typically 5).
            adamw_betas: (β1, β2) for exponential moving averages in AdamW.
            adamw_eps: Small constant for numerical stability.
        """
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        super().__init__(param_groups, defaults)
        self.distributed_mode = False
        
        # Classify parameters by dimensionality and size for Muon vs AdamW selection
        for group in self.param_groups:
            for p in group["params"]:
                # Use Muon for high-dimensional parameters (embeddings excluded by size threshold)
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
        Enable distributed training mode with data and tensor parallelism support.
        
        This method configures the optimizer for distributed training, setting up
        buffer allocations for data-parallel (DP) and tensor-parallel (TP) updates.
        
        Args:
            global_buffer_sizes: Hierarchical buffer size information.
            dist_group: PyTorch process group for data parallelism.
            tp_group: PyTorch process group for tensor parallelism.
            dist_metas: Dictionary mapping parameters to their MuonDistMeta objects.
        """
        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.tp_group = tp_group
        self.dist_metas = dist_metas

        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)

        # Calculate local buffer ranges for this rank's data parallel shard
        self.local_buffer_sizes = []
        self.local_buffer_ranges = []
        for bucket_sizes in global_buffer_sizes:
            local_bucket_sizes = []
            local_bucket_ranges = []
            for global_bucket_size, bucket_offset in bucket_sizes:
                assert global_bucket_size % world_size == 0, \
                    f"Global buffer size {global_bucket_size} must be divisible by world size {world_size}"
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

        # Map each parameter to its local buffer range
        for dist_meta in dist_metas.values():
            local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                dist_meta.bucket_idx
            ]
            dist_meta.set_local_buffer_range(local_buffer_range)

        self.distributed_mode = True

    def adjust_lr_for_muon(self, lr, matched_adamw_rms, param_shape):
        """
        Adjust learning rate based on parameter dimensions using geometric scaling.
        
        This implements the theoretical result that the spectral norm of updates
        should scale with √(max(m,n)) to maintain consistent optimization behavior
        across parameters of different sizes.
        
        Args:
            lr (float): Base learning rate.
            matched_adamw_rms (float): Target matching coefficient.
            param_shape (torch.Size): Parameter tensor shape.
        
        Returns:
            float: Adjusted learning rate.
        
        Reference:
            This scaling follows the analysis in "Momentum Orthogonalization in 
            Manifold Spaces" and related work on adaptive learning rates for
            matrix parameters.
        """
        A, B = param_shape[:2]
        adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self):
        """
        Perform a single optimization step with variance-stabilized preprocessing.
        
        Algorithm outline:
        1. Update first-moment (M) and second-moment (V) exponential moving averages
        2. Apply bias correction to both moments
        3. Compute Nesterov-accelerated momentum: M_hat = (β/(1-β))*M + G
        4. Preprocess via variance stabilization: M_tilde = M_hat / sqrt(M_hat^2 + α*V)
        5. Orthogonalize via Newton-Schulz: O = NS5(M_tilde)
        6. Adjust learning rate by parameter dimensions
        7. Apply updates: W ← W - η*λ*W - η_adj*O (weight decay + orthogonalized update)
        8. For non-Muon parameters, apply standard AdamW with bias correction
        """
        dtype = torch.bfloat16
        device = torch.cuda.current_device()

        # Store preprocessing results for orthogonalization stage
        ns_inputs = {}

        ############################
        # Stage 1: Momentum Update  #
        # and Preprocessing         #
        ############################
        for group in self.param_groups:
            momentum = group["momentum"]
            eps = group["adamw_eps"]
            params = group["params"]

            one_minus_momentum = 1 - momentum
            nesterov_factor = momentum / one_minus_momentum
            variance_factor = momentum * one_minus_momentum

            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None, "Gradient is None for Muon-optimized parameter"
                # Non-distributed mode requires 2D gradients for matrix operations
                assert self.distributed_mode or g.dim() == 2, \
                    f"Expected 2D gradient in non-distributed mode, got {g.dim()}D"

                state = self.state[p]
                
                # Initialize momentum buffers if needed
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                    state["variance_buffer"] = torch.zeros_like(g)
                    state["muon_step"] = 0

                M = state["muon_buffer"]
                V = state["variance_buffer"]
                
                # Increment optimization step counter for bias correction
                state["muon_step"] += 1
                step = state["muon_step"]

                # Update first moment: M_t = β*M_{t-1} + (1-β)*G_t
                M.mul_(momentum).add_(g, alpha=one_minus_momentum)
                
                # Update second moment: V_t = β*V_{t-1} + β*(1-β)*(M_{t-1} - G_t)^2
                # This captures the variance of gradient changes
                V.mul_(momentum).addcmul_(M - g, M - g, value=variance_factor)

                # Compute bias correction factors
                # M uses update weight (1-β), V uses weight β(1-β)
                bias_correction_M = 1 - momentum ** step
                bias_correction_V = 1 - momentum ** step
                
                # Apply bias correction to denoise estimates
                M_corrected = M / bias_correction_M
                V_corrected = V / bias_correction_V

                # Nesterov acceleration: M_hat = (β/(1-β))*M_corrected + G_t
                M_hat = M_corrected * nesterov_factor + g
                
                # Variance-stabilized preprocessing: M_tilde = M_hat / sqrt(M_hat^2 + α*V)
                # This normalization prevents extreme magnitude variations
                M_tilde_snr = M_hat / (V_corrected).sqrt().add_(eps)

                # Store for orthogonalization stage
                ns_inputs[p] = M_tilde_snr.to(dtype)

        ########################################
        # Stage 2: Distributed Synchronization #
        # (Data Parallelism + Tensor Parallel) #
        ########################################
        if self.distributed_mode:
            # Build local and global buffers for communication
            def build_buffers():
                """Allocate temporary buffers for collective communication."""
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

            # Scatter local parameters into buffers for all-gather
            def fill_local_buffers(tensor_dict, local_bufs):
                """Pack local parameter updates into communication buffers."""
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

            # Synchronize across data parallel group
            def all_gather_dp(global_bufs, local_bufs):
                """Collective all-gather operation along data parallelism dimension."""
                for gbufs, lbufs in zip(global_bufs, local_bufs):
                    for gb, lb in zip(gbufs, lbufs):
                        dist.all_gather_into_tensor(
                            gb,
                            lb,
                            group=self.dist_group,
                        )

            all_gather_dp(ns_global_bufs, ns_local_bufs)

            # Reconstruct full parameters from global buffers
            def restore_global(tensor_dict, global_bufs):
                """Unpack globally synchronized buffers back to parameter view."""
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

            # Extract tensor parallelism configuration
            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        #######################################
        # Stage 3: Orthogonalization and      #
        # Parameter Update                    #
        #######################################
        for group in self.param_groups:
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            params = group["params"]

            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue

                ns_input = ns_inputs[p]  # M_tilde in preprocessing stage reference frame

                tp_split_dim = -1
                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim

                # Synchronize across tensor parallel group if needed
                if tp_split_dim != -1:
                    def tp_gather_cat(x):
                        """Gather tensor parallel shards and concatenate."""
                        shards = [torch.empty_like(x) for _ in range(tp_world_size)]
                        dist.all_gather(shards, x, self.tp_group)
                        return torch.cat(shards, dim=tp_split_dim)

                    ns_input = tp_gather_cat(ns_input)

                # Apply Newton-Schulz orthogonalization to preprocessed momentum
                O_full = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                # Shard orthogonalized update back to local tensor parallel rank
                if tp_split_dim != -1:
                    O_tp_local = O_full.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]
                else:
                    O_tp_local = O_full

                # Apply data parallel slicing if in distributed mode
                if self.distributed_mode:
                    local_range_in_global_range = normalize_range(
                        dist_meta.local_range, dist_meta.global_range[0]
                    )
                    start, end = local_range_in_global_range

                    def dp_slice(x):
                        """Extract data parallel local slice from global tensor."""
                        flat = x.reshape(-1)
                        sliced = flat[start:end]
                        return sliced.view_as(self.state[p]["muon_buffer"])

                    O_local = dp_slice(O_tp_local)
                else:
                    O_local = O_tp_local

                # Scale learning rate based on parameter shape
                adjusted_lr = self.adjust_lr_for_muon(lr, 0.2, O_local.shape)

                # Apply weight decay and orthogonalized update
                # W_{t+1} = W_t - η*λ*W_t - η_adj*O_t
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(O_local, alpha=-adjusted_lr)

        ############################
        # Stage 4: AdamW Fallback  #
        # for Low-Dim Parameters   #
        ############################
        for group in self.param_groups:
            # Initialize or increment global step counter
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
                    continue  # Skip Muon-optimized parameters

                g = p.grad
                assert g is not None, "Gradient is None for AdamW-optimized parameter"
                state = self.state[p]

                # Initialize AdamW state buffers
                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                buf1 = state["adamw_exp_avg"]
                buf2 = state["adamw_exp_avg_sq"]
                
                # Update biased first and second moment estimates
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                # Compute bias-corrected estimates
                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                
                # Apply weight decay (L2 regularization)
                p.data.mul_(1 - lr * weight_decay)
                
                # Apply scaled AdamW update
                p.data.add_(g, alpha=-lr / scale)