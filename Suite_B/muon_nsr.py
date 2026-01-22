import math
import torch
from torch.nn.utils import clip_grad_norm_

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py

@torch.compile
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
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-5)
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


class Muon_NSR(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz (Extended Version)

    This version supports two optimization strategies:
    1. Full-matrix Muon for 2D non-embedding parameters
    2. AdamW with adamw_lr for embedding parameters, 1D parameters and lm_head

    Arguments:
        muon_params: The 2D non-embedding parameters optimized by full-matrix Muon.
        adamw_params: The embedding parameters, 1D parameters and lm_head optimized by AdamW.
        lr: The learning rate for full-matrix Muon. (0.02 is a good default)
        wd: The weight decay for Muon.
        momentum: The momentum used for full-matrix Muon. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum. (recommended)
        ns_steps: The number of Newton-Schulz iterations. (5 is probably always enough)
        adamw_lr: The learning rate for AdamW (embedding, 1D parameters and lm_head).
        adamw_wd: The weight decay for AdamW.
        adamw_betas: The betas for AdamW.
        adamw_eps: The epsilon for AdamW.
        g_norm: The gradient norm threshold for gradient clipping.
        alpha: The alpha parameter for variance scaling in the Adam-style preprocessing.
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        lr=0.02,
        wd=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_lr=6e-4,
        adamw_wd=0.1,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        alpha=1000.0,
        g_norm=1.0,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_wd=adamw_wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            alpha=alpha,
            g_norm=g_norm,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Classify parameters into two groups
        for p in muon_params:
            # Full-matrix Muon for 2D non-embedding parameters
            assert p.ndim == 2, f"Muon parameters must be 2D, but got {p.ndim}"
            self.state[p]["opt_type"] = "muon"
        
        for p in adamw_params:
            # AdamW for embedding, 1D parameters and lm_head
            self.state[p]["opt_type"] = "adamw"

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = (max(1, A / B)) ** 0.5
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def _clip_grad(self, g, g_norm):
        """
        Apply gradient clipping using the formula:
        g_hat = g * max{1, g_norm / ||g||_2}
        
        Args:
            g: The gradient tensor
            g_norm: The gradient norm threshold
            
        Returns:
            Clipped gradient tensor
        """
        grad_norm = g.norm(2)
        clip_coef = g_norm / (grad_norm + 1e-8)
        # max{1, g_norm / ||g||_2}
        clip_coef = torch.clamp(clip_coef, min=1.0)
        return g * clip_coef

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            # Segregate parameters into two groups
            muon_p = [p for p in group["params"] if p.grad is not None and self.state[p]["opt_type"] == "muon"]
            adamw_p = [p for p in group["params"] if p.grad is not None and self.state[p]["opt_type"] == "adamw"]

            g_norm = group["g_norm"]

            ############################
            #    Full-Matrix Muon      #
            ############################
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            alpha = group["alpha"]
            eps = group["adamw_eps"]
            
            # Precompute constants for efficiency
            one_minus_momentum = 1 - momentum
            nesterov_factor = momentum / one_minus_momentum
            variance_factor = momentum * one_minus_momentum

            for p in muon_p:
                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # Apply gradient clipping: g_hat = g * max{1, g_norm / ||g||_2}
                g = self._clip_grad(g, g_norm)

                state = self.state[p]
                
                # Initialize state buffers
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["variance_buffer"] = torch.zeros_like(g)
                    state["muon_step"] = 0  # Step counter for bias correction
                
                # Get buffers
                M = state["momentum_buffer"]
                V = state["variance_buffer"]
                
                # Update step count
                state["muon_step"] += 1
                step = state["muon_step"]
                
                # 1. Update M_t, V_t
                # V_t = β*V_{t-1} + β*(1-β)*(M_{t-1} - G_t)^2
                V.mul_(momentum).addcmul_(M - g, M - g, value=variance_factor)
                # M_t = β*M_{t-1} + (1-β)*G_t
                M.mul_(momentum).add_(g, alpha=one_minus_momentum)

                # 2. Compute bias correction factors
                # First moment M: update weight is (1-β), bias correction factor is (1 - β^t)
                # Second moment V: update weight is β(1-β), bias correction factor is β(1 - β^t)
                bias_correction_M = 1 - momentum ** step
                bias_correction_V = 1 - momentum ** step
                
                # 3. Apply bias correction
                M_corrected = M / bias_correction_M
                V_corrected = V / bias_correction_V

                # 4. Compute Nesterov-style momentum: M_hat = (β/(1-β)) * M_corrected + G_t
                M_hat = M_corrected * nesterov_factor + g
                
                # 5. Adam-style preprocessing: M_tilde_snr = M_hat / sqrt(M_hat^2 + alpha * V_corrected)
                M_tilde_snr = M_hat / (M_hat.square() + alpha * V_corrected).sqrt().add_(1e-8)

                # 6. Orthogonalization: O_t = NS5(M_tilde_snr)
                u = zeropower_via_newtonschulz5(M_tilde_snr, steps=group["ns_steps"])

                # Scale update (using Kimi scaling)
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # Apply weight decay BEFORE update
                p.data.mul_(1 - lr * wd)
                
                # Apply scaled orthogonalized update AFTER weight decay
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #    AdamW for all other   #
            #    params (embed + 1D +  #
            #    lm_head)              #
            ############################
            adamw_lr = group['adamw_lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            adamw_wd = group["adamw_wd"]

            for p in adamw_p:
                g = p.grad

                # Apply gradient clipping: g_hat = g * max{1, g_norm / ||g||_2}
                g = self._clip_grad(g, g_norm)

                state = self.state[p]

                # Adam state initialization
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                # Adam update calculation
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                
                # Apply weight decay BEFORE update
                p.data.mul_(1 - adamw_lr * adamw_wd)
                
                # Apply Adam update AFTER weight decay
                p.data.add_(g, alpha=-adamw_lr / scale)

        return loss