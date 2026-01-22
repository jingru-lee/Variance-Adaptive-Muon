"""
Muon Optimizer: Momentum Orthogonalized by Newton-Schulz.

This module implements the Muon optimizer, which combines momentum-based gradient 
updates with orthogonalization via Newton-Schulz iteration. The optimizer supports
a hybrid two-group strategy where 2D parameters use full-matrix orthogonalization
while embeddings and 1D parameters use standard AdamW.

Reference:
    - Muon: https://github.com/KellerJordan/Muon/blob/master/muon.py
    - Moonlight: https://github.com/MoonshotAI/Moonlight
"""

import math
import torch
from torch.nn.utils import clip_grad_norm_


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Compute the zeroth power (orthogonalization) of matrix G via Newton-Schulz iteration.
    
    This function uses a quintic Newton-Schulz iteration with coefficients optimized
    to maximize the slope at zero. The iteration produces an approximate orthogonal
    matrix US'V^T where S' is diagonal with entries approximately in [0.5, 1.5],
    which empirically does not hurt model performance compared to exact UV^T.
    
    Args:
        G: Input matrix of shape (m, n) to be orthogonalized.
        steps: Number of Newton-Schulz iterations to perform.
        
    Returns:
        Orthogonalized matrix of the same shape as G.
        
    Note:
        The iteration is performed in bfloat16 for numerical stability and efficiency.
        For tall matrices (m > n), the computation is performed on the transpose.
    """
    assert len(G.shape) == 2, "Input must be a 2D matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    
    # Handle tall matrices by transposing
    if G.size(0) > G.size(1):
        X = X.T
        
    # Normalize to ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-5)
    
    # Perform Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if input was tall
    if G.size(0) > G.size(1):
        X = X.T
        
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-Schulz (Extended Version).
    
    This optimizer implements a hybrid optimization strategy:
        1. Full-matrix Muon for 2D non-embedding parameters: Uses momentum with
           Newton-Schulz orthogonalization for spectral norm control.
        2. AdamW for embedding parameters, 1D parameters, and lm_head: Uses
           standard Adam with decoupled weight decay.
    
    The key innovation is applying orthogonalization to the momentum-accumulated
    gradients, which helps maintain stable spectral properties during training.
    
    Args:
        muon_params: Parameters to optimize with full-matrix Muon (must be 2D).
        adamw_params: Parameters to optimize with AdamW (embeddings, 1D, lm_head).
        lr: Learning rate for Muon parameters. Default: 0.02
        wd: Weight decay coefficient for Muon parameters. Default: 0.0
        momentum: Momentum coefficient for Muon. Default: 0.95
        nesterov: Whether to use Nesterov momentum. Default: True
        ns_steps: Number of Newton-Schulz iterations. Default: 5
        adamw_lr: Learning rate for AdamW parameters. Default: 6e-4
        adamw_wd: Weight decay for AdamW parameters. Default: 0.1
        adamw_betas: Beta coefficients for AdamW. Default: (0.9, 0.95)
        adamw_eps: Epsilon for numerical stability. Default: 1e-8
        g_norm: Gradient norm threshold for clipping. Default: 1.0
        
    Example:
        >>> muon_params, adamw_params = classify_parameters(model)
        >>> optimizer = Muon(
        ...     muon_params=muon_params,
        ...     adamw_params=adamw_params,
        ...     lr=0.02,
        ...     momentum=0.95
        ... )
        
    Reference:
        https://github.com/KellerJordan/Muon
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
        g_norm=1.0,
    ):
        """Initialize the Muon optimizer with parameter groups and hyperparameters."""
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
            g_norm=g_norm,
        )

        # Combine parameter lists
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Classify parameters into optimization groups
        for p in muon_params:
            assert p.ndim == 2, f"Muon parameters must be 2D, got {p.ndim}D"
            self.state[p]["opt_type"] = "muon"
        
        for p in adamw_params:
            self.state[p]["opt_type"] = "adamw"

    def adjust_lr_for_muon(self, lr, param_shape):
        """
        Adjust learning rate based on parameter matrix dimensions.
        
        This scaling follows the Moonlight paper's recommendation to adjust
        the learning rate proportionally to sqrt(max(1, A/B)) where A and B
        are the matrix dimensions. This helps maintain consistent update
        magnitudes across parameters of different aspect ratios.
        
        Args:
            lr: Base learning rate.
            param_shape: Shape of the parameter tensor.
            
        Returns:
            Adjusted learning rate.
        """
        A, B = param_shape[:2]
        adjusted_ratio = (max(1, A / B)) ** 0.5
        return lr * adjusted_ratio

    def _clip_grad(self, g, g_norm):
        """
        Apply gradient clipping with the formula: g_hat = g * min(1, g_norm / ||g||_2).
        
        This implements a soft clipping that scales gradients when their norm
        exceeds the threshold, rather than hard truncation.
        
        Args:
            g: Gradient tensor to clip.
            g_norm: Maximum allowed gradient norm.
            
        Returns:
            Clipped gradient tensor.
        """
        grad_norm = g.norm(2)
        clip_coef = g_norm / (grad_norm + 1e-8)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        return g * clip_coef

    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        This method implements the hybrid Muon-AdamW update:
            1. For Muon parameters: Apply momentum, orthogonalize via Newton-Schulz,
               then apply weight decay and scaled update.
            2. For AdamW parameters: Apply standard Adam with decoupled weight decay.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
                    Optional for most use cases.
                    
        Returns:
            Loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            # Separate parameters by optimization type
            muon_p = [p for p in group["params"] 
                      if p.grad is not None and self.state[p]["opt_type"] == "muon"]
            adamw_p = [p for p in group["params"] 
                       if p.grad is not None and self.state[p]["opt_type"] == "adamw"]

            g_norm = group["g_norm"]

            # ============================
            # Full-Matrix Muon Update
            # ============================
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in muon_p:
                g = p.grad
                
                # Flatten higher-dimensional gradients to 2D
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # Apply gradient clipping
                g = self._clip_grad(g, g_norm)

                # Momentum accumulation
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                # Apply Nesterov momentum if enabled
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                # Orthogonalize via Newton-Schulz iteration
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # Compute dimension-adjusted learning rate
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # Apply decoupled weight decay
                p.data.mul_(1 - lr * wd)
                
                # Apply orthogonalized update
                p.data.add_(u, alpha=-adjusted_lr)

            # ============================
            # AdamW Update for Embeddings
            # ============================
            adamw_lr = group['adamw_lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            adamw_wd = group["adamw_wd"]

            for p in adamw_p:
                g = p.grad

                # Apply gradient clipping
                g = self._clip_grad(g, g_norm)

                state = self.state[p]

                # Initialize Adam state
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                # Update biased first and second moment estimates
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                # Compute bias-corrected update direction
                g = buf1 / (eps + buf2.sqrt())

                # Compute bias correction factors
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                
                # Apply decoupled weight decay
                p.data.mul_(1 - adamw_lr * adamw_wd)
                
                # Apply Adam update with bias correction
                p.data.add_(g, alpha=-adamw_lr / scale)

        return loss