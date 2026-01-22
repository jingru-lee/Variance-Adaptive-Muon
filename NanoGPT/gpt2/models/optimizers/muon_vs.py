import os
import math
import torch
import torch.distributed as dist
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration for computing the orthogonalization (zeroth power) of matrix G.
    
    This implementation uses a quintic iteration with coefficients optimized to maximize 
    the slope at zero. The iteration produces an approximation US'V^T where S' is diagonal 
    with S_{ii}' approximately uniformly distributed in [0.5, 1.5], which empirically shows 
    minimal performance degradation compared to exact SVD orthogonalization UV^T.
    
    Args:
        G: Input tensor of shape (..., M, N) where M, N are matrix dimensions.
        steps: Number of Newton-Schulz iterations to perform.
        
    Returns:
        Orthogonalized tensor of the same shape as input G.
        
    Reference:
        Adapted from https://github.com/KellerJordan/Muon
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize spectral norm to at most 1 for numerical stability
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform quintic Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon_VS(torch.optim.Optimizer):
    """
    Muon optimizer with Variance-Stabilized preprocessing (Muon_VS).
    
    This optimizer combines momentum-based gradient updates with orthogonalization via
    Newton-Schulz iteration, enhanced with variance stabilization. The key innovation is
    preprocessing the momentum estimate with a variance term before orthogonalization,
    which stabilizes the optimization trajectory.
    
    Algorithm:
        1. Update first moment: M_t = β*M_{t-1} + (1-β)*G_t
        2. Update second moment: V_t = β*V_{t-1} + β*(1-β)*(M_{t-1} - G_t)^2
        3. Apply bias correction to both moments
        4. Compute Nesterov-accelerated momentum: M_hat = (β/(1-β))*M_corrected + G_t
        5. Variance-stabilized preprocessing: M_tilde = M_hat / sqrt(V_corrected + eps)
        6. Orthogonalize via Newton-Schulz: O = NS5(M_tilde)
        7. Apply scaled update with learning rate adjustment based on parameter shape
    
    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate for parameter updates. Default: 0.02
        weight_decay: L2 regularization coefficient. Default: 0.01
        momentum: Momentum coefficient for exponential moving averages. Default: 0.95
        nesterov: Whether to use Nesterov-style momentum acceleration. Default: True
        ns_steps: Number of Newton-Schulz iterations for orthogonalization. Default: 5
        eps: Small constant for numerical stability in variance normalization. Default: 1e-8
        alpha: Scaling factor for variance term (unused in current implementation). Default: 10.0
        rank: Distributed training rank. Required for multi-GPU training.
        world_size: Total number of distributed processes. Required for multi-GPU training.
        
    Raises:
        Exception: If rank or world_size is None. For single GPU, pass rank=0 and world_size=1.
        
    Reference:
        - Muon: https://github.com/KellerJordan/modded-nanogpt
        - Variance stabilization inspired by Adam's second moment estimation
    """
    
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, 
                 ns_steps=5, eps=1e-8, alpha=10.0, rank=None, world_size=None):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this "
                          "optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, 
                       nesterov=nesterov, ns_steps=ns_steps, eps=eps, alpha=alpha)
        params: list[Tensor] = [*params]
        param_groups = []
        
        # Group parameters by size for efficient distributed communication
        for size in {p.numel() for p in params}:
            buf = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=buf, 
                update_buffer_views=[buf[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    def adjust_lr_for_muon(self, lr, matched_adamw_rms, param_shape):
        """
        Adjust learning rate based on parameter matrix dimensions.
        
        This scaling follows the Moonlight paper's recommendation to adjust the 
        learning rate proportionally to the square root of the maximum dimension,
        which helps maintain consistent update magnitudes across parameters of 
        different sizes.
        
        Args:
            lr: Base learning rate.
            matched_adamw_rms: Target RMS scaling factor for AdamW matching.
            param_shape: Shape of the parameter tensor.
            
        Returns:
            Adjusted learning rate scaling ratio.
            
        Reference:
            https://github.com/MoonshotAI/Moonlight
        """
        A, B = param_shape[:2]
        adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
        return adjusted_ratio
        
    @torch.no_grad()
    def step(self):
        """
        Perform a single optimization step with variance-stabilized preprocessing.
        
        This method implements the full Muon_VS algorithm:
        1. Compute and update momentum and variance estimates
        2. Apply bias correction for both estimates
        3. Compute Nesterov-accelerated momentum
        4. Apply variance-stabilized preprocessing
        5. Orthogonalize via Newton-Schulz iteration
        6. Scale learning rate by parameter dimensions
        7. Synchronize updates across distributed workers
        8. Apply weight decay and parameter updates
        """
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            eps = group["eps"]
            momentum = group["momentum"]
            alpha = group["alpha"]
            lr = group["lr"]
            handle = None
            params_world = None
            
            # Precompute constants for computational efficiency
            one_minus_momentum = 1 - momentum
            nesterov_factor = momentum / one_minus_momentum
            variance_factor = momentum * one_minus_momentum

            def update_prev():
                """Apply parameter updates after distributed synchronization."""
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    # Apply weight decay (L2 regularization)
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    # Apply orthogonalized gradient update
                    p_world.add_(g_world.view_as(p_world), alpha=-group["lr"])

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None

                    state = self.state[p]
                    
                    # Store original shape for learning rate adjustment
                    original_shape = p.shape

                    # Reshape high-dimensional tensors (e.g., conv filters) to 2D
                    if g.ndim == 4:
                        g = g.view(g.size(0), -1)
                    elif g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    # Initialize state buffers on first iteration
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["variance_buffer"] = torch.zeros_like(g)
                        state["step"] = 0  # Step counter for bias correction

                    # Retrieve state buffers
                    M = state["momentum_buffer"]
                    V = state["variance_buffer"]
                    
                    # Increment step counter
                    state["step"] += 1
                    step = state["step"]
                    
                    # Step 1: Update momentum (M) and variance (V) estimates
                    # V_t = β*V_{t-1} + β*(1-β)*(M_{t-1} - G_t)^2
                    # This captures the variance of gradient prediction errors
                    V.mul_(momentum).addcmul_(M - g, M - g, value=variance_factor)
                    # M_t = β*M_{t-1} + (1-β)*G_t
                    M.mul_(momentum).add_(g, alpha=one_minus_momentum)

                    # Step 2: Compute bias correction factors
                    # First moment M: update weight is (1-β), bias correction factor is (1 - β^t)
                    # Second moment V: update weight is β(1-β), bias correction factor is β(1 - β^t)
                    bias_correction_M = 1 - momentum ** step
                    bias_correction_V = 1 - momentum ** step
                    
                    # Step 3: Apply bias correction to denoise estimates
                    M_corrected = M / bias_correction_M
                    V_corrected = V / bias_correction_V

                    # Step 4: Compute Nesterov-accelerated momentum
                    # M_hat = (β/(1-β)) * M_corrected + G_t
                    M_hat = M_corrected * nesterov_factor + g
                    
                    # Step 5: Variance-stabilized preprocessing
                    # M_tilde = M_hat / sqrt(V_corrected + eps)
                    # This normalization prevents extreme magnitude variations
                    M_tilde_snr = M_hat / (V_corrected).sqrt().add_(eps)

                    # Step 6: Orthogonalize via Newton-Schulz iteration
                    # O_t = NS5(M_tilde)
                    O = zeropower_via_newtonschulz5(M_tilde_snr, steps=group["ns_steps"])
                    
                    # Step 7: Adjust learning rate based on parameter shape
                    # Uses fixed scaling based on parameter dimensions rather than dynamic norm-based scaling
                    adjusted_lr = self.adjust_lr_for_muon(lr, 0.2, original_shape)
                    
                    # Apply scaled orthogonalized update
                    g = O.flatten()
                    g = g * adjusted_lr  # Scale by adjusted learning rate ratio
                    
                    g = g.to(update_buffer.dtype)

                else:
                    g = update_buffer_views[self.rank]
                    
                # Perform asynchronous all-gather for distributed synchronization
                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
                
            # Apply final batch of updates
            update_prev()