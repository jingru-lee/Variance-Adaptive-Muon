import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer with gradient clipping support.
    
    This implementation follows the standard AdamW algorithm with an additional
    gradient clipping feature using the same logic as Muon optimizer.
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient
               and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
        amsgrad: whether to use the AMSGrad variant (default: False)
        g_norm: gradient norm threshold for gradient clipping (default: 1.0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        g_norm=1.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= g_norm:
            raise ValueError(f"Invalid g_norm value: {g_norm}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            g_norm=g_norm,
        )
        super(AdamW, self).__init__(params, defaults)

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

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            amsgrad = group['amsgrad']
            g_norm = group['g_norm']

            for p in group['params']:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                
                # Apply gradient clipping
                g = self._clip_grad(p.grad, g_norm)
                grads.append(g)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state['step'] += 1
                state_steps.append(state['step'])

            # Perform AdamW update
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    max_exp_avg_sq = max_exp_avg_sqs[i]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1

                # Compute the bias-corrected denominator
                bias_correction2_sqrt = bias_correction2 ** 0.5
                denom = denom / bias_correction2_sqrt

                # Apply weight decay (decoupled weight decay as in AdamW)
                param.mul_(1 - lr * weight_decay)

                # Apply update
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss