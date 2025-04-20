import torch
from torch.optim import Optimizer

class TwoSided(Optimizer):
    """
    Two-Sided version
    """
    def __init__(self, params, lr=3e-4, momentum=0.9, cov_momentum=0.99, damping=1e-5,
                 weight_decay=0.0, nesterov=True, eps=1e-8, T_stats=1):
        defaults = dict(lr=lr, momentum=momentum, cov_momentum=cov_momentum,
                        damping=damping, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, T_stats=T_stats)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['momentum']
            beta2 = group['cov_momentum']
            damping = group['damping']
            wd = group['weight_decay']
            nesterov = group['nesterov']
            eps = group['eps']
            T_stats = group['T_stats']

            for p in group['params']:
                if p.grad is None:
                    continue
                G = p.grad.data

                assert G.ndim >= 2
                # flatten other dims
                if G.ndim > 2:
                    G = G.view(G.size(0), -1)

                state = self.state[p]
                # momentum buffer
                M = state.get('momentum_buffer')
                if M is None:
                    M = state['momentum_buffer'] = torch.zeros_like(G)
                # update momentum
                M.mul_(1 - beta1).add_(beta1, G)
                grad_eff = G + beta1 * M if nesterov else M

                m, n = G.shape
                # initialize covariances
                if 'S_C' not in state:
                    state['S_C'] = damping * torch.eye(m, device=G.device, dtype=G.dtype)
                if 'S_K' not in state:
                    state['S_K'] = damping * torch.eye(n, device=G.device, dtype=G.dtype)
                S_C = state['S_C']
                S_K = state['S_K']

                # update cov every T_stats steps
                if self.step_count % T_stats == 0:
                    # row-covariance
                    S_C.mul_(1 - beta2)
                    S_C.add_(beta2, G @ G.t().div(n) + damping * torch.eye(m, device=G.device, dtype=G.dtype))
                    # col-covariance
                    S_K.mul_(1 - beta2)
                    S_K.add_(beta2, G.t() @ G.div(m) + damping * torch.eye(n, device=G.device, dtype=G.dtype))

                # eigendecomposition for inv_sqrt
                dC, QC = torch.linalg.eigh(S_C)
                inv_sqrt_C = QC @ torch.diag(dC.add(eps).rsqrt()) @ QC.t()
                dK, QK = torch.linalg.eigh(S_K)
                inv_sqrt_K = QK @ torch.diag(dK.add(eps).rsqrt()) @ QK.t()

                # two-sided preconditioned update
                W = inv_sqrt_C @ grad_eff @ inv_sqrt_K
                norm = W.mul(W).mean().sqrt().clamp_min(eps)
                step_size = lr / norm
                # weight decay on parameter
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                # apply update
                update = W.mul(step_size)
                update = update.view_as(p.data)
                p.data.add_(-1.0, update)

        self.step_count += 1
