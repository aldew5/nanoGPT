import torch

class OneSided(torch.optim.Optimizer):
    """

    **Warnings:**
      - Assumes that all parameters are 2D.
      - Do not use it for parameters that are 0D/1D (e.g. embeddings or final FC layers).
      - For convolutional filters (4D tensors) you should flatten the last dimensions.
    """
    
    def __init__(self, params, lr=3e-4, momentum=0.80, cov_momentum=0.98, damping=1e-5, 
                 weight_decay=1e-4, nesterov=True, eps=1e-8, T_stats=5):
        defaults = dict(lr=lr, momentum=momentum, cov_momentum=cov_momentum, damping=damping,
                        weight_decay=weight_decay, nesterov=nesterov, eps=eps)
        super().__init__(params, defaults)
        self.mystep = 0
        self.T_stats = T_stats

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['momentum']         # for momentum update
            beta2 = group['cov_momentum']     # for covariance (running average) update
            damping = group['damping']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                # skip 1D params
                if p.grad.ndim < 2:
                    continue  
                # if dim > 2, flatten other dimensions
                G = p.grad.data if p.grad.ndim == 2 else p.grad.data.view(p.grad.data.shape[0], -1)

                state = self.state[p]
                
                # Initialize the momentum buffer if needed.
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(G)
                M = state['momentum_buffer']
                # Momentum update: M = (1 - beta1) * M + beta1 * G
                M.mul_(1 - beta1).add_(beta1 * G)
                # Optionally use Nesterov momentum:
                effective_grad = G + beta1 * M if nesterov else M
                
                m, n = G.shape
                # row cov case
                if m <= n:
                    if 'cov' not in state:
                        # init S^(C) as damping * I
                        # TODO: check init
                        state['cov'] = damping * torch.eye(m, device=p.device, dtype=p.dtype)
                    S_C = state['cov']
                    
                    if self.mystep % self.T_stats == 0:
                        # S = (1 - beta2)*S + beta2*(G G^T / n + damping*I)
                        S_C.mul_(1 - beta2)
                        S_C.add_(beta2 * (1/n * G @ G.T + damping * torch.eye(m, device=p.device, dtype=p.dtype)))
                    # Compute S^{-1/2} via eigendecomp TODO: check this method
                    eigvals, eigvecs = torch.linalg.eigh(S_C)
                    inv_sqrt_S_C = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + eps)) @ eigvecs.T
                    precond_update = inv_sqrt_S_C @ effective_grad
                else:
                    if 'cov' not in state:
                        state['cov'] = damping * torch.eye(n, device=p.device, dtype=p.dtype)
                    S_K = state['cov']

                    if self.mystep % self.T_stats == 0:
                        S_K.mul_(1 - beta2)
                        S_K.add_(beta2 * (1/m * (G.T @ G) + damping * torch.eye(n, device=p.device, dtype=p.dtype)))
                    eigvals, eigvecs = torch.linalg.eigh(S_K)
                    inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + eps)) @ eigvecs.T
                    precond_update = effective_grad @ inv_sqrt
                
                # TODO: weight decay?
                #p.data.mul_(1 - lr * weight_decay)
                p.data.add_(-lr, precond_update)
        self.mystep += 1
