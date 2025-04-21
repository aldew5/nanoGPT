import torch


class TwoSided2(torch.optim.Optimizer):
    '''
    two-sided
   '''
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
            weight_decay=1e-2, beta2=1e-2, freq=2, damping=1e-16,
            using_raw_grad=True):
        print('my two-sided')
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                weight_decay=weight_decay,
                beta2=beta2,
                damping=damping
                )
        super().__init__(params, defaults)
        self.cutoff = 1e-30
        self.mystep = 0
        self.freq = freq
        self.using_raw_grad = using_raw_grad 
        if self.using_raw_grad:
            print('using raw ggt')
        else:
            print('using momentum ggt')


    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta2 = group['beta2']
            freq = self.freq
            damping = group['damping']
            using_raw_grad = self.using_raw_grad

            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cpu', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                assert len(p.grad.shape)>=2
                if p.grad.shape[0]>p.grad.shape[-1]:
                    g = p.grad.view(p.grad.shape[0], -1) #.to(dtype=torch.bfloat16)
                else:
                    g = p.grad.view(-1, p.grad.shape[-1]) #.to(dtype=torch.bfloat16)
                if self.mystep == 0:
                    print('dtype', g.dtype)
                state = self.state[p]

                if self.mystep == 0 and len(p.grad.shape)>2:
                    print('special merging', p.grad.shape, g.shape)

                if 'left' not in state:
                    state['left'] = torch.diag(g.new(g.size(0)).fill_(1))
                    state['eigen_lQ'] = torch.diag(g.new(g.size(0)).fill_(1))
                    state['eigen_sqrt_inv_ld'] = g.new(g.size(0)).fill_(1)

                if 'right' not in state:
                    state['right'] = torch.diag(g.new(g.size(1)).fill_(1))
                    state['eigen_rQ'] = torch.diag(g.new(g.size(1)).fill_(1))
                    state['eigen_sqrt_inv_rd'] = g.new(g.size(1)).fill_(1)

                left = state['left'] 
                right = state['right'] 

                lfactor = right.shape[0]
                ltrace = damping*torch.sum(state['eigen_sqrt_inv_rd']**2)
                rfactor = left.shape[0]
                rtrace = damping*torch.sum(state['eigen_sqrt_inv_ld']**2)

                assert g.size(1) == right.shape[0]
                assert g.size(0) == left.shape[0]

                if using_raw_grad:
                    #lmat  = g @ right_inv @ g.T
                    lhalf = (state['eigen_rQ'].T @ g.T) * state['eigen_sqrt_inv_rd'].view(-1,1) 
                    lmat = lhalf.T @ lhalf

                    #rmat = g.T @ left_inv @ g
                    rhalf = (state['eigen_lQ'].T @ g) * state['eigen_sqrt_inv_ld'].view(-1,1) 
                    rmat = rhalf.T @ rhalf 

                assert len(g.shape) == 2
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)

                if not using_raw_grad:
                    lhalf = (state['eigen_rQ'].T @ g.T) * state['eigen_sqrt_inv_rd'].view(-1,1) 
                    lmat = lhalf.T @ lhalf

                    rhalf = (state['eigen_lQ'].T @ g) * state['eigen_sqrt_inv_ld'].view(-1,1) 
                    rmat = rhalf.T @ rhalf 

                left.mul_(1-beta2).add_(lmat + ltrace*torch.eye(left.shape[0], device=left.device), alpha=beta2/lfactor)
                right.mul_(1-beta2).add_(rmat + rtrace*torch.eye(right.shape[0], device=right.device), alpha=beta2/rfactor)

                if self.mystep % freq == 0:
                    cutoff = self.cutoff
                    try:
                        ld, lQ = torch.linalg.eigh( (left+left.T).to(dtype=torch.float32)/2.0)
                        success = True
                    except Exception as exception:
                        print('skipped eigh' + str(exception) )
                        print('left info', left)
                        success = False

                    if success:
                        #ld = torch.abs(ld)
                        #ld[ld<cutoff]=cutoff
                        ld[ld<cutoff]=0.0
                        sqrt_inv_ld = (1.0/torch.sqrt(ld)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        state['eigen_lQ'] = lQ.to(dtype=g.dtype)
                        state['eigen_sqrt_inv_ld'] = sqrt_inv_ld.to(dtype=g.dtype)

                    try:
                        rd, rQ = torch.linalg.eigh( (right+right.T).to(dtype=torch.float32)/2.0)
                        success = True
                    except Exception as exception:
                        print('skipped eigh' + str(exception) )
                        print('right info', right)
                        success = False


                    if success:
                        #rd = torch.abs(rd)
                        #rd[rd<cutoff]=cutoff
                        rd[rd<cutoff]=0.0
                        sqrt_inv_rd = (1.0/torch.sqrt(rd)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        state['eigen_rQ'] = rQ.to(dtype=g.dtype)
                        state['eigen_sqrt_inv_rd'] = sqrt_inv_rd.to(dtype=g.dtype)

                lQ = state['eigen_lQ'] 
                sqrt_inv_ld = state['eigen_sqrt_inv_ld'] 
                rQ = state['eigen_rQ'] 
                sqrt_inv_rd = state['eigen_sqrt_inv_rd'] 

                grad = lQ @ ( (lQ.T @ g) * sqrt_inv_ld.view(-1,1) )
                grad = (grad @ rQ) @ ( rQ.T * sqrt_inv_rd.view(-1,1) )

                grad *= (1.0/grad.square().mean())**0.5 # scale to have update.square().mean() == 1
                updates_flat[curr_idx:curr_idx+p.numel()] = grad.flatten().to(dtype=torch.bfloat16)
                curr_idx += p.numel()

            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.mul_(1 - group['lr'] * group['weight_decay']).add_(g, alpha=-lr)
                curr_idx += p.numel()

        self.mystep += 1