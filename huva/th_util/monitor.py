import torch
from torch.optim.optimizer import required
import math

"""
Monitored optimizers:
- MonitoredAdam
- MonitoredRMSprop
- MonitoredSGD
"""


class MonitoredAdam(torch.optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, separate_decay=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        torch.optim.Optimizer.__init__(self, params, defaults)
        self.separate_decay = separate_decay

    def step(self, closure=None, update_monitor=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if update_monitor:
            self.update_sqr = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0 and not self.separate_decay:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if update_monitor:
                    self.update_sqr += (step_size * exp_avg.div(denom).norm()) **2

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0 and self.separate_decay:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        if update_monitor:
            self.update_norm = math.sqrt(self.update_sqr+1e-8)
        return loss


class MonitoredRMSprop(torch.optim.RMSprop):

    def step(self, closure=None, monitor_update=True):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.update_sqr = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = grad.new().resize_as_(grad).zero_()
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = grad.new().resize_as_(grad).zero_()
                    if group['centered']:
                        state['grad_avg'] = grad.new().resize_as_(grad).zero_()

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                    if monitor_update:
                        self.update_sqr += (group['lr'] * buf.norm())**2
                else:
                    #p.data.addcdiv_(-group['lr'], grad, avg)
                    normed_grad = grad.div(avg)
                    p.data.add_(-group['lr'], normed_grad)
                    if monitor_update:
                        self.update_sqr += (group['lr'] * normed_grad.norm())**2

        self.update_norm = math.sqrt(self.update_sqr+1e-8) * group['lr']
        return loss


class MonitoredNorms(object):

    def __init__(self):
        """
        - w_norm: norm of the parameter
        - g_norm: norm of back-propagated error
        - d_norm: norm of weight decay
        - u_norm: norm of update made to this parameter
        """
        self.w_norm = None
        self.g_norm = None
        self.d_p_ema = None
        self.mg_norm= None
        self.g_proj_d = None
        self.d_norm = None
        self.u_norm = None


class MonitoredSGD(torch.optim.SGD):
    """
    MonitoredSGD adds several features to torch.optim.SGD:
    - tracks size of total update in self.update_norm
    - tracks MonitoredNorms for every parameter
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(torch.optim.SGD, self).__init__(params, defaults)

    def step(self, closure=None, update_monitor=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.update_sqr = 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            if 'norms' not in group:
                group['norms'] = [MonitoredNorms() for p in group['params']] # one norm per param
            L1 = weight_decay != 0 and 'L1' in group and group['L1']

            for p, norm in zip(group['params'], group['norms']):
                if p.grad is None:
                    continue

                """ record norms """
                if update_monitor:
                    norm.w_norm = p.data.norm()
                    norm.g_norm = p.grad.data.norm()
                    if momentum != 0:
                        norm.mg_norm= self.state[p]['momentum_buffer'].norm()

                d_p = p.grad.data
                p.grad_var = d_p.var()

                if weight_decay != 0:
                    # L1 decay
                    if L1:
                        decay = weight_decay * p.data.sign()
                    # L2 decay
                    else:
                        decay = weight_decay * p.data
                    if update_monitor:
                        norm.d_norm = decay.norm()
                        norm.g_proj_d = decay.dot(d_p) / decay.norm()
                    d_p.add_(decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                self.update_sqr += (group['lr'] * d_p.norm())**2

                """ check for 0-crossing if we are using L1 decay, which is unstable """
                if L1:
                    pre_sign = p.data.sign()
                    p.data.add_(-group['lr'], d_p)
                    post_sign = p.data.sign()
                    mask = (pre_sign * post_sign != -1).float()
                    p.data.mul_(mask) # only keep those that didn't change sign
                    # L1 doesn't support u_norm tracking
                else:
                    p.data.add_(-group['lr'], d_p)
                    if update_monitor:
                        norm.u_norm = d_p.norm() * group['lr']

        """ norm of total update """
        self.update_norm = math.sqrt(self.update_sqr+1e-8)
        return loss


class MonitoredAdagrad(torch.optim.Adagrad):

    def step(self, closure=None, update_monitor=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if update_monitor:
            self.update_sqr = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad_indices = grad.indices()
                    grad_values = grad.values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum'].sparse_mask(grad)
                    std_values = std.values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                    if update_monitor:
                        assert False, 'currently does not support sparse tensor Adagrad under monitor mode'
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

                    if update_monitor:
                        self.update_sqr += (clr * grad.div(std).norm()) **2

        if update_monitor:
            self.update_norm = math.sqrt(self.update_sqr+1e-8)

        return loss


class MonitoredSpecificSGD(torch.optim.SGD):
    """
    MonitoredSGD adds several features to torch.optim.SGD:
    - tracks size of total update in self.update_norm
    - tracks MonitoredNorms for every parameter
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, specific_mode=None, d_p_momentum=0.98, mult_weight_norm=True):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay)
        assert specific_mode in [None, 'layer', 'unit']
        self.specific_mode = specific_mode
        self.d_p_momentum = d_p_momentum
        self.mult_weight_norm = mult_weight_norm
        super(torch.optim.SGD, self).__init__(params, defaults)

    def step(self, closure=None, update_monitor=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if update_monitor:
            self.update_sqr = 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            decay_mults = [None] * len(group['params']) if 'decay_mults' not in group else group['decay_mults']
            if 'norms' not in group:
                group['norms'] = [MonitoredNorms() for p in group['params']] # one norm per param

            for p, norm, decay_mult in zip(group['params'], group['norms'], decay_mults):
                if p.grad is None:
                    continue

                """ record norms """
                if update_monitor:
                    norm.w_norm = p.data.norm()
                    norm.g_norm = p.grad.data.norm()
                    if momentum != 0:
                        norm.mg_norm= self.state[p]['momentum_buffer'].norm() if 'momentum_buffer' in self.state[p] else 0

                d_p = p.grad.data
                p.grad_var = d_p.var()

                if weight_decay != 0:
                    decay = weight_decay * p.data
                    if decay_mult is not None:
                        decay.mul_(decay_mult)
                        if not hasattr(self, 'printed'):
                            print('using decay_mult')
                            print(decay_mult)
                            self.printed = True
                    if update_monitor:
                        norm.d_norm = decay.norm()
                        norm.g_proj_d = decay.dot(d_p) / (decay.norm() + 1e-8)
                    d_p.add_(decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = buf


                """ apply update """
                """ perform layer/unit-specific adjustment before application """
                if self.specific_mode is not None:
                    if self.specific_mode == 'layer':
                        """ normalize d_p to same magnitude as p """
                        if norm.d_p_ema is None:
                            norm.d_p_ema = d_p.norm()
                        else:
                            norm.d_p_ema = self.d_p_momentum * norm.d_p_ema + (1-self.d_p_momentum) * d_p.norm()
                        multiplier  = 1 / (norm.d_p_ema + 1e-8)
                        if self.mult_weight_norm: multiplier *= p.data.norm()
                        d_p.mul_(multiplier)
                    elif self.specific_mode == 'unit':
                        assert False, 'unit-specific mode not implemented'
                    else:
                        assert False, 'unknown mode {}'.format(self.specific_mode)
                p.data.add_(-group['lr'], d_p)

                if update_monitor:
                    self.update_sqr += (group['lr'] * d_p.norm())**2
                    norm.u_norm = d_p.norm() * group['lr']

        """ record norm of total update """
        if update_monitor:
            self.update_norm = math.sqrt(self.update_sqr+1e-8)
        return loss
