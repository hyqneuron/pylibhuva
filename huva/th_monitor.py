import torch
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

                if monitor_update:
                    self.update_sqr += (step_size * exp_avg.div(denom).norm()) **2

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0 and self.separate_decay:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

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

class MonitoredSGD(torch.optim.SGD):
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
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

                if monitor_update:
                    self.update_sqr += (group['lr'] * d_p.norm())**2
                p.data.add_(-group['lr'], d_p)

        self.update_norm = math.sqrt(self.update_sqr+1e-8)
        return loss
