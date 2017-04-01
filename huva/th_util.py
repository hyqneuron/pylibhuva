import torch
import cv2
import matplotlib.pyplot as plt
from np_util import *

def th_get_jet(img, label):
    img_np = bgr_to_numpy(img)
    H, W         = label.size(1), label.size(2)
    img_H, img_W = img.size(1),   img.size(2)
    label_np = label.numpy().reshape(H, W, 1)
    jet = get_jet(img_np, cv2.resize(label_np, (img_W, img_H)))
    return jet

def img_to_numpy(img):
    """
    img of range[0,1] shape 3xHxW
    return uint8 numpy array HxWx3
    """
    return (img*255).numpy().astype('uint8').transpose([1,2,0])

def bgr_to_numpy(torch_bgr_img, rgb=False):
    """
    torch_bgr_img is a 3xHxW torch.FloatTensor, range [0,255]
    returns a numpy uint8 array, HxWx3
    """
    result = torch_bgr_img.numpy().astype('uint8').transpose([1,2,0])
    if rgb:
        result = result[:,:,[2,1,0]]
    return result

def save_bgr_3hw(path, img):
    """
    img is a 3xHxW torch.Tensor, values range [0,255]
    """
    return cv2.imwrite(path, img.numpy().transpose([1,2,0]))


def load_bgr_3hw(path):
    """
    returns a 3xHxW torch.Tensor, values range [0,255]
    """
    return torch.from_numpy(cv2.imread(path).transpose([2,0,1]))

def imshow_th(img_th):
    imshow_np(torch.img_to_numpy(img_th))

def get_model_param_norm(model, simple=True):
    """ get parameter length of a model """
    if simple:
        return sum([p.data.norm() for p in model.parameters()])
    else:
        return math.sqrt(sum([p.data.norm()**2 for p in model.parameters()]))


def get_num_correct(output, labels):
    """
    Compute number of correrct predictions for CrossEntropyLoss
    output is N by num_class
    labels is N by 1
    """
    maxval, maxpos = output.max(1)
    equals = maxpos == labels
    return equals.sum()


import math
class MonitoredAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, separate_decay=False):
        print(lr)
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        torch.optim.Optimizer.__init__(self, params, defaults)
        self.separate_decay = separate_decay
        print(self.param_groups[0]['lr'])

    def step(self, closure=None, monitor_update=True):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.update_norm = 0

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
                    self.update_norm += step_size * (exp_avg.div(denom)).norm() **2

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0 and self.separate_decay:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        self.update_norm = math.sqrt(self.update_norm+1e-8)
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

        self.update_norm = 0

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
                    self.update_norm += d_p.norm()**2 * group['lr']
                p.data.add_(-group['lr'], d_p)

        self.update_norm = math.sqrt(self.update_norm+1e-8)
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

        self.update_norm = 0

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
                        self.update_norm += buf.norm()**2 * group['lr']
                else:
                    #p.data.addcdiv_(-group['lr'], grad, avg)
                    normed_grad = grad.div(avg)
                    p.data.add_(-group['lr'], normed_grad)
                    if monitor_update:
                        self.update_norm += normed_grad.norm()**2 * group['lr']

        self.update_norm = math.sqrt(self.update_norm+1e-8)
        return loss

class CachedSequential(torch.nn.Sequential):
    def set_cache_targets(self, cache_targets):
        self.cache_targets = cache_targets
    def forward(self, input):
        self.cached_outs = {}
        for name, module in self._modules.iteritems():
            input = module(input)
            if name in self.cache_targets:
                self.cached_outs[name] = input
        return input

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def init_weights(module, use_in_channel=False):
    """
    Initialize layers using MSRinit if use_in_channel==False
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            """
            Now that we can count on the variance of output units to be 1, or maybe 0.5 (due to ReLU), we have 
            Nin * var(w) = [1 or 2]
            var(w) = [1 or 2]/Nin
            std(w) = sqrt([1 or 2]/Nin)
            """
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, math.sqrt(2. / n)) # 2 accounts for ReLU's half-slashing
            if m.bias is not None:
                m.bias.data.zero_()
            print (n, m.weight.data.norm())
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            n = m.weight.size(1) # in_features
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def get_layer_utilization(layer):
    """
    Analyze dead-unit statistics

    weight dimension: [num_out, num_in, k, k]
    Return a summary of size [num_in]
    """
    W = layer.weight.data.cpu()
    W_mean = W.view(W.size(0), W.size(1), -1).abs().mean(2).squeeze(2)
    W_summary = W_mean.mean(0).squeeze(0)
    return W_mean, W_summary

def get_all_utilization(module, threshold=1e-20, result=None, prefix=''):
    """
    Analyze dead-unit statistics
    """
    if result is None:
        result = {}
    for name, sub_mod in module._modules.iteritems():
        full_name = prefix + name
        if isinstance(sub_mod, torch.nn.Conv2d):
            result[full_name] = get_layer_utilization(sub_mod)[1].lt(threshold).sum(), sub_mod.weight.size(1)
        elif hasattr(sub_mod, '_modules'):
            get_all_utilization(sub_mod, threshold, result=result, prefix=full_name+'/')
    return result
