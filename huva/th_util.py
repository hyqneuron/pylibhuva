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

def get_model_param_norm(model):
    """ get parameter length of a model """
    return sum([param.norm().data[0] for param in model.parameters()])

import math
class MonitoredAdam(torch.optim.Adam):
    def step(self, closure=None, monitor_update=False):
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

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if monitor_update:
                    self.update_norm += step_size * (exp_avg.div(denom)).norm()
                p.data.addcdiv_(-step_size, exp_avg, denom)

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

def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
