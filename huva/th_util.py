import cv2
import torch
import matplotlib.pyplot as plt
from np_util import *
import math
from th_monitor import *
from th_nn_stats import *
from th_visualize import *

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

def get_num_correct(output, labels):
    """
    Compute number of correrct predictions for CrossEntropyLoss
    output is N by num_class
    labels is N by 1
    """
    maxval, maxpos = output.max(1)
    equals = maxpos == labels
    return equals.sum()

class LogPrinter:
    def __init__(self, filename):
        self.logtxt = ''
        self.file = open(filename, 'w')
    def log(self, val, show_onscreen=True):
        try:
            txt = str(val) + '\n'
            self.logtxt += txt
            self.file.write(txt)
            self.file.flush()
        except Exception as e:
            print("failed to write to logger: {}".format(e.message))
        if show_onscreen:
            print(val)
    def close(self):
        self.file.close()


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
    def __repr__(self):
        return "Flatten()"

class View(torch.nn.Module):
    def __init__(self, *sizes):
        torch.nn.Module.__init__(self)
    def forward(self, x):
        return x.view(x.size(0), *sizes)
    def __repr__(self):
        return "Flatten{}".format(str(sizes))

class InfoU(torch.nn.Module):
    def __init__(self, alpha=2, inplace=False):
        super(InfoU, self).__init__()
        self.alpha = alpha
        # ignore inplace
    def forward(self, x):
        exped   = (-x*self.alpha).exp()
        divided = exped.div(self.alpha+exped)
        logged  = divided.log() / self.alpha
        return -logged


def init_weights(module):
    """
    Initialize Conv2d, Linear and BatchNorm2d
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
            std = math.sqrt(2.0 / n) # 2 accounts for ReLU's half-slashing
            m.weight.data.normal_(0, std) 
            m.bias.data.zero_()
            print (n, m.weight.data.norm())
        elif isinstance(m, torch.nn.Linear):
            n = m.in_features # in_features
            std = math.sqrt(2.0 / n) # 2 accounts for ReLU's half-slashing
            m.weight.data.normal_(0, std)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def decay_learning_rate(optimizer, decay):
    for group in optimizer.param_groups:
        group['lr'] *= decay

