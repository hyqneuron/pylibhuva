import torch
from torch.autograd import Function
from huva.th_util.th_math import gumbel_max, multinomial_max, plain_max


class FuncSTCategorical(Function):

    def __init__(self, stochastic=False, forget_mask=False):
        super(FuncSTCategorical, self).__init__()
        self.forget_mask = forget_mask

    def forward(self, x, mask):
        if not self.forget_mask:
            self.save_for_backward(mask)
        return mask * x

    def backward(self, grad_output):
        if self.forget_mask:
            return grad_output, grad_output.new().resize_as_(grad_output).fill_(0)
        mask = self.saved_tensors[0]
        return mask * grad_output, grad_output.new().resize_as_(grad_output).fill_(0)


class FuncOneHotSTCategorical(Function):
    """ One-hot straight-through categorical """

    def __init__(self, stochastic=False, forget_mask=False):
        super(FuncOneHotSTCategorical, self).__init__()
        self.stochastic  = stochastic
        self.forget_mask = forget_mask

    def forward(self, x):
        # x is logit! can be probability only in the case of non-stochastic sampling
        if self.stochastic:
            mask = gumbel_max(x)
        else:
            mask = plain_max(x)
        if not self.forget_mask:
            self.save_for_backward(mask)
        return mask # one-hot

    def backward(self, grad_output):
        if self.forget_mask:
            return grad_output # truly straight-through
        mask = self.saved_tensors[0]
        return mask * grad_output

