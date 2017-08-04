import torch
from torch.autograd import Function
from .math_ops import gumbel_max, multinomial_max, plain_max
from .cuda import cpu_spatial_gather_hwc,   cuda_spatial_gather_hwc,                \
                              cpu_spatial_scatter_hwc,  cuda_spatial_scatter_hwc,   \
                              cpu_depthwise_conv2d_chw, cuda_depthwise_conv2d_chw
from .base import new_as


class NegateGradient(Function):
    """ 
    this function negates gradient without changing forward pass, useful in minimax context
    """

    def forward(self, input):
        return input

    def backward(self, grad_output):
        return - grad_output


class SpatialGatherHWC(Function):

    def forward(self, hwc, indices):
        N, H, W, C = hwc.size()
        K = indices.size(0)
        kc = hwc.new().resize_(K, C)
        if hwc.is_cuda:
            cuda_spatial_gather_hwc(kc, hwc, indices)
        else:
            cpu_spatial_gather_hwc(kc, hwc, indices)
        # save for backward
        self.save_for_backward(indices)
        self.HWC = H, W, C
        return kc

    def backward(self, grad_kc):
        """
        Scatter grad_output onto 
        """
        H, W, C = self.HWC
        hwc_grad = grad_kc.new().resize_(1, H, W, C)
        indices = self.saved_tensors[0]
        if grad_kc.is_cuda:
            cuda_spatial_scatter_hwc(grad_kc, hwc_grad, indices)
        else:
            cpu_spatial_scatter_hwc(grad_kc, hwc_grad, indices)
        return hwc_grad, None


class SpatialScatterHWC(Function):

    def __init__(self, H, W):
        super(SpatialScatterHWC, self).__init__()
        self.HW = H, W

    def forward(self, kc, indices):
        H, W = self.HW
        K, C = kc.size()
        hwc = kc.new().resize_(1, H, W, C)
        if kc.is_cuda:
            cuda_spatial_scatter_hwc(kc, hwc, indices)
        else:
            cpu_spatial_scatter_hwc(kc, hwc, indices)
        # save for backward
        self.save_for_backward(indices)
        self.KC = K, C
        return hwc

    def backward(self, grad_hwc):
        K, C = self.KC
        grad_kc = grad_hwc.new().resize_(K, C)
        if grad_hwc.is_cuda:
            cuda_spatial_gather_hwc(grad_kc, grad_hwc, indices)
        else:
            cpu_spatial_gather_hwc(grad_kc, grad_hwc, indices)


class DepthwiseConv2dCHW(Function):

    def forward(self, input, weight):
        output = new_as(input)
        if input.is_cuda:
            cuda_depthwise_conv2d_chw(input, weight, output)
        else:
            cpu_depthwise_conv2d_chw(input, weight, output)
        return output

    def backward(self, grad_output):
        raise NotImplementedError


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

