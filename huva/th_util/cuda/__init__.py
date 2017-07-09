from cffi import FFI
import torch
import torch.cuda
import os

this_folder = os.path.dirname(os.path.abspath(__file__))

header_content = open(os.path.join(this_folder, 'cudalib.h')).read()
ffi = FFI()
ffi.cdef(header_content)

LIB = ffi.dlopen(os.path.join(this_folder, 'cudalib.so'))

def fptr(tensor):
    return ffi.cast('float*', tensor.data_ptr())

def iptr(tensor):
    return ffi.cast('int*', tensor.data_ptr())

def check_hwc_args(compact, sparse, indices, use_gpu):
    """
    compact: KxC
    sparse:  1xHxWxC
    indices: K
    """
    if use_gpu:
        assert isinstance(compact, torch.cuda.FloatTensor)
        assert isinstance(sparse,  torch.cuda.FloatTensor)
        assert isinstance(indices, torch.cuda.IntTensor)
    else:
        assert isinstance(compact, torch.FloatTensor)
        assert isinstance(sparse,  torch.FloatTensor)
        assert isinstance(indices, torch.IntTensor)
    """
    assert sparse.is_contiguous()
    assert compact.is_contiguous()
    assert indices.is_contiguous()
    """
    assert sparse.dim()==4
    assert compact.dim()==2
    assert indices.dim()==1
    N, H, W, C = sparse.size()
    K = indices.size(0)
    assert N == 1
    assert compact.size(0)==K and compact.size(1)==C
    return C, H, W, K

def to_hwc(chw):
    return chw.permute(0,2,3,1) # move C to last dimension, giving NHWC

def to_chw(hwc):
    return hwc.permute(0, 3, 1, 2) # move C to second dimension, giving NCHW

"""
HWC spatial transfer
"""
def cpu_spatial_gather_hwc(compact, sparse, indices):
    C, H, W, K = check_hwc_args(compact, sparse, indices, False)
    LIB.CPU_spatial_gather_hwc(fptr(compact), fptr(sparse), iptr(indices), C, H, W, K)

def cpu_spatial_scatter_hwc(compact, sparse, indices):
    C, H, W, K = check_hwc_args(compact, sparse, indices, False)
    LIB.CPU_spatial_scatter_hwc(fptr(compact), fptr(sparse), iptr(indices), C, H, W, K)

def cuda_spatial_gather_hwc(compact, sparse, indices):
    C, H, W, K = check_hwc_args(compact, sparse, indices, True)
    LIB.launch_spatial_gather_hwc(fptr(compact), fptr(sparse), iptr(indices), C, H, W, K)

def cuda_spatial_scatter_hwc(compact, sparse, indices):
    C, H, W, K = check_hwc_args(compact, sparse, indices, True)
    LIB.launch_spatial_scatter_hwc(fptr(compact), fptr(sparse), iptr(indices), C, H, W, K)

"""
CHW spatial transfer
not implemented
"""

"""
depthwise convolution on CHW
"""

def check_chw_args(input, weight, output, use_gpu):
    proper_type = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    assert isinstance(input,  proper_type)
    assert isinstance(weight, proper_type)
    assert isinstance(output, proper_type)
    assert input.dim()==4
    assert weight.dim()==3
    assert output.dim()==4
    """
    assert input.is_contiguous()
    assert weight.is_contiguous()
    assert output.is_contiguous()
    """
    assert input.size() == output.size()
    N, C, H, W = input.size()
    assert weight.size() == (C, 3, 3), 'only support 3x3 depthwise conv'
    return N, C, H, W


def cpu_depthwise_conv2d_chw(input, weight, output):
    N, C, H, W = check_chw_args(input, weight, output, False)
    LIB.CPU_depthwise_conv2d_chw_k3(fptr(input), fptr(weight), fptr(output), N, C, H, W)

def cuda_depthwise_conv2d_chw(input, weight, output, perregion=False):
    N, C, H, W = check_chw_args(input, weight, output, True)
    LIB.launch_depthwise_conv2d_chw_k3(fptr(input), fptr(weight), fptr(output), N, C, H, W, perregion)
