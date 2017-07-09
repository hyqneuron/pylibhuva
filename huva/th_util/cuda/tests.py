import torch
import torch.cuda
import torch.nn as nn
from torch.autograd import Variable
from .__init__ import *


def test_spatial_transfer_hwc(C, H, W, K):
    # cpu tests
    sparse  = torch.Tensor(1, H, W, C)
    compact = torch.Tensor(K, C)
    indices = torch.randperm(W*H)[:K].int()
    for i in xrange(W*H):
        sparse[0, i/W, i%W, :] = i
    cpu_spatial_gather_hwc(compact, sparse, indices)
    for i in xrange(indices.size(0)):
        assert (compact[i]==indices[i]).all()
    sparse_target = sparse.clone().fill_(-1)
    cpu_spatial_scatter_hwc(compact, sparse_target, indices)
    for i in xrange(indices.size(0)):
        idx = indices[i]
        assert (sparse_target[0, idx/W, idx%W, :] == idx).all()
    # gpu tests
    cu_sparse  = sparse.cuda()
    cu_compact = compact.cuda()
    cu_indices = indices.cuda()
    cuda_spatial_gather_hwc(cu_compact, cu_sparse, cu_indices)
    assert (cu_compact.cpu() == compact).all()
    cu_sparse_target = cu_sparse.clone().fill_(-1)
    cuda_spatial_scatter_hwc(cu_compact, cu_sparse_target, cu_indices)
    assert (cu_sparse_target.cpu() == sparse_target).all()


test_spatial_transfer_hwc(10, 4,4,4)
test_spatial_transfer_hwc(20, 4,4,16)
test_spatial_transfer_hwc(200, 10,10,20)
test_spatial_transfer_hwc(1024, 10,10,100)


def fT(weight):
    w2 = weight.clone()
    assert weight.dim() == 3
    for c in xrange(weight.size(0)):
        w2[c] = weight[c].t()
    return w2
    """
    for y in xrange(weight.size(1)):
        for x in xrange(weight.size(2)):
        """

def test_depthwise_conv2d_chw(N,C,H,W, gpu_only=False):
    # gold
    conv = nn.Conv2d(C, C, 3, stride=1, padding=1, groups=C, bias=False)
    input    = torch.Tensor(N, C, H, W).normal_()
    if False: # debugging by setting fixed value to weight
        for c in xrange(C):
            conv.weight.data[c, 0, 1, 1] = c
    v_input  = Variable(input)
    v_output = conv(v_input)
    output   = v_output.data
    weight   = conv.weight.data.squeeze(1)
    # cpu tests
    if gpu_only:
        print 'skipped cpu test for {}'.format((N,C,H,W))
    else:
        cpu_input  = input.clone()
        cpu_output = output.clone().fill_(999)
        cpu_weight = weight.clone()
        cpu_depthwise_conv2d_chw(cpu_input, cpu_weight, cpu_output)
        diff = cpu_output - output
        assert diff.abs().max() < 1e-6, torch.cat([output, cpu_output], 3)
    # gpu tests
    gpu_input  = input.clone().cuda()
    gpu_output = output.clone().fill_(888).cuda()
    gpu_weight = weight.clone().cuda()
    cuda_depthwise_conv2d_chw(gpu_input, gpu_weight, gpu_output)
    diff = gpu_output.cpu() - output
    assert diff.abs().max() < 1e-6, torch.cat([output, gpu_output.cpu()], 3)


test_depthwise_conv2d_chw(1, 10, 1, 1)
test_depthwise_conv2d_chw(1, 1, 1, 2)
test_depthwise_conv2d_chw(1, 10, 1, 2)
test_depthwise_conv2d_chw(1, 10, 2, 1)
test_depthwise_conv2d_chw(1, 10, 2, 2)
test_depthwise_conv2d_chw(1, 10, 4, 4)
test_depthwise_conv2d_chw(1, 10, 5, 5)
test_depthwise_conv2d_chw(1, 10, 32, 32)
test_depthwise_conv2d_chw(1, 10, 63, 63)
test_depthwise_conv2d_chw(1, 10, 64, 64)
test_depthwise_conv2d_chw(1, 10, 65, 65)
test_depthwise_conv2d_chw(1, 10, 256, 256)
test_depthwise_conv2d_chw(1, 256, 256, 256)
test_depthwise_conv2d_chw(11, 10, 256, 256)
test_depthwise_conv2d_chw(10, 256, 256, 256, True)
