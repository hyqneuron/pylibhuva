from .model import *
from torch.autograd import Variable
import torch.nn.functional as F
import math
import itertools


def clear_grads(variables):
    for var in variables:
        if var.grad is not None:
            var.grad.data.zero_()


def test_space_to_channel(C=3, H=4, W=4, s=2, leak=False):
    global va, vb, vc, a_grad, multiplier
    Hs = H/s # strided H
    Ws = W/s # strided W
    s2c = SpaceToChannel(s)
    c2s = ChannelToSpace(s)
    # a is a Tensor where every channel in every region (sxs block) has the same value
    a = torch.Tensor(1,C,H,W)
    for c in range(C):         
        for y in range(Hs):
            for x in range(Ws):
                a[0, c, y*s:(y+1)*s, x*s:(x+1)*s] = c*100000 + y*Ws + x
    va = Variable(a, requires_grad=True)
    vb = s2c(va)
    # verify that channels that feed to the same region have the same value
    for c in range(C):
        for y in range(Hs):
            for x in range(Ws):
                val = c * 100000 + y*Ws + x
                assert (vb.data[0, c*s*s:(c+1)*s*s, y, x] == val).all()
    # verify that space-to-chanel-to-space is identity
    vc = c2s(vb)
    assert (vc.data == va.data).all()
    # verify gradient of space to channel
    multiplier = torch.range(0, C*H*W-1).view(1, C*s*s, Hs, Ws)
    vsum = (vb * Variable(multiplier)).sum()
    vsum.backward()
    a_grad = torch.Tensor(1,C,H,W).zero_()
    for c in range(C):
        for y in range(H):
            for x in range(W):
                y_jump, y_shift = y / s, y % s
                x_jump, x_shift = x / s, x % s
                a_grad[0, c, y, x] = (c * H * W) + (y_jump*Ws) + (x_jump) + (y_shift*H*Ws) + (x_shift*Hs*Ws)
    assert (va.grad.data==a_grad).all()
    # verify gradient of channel to space
    clear_grads([va,vb, vc])
    multiplier = torch.range(0, C*H*W-1).view(1, C, H, W)
    vc.backward(multiplier)
    assert (va.grad.data==multiplier).all()


def test_space_to_channel_multiple():
    test_space_to_channel(3,   4,  4, 2)
    test_space_to_channel(10,  4,  4, 1)
    test_space_to_channel(10, 16, 16, 1)
    test_space_to_channel(10, 16, 16, 2)
    test_space_to_channel(10, 16, 16, 4)
    test_space_to_channel(1, 12, 12, 3)
    test_space_to_channel(1, 12, 12, 4)
    test_space_to_channel(1, 20, 20, 5)
    test_space_to_channel(1, 20, 10, 5)


def test_kld_for_uniform_categorical(C=10, num_samples=1):
    # verify uniforms have 0 KLD
    a = torch.Tensor(num_samples,C).fill_(1.0/C)
    assert kld_for_uniform_categorical(a) == 0
    # verify half-zero, half-double have log(2)
    if C % 2 == 0:
        b = torch.zeros(num_samples,C)
        b[:, :C/2] = 2.0 / C
        b[:, C/2:] = 1e-15
        ans  = kld_for_uniform_categorical(b) 
        gold = math.log(2) * num_samples
        assert abs(ans - gold) < 1e-7, '{} == {} failed'.format(ans, gold)
    # verify the sequence 1/2, 1/4, 1/8, 1/16 ... has KLD log(C) - log(2)
    # note: not enough precision..., so it's going to be very approximate
    num_ladders = 100 # only work with 100 ladders
    c = torch.zeros(num_samples, num_ladders)
    for i in xrange(1, num_ladders+1, 1):
        c[:, i-1] = 0.5**i
    c[:, num_ladders-1] = 0.5**(num_ladders-1)
    ans  = kld_for_uniform_categorical(c)
    gold = (math.log(num_ladders) - 2*math.log(2)) * num_samples
    assert abs(ans - gold) < 1e-7, '{} == {} failed'.format(ans, gold)


def test_all():
    tests = [
        test_space_to_channel_multiple,
    ]
    for test in tests:
        test() # forget about exceptions, lol

