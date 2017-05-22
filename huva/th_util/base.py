import torch
from torch.autograd import Variable

def has_nan(val):
    if isinstance(val, Variable):
        val = val.data
    return not (val==val).all()

def new_as(val):
    assert torch.is_tensor(val)
    return val.new().resize_as_(val)
