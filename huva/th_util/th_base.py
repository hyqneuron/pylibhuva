import torch
from torch.autograd import Variable

def has_nan(val):
    if isinstance(val, Variable):
        val = val.data
    return not (val==val).all()
