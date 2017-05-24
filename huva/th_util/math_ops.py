import torch
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict
import math


def new_add_channels(x, dim, additional_channels, init_val=None, add_val=None):
    """
    Purpose: this function allows us to mix categorical variables with continuous variables in a single Tensor.

    Say the tensor we're dealing with is NxC, where C is the number of categories. Upon sampling, Each C-vector will
    become one-hot. In order to mix continuous variables in the same tensor, we make a Nx(C+S) tensor, where S is the
    number of continuous components.

    Operations: build a new tensor, whose size is similar to x    (e.g. NxC).
    However, at dimension=dim, expand size by additional_channels (e.g. NxC -> Nx(C+S))
    If init_val is not None, initialize all values to init_val    (e.g. the NxC part)
    If add_val is not None, initialize the added cells to add_val (e.g. the NxS part)
    Both init_val and add_val can be a single scalar or a tensor of the right size
    """
    sizes = list(x.size())
    sizes[dim] += additional_channels
    result = x.new().resize_(*sizes)
    if init_val is not None:
        result[:] = init_val
    if add_val is not None:
        starting_pos = x.size(dim)
        slices = tuple([slice() if i != dim else slice(starting_pos, None) for i in xrange(dim)])
        result[slices] = add_val
    return  result


def modify_max(func_max):
    """
    this is a wrapper for the max functions below
    It allows *_max to expand the size of the input along dimension 1
    """
    def max_wrapper(x, additional_channels=0, add_val=None, T=1):
        assert torch.is_tensor(x)
        if additional_channels > 0:
            expanded = new_add_channels(x, 1, additional_channels, add_val=add_val)
            return func_max(x, expanded[:, :x.size(1)], T=T)
        else:
            return func_max(x, x.new().resize_as_(x), T=T)
    return max_wrapper


"""
===============================================================================
Several categorical sampling methods
===============================================================================
"""

def gumbel_noise(x, eps=1e-10):
    """ 
    x is [N, ]
    """
    assert torch.is_tensor(x)
    noise = x.new().resize_as_(x).uniform_(0, 1)
    return -(-(noise + eps).log() + eps).log()


@modify_max
def gumbel_max(x, out, T=None):
    """ x: logit, or output of log_softmax """
    noisy_x = x + gumbel_noise(x)
    max_x = noisy_x.max(1)[0].expand_as(x)
    return torch.eq(max_x, noisy_x, out=out).float()


@modify_max
def gumbel_softmax(x, out, T=1):
    """ x: logit, or output of log_softmax """
    noisy_x = (x + gumbel_noise(x)) / (T+1e-8)
    softmax_x = F.softmax(Variable(noisy_x)).data
    return softmax_x # return the thing without making it one-hot


@modify_max
def multinomial_max(p, out, T=None):
    """ x: probability for categories """
    out.zero_()
    labels  = p.view(p.size(0), -1).multinomial(1).view(p.size())
    out.scatter_(1, labels, 1)
    return out.float()


@modify_max
def plain_max(x, out, T=None):
    """ x: both p and logit would work, since we're just taking max """
    max_x = x.max(1)[0].expand_as(x)
    return torch.eq(max_x, x, out=out).float()

"""
===============================================================================
Gaussian costs for VAEs
===============================================================================
"""

def kld_for_gaussians((mean1, logvar1, var1), (mean2, logvar2, var2), do_sum=True):
    """
    Compute the KL-divergence between two diagonal Gaussians.

    arguments: (q_mean, q_logvar, q_var), (p_mean, p_logvar, p_var) 
    We require var in addition to logvar, because var is computed in the sampling pass, so we don't have to compute it
    again as logvar.exp()

    See https://stats.stackexchange.com/a/7443/125143 for derivation
    """
    diff = mean1 - mean2
    term1 = logvar2 - logvar1
    term2 = (var1 + (diff*diff)) / (var2+1e-8 )
    result = (term1 + term2 - 1) * 0.5
    if do_sum:
        result =  result.sum()
    return result


def kld_for_unit_gaussian(mean, logvar, var, do_sum=True):
    """
    Compute the KL-divergence from a diagonal Gaussian to an isotropic Gaussian

    See Autoencoding Variational Bayes by Kingma for derivation.
    """
    result = -0.5 * (1 + logvar - mean*mean - var)
    if do_sum:
        result =  result.sum()
    return result


def nl_for_gaussian(x, (mean, logvar, var), do_sum=True):
    """
    Compute the negative logarithm of P(x|z) under the Gaussian distribution
    P(x|z) = [1/(2pi*var)**0.5] * exp((x-mean)**2 / (2*var))
           = [(2pi*var)**-0.5] * exp(-(x-mean)**2 / (2*var))
    -logP(x|z) = - ( -0.5*(log(2pi)+logvar) - 0.5*(x-mean)**2 / var)
               = 0.5 * ( log(2pi) + logvar + (x-mean)**2/var )
    """
    term1 = math.log(2*math.pi) + logvar
    term2 = (x-mean)**2 / (var + 1e-10)
    result = 0.5 * (term1+term2)
    if do_sum:
        result = result.sum()
    return result

"""
===============================================================================
Categorical costs for VAEs
===============================================================================
"""

def kld_for_categoricals(q, p, do_sum=True, sample_mask=None, eps=1e-10):
    """
    - (q * (p / q).log()).sum()
    = (q * (q / p).log()).sum()

    persample should be set True if we want a high-variance, per-sample estimate of KLD.
    """
    result = (q / (p+eps) + eps).log()
    result = result * (q if sample_mask is None else sample_mask)
    if do_sum:
        result = result.sum()
    return result


def kld_for_uniform_categorical(q, do_sum=True, sample_mask=None, eps=1e-10):
    C = q.size(1) # number of channels
    result = (q * C + eps).log()
    result = result * (q if sample_mask is None else sample_mask)
    if do_sum:
        result = result.sum()
    return result



