import torch
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict
import math


def new_add_channels(x, dim, additional_channels, init_val=None, add_val=None):
    """
    Build a new tensor, whose size is similar to x.
    However, at dimension=dim, expand size by additional_channels
    If init_val is not None, initialize all values to init_val
    If add_val is not None, initialize the added cells to add_val
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
    def max_wrapper(x, additional_channels=0, add_val=None):
        assert torch.is_tensor(x)
        if additional_channels > 0:
            expanded = new_add_channels(x, 1, additional_channels, add_val=add_val)
            return func_max(x, expanded[:, :x.size(1)])
        else:
            return func_max(x, x.new().resize_as_(x))
    return max_wrapper


def gumbel_noise(x, eps=1e-10):
    """ 
    x is [N, ]
    """
    assert torch.is_tensor(x)
    noise = x.new().resize_as_(x).uniform_(0, 1)
    return -(-(noise + eps).log() + eps).log()


@modify_max
def gumbel_max(x, out):
    """ x: logit, or output of log_softmax """
    noisy_x = x + gumbel_noise(x)
    max_x = noisy_x.max(1)[0].expand_as(x)
    return torch.eq(max_x, noisy_x, out=out)


@modify_max
def multinomial_max(p, out):
    """ x: probability for categories """
    out.zero_()
    labels  = p.multinomial(1)
    out.scatter_(1, labels, 1)
    return out


@modify_max
def plain_max(x, out):
    """ x: both p and logit would work, since we're just taking max """
    max_x = x.max(1)[0].expand_as(x)
    return torch.eq(max_x, x, out=out)


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
    term = term1 + term2 - 1
    if do_sum:
        return term.sum()*0.5
    else:
        return term * 0.5


def kld_for_unit_gaussian(mean, logvar, var, do_sum=True):
    """
    Compute the KL-divergence from a diagonal Gaussian to an isotropic Gaussian

    See Autoencoding Variational Bayes by Kingma for derivation.
    """
    term = -0.5 * (1 + logvar - mean*mean - var)
    if do_sum:
        return term.sum()
    else:
        return term


def kld_for_categoricals(q, p):
    """
    - (q * (p / (q+1e-8)).log()).sum()
    """
    return (q * (q / (p+1e-8)).log()).sum()


def kld_for_uniform_categorical(q):
    # FIXME the line below is for debuggin only
    if True: # if debug
        m = q
        if isinstance(m, Variable): m = m.data
        m.add_(1e-7)
        assert m.gt(0).all() # all q must > 0
        assert (m.sum(1)-1).abs().lt(1e-4).all(), 'summed to.. {}'.format(m.sum(1)) # all sum to 1
    C = q.size(1) # number of channels
    return (q * (q * C).log()).sum()


def nl_for_gaussian(x, (mean, logvar, var)):
    """
    Compute the negative logarithm of x under the Gaussian distribution
    P(x) = [1/(2pi*var)**0.5] * exp((x-mean)**2 / (2*var))
         = [(2pi*var)**-0.5] * exp(-(x-mean)**2 / (2*var))
    -logP(x) = - ( -0.5*(log(2pi)+logvar) - 0.5*(x-mean)**2 / var)
             = 0.5 * ( log(2pi) + logvar + (x-mean)**2/var )
    """
    term1 = math.log(2*math.pi) + logvar
    term2 = (x-mean)**2 / (var + 1e-8)
    return 0.5 * (term1+term2).sum()
