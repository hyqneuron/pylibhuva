import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base import *
from .math_ops import *


class Distribution(nn.Module):
    """
    A Distribution can be a Gaussian, Laplacian, Bernoulli and so on. 
    - It computes a representation of the distribution, given some raw input (forward)
    - It computes logP, KLD, NLL, given a representation of the distribution.
    - It takes samples from the distribution given its representation (sample)

    A sub-class must implement:
    - forward
    - logP
    - prior_P
    - sample
    """

    def forward(self, input):
        # returns a representation of the computed distribution
        # E.g. Gaussian returns (mean, logvar, var) while Bernoulli returns just P
        raise NotImplementedError

    def logP(self, x, P):
        raise NotImplementedError

    def P(self, x, P):
        return self.logP(x, P).exp()

    def KLD(self, z, Q, P):
        logQ = self.logP(z, Q)
        logP = self.logP(z, P)
        return logQ - logP

    def NLL(self, x, P):
        return - self.logP(x, P)

    def prior_P(self, template):
        raise NotImplementedError

    def sample(self, P):
        raise NotImplementedError

    def sample_prior(self, template):
        P = self.prior_P(template)
        return self.sample(P)

    def __repr__(self, additional=''):
        return '{} ({})'.format(self.__class__.__name__, additional)


class Coder(nn.Module):
    """
    A Coder combines a list of nn.Modules with a Distribution.  It maps an input value to an output distribution with
    the help that list of layers.
    """

    def __init__(self, layers, distribution):
        super(Coder, self).__init__()
        self.layers       = nn.Sequential(*layers)
        self.distribution = distribution

    def forward(self, x):
        raw = self.layers(x)
        P   = self.distribution(raw)
        return P

    def __getattr__(self, key):
        """
        delegate P, logP, KLD, NLL, prior_P, sample, sample_prior to self.distribution
        """
        try:
            return super(Coder, self).__getattr__(key)
        except:
            assert key in ['P', 'logP', 'KLD', 'NLL', 'prior_P', 'sample', 'sample_prior']
            return getattr(self.distribution, key)


def extract_mean(P):
    return (P[0]      if type(P) in [tuple, list] else 
            P)


"""
==================================================================================================
Distributions

Gaussian, Laplacian, Bernoulli, BinaryContinuous, AltGaussian
==================================================================================================
"""

class Gaussian(Distribution):

    def forward(self, x):
        mean, logvar = x.chunk(2, dim=1)
        return (mean, logvar, logvar.exp())

    def logP(self, x, P):
        return -nl_for_gaussian(x, P)

    def KLD(self, z, Q, P):
        """ closed-form KLD """
        KLD = kld_for_gaussians(Q, P)
        return KLD

    def NLL(self, x, P):
        NLL = nl_for_gaussian(x, P)
        return NLL

    def prior_P(self, template):
        mean, logvar, var = new_as(template.data), new_as(template.data), new_as(template.data)
        mean.fill_(0)
        logvar.fill_(0)
        var.fill_(1)
        return Variable(mean), Variable(logvar), Variable(var)

    def sample(self, P):
        mean, logvar, var = P
        std = var.sqrt()
        noise = Variable(new_as(std.data).normal_())
        return mean + noise * std


class Laplacian(Distribution):

    def forward(self, x):
        mean, logstd = x.chunk(2, dim=1)
        return (mean, logstd, logstd.exp())

    def logP(self, x, P):
        return - logstd - (z - mean).abs() / std + math.log(2)

    def prior_P(self, template):
        mean, logstd, std = new_as(template.data), new_as(template.data), new_as(template.data)
        mean.fill_(0)
        logstd.fill_(0)
        std.fill_(1)
        return Variable(mean), Variable(logvar), Variable(var)

    def sample(self, P):
        mean, logstd, std = P
        return mean + std * sample_unit_laplacian(std.data)


class Bernoulli(Distribution):

    def __init__(self, sigmoid=False):
        """
        sigmoid: bool, if True, will apply sigmoid transform on input before using it as probability
        """
        super(Bernoulli, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, x):
        P = x
        if self.sigmoid:
            P = F.sigmoid(P)
        return P

    def logP(self, x, P):
        return x * P.log() + (1-x) * (1-P).log()

    def KLD(self, z, Q, P, eps=1e-8):
        """ closed-form KLD """
        positive =      Q  * ((    Q+eps) / (    P+eps)).log()
        negative = (1 - Q) * ((1 - Q+eps) / (1 - P+eps)).log()
        return positive + negative

    def prior_P(self, template):
        result = new_as(template.data).fill_(0.5)
        return Variable(result)

    def sample(self, P):
        return (Variable(new_as(P.data).uniform_()) < P).float() # needa return float instead of byte

    def __repr__(self, additional=''):
        return '{} (sigmoid={}, {})'.format(self.__class__.__name__, self.sigmoid, additional)


class BinaryContinuous(Distribution):

    def __init__(self, bernoulli, continuous):
        """
        bernoulli: Bernoulli instance
        continuous: a continuous Distribution. It has to be numerically stable when logP or KLD is computed with x=0,
                    otherwise BinaryContinuous would be numerically unstable.
        """
        super(BinaryContinuous, self).__init__()
        self.bernoulli  = bernoulli
        self.continuous = continuous

    def forward(self, x):
        mean, logxxx = x.chunk(2, dim=1)
        bina_distro = self.bernoulli(mean)
        cont_distro = self.continuous(x)
        mean = extract_mean(bina_distro) * extract_mean(cont_distro) # in cae hvae.pass_sample=False
        # bina_distro = Variable(new_as(bina_distro.data).fill_(1)) # FIXME debugging
        return (mean, bina_distro, cont_distro)

    def logP(self, x, P):
        assert False, 'do not call this yet'
        mean, bina_distro, cont_distro = P
        m = (x != 0).float()
        bina_logP =     self.bernoulli .logP(m, bina_distro)
        cont_logP = m * self.continuous.logP(x, cont_distro)
        # FIXME do NaN-removal to guarantee numerical stability
        return bina_logP + cont_logP

    def KLD(self, z, Q, P):
        """ This unnecessary implementation allows closed-form KLD """
        mean_Q, bina_Q, cont_Q = Q
        mean_P, bina_P, cont_P = P
        m = (z != 0).float()
        bina_KLD =          self.bernoulli.KLD (m, bina_Q, bina_P)
        cont_KLD = bina_Q * self.continuous.KLD(z, cont_Q, cont_P)
        # FIXME do NaN-removal to guarantee numerical stability
        # return cont_KLD # bina_KLD + cont_KLD FIXME debugging
        return bina_KLD + cont_KLD 

    def prior_P(self, template):
        bina_prior = self.bernoulli. prior_P(template)
        cont_prior = self.continuous.prior_P(template)
        mean = extract_mean(bina_prior) * extract_mean(cont_prior)
        return mean, bina_prior, cont_prior

    def sample(self, P):
        mean, bina_distro, cont_distro = P
        m = self.bernoulli .sample(bina_distro)
        s = self.continuous.sample(cont_distro)
        return m * s


def base_to_full(base, s, extra_value):
    N, C, H, W = base.size()
    assert (H % s) == 0 and (W % s) == 0
    Hs, Ws = H / s, W / s
    part = base.view   (N, C, Hs, s, Ws, s)\
               .permute(0, 1, 2,  4, 3,  5).contiguous()\
               .view   (N, C, Hs, Ws, s**2)
    full_size = list(part.size())
    full_size[-1] += 1
    full = Variable(part.data.new().resize_(*full_size))
    full[:,:,:,:,:-1] = part
    full[:,:,:,:, -1] = extra_value
    return full


def full_to_base(full, s):
    part = full[:,:,:,:,:-1].contiguous()
    N, C, Hs, Ws, _ = part.size()
    return part.view   (N, C, Hs, Ws, s, s)\
               .permute(0, 1, 2,  4,  3, 5).contiguous()\
               .view   (N, C, Hs*s,   Ws*s)


class PoolingContinuous(Distribution):
    # FIXME: This is actually better implemented using CategoricalContinuous with preceding SpaceToChannel
    # If we implement CategoricalContinuous, it is easier to test because it can be lowered to BinaryContinuous

    def __init__(self, continuous, stride=2, num_off=None):
        """
        continuous: the continuous distribution to use
        num_off: number of off-components. Usually the same as number of shift locations (stride**2)
        """
        super(PoolingContinuous, self).__init__()
        self.continuous  = continuous
        self.stride      = stride
        if num_off is None: num_off = stride ** 2
        self.num_off = num_off

    def forward(self, x):
        """
        base shape: N x C x H x W
        full shape: N x C x Hs x Ws x (ss+1)
        flat shape: (N*C*Hs*Ws) x (ss+1)

        steps: exponentiate, expand, normalize
        """
        base_conm, base_conl = x.chunk(2, dim=1) # continuous_mean and continuous_logxxx
        N,C,H,W = base_conm.size()
        assert H % self.stride == 0 and W % self.stride == 0
        # exponentiate, expand, normalize
        base_expo = base_conm.exp()
        full_expo = base_to_full(base_expo, self.stride, (self.num_off+1e-5))
        full_prob = full_expo / full_expo.sum(4).expand_as(full_expo)
        # compute mean
        base_prob = full_to_base(full_prob, self.stride)
        base_mean = base_conm * base_prob
        # compute continuous distribution
        cont_distro = self.continuous(x)
        return base_mean, full_prob, cont_distro

    def logP(self, x, P):
        raise NotImplementedError

    def KLD(self, z, Q, P):
        base_mean_Q, full_prob_Q, cont_Q = Q
        base_mean_P, full_prob_P, cont_P = P
        full_KLD = kld_for_categoricals(full_prob_Q, full_prob_P)
        # need to retain KLD due to off-components
        # divide by ss times, because it'll be duplicated ss times and added into base_KLD
        last_KLD    = (full_KLD[:,:,:,:,-1] / (self.stride**2)).unsqueeze(4).expand_as(full_KLD)
        last_KLD    = full_to_base(last_KLD, self.stride)
        base_KLD    = full_to_base(full_KLD, self.stride) + last_KLD
        base_prob_Q = full_to_base(full_prob_Q, self.stride)
        cont_KLD = base_prob_Q * self.continuous.KLD(z, cont_Q, cont_P)
        return base_KLD + cont_KLD

    def prior_P(self, template):
        raise NotImplementedError

    def sample(self, P):
        base_mean, full_prob, cont_distro = P
        base_cons = self.continuous.sample(cont_distro)     # continuous_sample
        # flatten to sample mask
        flat_prob = full_prob.view(-1, full_prob.size()[-1])
        flat_mask = Variable(multinomial_max(flat_prob.data))
        full_mask = flat_mask.view(full_prob.size())
        base_mask = full_to_base(full_mask, self.stride)
        return base_mask * base_cons


class AltGaussian(Gaussian):
    """
    Alternative gaussian distribution, with non-standard prior

    Specifically, the prior is set to approximate a Dirichlet.  Samples from this coder are supposed to be transformed
    using a softmax layer. To account for the distortion introduced by softmax, set softmax_correction=True to alter the
    KLD objective.
    """

    def __init__(self, num_latent, alpha, softmax_correction):
        GaussianCoder.__init__(self)
        K = num_latent
        self.alpha   = alpha
        self.softmax_correction = softmax_correction
        alpha        = torch.Tensor(1, K).fill_(alpha)                      # alpha of dirichlet. default at 1
        prior_mean   = alpha.log() -  alpha.log().sum() / K                 # lognormal mean: default at 0
        prior_var    = ((1 - 2.0 / K) / alpha +                             # variance of normal
                        alpha.reciprocal().sum() / K**2)
        prior_logvar = prior_var.log()                                      # corresponding log variance
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)

    def KLD(self, z, Q, P):
        # this should not happen. But we keep it for backward compatibility
        if P is None: P = self.prior_P(Q)
        KLD = kld_for_gaussians(Q, P)
        if self.softmax_correction:
            prior_mean,     prior_logvar,     prior_var     = P
            posterior_mean, posterior_logvar, posterior_var = Q
            Q_a = 0.5 / posterior_var
            P_a = 0.5 / prior_var    
            Q_b = (z - posterior_mean) / posterior_var
            P_b = (z - prior_mean    ) / prior_var    
            logQ_corr = (Q_b*Q_b) / (4*Q_a) - 0.5 * Q_a.log()
            logP_corr = (P_b*P_b) / (4*P_a) - 0.5 * P_a.log()
            KLD += (logQ_corr - logP_corr)
        return KLD

    def prior_P(self, template):
        mean   = Variable(self.prior_mean.  expand_as(template.data))
        logvar = Variable(self.prior_logvar.expand_as(template.data))
        var    = Variable(self.prior_var.   expand_as(template.data))
        return mean, logvar, var

