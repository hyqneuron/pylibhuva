import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base import *
from .math_ops import *

class Distribution(nn.Module):

    def forward(self, input):
        # returns a representation of the computed distribution
        raise NotImplementedError

    def P(self, x, P):
        return self.logP(x, P).exp()

    def logP(self, x, P):
        raise NotImplementedError

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
        raise NotImplementedError


class Coder(nn.Sequential):
    """
    A Coder is either an encoder or a decoder. It has a list of internal layers, which it uses to convert an input value
    to an output distribution.
    """

    def __init__(self, layers, distribution, critic=False):
        """
        critic: if a critic layer is used in this coder

        We should implement critic as a plugin for generality. But right now the critic plugin is the only one I can
        think of, so we go for a specific implementation instead.
        """
        super(Coder, self).__init__(*layers)
        self.distribution = distribution
        self.critic = critic
        if critic:
            self.critic_value = None

    def forward(self, x):
        output = super(Coder, self).forward(x)
        # if this coder has an interval layer that outputs a critic value, extract it
        if self.critic:
            assert type(output) in [tuple, list], 'a coder containing critic must output a tuple/list, with the last element being a critic value'
            assert len(output) >= 2
            self.critic_value = output[-1]
            output = output[:-1]
            output = output[0] if len(output)==1 else output
        return self.distribution(output)

    def __getattr__(self, key):
        """
        delegate P, logP, KLD, NLL, prior_P, sample, sample_prior to self.distribution
        """
        assert key in ['P', 'logP', 'KLD', 'NLL', 'prior_P', 'sample', 'sample_prior']
        return getattr(self.distribution, key)


def extract_mean(P):
    return P[0] if type(P) in [tuple, list] else P


def split_mean_logvar(mean_logvar, num_latent):
    assert mean_logvar.size(1) == num_latent * 2
    mean   = mean_logvar[:, :num_latent]
    logvar = mean_logvar[:, num_latent:]
    return mean, logvar


"""
==================================================================================================
Distributions
==================================================================================================
"""

class Gaussian(Distribution):

    def __init__(self, num_latent):
        super(GaussianCoder, self).__init__()
        self.num_latent = num_latent

    def forward(self, x):
        mean, logvar = split_mean_logvar(x, self.num_latent)
        return (mean, logvar, logvar.exp())

    def logP(self, x, P):
        return -nl_for_gaussian(x, P)

    def KLD(self, z, Q, P):
        KLD = kld_for_gaussians(Q, P)
        return KLD

    def NLL(self, x, P):
        NLL = nl_for_gaussian(x, P)
        return NLL

    def prior_P(self, template):
        mean, logvar, var = template
        mean, logvar, var = new_as(mean.data), new_as(logvar.data), new_as(var.data)
        mean.fill_(0)
        logvar.fill_(0)
        var.fill_(1)
        return Variable(mean), Variable(logvar), Variable(var)

    def sample(self, P):
        mean, logvar, var = P
        std = var.sqrt()
        noise = Variable(new_as(std.data).normal_())
        return mean + noise * std

    def sample_prior(self, template):
        assert template.size(1) == self.num_latent
        return Variable(new_as(template.data).normal_())


class Laplacian(Distribution):

    def __init__(self, num_latent):
        super(Laplacian, self).__init__()
        self.num_latent = num_latent

    def forward(self, x):
        mean, logstd = split_mean_logvar(x, self.num_latent)
        return (mean, logstd, logstd.exp())

    def logP(self, x, P):
        return - logstd - (z - mean).abs() / std + math.log(2)

    def prior_P(self, template):
        mean, logstd, std = template
        mean, logstd, std = new_as(mean.data), new_as(logstd.data), new_as(std.data)
        mean.fill_(0)
        logstd.fill_(0)
        std.fill_(1)
        return Variable(mean), Variable(logvar), Variable(var)

    def sample(self, P):
        mean, logstd, std = P
        return mean + std * sample_unit_laplacian(std.data)

    def sample_prior(self. template):
        return Variable(sample_unit_laplacian(template.data))


class Bernoulli(Distribution):

    def __init__(self, sigmoid=False):
        super(Bernoulli, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, x):
        P = x
        if self.sigmoid:
            P = F.sigmoid(P)
        return P

    def logP(self, x, P):
        return x * P.log() + (1-x) * (1-P).log()

    def prior_P(self, template):
        result = new_as(template.data).fill_(0.5)
        return Variable(result)

    def sample(self, P):
        return Variable(new_as(P.data)) < P

    def sample_prior(self, template):
        raise return Variable(new_as(template).uniform_() > 0.5)

    def __repr__(self):
        return PSequential.__repr__(self, 'sigmoid={}'.format(self.sigmoid))


class AltGaussian(Gaussian):
    """
    Alternative gaussian distribution, with non-standard prior

    Specifically, the prior is set to approximate a Dirichlet.  Samples from this coder are supposed to be transformed
    using a softmax layer. To account for the distortion introduced by softmax, set softmax_correction=True to alter the
    KLD objective.
    """

    def __init__(self, num_latent, alpha, softmax_correction):
        GaussianCoder.__init__(self, num_latent)
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
        mean, logvar, var = template
        mean   = Variable(self.prior_mean.  expand_as(mean.data))
        logvar = Variable(self.prior_logvar.expand_as(logvar.data))
        var    = Variable(self.prior_var.   expand_as(var.data))
        return mean, logvar, var

    def sample_prior(self, template):
        assert template.size(1) == self.num_latent
        return Variable(new_as(template.data).normal_(self.prior_mean[0,0], math.sqrt(self.prior_var[0,0])))


