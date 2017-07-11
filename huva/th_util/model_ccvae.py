import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base import *
from .model import *
from .math_ops import *
#from .distribution import *
from .functions import FuncSTCategorical, FuncOneHotSTCategorical
import os # for debugging with os.getenv
import itertools


"""
I am a bit troubled by the seemingly excessive separation of concern in this file. Sure it makes everything quite a bit
more extensible, but it comes at a huge cost of readability to those who don't implement VAEs all-day. 

Indeed this is a presentational issue plaguing most, if not all, programming languages. Separation of concern should not
lead to code that's scattered all over the place. Unfortunately this is the case for all non-graphical languages right
now.
"""


"""

Non-hierarchical VAE:

    A non-hierarchical VAE (the VAE class) has an encoder and a decoder. 
    
    Encoder has the following responsibility:
    - computes Q given x            : forward(self, input)
    - sample z from Q               : sample(self, P)               # P is set to Q
    - compute KLD from Q to P       : KLD(self, z, Q, P)

    Decoder has the following responsibility:
    - reconstruct input given z     : forward(self, input)
    - compute input's cross entropy : NLL(self, x, P)    # P is set to reconstructed distribution of input
    Note that (reconstructed x) is a distribution

    The VAE class is a non-hierarchical VAE that wraps around an encoder and decoder


Hierarchical VAE

    A hierarchical VAE (the hVAE class) has a series of stages. Every stage is a VAE.
"""


"""
TODOs:
[x] return NLL and KLD as (N,1)
[x] make sure VAE does sum/mean on its own
[x] Use logP to compute KLD in Distribution
[ ] make sample_prior return according to a template
[x] Support prior KLD
    [x] prior_P(template)
    [x] corresponding changes in VAE, SleepVAE, hVAE and so on
[x] Complete sVAE
    [x] SleepVAE
    [x] sVAE
[x] Complete hsVAE
    [x] hSleepVAE
    [x] hsVAE
[-] alternative costs
    [x] support critic in Coder
    [x] support critic as an extension of VAE. 
    [ ] implement importance-weighted updates
    [ ] implement critic-based updates

"""

"""
===============================================================================
Abstract classes

BaseVAE, Coder, Distribution
===============================================================================
"""

class BaseVAE(nn.Module):
    def forward(self, input):
        """
        algorithm-specific

        input: input
        returns: Q, z, P
            x: actual input used
            Q: approximate distribution Q(z|x)
            z: sample from Q(z|x)
            P: model distribution P(x|z)
        """
        raise NotImplementedError

    def get_losses(self):
        """
        algorithm-specific

        returns: (loss_overall, KLD, NLL [, loss_zs])
            loss_overall: KLD + NLL
            KLD: KL divergence in the latent layers KLD(Q||P). If there are multiple layers of latent code, this is a
                sum of loss_zs
            NLL: - logP(x|z)
            loss_zs: in the case there are multiple layers of latent code, loss_zs is a list of KLDs from different
                latent layers
        """
        raise NotImplementedError

    def sample_p(self, z, add_noise):
        """
        input:
            z: latent code from which to perform generation
            add_noise: bool, whether to add noise at P(x|z) (the final generation step)
        returns: a sample obtained by passing z down the chain of generation, and taking samples where appropriate
        """
        raise NotImplementedError

    def sample_prior(self, template):
        """
        input: template, a Variable of the shape and type same as expected returned value

        returns: a sample of latent z that can be sent to sample_p(z) for generating samples in input space
        """
        raise NotImplementedError


"""
===============================================================================
VAEs

done: VAE(WakeVAE), SleepVAE, sVAE, hVAE(hWakeVAE), hSleepVAE, hsVAE
todo: IWVAE, CriticVAE, 
===============================================================================
"""

def extract_mean(P):
    return P[0] if type(P) in [tuple, list] else P


class VAE(BaseVAE):

    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """
        encode, sample, decode
        save stuffs along the way so that we can easily compute loss later
        """
        Q = self.encoder(x)
        z = self.encoder.sample(Q)
        P = self.decoder(z)
        self.x, self.Q, self.P, self.z = x, Q, P, z
        return x, Q, z, P

    def get_losses(self, do_mean=True):
        P    = self.prior_P(self.Q)
        KLD  = self.KLD(self.z, self.Q, P)
        NLL  = self.NLL(self.x, self.P)
        if do_mean:
            KLD = KLD.mean()
            NLL = NLL.mean()
        loss = KLD + NLL
        return loss, KLD, NLL

    def KLD(self, z, Q, P):
        return self.encoder.KLD(z,Q,P).sum(1)

    def NLL(self, x, P):
        return self.decoder.NLL(x, P).sum(1)

    def prior_P(self, template):
        return self.encoder.prior_P(template)

    def sample_p(self, z, add_noise=False):
        P = self.decoder(z)
        return self.decoder.sample(P) if add_noise else extract_mean(P)

    def sample_prior(self, template):
        return self.encoder.sample_prior(template)

    def clear_states(self):
        self.x = None
        self.Q = None
        self.z = None
        self.P = None


class hVAE(nn.Sequential, BaseVAE): # inherits nn.Sequential for its nice printing (__repr__ method)

    def __init__(self, stages, pass_sample=False):
        nn.Sequential.__init__(self, *stages)
        assert len(stages) >= 1, 'Minimum 1 stage is required'
        self.stages = stages
        self.pass_sample = pass_sample

    def forward(self, x):
        self.x = x
        inp = x
        Qs, zs, Ps = [], [], []
        for stage in self.stages:
            _, Q, z, P = stage(inp)
            Qs.append(Q)
            zs.append(z)
            Ps.append(P)
            if self.pass_sample: # sample
                inp = z
            else:                # distro mean
                inp = extract_mean(Q)
        return x, Qs, zs, Ps

    def get_losses(self, do_mean=True):
        KLDs = []
        KLD = 0
        # compute KLD at every stage
        for i, lower in enumerate(self.stages):
            P = self.stages[i+1].P if i+1 < len(self.stages) else lower.prior_P(lower.Q)
            KLDi = lower.KLD(lower.z, lower.Q, P)
            KLDs.append(KLDi)
            KLD += KLDi
        # compute NLL at the lowest stage
        lowest = self.stages[0]
        NLL = lowest.NLL(self.x, lowest.P)
        if do_mean:
            KLD = KLD.mean()
            NLL = NLL.mean()
            for i in xrange(len(KLDs)):
                KLDs[i] = KLDs[i].mean()
        loss = KLD + NLL
        return loss, KLD, NLL, KLDs

    def sample_p(self, z, add_noise=False):
        for i, stage in enumerate(reversed(self.stages)):
            P = stage.decoder(z)
            z = stage.decoder.sample(P)
        return z if add_noise else extract_mean(P)

    def sample_prior(self, template):
        return self.stages[-1].sample_prior(template)

    def clear_states(self):
        self.x = None


"""
===============================================================================
Debug coders

MSE, Fake
===============================================================================
"""

class MSEDecoder(PSequential):
    """
    For debugging VAEs. Propagates input from layers as-is.
    - KLD: doesn't support
    - NLL: MSE
    """

    def __init__(self, layers):
        super(MSEDecoder, self).__init__(*layers)

    def KLD(self, z, Q, P):
        raise NotImplementedError, '{} does not support encoding z'.format(self.__class__)

    def NLL(self, x, P):
        NLL = (P - x).pow(2).mean(1) # MSE, note the change of shape is somewhat dangerous
        return NLL # NLL / x.size(0)

    def sample(self, P):
        return P

    def sample_prior(self, num_samples): # FIXME
        raise NotImplementedError


class FakeCoder(PSequential):
    """ 
    For debugging VAEs. Propagates input from layers as-is.
    - KLD: 0
    - NLL: doesn't support
    """

    def __init__(self, layers):
        super(FakeCoder, self).__init__(*layers)
        self.layers = layers

    def KLD(self, z, Q, P=None):
        return Variable(Q.data.new().resize_(Q.size(0), 1).fill_(0)) # return 0 tensor

    def NLL(self, x, P):
        raise NotImplementedError

    def sample(self, P):
        return P

    def sample_prior(self, num_samples): # FIXME
        if hasattr(self.layers[-1], 'sample_prior'):
            return self.layers[-1].sample_prior(num_samples)
        else:
            raise NotImplementedError

"""
===============================================================================
SOM and WTA

SOMSmoother, WTALayer, SWTAInputLayer, SWTACoder, WTACoder
===============================================================================
"""

class SOMSmoother(nn.Module):

    def __init__(self, som_dims, std=1.0, std_min=1.0, decay_rate=1.0, decay_interval=0):
        """
        This module applies a SOM-based smoothing convolution on input units organized on a low-dimensional grid.

        Arguments:
            som_dims: int tuple for the size of the SOM grid. Can be 1D (K,) or 2D (H,W)
            std: bandwidth of gaussian smoothing kernel
            std_min: minimum bandwidth. See decay below
            decay_rate: every time when std decays, it is multiplied by decay rate
            decay_interval: after this number of intervals, perform a decay step. If 0, no decay is performed.
        """
        super(SOMSmoother, self).__init__()
        self.som_dims       = som_dims
        K = reduce(lambda x1,x2: x1 * x2, som_dims, 1)
        self.K              = K
        self.std            = std
        self.std_min        = std_min
        self.decay_rate     = decay_rate
        self.decay_interval = int(decay_interval)
        self.iterations = 0
        if len(som_dims)==1:
            distance_matrix = self.get_1d_dist_matrix(*som_dims)
        elif len(som_dims)==2:
            distance_matrix = self.get_2d_dist_matrix(*som_dims)
        else:
            assert False, 'currently only support 1D and 2D SOM, but requested {}D'.format(len(som_dims))
        self.register_buffer('distance_matrix', distance_matrix)
        self.reset_weight()

    def reset_weight(self, std=None):
        std = std or self.std
        self.std = std
        # compute gaussian value
        som_matrix = (-self.distance_matrix / (std*std)).exp()
        assert (som_matrix == som_matrix.t()).all()
        # normalize sums so that a row sums to 1
        row_sum    = som_matrix.sum(1)
        row_matrix = som_matrix / row_sum.expand_as(som_matrix)
        self.register_buffer('row_matrix', row_matrix)

    def forward(self, input):
        smoothed = input.mm(Variable(self.row_matrix))          # smooth it
        if self.train and self.decay_interval>0:                # decay if in training mode
            self.iterations += 1
            if self.iterations % self.decay_interval == 0:
                self.std = max(self.std_min, self.std * self.decay_rate)
                self.reset_weight(self.std)
        return smoothed

    def get_1d_dist_matrix(self, K):
        distance_matrix = torch.zeros(K, K)
        for x1 in xrange(K):
            for x2 in xrange(K):
                dist = (x1-x2)**2
                distance_matrix[x1,x2] = dist
        return distance_matrix

    def get_2d_dist_matrix(self, H,W):
        K = H * W
        distance_matrix = torch.zeros(K, K)
        for y1, x1 in itertools.product(xrange(H), xrange(W)):
            for y2, x2 in itertools.product(xrange(H), xrange(W)):
                dist = (x1-x2)**2 + (y1-y2)**2
                i1 = y1*W + x1
                i2 = y2*W + x2
                distance_matrix[i1,i2] = dist
        return distance_matrix

    def __repr__(self, additional=""):
        prop_str = "{}(som_dims={}, stds=({},{}), decay_rate={}, decay_interval={}, {})".format(
                self.__class__.__name__, self.som_dims, self.std, self.std_min, self.decay_rate, self.decay_interval, additional)
        return prop_str


class WTALayer(nn.Module):

    def __init__(self, num_latent):
        """
        Winner-Take-All. Non-maximal units are wiped to 0. Only maximal unit outputs as-is
        """
        super(WTALayer, self).__init__()
        self.num_latent    = num_latent

    def forward(self, input):
        mask = plain_max(input.data)
        return input * Variable(mask)

    def sample_prior(self, num_samples): # FIXME
        assert num_samples <= self.num_latent
        sample = torch.zeros(num_samples, self.num_latent)
        for i in xrange(num_samples):
            sample[i,i] = 1
        return Variable(sample)

    def __repr__(self, additional=""):
        prop_str = "{}(num_latent={}, {})".format(self.__class__.__name__, self.num_latent, additional)
        return prop_str


class HierarchicalWTALayer(nn.Module):

    def __init__(self, num_groups, num_categories, smoother, p_transform, exp=True):
        super(HierarchicalWTALayer, self).__init__()
        self.num_groups     = num_groups        # number of groups
        self.num_categories = num_categories    # number of categories per group
        self.smoother       = smoother
        self.p_transform    = p_transform
        self.exp            = exp

    def forward(self, input):
        """
        3 views:
        total view: [N, num_groups*num_categories], one row per sample
        group view: [N*num_groups, num_categories], one row per group
        gsum  view: [N, num_groups]

        steps:
        1. for each unit, compute P(unit|x)
        2. within-group smoothing
        3. compute P(group|x) by summing P(unit|x) over units within a group
        4. identify winning group
        5. identify winning unit within winning group
        """
        N = input.size(0)
        num_groups, num_categories = self.num_groups, self.num_categories
        assert input.size(1) == num_groups * num_categories
        return self.method_exp(input) if self.exp else self.method_linear(input)

    def method_exp(self, input):
        """
        Linear smoothing, exponential mask
        """
        N = input.size(0)
        num_groups, num_categories = self.num_groups, self.num_categories
        total_input  = input
        group_input  = total_input.view(N*num_groups, num_categories)
        # smooth
        group_smooth = self.smoother(group_input)                           # 2. within-group smoothing, FIXME FakeSOMCoder gives 1-hot output!
        total_smooth = group_smooth.view(N, num_groups*num_categories)
        # exponentiate on total_smooth
        total_p      = self.p_transform(total_smooth)
        group_p      = total_p.view(N*num_groups, num_categories)
        # identify winners using p
        gsum         = group_p.sum(1).view(N, num_groups)                   # 3. compute P(group|x)
        gsum_mask    = Variable(plain_max(gsum.data))                       # 4. identify winning group
        group_mask   = Variable(plain_max(group_p.data))                    #    identify winning unit within every group
        total_mask   = (group_mask.view(N, num_groups*num_categories) *     # 5. identify winning unit within winnint group
                        gsum_mask.unsqueeze(2).expand(N, num_groups, num_categories).contiguous().view(N, num_groups*num_categories))
        return total_smooth * total_mask

    def method_linear(self, input):
        """
        Linear smoothing, linear mask
        """
        N = input.size(0)
        num_groups, num_categories = self.num_groups, self.num_categories
        total_input  = input
        group_input  = total_input.view(N*num_groups, num_categories)
        # smooth
        group_smooth = self.smoother(group_input)                           # 2. within-group smoothing, FIXME FakeSOMCoder gives 1-hot output!
        total_smooth = group_smooth.view(N, num_groups*num_categories)
        # identify winners
        gsum         = group_smooth.sum(1).view(N, num_groups)              # 3. compute P(group|x)
        gsum_mask    = Variable(plain_max(gsum.data))                       # 4. identify winning group
        group_mask   = Variable(plain_max(group_smooth.data))               #    identify winning unit within every group
        total_mask   = (group_mask.view(N, num_groups*num_categories) *     # 5. identify winning unit within winnint group
                        gsum_mask.unsqueeze(2).expand(N, num_groups, num_categories).contiguous().view(N, num_groups*num_categories))
        return total_smooth * total_mask

    def sample_prior(self, num_samples): # FIXME
        K = self.num_groups * self.num_categories
        assert num_samples <= K, 'not sure how to take more than {} samples from prior for WTA'.format(K)
        prior_sample = torch.zeros(num_samples, K)
        for i in xrange(num_samples):
            prior_sample[i,i] = 1 
        return Variable(prior_sample)


class SWTAInputLayer(nn.Module):

    def __init__(self, num_latent, smoother, p_transform, split, bn_mean, bn_logvar):
        super(SWTAInputLayer, self).__init__()
        self.num_latent  = num_latent    # where to split
        self.split       = split         # if to split
        self.bn_mean     = bn_mean
        self.bn_logvar   = bn_logvar
        self.smoother    = smoother
        self.p_transform = p_transform   # transforms 'mean' to P

    def forward(self, input):
        # get mean, logvar
        if self.split:
            mean, logvar = split_mean_logvar(input, self.num_latent)
        else:
            mean, logvar = input
        # apply BN
        mean   = self.bn_mean  (mean.contiguous())
        logvar = self.bn_logvar(logvar.contiguous())
        # apply smoothing
        mean   = self.smoother(mean)
        logvar = self.smoother(logvar)
        # transform mean to P
        P      = self.p_transform(mean)
        return P, mean, logvar


class SWTACoder(PSequential):
    """ 
    Simplified WTACoder

    Removed these from WTACoder:
    - mean_normalizer, logvar_normalizer: if needed, do this outside of SWTA
    - num_continuous:               removed texture component. If needed, do it in parallel to SWTA
    - share_logvar:                 does not support
    - concrete, gumbel*:            does not support
    - bypass_mode, mult*:           does not support

    Supported options:
    - sample mode   : ['stochastic','deterministic']    # does not support gumbel or concrete
    - upward mode   : ['P', 'mean', 'sample']           # upward output
    - persample kld : [True, False]                     # persample KLD
    - use gaus      : [True, False]                     # include gaussian cost

    Inputs:
    1. P        (Categorical probability)
    2. mean     (Gaussian mean)
    3. logvar   (Gaussian logvar)

    forward:
    - Q = (upward_value, P, mean, logvar, var)
    """

    def __init__(self, layers, num_latent, sample_mode, upward_mode, persample_kld, use_gaus):
        super(SWTACoder, self).__init__(*layers)
        self.layers = layers
        self.num_latent     = num_latent
        self.sample_mode    = sample_mode
        self.upward_mode    = upward_mode
        self.persample_kld  = persample_kld
        self.use_gaus       = use_gaus
        assert sample_mode in ['stochastic','deterministic']
        assert upward_mode in ['P', 'mean'] # does not support upward_mode='sample' yet
        # we will adaptively estimate the mean and var of gaussian samples
        self.adaptive_mean = 0.0
        self.adaptive_var  = 1.0

    def forward(self, input):
        inp = input
        for layer in self.layers:
            inp = layer(inp)
        P, mean, logvar = inp
        """
        print '=='
        print P.var().data[0], mean.var().data[0], logvar.var().data[0]
        print P.mean().data[0], mean.mean().data[0], logvar.mean().data[0]
        """
        var = logvar.exp()
        upward_value = P if self.upward_mode=='P' else mean
        return (upward_value, P, mean, logvar, var)

    def sample(self, P):    # P is returned by forward
        upward_value, P, mean, logvar, var = P
        # take cat_mask as 1-hot categorical sample
        if self.sample_mode=='stochastic':
            cat_mask = gumbel_max(P.data.log())
        elif self.sample_mode=='deterministic':
            cat_mask = plain_max (P.data)
        else:
            assert False
        cat_mask = Variable(cat_mask)
        # save categorical mask for per-sample KLD computation
        self.cat_mask = cat_mask
        # take gaussian sample
        gaussian = mean
        if self.use_gaus:
            std   = var.sqrt()
            noise = Variable(std.data.new().resize_as_(std.data).normal_())
            gaussian = mean + (std * noise)
        return cat_mask * gaussian

    def KLD(self, z, Q, P=None):
        Q_upward_value, Q_cat, Q_mean, Q_logvar, Q_var = Q
        sample_mask = self.cat_mask if self.persample_kld else Q_cat
        # compute adaptive statistics
        if self.train:
            current_vals = (sample_mask * Q_mean).sum(1)
            current_mean = current_vals.mean().data[0]
            current_var  = current_vals.var() .data[0]
            self.adaptive_mean += 0.05 * (current_mean - self.adaptive_mean)
            self.adaptive_var  += 0.05 * (current_var  - self.adaptive_var )
        # fill up P if it is None, for top-layer encoder
        if P is not None:
            P_upward_value, P_cat, P_mean, P_logvar, P_var = P
        else:
            P_cat = Variable(new_as(Q_cat.data).fill_(1.0 / self.num_latent ))
            if self.use_gaus:
                P_mean   = Variable(new_as(Q_mean.data).fill_(self.adaptive_mean))
                P_var    = Variable(new_as(Q_mean.data).fill_(self.adaptive_var))
                P_logvar = Variable(new_as(Q_mean.data).fill_(math.log(self.adaptive_var)))
        # losses
        loss_cat  = kld_for_categoricals(Q_cat, P_cat, sample_mask=sample_mask)
        loss_gaus = 0.0 if not self.use_gaus else kld_for_gaussians((Q_mean, Q_logvar, Q_var),(P_mean, P_logvar, P_var))
        # save some stats, saved as floats so no need to clear in clear_states
        self.loss_cat  = loss_cat .data[0]
        self.loss_gaus = loss_gaus.data[0] if self.use_gaus else 0.0
        return loss_cat + loss_gaus

    def NLL(self, x, P):
        raise NotImplementedError, '{} does not support decoding x'.format(self.__class__.__name__)

    def sample_prior(self, num_samples): # FIXME
        assert num_samples <= self.num_latent, 'not sure how to take more than self.num_latent samples from prior for SOM'
        prior_sample = torch.zeros(num_samples, self.num_latent)
        mean, std    = self.adaptive_mean, math.sqrt(self.adaptive_var)
        #gaus_samples = torch.Tensor(num_samples).normal_(mean, std)
        for i in xrange(num_samples):
            prior_sample[i,i] = mean #gaus_samples[i]
        return Variable(prior_sample)

    def clear_states(self):
        if hasattr(self, 'cat_mask'):
            self.cat_mask = None

    def __repr__(self, additional=''):
        extra_str = "sample_mode={}, upward_mode={}, persample_kld={}, use_gaus={}".format(
                self.sample_mode,self.upward_mode, self.persample_kld, self.use_gaus)
        return super(SWTACoder, self).__repr__(additional="{}, {}".format(extra_str, additional))


class WTACoder(PSequential):
    """
    At this point WTACoder has grown to such a monstrous size we must consider refactoring it. We can create several
    subclasses of it depending on the sampling mode, though that would break the external interface. An alternative
    method, which preserves the external interface, is to use internal delegation. For every sampling mode, we delegate
    forward_hook, sample, and KLD

    These are the sampling modes we currently have:
    - deterministic:    fake loss for debugging
    - stochastic:       one-hot stochastic
    - gumbel:           relaxed one-hot, using discrete KLD
    - concrete:         relaxed one-hot, using concrete KLD

    persample_kld is not itself a sampling mode, however it does modify the way KLD is computed.
    """
    def __init__(self, 
            layers, mean_normalizer, logvar_normalizer,
            num_latent, num_continuous=0,
            output_nonlinear=False, share_logvar=False,
            persample_kld=False, 
            stochastic=True, concrete=False,
            gumbel=False, gumbel_init=1, gumbel_decay=1, gumbel_step=999999999,
            bypass_mode='BN', mult=1, mult_learn=False, mult_mode="multiply",
            use_gaus=True):
        """
        bypass_mode:
            'BNM': use output of mean-bn-mult, bypass nothing
            'BN':  use output of mean-bn,      bypass mult
            'ST':  use output of mean,         bypass bn and mult
        """
        super(WTACoder, self).__init__(*layers)
        """ modules """
        self.layers          = layers
        self.mean_normalizer = mean_normalizer
        self.logvar_normalizer = logvar_normalizer
        """ sizes """
        self.num_latent      = num_latent
        self.num_continuous  = num_continuous
        self.num_wta         = num_latent - num_continuous
        assert num_latent >= num_continuous, ('How could number of continuous components be greater than total number of '
                                              'latent componentes??')
        """ upward passing output """
        self.output_nonlinear = output_nonlinear
        """ sharing logvar between categorical and gaussian """
        self.share_logvar     = share_logvar
        if share_logvar: # when logvar is shared, it cannot be an independent multiplier
            assert mult==1 and mult_learn==False, ('when logvar is shared between categorical and gaussian, '
                                                   'multiplier must not be learnable and must be 1')
        """ KLD mode """
        self.persample_kld = persample_kld      # per-sample KLD provides high-variance KLD gradient estimate
        """ 
        sampling mode 
        - non-stochastic:   fake loss for debugging
        - stochastic:       one-hot
        - concrete:         relaxed one-hot with concrete loss
        - gumbel:           relaxed one-hot with discrete loss
        """
        self.stochastic    = stochastic         # stochastic sampling, can be disabled for debugging
        self.concrete      = concrete
        self.gumbel        = gumbel             # gumbel_softmax sampling reduces variance for both KLD and logP
        self.gumbel_init   = gumbel_init        # initial temperature
        self.gumbel_decay  = gumbel_decay       # gumbel softmax annealing
        self.gumbel_step   = gumbel_step        # gunbel softmax annealing
        # sanity checks for sampling modes
        assert sum([stochastic, concrete, gumbel]) <= 1, 'At most one of stochastic, gumbel can be True. '
        if gumbel:
            self.steps_taken = 0
        else:
            assert gumbel_decay == 1, 'When gumbel-softmax is not used, gumbel_decay must be 1'
        """ bypassing and multiplier """
        assert bypass_mode in ['BNM', 'BN', 'ST']
        self.bypass_mode = bypass_mode
        self.mult = mult
        self.mult_learn = mult_learn
        if mult_learn:
            assert mult_mode in ['multiply', 'divide', 'multiply_exp', 'divide_exp']
            self.mult_mode = mult_mode
            if mult_mode == 'multiply':
                self.bn_mult = MultScalar  (             self.mult,  learnable=True, apply_exp=False)
            elif mult_mode == 'multiply_exp':
                self.bn_mult = MultScalar  (math.log(    self.mult), learnable=True, apply_exp=True)
            elif mult_mode == 'divide':
                self.bn_mult = DivideScalar(         1.0/self.mult,  learnable=True, apply_exp=False)
            elif mult_mode == 'divide_exp':
                self.bn_mult = DivideScalar(math.log(1.0/self.mult), learnable=True, apply_exp=True)
            else:
                assert False, 'WTF? mult_mode={}'.format(mult_mode)
        else:
            assert mult_mode=='multiply', 'When multiplier is not learnable, mult_mode must be "multiply"'
        """ whether to include gaussian KLD """
        self.use_gaus = use_gaus

    def forward(self, x):
        """ forward propagate and split mean from logvar """
        inp = x
        for layer in self.layers:
            inp = layer(inp)
        mean, logvar = split_mean_logvar(inp, self.num_latent)
        if self.logvar_normalizer is not None:
            logvar = self.logvar_normalizer(logvar)
        var = logvar.exp() # D5 * 0.1
        """ extract categorical components """
        ST  = mean[:, :self.num_wta]                    # ST
        BN  = self.mean_normalizer(ST.contiguous())     # BN
        if self.share_logvar:                           # BNM
            BNM = BN / var
        elif self.mult_learn:
            BNM = self.bn_mult(BN)
        else:
            BNM = self.mult * BN
        P_cat = F.softmax(BNM)
        """ choose which categorical mean to output as mean"""
        if self.bypass_mode != 'ST':
            mean = mean.clone()
            mean[:, :self.num_wta] = BN if self.bypass_mode == 'BN' else BNM 
        """ upward output """
        if self.output_nonlinear:
            upward_output = mean.clone()
            upward_output[:, :self.num_wta] = mean[:, :self.num_wta] * P_cat
        else:
            upward_output = mean
        """ increment gumbel steps, this is likely the most sensible place to record number of iterations """
        if self.training and self.gumbel:
            self.steps_taken += 1
        return (upward_output, mean, logvar, var, P_cat)

    def KLD(self, z, Q, P=None):
        Q_gaus, Q_cat = Q[1:-1], Q[-1]
        """ 
        handle persample_kld
        - categorical KLD normalized with cat_mask instead of Q_cat: sample_mask          * log(q/p) 
        - gaussian    KLD normalized with cat_mask instead of Q_cat: expanded_sample_mask * kld(q||p)
        """
        sample_mask   = self.cat_mask if self.persample_kld else Q_cat

        """ fill up absent P for top-layer prior """
        if P is None:
            P_cat = Variable(new_as(Q_cat.data).fill_(1.0/self.num_wta))
            """ below is needed for sampling """
            # compute current mean and var
            q_mean, q_logvar, q_var = Q_gaus
            wta_mean = q_mean[:, :self.num_wta]
            current_vals = (wta_mean * sample_mask).sum(1)
            current_adaptive_mean = current_vals.mean().data[0]
            current_adaptive_var  = current_vals.var().data[0]
            # update adaptive mean and var
            if not hasattr(self, 'adaptive_mean'):
                self.adaptive_mean = 0
                self.adaptive_var  = 1
            self.adaptive_mean = 0.95 * self.adaptive_mean + 0.05 * current_adaptive_mean
            self.adaptive_var  = 0.95 * self.adaptive_var  + 0.05 * current_adaptive_var
            if self.use_gaus:
                # fill up P_gaus and P_cat
                p_mean = Variable(new_as(q_mean.data))
                p_mean.data[:, :self.num_wta] = self.adaptive_mean # WTA
                p_logvar = Variable(new_as(q_mean.data).fill_(math.log(self.adaptive_var+1e-10)))      # prior unit variance
                p_var    = Variable(new_as(q_mean.data).fill_(         self.adaptive_var ))      # prior unit variance
                # continuous components
                if self.num_continuous > 0:
                    p_mean  .data[:, self.num_wta:] = 0  # 0 mean
                    p_logvar.data[:, self.num_wta:] = 0  # unit variance
                    p_var   .data[:, self.num_wta:] = 1  # unit variance
                P_gaus   = (p_mean, p_logvar, p_var)
        else:
            P_gaus, P_cat = P[1:-1], P[-1]
        """ compute loss """
        if self.concrete:
            loss_cat = kld_for_concretes(Q_cat, P_cat) # concrete KLD does not support persample_kld
        else:
            loss_cat = kld_for_categoricals(Q_cat, P_cat, sample_mask=sample_mask)
        if self.use_gaus:
            loss_gaus = kld_for_gaussians(Q_gaus, P_gaus, do_sum=False)
            full_mask = self.expand_mask(sample_mask) if not self.concrete else 1
            loss_gaus = (loss_gaus * full_mask).sum()
        else:
            loss_gaus = 0
        self.loss_gaus = loss_gaus
        self.loss_cat  = loss_cat
        return loss_gaus + loss_cat

    def print_losses(self):
        print(self.loss_cat, self.loss_gaus)

    def NLL(self, x, P):
        raise NotImplementedError, '{} does not support decoding x'.format(self.__class__)

    def sample(self, P):
        upward_output, mean, logvar, var, P_cat = P
        """ sample a categorical mask """
        if self.stochastic:
            cat_mask = Variable(gumbel_max      (P_cat.data.log()))         # gumbel_max works with logp, more flexible than multinomial_max
        elif self.gumbel or self.concrete:
            T = 0 if not self.training else self.gumbel_init * self.gumbel_decay ** int(self.steps_taken / self.gumbel_step)
            cat_mask = Variable(gumbel_softmax  (P_cat.data.log(), T=T))    # gumbel_softmax requires logp
        else:
            cat_mask = Variable(plain_max       (P_cat.data))               # plain_max works with both logp and p
        self.cat_mask = cat_mask
        """ take gaussian sample """
        if self.use_gaus: 
            std    = var.sqrt()
            noise  = Variable(std.data.new().resize_as_(std.data).normal_())
            smooth = mean + (std * noise)
        else:
            smooth = mean
        """ expand categorical mask from NxC to full mask Nx(C+S) """
        full_mask = self.expand_mask(cat_mask)
        """ 
        STS: Smooth times stochastic 
        full_mask[:, :self.num_wta] is (almost) one-hot (when gumbel_softmax isn't used)
        full_mask[:, self.num_wta:] == 1, dummy values so that we can perform a single multiplication
        Alternative:
            sts_cat     = smooth[:, :self.num_wta] * cat_mask
            smooth_cont = smooth[:, self.num_wta:]
            return torch.cat([sts_cat, smooth_cont], 1)
        """
        return smooth * full_mask.float()

    def expand_mask(self, cat_mask):
        # expand the categorical mask to cover the continuous components with 1 to simplify computation
        full_size = list(cat_mask.size())
        full_size[1] += self.num_continuous
        if self.num_wta < self.num_latent: # has continuous components
            full_mask = Variable(cat_mask.data.new().resize_(*full_size))
            full_mask[:, :self.num_wta] = cat_mask  # gradient not needed
            full_mask[:, self.num_wta:] = 1         # simply copy continous components
        else:
            full_mask = cat_mask
        return full_mask

    def __repr__(self):
        extra_str = "(num_wta={}, num_continuous={}, output_nonlinear={}, persample_kld={}, stochastic={}, gumbel={}, gumbel_decay={}, gumbel_step={}, "
        extra_str += "bypass_mode={}, mult={}, mult_learn={}, use_gaus={})"
        extra_str = extra_str.format(
                self.num_wta, self.num_continuous, self.output_nonlinear, self.persample_kld, self.stochastic, self.gumbel, self.gumbel_decay, self.gumbel_step,
                self.bypass_mode, self.mult, self.mult_learn, self.use_gaus
                )
        return super(WTACoder, self).__repr__(extra_str)

    def extract_variance_stats(self):
        """ for debugging variance statistics """
        multiplier = self.bn_mult.weight.data[0] if self.mult_learn else self.mult
        final_layer = self.layers[-1]
        # extract only the logvar of wta components
        nw = self.num_wta
        nl = self.num_latent
        logvar_weight  = final_layer.weight[nl:nl+nw]
        logvar_bias    = final_layer.bias[nl:nl+nw] if hasattr(final_layer, 'bias') and final_layer.bias is not None else 0
        logvar_wvar  = (logvar_weight/nw).var().data[0]
        logvar_bmean = logvar_bias.mean().data[0] if logvar_bias != 0 else 0 
        return multiplier, logvar_wvar, logvar_bmean

