import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base import *
from .model import *
from .math_ops import *
from .functions import FuncSTCategorical, FuncOneHotSTCategorical
import os # for debugging with os.getenv

"""
prototype:
class BaseVAE(nn.Module):
    def forward(self, x):
        x: input
        returns: Q, z
            Q: latent distribution
            z: sample from Q

    def get_losses(self):
        returns: (loss_overall, loss_z, loss_x [, loss_zs])
            loss_overall: loss_z + loss_x
            loss_z: KL divergence in the latent layers KLD(Q||P). If there are multiple layers of latent code, this is a
                sum of loss_zs
            loss_x: negative log probability of P(x|z)
            loss_zs: in the case there are multiple layers of latent code, loss_zs is a list of KLDs from different
                latent layers

    def sample_p(self, z):
        returns: a sample obtained by passing z down the chain of generation, and taking samples where appropriate

    def clear_state(self):
        returns: None
        Our VAEs save some states for computing losses. This is why get_losses does not require other parameters. This
        method can be used to clear the saved states before serialization.
"""

class VAE(nn.Module):

    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        Q = self.encoder(x)
        z = self.encoder.sample(Q)
        P = self.decoder(z)
        self.x = x
        self.Q = Q
        self.P = P
        return Q, z

    def get_losses(self):
        loss_z = self.encoder.get_loss_z(self.Q, None)
        loss_x = self.decoder.get_loss_x(self.P, self.x)
        loss = loss_z + loss_x
        return loss, loss_z, loss_x

    def sample_p(self, z, sample_x=False):
        P = self.decoder(z)
        if sample_x:
            return self.decoder.sample(P)
        else:
            if type(P) in [tuple, list]:
                return P[0]
            else:
                return P

    def clear_states(self):
        self.x = None
        self.Q = None
        self.P = None


class hVAE(nn.Sequential): # inherits nn.Sequential for its nice printing (__repr__ method)

    def __init__(self, stages):
        nn.Sequential.__init__(self, *stages)
        self.stages = stages

    def forward(self, x):
        self.x = x
        inp = x
        for stage in self.stages:
            Q = stage.encoder(inp)
            z = stage.encoder.sample(Q)
            P = stage.decoder(z)
            stage.Q = Q
            stage.P = P
            inp = Q[0] if type(Q) in [tuple, list] else Q # pass the first element in Q (usually mean) to higher stage
        return Q, z

    def get_losses(self):
        loss_zs = []
        loss_z = 0
        for i, lower in enumerate(self.stages):
            higher = self.stages[i+1] if i+1 < len(self.stages) else None
            layer_lossz = lower.get_loss_z(lower.Q, higher.P if higher else None)
            loss_zs.append(layer_lossz)
            loss_z += layer_lossz
        lowest = self.stages[0]
        loss_x = lowest.get_loss_x(lowest.P, self.x)
        loss_sum = loss_z + loss_x
        return loss_sum, loss_z, loss_x, loss_zs

    def sample_p(self, z, sample_x=False):
        for i, stage in enumerate(reversed(self.stages)):
            P = stage.decoder(z)
            z = stage.decoder.sample(P)
        if sample_x:
            return z
        else:
            # return P, but P may be a tuple, so we take the mean
            if type(P) in [tuple, list]:
                return P[0]
            else:
                return P

    def clear_states(self):
        self.x = None
        for stage in self.stages:
            stage.Q = None
            stage.P = None


class hVAEStage(nn.Module):

    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder

    def get_loss_z(self, Q, P):
        return self.encoder.get_loss_z(Q,P)

    def get_loss_x(self, P, x): # only callable on bottom stage directly connected with x
        return self.decoder.get_loss_x(P,x)


def split_gaussian(mean_logvar, num_latent):
    assert mean_logvar.size(1) == num_latent * 2
    mean   = mean_logvar[:, :num_latent]
    logvar = mean_logvar[:, num_latent:]
    return mean, logvar


class GaussianCoder(nn.Sequential):

    def __init__(self, num_latent, layers):
        nn.Sequential.__init__(self, *layers)
        self.num_latent = num_latent

    def forward(self, x):
        mean, logvar = split_gaussian(nn.Sequential.forward(self, x), self.num_latent)
        return (mean, logvar, logvar.exp())

    def get_loss_z(self, Q, P=None):
        if P is not None:
            loss_z = kld_for_gaussians(Q, P)
        else: # P will be None for top-layer encoder
            loss_z = kld_for_unit_gaussian(*Q) 
        return loss_z / Q[0].size(0)

    def get_loss_x(self, P, x):
        loss_x = nl_for_gaussian(x, P)
        return loss_x / x.size(0)

    def sample(self, P):
        mean, logvar, var = P
        std = var.sqrt()
        noise = Variable(std.data.new().resize_as_(std.data).normal_())
        return mean + noise * std


class WTACoder(PSequential):
    """
    At this point WTACoder has grown to such a monstrous size we must consider refactoring it. We can create several
    subclasses of it depending on the sampling mode, though that would break the external interface. An alternative
    method, which preserves the external interface, is to use internal delegation. For every sampling mode, we delegate
    forward_hook, sample, and get_loss_z

    These are the sampling modes we currently have:
    - deterministic:    fake loss for debugging
    - stochastic:       one-hot stochastic
    - gumbel:           relaxed one-hot, using discrete KLD
    - concrete:         relaxed one-hot, using concrete KLD

    persample_kld is not itself a sampling mode, however it does modify the way loss_z is computed.
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
        """ loss_z mode """
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
        mean, logvar = split_gaussian(inp, self.num_latent)
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

    def get_loss_z(self, Q, P=None):
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
            full_mask = self.expand_mask(sample_mask)
            loss_gaus = (loss_gaus * full_mask).sum()
        else:
            loss_gaus = 0
        loss_gaus /= Q_gaus[0].size(0)
        loss_cat  /= Q_gaus[0].size(0)
        self.loss_gaus = loss_gaus
        self.loss_cat  = loss_cat
        return loss_gaus + loss_cat

    def print_losses(self):
        print(self.loss_cat, self.loss_gaus)

    def get_loss_x(self, P, x):
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
            smooth = mean + (noise * std)
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
            assert False # FIXME debugging
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
        logvar_bias    = final_layer.bias[nl:nl+nw] if hasattr(final_layer, 'bias') else 0
        logvar_wvar  = (logvar_weight/nw).var().data[0]
        logvar_bmean = logvar_bias.mean().data[0] if logvar_bias != 0 else 0 
        return multiplier, logvar_wvar, logvar_bmean


class FWTACoder(PSequential):

    def __init__(self, num_latent, layers, 
            mean_normalizer=None, stochastic=True, num_continuous=0, bypass_mode='BNM', mult=1, use_gaus=True):
        """
        bypass_mode:
            'BNM': use output of mean-bn-mult, bypass nothing
            'BN':  use output of mean-bn,      bypass mult
            'ST':  use output of mean,         bypass bn and mult
        """
        super(WTACoder, self).__init__(*layers)
        self.layers = layers
        self.num_latent     = num_latent
        self.num_continuous = num_continuous
        self.num_wta        = num_latent - num_continuous
        self.bypass_mode = bypass_mode
        assert bypass_mode in ['BNM', 'ST'] # doesn't support BN yet
        assert num_latent >= num_continuous
        self.stochastic = stochastic
        self.mean_normalizer= mean_normalizer
        self.mult = 1
        self.use_gaus = use_gaus
        # FIXME debugging
        assert self.num_latent==self.num_wta==256
        assert self.num_continuous == 0

    def forward(self, x):
        inp = x
        for layer in self.layers:
            inp = layer(inp)
        mean, logvar = split_gaussian(inp, self.num_latent)
        cat_input = mean[:, :self.num_wta]
        if self.mean_normalizer is not None:
            logit = self.mean_normalizer(cat_input.contiguous())
            # FIXME remove this assert
            assert self.bypass_mode=='BNM'
            if self.bypass_mode=='BNM':
                mean = mean.clone()
                mean[:, :self.num_wta] = logit
        P_cat = F.softmax(logit) # F.softmax covers both 2D and 4D
        return (mean, logvar, logvar.exp(), P_cat)

    def get_loss_z(self, Q, P=None):
        Q_gaus, Q_cat = Q[:-1], Q[-1]
        if P is None:
            loss_cat  = kld_for_uniform_categorical(Q_cat)
            loss_gaus = kld_for_unit_gaussian(*Q_gaus, do_sum=False)
        else:
            P_gaus, P_cat = P[:-1], P[-1]
            loss_cat  = kld_for_categoricals(Q_cat, P_cat)
            loss_gaus = kld_for_gaussians(Q_gaus, P_gaus, do_sum=False)
        full_mask = self.expand_mask(Q_cat)
        loss_gaus = (loss_gaus * full_mask).sum() if self.use_gaus else 0
        return (loss_gaus + loss_cat) / Q_gaus[0].size(0) 

    def get_loss_x(self, P, x):
        raise NotImplementedError, '{} does not support decoding x'.format(self.__class__)

    def sample(self, P):
        mean, logvar, var, P_cat = P
        if self.stochastic:
            cat_mask = Variable(multinomial_max(P_cat.data)) # gumbel_max doesn't work with P
        else:
            cat_mask = Variable(plain_max(P_cat.data))
        full_mask = self.expand_mask(cat_mask)
        if self.use_gaus:
            std = var.sqrt()
            noise = Variable(std.data.new().resize_as_(std.data).normal_())
            gaus = mean + (noise * std)
        else:
            gaus = mean
        return gaus  * full_mask.float()

    def expand_mask(self, wta_mask):
        # expand the categorical mask to cover the continuous components with 1 to simplify computation
        full_size = list(wta_mask.size())
        assert self.num_continuous==0 # FIXME debugging
        full_size[1] += self.num_continuous
        if self.num_wta < self.num_latent: # has continuous components
            assert False # FIXME debugging
            full_mask = Variable(wta_mask.data.new().resize_(*full_size))
            full_mask[:, :self.num_wta] = wta_mask # would the gradient be regisered???
            full_mask[:, self.num_wta:] = 1 # do not mask over continuous components
        else:
            full_mask = wta_mask
        return full_mask

    def __repr__(self):
        return super(WTACoder, self).__repr__('(stochastic={}, num_wta={}, num_continuous={})'.format(
                                            self.stochastic, self.num_wta, self.num_continuous))


class BernoulliCoder(nn.Sequential): # nobody use Bernoulli for encoder!

    def __init__(self, layers):
        nn.Sequential.__init__(self, *layers)

    def forward(self, x):
        P = nn.Sequential.forward(self, x)
        return P

    def get_loss_z(self, Q, P):
        raise NotImplementedError, '{} does not support encoding z'.format(self.__class__)

    def get_loss_x(self, P, x):
        loss_x = F.binary_cross_entropy(P, x, size_average=False)
        return loss_x / x.size(0)

    def sample(self, P):
        assert type(P)==Variable
        return Variable(P.data.new().resize_as_(P.data).uniform_()) < P


class MSEDecoder(nn.Sequential):

    def __init__(self, layers):
        super(MSEDecoder, self).__init__(*layers)

    def get_loss_z(self, Q, P):
        raise NotImplementedError, '{} does not support encoding z'.format(self.__class__)

    def get_loss_x(self, P, x):
        loss_x = (P - x).pow(2).mean() # MSE
        return loss_x # loss_x / x.size(0)

    def sample(self, P):
        return P


class FakeCoder(PSequential):
    """ For debugging WTACoder """

    def __init__(self, layers, 
            stochastic=True, KLD=True, forward_sample=False, use_variance=False, beta=0.9, debug=False):
        super(FakeCoder, self).__init__(*layers)
        self.stochastic=stochastic
        self.KLD = KLD
        self.forward_sample = forward_sample
        self.use_variance = use_variance
        #assert not (use_variance and not stochastic), 'variance can only be estimated in stochastic mode'
        self.beta = beta
        self.debug = debug
        if debug:
            self.layers = layers
            self.num_latent = 256
            self.num_wta    = 256
            self.num_continuous = 0
            self.mean_normalizer = nn.Sequential(nn.BatchNorm1d(256, affine=False))
            assert self.num_latent==self.num_wta==256

    def forward(self, x):
        if not self.debug:
            linear = nn.Sequential.forward(self, x)
            resp = None
            if self.forward_sample:
                return self.sample((linear, resp))
            return linear, resp
        else:
            num_latent=256
            inp = x
            for layer in self.layers:
                inp = layer(inp)
            mean, logvar = split_gaussian(inp, num_latent)
            cat_input = mean[:, :self.num_wta]
            if self.mean_normalizer is not None:
                logit = self.mean_normalizer(cat_input.contiguous())
                mean = mean.clone()
                mean[:, :self.num_wta] = logit
            P_cat = F.softmax(logit)
            return mean, P_cat

    def get_loss_z(self, Q, P=None):
        if not self.debug:
            if self.KLD:
                assert P==None
                linear, resp = Q
                Q_cat = resp if resp is not None else F.softmax(linear)
                loss_cat  = kld_for_uniform_categorical(Q_cat)
                return loss_cat / linear.size(0) 
            else:
                return Variable(Q[0].data.new().resize_(1).fill_(0)) # no loss, that's why it's fake
        else:
            Q_mean, Q_cat = Q
            loss_cat = kld_for_uniform_categorical(Q_cat)
            return loss_cat / Q_mean.size(0)

    def get_loss_x(self, P, x):
        raise NotImplementedError

    def sample(self, P):
        linear, resp = P
        data = resp.data if resp is not None else linear.data
        if self.stochastic:
            mask = Variable(multinomial_max(data))
        else:
            mask = Variable(plain_max(data))
        """
        if self.use_variance: # self.use_variance
            # if variance buffer doesn't exist, create it first
            if not hasattr(self, 'buffer_mean'):
                p_cat = F.softmax(linear)
                # assume 0 mean, unit variance at initialization
                buffer_mean = linear.data.mean(0).fill_(0) # FIXME doesn't work for 4D tensor
                buffer_var  = buffer_mean.clone().fill_(1)
                m1 = buffer_mean.clone()
                v1 = buffer_var.clone()
                p1 = buffer_var.clone().fill_(1)
                self.register_buffer('buffer_mean', buffer_mean)
                self.register_buffer('buffer_var',  buffer_var)
                self.register_buffer('m1', m1)
                self.register_buffer('v1', v1)
                self.register_buffer('p1', p1)
                assert self.buffer_mean is not None
            # maintain a running estimation of mean and variance, normalized by responsibility
            else:
                # use p_cat to compute mean and var
                diff  = linear - Variable(self.buffer_mean.expand_as(linear))
                diff2 = diff.pow(2)
                exponent = - diff2 / (Variable(self.buffer_var.expand_as(linear))*2)
                resp = F.softmax(exponent)
                mask = Variable(multinomial_max(resp.data))
                self.m1 *= self.beta
                self.p1 *= self.beta
                self.v1 *= self.beta
                self.m1 += (1-self.beta)*(mask * linear).mean(0).data
                self.v1 += (1-self.beta)*(mask * diff2).mean(0).data
                self.p1 += (1-self.beta)*(mask).mean(0).data
                self.buffer_mean = self.m1 / self.p1
                self.buffer_var  = self.v1 / self.p1
        """
        if self.debug:
            full_mask = self.expand_mask(mask)
            return linear * full_mask
        return linear * mask.float()

    # FIXME debugging
    def expand_mask(self, wta_mask):
        # expand the categorical mask to cover the continuous components with 1 to simplify computation
        full_size = list(wta_mask.size())
        assert self.num_continuous==0 # FIXME debugging
        full_size[1] += self.num_continuous
        if self.num_wta < self.num_latent: # has continuous components
            assert False # FIXME debugging
            full_mask = Variable(wta_mask.data.new().resize_(*full_size))
            full_mask[:, :self.num_wta] = wta_mask # would the gradient be regisered???
            full_mask[:, self.num_wta:] = 1 # do not mask over continuous components
        else:
            full_mask = wta_mask
        return full_mask

    def __repr__(self):
        return super(FakeCoder, self).__repr__(
            '(stochastic={}, KLD={}, forward_sample={}, use_variance={}, beta={})'.format(
                self.stochastic, self.KLD, self.forward_sample, self.use_variance, self.beta)
        )


class STCategory(torch.nn.Module):
    """ straight-through categorical """

    def __init__(self, stochastic=True, forget_mask=False):
        super(STCategory, self).__init__()
        self.stochastic  = stochastic
        self.forget_mask = forget_mask

    def forward(self, x):
        """
        if self.stochastic:
            mask = Variable(gumbel_max(x.data)) # gumbel_max must use logit
        else:
            mask = Variable(plain_max(x.data)) # plain_max accepts either logit or probability
        return x * mask.float()
        """

        if self.stochastic:
            mask = Variable(gumbel_max(x.data))
        else:
            mask = Variable(plain_max(x.data))
        return FuncSTCategorical(self.stochastic, self.forget_mask)(x, mask)

    def __repr__(self):
        return "{}(stochastic={}, forget_mask={})".format(self.__class__.__name__, self.stochastic, self.forget_mask)


class OneHotSTCategory(torch.nn.Module):

    def __init__(self, stochastic=False, forget_mask=False):
        super(OneHotSTCategory, self).__init__()
        self.stochastic  = stochastic
        self.forget_mask = forget_mask

    def forward(self, x):
        return FuncOneHotSTCategorical(self.stochastic, self.forget_mask)(x)


