import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .th_model import *
from .th_math import *
from .th_functions import FuncSTCategorical, FuncOneHotSTCategorical
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
            loss_zs.append(lower.get_loss_z(lower.Q, higher.P if higher else None))
            loss_z += loss_zs[-1]
        lowest = self.stages[0]
        loss_x = lowest.get_loss_x(lowest.P, self.x)
        return loss_z + loss_x, loss_z, loss_x, loss_zs

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
        self.mult = mult
        self.use_gaus = use_gaus
        # FIXME debugging
        assert self.num_latent==self.num_wta==256
        assert self.num_continuous == 0
        if int(os.getenv('learn_mult')):
            self.logit_mult = MultScalar(self.mult, learnable=True) # D7

    def forward(self, x):
        inp = x
        for layer in self.layers:
            inp = layer(inp)
        mean, logvar = split_gaussian(inp, self.num_latent)
        var = logvar.exp() # D5 * 0.1
        cat_input = mean[:, :self.num_wta]
        logit = self.mean_normalizer(cat_input.contiguous())
        mean = mean.clone()
        mean[:, :self.num_wta] = logit
        if int(os.getenv('learn_mult')):
            P_cat = F.softmax(self.logit_mult(logit))
        else:
            P_cat = F.softmax((logit*self.mult)) # F.softmax covers both 2D and 4D # D3, D7
        # P_cat = F.softmax(logit / var ) # divide by variance # D4
        """ adaptive mean """
        #self.cat_mean = mean.mean(0)
        return (mean, logvar, var, P_cat)

    def get_loss_z(self, Q, P=None):
        Q_gaus, Q_cat = Q[:-1], Q[-1]
        assert P is None
        loss_cat  = kld_for_uniform_categorical(Q_cat)
        loss_gaus = kld_for_unit_gaussian(*Q_gaus, do_sum=False)            #D1
        full_mask = self.expand_mask(Q_cat)                                 #D1
        loss_gaus = (loss_gaus * full_mask).sum() if self.use_gaus else 0   #D1
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
        #if self.use_gaus: # FIXME, D2
        if int(os.getenv('use_gaussian_sample')):
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
            assert False
            """
            P_gaus, P_cat = P[:-1], P[-1]
            loss_cat  = kld_for_categoricals(Q_cat, P_cat)
            loss_gaus = kld_for_gaussians(Q_gaus, P_gaus, do_sum=False)
            """
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

    def __init__(self, layers, stochastic=True, KLD=True, forward_sample=False, use_variance=False, beta=0.9, debug=False):
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


