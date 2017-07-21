import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
from .model import Model, extract_float

"""
Issues to double-check:
[x] loss sizes (N,K) to (N,1) to (1)
[x] prior_P inheritance and invokation
[x] sample_prior and sample_input inheritance
[x] upper_prior in both wake and sleep
[x] add_noise in sample_input
[x] invert
[x] get_losses
[x] extract_float on report_losses
[ ] KLD, NLL sizes
"""

def extract_mean(P):
    """
    P is the representation of a distribution output by Distribution.forward
    sometimes it is just a Variable(Tensor). Sometimes, it is a tuple of Variable, the first in which is the mean
    """
    return P[0] if type(P) in [tuple, list] else P


VAEState = namedtuple('VAEState', ['input', 'upper', 'code', 'lower'])
# wake:                             x,       Q(z|x),  z,      P(x|z)
# sleep:                            z,       P(x|z),  x,      Q(z|x)


class SimpleNetwork(nn.Module):

    def __init__(self, encoder, decoder):
        super(SimpleNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        upper = self.encoder(input)
        code  = self.encoder.sample(upper)
        lower = self.decoder(code)
        state = VAEState(input, upper, code, lower)
        return state

    def get_losses(self, state, loss_transform=torch.mean, take_top_prior=True):
        upper_prior = self.encoder.prior_P(template=extract_mean(state.upper)) if take_top_prior else None
        KLD = self.KLD(state, upper_prior)
        NLL = self.NLL(state)
        loss = KLD + NLL
        loss = loss_transform(loss) if loss_transform else loss
        return loss, KLD.mean(), NLL.mean()
    
    def KLD(self, state, upper_prior):
        if upper_prior is not None:
            return self.encoder.KLD(state.code, state.upper, upper_prior).sum(1)
        else: # in sleep-phase, upper_prior at top-level is unavailable, so we can only take the logP part of a KLD
            return self.encoder.logP(state.code, state.upper).sum(1)

    def NLL(self, state):
        return self.decoder.NLL(state.input, state.lower).sum(1)

    def sample_prior(self, template=None, state=None):
        if template is None: template = extract_mean(state.upper)
        return self.encoder.sample_prior(template)

    def sample_input(self, code, add_noise=False):
        lower = self.decoder(code)
        return extract_mean(lower) if not add_noise else self.decoder.sample(lower)

    def invert(self):
        return SimpleNetwork(self.decoder, self.encoder)


class HierarchicalNetwork(nn.Module):

    def __init__(self, stages, pass_sample=False):
        super(HierarchicalNetwork, self).__init__()
        self.stages = stages # stages: [SimpleNetwork]
        for i, stage in enumerate(stages):
            self.add_module(str(i), stage)
        self.pass_sample = pass_sample

    def forward(self, input):
        states = []
        for stage in self.stages:
            state = stage(input)
            states.append(state)
            input = state.code if self.pass_sample else extract_mean(state.upper)
        return states
    
    def get_losses(self, states, loss_transform=torch.mean, take_top_prior=True):
        KLDs = []
        KLD = 0
        # compute KLD at every stage
        for i, (stage, state) in enumerate(zip(self.stages, states)):
            if (i+1) < len(states):
                upper_prior = states[i+1].lower
            elif take_top_prior:
                upper_prior = self.stages[-1].encoder.prior_P(template=extract_mean(state.upper))
            else:
                upper_prior = None
            KLDi = stage.KLD(state, upper_prior)
            KLDs.append(KLDi.mean())
            KLD += KLDi
        # compute NLL at the lowest stage
        lowest_stage = self.stages[0]
        lowest_state =      states[0]
        NLL = lowest_stage.NLL(lowest_state)
        loss = KLD + NLL
        loss = loss_transform(loss) if loss_transform else loss
        return loss, KLD.mean(), NLL.mean(), KLDs

    def sample_prior(self, template=None, state=None):
        if template is None: template = extract_mean(state[-1].upper)
        return self.stages[-1].encoder.sample_prior(template)

    def sample_input(self, code, add_noise=False):
        for stage in reversed(self.stages):
            lower = stage.decoder(code)
            code  = stage.decoder.sample(lower)
        return extract_mean(lower) if not add_noise else code 

    def invert(self):
        return HierarchicalNetwork([stage.invert() for stage in reversed(self.stages)])


def mean_transformer(vae, state):
    return torch.mean


def identity_transformer(vae, state):
    return lambda x:x


mt = mean_transformer


class AsymmetricVAE(Model):

    def __init__(self, network, loss_transformer=mt):
        # loss_transformer allows importance-weighted updates and critic-based updates
        # it defaults to mean_transformer, which simply computes torch.mean
        super(AsymmetricVAE, self).__init__()
        self.network = network
        self.loss_transformer = loss_transformer

    def forward(self, input): # have to implement this, otherwise nn.Module complains
        state = self.network(input)
        return state

    def get_losses(self, state, label, loss_transform=None):
        if loss_transform is None: loss_transform = self.loss_transformer(self, state)
        return self.network.get_losses(state, loss_transform, True)

    def report_losses(self, losses, include_all=False):
        if not include_all:
            return extract_float(losses[:3]) # only loss, KLD, NLL
        else:
            return extract_float(losses)

    def __getattr__(self, name):
        try:
            return super(AsymmetricVAE, self).__getattr__(name)
        except:
            assert name in ['sample_prior', 'sample_input'], '{} is not a member of {}'.format(name, self.__class__.__name__)
            return getattr(self.network, name)


class SymmetricVAE(Model):
    
    def __init__(self, network, wake_transformer=mt, sleep_transformer=mt, sleep=True):
        super(SymmetricVAE, self).__init__()
        self.wake_network  = network
        self.sleep_network = network.invert() # invert encoder-decoder and order of stages
        self.wake_transformer  = wake_transformer
        self.sleep_transformer = sleep_transformer
        self.sleep = sleep

    def forward(self, x, z=None):
        state_wake  = self.wake_network(x)
        if not self.sleep:
            return state_wake, None
        if z is None: 
            z = self.wake_network.sample_prior(state=state_wake)
        state_sleep = self.sleep_network(z)
        return state_wake, state_sleep

    def get_losses(self, state, label, wake_transform=None, sleep_transform=None):
        if not self.sleep:
            state_wake, _ = state
            if wake_transform  is None: wake_transform  = self.wake_transformer (self, state_wake)
            losses_wake  = self.wake_network. get_losses(state_wake,  wake_transform,  True)
            return losses_wake
        state_wake, state_sleep = state
        if wake_transform  is None: wake_transform  = self.wake_transformer (self, state_wake)
        if sleep_transform is None: sleep_transform = self.sleep_transformer(self, state_sleep)
        losses_wake  = self.wake_network. get_losses(state_wake,  wake_transform,  True)
        losses_sleep = self.sleep_network.get_losses(state_sleep, sleep_transform, False)
        loss = losses_wake[0] + losses_sleep[0] * 0.5
        return loss, losses_wake, losses_sleep

    def report_losses(self, losses, include_all=False):
        if self.sleep:
            loss, losses_wake, losses_sleep = losses
            if not include_all:
                return extract_float([loss, losses_wake[0], losses_sleep[0]])
            else:
                return extract_float([loss, losses_wake, losses_sleep])
        else:
            losses_wake = losses
            if not include_all:
                return extract_float(losses_wake[:3])
            else:
                return extract_float(losses_wake)

    def __getattr__(self, name):
        try:
            return super(SymmetricVAE, self).__getattr__(name)
        except:
            assert name in ['sample_prior', 'sample_input']
            return getattr(self.wake_network, name)

    def __repr__(self, additional=''):
        # We don't want sleep_network to be printed, so we modify a bit
        result = self.__class__.__name__ + ' ({},'.format(additional)+'\n'
        for key, module in self._modules.items():
            if module is self.sleep_network: 
                modstr = "..."
            else:
                modstr = module.__repr__()
                modstr = nn.modules.module._addindent(modstr, 2)
            result = result + '  (' + key + '): ' + modstr + '\n'
        result = result + ')'
        return result

