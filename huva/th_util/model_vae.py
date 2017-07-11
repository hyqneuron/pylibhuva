import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

"""
Issues to double-check:
[x] loss sizes (N,K) to (N,1) to (1)
[x] prior_P inheritance and invokation
[x] sample_prior and sample_input inheritance
[x] higher_prior in both wake and sleep
[x] add_noise in sample_input
[x] invert
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
        higher_prior = encoder.prior_P(template=state.higher) if take_top_prior else None
        KLD = self.KLD(state. higher_prior)
        NLL = self.NLL(state)
        loss = KLD + NLL
        loss = loss_transform(loss) if loss_transform else loss
        return loss, KLD, NLL
    
    def KLD(self, state, higher_prior):
        if higher_prior is not None:
            return self.encoder.KLD(state.code, state.higher, higher_prior).sum(1)
        else: # in sleep-phase, higher_prior at top-level is unavailable, so we can only take the logP part of a KLD
            return self.encoder.logP(state.code, state.higher).sum(1)

    def NLL(self, state):
        return self.decoder.NLL(state.input, state.lower).sum(1)

    def sample_prior(self, template=None, state=None):
        if template is None: template = state.higher
        return self.encoder.sample_prior(template)

    def sample_input(self, code, add_noise=False):
        lower = self.decoder(code)
        return extract_mean(lower) if not add_noise else self.decoder.sample(lower)

    def invert(self):
        return SimpleNetwork(self.decoder, self.encoder)


class HierarchicalNetwork(nn.Module):

    def __init__(self, stages):
        super(HierarchicalNetwork, self).__init__()
        self.stages = stages # stages: [SimpleNetwork]

    def forward(self, input, pass_sample=False):
        states = []
        for stage in stages:
            state = stage(input)
            states.append(state)
            input = state.code if pass_sample else extract_mean(state.upper)
        return states
    
    def get_losses(self, states, loss_transform=torch.mean, take_top_prior=True):
        KLDs = []
        KLD = 0
        # compute KLD at every stage
        for i, (stage, state) in enumerate(zip(self.stages, states)):
            if (i+1) < len(states):
                higher_prior = states[i+1].lower
            elif take_top_prior:
                higher_prior = self.stages[-1].encoder.prior_P(template=state.higher)
            else:
                higher_prior = None
            KLDi = stage.KLD(state, higher_prior)
            KLDs.append(KLDi)
            KLD += KLDi
        # compute NLL at the lowest stage
        lowest_stage = self.stages[0]
        lowest_state =      states[0]
        NLL = lowest_stage.NLL(lowest_state)
        loss = KLD + NLL
        loss = loss_transform(loss) if loss_transform else loss
        return loss, KLD, NLL, KLDs

    def sample_prior(self, template=None, states=None):
        if template is None: template = states[-1].higher
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

mt = mean_transformer


class AsymmetricVAE(nn.Module):

    def __init__(self, network, loss_transformer=mt):
        # loss_transformer allows importance-weighted updates and critic-based updates
        # it defaults to mean_transformer, which simply computes torch.mean
        super(AsymmetricVAE, self).__init__()
        self.network = network
        self.loss_transformer = loss_transformer

    def get_losses(self, state, loss_transform=None):
        if loss_transform is None: loss_tramsform = self.loss_transformer(self, state)
        return network.get_losses(state, loss_transform, True)

    def __getattr__(self, name):
        assert name in ['sample_prior', 'sample_input', 'forward']
        return getattr(self.network, name)


class SymmetricVAE(nn.Module):
    
    def __init__(self, network, wake_transformer=mt, sleep_transformer=mt):
        super(SymmetricVAE, self).__init__()
        self.wake_network  = network
        self.sleep_network = network.invert() # invert encoder-decoder and order of stages
        self.wake_transformer  = wake_transformer
        self.sleep_transformer = sleep_transformer

    def forward(self, x, z=None, sleep=True):
        state_wake  = self.wake_network(x)
        if not sleep: 
            return state_wake
        if z is None: 
            z = self.wake_network.sample_prior(state=state_wake)
        state_sleep = self.sleep_network(z)
        return state_wake, state_sleep

    def get_losses(self, state, wake_transform=None, sleep_transform=None, sleep=True):
        if sleep:
            state_wake, state_sleep = state
            if wake_transform  is None: wake_transform  = self.wake_transformer (self, state_wake)
            if sleep_transform is None: sleep_transform = self.sleep_transformer(self, state_sleep)
            losses_wake  = self.wake_network. get_losses(state_wake,  wake_transform,  True)
            losses_sleep = self.sleep_network.get_losses(state_sleep, sleep_transform, False)
            loss = losses_wake[0] + losses_sleep[0]
            return loss, losses_wake, losses_sleep
        else:
            state_wake = state
            if wake_transform  is None: wake_transform  = self.wake_transformer (self, state_wake)
            losses_wake  = self.wake_network. get_losses(state_wake,  wake_transform,  True)
            return losses_wake

    def __getattr__(self, name):
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

