import torch
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict
import math
from .th_functions import *


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


class CachedSequential(torch.nn.Sequential):
    """
    Convenience container that caches the output of a specified set of layers for later use.
    """

    def set_cache_targets(self, cache_targets):
        """
        cache_target: a list of layer names whose output we'll cache
        """
        self.cache_targets = cache_targets

    def forward(self, input):
        self.cached_outs = {}
        for name, module in self._modules.iteritems():
            input = module(input)
            if name in self.cache_targets:
                self.cached_outs[name] = input
        return input


class PSequential(torch.nn.Sequential): # Pretty Sequential, allow __repr__ customization

    def __repr__(self, additional=''):
        tmpstr = self.__class__.__name__ + ' ({},'.format(additional)+'\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = torch.nn.modules.module._addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


class Flatten(torch.nn.Module):
    """
    Flatten the output from dim-1 and onwards
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return "Flatten()"


class View(torch.nn.Module):
    """
    View output as a specific size
    """

    def __init__(self, *sizes):
        torch.nn.Module.__init__(self)
        self.sizes = sizes

    def forward(self, x):
        return x.view(x.size(0), *self.sizes)

    def __repr__(self):
        return "View({})".format(str(self.sizes))


class StridingTransform(torch.nn.Module):

    def __init__(self, stride):
        torch.nn.Module.__init__(self)
        self.stride = (stride, stride) if type(int(stride)) in [int, float] else stride
        assert all(map(lambda x:type(x)==int, self.stride))

    def __repr__(self):
        return '{}(stride={})'.format(self.__class__.__name__, self.stride)


class SpaceToChannel(StridingTransform):

    def forward(self, x):
        assert x.dim() == 4 
        assert x.size(2) % self.stride[0] == 0 
        assert x.size(3) % self.stride[1] == 0
        N, C, H, W = x.size()
        Cs = C * self.stride[0] * self.stride[1]
        Hs = H / self.stride[0]
        Ws = W / self.stride[1]
        return x.view(N, C, Hs, self.stride[0], Ws, self.stride[1])\
                .permute(0, 1, 3, 5, 2, 4)\
                .contiguous()\
                .view(N, Cs, Hs, Ws)


class ChannelToSpace(torch.nn.Module):

    def forward(self, x):
        assert x.dim() == 4 
        N, Cs, Hs, Ws = x.size()
        assert Cs % (self.stride[0] * self.stride[1]) == 0
        C = Cs / self.stride[0] / self.stride[1]
        H = Hs * self.stride[0]
        W = Ws * self.stride[1]
        return x.view(N, C, self.stride[0], self.stride[1], Hs, Ws)\
                .permute(0, 1, 4, 2, 5, 3)\
                .contiguous()\
                .view(N, C, H, W)


class MultScalar(torch.nn.Module):

    def __init__(self, scalar=1, learnable=True):
        super(MultScalar, self).__init__()
        self.mult = scalar
        self.learnable = learnable
        if learnable:
            weight = Parameter(torch.Tensor(1).fill_(scalar))
            self.weight = weight

    def forward(self, x):
        if self.learnable:
            return x * self.weight.expand_as(x)
        else:
            return x * self.mult

    def __repr__(self):
        return "{}(scalar={}, learnable={})".format(self.__class__.__name__, self.mult, self.learnable)


class STCategory(torch.nn.Module):
    """ straight-through categorical """

    def __init__(self, stochastic=True):
        super(STCategory, self).__init__()
        self.stochastic = stochastic

    def forward(self, x):
        if self.stochastic:
            #p = F.softmax(x)
            #mask = Variable(multinomial_max(p.data)) # gumbel_max must use logit
            mask = Variable(gumbel_max(x.data)) # gumbel_max must use logit
            """
            Both multinomial_max and gumbel_max seem to have the same behaviour so far. So maybe the multinomial bug in
            torch isn't so bad. multinomial_max is faster than gumbel_max, but it requires one more softmax, so not sure
            what the end result is.
            """
        else:
            mask = Variable(plain_max(x.data)) # plain_max accepts either logit or probability
        return x * mask.float()

    def __repr__(self):
        return "{}(stochastic={})".format(self.__class__.__name__, self.stochastic)


class GaussianSplit(torch.nn.Module):

    def __init__(self, split_position):
        torch.nn.Module.__init__(self)
        self.split_position = split_position

    def forward(self, x):
        mean   = x[:, :self.split_position]
        logvar = x[:, self.split_position:]
        return (mean, logvar, logvar.exp())


def init_weights(module):
    """
    Initialize Conv2d, Linear and BatchNorm2d
    """

    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            """
            Now that we can count on the variance of output units to be 1, or maybe 0.5 (due to ReLU), we have 
            Nin * var(w) = [1 or 2]
            var(w) = [1 or 2]/Nin
            std(w) = sqrt([1 or 2]/Nin)
            """
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            std = math.sqrt(2.0 / n) # 2 accounts for ReLU's half-slashing
            m.weight.data.normal_(0, std) 
            m.bias.data.zero_()
            print (n, m.weight.data.norm())
        elif isinstance(m, torch.nn.Linear):
            n = m.in_features # in_features
            std = math.sqrt(2.0 / n) # 2 accounts for ReLU's half-slashing
            m.weight.data.normal_(0, std)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def compute_i_to_coord(num_channels, grid_size):
    num_cells = grid_size**2
    units_per_cell = int(math.ceil(float(num_channels) / num_cells))
    i_to_coord = {}
    for i in xrange(num_channels):
        cell_id = i / units_per_cell
        cell_x = cell_id % grid_size
        cell_y = cell_id / grid_size
        i_to_coord[i] = (cell_x, cell_y)
    return i_to_coord


def get_topographic_decay_multiplier(conv_layer, grid_size=3, mults=[1,2,4]):
    """
    Topographically structured weight decay
    Compute a topographic_multiplier for conv_layer.weight
    """
    assert isinstance(conv_layer, torch.nn.Conv2d)
    CJ = conv_layer.in_channels
    CI = conv_layer.out_channels
    i_to_coord = compute_i_to_coord(CI, grid_size) # current layer
    j_to_coord = compute_i_to_coord(CJ, grid_size) # previous layer
    decay_multiplier = conv_layer.weight.data.clone().fill_(1)
    for i in xrange(CI):
        i_x, i_y = i_to_coord[i]
        for j in xrange(CJ):
            j_x, j_y = j_to_coord[j]
            dist = math.sqrt((i_x - j_x)**2 + (i_y - j_y)**2)
            dist = int(math.floor(dist))
            assert len(mults) > dist, '{} need at least {} elements'.format(mults, dist+1)
            decay_multiplier[i,j] = mults[dist]
    return decay_multiplier


def make_cnn_with_conf(model_conf, 
            dropout=torch.nn.Dropout, 
            activation=torch.nn.ReLU,
            batchnorm=torch.nn.BatchNorm2d):
    """
    Use a declarative cnn specification to create a model

    model_conf example: (vgg16 with BN and dropout)
    model_conf_16 = [
        ('input'  , (3,   None)),  # (3, None) or (1, None) for RGB or Black-white
        ('conv1_1', (64,  0.3)),   # (64 channels, 0.3 dropout)
        ('conv1_2', (64,  None)),  # (64 channels, no dropout)
        ('pool1'  , (2,   2)),     # max-pool, kernel=2, stride=2
        ('conv2_1', (128, 0.4)),
        ('conv2_2', (128, None)),
        ('pool2'  , (2,   2)),
        ('conv3_1', (256, 0.4)),
        ('conv3_2', (256, 0.4)),
        ('conv3_3', (256, None)),
        ('pool3'  , (2,   2)),
        ('conv4_1', (512, 0.4)),
        ('conv4_2', (512, 0.4)),
        ('conv4_3', (512, None)),
        ('pool4'  , (2,   2)),
        ('conv5_1', (512, 0.4)),
        ('conv5_2', (512, 0.4)),
        ('conv5_3', (512, None)),
        ('pool5'  , (2,   2)),
        ('drop5'  , (None,0.5)),  # plain dropout with 0.5 probability
        ('fc6'    , (512, 0.5)),  # fully-connected (with CNN), 512-channels, followd by 0.5 dropout
        ('logit'  , (10,  None)), # fully-connected (with CNN), 10-channels
        ('flatter', (None,None))
    ]
    """
    in_channels = -1
    """ early layers """
    layers = OrderedDict()
    for name, info in model_conf:
        if name=='input':
            in_channel, _ = info
        elif name.startswith('conv') or name.startswith('fc'):
            num_chan, drop_p = info
            k,pad = (3,1) if name.startswith('conv') else (1,0)
            print('number of output channels: {}'.format(num_chan))
            sub_layers = [
                torch.nn.Conv2d(in_channel, num_chan, kernel_size=k, padding=pad),
                activation(inplace=True)
            ]
            if batchnorm is not None:
                sub_layers.insert(1, batchnorm(num_chan))
            if drop_p is not None:
                sub_layers += [dropout(p=drop_p)]
            layers[name] = torch.nn.Sequential(*sub_layers)
            in_channel = num_chan
        elif name.startswith('pool'):
            k, s = info
            layers[name] = torch.nn.MaxPool2d(kernel_size=k, stride=s)
        elif name.startswith('drop'):
            _, drop_p = info
            layers[name] = dropout(p=drop_p)
        elif name.startswith('logit'):
            num_class, _ = info
            layers[name] = torch.nn.Conv2d(in_channel, num_class, kernel_size=1, padding=0)
        elif name.startswith('flatter'):
            layers[name] = Flatten()
        elif name.startswith('softmax'):
            layers[name] = torch.nn.Softmax()
        else:
            assert False
    model = torch.nn.Sequential(layers)
    model.model_conf = model_conf # capture the model_conf for serialization
    init_weights(model)
    return model
