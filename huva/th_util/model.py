import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict
import math
from functools import wraps
from .base import new_as
import functions as FS



"""
==================================================================================================
Model
==================================================================================================
"""

class Model(nn.Module):
    """
    A Model has 2 parts:
    - network: computes forward pass (not defined in this class)
    - loss   : computes objective of optimization.

    Note that when loss has a complex structure (as in VAEs), the computation of loss might be delegated to network.

    losses = self.get_losses(state, label) has the freedom to return whatever it wants, provided:
    1. It is a tuple, whose first element is the default optimization objective
    1. self.backward      (state, losses)       can perform backprop from objective
    2. self.report_losses (state, losses)       can extract a report structure
    3. self.average_losses(state, list_losses)  can compute average of a list of losses
    """

    def forward(self, input):
        raise NotImplementedError

    def get_losses(self, state, label):
        """
        state: return value of forward
        """
        raise NotImplementedError

    def backward(self, state, losses):
        """
        state: return value of forward
        losses: return value of get_losses

        Backpropagation. optimizer.zero_grad() should be called before this. optimizer.step() should be called after
        this.
        """
        while type(losses) in [list, tuple]:
            losses = losses[0]
        losses.backward()

    def average_losses(self, list_losses):
        """
        Given a list of losses returned by get_losses, computes the average.
        """
        length = len(list_losses)
        if length == 0: 
            return 0
        result = reduce(structured_add, list_losses)
        return structured_divide(result, float(length))

    def report_losses(self, losses):
        """
        losses: return value of get_losses or average_losses

        Returns a subset of losses that should be reported, as structured floats
        """
        return extract_float(losses)

    def format_losses(self, reported, prec=3):
        """
        reported: return value of report_losses

        Returns a string representation of "reported" that can be reasonably printed.
        """
        return str_float(reported, prec)


def extract_float(value):
    if type(value) in [tuple, list]:
        return map(extract_float, value)
    assert value.size() == (1,)
    return value.data[0]


def str_float(value, prec):
    if type(value) in [tuple, list]:
        return '({})'.format(', '.join([str_float(val, prec) for val in value]))
    return '{:.{prec}f}'.format(value, prec=prec)


def structured_add(value1, value2):
    """
    Element-wise addition, permitting tuple and list as structures. Can be implemented as fmap, but unrolled for
    simplicity.

    E.g.: 
        value1 = (1,2,(3,4)); value2 = (5,6,(7,8))
        structured_add(value1, value2) = (6,8,(10,12))
    """
    assert type(value1) == type(value2) or (set(map(type, [value1, value2]))==set([list, tuple]))
    if type(value1) in [tuple, list]:
        assert len(value1) == len(value2)
        return [structured_add(v1, v2) for v1,v2 in zip(value1, value2)]
    else:
        return value1 + value2


def structured_divide(value, denominator):
    """
    Element-wise division by a single scalar

    E.g.:
        value = (2,4,(6,8))
        structured_divide(value, 2) = (1,2,(3,4))
    """
    if type(value) in [tuple, list]:
        return [structured_divide(val, denominator) for val in value]
    return value / denominator


"""
==================================================================================================
Modules
==================================================================================================
"""

class CachedSequential(nn.Sequential):
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


class PSequential(nn.Sequential): 

    # Pretty Sequential, allow __repr__ customization
    def __repr__(self, additional=''):
        tmpstr = self.__class__.__name__ + ' ({},'.format(additional)+'\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = nn.modules.module._addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


"""
==================================================================================================
Convolutions
==================================================================================================
"""

class ConvTranspose2d(nn.ConvTranspose2d):
    """
    Fixes ConvTransposed2d's weight initialization.
    Conv2d           input size is [in_channels, k,k], output size is [out_channels,1,1]
    ConvTransposed2d input size is [out_channels,1,1], output size is [in_channels, k,k]
    """

    def reset_parameters(self):
        n = self.in_channels
        """
        remove below, because input size is in_channels by 1x1, and output size is out_channels by kxk
        """
        # for k in self.kernel_size:
        #     n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class DepthwiseConv2d(nn.Module):

    def __init__(self, in_channels, k=3, stride=1, padding=1):
        assert k==3 and stride==1 and padding==1, 'currently only support k=3, stride=1, padding=1'
        super(DepthwiseConv2d, self).__init__()
        self.in_channels = in_channels
        self.k           = k
        self.stride      = stride
        self.padding     = padding
        self.weight = Parameter(torch.Tensor(in_channels, k, k))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.k*self.k)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return FS.DepthwiseConv2dCHW()(input, self.weight)

    def __repr__(self):
        return '{}(in_channels={}, k={}, stride={}, padding={})'.format(
                self.__class__.__name__, self.in_channels, self.k, self.stride, self.padding)


class SparsePointwiseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SparsePointwiseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, c1hw, indices):
        """
        c1hw -> hwc1 -> kc1 -> kc2 -> hwc2 -> c2hw
        """
        assert c1hw.dim() == 4
        N, C, H, W = c1hw.size()
        assert N==1 and C==self.in_channels
        hwc1 = c1hw.permute(0, 2, 3, 1).contiguous()# this one requires contiguous
        kc1 = FS.SpatialGatherHWC()(hwc1, indices)
        kc2 = self._backend.Linear()(kc1, self.weight)
        hwc2 = FS.SpatialScatterHWC(H, W)(kc2, indices)
        c2hw = hwc2.permute(0, 3, 1, 2).contiguous()
        return c2hw

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(
                self.__class__.__name__, self.in_channels, self.out_channels)

"""
==================================================================================================
Internal batchnorm
==================================================================================================
"""

class BatchNorm2d(nn.Module):

    def __init__(self, num_channels, affine=True, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.num_channels = num_channels
        running_mean = torch.Tensor(1, num_channels, 1, 1)
        running_var  = torch.Tensor(1, num_channels, 1, 1)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var',  running_var)
        if affine:
            self.weight = Parameter(torch.Tensor(1, num_channels, 1, 1).fill_(1))
            self.bias   = Parameter(torch.Tensor(1, num_channels, 1, 1).fill_(0))
        self.affine = affine
        self.momentum = momentum

    def forward(self, input):
        mean = input.data   .mean(0).view(1, input.size(1), -1).mean(2).unsqueeze(-1)
        diff = input.data - mean.expand_as(input.data)
        var  = (diff * diff).mean(0).view(1, input.size(1), -1).mean(2).unsqueeze(-1)
        output = (input - Variable(mean.expand_as(input.data))) * Variable(var.rsqrt().expand_as(input.data))

        self.running_mean = self.running_mean * self.momentum + mean * (1 - self.momentum)
        self.running_var  = self.running_var  * self.momentum + var  * (1 - self.momentum)

        if self.affine:
            weight = self.weight.expand_as(input)
            bias   = self.bias  .expand_as(input)
            output = output * weight + bias
        return output

    def __repr__(self, additional=''):
        return '{}(num_channels={})'.format(self.__class__.__name__, self.num_channels)


"""
==================================================================================================
Shape-changing
==================================================================================================
"""

class Identity(nn.Module):

    def forward(self, x):
        return x


class Flatten(nn.Module):
    """
    Flatten the output from dim-1 and onwards
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return "Flatten()"


class View(nn.Module):
    """
    View output as a specific size
    """

    def __init__(self, *sizes):
        nn.Module.__init__(self)
        self.sizes = sizes

    def forward(self, x):
        return x.view(x.size(0), *self.sizes)

    def __repr__(self):
        return "View({})".format(str(self.sizes))


class Chunk(nn.Module):
    """
    chunk tensor into pieces
    """

    def __init__(self, num_chunks, dim):
        nn.Module.__init__(self)
        self.num_chunks = num_chunks
        self.dim = dim

    def forward(self, x):
        return x.chunk(self.num_chunks, self.dim)

    def __repr__(self):
        return 'Chunk(num_chunks={}, dim={})'.format(self.num_chunks, self.dim)


class StridingTransform(nn.Module):

    def __init__(self, stride):
        """
        A: N x C x H x W
        B: N x C*ss x Hs x Ws
        SpaceToChannel: A --> B
        ChannelToSpace: B --> A
        """
        nn.Module.__init__(self)
        self.stride = (int(stride), int(stride)) if type(stride) in [int, float] else stride
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


class ChannelToSpace(StridingTransform):

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


class SplitTake(nn.Module):

    def __init__(self, split_position):
        nn.Module.__init__(self)
        self.split_position = split_position

    def forward(self, x):
        return x[:, :self.split_position].contiguous()


class GaussianSplit(nn.Module):

    def __init__(self, split_position):
        nn.Module.__init__(self)
        self.split_position = split_position

    def forward(self, input):
        mean   = input[:, :self.split_position]
        logvar = input[:, self.split_position:]
        return (mean, logvar, logvar.exp())


class SplitBatchNorm(nn.Module):

    def __init__(self, num_part1, num_part2, spatial, affines=(True, True)):
        nn.Module.__init__(self)
        self.num_part1 = num_part1
        self.num_part2 = num_part2
        self.spatial = spatial
        if spatial:
            self.bn1 = nn.BatchNorm2d(num_part1, affine=affines[0])
            self.bn2 = nn.BatchNorm2d(num_part2, affine=affines[1])
        else:
            self.bn1 = nn.BatchNorm1d(num_part1, affine=affines[0])
            self.bn2 = nn.BatchNorm1d(num_part2, affine=affines[1])

    def forward(self, input):
        part1 = input[:, :self.num_part1].contiguous()
        part2 = input[:, self.num_part1:].contiguous()
        part1_bned = self.bn1(part1)
        part2_bned = self.bn2(part2)
        return torch.cat([part1_bned, part2_bned], 1)


def batchnorm_set_halfbias(bn, bias=0.0):
    old_forward = bn.forward
    def new_forward(x):
        bn.bias.data.chunk(2)[0].fill_(bias)
        return old_forward(x)
    bn.forward = new_forward


"""
==================================================================================================
Scalar/Tensor Op
==================================================================================================
"""

class ScalarOp(nn.Module):
    def __init__(self, init_val=1, learnable=True, apply_exp=False):
        super(ScalarOp, self).__init__()
        self.init_val = init_val
        self.learnable= learnable
        self.apply_exp = apply_exp
        if apply_exp: 
            assert learnable, 'Only learnable scalar ops support apply_exp=True'
        if learnable:
            weight = Parameter(torch.Tensor(1).fill_(init_val))
            self.weight = weight

    def get_scalar(self, expand_as=None):
        if not self.learnable:
            return self.init_val
        scalar = self.weight.exp() if self.apply_exp else self.weight
        if expand_as is not None:
            scalar = scalar.expand_as(expand_as)
        return scalar

    def __repr__(self):
        return "{}(init_val={}, learnable={}, apply_exp={})".format(self.__class__.__name__, self.init_val, self.learnable, self.apply_exp)


class MultScalar(ScalarOp):

    def forward(self, x):
        return x * self.get_scalar(expand_as=x)


class DivideScalar(ScalarOp):

    def forward(self, x):
        return x / self.get_scalar(expand_as=x)


class TensorOp(nn.Module):

    def __init__(self, init_val, learnable=True, apply_exp=False):
        super(TensorOp, self).__init__()
        self.init_val = init_val
        self.learnable = learnable
        self.apply_exp = apply_exp
        if apply_exp:
            assert learnable, 'Only learnable tensor ops support apply_exp=True'
        if learnable:
            weight = Parameter(init_val.clone())
            self.weight = weight

    def get_tensor(self, expand_as=None):
        result = self.init_val
        if self.learnable:
            result = self.weight
        if expand_as is not None:
            result = result.expand_as(expand_as)
        return result

    def __repr__(self):
        return "{}(size={})".format(self.__class__.__name__, list(self.init_val.size()))


class AddTensor(TensorOp):

    def forward(self, x):
        return x + self.get_tensor(expand_as=x)


class PCALayer(nn.Module):

    def __init__(self, V, mean, direction):
        assert direction in ['encode', 'decode']
        super(PCALayer, self).__init__()
        self.direction = direction
        self.register_buffer('V', V)
        self.register_buffer('mean', mean)

    def reset(self, V, mean):
        self.register_buffer('V', V)
        self.register_buffer('mean', mean)

    def forward(self, input):
        if self.direction=='encode':
            return self.encode(input)
        else:
            return self.decode(input)

    def encode(self, data):
        shifted = data - Variable(self.mean.expand_as(data))
        coded   = torch.mm(shifted, Variable(self.V))
        return coded

    def decode(self, coded):
        shifted = torch.mm(coded, Variable(self.V.t()))
        data    = shifted + Variable(self.mean.expand_as(shifted))
        return data


"""
==================================================================================================
Miscellaneous
==================================================================================================
"""


def init_weights(module):
    """
    Initialize Conv2d, Linear and BatchNorm2d
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
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
        elif isinstance(m, nn.Linear):
            n = m.in_features # in_features
            std = math.sqrt(2.0 / n) # 2 accounts for ReLU's half-slashing
            m.weight.data.normal_(0, std)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
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
    assert isinstance(conv_layer, nn.Conv2d)
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
            dropout=nn.Dropout, 
            activation=nn.ReLU,
            batchnorm=nn.BatchNorm2d):
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
                nn.Conv2d(in_channel, num_chan, kernel_size=k, padding=pad),
                activation(inplace=True)
            ]
            if batchnorm is not None:
                sub_layers.insert(1, batchnorm(num_chan))
            if drop_p is not None:
                sub_layers += [dropout(p=drop_p)]
            layers[name] = nn.Sequential(*sub_layers)
            in_channel = num_chan
        elif name.startswith('pool'):
            k, s = info
            layers[name] = nn.MaxPool2d(kernel_size=k, stride=s)
        elif name.startswith('drop'):
            _, drop_p = info
            layers[name] = dropout(p=drop_p)
        elif name.startswith('logit'):
            num_class, _ = info
            layers[name] = nn.Conv2d(in_channel, num_class, kernel_size=1, padding=0)
        elif name.startswith('flatter'):
            layers[name] = Flatten()
        elif name.startswith('softmax'):
            layers[name] = nn.Softmax()
        else:
            assert False
    model = nn.Sequential(layers)
    model.model_conf = model_conf # capture the model_conf for serialization
    init_weights(model)
    return model

"""
==================================================================================================
Weight normalization
Modified from https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f

weight_norm: 
    function to wrap a nn.Module instance, making it weight-normalized

weight_norm_ctor: 
    function to wrap a nn.Module class. Result can be called as a constructor. This constructor will produce
    weight-normalized modules.

wn_decorate: 
    modifies module's forward function to include a normalizatio step
==================================================================================================
"""

def wn_decorate(forward, module, name, name_g, name_v, fixed_norm=None):
    @wraps(forward)
    def decorated_forward(*args, **kwargs):
        g = module.__getattr__(name_g)
        v = module.__getattr__(name_v)
        if fixed_norm is not None: # fix norm at a certain value
            assert isinstance(fixed_norm, float), '{} is not float'.format(fixed_norm)
            v.data.div_(v.data.norm()/fixed_norm) # bypass gradients
            w = v * g.expand_as(v)
        else:
            w = v*(g/torch.norm(v)).expand_as(v)
        module.__setattr__(name, w)
        return forward(*args, **kwargs)
    return decorated_forward


def weight_norm(module, name='weight', fix_norm=False, init_prop=False):
    param = module.__getattr__(name)

    # construct g,v such that w = g/||v|| * v
    g = torch.norm(param)
    if fix_norm:
        """
        Without init_prop, we fix norm at 10
        With init_prop, we fix norm at initial norm
        """
        if init_prop:
            v = param # v at this point has norm=original norm
            fixed_norm=g.data[0]
            g.data.fill_(1) # g start at 1
        else:
            v = param/g.expand_as(param) * 10.0 # v at this point has norm=10.0
            fixed_norm=10.0
            g.data.div_(10.0) # g start at 0.1 * original norm
    else:
        v = param/g.expand_as(param) # v at this point has norm=1
        fixed_norm = None
        # g start at original norm
    g = Parameter(g.data)
    v = Parameter(v.data)
    name_g = name + '_g'
    name_v = name + '_v'

    # remove w from parameter list
    del module._parameters[name]

    # add g and v as new parameters
    module.register_parameter(name_g, g)
    module.register_parameter(name_v, v)

    # construct w every time before forward is called
    module.forward = wn_decorate(module.forward, module, name, name_g, name_v, fixed_norm=fixed_norm)
    return module


def weight_norm_ctor(mod_type, name='weight', fix_norm=False, init_prop=False):
    def init_func(*args, **kwargs):
        mod = mod_type(*args, **kwargs)
        return weight_norm(mod, name=name, fix_norm=fix_norm, init_prop=init_prop)
    return init_func

