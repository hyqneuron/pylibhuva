import torch
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict
import math


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


class ScalarOp(torch.nn.Module):
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


class TensorOp(torch.nn.Module):

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


class SplitTake(torch.nn.Module):

    def __init__(self, split_position):
        torch.nn.Module.__init__(self)
        self.split_position = split_position

    def forward(self, x):
        return x[:, :self.split_position].contiguous()


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
