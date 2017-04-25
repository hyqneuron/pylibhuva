import torch
from collections import OrderedDict
import math

def gumbel_noise(x, eps=1e-10):
    """ 
    x is [N, ]
    """
    noise = x.new().resize_as_(x).uniform_(0, 1)
    return -(-(noise + eps).log() + eps).log()

def gumbel_max(x):
    noisy_x = x + gumbel_noise(x)
    max_x = noisy_x.max(1)[0].expand_as(x)
    mask_x = max_x == noisy_x
    return mask_x

def plain_max(x):
    max_x = x.max(1)[0].expand_as(x)
    mask_x = max_x == x
    return mask_x

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
    def forward(self, x):
        return x.view(x.size(0), *sizes)
    def __repr__(self):
        return "Flatten{}".format(str(sizes))

class InfoU(torch.nn.Module):
    def __init__(self, alpha=2, inplace=False):
        super(InfoU, self).__init__()
        self.alpha = alpha
        # ignore inplace
    def forward(self, x):
        exped   = (-x*self.alpha).exp()
        divided = exped.div(self.alpha+exped)
        logged  = divided.log() / self.alpha
        return -logged

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
