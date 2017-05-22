import torch
from torch.autograd import Variable
from .image import tile_images, save_image, enlarge_image_pixel
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def normalize(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / (tensor.max() + 1e-8)
    return tensor


def get_model_param_norm(model):
    """ get parameter length of a model """
    return math.sqrt(sum([p.data.norm()**2 for p in model.parameters()]))

""" 
========================= Layer unit utilization statistics ===========================================
"""

def get_layer_utilization(layer):
    """
    Analyze whether the units in previous layer are dead

    weight dimension: [num_out, num_in, k, k]
    Return a summary of size [num_in]
    """
    W = layer.weight.data.cpu()
    W_mean = W.view(W.size(0), W.size(1), -1).abs().mean(2).squeeze(2)
    W_summary = W_mean.mean(0).squeeze(0)
    return W_summary


def get_model_utilization(module, threshold=1e-20, result=None, prefix=''):
    """
    Analyze dead-unit statistics. Perform analysis on all Conv2d modules in module
    """
    if result is None:
        result = {}
    for name, sub_mod in module._modules.iteritems():
        full_name = prefix + name
        if isinstance(sub_mod, torch.nn.Conv2d):
            result[full_name] = get_layer_utilization(sub_mod).lt(threshold).sum(), sub_mod.weight.size(1)
        elif hasattr(sub_mod, '_modules'):
            get_model_utilization(sub_mod, threshold, result=result, prefix=full_name+'/')
    return result

""" 
========================= Layer output statistics ====================================================
"""

def make_layer_keep_output(layer):
    """
    An obsolete method for keeping the output of a layer after a forward pass. Output is kept at layer.output
    """
    def forward_keep_output(module, input, output):
        assert module == layer
        layer.output = output
    return layer.register_forward_hook(forward_keep_output)


def collect_output_over_loader(model, name_layer, loader, max_batches=999999, selector=0, flatten=False, mode='output'):
    """
    loader: the loader from which to extract batches, should not use shuffling
    max_batches: maximum number of batches to collect
    selector: each batch sometimes is a (input, label) tuple, selector is the index to choose which one. Set it to 0 to
        choose input, set it to 1 to choose label. Set it to 'all' to return the whole thing.
    """
    batches = []
    for batch, content in enumerate(loader):
        if batch >= max_batches: break
        if selector == 'all':
            selected = content
        elif type(selector)==int:
            selected = content[selector]
        else:
            assert False, 'Unknown selector type'
        if flatten:
            selected = selected.view(selected.size(0), -1)
        batches.append(selected)
    return collect_output_over_batches(model, name_layer, batches, mode)


def collect_output_over_batches(model, name_layer, batches, mode='output'):
    """
    model: the model whose output we collect
    name_layer: {str_name:layer}, only collect output from layers listed here
    batches: [Tensor], a list of batch inputs
    mode: 'output' collects output, 'input' collects input instead
    """
    name_output = {}    # every target layer has an output buffer
    name_hook = {}      # every target layer has a forward_hook, which collects output on each pass
    total_size = sum(map(lambda inputs:inputs.size(0), batches))
    batch_size = batches[0].size(0)
    """ register output-saving hooks """
    for name, layer in name_layer.iteritems():
        def save_output(module, input, output, name=name, layer=layer, mode=mode): # stupid python scoping
            assert module == layer
            if mode=='output':
                target = output.data.cpu()
            elif mode=='input':
                if type(input) in [tuple, list]: 
                    assert len(input) == 1
                    input = input[0]
                target = input.data.cpu()
            else:
                assert False, 'unknown mode: {}'.format(mode)
            # first run
            if name not in name_output:
                name_output[name] = target
                sizel = list(name_output[name].size())
                sizel[0] = total_size # capture the NxCxHxW, but change N to total_size
                name_output[name].resize_(sizel) 
            # subsequent runs
            else:
                name_output[name][i*batch_size:i*batch_size+inputs.size(0)] = target
        name_hook[name] = layer.register_forward_hook(save_output)
    """ collect data over batches """
    for i, inputs in enumerate(batches):
        v_inputs = Variable(batches[i]).cuda()
        model(v_inputs)
    """ remove hooks """
    for name, hook in name_hook.iteritems(): 
        hook.remove()
    return name_output


class OutputStats(object):

    def __init__(self, outputt, output_std, order_std, output_skew, order_skew, output_kurtosis, order_kurtosis, covariance):
        self.outputt = outputt
        self.std      = output_std
        self.skew     = output_skew
        self.kurtosis = output_kurtosis
        self.std_order = order_std
        self.skew_order= order_skew
        self.kurtosis_order = order_kurtosis
        self.covariance = covariance


def get_outputt_kurtosis(outputt):
    """
    kurtosis   = E[(x-mu)**4] / E[(x-mu)**2] ** 2
    """
    mu = outputt.mean(1)
    x_minus_mu = outputt - mu.expand_as(outputt)
    numerator  = (x_minus_mu.pow(4)).mean(1)
    denominator= outputt.var(1).pow(2)
    return numerator / (denominator + 0.0000001)


def get_outputt_skewedness(outputt):
    """
    skewedness = E[(x-mu)**3] / E[(x-mu)**2] **1.5
    """
    mu = outputt.mean(1)
    x_minus_mu = outputt - mu.expand_as(outputt)
    numerator  = (x_minus_mu.pow(2)).mean(1)
    denominator= outputt.var(1).pow(1.5)
    return numerator / (denominator + 0.0000001)


def get_outputt_covariance(outputt):
    shifted = outputt - outputt.mean(1).expand_as(outputt)
    normed  = shifted / shifted.std(1).expand_as(shifted)
    covariance = normed.mm(normed.transpose(0,1)) / normed.size(1)
    return covariance


def get_output_stats(output):
    """
    output: [N,C] or [N,C,K,K]
    transpose to [C,N] or [C,N,K,K]
    compute std for every c, and order them
    """
    assert output.dim()>=2
    outputt = output.transpose(0,1).contiguous()
    outputt = outputt.view(outputt.size(0), -1) # Cx(NxHxW)

    output_std = outputt.std(1).squeeze(1)
    ordered_std, order_std = output_std.sort()

    output_skew = get_outputt_skewedness(outputt).squeeze(1)
    ordered_skew, order_skew = output_skew.sort()

    output_kurtosis = get_outputt_kurtosis(outputt).squeeze(1)
    ordered_kurtosis, order_kurtosis = output_kurtosis.sort()

    covariance = get_outputt_covariance(outputt)
    return OutputStats(outputt, output_std, order_std, output_skew, order_skew, output_kurtosis, order_kurtosis, covariance)


def show_output_hist(outputt, i, bins=40):
    """ visualize i'th ordered outputt's histogram, i.e. outputt[order[i]] """
    plt.hist(outputt[i].numpy(), bins=bins)
    plt.show()


def save_output_hist(outputt, i, path, bins=40):
    """ save i'th ordered outputt's histogram to path """
    plt.close()
    plt.hist(outputt[i].numpy(), bins=bins)
    plt.savefig(path)
    plt.close()


def save_all_output_hist(outputt, output_std, order, path_pattern, bins=40):
    for order_i in xrange(order.size(0)):
        i = order[order_i]
        save_output_hist(outputt, order, i, path_pattern.format(order_i, output_std[i]), bins=bins)


def collect_output_save_hist_for_layer(model, layer, loader, path_folder, max_batches=20, selector=0, mode='output'):
    name_layer = {name:module for name,module in model.named_modules() if module is layer}
    name = name_layer.keys()[0]
    name_output = collect_output_over_loader(model, name_layer, loader, max_batches=max_batches, selector=selector, mode=mode)
    output = name_output[name]
    stats = get_output_stats(output)
    num_units = stats.outputt.size(0)
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    for i in xrange(num_units):
        savepath = os.path.join(path_folder, 'hist_{}_{}_{}.jpg'.format(mode, name, i))
        save_output_hist(stats.outputt, i, savepath)

""" 
========================= Layer weight statistics ====================================================
"""

def get_weight_cosine(W):
    """
    W is [out, in, k, k] or [out, in]
    """
    assert W.dim() in [2,4]
    W = W.view(W.size(0), -1)
    norm = W.norm(2, 1) # L2 norm, across dim-1
    normed = W / norm.expand_as(W)
    dotted = normed.mm(normed.t())
    return dotted


def show_conv1_weights(conv1):
    W = conv1.weight.data.cpu() # [Cout, 3, k, k]
    assert W.dim() == 4
    assert W.size(1) == 3
    assert W.size(2) == W.size(3)
    k = W.size(2)
    num_unit = W.size(0)
    grid_side = int(math.ceil(math.sqrt(num_unit)))
    grid_length = grid_side * (k+1)-1
    grid = W.new().resize_(3, grid_length, grid_length).fill_(0)
    for i in xrange(num_unit):
        x = (i % grid_side)*(k+1)
        y = (i / grid_side)*(k+1)
        grid[:, y:y+k, x:x+k] = W[i]
    grid = normalize(grid)
    plt.imshow(grid.numpy().transpose([1,2,0]))
    plt.show()


def save_mlp_decoder_weights(w, filename):
    w = w.t().contiguous()
    assert w.dim()==2
    num_in, num_out = w.size
    num_side = int(math.sqrt(num_out))
    assert num_side**2 == num_out, 'cannot handle non-square decoder'
    w = w.t().contiguous().view(num_in, 1, num_side, num_side)
    save_image(w, filename)


"""
========================= Model Summary Report =======================================================
"""


"""
build a tracer:
* outgoing weight distribution, how many units in the next layer use     this unit
* incoming weight distribution, how many units in the prev layer used by this unit
* received gradient distribution
* outgoing gradient distribution
"""

