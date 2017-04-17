import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt

def get_model_param_norm(model):
    """ get parameter length of a model """
    return math.sqrt(sum([p.data.norm()**2 for p in model.parameters()]))

""" 
========================= Layer unit utilization statistics ===========================================
"""
def get_layer_kk_utilization(layer):
    """
    Analyze dead-cell statistics
    weight dimension: [num_out, num_in, k, k]
    return a summary of size [num_in, k, k]
    """
    W = layer.weight.data.cpu()
    mean = W.abs().mean(0).squeeze(0)
    return mean

def get_layer_utilization(layer):
    """
    Analyze dead-unit statistics

    weight dimension: [num_out, num_in, k, k]
    Return a summary of size [num_in]
    """
    W = layer.weight.data.cpu()
    W_mean = W.view(W.size(0), W.size(1), -1).abs().mean(2).squeeze(2)
    W_summary = W_mean.mean(0).squeeze(0)
    return W_mean, W_summary

def get_all_utilization(module, threshold=1e-20, result=None, prefix=''):
    """
    Analyze dead-unit statistics
    """
    if result is None:
        result = {}
    for name, sub_mod in module._modules.iteritems():
        full_name = prefix + name
        if isinstance(sub_mod, torch.nn.Conv2d):
            result[full_name] = get_layer_utilization(sub_mod)[1].lt(threshold).sum(), sub_mod.weight.size(1)
        elif hasattr(sub_mod, '_modules'):
            get_all_utilization(sub_mod, threshold, result=result, prefix=full_name+'/')
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

def collect_output_over_loader(model, name_layer, loader, max_batches=999999, selector=0, flatten=False):
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
    return collect_output_over_batches(model, name_layer, batches)


def collect_output_over_batches(model, name_layer, batches):
    """
    model: the model whose output we collect
    name_layer: {str_name:layer}, only collect output from layers listed here
    batches: [Tensor], a list of batch inputs
    """
    name_output = {}    # every target layer has an output buffer
    name_hook = {}      # every target layer has a forward_hook, which collects output on each pass
    total_size = sum(map(lambda inputs:inputs.size(0), batches))
    batch_size = batches[0].size(0)
    """ register output-saving hooks """
    for name, layer in name_layer.iteritems():
        def save_output(module, input, output, name=name, layer=layer): # stupid python scoping
            assert module == layer
            # first run
            if name not in name_output:
                name_output[name] = output.data.cpu()
                sizel = list(name_output[name].size())
                sizel[0] = total_size # capture the NxCxHxW, but change N to total_size
                name_output[name].resize_(sizel) 
            # subsequent runs
            else:
                name_output[name][i*batch_size:i*batch_size+inputs.size(0)] = output.data.cpu()
        name_hook[name] = layer.register_forward_hook(save_output)
    """ collect data over batches """
    for i, inputs in enumerate(batches):
        v_inputs = Variable(batches[i]).cuda()
        model(v_inputs)
    """ remove hooks """
    for name, hook in name_hook.iteritems(): 
        hook.remove()
    return name_output

def get_output_std(output):
    """
    output: [N,C] or [N,C,K,K]
    transpose to [C,N] or [C,N,K,K]
    compute std for every c, and order them
    """
    assert output.dim()>=2
    outputt = output.transpose(0,1).contiguous()
    outputt = outputt.view(outputt.size(0), -1) # Cx(NxHxW)
    output_std = output.std(1).squeeze(1)
    ordered_std, order = output_std.sort()
    return outputt, output_std, order

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


"""
build a tracer:
* unit activation mean, std, histogram
* outgoing weight distribution, how many units in the next layer use     this unit
* incoming weight distribution, how many units in the prev layer used by this unit
* received gradient distribution
* outgoing gradient distribution
"""

