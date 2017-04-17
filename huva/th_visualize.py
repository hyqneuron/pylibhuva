import torch
import math
from th_nn_stats import *

"""
1. Collect network activation
2. Pick maximal activation
3. Perform back-propagation from that point

We need the layers to keep output, after that, we need the particular sample which led to maximal output
1. run collect_output_over_loader 
"""

def collect_output_and_visualization(model, layer, unit_idx, dataset, loader, max_batches=99999, top_k=1, backward_thresh=True):
    name_layer = {name: module for name, module in model.named_modules() if module is layer}
    name_output = collect_output_over_loader(model, name_layer, loader, max_batches=max_batches)
    assert len(name_output)==1
    name, output = name_output.items()[0]
    return visualize_layer_unit(model, name, layer, output, unit_idx, dataset, top_k=top_k, backward_thresh=backward_thresh)

def visualize_layer_unit(model, name, layer, output, unit_idx, dataset, top_k=1, backward_thresh=True):
    """
    Visualize a unit's top-activation, using guided back-propagation

    Steps:
    1. for the given unit, select top-k activations, and their sample index
    2. group those top-k samples into a batch
    3. forward-backward-hook prop
    """
    """ find top-k activation """
    N = output.size(0) # number of samples
    K = top_k
    assert N >= K
    n_outputflat  = output[:, unit_idx].contiguous().view(N, -1)
    n_maxval, n_maxid = n_outputflat.max(1)                         # find max in every sample
    k_n     = n_maxval.squeeze().sort(0, descending=True)[1][:K]    # sort samples by max, take top-K
    k_maxid = n_maxid.squeeze()[k_n]
    """ group top-k samples into batch """
    batch_sizes = torch.Size([K] + list(dataset[0][0].size()))
    k_sample = torch.Tensor(batch_sizes)
    for k in xrange(K): # j indexes top-activation
        k_sample[k] = dataset[k_n[k]][0]
    """ apply hooks to backward passes """
    backward_hooks=[]
    if backward_thresh:
        print('doing backwards')
        def backward_relu_hook(module, grad_input, grad_output):
            # print((grad_input.eq(grad_output)).long().sum(), grad_input.nelement())
            assert isinstance(module, torch.nn.ReLU)
            assert type(grad_input)==tuple and len(grad_input)==1
            assert type(grad_output)==tuple and len(grad_output)==1
            modified_grad = grad_input[0].ge(0).float() * grad_input[0]
            return (modified_grad,)

        """ tag all ReLU layers up to the target layer """
        for name, module in model.named_modules():
            if module is layer: break
            if isinstance(module, torch.nn.ReLU) or module is layer:
                print('register hook for {}'.format(name))
                backward_hooks.append(module.register_backward_hook(backward_relu_hook))
    forward_hook = make_layer_keep_output(layer)
    """ forward-backward """
    model.eval()
    v_inputs = Variable(k_sample.cuda(), requires_grad=True)
    v_output = model(v_inputs)
    if not isinstance(layer, torch.nn.ReLU):
        v_layer_output = torch.nn.ReLU()(layer.output)
    else:
        v_layer_output = layer.output
    v_layer_output.data.zero_()
    for k in xrange(K):
        v_layer_output.data[k].view(-1)[k_maxid[k]] = 1
    summed = v_layer_output.sum()
    summed.backward()
    """ remove hooks """
    forward_hook.remove()
    for h in backward_hooks:h.remove()
    return v_inputs.data, v_inputs.grad.data

