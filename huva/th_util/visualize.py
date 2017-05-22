import torch
import math
from .nn_stats import *
from .image import tile_images, save_image, enlarge_image_pixel
import os

"""
1. Collect network activation
2. Pick maximal activation
3. Perform back-propagation from that point

We need the layers to keep output, after that, we need the particular sample which led to maximal output
1. run collect_output_over_loader 
"""


def collect_output_and_guided_backprop(
        model, layer, unit_idx, dataset, loader, max_batches=99999, top_k=1, backward_thresh=True, mode='output'):
    name_layer = {name: module for name, module in model.named_modules() if module is layer}
    name_output = collect_output_over_loader(model, name_layer, loader, max_batches=max_batches, mode=mode)
    assert len(name_output)==1
    name, output = name_output.items()[0]
    return visualize_layer_unit(model, layer, output, unit_idx, dataset, top_k=top_k, backward_thresh=backward_thresh)


def guided_backprop_layer_unit(model, layer, output, unit_idx, dataset, top_k=1, backward_thresh=True, selector=0):
    """
    Visualize a unit's top-activation, using guided back-propagation

    layer: the layer to analyze
    unit_idx: the unit to analyze (inside layer)
    model: the model to which the layer belongs

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
    batch_sizes = torch.Size([K] + list((dataset[0] if selector=='all' else dataset[0][selector]).size()))
    k_sample = torch.Tensor(batch_sizes)
    for k in xrange(K): # j indexes top-activation
        if selector=='all':
            k_sample[k] = dataset[k_n[k]]
        else:
            k_sample[k] = dataset[k_n[k]][selector]
    """ apply hooks to backward passes """
    backward_hooks=[]
    if backward_thresh:
        def backward_relu_hook(module, grad_input, grad_output):
            assert isinstance(module, torch.nn.ReLU)
            assert type(grad_input)==tuple and len(grad_input)==1
            assert type(grad_output)==tuple and len(grad_output)==1
            modified_grad = grad_input[0].ge(0).float() * grad_input[0]
            return (modified_grad,)

        """ tag all ReLU layers up to the target layer """
        for name, module in model.named_modules():
            if module is layer: break
            if isinstance(module, torch.nn.ReLU) or module is layer:
                backward_hooks.append(module.register_backward_hook(backward_relu_hook))
    forward_hook = make_layer_keep_output(layer)
    """ forward-backward """
    model.eval()
    v_inputs = Variable(k_sample.cuda(), requires_grad=True)
    v_output = model(v_inputs)
    output_grad = layer.output.data.clone().zero_()
    for k in xrange(K):
        if output_grad.dim()==2: # MLP
            output_grad[k, unit_idx] = 1
        else: # CNN
            output_grad[k, unit_idx].view(-1)[k_maxid[k]] = 1
    v_layer_output = layer.output
    v_layer_output.backward(output_grad)
    """ remove hooks """
    forward_hook.remove()
    for h in backward_hooks:
        h.remove()
    return v_inputs.data, v_inputs.grad.data


def collect_output_save_gbp_for_layer(
        model, layer, dataset, loader, path_folder, max_batches=20, K=10, selector=0, mode='output'):
    name_layer = {name:module for name,module in model.named_modules() if module is layer}
    name = name_layer.keys()[0]
    name_output = collect_output_over_loader(model, name_layer, loader, max_batches=20, selector=selector, mode=mode)
    output = name_output[name]
    num_units = output.size(1)
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    for i in xrange(num_units):
        inputs, grads = guided_backprop_layer_unit(model, layer, output, i, dataset, top_k=K, selector=selector)
        inputs = normalize(inputs)
        grads  = normalize(grads)
        if inputs.dim() == 2: # MLP data
            num_inputs = inputs.size(1)
            side_len = int(math.sqrt(num_inputs))
            assert side_len**2 == num_inputs, "Don't know how to handle MLP input that's not sqaure"
            inputs = inputs.contiguous().view(inputs.size(0), 1, side_len, side_len)
            grads  = grads .contiguous().view(grads .size(0), 1, side_len, side_len)
        images = torch.cat([inputs, grads], 0)
        tiled = tile_images(images, rows=2, cols=K, padding=4)
        tiled = enlarge_image_pixel(tiled, 5)
        savepath = os.path.join(path_folder, 'gbp_{}_{}.jpg'.format(name, i))
        save_image(tiled, savepath)
