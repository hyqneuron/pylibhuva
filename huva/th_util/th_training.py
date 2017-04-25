import torch
import math

def get_num_correct(output, labels):
    """
    Compute number of correrct predictions for CrossEntropyLoss
    output is N by num_class
    labels is N by 1
    """
    maxval, maxpos = output.max(1)
    equals = maxpos == labels
    return equals.sum()


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def decay_learning_rate(optimizer, decay):
    for group in optimizer.param_groups:
        group['lr'] *= decay
