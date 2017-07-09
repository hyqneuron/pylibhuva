import torch
import math


class TrainingState:
    """
    This class wraps some training state

    It is designed as a state wrapper, with very minimal logic.
    
    Usually you should write your own training loop. If you are too lazy to do even that, you can use the default
    training loop, with a set of batch/epoch/age/final plugin functions. (note: using the default loop may reduce
    readability of your code)

    When you use the default training loop, remember to specify self.epochs and self.loader
    - epochs: a list of number of epochs
    - loader: a DataLoader for training
    """

    def __init__(self, model, optimizer, 
            batch_plugins=[], epoch_plugins=[], age_plugins=[], final_plugins=[],
            **kwargs):
        self.model         = model
        self.optimizer     = optimizer
        self.batch_plugins = list(batch_plugins)
        self.epoch_plugins = list(epoch_plugins)
        self.age_plugins   = list(age_plugins)
        self.final_plugins = list(final_plugins)
        self.__dict__.update(kwargs)

    def train(self):
        # default training loop for the uber-lazy
        # you should write your own training loop if you want decent code readability
        assert hasattr(self, 'epochs'), 'To use TrainingState.train, you must specify self.epochs'
        assert hasattr(self, 'loader'), 'To use TrainingState.train, you must specify self.loader'
        for age, num_epochs in enumerate(self.epochs):
            state.age = age
            for epoch in xrange(num_epochs):
                state.epoch = epoch
                for batch, input in enumerate(self.loader):
                    state.batch = batch
                    state.input = input
                    self.run_plugins(self.batch_plugins)
                self.run_plugins(self.epoch_plugins)
            self.run_plugins(self.age_plugins)
        self.run_plugins(self.final_plugins)

    def run_plugins(self, plugins):
        for plugin in plugins:
            plugin(self)


class TrainingPlugin(object):
    """
    This is one way to define a plugin
    alternatively, just use:
    def plugin_func(state):
        pass
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, state):
        pass


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

