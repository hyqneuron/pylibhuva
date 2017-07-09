from .base import *
from .data import *       # datasets
from .math_ops import *       # max, kld and stuff
from .model import *      # custom modules
from .model_ccvae import *# ccvae modules
from .training import *   # training utilities

from .monitor import *    # monitor gradient descent
from .nn_stats import *   # nn statistics
from .visualize import *  # guided backpropagation visualization

from .tests import *

from .. import LogPrinter


def parse_bool(inp):
    return {'True':True, 'False':False}[inp]


def make_argument_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # name of experiment
    parser.add_argument('-n', '--name',         type=str)
    # dataset
    parser.add_argument('-d', '--dataset',      type=str)
    # optimization
    parser.add_argument('-b', '--batch-size',   type=int)
    parser.add_argument('-o', '--optimizer')
    parser.add_argument('-lr','--learning-rate',type=float)
    parser.add_argument('--momentum',           type=float,     default=0.0)
    parser.add_argument('-wd','--weight-decay', type=float,     default=0.0)
    parser.add_argument('-e', '--epochs',       type=int,       nargs='+')
    # LR decay
    parser.add_argument('--decays-epoch',       type=float,     nargs='+', default=[]) # per-epoch decay, one per age
    parser.add_argument('--decays-age',         type=float,     nargs='+', default=[]) # per-age   decay, one per age
    # logging
    parser.add_argument('--logfile',            type=str,       default='')
    parser.add_argument('--graphfolder',        type=str,       default='')
    parser.add_argument('--force-name',         action='store_true')
    parser.add_argument('--start',              action='store_true')
    parser.add_argument('-ri', '--report-interval', type=int,   default=50)
    return parser

