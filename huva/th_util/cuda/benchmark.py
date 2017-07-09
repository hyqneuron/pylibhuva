import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter


"""
Order of operation:
    sparse1: C1HW
    sparse2: HWC1   -> first transpose
    compact: KC1
    mm     : KC2
    sparse3: HWC2
    sparse4: C2HW   -> second transpose

"""

