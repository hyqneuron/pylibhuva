
"""
The huva utility package

General utilities in this file
"""
import cv2
import numpy as np
from collections import OrderedDict
import cPickle


def clip(val, minval, maxval):
    """
    Clip val to [minval, maxval]
    """
    return max(minval, min(maxval, val))


def make_multiple(val, multiple_of):
    return int(round(val / float(multiple_of)) * multiple_of)


class LazyImage:
    """
    lazy image constructor. Loads the image only when get() is called
    """

    def __init__(self, full_path):
        self.full_path = full_path
        self.img = None

    def get(self):
        if self.img is None:
            self.img = cv2.imread(self.full_path)
        return self.img


class LogPrinter:
    """ 
    write logfile to a filename 
    In addition to printing the message, also log it into the logfile
    """

    def __init__(self, filename, struct_name=None):
        self.logtxt = ''
        self.struct_dict = OrderedDict()
        struct_name = struct_name or filename+'.pkl'
        self.file        = open(filename, 'w')
        self.struct_file = open(struct_name, 'w')

    def log(self, val, show_onscreen=True):
        # print a log message and add it to file also
        try:
            txt = str(val) + '\n'
            self.logtxt += txt
            self.file.write(txt)
            self.file.flush()
        except Exception as e:
            print("failed to write to logger: {}".format(e.message))
        if show_onscreen:
            print(val)

    def log_struct(self, key, value):
        # add to dict without printing
        self.struct_dict[key] = value

    def close(self):
        self.file.close()
        cPickle.dump(self.struct_dict, self.struct_file, protocol=-1)
        self.struct_file.close()

