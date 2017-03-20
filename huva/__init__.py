
"""
The huva utility package

General utilities in this file
"""
import cv2
import numpy as np

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


