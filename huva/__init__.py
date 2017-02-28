
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




    


