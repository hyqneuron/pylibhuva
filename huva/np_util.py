import numpy as np
import cv2
import matplotlib.pyplot as plt
cmap = plt.get_cmap('jet')

def get_jet(img, label):
    """
    img is numpy uint8 array HxWx3
    label is numpy float32 array HxWx1 [0,1]
    """
    assert img.dtype == np.uint8
    assert label.dtype == np.float32
    H = label.shape[0]
    W = label.shape[1]
    heat_img = (cmap(label.reshape(H,W))[:,:,[0,1,2]] * 255).astype('uint8')
    return (img/2 + heat_img/2)

def imshow_np(img_np):
    plt.imshow(img_np)
    plt.show()

"""
image augmentation functions
"""
def mess_contrast(img_np, (alpha_min, alpha_max), (beta_min, beta_max)):
    assert img_np.dtype == np.uint8
    alpha = random.uniform(alpha_min, alpha_max)
    beta  = random.uniform(beta_min,  beta_max)
    return (alpha * img_np + beta).clip(0,255).astype(img_np.dtype)

def normalize_image(img_np):
    """
    normalize values to [0,255]
    """
    assert img_np.dtype == np.uint8
    alpha = 255.0 / (img_np.max() - img_np.min())
    beta  = - img_np.min() * alpha
    return (alpha * (img_np - img_np.min())).astype(img_np.dtype)

def swap_channels(img_np):
    """
    Swap BGR<->RGB
    """
    return img_np[:,:,[2,1,0]]

def add_black_band(img_np):
    width = random.randint(20,40)
    height= random.randint(20,40)
    img_np[-height:, :] = 0
    img_np[:,  -width:] = 0
    return width, height
