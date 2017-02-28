import torch
import cv2
import matplotlib.pyplot as plt
from np_util import *

def th_get_jet(img, label):
    img_np = bgr_to_numpy(img)
    H, W         = label.size(1), label.size(2)
    img_H, img_W = img.size(1),   img.size(2)
    label_np = label.numpy().reshape(H, W, 1)
    jet = get_jet(img_np, cv2.resize(label_np, (img_W, img_H)))
    return jet

def img_to_numpy(img):
    """
    img of range[0,1] shape 3xHxW
    return uint8 numpy array HxWx3
    """
    return (img*255).numpy().astype('uint8').transpose([1,2,0])

def bgr_to_numpy(torch_bgr_img, rgb=False):
    """
    torch_bgr_img is a 3xHxW torch.FloatTensor, range [0,255]
    returns a numpy uint8 array, HxWx3
    """
    result = torch_bgr_img.numpy().astype('uint8').transpose([1,2,0])
    if rgb:
        result = result[:,:,[2,1,0]]
    return result

def save_bgr_3hw(path, img):
    """
    img is a 3xHxW torch.Tensor, values range [0,255]
    """
    return cv2.imwrite(path, img.numpy().transpose([1,2,0]))


def load_bgr_3hw(path):
    """
    returns a 3xHxW torch.Tensor, values range [0,255]
    """
    return torch.from_numpy(cv2.imread(path).transpose([2,0,1]))

def imshow_th(img_th):
    imshow_np(torch.img_to_numpy(img_th))
