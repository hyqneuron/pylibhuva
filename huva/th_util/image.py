import cv2
import torch
from PIL import Image
import os


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


def tile_images(imgs, rows, cols, padding=0):
    """
    imgs is Nx3xHxW, where N=rows*cols
    """
    assert imgs.dim()==4
    N,C,H,W = list(imgs.size())
    assert C in [3,1]
    assert N==rows*cols
    tiled = imgs.new().resize_(C, (H+padding)*rows-padding, (W+padding)*cols-padding).fill_(0)
    for i in xrange(N):
        x = (i % cols)*(W+padding)
        y = (i / cols)*(H+padding)
        tiled[:, y:y+H, x:x+W] = imgs[i]
    return tiled


def save_image(img, filename, create_folder=False):
    img = img.cpu().float()
    img = img -  img.min()
    img = img / (img.max() + 1e-8)
    img = img.mul(255).byte().permute(1,2,0).squeeze()
    folder = os.path.dirname(filename)
    if not os.path.exists(folder) and create_folder:
        os.mkdir(folder)
    Image.fromarray(img.numpy()).save(filename)


def enlarge_image_pixel(img, times):
    """
    img: 3xHxW
    """
    assert img.dim()==3
    C,H,W = img.size()
    assert C in [1,3]
    enlarged = img.unsqueeze(3).unsqueeze(2).expand(C,H,times,W,times).contiguous()
    enlarged = enlarged.view(C,H*times, W*times)
    return enlarged
