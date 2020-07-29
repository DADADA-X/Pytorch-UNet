#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    """返回目录中的id列表，是一个生成器。生成器：()， 列表生成式：[]"""
    return (f[:-4] for f in os.listdir(dir))  # 返回指定的文件夹包含的文件或文件夹的名字的列表


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for id in ids for i in range(n))
    # 得到生成器((id1,0),(id1,1),(id2,0),(id2,1),...,(idn,0),(idn,1))
    # 这样的作用是后面会通过后面的0,1作为utils.py中get_square函数的pos参数，pos=0的取左边的部分，pos=1的取右边的部分


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)


def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.tif', scale) # 生成器

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.png', scale)  #

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)