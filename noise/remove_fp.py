import os
from tqdm import tqdm
from ast import literal_eval
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import cv2
import skimage.io
import numpy as np

from eval_coco import *
from augmentation import augment_img

from pipeline_utils import *
from train_clutter import *
import model as modellib, visualize, utils


def remove_bin_fps(pred_mask_dir, gt_mask_dir, bin_mask_dir):
    """Given GT segmasks, predictions, and binary bin-or-not-bin segmasks,
    bitwise-ANDs every GT and predicted segmask as to remove any non-bin
    pixels.

    Segmasks in bin_mask_dir are .pngs.
    """

    N = len([p for p in os.listdir(bin_mask_dir) if p.endswith('.png')])
    print("Removing bin pixels from GT and predictions")
    for i in tqdm(range(N)):
        im_name = 'image_{:06d}.npy'.format(i)
        bin_im_name = 'image_{:06d}.png'.format(i)
        pred_mask = np.load(os.path.join(pred_mask_dir, im_name))
        gt_mask = np.load(os.path.join(gt_mask_dir, im_name))
        bin_mask = skimage.io.imread(os.path.join(bin_mask_dir, bin_im_name))
        print(pred_mask.shape, gt_mask.shape, bin_mask.shape)
