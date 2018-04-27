import os
import sys
import logging
from tqdm import tqdm

import cv2
import skimage.io
import numpy as np
import tensorflow as tf

from clutter import ClutterConfig
from augmentation import augment_img
from eval_coco import *

from train_clutter import mkdir_if_missing
import model as modellib, visualize, utils

"""
SimImageDataset creates a Matterport dataset for a directory of simulated
images in order to ensure compatibility with functions from Saurabh's code and
MaskRCNN code, e.g. old benchmarking tools and image resizing for networks.

Directory structure must be as follows:

$base_path/
    train_indices.npy
    depth_ims/ (Depth images here)
        image_000000.png
        image_000001.png
        ...
    modal_segmasks/ (GT segmasks here, one channel)
        image_000000.png
        image_000001.png
        ...
"""

class SimImageDataset(utils.Dataset):
    def __init__(self, base_path=""):
        assert base_path != "", "You must provide the path to a dataset!"

        self.base_path = base_path
        super().__init__()

    def load(self, imset):
        # Load the indices for imset.

        split_file = os.path.join(self.base_path, '{:s}_indices.npy'.format(imset))
        self.image_id = np.load(split_file)
        self.add_class('clutter', 1, 'fg')

        # because we are training
        flips = [0, 1, 2, 3]

        for i in self.image_id:
            p = os.path.join(self.base_path, 'depth_ims',
                             'image_{:06d}.png'.format(i))
            self.add_image('clutter', image_id=i, path=p, width=256, height=256)

            for flip in flips:
                self.add_image('clutter', image_id=i, path=p, width=600, height=400, flip=flip)

    def flip(self, image, flip):
        # flips during training for augmentation

        if flip == 0:
            image = image
        elif flip == 1:
            image = image[::-1,:,:]
        elif flip == 2:
            image = image[:,::-1,:]
        elif flip == 3:
            image = image[::-1,::-1,:]
        return image

    def load_image(self, image_id):
        # loads image from path

        info = self.image_info[image_id]
        image = cv2.imread(info['path'])
        assert(image is not None)
        if image.ndim == 2: image = np.tile(image[:,:,np.newaxis], [1,1,3])
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "clutter":
            return info["path"] + "-{:d}".format(info["flip"])
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        # loads mask from path

        info = self.image_info[image_id]
        _image_id = info['id']
        Is = []
        file_name = os.path.join(self.base_path, 'modal_segmasks',
          'image_{:06d}.png'.format(_image_id))

        all_masks = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

        for i in range(25):
            I = all_masks == i+1 # We ignore the background, so the first instance is 0-indexed.
            if np.any(I):
                I = I[:,:,np.newaxis]
                Is.append(I)
        if len(Is) > 0:
            mask = np.concatenate(Is, 2)
        else:
            mask = np.zeros([info['height'], info['width'], 0], dtype=np.bool)

        class_ids = np.array([1 for _ in range(mask.shape[2])])
        return mask, class_ids.astype(np.int32)
