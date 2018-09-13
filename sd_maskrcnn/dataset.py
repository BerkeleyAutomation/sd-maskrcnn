import os
import sys
import logging
from tqdm import tqdm

import cv2
import skimage.io
import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import Mask R-CNN repo
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn import model as modellib, visualize, utils

"""
ImageDataset creates a Matterport dataset for a directory of
images in order to ensure compatibility with benchmarking tools 
and image resizing for networks.
Directory structure must be as follows:
$base_path/
    test_indices.npy
    train_indices.npy
    depth_ims/ (Depth images here)
        image_000000.png
        image_000001.png
        ...
    color_ims/ (Color images here)
        image_000000.png
        image_000001.png
        ...
    modal_segmasks/ (GT segmasks here, one channel)
        image_000000.png
        image_000001.png
        ...
"""

class ImageDataset(utils.Dataset):
    def __init__(self, base_path, images, masks):
        assert base_path != "", "You must provide the path to a dataset!"

        self.base_path = base_path
        self.images = images
        self.masks = masks
        super().__init__()

    def load(self, imset, augment=False):
        
        # Load the indices for imset.
        split_file = os.path.join(self.base_path, '{:s}'.format(imset))
        self.image_id = np.load(split_file)
        self.add_class('clutter', 1, 'fg')

        flips = [1, 2, 3]
        for i in self.image_id:
            if 'numpy' in self.images:
                p = os.path.join(self.base_path, self.images,
                                'image_{:06d}.npy'.format(i))
            else:
                p = os.path.join(self.base_path, self.images,
                                'image_{:06d}.png'.format(i))
            self.add_image('clutter', image_id=i, path=p, width=512, height=384)

            if augment:
                for flip in flips:
                    self.add_image('clutter', image_id=i, path=p, width=512, height=384, flip=flip)

    def flip(self, image, flip):
        # flips during training for augmentation

        if flip == 1:
            image = image[::-1,:,:]
        elif flip == 2:
            image = image[:,::-1,:]
        elif flip == 3:
            image = image[::-1,::-1,:]
        return image

    def load_image(self, image_id):
        # loads image from path
        if 'numpy' in self.images:
            image = np.load(self.image_info[image_id]['path']).squeeze()
        else:
            image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
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
        file_name = os.path.join(self.base_path, self.masks,
          'image_{:06d}.png'.format(_image_id))

        all_masks = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        for i in np.arange(1,np.max(all_masks)+1):
            I = all_masks == i # We ignore the background, so the first instance is 0-indexed.
            if np.any(I):
                I = I[:,:,np.newaxis]
                Is.append(I)
        if len(Is) > 0:
            mask = np.concatenate(Is, 2)
        else:
            mask = np.zeros([info['height'], info['width'], 0], dtype=np.bool)

        class_ids = np.array([1 for _ in range(mask.shape[2])])
        return mask, class_ids.astype(np.int32)

    @property
    def indices(self):
        return self.image_id
