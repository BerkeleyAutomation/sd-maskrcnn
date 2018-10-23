"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Author: Mike Danielczuk
"""

import os
import skimage
import numpy as np

from mrcnn.utils import Dataset


"""
TargetDataset creates a Matterport dataset for a directory of
target images, used for target detection branch training.
Directory structure must be as follows:
$base_path/
    target_indices.npy
    depth_ims/ (Depth images here)
        image_000000.png
        image_000001.png
        ...
    color_ims/ (Color images here)
        image_000000.png
        image_000001.png
        ...
"""

class TargetDataset(utils.TargetDataset):
    def __init__(self, base_path, images, masks):
        assert base_path != "", "You must provide the path to a dataset!"

        self.base_path = base_path
        self.images = images
        self.masks = masks
        super().__init__()

    def load(self, imset):
        # Load the indices for imset.
        split_file = os.path.join(self.base_path, '{:s}'.format(imset))
        self.image_id = np.load(split_file)

        flips = [1, 2, 3]
        for i in self.image_id:
            if 'numpy' in self.images:
                p = os.path.join(self.base_path, self.images,
                                'image_{:06d}.npy'.format(i))
            else:
                p = os.path.join(self.base_path, self.images,
                                'image_{:06d}.png'.format(i))
            self.add_image('target', image_id=i, path=p)

    def load_target(self, target_id):
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

    def load_bb(self, target_id):
        """These are cropped, so we'll use utils.resize_image instead."""
        return None


"""
ImageDataset creates a Matterport dataset for a directory of
images in order to ensure compatibility with benchmarking tools
and image resizing for networks.
Directory structure must be as follows:
$base_path/
    test_indices.npy
    train_indices.npy
    images/ (Train/Test Images here)
        image_000000.png
        image_000001.png
        ...
    segmasks/ (GT segmasks here, one channel)
        image_000000.png
        image_000001.png
        ...
"""

class ImageDataset(Dataset):
    def __init__(self, config):
        assert config['dataset']['path'] != "", "You must provide the path to a dataset!"

        self.dataset_config = config['dataset']
        self.base_path = config['dataset']['path']
        self.images = config['dataset']['images']
        self.masks = config['dataset']['masks']

        self._channels = config['model']['settings']['image_channel_count']
        super().__init__()

    def load(self, indices_file, augment=False):
        # Load the indices for imset.
        split_file = os.path.join(self.base_path, '{:s}'.format(indices_file))
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
            self.add_image('clutter', image_id=i, path=p)

            if augment:
                for flip in flips:
                    self.add_image('clutter', image_id=i, path=p, flip=flip)

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
        
        if self._channels < 4 and image.shape[-1] == 4 and image.ndim == 3:
            image = image[...,:3]
        if self._channels == 1 and image.ndim == 2:
            image = image[:,:,np.newaxis]
        elif self._channels == 1 and image.ndim == 3:
            image = image[:,:,0,np.newaxis]
        elif self._channels == 3 and image.ndim == 3 and image.shape[-1] == 1:
            image = skimage.color.gray2rgb(image)
        elif self._channels == 4 and image.shape[-1] == 3:
            concat_image = np.concatenate([image, image[:,:,0:1]], axis=2)
            assert concat_image.shape == (image.shape[0], image.shape[1], image.shape[2] + 1), concat_image.shape
            image = concat_image
            
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

        all_masks = skimage.io.imread(file_name)
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
