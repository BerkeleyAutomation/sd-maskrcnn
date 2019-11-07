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

import json
import os
import time
import skimage
import numpy as np

from mrcnn.utils import Dataset

from sd_maskrcnn.data_augmentation import *
from mrcnn import model as modellib, visualize, utils


"""
TargetDataset creates a Matterport dataset for a directory of
target images, used for target detection branch training.
Directory structure must be as follows:
$base_path/
    target.json
    images/ (target images here)
        ...
    piles/ (color pile images here)
        ...
    masks/ (masks corresponding to piles)

target.json must contain a list of tuples with the following format:
(target_path, pile_path, target_index)

In this class, image_id does not map one-to-one to different piles, rather
to different pile/target pairs.
"""

class TargetDataset(utils.Dataset):
    def __init__(self, config, base_path, images='piles', masks='masks', targets='images', augment_targets=False):
        # omg fix this above before u get confused!!!
        assert base_path != "", "You must provide the path to a dataset!"
        self.targets = targets
        self.images = images
        self.masks = masks
        self.base_path = base_path
        self.data_tuples = None
        super().__init__(config)

    def load(self, imset=None):
        self.add_class('clutter', 1, 'fg')
        import json
        self.data_tuples = json.load(open(os.path.join(self.base_path, 'target.json')))

        # Provide optional index file. NOTE: This operates on the JSON files!
        if imset:
            imset = os.path.join(self.base_path, imset)
            indices = np.load(imset)
        else:
            indices = list(range(len(self.data_tuples)))

        for i in indices:
            pile_path = os.path.join(self.base_path, self.images,
                                     self.data_tuples[i][1])
            mask_path = os.path.join(self.base_path, self.masks,
                                     self.data_tuples[i][1])
            target_path = os.path.join(self.base_path, self.targets,
                                       self.data_tuples[i][0])
            target_ind = int(self.data_tuples[i][2]) - 1
            self.add_example(source='clutter', image_id=i, pile_path=pile_path,
                           pile_mask_path=mask_path, target_path=target_path,
                           target_index=target_ind)

    def load_example(self, example_id):
        """Returns a dictionary containing inputs from a training example."""
        info = self.example_info[example_id]
        example = {}
        example['pile_image'] = self._load_image(info['pile_path'])
        example['target_image'] = self._load_image(info['target_path'])
        example['pile_mask'], example['class_ids'] = self._load_mask(info['pile_mask_path'])
        example['target_index'] = info['target_index']

        return example


"""
Same directory organization as TargetDataset, but one example JSON tuple is now:
((target_path_1, target_path_2, ...), pile_image, target_index)
"""
class TargetStackDataset(utils.Dataset):
    def __init__(self, config, base_path, tuple_file, images='piles', masks='masks', targets='images', augment_targets=False):
        # omg fix this above before u get confused!!!
        assert base_path != "", "You must provide the path to a dataset!"
        self.targets = targets
        self.images = images
        self.masks = masks
        self.base_path = base_path
        self.target_stack_size = config['model']['settings']['stack_size']
        self.bg_pixel = config['model']['settings']['bg_pixel']

        self.augment_targets = augment_targets
        self.data_tuples = json.load(open(os.path.join(self.base_path, tuple_file)))
        super().__init__(config)

        assert len(self.bg_pixel) == self._channels, "background pixel must match # of channels"
        self.bg = np.stack(
            [np.stack([np.array(self.bg_pixel)] * 512)] * 512)

    def load(self, imset=None):
        self.add_class('clutter', 1, 'fg')

        # Provide optional index file. NOTE: This operates on the JSON files!
        if imset:
            imset = os.path.join(self.base_path, imset)
            indices = np.load(imset)
        else:
            indices = list(range(len(self.data_tuples)))

        for i in indices:
            pile_path = os.path.join(self.base_path, self.images,
                                     self.data_tuples[i][1])

            mask_path = os.path.join(self.base_path, self.masks,
                                     self.data_tuples[i][1])

            if not isinstance(self.data_tuples[i][0], list):
                print("Singleton target detected.")
                self.data_tuples[i][0] = [self.data_tuples[i][0]]

            assert len(self.data_tuples[i][0]) == self.target_stack_size, \
            "assert self.bg_pixel.shape == Expected {} target images to stack, but instead found {}.".format(
                self.target_stack_size, len(self.data_tuples[i][0]))
            target_stack_paths = [os.path.join(self.base_path, self.targets, path) for path in self.data_tuples[i][0]]
            target_ind = int(self.data_tuples[i][2]) - 1

            if 'numpy' in self.images:
                pile_path = pile_path.replace('.png', '.npy')
                target_stack_paths = [path.replace('.png', '.npy') for path in target_stack_paths]

            self.add_example(source='clutter', image_id=i, pile_path=pile_path,
                           pile_mask_path=mask_path, target_stack_paths=target_stack_paths,
                           target_index=target_ind)

    def load_example(self, example_id):
        """Returns a dictionary containing inputs from a training example."""
        info = self.example_info[example_id]
        example = {}
        example['target_images'] = []
        for path in info['target_stack_paths']:
            im = self._load_image(path)
            example['target_images'].append(im)
        example['pile_image'] = self._load_image(info['pile_path'])
        example['pile_mask'], example['class_ids'] = self._load_mask(info['pile_mask_path'])
        example['target_index'] = info['target_index']
        return example

    def _get_target_bb(self, target_image):
        target_mask = np.sum(target_image - self.bg_pixel, axis=2)
        bb = utils.extract_bboxes(
            target_mask.reshape((target_image.shape[0], target_image.shape[1], 1)))[0]
        return bb

    def _rotate(self, target_image, rotation):
        im_h, im_w = target_image.shape[:2]
        y1, x1, y2, x2 = self._get_target_bb(target_image)
        crop = target_image[y1-1:y2+1,x1-1:x2+1,:] # pad by one so nearest mode doesn't take colored pixels
        rotated_crop = scipy.ndimage.rotate(crop, rotation, mode='nearest')
        crop_h, crop_w = rotated_crop.shape
        insert_y, insert_x = (im_h - crop_h) // 2, (im_w - crop_w) // 2

        rotated_target = np.copy(self.bg)
        rotated_target[insert_y:(insert_y + crop_h) , insert_x:(insert_x + crop_w), :] = rotated_crop
        return rotated_target

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
        super().__init__(config)

    def load(self, indices_file, augment=False):
        # Load the indices for imset.
        split_file = os.path.join(self.base_path, '{:s}'.format(indices_file))
        self.image_id = np.load(split_file)
        self.add_class('clutter', 1, 'fg')

        flips = [1, 2, 3]
        for i in self.image_id:
            if 'numpy' in self.images:
                path = os.path.join(self.base_path, self.images,
                                'image_{:06d}.npy'.format(i))
                mask_path = os.path.join(self.base_path, self.masks,
                                'image_{:06d}.npy'.format(i))
            else:
                path = os.path.join(self.base_path, self.images,
                                'image_{:06d}.png'.format(i))
                mask_path = os.path.join(self.base_path, self.masks,
                                'image_{:06d}.png'.format(i))

            self.add_example(source='clutter', image_id=i, pile_path=path, pile_mask_path=mask_path)

            # if augment:
    #             for flip in flips:
    #                 self.add_image('clutter', image_id=i, path=p, flip=flip)

    # def flip(self, image, flip):
    #     # flips during training for augmentation
    #     if flip == 1:
    #         image = image[::-1,:,:]
    #     elif flip == 2:
    #         image = image[:,::-1,:]
    #     elif flip == 3:
    #         image = image[::-1,::-1,:]
    #     return image

    def load_example(self, example_id):
        """Returns a dictionary containing inputs from a training example."""
        info = self.example_info[example_id]
        example = {}
        example['pile_image'] = self._load_image(info['pile_path'])
        example['pile_mask'], example['class_ids'] = self._load_mask(info['pile_mask_path'])
        return example

    # def load_image(self, image_id):
    #     # loads image from path
    #     if 'numpy' in self.images:
    #         image = np.load(self.image_info[image_id]['path']).squeeze()
    #     else:
    #         image = skimage.io.imread(self.image_info[image_id]['path'])

    #     if self._channels < 4 and image.shape[-1] == 4 and image.ndim == 3:
    #         image = image[...,:3]
    #     if self._channels == 1 and image.ndim == 2:
    #         image = image[:,:,np.newaxis]
    #     elif self._channels == 1 and image.ndim == 3:
    #         image = image[:,:,0,np.newaxis]
    #     elif self._channels == 3 and image.ndim == 3 and image.shape[-1] == 1:
    #         image = skimage.color.gray2rgb(image)
    #     elif self._channels == 4 and image.shape[-1] == 3:
    #         concat_image = np.concatenate([image, image[:,:,0:1]], axis=2)
    #         assert concat_image.shape == (image.shape[0], image.shape[1], image.shape[2] + 1), concat_image.shape
    #         image = concat_image

    #     return image

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
