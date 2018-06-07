"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np, cv2, os
from config import Config
# from maskrcnn.config import Config
from tqdm import tqdm
# import maskrcnn.utils as utils
import utils
import logging

# Structure of data from Jeff.
# depth_ims: image_{:06d}.png
# gray_ims: image_{:06d}.png
# occluded_segmasks: image_{:06d}_channel_{:03d}.png
# semantic_segmasks: image_{:06d}
# unoccluded_segmasks: image_{:06d}_channel_{:03d}.png
# splits/fold_{:02d}/{train,test}_indices.npy

base_dir = '/nfs/diskstation/projects/dex-net/segmentation/datasets/noisy_sim_dataset'

class ClutterConfig(Config):
  """Configuration for training on the toy shapes dataset.
  Derives from the base Config class and overrides values specific
  to the toy shapes dataset.
  """
  # Give the configuration a recognizable name
  NAME = "clutter"

  # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
  # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
  GPU_COUNT = 1
  IMAGES_PER_GPU = 4

  # Number of classes (including background)
  NUM_CLASSES = 1 + 1  # background + 3 shapes

  # Use small images for faster training. Set the limits of the small side
  # the large side, and that determines the image shape.
  IMAGE_MIN_DIM = 512
  IMAGE_MAX_DIM = 512

  # Use smaller anchors because our image and objects are small
  #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

  # Reduce training ROIs per image because the images are small and have
  # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
  #TRAIN_ROIS_PER_IMAGE = 32

  # Use a small epoch since the data is simple
  STEPS_PER_EPOCH = 8000/IMAGES_PER_GPU

  # use small validation steps since the epoch is small
  #VALIDATION_STEPS = 50

  #DETECTION_MIN_CONFIDENCE = 0.4

  def __init__(self, mean):
    # Overriding things here.
    super(ClutterConfig, self).__init__()
    self.IMAGE_SHAPE[2] = 3
    self.MEAN_PIXEL = np.array([mean, mean, mean])


class ClutterDataset(utils.Dataset):
  """Generates the shapes synthetic dataset. The dataset consists of simple
  shapes (triangles, squares, circles) placed randomly on a blank surface.
  The images are generated on the fly. No file access required.
  """
  def load(self, imset, typ='depth', fold=0):
    # Load the indices for imset.
    # self.base_path = os.path.join('/nfs/diskstation/projects/dex-net/segmentation/datasets/pile_segmasks_01_28_18')
    self.base_path = os.path.join("/nfs/diskstation/projects/dex-net/segmentation/datasets/noisy_sim_dataset")
    # with folds
    # split_file = os.path.join(self.base_path, 'splits',
    # 'fold_{:02d}'.format(fold), '{:s}_indices.npy'.format(imset))
    split_file = os.path.join(self.base_path, '{:s}_indices.npy'.format(imset))
    self.image_id = np.load(split_file)
    self.image_id = self.image_id[self.image_id < 10000]

    self.add_class('clutter', 1, 'fg')
    flips = [0, 1, 2, 3] if imset == 'train' else [0]

    count = 0

    for i in self.image_id:
      # make sure that i is not too big if incorrect indices are given
      if i > 10000:
        continue
    # p = os.path.join(self.base_path, '{:s}_ims'.format(typ),
      p = os.path.join(self.base_path, '{:s}_ims'.format(typ),
        'image_{:06d}.png'.format(i))
      count += 1

      for flip in flips:
        self.add_image('clutter', image_id=i, path=p, width=600, height=400, flip=flip)

  def flip(self, image, flip):
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
    """Generate an image from the specs of the given image ID.
    Typically this function loads the image from a file, but
    in this case it generates the image on the fly from the
    specs in image_info.
    """
    info = self.image_info[image_id]

    # modify path- depth_ims to depth_ims_resized
    image = cv2.imread(info['path'], cv2.IMREAD_UNCHANGED)
    logging.log(level=1, msg=str(info['path']))
    # image = cv2.imread(info['path'])
    assert(image is not None)
    if image.ndim == 2: image = np.tile(image[:,:,np.newaxis], [1,1,3])
    image = self.flip(image, info['flip'])
    return image

  def image_reference(self, image_id):
    """Return the shapes data of the image."""
    info = self.image_info[image_id]
    if info["source"] == "clutter":
      return info["path"] + "-{:d}".format(info["flip"])
    else:
      super(self.__class__).image_reference(self, image_id)

  def load_mask(self, image_id):
    """Generate instance masks for shapes of the given image ID.
    """
    info = self.image_info[image_id]
    _image_id = info['id']
    Is = []
    file_name = os.path.join(self.base_path, 'modal_segmasks_project_resized',
      'image_{:06d}.png'.format(_image_id))

    all_masks = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    for i in range(25):
      # file_name = os.path.join(self.base_path, 'occluded_segmasks',
      #   'image_{:06d}_channel_{:03d}.png'.format(_image_id, i))
      # I = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) > 0
      I = all_masks == i+1
      if np.any(I):
        I = I[:,:,np.newaxis]
        Is.append(I)
    if len(Is) > 0: mask = np.concatenate(Is, 2)
    else: mask = np.zeros([info['height'], info['width'], 0], dtype=np.bool)
    # Making sure masks are always contiguous.
    # block = np.any(np.any(mask,0),0)
    # assert((not np.any(block)) or (not np.any(block[np.where(block)[0][-1]+1:])))
    # print(block)
    mask = self.flip(mask, info['flip'])
    class_ids = np.array([1 for _ in range(mask.shape[2])])
    return mask, class_ids.astype(np.int32)


# class RealDataset(ClutterDataset):
#   def load(self, imset, typ='depth', fold=0):
#     # Load the indices for imset.
#     # self.base_path = os.path.join('/nfs/diskstation/projects/dex-net/segmentation/datasets/pile_segmasks_01_28_18')
#     self.base_path = os.path.join('/nfs/diskstation/projects/dex-net/segmentation/datasets/segmasks_04_13_18')
#     split_file = os.path.join(self.base_path, 'splits',
#       'real_{:02d}'.format(fold), '{:s}_indices.npy'.format(imset))
#     self.image_id = np.load(split_file)

#     self.add_class('clutter', 1, 'fg')
#     flips = [0]

#     count = 0

#     for i in self.image_id:
#       if i > 10000:
#         continue
#     # p = os.path.join(self.base_path, '{:s}_ims'.format(typ),
#       p = os.path.join(self.base_path, 'noisy_{:s}_ims'.format(typ),
#         'image_{:06d}.png'.format(i))
#       count += 1

#       for flip in flips:
#         self.add_image('clutter', image_id=i, path=p, width=256, height=256, flip=flip)

#   def flip(self, image, flip):
#     if flip == 0:
#       image = image
#     elif flip == 1:
#       image = image[::-1,:,:]
#     elif flip == 2:
#       image = image[:,::-1,:]
#     elif flip == 3:
#       image = image[::-1,::-1,:]
#     return image

#   def load_image(self, image_id):
#     """Generate an image from the specs of the given image ID.
#     Typically this function loads the image from a file, but
#     in this case it generates the image on the fly from the
#     specs in image_info.
#     """
#     info = self.image_info[image_id]

#     # modify path- depth_ims to depth_ims_resized
#     image = cv2.imread(info['path'].replace('depth_ims', 'noisy_depth_ims'), cv2.IMREAD_UNCHANGED)
#     # image = cv2.imread(info['path'])
#     assert(image is not None)
#     if image.ndim == 2: image = np.tile(image[:,:,np.newaxis], [1,1,3])
#     image = self.flip(image, info['flip'])
#     return image

#   def image_reference(self, image_id):
#     """Return the shapes data of the image."""
#     info = self.image_info[image_id]
#     if info["source"] == "clutter":
#       return info["path"] + "-{:d}".format(info["flip"])
#     else:
#       super(self.__class__).image_reference(self, image_id)

#   def load_mask(self, image_id):
#     """Generate instance masks for shapes of the given image ID.
#     """
#     info = self.image_info[image_id]
#     _image_id = info['id']
#     Is = []
#     file_name = os.path.join(self.base_path, 'real_segmasks',
#       'image_{:06d}.png'.format(_image_id))

#     all_masks = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

#     for i in range(25):
#       # file_name = os.path.join(self.base_path, 'occluded_segmasks',
#       #   'image_{:06d}_channel_{:03d}.png'.format(_image_id, i))
#       # I = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) > 0
#       I = all_masks == i+1
#       if np.any(I):
#         I = I[:,:,np.newaxis]
#         Is.append(I)
#     if len(Is) > 0: mask = np.concatenate(Is, 2)
#     else: mask = np.zeros([info['height'], info['width'], 0], dtype=np.bool)
#     # Making sure masks are always contiguous.
#     # block = np.any(np.any(mask,0),0)
#     # assert((not np.any(block)) or (not np.any(block[np.where(block)[0][-1]+1:])))
#     # print(block)
#     mask = self.flip(mask, info['flip'])
#     class_ids = np.array([1 for _ in range(mask.shape[2])])
#     return mask, class_ids.astype(np.int32)

def test_clutter_dataset():
  clutter_dataset = ClutterDataset()
  # clutter_dataset.load('train', 'gray')
  clutter_dataset.load('test', 'depth')
  clutter_dataset.prepare()
  image_ids = clutter_dataset.image_ids
  Is = []
  for i in tqdm(image_ids):
    I = clutter_dataset.load_image(i)
    clutter_dataset.load_mask(i)
    Is.append(I)
  print(np.mean(np.array(Is)))

def concat_segmasks():
  print("CONCATENATING SEGMASKS IN " + base_dir)
  bads = []
  for i in tqdm(range(10000)):
    Is = []
    masks = np.zeros((150, 200), dtype=np.uint8)
    for j in range(21):
      file_name = os.path.join(base_dir, 'modal_segmasks',
        'image_{:06d}_channel_{:03d}.png'.format(i, j))

      im = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
      if im is not None:
        I = im > 0
        masks[I] = j+1
        I = I[:,:,np.newaxis]; Is.append(I)
    Is = np.concatenate(Is, 2)
    Is = Is*1
    file_name = os.path.join(base_dir, 'modal_segmasks_project',
      'image_{:06d}.png'.format(i))
    cv2.imwrite(file_name, masks)
    bads.append(len(np.where(np.sum(Is,2) > 1)[0]))
  print(bads)

def resize_images(max_dim=512):
  """Resizes all images so their maximum dimension is 512. Saves to new directory."""
  print("RESIZING IMAGES")
  image_dirs = ['depth_ims'] # directories of images that need resizing
  mask_dirs = ['modal_segmasks_project']
  resized_image_dirs = [d + '_resized' for d in image_dirs]
  resized_mask_dirs = [d + '_resized' for d in mask_dirs]
  for d in (resized_image_dirs + resized_mask_dirs):
    # create new dirs for resized images
    if not os.path.exists(os.path.join(base_dir, d)):
      os.makedirs(os.path.join(base_dir, d))
  print(mask_dirs, resized_image_dirs, resized_mask_dirs)
  vaughan = 0
  for d, md, r_d, r_md in zip(image_dirs, mask_dirs, resized_image_dirs, resized_mask_dirs):
    old_im_path = os.path.join(base_dir, d)
    new_im_path = os.path.join(base_dir, r_d)
    old_mask_path = os.path.join(base_dir, md)
    new_mask_path = os.path.join(base_dir, r_md)
    for im_path in os.listdir(old_im_path):
      if vaughan % 100 == 0:
          print("Image #{} processed.".format(vaughan))
      vaughan += 1
      im_old_path = os.path.join(old_im_path, im_path)
      try:
        mask_old_path = os.path.join(old_mask_path, im_path)
      except:
        continue
      im = cv2.imread(im_old_path, cv2.IMREAD_UNCHANGED)
      mask = cv2.imread(mask_old_path, cv2.IMREAD_UNCHANGED)
      if mask.shape[0] == 0 or mask.shape[1] == 0:
          print("mask empty")
          continue
      im_box = zero_crop(im, mask)
      im = im[im_box[0] : im_box[2], im_box[1] : im_box[3], :]
      mask = mask[im_box[0] // 4 : im_box[2] // 4, im_box[1] // 4 : im_box[3] // 4]
      im = scale_to_square(im)
      mask = scale_to_square(mask)
      new_im_file = os.path.join(new_im_path, im_path)
      new_mask_file = os.path.join(new_mask_path, im_path)
      cv2.imwrite(new_im_file, im)
      cv2.imwrite(new_mask_file, mask)

def scale_to_square(im, dim=512):
  """Resizes an image to a square image of length dim."""
  scale = 512.0 / min(im.shape[0:2]) # scale so min dimension is 512
  scale_dim = tuple(reversed([int(np.ceil(d * scale)) for d in im.shape[:2]]))
  im = cv2.resize(im, scale_dim, interpolation=cv2.INTER_NEAREST)
  y_margin = abs(im.shape[0] - 512) // 2
  x_margin = abs(im.shape[1] - 512) // 2

  check_y = 512 - (im.shape[0] - y_margin - y_margin)
  check_x = 512 - (im.shape[1] - x_margin - x_margin)

  im = im[y_margin : im.shape[0] - y_margin + check_y, x_margin : im.shape[1] - x_margin + check_x]


  assert im.shape[0] == 512 and im.shape[1] == 512, "shapes messed up " + str(im.shape)

  return im

def zero_crop(im, mask, margin=15):
  """Assuming im and mask are already resized, remove any erroneous zeros while retaining shape"""
  dim = im.shape[0]
  mask = mask.reshape((150, 200, 1))
  boxes = utils.extract_bboxes(mask)
  top = np.min(boxes[:, 0]) * 4
  left = np.min(boxes[:, 1]) * 4
  bot = np.max(boxes[:, 2]) * 4
  right = np.max(boxes[:, 3]) * 4

  # find the location of the farthest points that are non-zero, and set this to be the new bbox
  im_magnitudes = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2]).reshape(im.shape[0:2])
  im_magnitudes[im_magnitudes == 0] = 1000
  new_top = np.max(np.argmin(im_magnitudes, axis=1))

  # don't use left
  top, left, bot, right = min(top, new_top), 0, 599, 799
  return np.array([check_bounds(top, 0, 600 - 1), check_bounds(left, 0, 800 - 1),
                   check_bounds(bot, 0, 600 - 1), check_bounds(right, 0, 800 - 1)])

def check_bounds(x, left, right):
  """If number is out of bounds of a range, cutoff."""
  if x < left:
      return left
  if x > right:
      return right
  return x


if __name__ == '__main__':
  # test_clutter_dataset()
  concat_segmasks()
  # resize_images()
  # test_display_images()
