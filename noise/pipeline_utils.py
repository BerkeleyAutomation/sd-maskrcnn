import os
import json
from tqdm import tqdm
from ast import literal_eval
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask

import model as modellib, visualize, utils
from datasets import RealImageDataset, prepare_real_image_test

def mkdir_if_missing(output_dir):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            print("Something went wrong in mkdir_if_missing. "
                  "Probably some other process created the directory already.")


def get_model_and_path(config, clutter_config):
    mode = config["model_mode"]
    model_folder = config["model_weights_folder"]
    base_path = config["base_path"]
    model_path = os.path.join(base_path, model_folder)
    print(model_path)
    mkdir_if_missing(model_path)
    model = modellib.MaskRCNN(mode=mode, config=clutter_config,
                              model_dir=model_path)
    return model, model_path


def get_conf_dict(config_obj):
    # create a dictionary of the proper arguments, including
    # the requested task
    task = config_obj.get("GENERAL", "task").upper()
    task = literal_eval(task)
    conf_dict = dict(config_obj.items(task))

    # return a type-sensitive version of the dictionary;
    # prevents further need to cast from string to other types
    out = {}
    for key, value in conf_dict.items():
        out[key] = literal_eval(value)

    out["task"] = task

    return out

def scale_to_square(im, dim=512):
    """Resizes an image to a square image of length dim."""
    scale = 512.0 / max(im.shape[0:2]) # scale so min dimension is 512
    scale_dim = tuple(reversed([int(np.ceil(d * scale)) for d in im.shape[:2]]))
    im = cv2.resize(im, scale_dim, interpolation=cv2.INTER_NEAREST)

    return im

def detect(run_dir, inference_config, model, dataset_real, bin_mask_dir=False,
           overlap_thresh=0.5):
    """
    Given a run directory, a MaskRCNN config object, a MaskRCNN model object,
    and a Dataset object,
    - Loads and processes ground-truth masks, saving them to a new directory
      for annotation
    - Makes predictions on images
    - Saves prediction masks in a certain directory
    - Saves other prediction info (scores, bboxes) in a separate directory

    Returns paths to directories for prediction masks, prediction info, and
    modified GT masks.

    If bin_mask_dir is specified, then we will be checking predictions against
    the "bin-vs-no bin" mask for the test case.
    For each predicted instance, if less than overlap_thresh of the mask actually
    consists of non-bin pixels, we will toss out the mask.
    """

    # Create subdirectory for prediction masks
    pred_dir = os.path.join(run_dir, 'pred_masks')
    mkdir_if_missing(pred_dir)

    # Create subdirectory for prediction scores & bboxes
    pred_info_dir = os.path.join(run_dir, 'pred_info')
    mkdir_if_missing(pred_info_dir)

    # Create subdirectory for transformed GT segmasks
    resized_segmask_dir = os.path.join(run_dir, 'modal_segmasks_processed')
    mkdir_if_missing(resized_segmask_dir)

    # Feed images into model one by one. For each image, predict, save, visualize?
    image_ids = dataset_real.image_ids
    indices = dataset_real.indices
    print('MAKING PREDICTIONS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
          modellib.load_image_gt(dataset_real, inference_config, image_id,
            use_mini_mask=False)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        # If we choose to mask out bin pixels, load the bin masks and
        # transform them properly.
        # Then, delete the mask, score, class id, and bbox corresponding
        # to each mask that is entirely bin pixels.
        if bin_mask_dir:
            name = 'image_{:06d}.png'.format(indices[image_id])
            bin_mask = io.imread(os.path.join(bin_mask_dir, name))
            # HACK: stack bin_mask 3x
            bin_mask = np.stack((bin_mask, bin_mask, bin_mask), axis=2)
            bin_mask, window, scale, padding = utils.resize_image(
                bin_mask,
                max_dim=inference_config.IMAGE_MAX_DIM,
                min_dim=inference_config.IMAGE_MIN_DIM,
                padding=inference_config.IMAGE_PADDING,
                interp='nearest'
            )

            bin_mask = bin_mask[:,:,0]
            deleted_masks = [] # which segmasks are gonna be tossed?
            num_detects = r['masks'].shape[2]
            for k in range(num_detects):
                # compute the area of the overlap.
                inter = np.logical_and(bin_mask, r['masks'][:,:,k])
                frac_overlap =  np.sum(inter) / np.sum(r['masks'][:,:,k])
                if frac_overlap <= overlap_thresh:
                    deleted_masks.append(k)

            r['masks'] = np.stack([r['masks'][:,:,k] for k in range(num_detects)
                                   if k not in deleted_masks], axis=2)
            r['rois'] = np.stack([r['rois'][k,:] for k in range(num_detects)
                                  if k not in deleted_masks], axis=0)
            r['class_ids'] = np.array([r['class_ids'][k] for k in range(num_detects)
                                       if k not in deleted_masks])
            r['scores'] = np.array([r['scores'][k] for k in range(num_detects)
                                       if k not in deleted_masks])

        # Save copy of transformed GT segmasks to disk in preparation for annotations
        mask_name = 'image_{:06d}'.format(image_id)
        mask_path = os.path.join(resized_segmask_dir, mask_name)

        molded_images = modellib.mold_image(image, inference_config)
        molded_images = np.expand_dims(molded_images, 0)

        # save the transpose so it's (n, h, w) instead of (h, w, n)
        np.save(mask_path, gt_mask.transpose(2, 0, 1))

        # Save masks
        save_masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])])
        save_masks_path = os.path.join(pred_dir, 'image_{:06d}.npy'.format(image_id))
        np.save(save_masks_path, save_masks)

        # Save info
        r_info = {
            'rois': r['rois'],
            'scores': r['scores'],
            'class_ids': r['class_ids']
        }
        r_info_path = os.path.join(pred_info_dir, 'image_{:06d}.npy'.format(image_id))
        np.save(r_info_path, r_info)

    print('Saved prediction masks to:\t {}'.format(pred_dir))
    print('Saved prediction info (bboxes, scores, classes) to:\t {}'.format(pred_info_dir))
    print('Saved transformed GT segmasks to:\t {}'.format(resized_segmask_dir))

    return pred_dir, pred_info_dir, resized_segmask_dir


def visualize_predictions(run_dir, dataset_real, inference_config, pred_mask_dir, pred_info_dir, show_bbox=True, show_class=True):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, 'vis')
    mkdir_if_missing(vis_dir)

    # Feed images into model one by one. For each image, predict, save, visualize?
    image_ids = dataset_real.image_ids

    print('VISUALIZING PREDICTIONS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
          modellib.load_image_gt(dataset_real, inference_config, image_id,
            use_mini_mask=False)

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'image_{:06}.npy'.format(image_id)))
        # Must transpose from (n, h, w) to (h, w, n)
        r['masks'] = np.transpose(r_masks, (1, 2, 0))      
        # Visualize
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['bg', 'obj'], r['scores'], show_bbox=show_bbox, show_class=show_class)
        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

def visualize_gts(run_dir, dataset_real, inference_config, show_scores=False, show_bbox=True, show_class=True):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, 'gt_vis')
    mkdir_if_missing(vis_dir)

    # Feed images into model one by one. For each image, predict, save, visualize?
    image_ids = dataset_real.image_ids

    print('VISUALIZING GROUND TRUTHS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
          modellib.load_image_gt(dataset_real, inference_config, image_id,
            use_mini_mask=False)

        # Visualize
        scores = np.ones(gt_class_id.size) if show_scores else None
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                                    ['bg', 'obj'], scores, show_bbox=show_bbox, show_class=show_class)
        file_name = os.path.join(vis_dir, 'gt_vis_{:06d}'.format(image_id))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()
