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

Benchmark Usage Notes:

Please edit "cfg/benchmark.yaml" to specify the necessary parameters for that task.

Run this file with the tag --config [config file name] if different config from the default location (cfg/benchmark.yaml).

Here is an example run command (GPU selection included):
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py --config cfg/benchmark.yaml
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import skimage.io as io
from copy import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from autolab_core import YamlConfig

from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig
from sd_maskrcnn.dataset import ImageDataset
from sd_maskrcnn.coco_benchmark import coco_benchmark
from sd_maskrcnn.supplement_benchmark import s_benchmark

from mrcnn import model as modellib, utils as utilslib, visualize

def benchmark(config):
    """Benchmarks a model, computes and stores model predictions and then
    evaluates them on COCO metrics and supplementary benchmarking script."""

    print("Benchmarking model.")

    # Create new directory for outputs
    output_dir = config['output_dir']
    utils.mkdir_if_missing(output_dir)

    # Save config in output directory
    config.save(os.path.join(output_dir, config['save_conf_name']))
    image_shape = config['model']['settings']['image_shape']
    config['model']['settings']['image_min_dim'] = min(image_shape)
    config['model']['settings']['image_max_dim'] = max(image_shape)
    config['model']['settings']['gpu_count'] = 1
    config['model']['settings']['images_per_gpu'] = 1
    inference_config = MaskConfig(config['model']['settings'])

    model_dir, _ = os.path.split(config['model']['path'])
    model = modellib.MaskRCNN(mode=config['model']['mode'], config=inference_config,
                              model_dir=model_dir)

    # Load trained weights
    print("Loading weights from ", config['model']['path'])
    model.load_weights(config['model']['path'], by_name=True)

    # Create dataset
    test_dataset = ImageDataset(config)
    test_dataset.load(config['dataset']['indices'])
    test_dataset.prepare()

    vis_config = copy(config)
    vis_config['dataset']['images'] = 'depth_ims'
    vis_config['dataset']['masks'] = 'modal_segmasks'
    vis_dataset = ImageDataset(config)
    vis_dataset.load(config['dataset']['indices'])
    vis_dataset.prepare()

    ######## BENCHMARK JUST CREATES THE RUN DIRECTORY ########
    # code that actually produces outputs should be plug-and-play
    # depending on what kind of benchmark function we run.

    # If we want to remove bin pixels, pass in the directory with
    # those masks.
    if config['mask']['remove_bin_pixels']:
        bin_mask_dir = os.path.join(config['dataset']['path'], config['mask']['bin_masks'])
        overlap_thresh = config['mask']['overlap_thresh']
    else:
        bin_mask_dir = False
        overlap_thresh = 0

    # Create predictions and record where everything gets stored.
    pred_mask_dir, pred_info_dir, gt_mask_dir = \
        detect(config['output_dir'], inference_config, model, test_dataset, bin_mask_dir, overlap_thresh)

    ap, ar = coco_benchmark(pred_mask_dir, pred_info_dir, gt_mask_dir)
    if config['vis']['predictions']:
        visualize_predictions(config['output_dir'], vis_dataset, inference_config, pred_mask_dir, pred_info_dir,
                              show_bbox=config['vis']['show_bbox_pred'], show_scores=config['vis']['show_scores_pred'], show_class=config['vis']['show_class_pred'])
    if config['vis']['ground_truth']:
        visualize_gts(config['output_dir'], vis_dataset, inference_config, show_scores=False, show_bbox=config['vis']['show_bbox_gt'], show_class=config['vis']['show_class_gt'])
    if config['vis']['s_bench']:
        s_benchmark(config['output_dir'], vis_dataset, inference_config, pred_mask_dir, pred_info_dir)

    print("Saved benchmarking output to {}.\n".format(config['output_dir']))
    return ap, ar

def detect(run_dir, inference_config, model, dataset, bin_mask_dir=False, overlap_thresh=0.5):
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
    utils.mkdir_if_missing(pred_dir)

    # Create subdirectory for prediction scores & bboxes
    pred_info_dir = os.path.join(run_dir, 'pred_info')
    utils.mkdir_if_missing(pred_info_dir)

    # Create subdirectory for transformed GT segmasks
    resized_segmask_dir = os.path.join(run_dir, 'modal_segmasks_processed')
    utils.mkdir_if_missing(resized_segmask_dir)

    # Feed images into model one by one. For each image, predict and save.
    image_ids = dataset.example_indices
    indices = dataset.indices
    times = []
    print('MAKING PREDICTIONS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, _, _, gt_mask =\
          modellib.load_image_gt(dataset, inference_config, image_id,
            use_mini_mask=False)

        # Run object detection
        results = model.detect(verbose=1, images=[image])
        r = results[0]
        times.append(r['time'])
        print('rois', r['rois'])

        # If we choose to mask out bin pixels, load the bin masks and
        # transform them properly.
        # Then, delete the mask, score, class id, and bbox corresponding
        # to each mask that is entirely bin pixels.
        if bin_mask_dir:
            name = 'image_{:06d}.png'.format(indices[image_id])
            bin_mask = io.imread(os.path.join(bin_mask_dir, name))[:,:,np.newaxis]
            bin_mask, _, _, _, _ = utilslib.resize_image(
                bin_mask,
                max_dim=inference_config.IMAGE_MAX_DIM,
                min_dim=inference_config.IMAGE_MIN_DIM,
                mode=inference_config.IMAGE_RESIZE_MODE
            )

            bin_mask = bin_mask.squeeze()
            deleted_masks = [] # which segmasks are gonna be tossed?
            num_detects = r['masks'].shape[2]
            for k in range(num_detects):
                # compute the area of the overlap.
                inter = np.logical_and(bin_mask, r['masks'][:,:,k])
                frac_overlap =  np.sum(inter) / np.sum(r['masks'][:,:,k])
                if frac_overlap <= overlap_thresh:
                    deleted_masks.append(k)

            r['masks'] = [r['masks'][:,:,k] for k in range(num_detects) if k not in deleted_masks]
            r['masks'] = np.stack(r['masks'], axis=2) if r['masks'] else np.array([])
            r['rois'] = [r['rois'][k,:] for k in range(num_detects) if k not in deleted_masks]
            r['rois'] = np.stack(r['rois'], axis=0) if r['rois'] else np.array([])
            r['class_ids'] = np.array([r['class_ids'][k] for k in range(num_detects)
                                       if k not in deleted_masks])
            r['scores'] = np.array([r['scores'][k] for k in range(num_detects)
                                       if k not in deleted_masks])

        # Save copy of transformed GT segmasks to disk in preparation for annotations
        mask_name = 'image_{:06d}'.format(image_id)
        mask_path = os.path.join(resized_segmask_dir, mask_name)

        # save the transpose so it's (n, h, w) instead of (h, w, n)
        np.save(mask_path, gt_mask.transpose(2, 0, 1))

        # Save masks
        save_masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])]) if np.any(r['masks']) else np.array([])
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
    print('Took {} s'.format(sum(times)))
    print('Saved prediction masks to:\t {}'.format(pred_dir))
    print('Saved prediction info (bboxes, scores, classes) to:\t {}'.format(pred_info_dir))
    print('Saved transformed GT segmasks to:\t {}'.format(resized_segmask_dir))

    return pred_dir, pred_info_dir, resized_segmask_dir

def visualize_predictions(run_dir, dataset, inference_config, pred_mask_dir, pred_info_dir, show_bbox=True, show_scores=True, show_class=True):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, 'vis')
    utils.mkdir_if_missing(vis_dir)

    # Feed images into model one by one. For each image visualize predictions
    image_ids = dataset.example_indices

    print('VISUALIZING PREDICTIONS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, _, _, _ = modellib.load_image_gt(dataset, inference_config, image_id, use_mini_mask=False)
        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'image_{:06}.npy'.format(image_id)))
        # Must transpose from (n, h, w) to (h, w, n)
        r['masks'] = np.transpose(r_masks, (1, 2, 0))
        # Visualize
        scores = r['scores'] if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.,0.,1.,1.])
        fig.add_axes(ax)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['bg', 'obj'],
                                    ax=ax, scores=scores, show_bbox=show_bbox, show_class=show_class)
        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()

def visualize_gts(run_dir, dataset, inference_config, show_bbox=True, show_scores=False, show_class=True):
    """Visualizes gts."""
    # Create subdirectory for gt visualizations
    vis_dir = os.path.join(run_dir, 'gt_vis')
    utils.mkdir_if_missing(vis_dir)

    # Feed images one by one
    image_ids = dataset.image_ids

    print('VISUALIZING GROUND TRUTHS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config, image_id,
                                                                        use_mini_mask=False)

        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # Visualize
        scores = np.ones(gt_class_id.size) if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.,0.,1.,1.])
        fig.add_axes(ax)
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, ['bg', 'obj'],
                                    scores, ax=ax, show_bbox=show_bbox, show_class=show_class)
        file_name = os.path.join(vis_dir, 'gt_vis_{:06d}'.format(image_id))
        height, width = image.shape[:2]
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark SD Mask RCNN model")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    # Set up tf session to use what GPU mem it needs and benchmark
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        set_session(sess)
        benchmark(config)
