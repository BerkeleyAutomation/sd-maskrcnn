"""
Copyright ©2019. The Regents of the University of California (Regents). All Rights Reserved.
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

import argparse
import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from autolab_core import YamlConfig
from keras.backend.tensorflow_backend import set_session
from mrcnn import model as modellib
from mrcnn import visualize
from tqdm import tqdm

from sd_maskrcnn import utils
from sd_maskrcnn.coco_benchmark import coco_benchmark
from sd_maskrcnn.dataset import ImageDataset
from sd_maskrcnn.model import SDMaskRCNNModel
from sd_maskrcnn.supplement_benchmark import s_benchmark


def benchmark(config):
    """Benchmarks a model, computes and stores model predictions and then
    evaluates them on COCO metrics and supplementary benchmarking script."""

    print("Benchmarking model.")

    # Create new directory for outputs
    output_dir = config["output_dir"]
    utils.mkdir_if_missing(output_dir)

    # Save config in output directory
    config.save(os.path.join(output_dir, config["save_conf_name"]))
    model = SDMaskRCNNModel("inference", config["model"])

    # Create dataset
    test_dataset = ImageDataset(config)
    test_dataset.load(config["dataset"]["indices"])
    test_dataset.prepare()

    vis_config = copy(config)
    vis_config["dataset"]["images"] = "depth_ims"
    vis_config["dataset"]["masks"] = "modal_segmasks"
    vis_dataset = ImageDataset(config)
    vis_dataset.load(config["dataset"]["indices"])
    vis_dataset.prepare()

    # Overarching benchmark function just creates the directory
    # code that actually produces outputs should be plug-and-play
    # depending on what kind of benchmark function we run.

    # If we want to remove bin pixels, pass in the directory with
    # those masks.
    if config["mask"]["remove_bin_pixels"]:
        bin_mask_dir = os.path.join(
            config["dataset"]["path"], config["mask"]["bin_masks"]
        )
        overlap_thresh = config["mask"]["overlap_thresh"]
    else:
        bin_mask_dir = None
        overlap_thresh = 0

    # Create predictions and record where everything gets stored.
    pred_mask_dir, pred_info_dir, gt_mask_dir = model.detect_dataset(
        config["output_dir"],
        test_dataset,
        bin_mask_dir,
        overlap_thresh,
    )

    ap, ar = coco_benchmark(pred_mask_dir, pred_info_dir, gt_mask_dir)
    if config["vis"]["predictions"]:
        visualize_predictions(
            config["output_dir"],
            vis_dataset,
            model.mask_config,
            pred_mask_dir,
            pred_info_dir,
            show_bbox=config["vis"]["show_bbox_pred"],
            show_scores=config["vis"]["show_scores_pred"],
            show_class=config["vis"]["show_class_pred"],
        )
    if config["vis"]["ground_truth"]:
        visualize_gts(
            config["output_dir"],
            vis_dataset,
            model.mask_config,
            show_scores=False,
            show_bbox=config["vis"]["show_bbox_gt"],
            show_class=config["vis"]["show_class_gt"],
        )
    if config["vis"]["s_bench"]:
        s_benchmark(
            config["output_dir"],
            vis_dataset,
            model.mask_config,
            pred_mask_dir,
            pred_info_dir,
        )

    print("Saved benchmarking output to {}.\n".format(config["output_dir"]))
    return ap, ar


def visualize_predictions(
    run_dir,
    dataset,
    inference_config,
    pred_mask_dir,
    pred_info_dir,
    show_bbox=True,
    show_scores=True,
    show_class=True,
):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, "vis")
    utils.mkdir_if_missing(vis_dir)

    # Feed images into model one by one. For each image visualize predictions
    image_ids = dataset.image_ids

    print("VISUALIZING PREDICTIONS")
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, _, _, _ = modellib.load_image_gt(
            dataset, inference_config, image_id, use_mini_mask=False
        )
        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)
        elif inference_config.IMAGE_CHANNEL_COUNT > 3:
            image = image[:, :, :3]

        # load mask and info
        r = np.load(
            os.path.join(pred_info_dir, "image_{:06}.npy".format(image_id)),
            allow_pickle=True,
        ).item()
        r_masks = np.load(
            os.path.join(pred_mask_dir, "image_{:06}.npy".format(image_id))
        )
        # Must transpose from (n, h, w) to (h, w, n)
        if r_masks.ndim == 3:
            r["masks"] = np.transpose(r_masks, (1, 2, 0))
        else:
            r["masks"] = r_masks
        # Visualize
        scores = r["scores"] if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        visualize.display_instances(
            image,
            r["rois"],
            r["masks"],
            r["class_ids"],
            ["bg", "obj"],
            ax=ax,
            scores=scores,
            show_bbox=show_bbox,
            show_class=show_class,
        )
        file_name = os.path.join(vis_dir, "vis_{:06d}".format(image_id))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()


def visualize_gts(
    run_dir,
    dataset,
    inference_config,
    show_bbox=True,
    show_scores=False,
    show_class=True,
):
    """Visualizes gts."""
    # Create subdirectory for gt visualizations
    vis_dir = os.path.join(run_dir, "gt_vis")
    utils.mkdir_if_missing(vis_dir)

    # Feed images one by one
    image_ids = dataset.image_ids

    print("VISUALIZING GROUND TRUTHS")
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, inference_config, image_id, use_mini_mask=False
        )

        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # Visualize
        scores = np.ones(gt_class_id.size) if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        visualize.display_instances(
            image,
            gt_bbox,
            gt_mask,
            gt_class_id,
            ["bg", "obj"],
            scores,
            ax=ax,
            show_bbox=show_bbox,
            show_class=show_class,
        )
        file_name = os.path.join(vis_dir, "gt_vis_{:06d}".format(image_id))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()


if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(
        description="Benchmark SD Mask RCNN model"
    )
    conf_parser.add_argument(
        "--config",
        action="store",
        default="cfg/benchmark.yaml",
        dest="conf_file",
        type=str,
        help="path to the configuration file",
    )
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    # Set up tf session to use what GPU mem it needs and benchmark
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=tf_config) as sess:
        set_session(sess)
        benchmark(config)
