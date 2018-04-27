import os
import sys
import logging
import argparse
import configparser
from tqdm import tqdm
from ast import literal_eval
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import cv2
import skimage.io
import numpy as np
import tensorflow as tf

from eval_coco import *
from augmentation import augment_img

from pipeline_utils import *
from clutter import ClutterConfig
import model as modellib, visualize, utils
from real_dataset import RealImageDataset, prepare_real_image_test

def augment_data(config):
    """
    Using provided image directory and output directory, perform data
    augmentation methods on each image and save the new copy to the
    output directory.
    """
    img_dir = config["img_dir"]
    out_dir = config["out_dir"]

    mkdir_if_missing(out_dir)

    print("Augmenting data in directory {}.\n".format(img_dir))
    num_imgs = int(config["num_imgs"])
    count = 0
    for img_file in tqdm(os.listdir(img_dir), total=num_imgs):
        if count == num_imgs:
            break
        if img_file.endswith(".png"):
            # read in image
            img_path = os.path.join(img_dir, img_file)
            # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = skimage.io.imread(img_path, as_grey=True)

            # return list of augmented images and save
            new_img = augment_img(img, config)
            out_path = os.path.join(out_dir, img_file)
            # cv2.imwrite(out_path, new_img)
            skimage.io.imsave(out_path, new_img)
        count += 1

    print("Augmentation complete; files saved in {}.\n".format(out_dir))


def train(config):
    # Load the datasets, configs.
    train_config = ClutterConfig(mean=config["mean_pixel"])
    config.display()

    # Training dataset
    dataset_train = ClutterDataset()
    dataset_train.load('train', config["img_type"], 0)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ClutterDataset()
    dataset_val.load('test', config["img_type"], 0)
    dataset_val.prepare()

    # Create the model.
    model = get_model(config, train_config)

    model.train(dataset_train, dataset_val, learning_rate=train_config.LEARNING_RATE,
                epochs=100, layers='all')
    model_path = os.path.join(model_dir, "mask_rcnn_clutter.h5")
    model.keras_model.save_weights(model_path)


def benchmark(config):
    print("Benchmarking model.")
    # Create new directory for run outputs
    output_dir = config['output_dir'] # In what location should we put this new directory?
    run_name = config['run_name'] # What is it called
    mkdir_if_missing(os.path.join(output_dir, run_name))

    # Create subdirectories for masks and visuals
    pred_dir = os.path.join(output_dir, run_name, 'pred')
    vis_dir = os.path.join(output_dir, run_name, 'vis')
    mkdir_if_missing(pred_dir)
    mkdir_if_missing(vis_dir)

    # Save config
    # TODO: actually do it

    model_path = config['model_path']
    test_dir = config['test_dir'] # directory of test images and segmasks

    inference_config, model, dataset_real = prepare_real_image_test(model_path, test_dir)

    # Feed images into model one by one. For each image, predict, save, visualize?
    image_ids = dataset_real.image_ids

    def get_ax(rows=1, cols=1, size=8):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

    # create directory in which we save transformed GT segmasks
    resized_segmask_dir = os.path.join(test_dir, 'modal_segmasks_processed')
    mkdir_if_missing(resized_segmask_dir)

    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
          modellib.load_image_gt(dataset_real, inference_config, image_id,
            use_mini_mask=False)

        # Save copy of transformed GT segmasks to disk in preparation for annotations
        mask_name = 'image_{:06d}'.format(image_id)
        mask_path = os.path.join(resized_segmask_dir, mask_name)
        print(mask_path)

        molded_images = modellib.mold_image(image, inference_config)
        molded_images = np.expand_dims(molded_images, 0)

        print('molded_images.shape', molded_images.shape)
        print('gt_mask.shape', gt_mask.shape)
        # save the transpose so it's (n, h, w) instead of (h, w, n)
        np.save(mask_path, gt_mask.transpose(2, 0, 1))
        # gt_stat, stat_name = compute_gt_stats(gt_bbox, gt_mask)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        save_masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])])
        print('save_masks.shape', save_masks.shape)
        save_masks_path = os.path.join(pred_dir, str(image_id) + '.npy')
        np.save(save_masks_path, save_masks)
        print(save_masks_path)

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['bg', 'obj'], r['scores'], ax=get_ax())
        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Generate prediction annotations
    encode_gt(resized_segmask_dir)
    encode_predictions(pred_dir)

    coco_benchmark(os.path.join(resized_segmask_dir, 'annos_gt.json'), os.path.join(pred_dir, 'annos_pred.json'))

    print("Saved benchmarking output to {}.\n".format(config["output_dir"]))


def read_config():
    # setting up flag parsing
    conf_parser = argparse.ArgumentParser(description="Augment data in path folder with various noise filters and transformations")

    # required argument for config file
    conf_parser.add_argument("--config", action="store", required=True,
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    conf = configparser.ConfigParser()
    conf.read([conf_args.conf_file])
    task = conf.get("GENERAL", "task").upper()
    task = literal_eval(task)

    # create a dictionary of the proper arguments, including
    # the requested task
    conf_dict = dict(conf.items(task))

    # return a type-sensitive version of the dictionary;
    # prevents further need to cast from string to other types
    out = {}
    for key, value in conf_dict.items():
        out[key] = literal_eval(value)
    out["task"] = task
    return out


if __name__ == "__main__":
    # parse the provided configuration file
    config = read_config()

    task = config["task"]
    print('config["task"]', config['task'])
    if task == "AUGMENT":
        augment_data(config)

    if task == "TRAIN":
        train(config)

    if task == "BENCHMARK":
        benchmark(config)
