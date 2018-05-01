import os
import sys
import time
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

from eval_coco import coco_benchmark
from eval_saurabh import s_benchmark
from augmentation import augment_img
from resize import scale_to_square

from pipeline_utils import *
from clutter import ClutterConfig
import model as modellib, visualize, utils
from real_dataset import RealImageDataset, prepare_real_image_test
from sim_dataset import SimImageDataset

"""
Pipeline Usage Notes:

Please edit "config.ini" to specify the task you wish to perform and the
necessary parameters for that task.

Run this file with the tag --config [config file name] (in this case,
config.ini).

You should include in your PYTHONPATH the locations of maskrcnn and clutter
folders.

Here is an example run command (GPU selection included):
CUDA_VISIBLE_DEVICES='0' PYTHONPATH='.:maskrcnn/:clutter/' python3 noise/pipeline.py --config noise/config.ini
"""


def augment_data(conf):
    """
    Using provided image directory and output directory, perform data
    augmentation methods on each image and save the new copy to the
    output directory.
    """
    config = get_conf_dict(conf)

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


def train(conf):
    config = get_conf_dict(conf)

    # read information from config
    dataset_path = config["base_path"]
    mean_pixel = config["mean_pixel"]
    img_type = config["img_type"]

    # Load the datasets, configs.
    train_config = ClutterConfig(mean=mean_pixel)

    # learning rate set
    train_config.LEARNING_RATE = config["learning_rate"]

    # future: override maskrcnn safety checks to allow non power-of-2 shapes
    # img_width = config["img_width"]
    # img_height = config["img_height"]
    # img_nchannels = config["img_nchannels"]
    # img_shape = (img_height, img_width, img_nchannels)
    # train_config.IMAGE_SHAPE = img_shape
    train_config.display()

    # Training datasetx
    dataset_train = SimImageDataset(dataset_path)
    dataset_train.load('train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SimImageDataset(dataset_path)
    dataset_val.load('test')
    dataset_val.prepare()

    # Create the model.
    model, model_path = get_model_and_path(config, train_config)

    # save config in run folder
    save_config(conf, os.path.join(model_path, config["save_conf_name"]))

    # train and save weights to model_path
    model.train(dataset_train, dataset_val, learning_rate=train_config.LEARNING_RATE,
                epochs=50, layers='all')

    # save in the dataset folder
    current_datetime = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_path, "mask_rcnn_clutter_{}.h5".format(current_datetime))
    model.keras_model.save_weights(model_path)


def benchmark(conf):
    config = get_conf_dict(conf)

    print("Benchmarking model.")
    # Create new directory for run outputs
    output_dir = config['output_dir'] # In what location should we put this new directory?
    run_name = config['run_name'] # What is it called
    run_dir = os.path.join(output_dir, run_name)
    mkdir_if_missing(run_dir)

    # Save config
    # TODO: actually do it
    model_path = config['model_path']
    test_dir = config['test_dir'] # directory of test images and segmasks

    inference_config, model, dataset_real = prepare_real_image_test(model_path, test_dir)

    ######## BENCHMARK JUST CREATES THE RUN DIRECTORY ########
    # code that actually produces outputs should be plug-and-play
    # depending on what kind of benchmark function we run.

    coco_benchmark(run_dir, inference_config, model, dataset_real)

    s_benchmark(run_dir, inference_config, model, dataset_real)


    print("Saved benchmarking output to {}.\n".format(run_dir))


def resize_images(conf):
    """Resizes all images so their maximum dimension is 512. Saves to new directory."""
    config = get_conf_dict(conf)
    base_dir = config["base_path"]

    # directories of images that need resizing
    image_dir = config["img_dir"]
    mask_dir = config["mask_dir"]

    # output: resized images
    image_out_dir = config["img_out_dir"]
    mkdir_if_missing(os.path.join(base_dir, image_out_dir))
    mask_out_dir = config["mask_out_dir"]
    mkdir_if_missing(os.path.join(base_dir, mask_out_dir))

    old_im_path = os.path.join(base_dir, image_dir)
    new_im_path = os.path.join(base_dir, image_out_dir)
    old_mask_path = os.path.join(base_dir, mask_dir)
    new_mask_path = os.path.join(base_dir, mask_out_dir)
    for im_path in tqdm(os.listdir(old_im_path)):
        im_old_path = os.path.join(old_im_path, im_path)
        try:
            mask_old_path = os.path.join(old_mask_path, im_path)
        except:
            continue
        im = cv2.imread(im_old_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_old_path, cv2.IMREAD_UNCHANGED)
        print('checking values', np.unique(im), np.unique(mask))
        print('im.shape, mask.shape', im.shape, mask.shape)
        if mask.shape[0] == 0 or mask.shape[1] == 0:
            print("mask empty")
            continue
        im = scale_to_square(im)
        mask = scale_to_square(mask)
        print('im.shape, mask.shape', im.shape, mask.shape)
        new_im_file = os.path.join(new_im_path, im_path)
        new_mask_file = os.path.join(new_mask_path, im_path)
        cv2.imwrite(new_im_file, im, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # 0 compression
        cv2.imwrite(new_mask_file, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])


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

    return conf


def save_config(conf, conf_path):
    # save config for current run in the folder
    with open(conf_path, "w") as f:
        conf.write(f)


if __name__ == "__main__":
    # parse the provided configuration file
    conf = read_config()

    task = conf.get("GENERAL", "task").upper()
    task = literal_eval(task)

    print("Task: {}".format(task))
    if task == "AUGMENT":
        augment_data(conf)

    if task == "TRAIN":
        train(conf)

    if task == "BENCHMARK":
        benchmark(conf)

    if task == "RESIZE":
        resize_images(conf)
