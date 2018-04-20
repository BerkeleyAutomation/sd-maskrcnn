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

from augmentation import augment_img
from eval_coco import *

from train_clutter import mkdir_if_missing
import model as modellib, visualize, utils
from clutter import ClutterConfig

def augment_data(config):
    """
    Using provided image directory and output directory, perform data
    augmentation methods on each image and save the new copy to the
    output directory.
    """
    print(config)
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
    pass


def benchmark(config):
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

    # Load gt dataset and generate annotations
    test_dir = config['test_dir'] # directory of test images
    test_segmasks_dir = config['test_segmasks_dir'] # directory of test image segmasks
    encode_gt(test_segmasks_dir)

    # Load specified model
    class InferenceConfig(ClutterConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig(mean=128)

    model_path = config['model_path']
    model_dir, model_name = os.path.split(model_path)
    model = modellib.MaskRCNN(mode='inference', config=inference_config,
                              model_dir=model_dir)

    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)



    # Feed images into model one by one. For each image, predict, save, visualize?
    N = len([p for p in os.listdir(test_dir) if p.endswith('.png')])
    for i in range(N):
        im_name = str(i) + '.png'
        print('evaluating', os.path.join(test_dir, im_name))
        image = io.imread(os.path.join(test_dir, im_name))

        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            max_dim=inference_config.IMAGE_MAX_DIM,
            padding=inference_config.IMAGE_PADDING)
        results = model.detect([image], verbose=1)
        r = results[0]

        # Save masks as .npy
        save_masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])])
        print(save_masks.shape)
        save_masks_path = os.path.join(pred_dir, str(i) + '.npy')
        np.save(save_masks_path, save_masks)
        print(save_masks_path)

        # Visualize
        def get_ax(rows=1, cols=1, size=8):
            """Return a Matplotlib Axes array to be used in
            all visualizations in the notebook. Provide a
            central point to control graph sizes.

            Change the default size attribute to control the size
            of rendered images
            """
            _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
            return ax


        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['bg', 'obj'], r['scores'], ax=get_ax())
        file_name = os.path.join(vis_dir, 'vis_{}'.format(im_name))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Generate prediction annotations
    encode_predictions(pred_dir)
    coco_benchmark(os.path.join(test_segmasks_dir, 'annos_gt.json'), os.path.join(pred_dir, 'annos_pred.json'))

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
        print(value)
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
