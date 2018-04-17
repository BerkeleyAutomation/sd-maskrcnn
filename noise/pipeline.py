import os
import logging
import argparse
import configparser
from tqdm import tqdm
from ast import literal_eval

import cv2
import numpy as np
import tensorflow as tf

from augmentation import augment_img
from utils import mkdir_if_missing


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
    for img_file in tqdm(os.listdir(img_dir), total=100):
        if img_file.endswith(".png"):
            # read in image
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # return list of augmented images and save
            new_img = augment_img(img, config)
            out_path = os.path.join(out_dir, img_file)
            cv2.imwrite(out_path, new_img)

    print("Augmentation complete; files saved in {}.\n".format(out_dir))


def train(config):
    pass


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

    # create a dictionary of the proper arguments, including
    # the requested task
    conf_dict = dict(conf.items(task) + [("task", task)])

    # return a type-sensitive version of the dictionary;
    # prevents further need to cast from string to other types
    return literal_eval(str(conf_dict))


if __name__ == "__main__":
    # parse the provided configuration file
    config = read_config()

    task = config["task"]
    if task == "AUGMENT":
        augment_data(config)

    if task == "train":
        train(config)

    if task == "benchmark":
        benchmark(config)
