import os
import sys
import time
import argparse
import numpy as np

from autolab_core import YamlConfig

import utils
from config import MaskConfig
from dataset import ImageDataset

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import Mask R-CNN repo
sys.path.append(ROOT_DIR) # To find local version of the library
from maskrcnn.mrcnn import model as modellib, utils as utilslib

COCO_WEIGHTS_PATH = '/nfs/diskstation/projects/dex-net/segmentation/models/mask_rcnn_coco.h5'

def train(config):

    # Training dataset
    dataset_train = ImageDataset(config['dataset']['path'], config['dataset']['images'], config['dataset']['masks'], config['dataset']['occlusions'])
    dataset_train.load(config['dataset']['train_indices'], augment=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ImageDataset(config['dataset']['path'], config['dataset']['images'], config['dataset']['masks'], config['dataset']['occlusions'])
    dataset_val.load(config['dataset']['val_indices'])
    dataset_val.prepare()

    # Load config
    train_config = MaskConfig(config['model']['settings'])
    train_config.STEPS_PER_EPOCH = dataset_train.indices.size/(train_config.IMAGES_PER_GPU*train_config.GPU_COUNT)
    # train_config.STEPS_PER_EPOCH = 5
    train_config.display()

    # Create the model.
    model = modellib.MaskRCNN(mode='training', config=train_config,
                              model_dir=config['model']['log_path'])

    # Select weights file to load
    if config['model']['weights'].lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utilslib.download_trained_weights(weights_path)
    elif config['model']['weights'].lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif config['model']['weights'].lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = config['model']['weights']

    # Load weights
    print("Loading weights ", weights_path)
    if config['model']['weights'].lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif config['model']['weights'].lower() == "new":
        model.set_log_dir()
    else:
        model.load_weights(weights_path, by_name=True)

    # save config in run folder
    config.save(os.path.join(config['model']['log_path'], config['save_conf_name']))

    # train and save weights to model_path
    model.train(dataset_train, dataset_val, learning_rate=train_config.LEARNING_RATE,
                epochs=config['model']['epochs'], layers='all')

    # save in the models folder
    current_datetime = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(config['model']['log_path'], "mask_rcnn_{}_{}.h5".format(train_config.NAME, current_datetime))
    model.keras_model.save_weights(model_path)

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and train
    conf_parser = argparse.ArgumentParser(description="Train SD Mask RCNN model")
    conf_parser.add_argument("--config", action="store", default="cfg/train.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    # utils.set_tf_config()
    train(config)
