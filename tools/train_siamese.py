
import os
import sys
import time
import argparse
import numpy as np

from autolab_core import YamlConfig

from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig
from sd_maskrcnn.dataset import TargetDataset, TargetStackDataset

from mrcnn import model as modellib, utils as utilslib

def train(config):

    # Training dataset
    dataset_train = TargetStackDataset(config, config['dataset']['path'],
                                       config['dataset']['tuples'],
                                       images=config['dataset']['images'],
                                       masks=config['dataset']['masks'],
                                       targets=config['dataset']['targets'],
                                       augment_targets=config['dataset']['augment'])
    #TODO: bring back augment
    dataset_train.load(config['dataset']['train_indices'])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = None
    # if config['dataset']['val_indices']:
    #     dataset_val = TargetDataset(config['dataset']['path'])
    #     dataset_val.load()
    #     dataset_val.prepare()

    # Load config
    train_config = MaskConfig(config['model']['settings'])
    train_config.STEPS_PER_EPOCH = dataset_train.example_indices.size/(train_config.IMAGES_PER_GPU*train_config.GPU_COUNT)
    # train_config.STEPS_PER_EPOCH = 5
    train_config.display()

    # Create the model.
    model = modellib.MaskRCNN(mode='training', config=train_config,
                              model_dir=config['model']['path'])


    # Select weights file to load
    if config['model']['weights'].lower() == "coco":
        weights_path = os.path.join(config['model']['path'], 'mask_rcnn_coco.h5')
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
        if config['model']['weight_type'].lower() == 'old':
            model.load_weights_from_sd_mrcnn_model(weights_path)
        else:
            model.load_weights(weights_path, by_name=True)

    # save config in run folder
    config.save(os.path.join(config['model']['path'], config['save_conf_name']))

    # train and save weights to model_path
    model.train(dataset_train, dataset_val, learning_rate=train_config.LEARNING_RATE,
                epochs=config['model']['epochs'], layers=config['model']['layers'].lower())

    # save in the models folder
    current_datetime = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(config['model']['path'], "mask_rcnn_{}_{}.h5".format(train_config.NAME, current_datetime))
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
