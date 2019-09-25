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
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from autolab_core import YamlConfig

from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig
from sd_maskrcnn.dataset import ImageDataset

from mrcnn import model as modellib, utils as utilslib

def train(config):

    # Training dataset
    dataset_train = ImageDataset(config)
    dataset_train.load(config['dataset']['train_indices'], augment=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ImageDataset(config)
    dataset_val.load(config['dataset']['val_indices'])
    dataset_val.prepare()

    # Load config
    image_shape = config['model']['settings']['image_shape']
    config['model']['settings']['image_min_dim'] = min(image_shape)
    config['model']['settings']['image_max_dim'] = max(image_shape)
    train_config = MaskConfig(config['model']['settings'])
    train_config.STEPS_PER_EPOCH = dataset_train.indices.size/(train_config.IMAGES_PER_GPU*train_config.GPU_COUNT)
    train_config.display()

    # Create directory if it doesn't currently exist
    utils.mkdir_if_missing(config['model']['path'])

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
    exclude_layers = []
    print("Loading weights ", weights_path)
    if config['model']['weights'].lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        if config['model']['settings']['image_channel_count'] == 1:
            exclude_layers = ['conv1']
        exclude_layers += ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        model.load_weights(weights_path, by_name=True, exclude=exclude_layers)
    elif config['model']['weights'].lower() == "imagenet":
        if config['model']['settings']['image_channel_count'] == 1:
            exclude_layers = ['conv1']
        model.load_weights(weights_path, by_name=True, exclude=exclude_layers)
    elif config['model']['weights'].lower() != "new":
        model.load_weights(weights_path, by_name=True)

    # save config in run folder
    config.save(os.path.join(config['model']['path'], config['save_conf_name']))

    # train and save weights to model_path
    model.train(dataset_train, dataset_val, learning_rate=train_config.LEARNING_RATE,
                epochs=config['model']['epochs'], layers='all')

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

    # Use mixed precision for speedup
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # Set up tf session to use what GPU mem it needs and train
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        set_session(sess)
        train(config)
