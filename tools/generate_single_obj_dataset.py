"""
Generates a dataset of individual object images in bins.
"""

import argparse
import gc
import numpy as np
import os
import json
import shutil
import time
import traceback
import matplotlib.pyplot as plt

from autolab_core import TensorDataset, YamlConfig, Logger
import autolab_core.utils as utils
from perception import DepthImage, GrayscaleImage, BinaryImage, ColorImage

from sd_maskrcnn.envs import BinHeapEnv
from sd_maskrcnn.envs.constants import *

from generate_mask_dataset import generate_segmask_dataset

SEED = 744

# set up logger
logger = Logger.get_logger('tools/generate_single_obj_dataset.py')


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Generate a dataset of single-object target images for a SD Mask R-CNN target branch')
    parser.add_argument('output_dataset_path', type=str, default=None, help='directory to store a dataset containing the images')

    args = parser.parse_args()
    output_dataset_path = args.output_dataset_path
    save_tensors = False
    warm_start = False

    # handle config filename
    config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   '..',
                                   'cfg_target/generate_mask_dataset.yaml')

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # create dataset root directory
    # (will contain an actual dataset dir per each mesh)
    if not os.path.exists(output_dataset_path):
        os.mkdir(output_dataset_path)

    # open config file
    config = YamlConfig(config_filename)

    # get list of object names
    mesh_dir = config['state_space']['heap']['objects']['mesh_dir']
    mesh_list = [] # (dataset_name, file_name)

    for root, dirs, files in os.walk(mesh_dir):
        dataset_name = os.path.basename(root)
        for f in files:
            mesh_name, ext = os.path.splitext(f)
            mesh_list.append((dataset_name, mesh_name))

    # generate dataset
    generation_start = time.time()

    for dataset_name, mesh_name in mesh_list:
        print('Generating images for', mesh_name, 'in', dataset_name)
        # new path per single mesh dataset
        mesh_output_dataset_path = os.path.join(output_dataset_path, mesh_name)
        # edit config to reflect
        config['state_space']['heap']['objects']['object_keys'] = {
            dataset_name: [mesh_name]
        }
        generate_segmask_dataset(mesh_output_dataset_path, config,
                                 save_tensors=False, warm_start=False)

    # log time
    generation_stop = time.time()
    logger.info('Mask dataset generation took %.3f sec' %(generation_stop-generation_start))
