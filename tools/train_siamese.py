import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from autolab_core import YamlConfig

from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig
from sd_maskrcnn.dataset import ImageDataset, TargetDataset
from sd_maskrcnn.coco_benchmark import coco_benchmark

from mrcnn import model as modellib, utils as utilslib, visualize

config = YamlConfig('../cfg/benchmark.yaml')
train_config = YamlConfig('../cfg/train.yaml')
inference_config = MaskConfig(config['model']['settings'])
inference_config.GPU_COUNT = 1
inference_config.IMAGES_PER_GPU = 1

training_config = MaskConfig(config['model']['settings'])
training_config.display()

#TODO: fix path
model = modellib.MaskRCNN(mode='training',
                          config=training_config,
                            model_dir='/nfs/diskstation/andrew_lee/sdmaskrcnn_target_results')

target_dataset = TargetDataset('/home/andrew_lee/sd-maskrcnn/data/')

target_dataset.load()
target_dataset.prepare()

data_generator = modellib.data_generator(target_dataset, training_config)

#TODO: FIX
filepath = '/nfs/diskstation/projects/dex-net/segmentation/models/mask_rcnn_coco_ft_wisdom_20180914-191150.h5'
model.load_weights_from_sd_mrcnn_model(filepath=filepath)

model.set_log_dir()
model.train(target_dataset, None, learning_rate=1e-4,
           epochs=20, layers='siamese')
