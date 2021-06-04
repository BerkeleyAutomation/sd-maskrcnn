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

import argparse
import os

import tensorflow as tf
from autolab_core import YamlConfig
from keras.backend.tensorflow_backend import set_session

from sd_maskrcnn.dataset import ImageDataset
from sd_maskrcnn.model import SDMaskRCNNModel


def train(config):

    # Training dataset
    dataset_train = ImageDataset(config)
    dataset_train.load(config["dataset"]["train_indices"], augment=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ImageDataset(config)
    dataset_val.load(config["dataset"]["val_indices"])
    dataset_val.prepare()

    # Load config
    config["model"]["settings"][
        "steps_per_epoch"
    ] = dataset_train.indices.size / (
        config["model"]["settings"]["images_per_gpu"]
        * config["model"]["settings"]["gpu_count"]
    )

    # Create the model.
    model = SDMaskRCNNModel("training", config["model"])

    # save config in run folder
    config.save(
        os.path.join(config["model"]["path"], config["save_conf_name"])
    )

    # train and save weights to model_path
    model.train(dataset_train, dataset_val)


if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and train
    conf_parser = argparse.ArgumentParser(
        description="Train SD Mask RCNN model"
    )
    conf_parser.add_argument(
        "--config",
        action="store",
        default="cfg/train.yaml",
        dest="conf_file",
        type=str,
        help="path to the configuration file",
    )
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    # Use mixed precision for speedup
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

    # Set up tf session to use what GPU mem it needs and train
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=tf_config) as sess:
        set_session(sess)
        train(config)
