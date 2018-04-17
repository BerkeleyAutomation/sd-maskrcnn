import tensorflow as tf
import numpy as np
import argparse
import configparser




def augment(config):
    """
    Using provided image directory and output directory, perform data
    augmentation methods on each image and save the new copy to the
    output directory.
    """
    img_path = config["img_path"]
    out_path = config["out_path"]



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

    # return a dictionary of the proper arguments
    return dict(conf.items(task) + ("task", task))


if __name__ == "__main__":
    # parse the provided configuration file
    config = read_config()

    task = config["task"]
    if task == "AUGMENT":
        augment(config)

    if task == "train":
        train(config)

    if task == "benchmark":
        benchmark(config)
