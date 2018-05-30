import os
import argparse
import configparser
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from ast import literal_eval

from eval_coco import coco_benchmark
# from eval_saurabh import s_benchmark
from pcl_detect import detect, EuclideanClusterExtractor, RegionGrowingSegmentor
from pcl_utils import get_conf_dict, mkdir_if_missing
from pcl_vis import visualize_predictions

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

def benchmark(conf):
    """Benchmarks a model, computes and stores model predictions and then
    evaluates them on COCO metrics and Saurabh's old benchmarking script."""

    config = get_conf_dict(conf)
    print("Benchmarking PCL method.")

    # Create new directory for run outputs
    # In what location should we put this new directory?
    output_dir = config['output_dir']
    
    # What is it called?
    run_name = config['run_name']
    run_dir = os.path.join(output_dir, run_name)
    mkdir_if_missing(run_dir)

    # Save config in run directory
    save_config(conf, os.path.join(run_dir, config["save_conf_name"]))

    # directory of test images and segmasks
    test_dir = config['test_dir']
    
    # get indices file name
    indices_name = config['indices_name'] + '_indices.npy'
    indices_arr = np.load(os.path.join(test_dir, indices_name))


    # Get location of file for relative path to binaries
    file_dir = os.path.dirname(__file__)

    # Get type of PCL detector and options for each
    detector_type = config['detector_type']
    if detector_type == 'euclidean':
        pcl_detector = EuclideanClusterExtractor(os.path.join(file_dir, 'euclidean_cluster_extraction'))
        tolerance = config['tolerance']
    elif detector_type == 'region_growing':
        pcl_detector = RegionGrowingSegmentor(os.path.join(file_dir, 'region_growing_segmentation'))
        n_neighbors = config['n_neighbors']
        smoothness = config['smoothness']
        curvature = config['curvature']
    else:
        print('PCL detector type not supported')
        exit()

    min_cluster_size = config['min_cluster_size']
    max_cluster_size = config['max_cluster_size']

    ######## BENCHMARK JUST CREATES THE RUN DIRECTORY ########
    # code that actually produces outputs should be plug-and-play
    # depending on what kind of benchmark function we run.

    # If we want to remove bin pixels, pass in the directory with
    # those masks.
    if config['remove_bin_pixels']:
        bin_mask_dir = os.path.join(test_dir, config['bin_masks'])
    else:
        bin_mask_dir = None

    # Create predictions and record where everything gets stored.
    pred_mask_dir, pred_info_dir, gt_mask_dir = \
        detect(pcl_detector, run_dir, test_dir, indices_arr, bin_mask_dir)

    coco_benchmark(pred_mask_dir, pred_info_dir, gt_mask_dir)
    visualize_predictions(run_dir, test_dir, pred_mask_dir, pred_info_dir, indices_arr)
    # s_benchmark(run_dir, dataset_real, inference_config, pred_mask_dir, pred_info_dir)

    print("Saved benchmarking output to {}.\n".format(run_dir))


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

    if task == "BENCHMARK":
        benchmark(conf)
