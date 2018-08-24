import argparse
import numpy as np
import os
from pycocotools.cocoeval import COCOeval
from autolab_core import YamlConfig

from ..benchmark import benchmark

# Parse input arguments to get directories and detector type
parser = argparse.ArgumentParser(description='Tune PCL implementations on dataset')
parser.add_argument('dataset_dir', help='directory of dataset')
parser.add_argument('output_dir',  help='output directory')
parser.add_argument('detector_type',  help='type of PCL detector, either euclidean or region_growing')
parser.add_argument('--indices', default='test_indices.npy', help='.npy file of indices for dataset')

args = parser.parse_args()

# Set up config for benchmarking
config = YamlConfig()
config['output_dir'] = args.output_dir
config['save_conf_name'] = 'pcl_tune.yaml'
config['test'] = {
    'path': args.dataset_dir,
    'bin_masks': 'segmasks_filled',
    'indices': args.indices
}

config['vis'] = {
    'predictions': 0,
    's_bench': 0
}

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

max_precision = 0
max_recall = 0

# Loop over two params that are meaningful for euclidean clustering and save best runs for both precision and recall
if args.detector_type == 'euclidean':
    best_prec_tol = 0
    best_prec_min_cluster = 0
    best_rec_tol = 0
    best_rec_min_cluster = 0
    for i in np.linspace(0.001, 0.01, 19):
        for j in np.linspace(100, 1000, 19):
            config['detector'] = {'type': 'euclidean'}
            config['detector']['euclidean'] = {
                'min_cluster_size': int(j),
                'max_cluster_size': 1000000,
                'tolerance': i
            }
            ap, ar = benchmark(config)
            if ap > max_precision:
                max_precision = ap
                best_prec_tol = i
                best_prec_min_cluster = int(j)
            if ar > max_recall:
                max_recall = ar
                best_rec_tol = i
                best_rec_min_cluster = int(j)
    # Print results at the end
    print('Results for Detector: {0}'.format(args.detector_type))
    print('Max Precision: {:.3f}, Best Tol: {:.3f}, Best Min Cluster: {:d}'.format(max_precision,best_prec_tol,best_prec_min_cluster))
    print('Max Recall: {:.3f}, Best Tol: {:.3f}, Best Min Cluster: {:d}'.format(max_recall,best_rec_tol,best_rec_min_cluster))
    config['detector']['min_cluster_size'] = best_prec_min_cluster
    config['detector']['tolerance'] = best_prec_tol
    config.save(os.path.join(args.output_dir, config['save_conf_name']))

# Loop over 2 params meaningful for region growing and record best runs
# Set the min cluster size according to results from running the euclidean tuning
elif args.detector_type == 'region_growing':
    best_prec_curvature = 0
    best_prec_smoothness = 0
    best_rec_curvature = 0
    best_rec_smoothness = 0
    for i in np.linspace(0.05, 0.5, 19):
        for j in np.linspace(0.1, 0.5, 17):
            config['detector'] = {'type': 'region_growing'}
            config['detector']['region_growing'] = {
                'min_cluster_size': 800,
                'max_cluster_size': 1000000,
                'n_neighbors': 5,
                'smoothness': j,
                'curvature': i
            }
            ap, ar = benchmark(config)
            if ap > max_precision:
                max_precision = ap
                best_prec_curvature = i
                best_prec_smoothness = j
            if ar > max_recall:
                max_recall = ar
                best_rec_curvature = i
                best_rec_smoothness = j
    # Print results at the end
    print('Results for Detector: {0}'.format(args.detector_type))
    print('Max Precision: {:.3f}, Best Curvature: {:.3f}, Best Smoothness: {:.4f}'.format(max_precision,best_prec_curvature,best_prec_smoothness))
    print('Max Recall: {:.3f}, Best Curvature: {:.3f}, Best Smoothness: {:.4f}'.format(max_recall,best_rec_curvature,best_rec_smoothness))
    config['detector']['curvature'] = best_prec_curvature
    config['detector']['smoothness'] = best_prec_smoothness
    config.save(os.path.join(args.output_dir, config['save_conf_name']))
else:
    print('Detector Type not supported for tuning')


