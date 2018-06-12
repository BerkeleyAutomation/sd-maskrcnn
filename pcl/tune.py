import configparser as cp
import argparse
import numpy as np
import os
from pipeline import benchmark
from pcl_utils import mkdir_if_missing
from pycocotools.cocoeval import COCOeval

# Parse input arguments to get directories and detector type
parser = argparse.ArgumentParser(description='Tune PCL implementations on dataset')
parser.add_argument('dataset_dir', help='directory of dataset')
parser.add_argument('output_dir',  help='output directory')
parser.add_argument('detector_type',  help='type of PCL detector, either euclidean or region_growing')
parser.add_argument('--indices', default='sample_test', help='.npy file of indices for dataset')

args = parser.parse_args()

# Set up config for benchmarking
config = cp.ConfigParser()
config['GENERAL'] = {'task': '"benchmark"'}
config['BENCHMARK'] = {
    'output_dir': '"' + args.output_dir + '"',
    'run_name': '"pcl_tune"',
    'save_conf_name': '"pcl_tune.ini"',
    'test_dir': '"' + args.dataset_dir + '"',
    'indices_name': '"' + args.indices + '"',
    'bin_masks': '"segmasks_filled"',
    'detector_type': '"' + args.detector_type + '"',
    'max_cluster_size': 1000000,
    'output_pred_vis': False,
    'show_bbox_pred': False,
    'show_class_pred': False,
    'output_s_bench': False
}

mkdir_if_missing(args.output_dir)

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
            config['BENCHMARK']['min_cluster_size'] = str(int(j))
            config['BENCHMARK']['tolerance'] = str(i)
            benchmark(config)
            out_coco = np.load(os.path.join(args.output_dir, 'pcl_tune', 'pred_masks', 'coco_eval.npy'))
            run_precision = np.mean(out_coco.item()['precision'].squeeze())
            run_recall = np.mean(out_coco.item()['recall'].squeeze())
            if run_precision > max_precision:
                max_precision = run_precision
                best_prec_tol = i
                best_prec_min_cluster = int(j)
            if run_recall > max_recall:
                max_recall = run_recall
                best_rec_tol = i
                best_rec_min_cluster = int(j)


# Loop over 4 params meaningful for region growing and record best runs
elif args.detector_type == 'region_growing':
    best_prec_curvature = 0
    best_prec_min_cluster = 0
    best_prec_knn = 0
    best_prec_smoothness = 0
    best_rec_curvature = 0
    best_rec_min_cluster = 0
    best_rec_knn = 0
    best_rec_smoothness = 0
    for i in np.linspace(0.01, 0.1, 19):
        for j in np.linspace(100, 1000, 19):
            for k in np.arange(4,11):
                for l in np.linspace(0.05, 0.5, 19):
                    config['BENCHMARK']['curvature'] = str(i)
                    config['BENCHMARK']['min_cluster_size'] = str(int(j))
                    config['BENCHMARK']['n_neighbors'] = str(k)
                    config['BENCHMARK']['smoothness'] = str(l)
                    benchmark(config)
                    out_coco = np.load(os.path.join(args.output_dir, 'pcl_tune', 'pred_masks', 'coco_eval.npy'))
                    run_precision = np.mean(out_coco.item()['precision'].squeeze())
                    run_recall = np.mean(out_coco.item()['recall'].squeeze())
                    if run_precision > max_precision:
                        max_precision = run_precision
                        best_prec_curvature = i
                        best_prec_min_cluster = int(j)
                        best_prec_knn = k
                        best_prec_smoothness = l
                    if run_recall > max_recall:
                        max_recall = run_recall
                        best_rec_curvature = i
                        best_rec_min_cluster = int(j)
                        best_rec_knn = k
                        best_rec_smoothness = l
else:
    print('Detector Type not supported for tuning')
    exit()

# Print results at the end
print('Results for Detector: {0}'.format(args.detector_type))
print('Max Precision: {:.3f}, Best Tol: {:.3f}, Best Min Cluster: {:d}'.format(max_precision,best_prec_tol,best_prec_min_cluster))
print('Max Recall: {:.3f}, Best Tol: {:.3f}, Best Min Cluster: {:d}'.format(max_recall,best_rec_tol,best_rec_min_cluster))

            


