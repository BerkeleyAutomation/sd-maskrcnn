"""
PCL Benchmark Usage Notes:

Please edit "cfg/pcl_benchmark.yaml" to specify the necessary parameters for that task.

Run this file with the tag --config [config file name] (in this case,
cfg/pcl_benchmark.yaml).

Here is an example run command (GPU selection included):
CUDA_VISIBLE_DEVICES='0' python3 pcl/benchmark.py --config cfg/pcl_benchmark.yaml
"""

import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt

from autolab_core import YamlConfig
from perception import DepthImage

# from cppdetect import detect
from pydetect import detect

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import Mask R-CNN repo
sys.path.append(ROOT_DIR) # To find local version of the library
from maskrcnn.mrcnn import visualize, utils as utilslib
from sd_maskrcnn.utils import mkdir_if_missing
from sd_maskrcnn.coco_benchmark import coco_benchmark
from sd_maskrcnn.saurabh_benchmark import compute_gt_stats, plot_stats, subplot, voc_ap_fast, calc_pr, inst_bench, inst_bench_image

def benchmark(config):
    """Benchmarks a model, computes and stores model predictions and then
    evaluates them on COCO metrics and Saurabh's old benchmarking script."""

    print("Benchmarking PCL method.")

    # Create new directory for run outputs
    # In what location should we put this new directory?
    output_dir = config['output_dir']
    mkdir_if_missing(output_dir)

    # Save config in run directory
    config.save(os.path.join(output_dir, config['save_conf_name']))

    # directory of test images and segmasks
    test_dir = config['test']['path']
    indices_arr = np.load(os.path.join(test_dir, config['test']['indices']))
    bin_mask_dir = os.path.join(test_dir, config['test']['bin_masks'])

    # Create predictions and record where everything gets stored.
    pred_mask_dir, pred_info_dir, gt_mask_dir = \
        detect(config['detector'], output_dir, test_dir, indices_arr, bin_mask_dir)

    ap, ar = coco_benchmark(pred_mask_dir, pred_info_dir, gt_mask_dir)
    if config['vis']['predictions']:
        visualize_predictions(output_dir, test_dir, indices_arr, pred_mask_dir, pred_info_dir, show_bbox=config['vis']['show_bbox_pred'], show_class=config['vis']['show_class_pred'])
    if config['vis']['s_bench']:
        s_benchmark(output_dir, test_dir, indices_arr, pred_mask_dir, pred_info_dir, gt_mask_dir)

    print("Saved benchmarking output to {}.\n".format(output_dir))
    return ap, ar

def visualize_predictions(run_dir, dataset_dir, indices_arr, pred_mask_dir, pred_info_dir, show_bbox=True, show_class=True):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, 'vis')
    depth_dir = os.path.join(dataset_dir, 'depth_ims_numpy')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    image_ids = np.arange(indices_arr.size)
    ##################################################################
    # Process each image
    ##################################################################
    print('VISUALIZING PREDICTIONS')
    for image_id in tqdm(image_ids):
        base_name = 'image_{:06d}'.format(indices_arr[image_id])
        depth_image_fn = base_name + '.npy'

        # Load image and ground truth data and resize for net
        depth_data = np.load(os.path.join(depth_dir, depth_image_fn))
        image = DepthImage(depth_data).to_color().data

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'image_{:06}.npy'.format(image_id)))
        # Must transpose from (n, h, w) to (h, w, n)
        if r_masks.any():
            r['masks'] = np.transpose(r_masks, (1, 2, 0))
        else:
            r['masks'] = r_masks
        image = cv2.resize(image, (r['masks'].shape[1], r['masks'].shape[0]))
        # Visualize
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['bg', 'obj'], show_bbox=show_bbox, show_class=show_class)
        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

def s_benchmark(run_dir, dataset_dir, indices_arr, pred_mask_dir, pred_info_dir, gt_mask_dir, vis_missed=False):
    """Runs Saurabh's old benchmarking code."""

    print("Computing Saurabh's bounding box metrics")

    results_dir = os.path.join(run_dir, 'results_saurabh')
    if not os.path.exists(os.path.join(results_dir, 'vis_fn')):
        os.makedirs(os.path.join(results_dir, 'vis_fn'))

    ms = [[] for _ in range(10)]
    thresh_all = [0.25, 0.5, 0.75]
    for ov in thresh_all:
        for m in ms:
            m.append([])
    ms.append(thresh_all)
    ms = list(zip(*ms))

    image_ids = np.arange(indices_arr.size)
    for image_id in tqdm(image_ids):
        
        # Load image and ground truth data
        image = cv2.imread(os.path.join(dataset_dir, 'depth_ims', 'image_{:06}.png'.format(indices_arr[image_id])))
        image = np.transpose(image, (1, 0, 2))
        gt_mask = np.load(os.path.join(gt_mask_dir, 'image_{:06}.npy'.format(image_id))).transpose()
        gt_class_id = np.array([1 for _ in range(gt_mask.shape[2])]).astype(np.int32)
        gt_bbox = utilslib.extract_bboxes(gt_mask)
        gt_stat, stat_name = compute_gt_stats(gt_bbox, gt_mask)

        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'image_{:06}.npy'.format(image_id)))
        # Must transpose from (n, h, w) to (h, w, n)
        r['masks'] = np.transpose(r_masks, (1, 2, 0))

        # Make sure scores are sorted.
        sc = r['scores']
        is_sorted = np.all(np.diff(sc) <= 0)
        assert(is_sorted)
        overlaps = utilslib.compute_overlaps(r['rois'], gt_bbox)
        dt = {'sc': sc[:,np.newaxis]*1.}
        gt = {'diff': np.zeros((gt_bbox.shape[0],1), dtype=np.bool)}

        for tps, fps, scs, num_insts, dup_dets, inst_ids, ovs, tp_inds, fn_inds, \
            gt_stats, thresh in ms:
            tp, fp, sc, num_inst, dup_det, inst_id, ov = inst_bench_image(dt, gt, {'minoverlap': thresh}, overlaps)
            tp_ind = np.sort(inst_id[tp]); fn_ind = np.setdiff1d(np.arange(num_inst), tp_ind)
            tps.append(tp); fps.append(fp); scs.append(sc); num_insts.append(num_inst)
            dup_dets.append(dup_det); inst_ids.append(inst_id); ovs.append(ov)
            tp_inds.append(tp_ind); fn_inds.append(fn_ind)
            gt_stats.append(gt_stat)

        # Visualize missing objects
        fn_ind = ms[1][8][-1] # missing objects at threshold 0.5
        if fn_ind.size > 0:
            _, _, axes = subplot(plt, (fn_ind.size+1, 1), sz_y_sz_x=(5,5))
            ax = axes.pop(); ax.imshow(image); ax.set_axis_off()
            class_names = {1: ''}
            for _ in range(fn_ind.size):
                j = fn_ind[_]
                ax = axes.pop()
                visualize.display_instances(image, gt_bbox[j:j+1,:],
                                            gt_mask[:,:,j:j+1], gt_class_id[j:j+1], class_names,
                                            ax=ax, title='')
            file_name = os.path.join(results_dir, 'vis_fn',
                                     'vis_{:06d}.png'.format(image_id))
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            plt.close()

    print('Computing AP and plotting PR curves...')
    # Compute AP
    for tps, fps, scs, num_insts, dup_dets, inst_ids, ovs, tp_inds, fn_inds, \
        gt_stats, thresh in ms:
        ap, rec, prec, npos, _ = inst_bench(None, None, None, tp=tps, fp=fps, score=scs, numInst=num_insts)
        str_ = 'mAP: {:.3f}, prec: {:.3f}, rec: {:.3f}, npos: {:d}'.format(
          ap[0], np.min(prec), np.max(rec), npos)
        # logging.error('%s', str_)
        # print("mAP: ", ap[0], "prec: ", np.max(prec), "rec: ", np.max(rec), "prec-1: ",
        #   prec[-1], "npos: ", npos)
        plt.style.use('fivethirtyeight') #bmh')
        _, _, axes = subplot(plt, (3,4), (8,8), space_y_x=(0.2,0.2))
        ax = axes.pop(); ax.plot(rec, prec, 'r'); ax.set_xlim([0,1]); ax.set_ylim([0,1])
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(str_) #'{:5.3f}'.format(ap[0]*100))
        plot_stats(stat_name, gt_stats, tp_inds, fn_inds, axes)
        file_name = os.path.join(results_dir, 'pr_stats_{:d}.png'.format(int(thresh*100)))
        # logging.error('plot file name: %s', file_name)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark SD Mask RCNN model")
    conf_parser.add_argument("--config", action="store", default="cfg/pcl_benchmark.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    benchmark(config)
