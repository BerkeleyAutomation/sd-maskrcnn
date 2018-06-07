"""Visualize PCL detection predictions.
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import visualize, utils

from pcl_utils import mkdir_if_missing
from perception import DepthImage
from eval_utils import *

def visualize_predictions(run_dir, dataset_dir, indices_arr, pred_mask_dir, pred_info_dir):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, 'vis')
    depth_dir = os.path.join(dataset_dir, 'depth_ims_numpy')
    mkdir_if_missing(vis_dir)

    ##################################################################
    # Process each image
    ##################################################################
    print('VISUALIZING PREDICTIONS')
    for image_id in tqdm(indices_arr):
        base_name = 'image_{:06d}'.format(image_id)
        depth_image_fn = base_name + '.npy'

        # Load image and ground truth data and resize for net
        depth_data = np.load(os.path.join(depth_dir, depth_image_fn))
        image = DepthImage(depth_data).to_color().data

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'image_{:06}.npy'.format(image_id)))
        # Must transpose from (n, h, w) to (h, w, n)
        r['masks'] = np.transpose(r_masks, (1, 2, 0))
        # Visualize
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['bg', 'obj'], r['scores'])
        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

def s_benchmark(run_dir, dataset_dir, indices_arr, pred_mask_dir, pred_info_dir, gt_mask_dir, vis_missed=False):
    """Runs Saurabh's old benchmarking code."""

    print("Computing Saurabh's bounding box metrics")

    results_dir = os.path.join(run_dir, 'results_saurabh')
    mkdir_if_missing(results_dir)

    mkdir_if_missing(os.path.join(results_dir, 'vis_fn'))

    ms = [[] for _ in range(10)]
    thresh_all = [0.25, 0.5, 0.75]
    for ov in thresh_all:
        for m in ms:
            m.append([])
    ms.append(thresh_all)
    ms = list(zip(*ms))

    for image_id in tqdm(indices_arr):
        
        # Load image and ground truth data
        image = cv2.imread(os.path.join(dataset_dir, 'depth_ims', 'image_{:06}.png'.format(image_id)))
        image = np.transpose(image, (1, 0, 2))
        gt_mask = np.load(os.path.join(gt_mask_dir, 'image_{:06}.npy'.format(image_id))).transpose()
        gt_class_id = np.array([1 for _ in range(gt_mask.shape[2])]).astype(np.int32)
        gt_bbox = utils.extract_bboxes(gt_mask)
        gt_stat, stat_name = compute_gt_stats(gt_bbox, gt_mask)

        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'image_{:06}.npy'.format(image_id)))
        # Must transpose from (n, h, w) to (h, w, n)
        r['masks'] = np.transpose(r_masks, (1, 2, 0))

        # Make sure scores are sorted.
        sc = r['scores']
        is_sorted = np.all(np.diff(sc) <= 0)
        assert(is_sorted)
        overlaps = utils.compute_overlaps(r['rois'], gt_bbox)
        dt = {'sc': sc[:,np.newaxis]*1.}
        gt = {'diff': np.zeros((gt_bbox.shape[0],1), dtype=np.bool)}

        for tps, fps, scs, num_insts, dup_dets, inst_ids, ovs, tp_inds, fn_inds, \
            gt_stats, thresh in ms:
            tp, fp, sc, num_inst, dup_det, inst_id, ov = inst_bench_image(dt, gt, {'minoverlap': thresh}, overlaps)
            tp_ind = np.sort(inst_id[tp]); fn_ind = np.setdiff1d(np.arange(num_inst), tp_ind);
            tps.append(tp); fps.append(fp); scs.append(sc); num_insts.append(num_inst);
            dup_dets.append(dup_det); inst_ids.append(inst_id); ovs.append(ov);
            tp_inds.append(tp_ind); fn_inds.append(fn_ind);
            gt_stats.append(gt_stat)

        # Visualize missing objects
        fn_ind = ms[1][8][-1] # missing objects at threshold 0.5
        if fn_ind.size > 0:
            fig, _, axes = subplot(plt, (fn_ind.size+1, 1), sz_y_sz_x=(5,5))
            ax = axes.pop(); ax.imshow(image); ax.set_axis_off();
            class_names = {1: ''}
            for _ in range(fn_ind.size):
                j = fn_ind[_]
                ax = axes.pop()
                visualize.display_instances(image, gt_bbox[j:j+1,:],
                                            gt_mask[:,:,j:j+1], gt_class_id[j:j+1], class_names,
                                            ax=ax, title='')
            file_name = os.path.join(results_dir, 'vis_fn',
                                     'vis_{:06d}.png'.format(np.flatnonzero(indices_arr == image_id)[0]))
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
        fig, _, axes = subplot(plt, (3,4), (8,8), space_y_x=(0.2,0.2))
        ax = axes.pop(); ax.plot(rec, prec, 'r'); ax.set_xlim([0,1]); ax.set_ylim([0,1]);
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(str_) #'{:5.3f}'.format(ap[0]*100))
        plot_stats(stat_name, gt_stats, tp_inds, fn_inds, axes)
        file_name = os.path.join(results_dir, 'pr_stats_{:d}.png'.format(int(thresh*100)))
        # logging.error('plot file name: %s', file_name)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

