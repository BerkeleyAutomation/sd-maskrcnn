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
from tqdm import tqdm
from matplotlib import pyplot as plt
import skimage
import numpy as np

from .utils import mkdir_if_missing
from mrcnn import model as modellib, visualize, utils as utilslib

def s_benchmark(run_dir, dataset_real, inference_config, pred_mask_dir, pred_info_dir, \
                vis_missed=False):
    """Runs supplementary benchmarking code."""

    print("Computing Supplementary's bounding box metrics")

    results_dir = os.path.join(run_dir, 'results_supplement')
    mkdir_if_missing(results_dir)

    image_ids = dataset_real.image_ids
    mkdir_if_missing(os.path.join(results_dir, 'vis_fn'))

    ms = [[] for _ in range(10)]
    thresh_all = [0.25, 0.5, 0.75]
    for ov in thresh_all:
        for m in ms:
            m.append([])
    ms.append(thresh_all)
    ms = list(zip(*ms))

    for image_id in tqdm(image_ids):
        # Load image and ground truth data
        image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_real,
                                                                        inference_config,
                                                                        image_id,
                                                                        use_mini_mask=False)
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
                                     'vis_{:06d}.png'.format(dataset_real.image_id[image_id]))
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

def compute_gt_stats(gt_bbox, gt_mask):
    # Compute statistics for all the ground truth things.
    hw = gt_bbox[:,2:] - gt_bbox[:,:2]
    hw = hw*1.
    min_side = np.min(hw,1)[:,np.newaxis]
    max_side = np.max(hw,1)[:,np.newaxis]
    aspect_ratio = np.max(hw, 1) / np.min(hw, 1)
    aspect_ratio = aspect_ratio[:,np.newaxis]
    log_aspect_ratio = np.log(aspect_ratio)
    box_area = np.prod(hw, 1)[:,np.newaxis]
    log_box_area = np.log(box_area)
    sqrt_box_area = np.sqrt(box_area)
    modal_area = np.sum(np.sum(gt_mask, 0), 0)[:,np.newaxis]*1.
    log_modal_area = np.log(modal_area)
    sqrt_modal_area = np.sqrt(modal_area)

    # Number of distinct components
    ov_connected = sqrt_box_area*1.
    for i in range(gt_mask.shape[2]):
        aa = skimage.measure.label(gt_mask[:,:,i], background=0)
        sz = np.bincount(aa.ravel())[1:]
        biggest = np.argmax(sz)+1
        big_comp = utilslib.extract_bboxes(aa[:,:,np.newaxis]==biggest)
        ov_connected[i,0] = utilslib.compute_overlaps(big_comp, gt_bbox[i:i+1,:])

    a = np.concatenate([min_side, max_side, aspect_ratio, log_aspect_ratio,
        box_area, log_box_area, sqrt_box_area, modal_area, log_modal_area,
        sqrt_modal_area, ov_connected], 1)
    n = ['min_side', 'max_side', 'aspect_ratio', 'log_aspect_ratio', 'box_area',
        'log_box_area', 'sqrt_box_area', 'modal_area', 'log_modal_area',
        'sqrt_modal_area', 'ov_connected']
    return a, n

def plot_stats(stat_name, gt_stats, tp_inds, fn_inds, axes):
    # Accumulate all stats for positives, and negatives.
    tp_stats = [gt_stat[tp_ind, :] for gt_stat, tp_ind in zip(gt_stats, tp_inds)]
    tp_stats = np.concatenate(tp_stats, 0)
    fn_stats = [gt_stat[fn_ind, :] for gt_stat, fn_ind in zip(gt_stats, fn_inds)]
    fn_stats = np.concatenate(fn_stats, 0)
    all_stats = np.concatenate(gt_stats, 0)

    for i in range(all_stats.shape[1]):
        ax = axes.pop()
        ax.set_title(stat_name[i])
        # np.histogram(all_stats
        min_, max_ = np.percentile(all_stats[:,i], q=[1,99])
        all_stats_ = all_stats[:,i]*1.
        all_stats_ = all_stats_[np.logical_and(all_stats_ > min_, all_stats_ < max_)]
        _, bins = np.histogram(all_stats_, 'auto')
        bin_size = bins[1]-bins[0]
        # s = np.unique(all_stats_); s = s[1:]-s[:-1]
        # if bin_size < np.min(s):
        #   bin_size = np.min(s)
        bins = np.arange(bins[0], bins[-1]+bin_size, bin_size)
        for _, (m, n) in \
            enumerate(zip([tp_stats[:,i], fn_stats[:,i]], ['tp', 'fn'])):
            ax.hist(m, bins, alpha=0.5, label=n, linewidth=1, linestyle='-', ec='k')
            ax2 = ax.twinx()
            t, _ = np.histogram(all_stats[:,i], bins)
            mis, _ = np.histogram(fn_stats[:,i], bins)
            mis_rate, = ax2.plot((bins[:-1] + bins[1:])*0.5, mis / np.maximum(t, 0.00001),
            'm', label='mis rate')
            ax2.set_ylim([0,1])
            ax2.tick_params(axis='y', colors=mis_rate.get_color())
            ax2.yaxis.label.set_color(mis_rate.get_color())
            ax2.grid('off')
            ax.legend(loc=2)
            ax2.legend()

def subplot(plt, Y_X, sz_y_sz_x=(10,10), space_y_x=(0.1,0.1), T=False):
    Y,X = Y_X
    sz_y, sz_x = sz_y_sz_x
    hspace, wspace = space_y_x
    plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
    fig, axes = plt.subplots(Y, X, squeeze=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if T:
        axes_list = axes.T.ravel()[::-1].tolist()
    else:
        axes_list = axes.ravel()[::-1].tolist()
    return fig, axes, axes_list

def voc_ap_fast(rec, prec):
    rec = rec.reshape((-1,1))
    prec = prec.reshape((-1,1))
    z = np.zeros((1,1)) 
    o = np.ones((1,1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))
    I = np.where(mrec[1:] != mrec[0:-1])[0]+1
    ap = np.sum((mrec[I] - mrec[I-1]) * mpre[I])
    return np.array(ap).reshape(1,)

def calc_pr(gt, out, wt=None):
    """Computes VOC 12 style AP (dense sampling).
    returns ap, rec, prec"""
    if wt is None:
        wt = np.ones((gt.size,1))

    gt = gt.astype(np.float64).reshape((-1,1))
    wt = wt.astype(np.float64).reshape((-1,1))
    out = out.astype(np.float64).reshape((-1,1))

    gt = gt*wt
    tog = np.concatenate([gt, wt, out], axis=1)*1.
    ind = np.argsort(tog[:,2], axis=0)[::-1]
    tog = tog[ind,:]
    cumsumsortgt = np.cumsum(tog[:,0])
    cumsumsortwt = np.cumsum(tog[:,1])
    prec = cumsumsortgt / cumsumsortwt
    rec = cumsumsortgt / np.sum(tog[:,0])

    ap = voc_ap_fast(rec, prec)
    return ap, rec, prec

def inst_bench_image(dt, gt, bOpts, overlap=None):
    nDt = len(dt['sc'])
    nGt = len(gt['diff'])
    numInst = np.sum(gt['diff'] == False)

    if overlap is None:
        overlap = bbox_utils.bbox_overlaps(dt['boxInfo'].astype(np.float), gt['boxInfo'].astype(np.float))
    
        # assert(issorted(-dt.sc), 'Scores are not sorted.\n');
    sc = dt['sc']

    det    = np.zeros((nGt,1)).astype(np.bool)
    tp     = np.zeros((nDt,1)).astype(np.bool)
    fp     = np.zeros((nDt,1)).astype(np.bool)
    dupDet = np.zeros((nDt,1)).astype(np.bool)
    instId = np.zeros((nDt,1)).astype(np.int32)
    ov     = np.zeros((nDt,1)).astype(np.float32)

    # Walk through the detections in decreasing score
    # and assign tp, fp, fn, tn labels
    for i in range(nDt):
        # assign detection to ground truth object if any
        if nGt > 0:
            maxOverlap = overlap[i,:].max(); maxInd = overlap[i,:].argmax()
            instId[i] = maxInd; ov[i] = maxOverlap
        else:
            maxOverlap = 0; instId[i] = -1; maxInd = -1
        # assign detection as true positive/don't care/false positive
        if maxOverlap >= bOpts['minoverlap']:
            if gt['diff'][maxInd] == False:
                if det[maxInd] == False:
                    # true positive
                    tp[i] = True
                    det[maxInd] = True
                else:
                    # false positive (multiple detection)
                    fp[i] = True
                    dupDet[i] = True
        else:
            # false positive
            fp[i] = True
    return tp, fp, sc, numInst, dupDet, instId, ov

def inst_bench(dt, gt, bOpts, tp=None, fp=None, score=None, numInst=None):
    """
    ap, rec, prec, npos, details = inst_bench(dt, gt, bOpts, tp = None, fp = None, sc = None, numInst = None)
    dt  - a list with a dict for each image and with following fields
        .boxInfo - info that will be used to cpmpute the overlap with ground truths, a list
        .sc - score
    gt
        .boxInfo - info used to compute the overlap,  a list
        .diff - a logical array of size nGtx1, saying if the instance is hard or not
    bOpt
        .minoverlap - the minimum overlap to call it a true positive
    [tp], [fp], [sc], [numInst]
        Optional arguments, in case the inst_bench_image is being called outside of this function
    """
    details = None
    if tp is None:
        # We do not have the tp, fp, sc, and numInst, so compute them from the structures gt, and out
        tp = []; fp = []; numInst = []; score = []; dupDet = []; instId = []; ov = []
        for i in range(len(gt)):
            # Sort dt by the score
            sc = dt[i]['sc']
            bb = dt[i]['boxInfo']
            ind = np.argsort(sc, axis = 0)
            ind = ind[::-1]
            if len(ind) > 0:
                sc = np.vstack((sc[i,:] for i in ind))
                bb = np.vstack((bb[i,:] for i in ind))
            else:
                sc = np.zeros((0,1)).astype(np.float)
                bb = np.zeros((0,4)).astype(np.float)

        dtI = dict({'boxInfo': bb, 'sc': sc})
        tp_i, fp_i, sc_i, numInst_i, dupDet_i, instId_i, ov_i = inst_bench_image(dtI, gt[i], bOpts)
        tp.append(tp_i); fp.append(fp_i); score.append(sc_i); numInst.append(numInst_i)
        dupDet.append(dupDet_i); instId.append(instId_i); ov.append(ov_i)
        details = {'tp': list(tp), 'fp': list(fp), 'score': list(score), 'dupDet': list(dupDet),
        'numInst': list(numInst), 'instId': list(instId), 'ov': list(ov)}

    tp = np.vstack(tp[:])
    fp = np.vstack(fp[:])
    sc = np.vstack(score[:])

    cat_all = np.hstack((tp,fp,sc))
    ind = np.argsort(cat_all[:,2])
    cat_all = cat_all[ind[::-1],:]
    tp = np.cumsum(cat_all[:,0], axis = 0)
    fp = np.cumsum(cat_all[:,1], axis = 0)
    npos = np.sum(numInst, axis = 0)

    # Compute precision/recall
    rec = tp / npos
    prec = np.divide(tp, (fp+tp))
    ap = voc_ap_fast(rec, prec)
    return ap, rec, prec, npos, details
