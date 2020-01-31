import os
import sys
import argparse
import time
from tqdm import tqdm
import numpy as np
import skimage.io as io
import skimage.transform
import imutils
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib as mpl
from matplotlib import patches
mpl.use('Agg')
import matplotlib.pyplot as plt

from autolab_core import YamlConfig

from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig
from sd_maskrcnn.dataset import TargetDataset, TargetStackDataset
from sd_maskrcnn.supplement_benchmark import s_benchmark

from mrcnn import model as modellib, utils as utilslib, visualize


def average_precision(retrievals, confidences, gt_index):
    """Calculates average precision COCO-style, 0.5:0.05:0.95.
    From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html.
    As there's only one positive label, recall can only be 0 or 1.
    Returns a single average precision metric."""
    p_r_s = []
    for thresh in np.arange(0.5, 1, 0.05):
        p_r_s.append(precision_recall(retrievals, confidences, gt_index, thresh))

    ap = 0
    last_r = 0
    for p, r in p_r_s:
        ap += (r - last_r) * p
        last_r = r
    assert ap <= 1.0
    return ap


def precision_recall(retrievals, confidences, gt_index, thresh):
    """Calculates precision and recall given a list of retrieved GT mask indices,
    confidence scores, and a single correct target mask index at a given threshold.

    Currently assumes exactly one target object per detection.

    Returns (precision, recall)."""
    filtered = [r for r, c in zip(retrievals, confidences) if c >= thresh]
    if gt_index in filtered:
        return (1 / len(filtered), 1)
    else:
        return (0, 0)


def iou(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2)) / np.sum(np.logical_or(mask1, mask2))


def benchmark(config):
    print("Benchmarking model.")

    # Create new directory for outputs
    output_dir = config['output_dir']
    utils.mkdir_if_missing(output_dir)

    # Save config in output directory
    config.save(os.path.join(output_dir, config['save_conf_name']))
    image_shape = config['model']['settings']['image_shape']
    config['model']['settings']['image_min_dim'] = min(image_shape)
    config['model']['settings']['image_max_dim'] = max(image_shape)

    inference_config = MaskConfig(config['model']['settings'])
    inference_config.GPU_COUNT = 1
    inference_config.IMAGES_PER_GPU = 1

    model_dir, _ = os.path.split(config['model']['path'])
    model = modellib.MaskRCNN(mode=config['model']['mode'], config=inference_config,
                              model_dir=model_dir)

    # Load trained weights
    print("Loading weights from ", config['model']['path'])
    model.load_weights_siamese(config['model']['path'],  config['model']['backbone_path'])

    # Create dataset
    test_dataset = TargetStackDataset(config, config['test']['path'],
                                      config['test']['tuples'],
                                      images=config['test']['images'],
                                      masks=config['test']['masks'],
                                      targets=config['test']['targets'],
                                      vis_file=config['test']['visibilities'])

    if config['test']['indices']:
        test_dataset.load(imset=config['test']['indices'])
    else:
        test_dataset.load()
    test_dataset.prepare()

    pred_mask_dir, pred_info_dir = detect(config['output_dir'], inference_config, model, test_dataset)

    calculate_statistics(output_dir, test_dataset, inference_config, pred_mask_dir, pred_info_dir)
    write_object_detections(output_dir, test_dataset, inference_config, pred_mask_dir, pred_info_dir)
    visualize_targets(output_dir, test_dataset, inference_config, pred_mask_dir, pred_info_dir)

    print('Saved benchmarking output to \t{}.\n'.format(config['output_dir']))


def benchmark_existing(config, pred_mask_dir, pred_info_dir):
    """Use an existing mask and info directory to calculate statistics and visualize targets."""
    print("Benchmarking model with existing infernece outputs.")

    # Create new directory for outputs
    output_dir = config['output_dir']
    utils.mkdir_if_missing(output_dir)
    image_shape = config['model']['settings']['image_shape']
    config['model']['settings']['image_min_dim'] = min(image_shape)
    config['model']['settings']['image_max_dim'] = max(image_shape)

    # Save config in output directory
    config.save(os.path.join(output_dir, config['save_conf_name']))
    inference_config = MaskConfig(config['model']['settings'])
    inference_config.GPU_COUNT = 1
    inference_config.IMAGES_PER_GPU = 1

    # Create dataset
    test_dataset = TargetStackDataset(config, config['test']['path'],
                                      config['test']['tuples'],
                                      images=config['test']['images'],
                                      masks=config['test']['masks'],
                                      targets=config['test']['targets'],
                                      vis_file=config['test']['visibilities'])

    if config['test']['indices']:
        test_dataset.load(imset=config['test']['indices'])
    else:
        test_dataset.load()
    test_dataset.prepare()

    calculate_statistics(output_dir, test_dataset, inference_config, pred_mask_dir, pred_info_dir)
    write_object_detections(output_dir, test_dataset, inference_config, pred_mask_dir, pred_info_dir)
    visualize_targets(output_dir, test_dataset, inference_config, pred_mask_dir, pred_info_dir)
    print('Saved benchmarking output to \t{}.\n'.format(config['output_dir']))



def detect(run_dir, inference_config, model, dataset):
    """
    Given a run directory, a MaskRCNN config object, a MaskRCNN model object,
    and a Dataset object, *all compatible with the Siamese branch*,
    - Makes predictions on images
    - Saves prediction masks in a certain directory
    - Saves other prediction info (scores, bboxes) in a separate directory

    Returns paths to directories for prediction masks, prediction info

    Continuing to hold over functionality from benchmark.py lest we choose to
    train from scratch and run all benchmarks at once.
    """
    full_pipeline_start = time.time()

    # Create subdirectory for prediction masks
    pred_dir = os.path.join(run_dir, 'pred_masks')
    utils.mkdir_if_missing(pred_dir)

    # Create subdirectory for prediction bboxes
    pred_info_dir = os.path.join(run_dir, 'pred_info')
    utils.mkdir_if_missing(pred_info_dir)

    image_ids = dataset.example_indices
    print('MAKING PREDICTIONS')

    for image_id in tqdm(image_ids):
        time_start = time.time()

        (pile_img, _, _, bbox, masks), (target_imgs, target_bbs, _, target_vector) \
            = modellib.load_inputs_gt(dataset, inference_config, image_id)

        target_mask = masks[:,:,np.argmax(target_vector)]
        target_pile_bb = bbox[np.argmax(target_vector)]
        r = model.detect(images=[pile_img], targets=[target_imgs], target_bbs=[target_bbs])
        r = r[0]

        # Save masks
        save_masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])]) if np.any(r['masks']) else np.array([])
        save_masks_path = os.path.join(pred_dir, 'example_{:06d}.npy'.format(image_id))
        np.save(save_masks_path, save_masks)

        pipeline_step_time = time.time() - time_start
        # Save info
        r_info = {
            'rois': r['rois'],
            'scores': r['scores'],
            'class_ids': r['class_ids'],
            'target_probs': r['target_probs'],
            'rois_prefilter': r['rois_prefilter'],
            'scores_prefilter': r['scores_prefilter'],
            'class_ids_prefilter': r['class_ids_prefilter'],
            'target_probs_prefilter': r['target_probs_prefilter'],
            'time': r['time'],
            'pipeline_step_time': pipeline_step_time
        }
        r_info_path = os.path.join(pred_info_dir, 'example_{:06d}.npy'.format(image_id))
        np.save(r_info_path, r_info)

    print('Saved prediction masks to:\t {}'.format(pred_dir))
    print('Saved prediction info (bboxes, scores, classes, target probs) to:\t {}'.format(pred_info_dir))

    full_pipeline_time = time.time() - full_pipeline_start
    print('full detection pipeline time', full_pipeline_time)

    return pred_dir, pred_info_dir


def normalize(color_im, crop_size=(512, 512)):
    """Centers and enlarges object detection masks.
    Taken from https://github.com/BerkeleyAutomation/perception/blob/mmatl/semantic_grasping_experiments/tools/generate_siamese_dataset.py."""

    # Center object in frame
    color_data = color_im

    nonzero_px = np.where(np.sum(color_im, axis=2) > 0)
    nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]].astype(np.int32)

    centroid = np.mean(nonzero_px, axis=0)
    cx, cy = color_data.shape[1] // 2, color_data.shape[0] // 2
    color_data = imutils.translate(color_data, cx - round(centroid[1]), cy - round(centroid[0]))

    # Crop about center to 256x256
    crop_x = crop_size[0] / 4
    crop_y = crop_size[1] / 4


    img = color_data
    x1, y1, x2, y2 = cx-crop_x, cy-crop_y, cx+crop_x, cy+crop_y
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = cv2.copyMakeBorder(img, -min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=BLACK)
    color_data = img[y1:y2, x1:x2, :]


    # resize to 512x512 to get a 2x scale increase
    color_data = skimage.transform.resize(color_data, (crop_size[1], crop_size[0], 3), clip=False, preserve_range=True).astype(np.uint8)

    return color_data


def write_object_detections(run_dir, dataset, inference_config, pred_mask_dir, pred_info_dir):
    """Writes separate images for each object detection per pile, crops and centers."""

    # Create subdirectory for masked predicted objects
    pred_obj_dir = os.path.join(run_dir, 'pred_objs')
    pred_roi_dir = os.path.join(run_dir, 'pred_obj_rois')

    image_ids = dataset.example_indices
    print('WRITING OBJECT DETECTION IMAGES')

    for image_id in tqdm(image_ids):
        (pile_img, _, _, bbox, masks), (target_imgs, target_bbs, _, target_vector) \
            = modellib.load_inputs_gt(dataset, inference_config, image_id)

        # load masks and info
        r = np.load(os.path.join(pred_info_dir, 'example_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'example_{:06}.npy'.format(image_id)))
        rois = r['rois']

        # Save object detection masks
        pile_dir = os.path.join(pred_obj_dir, 'example_{:06d}'.format(image_id))
        utils.mkdir_if_missing(pile_dir)
        for i in range(r_masks.shape[0]):
            pile_copy = np.array(pile_img)[:,:,:3] # RGB only, depth becomes alpha channel
            pred_mask = np.logical_not(r_masks[i,:,:]) # all non-mask pixels are True
            pile_copy[pred_mask] = 0

            pile_copy = normalize(pile_copy)

            obj_name = os.path.join(pile_dir, 'det_{:06d}.png'.format(i))
            io.imsave(obj_name, pile_copy, check_contrast=False)

        # Save object detection bounding boxes
        pile_roi_dir = os.path.join(pred_roi_dir, 'example_{:06d}'.format(image_id))
        utils.mkdir_if_missing(pile_roi_dir)
        for i in range(rois.shape[0]):
            pile_copy = np.array(pile_img)[:,:,:3] # RGB only, depth becomes alpha channel
            y1, x1, y2, x2 = rois[i,:]
            mask = np.zeros_like(pile_copy)
            mask[y1:y2,x1:x2] = 1
            pile_copy = pile_copy * mask # pointwise multiplication to keep bbox pixels only

            pile_copy = normalize(pile_copy)
            obj_name = os.path.join(pile_roi_dir, 'det_{:06d}.png'.format(i))
            io.imsave(obj_name, pile_copy, check_contrast=False)


def calculate_statistics(run_dir, dataset, inference_config, pred_mask_dir, pred_info_dir):
    """Calculates statistics (mean IoU, precision & recall)"""
    print('CALCULATING STATISTICS')
    # Create subdirectory for prediction visualizations
    image_ids = dataset.example_indices

    max_probs = np.zeros_like(dataset.example_indices, dtype=np.float)

    ### Top-n IoU
    n_s = [1, 2, 3] #SET

    max_top_n_ious = {n: np.zeros_like(dataset.example_indices, dtype=np.float) for n in n_s}
    max_top_n_indices = {n: np.zeros_like(dataset.example_indices, dtype=np.int) for n in n_s}

    correct_targets_masks = np.zeros_like(dataset.example_indices, dtype=np.int)

    average_precisions = np.zeros_like(dataset.example_indices, dtype=np.float)

    ### TODO: REFACTOR TO USE utilslib FUNCTIONS FOR MASK OVERLAPS ###
    def top_n_iou(pred_masks, target_mask, target_probs, n):
        """Given a set of prediction masks and a target mask, returns the
        argmax and maximum prediction-target IoU over the top n predictions by predicted
        probability.
        """
        target_probs = target_probs[:]
        top_n_indices = np.argsort(target_probs)[-n:] # grab indices of top n elements
        top_n_ious = [iou(target_mask, r_masks[i,:,:]) for i in top_n_indices]
        return np.argmax(top_n_ious), np.max(top_n_ious)


    for image_id in tqdm(image_ids):
        (pile_img, _, _, gt_bbox, gt_masks), (_, _, _, target_vector) \
            = modellib.load_inputs_gt(dataset, inference_config, image_id)

        example_meta = dataset.load_example_metadata(image_id)

        target_mask = gt_masks[:,:,np.argmax(target_vector)]

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'example_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'example_{:06}.npy'.format(image_id)))

        target_probs = r['target_probs'][:,1]

        for n in n_s:
            max_index, max_iou = top_n_iou(r_masks, target_mask,
                                            target_probs, n)
            max_top_n_ious[n][image_id] = max_iou
            max_top_n_indices[n][image_id] = max_index

        # Obtain pred-GT mask matrix of IoUs
        pred_masks = np.transpose(r_masks, axes=(1,2,0))
        iou_matrix = utilslib.compute_overlaps_masks(pred_masks, gt_masks) # k preds x n GTs
        retrievals = np.argmax(iou_matrix, axis=1)

        pred_index = np.argmax(target_probs)
        gt_index = np.argmax(target_vector)

        ###### Precision, Recall ######
        ap = average_precision(retrievals, target_probs, gt_index)
        average_precisions[image_id] = ap

        # pred_target_mask = r_masks[pred_index:pred_index+1,:,:] # shape (1, H, W)
        # pred_target_mask = np.transpose(pred_target_mask, axes=(1,2,0)) # shape (H, W, 1)

        # mask_ious = np.squeeze(utilslib.compute_overlaps_masks(gt_masks, pred_target_mask))
        target_mask_ious = iou_matrix[pred_index,:]

        if np.argmax(target_mask_ious) == gt_index:
            correct_targets_masks[image_id] = 1


    print('Correct target (by mask IoU) for {} out of {} cases'.format(np.sum(correct_targets_masks), len(image_ids)))
    print('mAP[0.5:0.05:0.95]:', np.mean(average_precisions))

    # Print top-n ious
    for n in n_s:
        print('Mean top-{} iou:'.format(n), np.mean(max_top_n_ious[n]))

    # Save
    for n in n_s:
        iou_path = os.path.join(pred_info_dir, 'max_top_{}_ious.npy'.format(n))
        indices_path = os.path.join(pred_info_dir, 'max_top_{}_indices.npy'.format(n))
        np.save(iou_path, max_top_n_ious[n])
        np.save(indices_path, max_top_n_indices[n])

    np.save(os.path.join(pred_info_dir, 'correct_targets_masks.npy'), correct_targets_masks)
    np.save(os.path.join(pred_info_dir, 'average_precisions.npy'), average_precisions)

    # Histogram of IoU values
    for n in n_s:
        plt.hist(max_top_n_ious[n])
        plt.title('Mean top-{} iou: {}'.format(n, np.mean(max_top_n_ious[n])))
        hist_path = os.path.join(pred_info_dir, 'hist_top_{}_iou.png'.format(n))
        plt.savefig(hist_path)
        plt.close()

    print('Saved statistics to:\t {}'.format(pred_info_dir))


def plot_detailed_predictions(file_name, pile_img, target_img, gt_masks, gt_bbs, target_vector,
                              pred_masks, pred_bbs, pred_target_probs,
                              obj_class='', obj_vis=0):
    assert pile_img.shape[-1] == target_img.shape[-1], "channel count must match"

    depth_channel = (pile_img.shape[-1] == 4)

    num_preds = len(pred_bbs)

    # cropping & splitting
    py1, px1, py2, px2 = 110, 90, 350, 410
    pile_rgb = pile_img[py1:py2,px1:px2,:3]

    target_rgb = target_img[:,:,:3]
    ty1, tx1, ty2, tx2 = utilslib.extract_bboxes(np.sum(target_rgb, axis=2)[:,:,np.newaxis] != 0)[0]
    target_rgb = target_img[ty1:ty2,tx1:tx2,:3]


    pred_masks = np.transpose(pred_masks, axes=(1,2,0))
    gt_target_mask = gt_masks[:,:,np.argmax(target_vector):np.argmax(target_vector)+1]

    pred_mask_ious = np.squeeze(utilslib.compute_overlaps_masks(pred_masks, gt_target_mask))

    fig, ax = plt.subplots(num_preds + 2, 2, figsize=(12, 6 * (num_preds + 2)))

    ax[0][0].imshow(target_rgb)
    ax[1][0].imshow(pile_rgb)

    # adjust for crop
    gt_target_pile_bb = gt_bbs[np.argmax(target_vector)] - np.array([py1, px1, py1, px1])

    ax[1][0].add_patch(patches.Rectangle((gt_target_pile_bb[1], gt_target_pile_bb[0]),
                                        gt_target_pile_bb[3] - gt_target_pile_bb[1],
                                        gt_target_pile_bb[2] - gt_target_pile_bb[0], linewidth=2,
                                      alpha=0.8,
                                      edgecolor='red', facecolor='none'))
    if depth_channel:
        pile_d = pile_img[py1:py2,px1:px2,3]
        target_d = target_img[ty1:ty2,tx1:tx2,3]
        ax[0][1].imshow(target_d, cmap='gray_r')
        ax[1][1].imshow(pile_d, cmap='gray_r')

        ax[1][1].add_patch(patches.Rectangle((gt_target_pile_bb[1], gt_target_pile_bb[0]),
                                             gt_target_pile_bb[3] - gt_target_pile_bb[1],
                                             gt_target_pile_bb[2] - gt_target_pile_bb[0],
                                             linewidth=2,
                                             alpha=0.8,
                                             edgecolor='red', facecolor='none'))

    ax[1][0].set_title('GT target - {}'.format(obj_class))
    ax[1][1].set_title('Visibility: {:.3f}'.format(obj_vis))

    target_plot_indices = np.argsort(np.argsort(pred_target_probs[:,1] * -1))

    for k in range(num_preds):
        j = target_plot_indices[k] + 2
        # print('pred', k, 'plot index', j, 'prob', pred_target_probs[k,1])
        # adjust for crop
        pred_bb = pred_bbs[k,:] - np.array([py1, px1, py1, px1])
        ax[j][0].imshow(pile_rgb)
        ax[j][0].add_patch(patches.Rectangle((pred_bb[1], pred_bb[0]), pred_bb[3] - pred_bb[1],
                                          pred_bb[2] - pred_bb[0], linewidth=2, alpha=0.8,
                                             edgecolor='red', linestyle='dashed',
                                             facecolor='none'))
        ax[j][0].set_title("Mask IoU: {:.3f}".format(pred_mask_ious[k]))
        ax[j][1].set_title("Target Confidence: {:.3f}".format(pred_target_probs[k,1]))

        if depth_channel:
            ax[j][1].imshow(pile_d, cmap='gray_r')
            ax[j][1].add_patch(patches.Rectangle((pred_bb[1], pred_bb[0]),
                                                 pred_bb[3] - pred_bb[1],
                                                 pred_bb[2] - pred_bb[0],
                                                 linewidth=2, alpha=0.8,
                                                 edgecolor='red', linestyle='dashed',
                                                 facecolor='none'))

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_predictions(file_name, pile_img, target_img, gt_masks, gt_bbs, target_vector,
                     pred_masks, pred_bbs, pred_target_probs, figsize=(14,6)):
    # If 4 channels, remove depth channel (assuming RGBD)
    if pile_img.ndim == 3 and pile_img.shape[-1] > 3:
        pile_img = pile_img[:,:,:3]
    if target_img.ndim == 3 and target_img.shape[-1] > 3:
        target_img = target_img[:,:,:3]

    # If 1 channel, squeeze
    if pile_img.ndim == 1 and pile_img.shape[-1] == 1:
        pile_img = pile_img[:,:,0]
    if target_img.ndim == 1 and target_img.shape[-1] == 1:
        target_img = target_img[:,:,0]

    target_pile_bb = gt_bbs[np.argmax(target_vector)]
    pred_index = np.argmax(pred_target_probs[:,1])
    pred_bb = pred_bbs[pred_index]

    _, axes = plt.subplots(1, 2, figsize=figsize)

    if target_img.ndim == 3 and target_img.shape[-1] == 1:
        target_img = target_img.squeeze()
    if target_img.ndim == 3 and target_img.shape[-1] == 4:
        target_img = target_img[:,:,:3] # don't show depth channel
    if target_img.ndim == 2:
        target_img = np.stack([target_img] * 3, axis=2)
    axes[0].imshow(target_img)


    if pile_img.ndim == 3 and pile_img.shape[-1] == 1:
        pile_img = pile_img.squeeze()
    if pile_img.ndim == 3 and pile_img.shape[-1] == 4:
        pile_img = pile_img[:,:,:3] # don't show depth channel
    if pile_img.ndim == 2:
        pile_img = np.stack([pile_img] * 3, axis=2)


    pred_masks_t = np.transpose(pred_masks, axes=[1, 2, 0])
    visualize.display_instances(pile_img, pred_bbs, pred_masks_t,
                                np.array([1] * len(pred_target_probs[:,1])),
                                ['', ''], pred_target_probs[:,1], figsize=figsize, ax=axes[1],
                                show_bbox=False, show_class=False)

    pred_bb_patch = patches.Rectangle((pred_bb[1], pred_bb[0]), pred_bb[3] - pred_bb[1],
                                      pred_bb[2] - pred_bb[0], linewidth=2,
                                      linestyle='dashed', alpha=0.8,
                                      edgecolor='cyan', facecolor='none')

    target_bb_patch = patches.Rectangle((target_pile_bb[1], target_pile_bb[0]),
                                        target_pile_bb[3] - target_pile_bb[1],
                                        target_pile_bb[2] - target_pile_bb[0], linewidth=2,
                                      alpha=0.8,
                                      edgecolor='red', facecolor='none')

    axes[1].add_patch(pred_bb_patch)
    axes[1].add_patch(target_bb_patch)
    plt.title('Max prob: {:06f}'.format(max(pred_target_probs[:,1])))
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_targets(run_dir, dataset, inference_config, pred_mask_dir, pred_info_dir):
    """Visualizes target predictions"""
    # Create subdirectory for prediction visualizations
    print('VISUALIZING TARGETS')
    vis_dir = os.path.join(run_dir, 'vis')
    utils.mkdir_if_missing(vis_dir)

    image_ids = dataset.example_indices

    for image_id in tqdm(image_ids):
        (pile_img, _, _, bbox, masks), (target_img, _, _, target_vector) \
            = modellib.load_inputs_gt(dataset, inference_config, image_id)

        example_meta = dataset.load_example_metadata(image_id)

        target_mask = masks[:,:,np.argmax(target_vector)]
        target_pile_bb = bbox[np.argmax(target_vector)]

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'example_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'example_{:06}.npy'.format(image_id)))

        pred_index = np.argmax(r['target_probs'][:,1])
        pred_mask = r_masks[pred_index,:,:] # Predicted masks are always saved in [N, H, W] instead of [H, W, N]

        pred_bb = r['rois'][pred_index]

        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))

        # if we have a stack, display first image in stack
        if len(target_img.shape) == 4:
            target_img = target_img[3,:,:,:]
        plot_detailed_predictions(file_name, pile_img, target_img, masks, bbox, target_vector,
                                  r_masks, r['rois'], r['target_probs'],
                                  obj_class=example_meta['obj_class'],
                                  obj_vis=example_meta['obj_visibility'])

    print('Saved prediction visualizations to:\t {}'.format(vis_dir))

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark Siamese SD Mask RCNN model")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark_siamese.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_parser.add_argument("--pred_masks", type=str, help="path to pred mask directory")
    conf_parser.add_argument("--pred_info", type=str, help="path to pred info directory")
    conf_args = conf_parser.parse_args()

    print(conf_args)
    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    if conf_args.pred_masks and conf_args.pred_info:
        benchmark_existing(config, conf_args.pred_masks, conf_args.pred_info)
    elif conf_args.pred_masks or conf_args.pred_info:
        raise ValueError("Need both masks dir and info dir")
    else:
        benchmark(config)
    # utils.set_tf_config()
