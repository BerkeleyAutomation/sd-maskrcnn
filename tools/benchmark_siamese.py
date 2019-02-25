import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import skimage.io as io
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

from autolab_core import YamlConfig

from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig
from sd_maskrcnn.dataset import TargetDataset
from sd_maskrcnn.coco_benchmark import coco_benchmark
from sd_maskrcnn.supplement_benchmark import s_benchmark

from mrcnn import model as modellib, utils as utilslib, visualize


def iou(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2)) / np.sum(np.logical_or(mask1, mask2))


def benchmark(config):
    print("Benchmarking model.")

    # Create new directory for outputs
    output_dir = config['output_dir']
    utils.mkdir_if_missing(output_dir)

    # Save config in output directory
    config.save(os.path.join(output_dir, config['save_conf_name']))
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
    test_dataset = TargetDataset(config['test']['path'], images=config['test']['images'],
                                 masks=config['test']['masks'], targets=config['test']['targets'])

    if config['test']['indices']:
        test_dataset.load(imset=os.path.join(config['test']['path'], config['test']['indices']))
    else:
        test_dataset.load()
    test_dataset.prepare()

    pred_mask_dir, pred_info_dir = detect(config['output_dir'], inference_config, model, test_dataset)

    calculate_statistics(output_dir, test_dataset, inference_config, pred_mask_dir, pred_info_dir)
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
    # Create subdirectory for prediction masks
    pred_dir = os.path.join(run_dir, 'pred_masks')
    utils.mkdir_if_missing(pred_dir)

    # Create subdirectory for prediction bboxes
    pred_info_dir = os.path.join(run_dir, 'pred_info')
    utils.mkdir_if_missing(pred_info_dir)

    image_ids = dataset.image_ids
    indices = dataset.indices
    print('MAKING PREDICTIONS')

    for image_id in tqdm(image_ids):

        (pile_img, _, _, bbox, masks), (target_img, target_bb, _, target_vector) \
            = modellib.load_inputs_gt(dataset, inference_config, image_id)

        target_mask = masks[:,:,np.argmax(target_vector)]
        target_pile_bb = bbox[np.argmax(target_vector)]

        r = model.detect([pile_img], [target_img], [target_bb])
        r = r[0]

        # Save masks
        save_masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])]) if np.any(r['masks']) else np.array([])
        save_masks_path = os.path.join(pred_dir, 'example_{:06d}.npy'.format(image_id))
        np.save(save_masks_path, save_masks)

        # Save info
        r_info = {
            'rois': r['rois'],
            'scores': r['scores'],
            'class_ids': r['class_ids'],
            'target_probs': r['target_probs'],
        }
        r_info_path = os.path.join(pred_info_dir, 'example_{:06d}.npy'.format(image_id))
        np.save(r_info_path, r_info)

    print('Saved prediction masks to:\t {}'.format(pred_dir))
    print('Saved prediction info (bboxes, scores, classes, target probs) to:\t {}'.format(pred_info_dir))

    return pred_dir, pred_info_dir


def calculate_statistics(run_dir, dataset, inference_config, pred_mask_dir, pred_info_dir):
    """Calculates statistics (mean IoU, precision & recall)"""
    print('CALCULATING STATISTICS')
    # Create subdirectory for prediction visualizations
    image_ids = dataset.image_ids

    ious = np.zeros_like(dataset.image_ids, dtype=np.float)
    max_probs = np.zeros_like(dataset.image_ids, dtype=np.float)

    max_top_n_ious = np.zeros_like(dataset.image_ids, dtype=np.float)
    max_top_n_indices = np.zeros_like(dataset.image_ids, dtype=np.int)

    for image_id in tqdm(image_ids):
        (pile_img, _, _, bbox, masks), (_, _, _, target_vector) \
            = modellib.load_inputs_gt(dataset, inference_config, image_id)

        target_mask = masks[:,:,np.argmax(target_vector)]

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'example_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'example_{:06}.npy'.format(image_id)))

        target_probs = r['target_probs'][:,1]

        pred_index = np.argmax(target_probs)
        pred_mask = r_masks[pred_index,:,:] # Predicted masks are always saved in [N, H, W] instead of [H, W, N]

        ious[image_id] = iou(target_mask, pred_mask)
        max_probs[image_id] = np.max(target_probs)

        ### Top-n IoU
        n = 3
        top_n_indices = target_probs.argsort()[-n:]
        top_n_ious = [iou(target_mask, r_masks[i,:,:]) for i in top_n_indices]

        max_top_n_ious[image_id] = max(top_n_ious)
        max_top_n_indices[image_id] = np.argmax(top_n_ious)


    ious_path = os.path.join(pred_info_dir, 'ious.npy')
    max_probs_path = os.path.join(pred_info_dir, 'max_probs.npy')

    max_top_n_ious_path = os.path.join(pred_info_dir, 'max_top_n_ious.npy')
    max_top_n_indices_path = os.path.join(pred_info_dir, 'max_top_n_indices.npy')

    np.save(ious_path, ious)
    np.save(max_probs_path, max_probs)
    np.save(max_top_n_ious, max_top_n_ious_path)
    np.save(max_top_n_indices, max_top_n_indices_path)

    mean_iou = np.mean(ious)

    # Histogram of IoU values
    plt.hist(ious)
    plt.title('IoU, mean IoU: {}'.format(mean_iou))

    hist_path = os.path.join(pred_info_dir, 'hist_iou.png')
    plt.savefig(hist_path)
    plt.close()

    # Plotting IoU vs. Prediction Probability
    plt.scatter(ious, max_probs)
    plt.title('IoU vs. Predicted Prob')

    plot_path = os.path.join(pred_info_dir, 'plot_max_prob.png')
    plt.savefig(plot_path)
    plt.close()

    print('Mean IoU {}'.format(mean_iou))
    print('Saved statistics to:\t {}'.format(pred_info_dir))


def plot_predictions(file_name, pile_img, target_mask, target_pile_bb, pred_mask, pred_bb, figsize=(6,6)):
    from matplotlib import patches

    _, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(pile_img)

    pred_mask_masked_array = np.ma.masked_where(np.logical_not(pred_mask), pred_mask)
    target_mask_masked_array = np.ma.masked_where(np.logical_not(target_mask), target_mask)

    ax.imshow(target_mask_masked_array, cmap='autumn', alpha=0.5)
    ax.imshow(pred_mask_masked_array, cmap='cool', alpha=0.5)

    pred_bb_patch = patches.Rectangle((pred_bb[1], pred_bb[0]), pred_bb[3] - pred_bb[1],
                                      pred_bb[2] - pred_bb[0], linewidth=2,
                                      linestyle='dashed', alpha=0.8,
                                      edgecolor='cyan', facecolor='none')

    target_bb_patch = patches.Rectangle((target_pile_bb[1], target_pile_bb[0]),
                                        target_pile_bb[3] - target_pile_bb[1],
                                        target_pile_bb[2] - target_pile_bb[0], linewidth=2,
                                      alpha=0.8,
                                      edgecolor='red', facecolor='none')

    ax.add_patch(pred_bb_patch)
    ax.add_patch(target_bb_patch)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_targets(run_dir, dataset, inference_config, pred_mask_dir, pred_info_dir):
    """Visualizes target predictions"""
    # Create subdirectory for prediction visualizations
    print('VISUALIZING TARGETS')
    vis_dir = os.path.join(run_dir, 'vis')
    utils.mkdir_if_missing(vis_dir)

    image_ids = dataset.image_ids

    for image_id in tqdm(image_ids):
        (pile_img, _, _, bbox, masks), (_, _, _, target_vector) \
            = modellib.load_inputs_gt(dataset, inference_config, image_id)

        target_mask = masks[:,:,np.argmax(target_vector)]
        target_pile_bb = bbox[np.argmax(target_vector)]

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'example_{:06}.npy'.format(image_id))).item()
        r_masks = np.load(os.path.join(pred_mask_dir, 'example_{:06}.npy'.format(image_id)))

        pred_index = np.argmax(r['target_probs'][:,1])
        pred_mask = r_masks[pred_index,:,:] # Predicted masks are always saved in [N, H, W] instead of [H, W, N]
        pred_bb = r['rois'][pred_index]

        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))

        plot_predictions(file_name, pile_img, target_mask, target_pile_bb, pred_mask, pred_bb)

    print('Saved prediction visualizations to:\t {}'.format(vis_dir))

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark Siamese SD Mask RCNN model")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark_siamese.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    # utils.set_tf_config()
    benchmark(config)
