"""Visualize PCL detection predictions.
"""
import numpy as np
import os
import visualize
from pcl_utils import mkdir_if_missing
from perception import DepthImage
import matplotlib.pyplot as plt

def visualize_predictions(run_dir, dataset_dir, pred_mask_dir, pred_info_dir, indices_arr):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, 'vis')
    depth_dir = os.path.join(dataset_dir, 'depth_ims_numpy')
    mkdir_if_missing(vis_dir)

    ##################################################################
    # Process each image
    ##################################################################
    print('VISUALIZING PREDICTIONS')
    for image_id in indices_arr:
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

