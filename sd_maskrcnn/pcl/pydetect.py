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

Detection utilites for using pcl-based pre-compiled segmentors.
"""
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import logging

try:
    import pcl
except ImportError:
    logging.warning('Failed to import python-pcl! Point cloud filtering will not be available.')
    logging.info('python-pcl can be installed from https://github.com/strawlab/python-pcl')
    sys.exit()

from perception import DepthImage, BinaryImage, CameraIntrinsics
from autolab_core import PointCloud, YamlConfig

from sd_maskrcnn.utils import mkdir_if_missing

def detect(detector_type, config, run_dir, test_config):
    """Run PCL-based detection on a depth-image-based dataset.

    Parameters
    ----------
    config : dict
        config for a PCL detector
    run_dir : str
        Directory to save outputs in. Output will be saved in pred_masks, pred_info,
        and modal_segmasks_processed subdirectories.
    test_config : dict
        config containing dataset information
    """

    ##################################################################
    # Set up output directories
    ##################################################################

    # Create subdirectory for prediction masks
    pred_dir = os.path.join(run_dir, 'pred_masks')
    mkdir_if_missing(pred_dir)

    # Create subdirectory for prediction scores & bboxes
    pred_info_dir = os.path.join(run_dir, 'pred_info')
    mkdir_if_missing(pred_info_dir)

    # Create subdirectory for transformed GT segmasks
    resized_segmask_dir = os.path.join(run_dir, 'modal_segmasks_processed')
    mkdir_if_missing(resized_segmask_dir)

    ##################################################################
    # Set up input directories
    ##################################################################

    dataset_dir = test_config['path']
    indices_arr = np.load(os.path.join(dataset_dir, test_config['indices']))

    # Input depth image data (numpy files, not .pngs)
    depth_dir = os.path.join(dataset_dir, test_config['images'])

    # Input GT binary masks dir
    gt_mask_dir = os.path.join(dataset_dir, test_config['masks'])

    # Input binary mask data
    if 'bin_masks' in test_config.keys():
        bin_mask_dir = os.path.join(dataset_dir, test_config['bin_masks'])

    # Input camera intrinsics
    camera_intrinsics_fn = os.path.join(dataset_dir, 'camera_intrinsics.intr')
    camera_intrs = CameraIntrinsics.load(camera_intrinsics_fn)

    image_ids = np.arange(indices_arr.size)

    ##################################################################
    # Process each image
    ##################################################################
    for image_id in tqdm(image_ids):
        base_name = 'image_{:06d}'.format(indices_arr[image_id])
        output_name = 'image_{:06d}'.format(image_id)
        depth_image_fn = base_name + '.npy'

        # Extract depth image
        depth_data = np.load(os.path.join(depth_dir, depth_image_fn))
        depth_im = DepthImage(depth_data, camera_intrs.frame)
        depth_im = depth_im.inpaint(0.25)

        # Mask out bin pixels if appropriate/necessary
        if bin_mask_dir:
            mask_im = BinaryImage.open(os.path.join(bin_mask_dir, base_name +'.png'), camera_intrs.frame)
            mask_im = mask_im.resize(depth_im.shape[:2])
            depth_im = depth_im.mask_binary(mask_im)

        point_cloud = camera_intrs.deproject(depth_im)
        point_cloud.remove_zero_points()
        pcl_cloud = pcl.PointCloud(point_cloud.data.T.astype(np.float32))
        tree = pcl_cloud.make_kdtree()
        if detector_type == 'euclidean':
            segmentor = pcl_cloud.make_EuclideanClusterExtraction()
            segmentor.set_ClusterTolerance(config['tolerance'])
        elif detector_type == 'region_growing':
            segmentor = pcl_cloud.make_RegionGrowing(ksearch=50)
            segmentor.set_NumberOfNeighbours(config['n_neighbors'])
            segmentor.set_CurvatureThreshold(config['curvature'])
            segmentor.set_SmoothnessThreshold(config['smoothness'])
        else:
            print('PCL detector type not supported')
            exit()

        segmentor.set_MinClusterSize(config['min_cluster_size'])
        segmentor.set_MaxClusterSize(config['max_cluster_size'])
        segmentor.set_SearchMethod(tree)
        cluster_indices = segmentor.Extract()

        # Set up predicted masks and metadata
        indiv_pred_masks = []
        r_info = {
            'rois': [],
            'scores': [],
            'class_ids': [],
        }

        for i,cluster in enumerate(cluster_indices):
            points = pcl_cloud.to_array()[cluster]
            indiv_pred_mask = camera_intrs.project_to_image(PointCloud(points.T, frame=camera_intrs.frame)).to_binary()
            indiv_pred_mask.data[indiv_pred_mask.data>0] = 1
            indiv_pred_masks.append(indiv_pred_mask.data)

            # Compute bounding box, score, class_id
            nonzero_pix = np.nonzero(indiv_pred_mask.data)
            min_x, max_x = np.min(nonzero_pix[1]), np.max(nonzero_pix[1])
            min_y, max_y = np.min(nonzero_pix[0]), np.max(nonzero_pix[0])
            r_info['rois'].append([min_y, min_x, max_y, max_x])
            r_info['scores'].append(1.0)
            r_info['class_ids'].append(1)

        r_info['rois'] = np.array(r_info['rois'])
        r_info['scores'] = np.array(r_info['scores'])
        r_info['class_ids'] = np.array(r_info['class_ids'])

        # Write the predicted masks and metadata
        if indiv_pred_masks:
            pred_mask_output = np.stack(indiv_pred_masks).astype(np.uint8)
        else:
            pred_mask_output = np.array(indiv_pred_masks).astype(np.uint8)

        # Save out ground-truth mask as array of shape (n, h, w)
        indiv_gt_masks = []
        gt_mask = cv2.imread(os.path.join(gt_mask_dir, base_name + '.png'))
        gt_mask = cv2.resize(gt_mask, (depth_im.shape[1], depth_im.shape[0])).astype(np.uint8)[:,:,0]
        num_gt_masks = np.max(gt_mask)
        for i in range(1, num_gt_masks+1):
            indiv_gt_mask = (gt_mask == i)
            if np.any(indiv_gt_mask):
                indiv_gt_masks.append(gt_mask == i)
        gt_mask_output = np.stack(indiv_gt_masks)

        np.save(os.path.join(resized_segmask_dir, output_name + '.npy'), gt_mask_output)
        np.save(os.path.join(pred_dir, output_name + '.npy'), pred_mask_output)
        np.save(os.path.join(pred_info_dir, output_name + '.npy'), r_info)

    print('Saved prediction masks to:\t {}'.format(pred_dir))
    print('Saved prediction info (bboxes, scores, classes) to:\t {}'.format(pred_info_dir))
    print('Saved transformed GT segmasks to:\t {}'.format(resized_segmask_dir))

    return pred_dir, pred_info_dir, resized_segmask_dir


if __name__ == '__main__':
    dataset_dir = '/nfs/diskstation/projects/dex-net/segmentation/datasets/real_test_cases_06_07_18/phoxi/images'
    run_dir = './run'

    # For euclidean
    config = {'min_cluster_size': 800, 'max_cluster_size': 1000000, 'tolerance': 0.002}
    test_config = {'path': dataset_dir, 'images': 'depth_ims_numpy', 'masks': 'modal_segmasks', 'indices': 'sample_test.npy', 'bin_masks': 'segmasks_filled'}
    # For region_growing
    # config = {'min_cluster_size': 800, 'max_cluster_size': 1000000, 'n_neighbors': 5, 'curvature': 0.095, 'smoothness': 0.200}

    detect('euclidean', config, run_dir, test_config)
