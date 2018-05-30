"""Detection utilites for using pcl-based pre-compiled segmentors.
"""
import cv2
import numpy as np
import os
import shutil
import subprocess

from perception import DepthImage, BinaryImage, CameraIntrinsics
from pcl_utils import mkdir_if_missing

def detect(pcl_detector, run_dir, dataset_dir, indices_arr, bin_mask_dir=None):
    """Run PCL-based detection on a depth-image-based dataset.

    Parameters
    ----------
    pcl_detector : PCLDetector
        Some pre-initialized PCL Detector.
    run_dir : str
        Directory to save outputs in. Output will be saved in pred_masks, pred_info,
        and modal_segmasks_processed subdirectories.
    dataset_dir : str
        Path to dataset. Should include depth_ims_numpy (.npy files)
        and modal_segmasks (.png files) as subdirectories.
    indices_arr : sequence of int
        Indices of images to perform detection on.
    bin_mask_dir : str
        Subdirectory of dataset_dir that contains binary masks for the bin.
        Should not be a full path, just the subdirectory name.
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

    # Input depth image data (numpy files, not .pngs)
    depth_dir = os.path.join(dataset_dir, 'depth_ims_numpy')

    # Input GT binary masks dir
    gt_mask_dir = os.path.join(dataset_dir, 'modal_segmasks')

    # Input binary mask data
    if bin_mask_dir:
        bin_mask_dir = os.path.join(dataset_dir, bin_mask_dir)

    # Input camera intrinsics
    camera_intrinsics_fn = os.path.join(dataset_dir, 'camera_intrinsics.intr')
    camera_intrs = CameraIntrinsics.load(camera_intrinsics_fn)

    ##################################################################
    # Process each image
    ##################################################################
    for index in indices_arr:
        base_name = 'image_{:06d}'.format(index)
        depth_image_fn = base_name + '.npy'

        # Extract depth image
        depth_data = np.load(os.path.join(depth_dir, depth_image_fn))
        depth_im = DepthImage(depth_data, camera_intrs.frame)

        # Mask out bin pixels if appropriate/necessary
        if bin_mask_dir:
            mask_im = BinaryImage.open(os.path.join(bin_mask_dir, base_name +'.png'), camera_intrs.frame)
            depth_im = depth_im.mask_binary(mask_im)

        # Run PCL detector
        pred_mask = pcl_detector.detect(depth_im, camera_intrs)

        # Save out ground-truth mask as array of shape (n, h, w)
        indiv_gt_masks = []
        gt_mask = cv2.imread(os.path.join(gt_mask_dir, base_name + '.png')).astype(np.uint8)[:,:,0]
        num_gt_masks = np.max(gt_mask)
        for i in range(1, num_gt_masks+1):
            indiv_gt_masks.append(gt_mask == i)
        gt_mask_output = np.stack(indiv_gt_masks)
        np.save(os.path.join(resized_segmask_dir, base_name + '.npy'), gt_mask_output)
        # Set up predicted masks and metadata
        indiv_pred_masks = []
        r_info = {
            'rois': [],
            'scores': [],
            'class_ids': [],
        }
        num_pred_masks = np.max(pred_mask)
        for i in range(1, num_pred_masks + 1):
            # Extract individual mask
            indiv_pred_mask = (pred_mask == i)
            indiv_pred_masks.append(indiv_pred_mask)

            # Compute bounding box, score, class_id
            nonzero_pix = np.nonzero(indiv_pred_mask)
            min_x, max_x = np.min(nonzero_pix[1]), np.max(nonzero_pix[1])
            min_y, max_y = np.min(nonzero_pix[0]), np.max(nonzero_pix[0])
            r_info['rois'].append([min_y, min_x, max_y, max_x])
            r_info['scores'].append(1.0)
            r_info['class_ids'].append(1)

        # Write the predicted masks and metadata
        pred_mask_output = np.stack(indiv_pred_masks).astype(np.uint8)
        np.save(os.path.join(pred_dir, base_name + '.npy'), pred_mask_output)
        np.save(os.path.join(pred_info_dir, base_name + '.npy'), r_info)

        if False: # Visualization of masks
            from visualization import Visualizer2D as vis
            print(num_pred_masks)
            bi = BinaryImage(pred_mask)
            bi._data = pred_mask * 10
            vis.figure()
            vis.imshow(bi)
            vis.show()

    print('Saved prediction masks to:\t {}'.format(pred_dir))
    print('Saved prediction info (bboxes, scores, classes) to:\t {}'.format(pred_info_dir))
    print('Saved transformed GT segmasks to:\t {}'.format(resized_segmask_dir))

    return pred_dir, pred_info_dir, resized_segmask_dir


class PCLDetector(object):
    """Base class for PCL-based object segmenters.
    """

    def __init__(self, executable, cache_dir='./.cache'):
        """Create a PCL detector.

        Parameters
        ----------
        executable : str
            Path to compiled executable that converts a depth image and camera
            intrinsics into a set of binary masks.
        cache_dir : str
            Temporary cache directory for subprocess IO.
        """
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self._executable = executable
        self._cache_dir = cache_dir
        self._depth_fn = os.path.join(self._cache_dir, '__depth.npy')
        self._mask_fn = os.path.join(self._cache_dir, '__mask.npy')
        self._intrs_fn = os.path.join(self._cache_dir, '__intrs.intr')
        self._arguments = [
            executable,
            '-d', self._depth_fn,
            '-m', self._mask_fn,
            '-i', self._intrs_fn
        ]

    def detect(self, depth_im, camera_intrs):
        """Perform detection on a depth image.

        Parameters
        ----------
        depth_im : perception.DepthImage
            A depth image to peform detection on.
        camera_intrs : perception.CameraIntrinsics
            Camera intrinsics corresponding to the depth image.

        Returns
        -------
        mask : (h,w) uint8
            Mask array where 0 is backround and each individual object is labelled
            with increasing numbers.
        """
        np.save(self._depth_fn, depth_im.data)
        camera_intrs.save(self._intrs_fn)
        subprocess.call(self._arguments)
        mask = np.load(self._mask_fn).astype(np.uint8)
        return mask

    def __del__(self):
        shutil.rmtree(self._cache_dir)

class EuclideanClusterExtractor(PCLDetector):
    """Runs Euclidean cluster extraction.
    """

    def __init__(self, executable,
                 min_cluster_size=500, max_cluster_size=1000000, tolerance=0.002):
        """Create a EuclideanClusterExtractor.

        Parameters
        ----------
        executable : str
            Path to compiled executable that converts a depth image and camera
            intrinsics into a set of binary masks.
        min_cluster_size : int
            Minimum cluster size.
        max_cluster_size : int
            Maximum cluster size.
        tolerance : float
            Max distance between closest points in same cluster.
        """
        super().__init__(executable)
        self._arguments.extend([
            '--min_cluster_size', str(min_cluster_size),
            '--max_cluster_size', str(max_cluster_size),
            '--tolerance', str(tolerance)
        ])

class RegionGrowingSegmentor(PCLDetector):
    """Runs region-growing segmentation.
    """

    def __init__(self, executable,
                 min_cluster_size=500, max_cluster_size=1000000, n_neighbors=5,
                 smoothness=10.0/180.0*np.pi, curvature=0.05):
        """Create a RegionGrowingSegmentor.

        Parameters
        ----------
        executable : str
            Path to compiled executable that converts a depth image and camera
            intrinsics into a set of binary masks.
        min_cluster_size : int
            Minimum cluster size.
        max_cluster_size : int
            Maximum cluster size.
        n_neighbors : int
            Number of neighbors to use for kNN.
        smoothness : float
            Max allowed angle in radians between nearby points in same cluster.
        curvature : float
            Max allowed curvature.
        """
        super().__init__(executable)
        self._arguments.extend([
            '--min_cluster_size', str(min_cluster_size),
            '--max_cluster_size', str(max_cluster_size),
            '--n_neighbors', str(n_neighbors),
            '--smoothness', str(smoothness),
            '--curvature', str(curvature)
        ])


if __name__ == '__main__':
    dataset_dir = '/nfs/diskstation/projects/dex-net/segmentation/datasets/real_test_cases_05_22/images'
    indices_arr = np.arange(0, 10)
    bin_mask_dir = 'segmasks_filled'
    run_dir = './run'

    pcl_detector = EuclideanClusterExtractor('./euclidean_cluster_extraction')
    #pcl_detector = RegionGrowingSegmentor('./region_growing_segmentation')
    detect(pcl_detector, run_dir, dataset_dir, indices_arr, bin_mask_dir=bin_mask_dir)
