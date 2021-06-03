"""Detection utilites for using gop/mcg pre-compiled segmentors.
"""
import cv2
import numpy as np
import os
import scipy.io as scio
from tqdm import tqdm

from autolab_core import BinaryImage

from .src.gop import segmentation, proposals, contour, imgproc
from .src.util import setupLearned

from sd_maskrcnn.utils import mkdir_if_missing

def detect(detector_type, config, run_dir, test_config):

    """Run RGB Object Proposal-based detection on a color-image-based dataset.

    Parameters
    ----------
    detector_type : str
        type of detector (either mcg or gop)
    config : dict
        config for a GOP/MCG detector
    run_dir : str
        Directory to save outputs in. Output will be saved in pred_masks, pr$
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
    rgb_dir = os.path.join(dataset_dir, test_config['images'])

    # Input GT binary masks dir
    gt_mask_dir = os.path.join(dataset_dir, test_config['masks'])

    # Input binary mask data
    if 'bin_masks' in test_config.keys():
        bin_mask_dir = os.path.join(dataset_dir, test_config['bin_masks'])

    image_ids = np.arange(indices_arr.size)

    ##################################################################
    # Process each image
    ##################################################################
    for image_id in tqdm(image_ids):
        base_name = 'image_{:06d}'.format(indices_arr[image_id])
        output_name = 'image_{:06d}'.format(image_id)
        rgb_image_fn = os.path.join(rgb_dir, base_name + '.png')

        # Run GOP detector
        if detector_type == 'gop':
            detector = GOP()
        elif detector_type == 'mcg':
            mcg_dir = os.path.join(dataset_dir, 'mcg', config['mode'])
            detector = MCG(mcg_dir, nms_thresh=config['nms_thresh'])

        pred_mask = detector.detect(rgb_image_fn)

        # Save out ground-truth mask as array of shape (n, h, w)
        indiv_gt_masks = []
        gt_mask = cv2.imread(os.path.join(gt_mask_dir, base_name + '.png')).astype(np.uint8)[:,:,0]
        num_gt_masks = np.max(gt_mask)
        for i in range(1, num_gt_masks+1):
            indiv_gt_masks.append(gt_mask == i)
        gt_mask_output = np.stack(indiv_gt_masks)
        np.save(os.path.join(resized_segmask_dir, output_name + '.npy'), gt_mask_output)

        # Set up predicted masks and metadata
        indiv_pred_masks = []
        r_info = {
            'rois': [],
            'scores': [],
            'class_ids': [],
        }

        if bin_mask_dir:
            mask_im = BinaryImage.open(os.path.join(bin_mask_dir, base_name +'.png'), 'phoxi')
            bin_mask = cv2.resize(mask_im.data, (pred_mask.shape[1], pred_mask.shape[0])) 

        # Number of predictions to use (larger number means longer time)
        num_pred_masks = min(pred_mask.shape[2], 100)
        # num_pred_masks = pred_mask.shape[2]
        for i in range(1, num_pred_masks + 1):

            # Extract individual mask
            indiv_pred_mask = pred_mask[:,:,i-1]
            if not np.any(indiv_pred_mask):
                continue
            if bin_mask_dir:
                inter = np.logical_and(bin_mask, indiv_pred_mask)
                frac_overlap =  np.sum(inter) / np.sum(indiv_pred_mask)
                if frac_overlap <= 0.5:
                    continue
            inter = np.logical_and(indiv_pred_mask, np.sum(indiv_pred_masks, axis=0))
            frac_overlap =  np.sum(inter) / np.sum(indiv_pred_mask)
            if frac_overlap >= 0.5:
                continue
            indiv_pred_masks.append(indiv_pred_mask)

            # Compute bounding box, score, class_id
            nonzero_pix = np.nonzero(indiv_pred_mask)
            min_x, max_x = np.min(nonzero_pix[1]), np.max(nonzero_pix[1])
            min_y, max_y = np.min(nonzero_pix[0]), np.max(nonzero_pix[0])
            r_info['rois'].append([min_y, min_x, max_y, max_x])
            if detector.mock_score:
                # Generates a meaningful mock score for MCG (first region scores
                # highest, etc.)
                r_info['scores'].append(-i)
            else:
                r_info['scores'].append(1.0)
            r_info['class_ids'].append(1)
        r_info['rois'] = np.array(r_info['rois'])
        r_info['scores'] = np.array(r_info['scores'])
        r_info['class_ids'] = np.array(r_info['class_ids'])
        # Write the predicted masks and metadata
        pred_mask_output = np.stack(indiv_pred_masks).astype(np.uint8) if indiv_pred_masks else np.array([])
        np.save(os.path.join(pred_dir, output_name + '.npy'), pred_mask_output)
        np.save(os.path.join(pred_info_dir, output_name + '.npy'), r_info)
        pred_mask_output = np.stack(indiv_pred_masks).astype(np.uint8)

    print('Saved prediction masks to:\t {}'.format(pred_dir))
    print('Saved prediction info (bboxes, scores, classes) to:\t {}'.format(pred_info_dir))
    print('Saved transformed GT segmasks to:\t {}'.format(resized_segmask_dir))

    return pred_dir, pred_info_dir, resized_segmask_dir

def nms(overlaps, score, thresh):
    order = score.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = overlaps[i, order[1:]]
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep
 
class MCG(object):
    """Running MCG."""

    def __init__(self, mcg_dir, nms_thresh=1.0):
        self.mcg_dir = mcg_dir
        self.mock_score = True
        self.nms_thresh = nms_thresh

    def compute_iou(self, sp2reg, sp):
        sp_area = np.bincount(sp.ravel())[np.newaxis, 1:]*1.
        sp2reg_ = sp2reg*1.
        reg_area = np.dot(sp2reg_, sp_area.T)
        inter = np.dot((sp_area * sp2reg_), sp2reg_.T)
        union = reg_area + reg_area.T - inter
        iou = inter / np.maximum(union, 1)
        return iou

    def fast_nms(self, sp2reg, sp, scores, nms_thresh):
        """Do NMS between things."""
        # Compute IoU
        iou = self.compute_iou(sp2reg, sp)
        keep = nms(iou, scores, nms_thresh)
        return keep

    def detect(self, rgb_fn):
        """Loads masks.

        Returns
        -------
        mask : (h,w,N) binary
            mask for the N object proposals
        """
        base_name = os.path.basename(rgb_fn).split('.')[0]
        mcg_fn = os.path.join(self.mcg_dir, base_name+'.mat')
        mcg = scio.loadmat(mcg_fn)
        sp, sp2reg = mcg['superpixels'], mcg['sp2reg']
        scores = -np.arange(sp2reg.shape[0])

        if self.nms_thresh < 1:
          ids = self.fast_nms(sp2reg, sp, scores, self.nms_thresh)
          sp2reg = sp2reg[ids, :]
        N = sp2reg.shape[0]
        mask = np.zeros((sp.shape[0], sp.shape[1], N), dtype=np.bool)
        for i in tqdm(range(N)):
            mask[:,:,i] = sp2reg[i,:][sp-1]
        return mask

class GOP(object):
    """Running GOP."""

    def __init__(self):
        """ """
        prop = proposals.Proposal( setupLearned( 140, 4, 0.8 ) )
        dir_path = os.path.dirname(os.path.realpath(__file__))
        detector = contour.MultiScaleStructuredForest()
        detector.load(os.path.join(dir_path, 'data', 'sf.dat'))
        self.detector = detector
        self.prop = prop
        self.mock_score = False

    def compute_iou(self, mask):
        m = np.reshape(mask, [mask.shape[0],-1]).astype(np.float32)
        inter = np.dot(m, m.T)
        ar = np.sum(m, 1)[:,np.newaxis]
        union = ar + ar.T - inter
        iou = inter / union
        return iou

    def detect(self, rgb_fn):
        """Perform detection on a RGB image.

        Returns
        -------
        mask : (h,w,N) binary
            mask for the N object proposals
        """
        s = segmentation.geodesicKMeans(imgproc.imread(rgb_fn), self.detector, 1000)
        b = self.prop.propose( s )
        
        # If you just want the boxes use
        boxes = s.maskToBox( b )

        # Read out all the masks
        N = b.shape[0]
        mask = np.zeros((s.s.shape[0], s.s.shape[1], N), dtype=np.bool)
        for i in range(b.shape[0]):
            mask[:,:,i] = b[i,s.s]
        return mask

if __name__ == '__main__':
    dataset_dir = '/nfs/diskstation/projects/dex-net/segmentation/datasets/real_test_cases_06_07_18/phoxi/images'
    indices_arr = np.arange(0, 10)
    bin_mask_dir = 'segmasks_filled'
    run_dir = './run'

    # For MCG
    # config = {'mode': 'fast', 'nms_thresh': 0.5}

    # For GOP
    config = {}

    detect('gop', config, run_dir, dataset_dir, indices_arr, bin_mask_dir=bin_mask_dir)
