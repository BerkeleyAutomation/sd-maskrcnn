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

Responsible for encoding both ground-truth segmentation masks and network
predictions(instance masks, class predictions, scores) into the COCO
annotations format, and calling the COCO API benchmarking metrics on said
annotations.
"""

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask

import numpy as np
import os
import json
import fnmatch

def encode_gt(mask_dir):
    """Given a path to a directory of ground-truth image segmentation masks,
    encodes them into the COCO annotations format using the COCO API.
    GT segmasks are 3D Numpy arrays of shape (n, h, w) for n predicted instances,
    in case of overlapping masks.
    These MUST be the same size as predicted masks, ensure this by using masks returned from
    model.load_image_gt.
    DO NOT INCLUDE THE BACKGROUND. model.load_image_gt will automatically remove it.
    Requires that GT masks are named image_000000.npy, image_000001.npy, etc. in order
    without any missing numbers.

    mask_dir: str, directory in which GT masks are stored. Avoid relative
        paths if possible.
    """
    # Constructing GT annotation file per COCO format:
    # http://cocodataset.org/#download
    gt_annos = {
        'images': [],
        'annotations': [],
        'categories': [
            {'name': 'object',
             'id': 1,
             'supercategory': 'object'}
        ]
    }

    N = len(fnmatch.filter(os.listdir(mask_dir), 'image_*.npy'))

    for i in range(N):
        # load image
        im_name = 'image_{:06d}.npy'.format(i)
        I = np.load(os.path.join(mask_dir, im_name))
        im_anno = {
            'id': i,
            'width': int(I.shape[1]),
            'height': int(I.shape[2]),
            'file_name': im_name
        }

        gt_annos['images'].append(im_anno)

        # leaving license, flickr_url, coco_url, date_captured
        # fields incomplete

        # mask each individual object
        # NOTE: We assume these masks do not include backgrounds.
        # This means that the 1st object instance will have index 0!
        for val in range(I.shape[0]):
            # get binary mask
            bin_mask = I[val,:,:].astype(np.uint8)
            instance_id = i * 100 + (val + 1) # create id for instance, increment val
            # find bounding box
            def bbox2(img):
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                return int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)

            # encode mask
            encode_mask = mask.encode(np.asfortranarray(bin_mask))
            encode_mask['counts'] = encode_mask['counts'].decode('ascii')
            size = int(mask.area(encode_mask))
            x, y, w, h = bbox2(bin_mask)

            instance_anno = {
                "id" : instance_id,
                "image_id" : i,
                "category_id" : 1,
                "segmentation" : encode_mask,
                "area" : size,
                "bbox" : [x, y, w, h],
                "iscrowd" : 0,
            }

            gt_annos['annotations'].append(instance_anno)

    anno_path = os.path.join(mask_dir, 'annos_gt.json')
    json.dump(gt_annos, open(anno_path, 'w+'))
    print("successfully wrote GT annotations to", anno_path)


def encode_predictions(mask_dir, info_dir):
    """Given a path to a directory of predicted image segmentation masks,
    encodes them into the COCO annotations format using the COCO API.
    Predictions are 3D Numpy arrays of shape (n, h, w) for n predicted
    instances, in case of overlapping masks.
    Requires that pred masks are named image_000000.npy, image_000001.npy,
    etc. in order without any missing numbers and correspond to the
    associated GT masks.

    mask_dir: str, directory in which pred masks are stored. Avoid relative
        paths if possible.
    """
    annos = []

    N = len(fnmatch.filter(os.listdir(mask_dir), 'image_*.npy'))

    for i in range(N):
        # load .npy
        im_name = 'image_{:06d}.npy'.format(i)
        I = np.load(os.path.join(mask_dir, im_name))
        info = np.load(os.path.join(info_dir, im_name)).item()

        for j,bin_mask in enumerate(I):
            # encode mask
            encode_mask = mask.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
            encode_mask['counts'] = encode_mask['counts'].decode('ascii')

            # info_score = float(info['scores'][j])
            info_score = 1.0

            # assume one category (object)
            pred_anno = {
                'image_id': i,
                'category_id': 1,
                'segmentation': encode_mask,
                'score': info_score
            }
            annos.append(pred_anno)

    anno_path = os.path.join(mask_dir, 'annos_pred.json')
    json.dump(annos, open(anno_path, 'w+'))
    print("successfully wrote prediction annotations to", anno_path)


def compute_coco_metrics(gt_dir, pred_dir):
    """Given paths to two directories, one containing a COCO ground truth annotations
    file and the other a path to a COCO prediction annotations file, compute the COCO
    evaluation metrics on the predictions.
    """

    gt_path = os.path.join(gt_dir, 'annos_gt.json')
    pred_path = os.path.join(pred_dir, 'annos_pred.json')

    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(pred_path)

    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')

    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.params.useCats = False
    cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    cocoEval.params.areaRngLbl = ['all']
    cocoEval.params.maxDets = np.arange(1,101)
    cocoEval.params.iouThrs = np.linspace(0.0, 1.0, 101)

    cocoEval.evaluate()
    cocoEval.accumulate()

    np.save(os.path.join(pred_dir, 'coco_eval.npy'), cocoEval.eval)
    np.save(os.path.join(pred_dir, 'coco_evalImgs.npy'), cocoEval.evalImgs)

    # cocoEval.summarize()

    precisions = cocoEval.eval['precision'].squeeze()[50:100:5,:,-1]
    recalls = cocoEval.eval['recall'].squeeze()[50:100:5,-1]
    
    ap = np.mean(precisions[precisions>-1])
    ar = np.mean(recalls[recalls>-1])
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}\n'
    precStr = iStr.format('Average Precision', '(AP)', '0.5:0.05:0.95', 'all', 100, ap)
    recStr = iStr.format('Average Recall', '(AR)', '0.5:0.05:0.95', 'all', 100, ar)
    print(precStr)
    print(recStr)

    
    with open(os.path.join(pred_dir, 'coco_summary.txt'), 'w') as coco_file:
        coco_file.write(precStr)
        coco_file.write(recStr)
        print('COCO Metric Summary written to {}'.format(os.path.join(pred_dir,
                                                                      'coco_summary.txt')))
    return ap, ar

def coco_benchmark(pred_mask_dir, pred_info_dir, gt_mask_dir):
    """Given directories for prediction masks, prediction infos (bboxes, scores, classes),
    and properly-transformed ground-truth masks, create COCO annotations and compute and
    write COCO metrics for said predictions.
    """
    # Generate prediction annotations
    encode_gt(gt_mask_dir)
    encode_predictions(pred_mask_dir, pred_info_dir)
    return compute_coco_metrics(gt_mask_dir, pred_mask_dir)
