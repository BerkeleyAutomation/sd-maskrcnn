"""
Responsible for encoding both ground-truth segmentation masks and network
predictions(instance masks, class predictions, scores) into the COCO
annotations format, and calling the COCO API benchmarking metrics on said
annotations.
"""

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
# import pylab
import os
import json
from tqdm import tqdm
import model as modellib, visualize, utils
from pipeline_utils import *
from real_dataset import RealImageDataset, prepare_real_image_test



def encode_predictions(mask_dir):
    """Given a path to a directory of predicted image segmentation masks,
    encodes them into the COCO annotations format using the COCO API.
    Predictions are 3D Numpy arrays of shape (n, h, w) for n predicted
    instances, in case of overlapping masks.
    Requires that pred masks are named 0.npy, 1.npy, etc. in order without any
    missing numbers and correspond to the associated GT masks.

    mask_dir: str, directory in which pred masks are stored. Avoid relative
        paths if possible.
    """
    annos = []

    N = len([p for p in os.listdir(mask_dir) if p.endswith('.npy')])

    for i in range(N):
        # load .npy
        im_name = str(i) + '.npy'
        I = np.load(os.path.join(mask_dir, im_name))

        for bin_mask in I:
            # encode mask
            encode_mask = mask.encode(np.asfortranarray(bin_mask))
            encode_mask['counts'] = encode_mask['counts'].decode('ascii')

            # assume one category (object)
            pred_anno = {
                'image_id': i,
                'category_id': 1,
                'segmentation': encode_mask,
                'score': 1.0
            }

            annos.append(pred_anno)

    anno_path = os.path.join(mask_dir, 'annos_pred.json')
    json.dump(annos, open(anno_path, 'w+'))
    print("successfully wrote prediction annotations to", anno_path)


def encode_gt(mask_dir):
    """Given a path to a directory of ground-truth image segmentation masks,
    encodes them into the COCO annotations format using the COCO API.
    GT segmasks are 3D Numpy arrays of shape (n, h, w) for n predicted instances,
    in case of overlapping masks.
    These MUST be the same size as predicted masks, ensure this by using masks returned from
    model.load_image_gt.
    DO NOT INCLUDE THE BACKGROUND. model.load_image_gt will automatically remove it.
    Requires that GT masks are named 000000.npy, 000001.npy, etc. in order without any
    missing numbers.

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

    N = len([p for p in os.listdir(mask_dir) if p.endswith('.npy')])

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
            instance_id = i * 100 + (val + 1) # create id for instance, incrmeent val

            # find bounding box

            def bbox2(img):
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
    #             return rmin, rmax, cmin, cmax
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

            # sanity check visualizations
            # plt.imshow(bin_mask)
            # plt.scatter([x, x, x + w, x + w], [y, y + h, y, y + h])
            # plt.savefig(os.path.join(mask_dir, 'bin_{:d}.png'.format(instance_id)))
            # plt.close()

    anno_path = os.path.join(mask_dir, 'annos_gt.json')

    json.dump(gt_annos, open(anno_path, 'w+'))
    print("successfully wrote GT annotations to", anno_path)


def compute_coco_metrics(gt_dir, pred_dir):
    """Given paths to two directories, one containing a COCO ground truth annotations
    file and the other a path to a COCO prediction annotations file, compute the COCO
    evaluation metrics on the predictions.

    Because the COCO API is weird and prints out summary values, we need this
    terrible hack to capture them from stdout.
https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    """

    gt_path = os.path.join(gt_dir, 'annos_gt.json')
    pred_path = os.path.join(pred_dir, 'annos_pred.json')

    import io
    from contextlib import redirect_stdout


    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(pred_path)

    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')

    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.params.useCats = False

    cocoEval.evaluate()
    cocoEval.accumulate()

    f = io.StringIO()
    with redirect_stdout(f):
        cocoEval.summarize()
    out = f.getvalue()
    print(out)

    with open(os.path.join(pred_dir, 'coco_summary.txt'), 'w') as coco_file:
        coco_file.write(out)
        print('COCO Metric Summary written to {}'.format(os.path.join(pred_dir,
                                                                      'coco_summary.txt')))



def coco_benchmark(run_dir, inference_config, model, dataset_real):
    """Given a run directory, a MaskRCNN config object, a MaskRCNN model object,
    and a Dataset object,
    - Runs the model on the images
    - Saves predictions in run directory
    - Creates and saves visuals for said predictions in run directory
    - Computes COCO statistics (mAP over certain ranges) on predictions.

    NOTE:
    The network transforms all inputs to a particular size with scaling and
    padding operations. This function will apply those transformations to the
    ground truth segmasks and save them in a new subdirectory of the test image
    directory. This helps for generating the annotations required to call the
    COCO API.
    """

    # Create subdirectories for masks and visuals
    pred_dir = os.path.join(run_dir, 'pred')
    mkdir_if_missing(pred_dir)
    vis_dir = os.path.join(run_dir, 'vis')
    mkdir_if_missing(vis_dir)

    # Create directory in which we save transformed GT segmasks
    resized_segmask_dir = os.path.join(run_dir, 'modal_segmasks_processed')
    mkdir_if_missing(resized_segmask_dir)

    # Feed images into model one by one. For each image, predict, save, visualize?
    image_ids = dataset_real.image_ids

    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
          modellib.load_image_gt(dataset_real, inference_config, image_id,
            use_mini_mask=False)

        # Save copy of transformed GT segmasks to disk in preparation for annotations
        mask_name = 'image_{:06d}'.format(image_id)
        mask_path = os.path.join(resized_segmask_dir, mask_name)

        molded_images = modellib.mold_image(image, inference_config)
        molded_images = np.expand_dims(molded_images, 0)

        # save the transpose so it's (n, h, w) instead of (h, w, n)
        np.save(mask_path, gt_mask.transpose(2, 0, 1))
        # gt_stat, stat_name = compute_gt_stats(gt_bbox, gt_mask)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        save_masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])])
        save_masks_path = os.path.join(pred_dir, str(image_id) + '.npy')
        np.save(save_masks_path, save_masks)

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['bg', 'obj'], r['scores'])
        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Generate prediction annotations
    encode_gt(resized_segmask_dir)
    encode_predictions(pred_dir)

    compute_coco_metrics(resized_segmask_dir, pred_dir)
