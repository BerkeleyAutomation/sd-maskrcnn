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

        print(I.shape)

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
        print('I.shape', I.shape)
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


            print('bin_mask.shape', bin_mask.shape)
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
            print('(val, x, y, w, h)', (val, x, y, w, h))

            # create instance annotation
    #         print('-------------Instance Annotation-------------')
    #         print('id:', instance_id)
    #         print('image_id:', i)
    #         print('category_id:', 1)
    #         print('segmentation:', encode_mask)
    #         print('area:', size)
    #         print('bbox:', [x, y, w, h])
    #         print('iscrowd:', 0)
    #         print('---------------------------------------------')

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


def coco_benchmark(gt_path, pred_path):
    """Given a path to a COCO ground truth annotations file and a path to a COCO
    prediction annotations file, compute the COCO evaluation metrics on the
    predictions.
    """

    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(pred_path)

    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')

    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.params.useCats = False


    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
