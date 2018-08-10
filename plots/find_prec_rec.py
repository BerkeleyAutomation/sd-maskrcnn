import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='either phoxi, sim, or primesense')
parser.add_argument('type', help='either euclidean, region_growing, or gop')
args = parser.parse_args()

results_path = '/nfs/diskstation/projects/dex-net/segmentation/final_results'
if args.type == 'gop':
    evalImgs = np.load(os.path.join(results_path, args.type+'_'+args.dataset, 'pred_masks', 'coco_evalImgs.npy'))
else:
    evalImgs = np.load(os.path.join(results_path, 'pcl_'+args.type+'_'+args.dataset, 'pred_masks', 'coco_evalImgs.npy'))

instances = []
detections = []
recalls = []
precisions = []

for iou in np.arange(90,101,1):
    for im in evalImgs: 
        gtMatches = im['gtMatches'][iou]
        dtMatches = im['dtMatches'][iou]
        instances.append(gtMatches.size)
        detections.append(dtMatches.size)
        if np.any(gtMatches):
            recalls.append(np.sum(gtMatches > 0)/gtMatches.size)
        else:
            recalls.append(0)
        if np.any(dtMatches):
            precisions.append(np.sum(dtMatches > 0)/dtMatches.size)
        else:
            precisions.append(0)

print('All IoU')
print(np.mean(detections))

instances = []
detections = []
recalls = []
precisions = []
for iou in np.arange(50,100,5):
    for im in evalImgs: 
        gtMatches = im['gtMatches'][iou]
        dtMatches = im['dtMatches'][iou]
        instances.append(gtMatches.size)
        detections.append(dtMatches.size)
        if np.any(gtMatches):
            recalls.append(np.sum(gtMatches > 0)/gtMatches.size)
        else:
            recalls.append(0)
        if np.any(dtMatches):
            precisions.append(np.sum(dtMatches > 0)/dtMatches.size)
        else:
            precisions.append(0)
print('0.5:0.05:0.95')
print(np.mean(detections), np.mean(instances), np.mean(recalls), np.mean(precisions))

instances = []
detections = []
recalls = []
precisions = []

for im in evalImgs: 
    gtMatches = im['gtMatches'][50]
    dtMatches = im['dtMatches'][50]
    instances.append(gtMatches.size)
    detections.append(dtMatches.size)
    if np.any(gtMatches):
        recalls.append(np.sum(gtMatches > 0)/gtMatches.size)
    else:
        recalls.append(0)
    if np.any(dtMatches):
        precisions.append(np.sum(dtMatches > 0)/dtMatches.size)
    else:
        precisions.append(0)
print('0.5')
print(np.mean(detections), np.mean(instances), np.mean(recalls), np.mean(precisions))