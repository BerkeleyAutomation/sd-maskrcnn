import os
from tqdm import tqdm
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dirs', nargs='+', help='path to dataset to be analyzed')
parser.add_argument('--indices', default='test', help='indices of dataset to be analyzed')
args = parser.parse_args()


hist,bins = np.histogram(0*[17], bins=np.arange(1,17))
f1 = plt.figure(0)
plt.plot(bins[:-1], [10,12,11,9.5,8,6,5,4,4,4,3,2.5,2,1,1], 'o-')
ax = plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
plt.title('Instances per Image')
plt.xlabel('Number of Instances')
plt.ylabel('Percentage of Images')
f2 = plt.figure(1)
plt.plot([4, 6, 10], [29, 36, 7.5], 'o-')
ax = plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
plt.title('Instance Size')
plt.xlabel('Percentage of Image Size')
plt.ylabel('Percentage of Instances')

for dataset_dir in args.dataset_dirs:

    if 'replication' in dataset_dir:
        label_name = 'WISDOM-Sim'
        mask_dir = os.path.join(dataset_dir, 'stacked_segmasks')
    else:
        label_name = 'WISDOM-Real'
        mask_dir = os.path.join(dataset_dir, 'modal_segmasks')
    
    if args.indices == 'all':
        N = len([p for p in os.listdir(mask_dir) if p.endswith('.png')])
        indices_arr = np.arange(N)
    else:
        indices_arr = np.load(os.path.join(dataset_dir, args.indices + '_indices.npy'))
        N = indices_arr.size

    segments = []
    sizes = []

    for i in tqdm(range(N)):
        
        # load image
        im_name = 'image_{:06d}.png'.format(indices_arr[i])
        I = cv2.imread(os.path.join(mask_dir, im_name))[:,:,0]
        image_area = I.shape[0]*I.shape[1]
        max_num_segments = np.max(I)

        true_num_segments = 0
        for j in np.arange(1,max_num_segments+1):
            mask = (I == j).astype(np.uint8)
            if np.sum(mask):
                sizes.append(np.sum(mask)/image_area)
                true_num_segments += 1
        segments.append(true_num_segments)

    total_instances = np.sum(segments)
    print('Total Number of Instances: {0}'.format(total_instances))
    print('Average Instances per Image: {:.2f}'.format(np.mean(segments)))
    print('Average Instance Size: {:.2f}% of image'.format(np.mean(sizes)*100))

    plt.figure(0)
    hist,bins = np.histogram(segments, bins=np.arange(1,17))
    plt.plot(bins[:-1],100*hist/N, 'o-')    

    plt.figure(1)
    hist,bins = np.histogram(np.array(sizes), bins=np.linspace(0,0.1,21))
    plt.plot(100*bins[1:], 100*hist/total_instances, 'o-')

plt.figure(0)
plt.legend(['COCO (7.7)', 'WISDOM-Sim (6.5)', 'WISDOM-Real (4.8)'])
plt.figure(1)
plt.legend(['COCO', 'WISDOM-Sim', 'WISDOM-Real'])
plt.show()