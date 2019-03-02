import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set()
cp = sns.color_palette('tab10',10)
results_dir = '/nfs/diskstation/projects/dex-net/segmentation/final_results'

remap_dict = {
    'gop' : 'GOP',
    'mcg_fast': 'MCG',
    'coco_ft_color' : 'MRCNN Color-FT',
    'coco_ft_depth' : 'MRCNN Depth-FT',
    'dopenet_ft'  : 'SD MRCNN-FT',
    'dopenet'   : 'SD MRCNN',
    'pcl_euclidean' : 'PCL ECE',
    'pcl_region_growing' : 'PCL RGE'
}

color_dict = {
    'GOP' : cp[0],
    'MCG': cp[1],
    'MRCNN Color-FT' : cp[2],
    'MRCNN Depth-FT' : cp[3],
    'SD MRCNN-FT'  : cp[4],
    'SD MRCNN'   : cp[5],
    'PCL ECE' : cp[6],
    'PCL RGE' : cp[7]

}

label_order = [
    'PCL ECE',
    'PCL RGE',
    'GOP',
    'MCG',
    'MRCNN Depth-FT',
    'MRCNN Color-FT',
    'SD MRCNN',
    'SD MRCNN-FT'
]

pcl_ece_ar = {
    'phoxi': (3.72, 0.446),
    'primesense': (2.8, 0.322),
    'sim': (2.835, 0.169)
}

pcl_rge_ar = {
    'phoxi': (3.88, 0.450),
    'primesense': (3.48, 0.274),
    'sim': (2.83, 0.131)
}

gop_ar = {
    'phoxi': (6.93, 0.447),
    'primesense': (4.38, 0.333)
}

pcl_ece_pr = {
    'phoxi': (0.668, 0.814),
    'primesense': (0.528, 0.786),
    'sim': (0.268, 0.657)
}

pcl_rge_pr = {
    'phoxi': (0.676, 0.784),
    'primesense': (0.542, 0.702),
    'sim': (0.225, 0.536)
}

gop_pr = {
    'phoxi': (0.723, 0.550),
    'primesense': (0.616, 0.735)
}

method_names = os.listdir(results_dir)

cam_info = []

# # Create recall/number of detections curves
# for method_name in method_names:
#     if method_name == 'pcl_tune' or 'fast' in method_name or 'rpn' in method_name or 'mcg' in method_name or 'gop' in method_name:
#         continue
#     hr_name = remap_dict[method_name.rsplit('_', 1)[0]]
#     cam_name = method_name.rsplit('_', 1)[1]
#     if cam_name not in cam_info:
#         cam_info.append(cam_name)
#         plt.figure(cam_name)
#         #plt.title('Recall versus number of instances at 0.5 IoU ({})'.format(cam_name))
#         plt.xlabel('Number of instances')
#         plt.ylabel('Recall')
#         # plt.style.context = 'seaborn-darkgrid'
#     else:
#         plt.figure(cam_name)
#     coco_eval_fn = os.path.join(results_dir, method_name, 'pred_masks', 'coco_eval.npy')
#     recall_arr = np.load(coco_eval_fn).item()['recall']
#     rec_at_50 = recall_arr[50,:,:,:20].flatten()
#     plt.plot(rec_at_50, label=hr_name, color=color_dict[hr_name])

# for cam_name in cam_info:
#     f = plt.figure(cam_name)
#     f.legend(bbox_to_anchor=(0.91, 0.31), ncol=2, frameon=True)
# plt.show()

cam_info = {}

# Average Recall vs number of detections
for method_name in method_names:
    if method_name == 'pcl_tune' or 'rpn' in method_name or 'accurate' in method_name or 'coco_ft_phoxi' in method_name:
        continue
    hr_name = remap_dict[method_name.rsplit('_', 1)[0]]
    cam_name = method_name.rsplit('_', 1)[1]
    if cam_name not in cam_info:
        cam_info[cam_name] = {}
    
    coco_eval_fn = os.path.join(results_dir, method_name, 'pred_masks', 'coco_eval.npy')
    recall_arr = np.load(coco_eval_fn).item()['recall']
    rec_at_50 = np.mean(recall_arr.squeeze()[50:100:5,:], axis=0)
    cam_info[cam_name][hr_name] = rec_at_50

for cam_name in cam_info:
    f = plt.figure(cam_name)
    #plt.title('Precision vs. Recall at 0.5 IoU ({})'.format(cam_name))
    plt.xlabel('Number of Detections')
    plt.ylabel('Average Recall')
    plt.ylim(0,1)
    methods = cam_info[cam_name]
    for label in label_order:
        if label not in methods:
            continue
        if label == 'PCL ECE':
            plt.plot(pcl_ece_ar[cam_name][0], pcl_ece_ar[cam_name][1], 'o', label=label, color=color_dict[label])
        elif label == 'PCL RGE':
            plt.plot(pcl_rge_ar[cam_name][0], pcl_rge_ar[cam_name][1], 'o', label=label, color=color_dict[label])
        elif label == 'GOP':
            plt.plot(gop_ar[cam_name][0], gop_ar[cam_name][1], 'o', label=label, color=color_dict[label])
        else:
            plt.plot(np.arange(1,13), cam_info[cam_name][label][:12], label=label, color=color_dict[label])
    f.legend(loc='upper right', bbox_to_anchor=(0.91, 0.89), ncol=2, frameon=True, framealpha=2.0)
plt.show()

cam_info = {}

# IoU vs number of detections
for method_name in method_names:
    if method_name == 'pcl_tune' or 'accurate' in method_name or 'rpn' in method_name or 'coco_ft_phoxi' in method_name:
        continue
    hr_name = remap_dict[method_name.rsplit('_', 1)[0]]
    cam_name = method_name.rsplit('_', 1)[1]

    if cam_name not in cam_info:
        cam_info[cam_name] = {}
    
    coco_eval_fn = os.path.join(results_dir, method_name, 'pred_masks', 'coco_eval.npy')
    recall_arr = np.load(coco_eval_fn).item()['recall']
    rec_at_50 = np.dot(np.linspace(0.005,0.995,100)[np.newaxis,:], (recall_arr.squeeze()[:-1,:] - recall_arr.squeeze()[1:,:]))[0,:]
    cam_info[cam_name][hr_name] = rec_at_50

for cam_name in cam_info:
    f = plt.figure(cam_name)
    #plt.title('Precision vs. Recall at 0.5 IoU ({})'.format(cam_name))
    plt.xlabel('Number of Detections')
    plt.ylabel('Jaccard Index')
    plt.ylim(0,1)
    methods = cam_info[cam_name]
    for label in label_order:
        if label not in methods:
            continue
        if label == 'PCL ECE':
            plt.plot(pcl_ece_ar[cam_name][0], cam_info[cam_name][label][-1], 'o', label=label, color=color_dict[label])
        elif label == 'PCL RGE':
            plt.plot(pcl_rge_ar[cam_name][0], cam_info[cam_name][label][-1], 'o', label=label, color=color_dict[label])
        elif label == 'GOP':
            plt.plot(gop_ar[cam_name][0], cam_info[cam_name][label][-1], 'o', label=label, color=color_dict[label])
        else:
            plt.plot(np.arange(1,13), cam_info[cam_name][label][:12], label=label, color=color_dict[label])
    f.legend(loc='lower right', bbox_to_anchor=(0.91, 0.1), ncol=2, frameon=True)

plt.show()

# Create precision/recall curves
cam_info = {}

for method_name in method_names:
    if method_name == 'pcl_tune' or 'accurate' in method_name or 'rpn' in method_name or 'coco_ft_phoxi' in method_name:
        continue

    hr_name = remap_dict[method_name.rsplit('_', 1)[0]]
    cam_name = method_name.rsplit('_', 1)[1]

    if cam_name not in cam_info:
        cam_info[cam_name] = {}

    coco_eval_fn = os.path.join(results_dir, method_name, 'pred_masks', 'coco_eval.npy')
    recall_arr = np.load(coco_eval_fn).item()['precision']
    rec_at_50 = np.zeros(recall_arr.shape[1])
    for i in range(recall_arr.shape[0]):
        rec_at_50 += recall_arr[50,:,:,:,20].flatten()
    rec_at_50 /= recall_arr.shape[0]
    cam_info[cam_name][hr_name] = rec_at_50

x = np.arange(0,1.01,0.01)
for cam_name in cam_info:
    f = plt.figure(cam_name)
    #plt.title('Precision vs. Recall at 0.5 IoU ({})'.format(cam_name))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1)
    plt.ylim(0,1)
    methods = cam_info[cam_name]
    for label in label_order:
        if label not in methods:
            continue
        elif label == 'PCL ECE':
            plt.plot(pcl_ece_pr[cam_name][0], pcl_ece_pr[cam_name][1], 'o', label=label, color=color_dict[label])
        elif label == 'PCL RGE':
            plt.plot(pcl_rge_pr[cam_name][0], pcl_rge_pr[cam_name][1], 'o', label=label, color=color_dict[label])
        elif label == 'GOP':
            plt.plot(gop_pr[cam_name][0], gop_pr[cam_name][1], 'o', label=label, color=color_dict[label])
        else:
            plt.plot(x, methods[label], label=label, color=color_dict[label])
    f.legend(loc='lower left', bbox_to_anchor=(0.115, 0.095), ncol=2, frameon=True)

plt.show()

