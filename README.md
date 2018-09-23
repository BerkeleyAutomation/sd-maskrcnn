# Mask-RCNN Data Pipeline
## Overview
This is a pipeline written to handle the data and run Mask-RCNN on depth-image datasets. It can turn raw images and segmasks to properly sized and transformed images (and corresponding masks), then train Mask-RCNN upon the result. Following that, the pipeline can benchmark the final model weights on a given test dataset, perhaps of real depth images. Typically, one sets the yaml file associated with the task to perform (e.g., train, benchmark) and then runs the associated script. Benchmarking code for PCL and GOP baselines is also included.

The following are the available tasks, or operations that the pipeline can perform.
#### Augment
This operation takes a dataset and injects noise/inpaints images/can apply arbitrary operations upon an image as a pre-processing step.
`python sd_maskrcnn/augment.py`
#### Resize
This operation takes folders of images and corresponding segmasks, and resizes them together to the proper shape as required by Mask-RCNN. `python sd_maskrcnn/augment.py`
#### Train
This operation runs the training for Mask-RCNN on a dataset specified in the yaml file. Note that CUDA_VISIBLE_DEVICES can be set if training using a GPU (recommended). `python sd_maskrcnn/train.py`
#### Benchmark
This operation, given model weights, runs COCO benchmarks and other benchmarking code on the indices and dataset specified in the yaml file. Note that CUDA_VISIBLE_DEVICES can be set if benchmarking using a GPU. `python benchmark.py`
## Requirements
```
- numpy
- scipy
- scikit-image
- tensorflow-gpu (v1.7)
- jupyter
- opencv-python
- pytest
- keras
- tqdm
- matplotlib
- perception (from BerkeleyAutomation)
- cython (for COCO API)
- flask (if labelling images)
```
These can all be installed by running `pip3 install -r requirements.txt`, except for `perception`, which must be cloned from `BerkeleyAutomation`. It can then be installed using `pip3 install -e .` from inside the `perception` folder.

Additionally, in order to compute COCO benchmarks, the COCO API must be installed inside the repository root directory.
Get it [here.](https://github.com/cocodataset/cocoapi).
Then, navigate to `cocoapi/PythonAPI/` and run `make install`.

## Datasets
Datasets for training and evaluation can be found at https://sites.google.com/view/wisdom-dataset/home. The latest version of the WISDOM dataset will be uploaded soon, along with pre-trained models. To create one's own dataset we use the image labelling tool from https://github.com/yuyu2172/image-labelling-tool.

## Standard Dataset Format
All datasets, both real and sim, are assumed to be in the following format:
```
<dataset root directory>/
    depth_ims/
        image_000000.png
        image_000001.png
        ...
    modal_segmasks/
        image_000000.png
        image_000001.png
        ...
    segmasks_filled/ (optional)
    train_indices.npy
    test_indices.npy
    <other names>_indices.npy
    ...
```
All segmasks inside `modal_segmasks/` must be single-layer .pngs with 0 corresponding to the background and 1, 2, ... corresponding to a particular instance. Additionally, depth images and ground truth segmasks must be the same size; use `resize.py` in the pipeline to accomplish this. If using bin-vs-no bin segmasks to toss out spurious predictions, `segmasks_filled/` must contain those segmasks.
These should be binary (0 if bin, 255 if object).

## Benchmark Output Format
Running `benchmark.py` will output a folder containing results, which is structured as follows:

```
<name>/ (results of one benchmarking run)
    modal_segmasks_processed/
    pred_info/
    pred_masks/
        coco_summary.txt
    results_saurabh/ (optional)
    vis/ (optional)
    ...
```

COCO performance scores are located in `pred_masks/coco_summary.txt`.
Images of the network's predictions for each test case can be found in `vis/` if the vis flag is set.
More benchmarking outputs (plots, missed images) can be found in `results_saurabh` if the flag is set.
