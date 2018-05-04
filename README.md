# Mask-RCNN Data Pipeline
## Overview
This is a pipeline written to handle the data and run Mask-RCNN on depth-image datasets. It can turn raw images and segmasks to properly sized and transformed images (and corresponding masks), then train Mask-RCNN upon the result. Following that, the pipeline can benchmark the final model weights on a given test dataset, perhaps of real depth images.

The following are the available tasks, or operations that the pipeline can perform.
#### Augment
This operation takes a dataset and injects noise/inpaints images/can apply arbitrary operations upon an image as a pre-processing step.
#### Resize
This operation takes folders of images and corresponding segmasks, and resizes them together to the proper shape as required by Mask-RCNN.
#### Train
This operation runs the training for Mask-RCNN on the standard dataset (described later).
#### Benchmark
This operation, given model weights, runs both COCO benchmarks and Saurabh's benchmarking code on a new, test dataset.
## Requirements
See requirements.txt.
## Standard Dataset Format
feiwofjoe
## Files
Note that most of these files reside within the subfolder "noise".
### pipeline.py
This is the main file to run for all tasks (train, benchmark, augment, resize). Run this file with the tag --config [config file name] (in this case, config.ini).

Include in the PYTHONPATH the locations of maskrcnn and clutter folders.

Here is an example run command (GPU selection included). Note that we are running this from the root clutter-det-maskrcnn folder, not the noise folder.
`CUDA_VISIBLE_DEVICES='0' PYTHONPATH='.:maskrcnn/:clutter/' python3 noise/pipeline.py --config noise/config.ini`
### config.ini
This is where all the parameters for each task are specified, as well as the current task to be run. It is sectioned off into sets of parameters for each type of task, as well as a flag at the top to set the task that will be executed when pipeline.py is run. Please read the inline comments of each parameter within config.ini before running pipeline.py.

### pipeline\_utils.py
This file contains some general helper functions that are used in pipeline.py.
### real\_dataset.py/sim\_dataset.py
blah
### Augmentation.py
This file contains the noise and inpainting functions that are used to modify an image dataset before it is resized and trained upon.
### eval\_coco.py/eval\_saurabh.py
whee
### resize.py
Resizing function for 512x512 images.
### noise.py
This is a standalone file that will look at areas of the image and generate depth histograms for noise analysis.



Deprecated README text from Saraubh's code:

### Code for running Mask-RCNN on images of object piles.

1. Dependencies: I developed and tested this code base with python3.5. I am using pip, and requirements are in `requirements.txt`. You will want to update your pip (with `pip install -U pip`) and then deactivate and activate before installing these requirements. Also you should install tensorflow with `pip install tensorflow-gpu`.

2. Code for running training.
`CUDA_VISIBLE_DEVICES='2' PYTHONPATH='.:maskrcnn/' python clutter/train_clutter.py --logdir outputs/v3_512_40_flip_depth --im_type depth --task train`

3. Code for benchmarking.
`CUDA_VISIBLE_DEVICES='2' PYTHONPATH='.:maskrcnn/' python clutter/train_clutter.py --logdir outputs/v3_512_40_flip_depth --im_type depth --task bench`

4 Misc. Info.
  - `clutter/clutter.py`, implements the class `ClutterDataset` that has functions for loading the images and the masks from Jeff's dataset.
  - `clutter/clutter.py:74` specifies the path where the data is stored.
  - `clutter/clutter.py` (function `concat_segmasks`) preprocesses the stored masks such that they are faster to load (basically pastes all the modal masks into a single image). So when switching to a new dataset you may want to re-run this function on the new data to use with the data class.
  -
