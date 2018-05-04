# Mask-RCNN Data Pipeline
## Overview
blah blah blah
## Requirements
```
- numpy
- scipy
- skimage
- tensorflow-gpu
- jupyter
- opencv-python
- pytest
- keras
- tqdm
- matplotlib
- flask (if labelling images)
```

Additionally, in order to compute COCO benchmarks, the COCO API must be installed inside the repository root directory.
Get it [here.](https://github.com/cocodataset/cocoapi)
Then, navigate to `cocoapi/PythonAPI/` and run `make install`.


`image-labelling-tool` contains a tool for labelling segmasks.
More instructions are contained in `image-labelling-tool/save_masks.ipynb`.

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
    segmasks/ (optional)
    train_indices.npy
    <other names>_indices.npy
    ...
```
All segmasks inside `modal_segmasks/` must be single-layer .pngs with 0 corresponding to the background and 1, 2, ... corresponding to a particular instance.
To convert from multiple channel segmasks to a single segmask per case, open `clutter/clutter.py`, point `base_dir` to your particular dataset, and run said file.
This will put the "stacked" segmasks in a new directory, `modal_segmasks_project`, which should be renamed.
Additionally, depth images and ground truth segmasks must be the same size; perform the "RESIZE" task in the pipeline to accomplish this.
If using bin-vs-no bin segmasks to toss out spurious predictions, `segmasks/` must contain those segmasks.
These should be binary (0 if bin, 255 if object).


## Benchmark Output Format

## Files
Note that most of these files reside within the subfolder "noise"
### pipeline.py
yadyadya
### config.ini
blakjfa
### pipeline_utils.py
blah
### real_dataset.py/sim_dataset.py
blah
### augmentation.py
blah
### eval_coco.py/eval_saurabh.py
whee
### noise.py
cheese
### resize.py
yes



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
