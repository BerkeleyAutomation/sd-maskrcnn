# Mask-RCNN Data Pipeline
  ## Overview
  blah blah blah
  ## Files
     ## `noise` folder
        ### `pipeline.py`
        yadyadya
        ### `config.ini`
        blakjfa
        ### `pipeline_utils.py`
        blah
        ### `real_dataset.py`/`sim_dataset.py`
        blah
        ### `augmentation.py`
        blah
        ### `eval_coco.py`/`eval_saurabh.py`
        whee
        ### `noise.py`
        cheese
        ### `resize.py`
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
