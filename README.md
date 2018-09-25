# Segmenting Unknown 3D Objects from Real<br/> Depth Images using Mask R-CNN Trained<br/> on Synthetic Point Clouds
Michael Danielczuk, Matthew Matl, Saurabh Gupta, Andrew Lee, Andrew Li, Jeffrey Mahler, and Ken Goldberg. Under review at ICRA 2019. [Project Page](https://sites.google.com/view/wisdom-dataset/home)

<p align="center">
    <img src="https://github.com/BerkeleyAutomation/sd-maskrcnn/blob/master/resources/seg_example.png" width="50%"/>
</p>

## Install SD Mask R-CNN
To begin using the SD Mask R-CNN repository, clone the repository using `git clone https://github.com/BerkeleyAutomation/sd-maskrcnn.git` and then run `sh install.sh` from inside the root directory of the repository. This script will install the repo, and download the available pre-trained model to the `models` directory.

Note that these instructions assume a Python 3 environment.

## Benchmark a Pre-trained Model
To benchmark a pre-trained model, first download the [pre-trained model](https://drive.google.com/open?id=1USddPiSrD9DWIGzlTZ4xGkhZ11GAgrvR) and extract it to `models/sd_maskrcnn.h5`. If you have run the install script, then the model has already been downloaded to the correct location. Next, download the [WISDOM-Real]() dataset for testing. Edit `cfg/benchmark.yaml` so that the test path points to the dataset to test on (typically, `/path/to/dataset/wisdom/wisdom-real/high-res/`). Finally, run `python sd_maskrcnn/benchmark.py` from the root directory of the project. You may also set the CUDA_VISIBLE_DEVICES if benchmarking using a GPU. Results will be available within the output directory specified in the `benchmark.yaml` file, and include visualizations of the masks if specified.

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
This operation, given model weights, runs COCO benchmarks and other benchmarking code on the indices and dataset specified in the yaml file. Note that CUDA_VISIBLE_DEVICES can be set if benchmarking using a GPU. `python sd_maskrcnn/benchmark.py`

## Requirements
```
- numpy
- scipy
- scikit-image
- tensorflow-gpu
- opencv-python
- keras
- tqdm
- pycocotools
- autolab_core
- ipython
```

These can be installed by running `pip install -r requirements.txt`, which is automatically run as part of the install script.

## Datasets
Datasets for training and evaluation can be found at https://sites.google.com/view/wisdom-dataset/home. The latest version of the WISDOM dataset will be uploaded soon, along with pre-trained models. To create one's own dataset we use the image labelling tool from https://github.com/yuyu2172/image-labelling-tool.

### Standard Dataset Format
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

### Benchmark Output Format
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
Images of the network's predictions for each test case can be found in `vis` if the vis flag is set.
More benchmarking outputs (plots, missed images) can be found in `results_verbose` if the flag is set.

## Citation
If you use this code for your research, please consider citing:
```
@article{danielczuk2018segmenting,
  title={Segmenting Unknown 3D Objects from Real Depth Images using Mask R-CNN Trained on Synthetic Point Clouds},
  author={Danielczuk, Michael and Matl, Matthew and Gupta, Saurabh and Li, Andrew and Lee, Andrew and Mahler, Jeffrey and Goldberg, Ken},
  journal={arXiv preprint arXiv:1809.05825},
  year={2018}
}
```
