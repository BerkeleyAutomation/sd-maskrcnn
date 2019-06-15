# SD Mask R-CNN Target Branch Experimentation

This is Andrew Lee's branch for development of the target identification SD Mask R-CNN variant.
Currently, given a dataset of `(target obj image, pile image, GT obj mask index)`, it is able to identify the desired target object mask in addition to performing segmentation on RGB images of object piles.

For questions please reach out to Andrew Lee or Mike. Andrew is interning and may not be responsive but can advise more generally.

## Setup
Train using `tools/train_siamese.py` and benchmark using `tools/benchmark_siamese.py`. Configs are specified in the same fashion as with the original SD Mask R-CNN. These scripts are somewhat similar to the original SD Mask R-CNN train and benchmark scripts but needed some tweaks to account for a new dataset format. Because changes have been made to the underlying Keras model inputs and outputs, the original scripts will no longer be functional.

Training with the config and dataset below with no changes should yield a steadily declining loss curve (loss < 0.03 after 20 epochs), and the benchmarking script should report an mIoU of ~0.55. Andrew got 0.567. If the net can't pull this off there's likely a bug in the implementation.

Please don't push changes to this branch- check out your own branch and make changes there. 

## Config
Current config is not on source control (and it probably should be)- located at `/nfs/diskstation/andrew_lee/cfg/train_siamese_one_loss.yaml`. There are new fields added to reflect changes in dataset format and for training the target branch.

## Dataset Format Changes
Dataset can be found at `/nfs/diskstation/andrew_lee/wisdom_real_targets/`. As the inputs to the model are no longer just a pile image and corresponding pile GT mask, we had to change the convention of the original Mask R-CNN repo. Each index in the dataset no longer corresponds to an identically-named image/mask stack in the dataset (`image_000123.png/.npy`), but rather a tuple of `(target obj image path, pile image path, index in corresponding GT mask stack)`. These tuples can be found in `target.json` at the root of the dataset directory. Please see the implementations at `TargetDataset` in `sd_maskrcnn/dataset.py`, as well as `model.py#load_inputs_gt` and `model.py#load_target`.

The JSON file containing the tuples is generated in a messy script located at `tools/generate_target_indices.ipynb` (sorry about that). If you plan to generate a new dataset or change the dataset format again you will likely need to run this notebook.

## Benchmarking
Benchmarking statistics are calculated ad-hoc in `tools/benchmark_siamese.py`. The convention is to put desired statistics into a Python dict and save it as a NumPy structured array. There is also a way to do benchmarking statistics & visualizations only (no inference, no GPU needed) by specifying an existing output directory.

## TODOs
These are things that Andrew wanted to do but didn't get to, in increasing order of difficulty by his estimates:
### Research
- [ ] Time for one forward pass- both network and total incl. pre- and post-processing check for improvements over separate segmentation & target ID pipeline (may already be supported since Andrew rebased recently)
- [ ] Get precision and recall curves and mean average precision (mAP) statistics. As this is a binary classification task (did you get the target or not), recommended ways to calculate are by either fixing an IoU threshold and varying network confidence, or vice-versa. mAP can be calculated with COCO benchmark tools (see original SD Mask R-CNN implementation).
- [ ] Double check if random rotations are useful
- [ ] Regularization and/or other strategies to prevent net from being equally confident in some predictions, and being overconfident. Suggested references include (https://arxiv.org/pdf/1512.00567.pdf, section 7 and https://arxiv.org/pdf/1701.06548.pdf).
- [ ] (*Mike and Andrew think this is most promising*) Pass in multiple images of target object in different stable poses, and explore architecture tweaks to utilize the additional information from that. Requires non-trivial engineering effort to modify dataset format and dataset loading methods- associated task in Engineering section.
- [ ] Reenable other losses and train all branches end-to-end. May require some tricks to balance the losses out.
- [ ] Add a 4th channel for depth data. Work by Andrew Li (ask Mike) indicates that even depth data alone is somewhat sufficient for this task.

### Engineering
- [ ] Put config in source control
- [ ] Support Siamese architecture changes in config (e.g. number of layers, layer size) 
- [ ] Support rotation & other dataset augmentations in config (could look into old Matterport implementation)
- [x] Support benchmarking from existing results directory instead of end-to-end (which requires a free GPU)
- [ ] Rebase changes from `master` of `BerkeleyAutomation/sd-maskrcnn`and `BerkeleyAutomation/maskrcnn` frequently (annoying, Andrew has neglected this, oops)
- [ ] Properly implement dataset generation workflow instead of using an iPython notebook- but make sure it's still possible to sanity check.
- [ ] Design a more formal way of calculating multiple benchmarks and recording their outputs.
- [ ] Support any number of target images in dataset and model. Requires changes in network architecture and the data generator (both in `model.py`), as well as in the `TargetDataset` class. Probably worth setting a corresponding config field as Keras doesn't do well with dynamic input sizes.
- [ ] Merge functionality of this branch with the master branch. **This is a difficult task which needs software design chops** (the idiosyncracies in the codebase are because of a failed attempt to do so). Areas to change include putting options for extra branches in config and supporting them in model code, supporting more general benchmarking and dataset functions. If you come up with a suitable abstraction that can support this please run it by Andrew & Mike.





# ---ORIGINAL README BELOW---

# Segmenting Unknown 3D Objects from Real<br/> Depth Images using Mask R-CNN Trained<br/> on Synthetic Point Clouds
>>>>>>> Knowledge transfer
Michael Danielczuk, Matthew Matl, Saurabh Gupta, Andrew Lee, Andrew Li, Jeffrey Mahler, and Ken Goldberg. https://arxiv.org/abs/1809.05825. [Project Page](https://sites.google.com/view/wisdom-dataset/home)

<p align="center">
    <img src="https://github.com/BerkeleyAutomation/sd-maskrcnn/blob/master/resources/seg_example.png" width="50%"/>
</p>

## Install SD Mask R-CNN
To begin using the SD Mask R-CNN repository, clone the repository using `git clone https://github.com/BerkeleyAutomation/sd-maskrcnn.git` and then run `bash install.sh` from inside the root directory of the repository. This script will install the repo, and download the available pre-trained model to the `models` directory, if desired. If dataset generation capabilities are desired, run `bash install.sh generation`.

Note that these instructions assume a Python 3 environment.

## Benchmark a Pre-trained Model
To benchmark a pre-trained model, first download the [pre-trained model](https://berkeley.box.com/shared/static/obj0b2o589gc1odr2jwkx4qjbep11t0o.h5) and extract it to `models/sd_maskrcnn.h5`. If you have run the install script, then the model has already been downloaded to the correct location. Next, download the [WISDOM-Real](https://berkeley.box.com/shared/static/7aurloy043f1py5nukxo9vop3yn7d7l3.rar) dataset for testing. Edit `cfg/benchmark.yaml` so that the test path points to the dataset to test on (typically, `/path/to/dataset/wisdom/wisdom-real/high-res/`). Finally, run `python tools/benchmark.py` from the root directory of the project. You may also set the CUDA_VISIBLE_DEVICES if benchmarking using a GPU. Results will be available within the output directory specified in the `benchmark.yaml` file, and include visualizations of the masks if specified. An example for getting started with benchmarking can be found in this [Benchmarking Google CoLab Notebook](https://colab.research.google.com/drive/1beJu7Pjmf9JLcyNR66Btw6Hij0F2f64k).

## Train a New Model
To train a new model, first download the [WISDOM-Sim](https://berkeley.box.com/shared/static/laboype774tjgu7dzma3tmcexhdd8l5a.rar) dataset. Edit `cfg/train.yaml` so that the test path points to the dataset to train on (typically, `/path/to/dataset/wisdom/wisdom-sim/`) and adjust training parameters for your GPU (e.g., number of images per GPU, GPU count). Then, run `python tools/train.py`, again setting CUDA_VISIBLE_DEVICES if necessary.

Note: If you wish to train using single channel images (such as those in WISDOM-Sim), you can change the image_channel_count and mean_pixel parameters to 1 and the single channel mean pixel value, respectively. This option also works when loading pre-trained weights (such as COCO or Imagenet).

## Generate a New Dataset
An example for getting started with dataset generation can be found in this [Dataset Generation Google CoLab Notebook](https://colab.research.google.com/drive/1iafphvk6oRT_RF0_fD6XwbHw8tpHqZeu). To generate a new dataset for training, use the `tools/generate_mask_dataset.py` script. Edit the corresponding config files (`cfg/generate_mask_dataset.yaml, cfg/partials/states.yaml, cfg/partials/mask_dataset.yaml`) to fit your needs (specifically, at minimum, you must configure `cfg/partials/states.yaml` to point at your directory of object meshes). The `--save_tensors` command line argument allows for saving the state of each heap generated, and the `--warm_start` option allows for resuming dataset generation if it is stopped. By default, the script outputs a dataset of images to the directory specified on the command line with the following structure:
```
<dataset root directory>/
    images/
        amodal_masks/
            image_000000/
                channel_000.png
                channel_001.png
                ...
            image_000001/
                channel_000.png
                channel_001.png
                ...
            ...
        depth_ims/
            image_000000.png
            image_000001.png
            ...
        modal_masks/
            image_000000/
                channel_000.png
                channel_001.png
                ...
            image_000001/
                channel_000.png
                channel_001.png
                ...
            ...
        semantic_masks/
            image_000000.png
            image_000001.png
            ...
        train_indices.npy
        test_indices.npy
    metadata.json
    dataset_generation.log
    dataset_generation_params.yaml
```
The modal and amodal masks directories give binary amodal and modal segmentation masks for each of the objects in the heap. Semantic masks are the single-channel stacked modal masks, and depth_ims contains depth images.

## Other Available Tools
Typically, one sets the yaml file associated with the task to perform (e.g., train, benchmark, augment) and then runs the associated script. Benchmarking code for PCL and GOP baselines is also included. 

#### Augment
This operation takes a dataset and injects noise/inpaints images/can apply arbitrary operations upon an image as a pre-processing step.
`python tools/augment.py`
#### Resize
This operation takes folders of images and corresponding segmasks, and resizes them together to the proper shape as required by Mask-RCNN. `python tools/augment.py`
#### Benchmark Baseline
This operation benchmarks the PCL or GOP baselines on a given dataset. Settings for each dataset and PCL method are commented in the corresponding yaml file, as well as visualization and output settings. Run with `python tools/benchmark_baseline.py`.

To run the GOP baseline, first run these commands from the project root directory to install GOP:
```
cd sd_maskrcnn/gop && mkdir build && cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=3 && make
```

To run the PCL baselines, first install python-pcl using the instructions here: https://github.com/strawlab/python-pcl.

## Datasets
Datasets for training and evaluation can be found at the [Project Page](https://sites.google.com/view/wisdom-dataset/home).

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
    ...
```
All segmasks inside `modal_segmasks/` must be single-layer .pngs with 0 corresponding to the background and 1, 2, ... corresponding to a particular instance. Additionally, depth images and ground truth segmasks must be the same size; use `resize.py` in the pipeline to accomplish this. If using bin-vs-no bin segmasks to toss out spurious predictions, `segmasks_filled/` must contain those segmasks.
These should be binary (0 if bin, 255 if object). More information can be found in the README.txt file.

### Benchmark Output Format
Running `benchmark.py` will output a folder containing results, which is structured as follows:

```
<name>/ (results of one benchmarking run)
    modal_segmasks_processed/
    pred_info/
    pred_masks/
        coco_summary.txt
    results_supplement/ (optional)
    vis/ (optional)
    ...
```

COCO performance scores are located in `pred_masks/coco_summary.txt`.
Images of the network's predictions for each test case can be found in `vis` if the vis flag is set.
More benchmarking outputs (plots, missed images) can be found in `results_supplement` if the flag is set.

## Citation
If you use this code for your research, please consider citing:
```
@article{danielczuk2018segmenting,
  title={Segmenting Unknown 3D Objects from Real Depth Images using Mask R-CNN Trained on Synthetic Data},
  author={Danielczuk, Michael and Matl, Matthew and Gupta, Saurabh and Li, Andrew and Lee, Andrew and Mahler, Jeffrey and Goldberg, Ken},
  journal={arXiv preprint arXiv:1809.05825},
  year={2018}
}
```
