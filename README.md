# Segmenting Unknown 3D Objects from Real<br/> Depth Images using Mask R-CNN Trained<br/> on Synthetic Data
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

## Dataset Generation using Docker
A Dockerfile is also provided for containerized dataset generation and training. To run using Docker, first run `docker build -t sdmaskrcnn:gen .` from the root directory of the project, then:
```shell
docker run --rm -it --gpus all -v <path/to/meshes>:/meshes:ro -v <path/to/dataset>:/dataset:rw -v <path/to/cfg>:/cfg:ro sdmaskrcnn:gen
```

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
@inproceedings{danielczuk2019segmenting,
  title={Segmenting Unknown 3D Objects from Real Depth Images using Mask R-CNN Trained on Synthetic Data},
  author={Danielczuk, Michael and Matl, Matthew and Gupta, Saurabh and Li, Andrew and Lee, Andrew and Mahler, Jeffrey and Goldberg, Ken},
  booktitle={Proc. IEEE Int. Conf. Robotics and Automation (ICRA)},
  year={2019}
}
```
