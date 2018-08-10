Geodesic Object Proposals
=========================

Dependencies
------------
Required:
 * cmake
 * c++11 compiler( g++-4.7 or higher, vc++2013 or higher, clang might work too )
 * Eigen 3.2 (optionally you can download eigen and put in in external/eigen such that external/eigen/Eigen is a valid directory)
 * libpng and libjpg (needed by cimg)

Optional for python3 bindings:
 * python3 (python 2.7 should work too, but I didn't test it extensively)
 * numpy
 * boost-python
 * matio (optional to load datasets)
 * matplotlib for some visualizations
 * MATLAB (r2013a on Ubuntu 14.04 tested, others might work too. You might have to specify a new gcc version in mexopts)

How to compile
--------------
 * create an build directory (eg. build)
 * call cmake: cmake -DCMAKE_BUILD_TYPE=Release relative_path_to_source
 * to compile the matlab bindings add the flag '-DUSE_PYTHON=3' (for python3), '-DUSE_PYTHON=2' (for python2.7)
 * to compile the matlab bindings add the flag '-DUSE_MATLAB=On'
You can define several variables to load various datasets from the python bindings:
 * BERKELEY_DIR: Directory of the BSD500 dataset (point to where 'groundTruth' and 'images' directories are located)
 * VOC_DIR: Directory of the VOCdevkit, with subfolders VOC2010, VOC2012, ... (point to where 'VOC2010', 'VOC2011', 'VOC2012' directories are located)

Example:
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DVOC_DIR=/home/user/Downloads/VOCdevkit -DBERKELEY_DIR=/home/user/Downloads/BSDS500

How to use/run
--------------
There is a small cpp example to illustate how to use the object proposals. The majority of examples are in python however.

eval_baseline.py : Reproduces the baseline numbers reported in the paper
eval_bnd.py: Compares various boundary detectors (fig 7 in paper).
eval_box.py: Runs GOP on the VOC 2012 detection dataset and evaluates the box overlap (run as:"python eval_box.py place_to_store_result.dat")
eval_coco.py: Evaluates GOP on the COCO dataset. This will take a while ~6 hours.
eval_learned.py : Trains a model, if none is provided and reproduces the learned numbers reported in the paper
eval_seed.py: Computes the number of undiscovered objects by various seed functions (fig 6a )
eval_size.py: Produces the accuracy vs size plot in the paper (fig 8 in paper). It does not produce the CPMC results.
example.py: A simple example that visualizes some proposals (required matplotlib)
plot_box.py: Plots the bounding box results and evaluates the VUS for 2000 windows (run as "python plot_box.py place_to_store_result.dat output.pdf")
train_seed.py: Trains a seed function (see eval_learned for example how to use)
train_unary.py: Trains a set of foreground and background masks (see eval_learned for example how to use)


Example (c++):
$ build/examples/example path-to-image.png

Example (python):
$ cd src
$ python eval_baseline.py
Results in Table 1
$ python eval_learned.py
Results in Table 1 (or very close, depends on what you learn)

Example (matlab):
See example.m
Make sure to compile the code with '-DUSE_MATLAB=On'

Seed proposals
--------------
I added seed proposals in version 1.2 (a proposals mask containing just the seed itself). Those proposals seem to help with smaller objects for both the COCO and VOC dataset. If you're insterested in just bouding boxes I'd recomment NOT to use them, as most small bounding boxes are labeled as difficult and wont be evaluated!

License
-------

All code here is under a BSD license and can be used freely for academic and non-academic purposes. One exception are the saliency based seeds (saliency.cpp), which use the patented saliency filter.

