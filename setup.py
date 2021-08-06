"""
Setup of SD Mask RCNN codebase

Author: Mike Danielczuk
"""

import sys
import os
from setuptools import setup

root_dir = os.path.dirname(os.path.realpath(__file__))

setup_requirements = set(["Cython", "numpy", "nvidia-pyindex"])
sub_requirements = set([f"mask-rcnn @ file://localhost{root_dir}/maskrcnn"])
requirements = set(
    [
        "pycocotools>=2.0",  # For benchmarking
        "scikit-image>=0.14.2",  # For image loading
        "keras>=2.2<2.3",  # For training
        "tqdm",  # For pretty progress bars
        "matplotlib",  # For visualization of results
        "h5py<3.0.0",  # Loading pretrained models
        "autolab_core>=1.1.0",  # For core utilities
        "torch",  # For training
        "torchvision",  # For models
    ]
)

generation_requirements = requirements.union(
    [
        "gym>=0.11",  # For sampling heaps
        "pyglet==1.4.0b1",  # For pyrender
        "pyrender>=0.1.23",  # For rendering images
        "pybullet",  # For dynamic sim
        "trimesh[easy]",  # For mesh loading/exporting
        "scipy",  # For random vars
    ]
)

# if someone wants to output a requirements file
# `python setup.py --list-train > requirements.txt`
# NOTE: maskrcnn lib must be installed separately
if "--list-setup" in sys.argv:
    print("\n".join(setup_requirements))
    exit()
elif "--list-train" in sys.argv:
    print("\n".join(requirements))
    exit()
elif "--list-gen" in sys.argv:
    print("\n".join(generation_requirements.union(requirements)))
    exit()

# load __version__
version_file = "sd_maskrcnn/version.py"
exec(open(version_file).read())

# load README.md as long_description
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        long_description = f.read()

setup(
    name="sd_maskrcnn",
    version=__version__,
    description="SD Mask RCNN project code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michael Danielczuk",
    author_email="mdanielczuk@berkeley.edu",
    license="MIT",
    url="http://github.com/BerkeleyAutomation/sd-maskrcnn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=["sd_maskrcnn", "sd_maskrcnn.envs"],
    package_data={"sd_maskrcnn": ["data/plane/*", "data/bin/*"]},
    setup_requires=list(setup_requirements),
    install_requires=list(requirements.union(sub_requirements)),
    extras_require={"generation": list(generation_requirements)},
)
