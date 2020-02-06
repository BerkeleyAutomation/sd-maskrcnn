"""
Setup of SD Mask RCNN codebase

Author: Mike Danielczuk
"""

import os
from setuptools import setup

# load __version__
version_file = 'sd_maskrcnn/version.py'
exec(open(version_file).read())

# load README.md as long_description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

setup_requirements = [
    'Cython',                
    'numpy'
]

requirements = [
    'pycocotools>=2.0',         # For benchmarking
    'scikit-image>=0.14.2',     # For image loading
    'keras>=2.2',               # For training
    'tqdm',                     # For pretty progress bars
    'matplotlib',               # For visualization of results
    'autolab_core>=0.0.9',      # For core utilities
    'autolab-perception',       # For image wrapping
    'tensorflow-gpu<1.13>=1.10'      # For training
]

generation_requirements = [
    'gym>=0.11',             # For sampling heaps
    'pyglet==1.4.0b1',       # For pyrender  
    'pyrender>=0.1.23',      # For rendering images
    'pybullet',              # For dynamic sim
    'trimesh[easy]',         # For mesh loading/exporting
    'scipy'                  # For random vars
]

setup(
    name='sd_maskrcnn',
    version=__version__,
    description='SD Mask RCNN project code',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Michael Danielczuk',
    author_email='mdanielczuk@berkeley.edu',
    license='MIT',
    url='http://github.com/BerkeleyAutomation/sd-maskrcnn',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    packages=['sd_maskrcnn', 'sd_maskrcnn.envs'],
    package_data={'sd_maskrcnn': ['data/plane/*', 'data/bin/*']},
    setup_requires=setup_requirements,
    install_requires=requirements,
    extras_require={
        'generation': generation_requirements
    }
)
