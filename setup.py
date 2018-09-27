"""
The build/compilations setup
>> pip install -r requirements.txt
>> python setup.py install
"""

import pip
import logging
import pkg_resources
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
    install_reqs += ['mask-rcnn']
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setup(name='sd_maskrcnn',
    version='0.1.0',
    description='SD Mask RCNN project code',
    author='Michael Danielczuk',
    author_email='mdanielczuk@berkeley.edu',
    packages=['sd_maskrcnn'],
    install_requires=install_reqs,
)
