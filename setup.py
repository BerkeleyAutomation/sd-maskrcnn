from setuptools import setup

requirements = [
    'mask-rcnn',
]

with open('requirements.txt') as f:
    requirements += f.read().splitlines()

setup(name='sd_maskrcnn',
      version='0.1.0',
      description='SD Mask RCNN project code',
      author='Michael Danielczuk',
      author_email='mdanielczuk@berkeley.edu',
      packages=['sd_maskrcnn'],
      install_requires=requirements,
     )
