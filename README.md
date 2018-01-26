### Code for running Mask-RCNN on images of object piles.

1. Code for running training. 
`CUDA_VISIBLE_DEVICES='2' PYTHONPATH='.:maskrcnn/' python clutter/train_clutter.py --logdir outputs/v3_512_40_flip_depth --im_type depth --task train`

2. Code for benchmarking.
`CUDA_VISIBLE_DEVICES='2' PYTHONPATH='.:maskrcnn/' python clutter/train_clutter.py --logdir outputs/v3_512_40_flip_depth --im_type depth --task bench`
