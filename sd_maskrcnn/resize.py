import os
from tqdm import tqdm
import cv2
import argparse
from autolab_core import YamlConfig
import utils

def resize_images(config):
    
    """Resizes all images so their maximum dimension is 512. Saves to new directory."""
    base_dir = config["dataset_path"]

    # directories of images that need resizing
    image_dir = config["img_dir"]
    mask_dir = config["mask_dir"]

    # output: resized images
    image_out_dir = config["img_out_dir"]
    utils.mkdir_if_missing(os.path.join(base_dir, image_out_dir))
    mask_out_dir = config["mask_out_dir"]
    utils.mkdir_if_missing(os.path.join(base_dir, mask_out_dir))

    old_im_path = os.path.join(base_dir, image_dir)
    new_im_path = os.path.join(base_dir, image_out_dir)
    old_mask_path = os.path.join(base_dir, mask_dir)
    new_mask_path = os.path.join(base_dir, mask_out_dir)
    for im_path in tqdm(os.listdir(old_im_path)):
        im_old_path = os.path.join(old_im_path, im_path)
        try:
            mask_old_path = os.path.join(old_mask_path, im_path)
        except:
            continue
        im = cv2.imread(im_old_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_old_path, cv2.IMREAD_UNCHANGED)
        if mask.shape[0] == 0 or mask.shape[1] == 0:
            print("mask empty")
            continue
        im = scale_to_square(im)
        mask = scale_to_square(mask)
        new_im_file = os.path.join(new_im_path, im_path)
        new_mask_file = os.path.join(new_mask_path, im_path)
        cv2.imwrite(new_im_file, im, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # 0 compression
        cv2.imwrite(new_mask_file, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def scale_to_square(im, dim=512):
    """Resizes an image to a square image of length dim."""
    scale = 512.0 / max(im.shape[0:2]) # scale so min dimension is 512
    scale_dim = tuple(reversed([int(np.ceil(d * scale)) for d in im.shape[:2]]))
    im = cv2.resize(im, scale_dim, interpolation=cv2.INTER_NEAREST)

    return im

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Resize images for SD Mask RCNN model")
    conf_parser.add_argument("--config", action="store", required=True,
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    resize_images(config)