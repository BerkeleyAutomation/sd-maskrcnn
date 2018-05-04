# Defines data augmentation functions and provides a
# method of composing them to operate upon lists of images.

import numpy as np
import skimage.color
from perception import DepthImage


def inject_noise(img, noise_level=0.0005):
    """
    Add a Gaussian noise to the image.
    """
    means = np.zeros(img.shape)
    std_devs = np.full(img.shape, noise_level)
    noise = np.random.normal(means, std_devs)

    # don't apply noise to some pixels
    # noise[img <= noise_threshold] = 0.0
    return img + noise


def inpaint(img):
    """
    Inpaint the image
    """
    # create DepthImage from gray version of img
    gray_img = skimage.color.rgb2gray(img)
    depth_img = DepthImage(gray_img)

    # zero out high-gradient areas and inpaint
    thresh_img = depth_img.threshold_gradients_pctile(0.95)
    inpaint_img = thresh_img.inpaint()
    return inpaint_img.data


# def augmenter(fn_lst):
#     """
#     Return a composition of all the desired augmentation functions.
#     """
#     if not fn_lst:
#         return lambda img: img
#     fn = fn_lst.pop(0)
#     return lambda img: augmenter(fn_lst)(fn(img))


def augment_img(img, config):
    """
    Compose augmentations.
    """
    if config["with_inpainting"]:
        img = inpaint(img)
    if config["with_noise"]:
        noise_level = config["noise_level"]
        img = inject_noise(img, noise_level)
    return img
