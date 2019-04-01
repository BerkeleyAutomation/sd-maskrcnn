"""Utility functions for performing data augmentations on images.
From https://github.com/BerkeleyAutomation/mech-search-siamese/blob/dev_dmwang/siamese/data_augmentation.py
"""

import cv2
import numpy as np
import scipy.ndimage
from skimage.transform import resize

def horizontal_flip(image):
    """
    Flips an image horizontally (reflect over the y-axis).
    Args:
      image: Numpy array of shape (w, h, c) representing the image.
    Returns:
      Flipped image of the same dimensions as the input image.
    """
    return image[:,::-1,:]

def vertical_flip(image):
    """
    Flips an image vertically (reflect over the y-axis).
    Args:
      image: Numpy array of shape (w, h, c) representing the image.
    Returns:
      Flipped image of the same dimensions as the input image.
    """
    return image[::-1,:,:]

def rotate(image, angle):
    """
    Rotates an image by the given angle.
    Args:
      image: Numpy array of shape (w, h, c) representing the image.
      angle: Float angle to rotate (in degrees).
    Returns:
      Rotated image of the same dimensions as the input image.
    """
    out = scipy.ndimage.rotate(image, angle, reshape=False)

    return out

# From https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions.
def scale(img, zoom_factor):
    """
    Scales an image by a factor (zoom in or out).
    Args:
      image: Numpy array of shape (w, h, c) representing the image.
      zoom_factor: Float of amount to scale (>1 means zoom in, <1 means zoom out).
    Returns:
      Scaled image of the same dimensions as the input image.
    """

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = scipy.ndimage.zoom(img, zoom_tuple)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = scipy.ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple)
        out = resize(out, img.shape)

    # If zoom_factor == 1, just return the input array
    else:
        out = img

    return out

def equalize_histogram(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output

def add_gaussian_noise(src):
    row,col,ch= src.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss.astype(np.int)

    clipped = np.clip(noisy, 0, 255)

    return clipped.astype(np.uint8)

def add_salt_pepper_noise(src):
    row,col,ch = src.shape
    s_vs_p = 0.5
    amount = 0.004
    out = src.copy()
    # Salt mode
    num_salt = np.ceil(amount * src.size * s_vs_p)
    coords = [np.random.randint(0, i-1 , int(num_salt))
                 for i in src.shape]
    out[coords[:-1]] = (255,255,255)

    # Pepper mode
    num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1 , int(num_pepper))
             for i in src.shape]
    out[coords[:-1]] = (0,0,0)
    return out

def transform_lighting(img_src):
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table
    gamma1 = 0.75
    gamma2 = 1.5

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )
    LUT_G1 = np.arange(256, dtype = 'uint8' )
    LUT_G2 = np.arange(256, dtype = 'uint8' )

    LUTs = []

    average_square = (10,10)

    for i in range(0, min_table):
        LUT_HC[i] = 0

    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table

    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # その他LUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255
        LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
        LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

    LUTs.append(LUT_HC)
    LUTs.append(LUT_LC)
    LUTs.append(LUT_G1)
    LUTs.append(LUT_G2)

    trans_img = []
    for i, LUT in enumerate(LUTs):
        trans_img.append( cv2.LUT(img_src, LUT))

    return trans_img
