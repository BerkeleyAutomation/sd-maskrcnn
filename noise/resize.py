import cv2
import numpy as np


def scale_to_square(im, dim=512):
    """Resizes an image to a square image of length dim."""
    scale = 512.0 / max(im.shape[0:2]) # scale so min dimension is 512
    scale_dim = tuple(reversed([int(np.ceil(d * scale)) for d in im.shape[:2]]))
    im = cv2.resize(im, scale_dim, interpolation=cv2.INTER_NEAREST)

    return im
