import cv2
import numpy as np


def scale_to_square(im, dim=512):
    """Resizes an image to a square image of length dim."""
    scale = 512.0 / min(im.shape[0:2]) # scale so min dimension is 512
    scale_dim = tuple(reversed([int(np.ceil(d * scale)) for d in im.shape[:2]]))
    im = cv2.resize(im, scale_dim, interpolation=cv2.INTER_NEAREST)
    y_margin = abs(im.shape[0] - 512) // 2
    x_margin = abs(im.shape[1] - 512) // 2

    check_y = 512 - (im.shape[0] - y_margin - y_margin)
    check_x = 512 - (im.shape[1] - x_margin - x_margin)

    im = im[y_margin : im.shape[0] - y_margin + check_y, x_margin : im.shape[1] - x_margin + check_x]
