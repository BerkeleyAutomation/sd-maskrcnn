import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
import autolab_core
import cv2
from perception import DepthImage


NUM_IMAGES = 1


def analyze_image_depths(path, bbox):
    """
    path should lead to a .npy file
    """
    print(path)
    img = np.load(path)
    img = np.reshape(img, img.shape[:2])

    img_slice = img[bbox[0] : bbox[2], bbox[1]: bbox[3]]
    vec = np.ndarray.flatten(img_slice)

    depth_img = DepthImage(img)

    n, bins, patches = plt.hist(vec, 10, density=True, facecolor="blue")

    plt.xlabel("depth value")
    plt.ylabel("count")
    plt.title("depth within region")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    base_path = "images/"
    for i in range(NUM_IMAGES):
        img_name = "depth_" + str(i) + ".npy"
        img_path = os.path.join(base_path, img_name)
        analyze_image_depths(img_path, [182, 333, 243, 396])
