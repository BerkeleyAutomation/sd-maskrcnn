import numpy as np
from scipy import ndimage
from perception import BinaryImage
import os

# Operate in current directory
for fn in os.listdir('./segmasks'):
    bi = BinaryImage.open('./segmasks/{}'.format(fn))
    bi.data = ndimage.binary_fill_holes(bi.data).astype(np.uint8) * 255
    bi.save('./segmasks_filled/{}'.format(fn))

