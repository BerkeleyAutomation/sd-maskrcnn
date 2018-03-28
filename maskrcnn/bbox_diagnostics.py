import numpy as np
import visualize
import imageio

base_path = "/nfs/diskstation/projects/dex-net/segmentation/datasets/pile_segmasks_01_28_18/"
mask_path = "semantic_segmasks/"
image_path = "depth_ims_resized/"

def print_mask_values(mask_name):
    mask_image = imageio.imread(base_path + mask_path + mask_name)
    print(np.unique(mask_image))
    print(mask_image.shape)


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


if __name__ == "__main__":
    # print_mask_values("image_009968.png")
    image = imageio.imread(base_path + image_path + "image_009968.png")
    masks = imageio.imread(base_path + mask_path + "image_009968.png")
    masks = masks.reshape(masks.shape + (1,))
    boxes = extract_bboxes(masks)
    visualize.display_instances(image, boxes, masks, None, None)
