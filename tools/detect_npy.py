import os
import numpy as np
import tqdm
from autolab_core import YamlConfig, DepthImage, SegmentationImage
import skimage.io as io
from sd_maskrcnn.model import SDMaskRCNNModel
import sd_maskrcnn.utils as utils

if __name__ == "__main__":

    config = YamlConfig("cfg/detect_npy.yaml")

    # Create new directory for outputs
    utils.mkdir_if_missing(config["out_dir"])

    # Load model and save predictions
    model = SDMaskRCNNModel("inference", config["model"])
    for path in tqdm.tqdm(os.listdir(config["in_dir"])):
        out_name = os.path.splitext(path.split("_")[-1])[0]
        image = np.load(os.path.join(config["in_dir"], f"{path}")).squeeze()
        if config["crop"][0] > 0:
            image[:config["crop"][0]] = 0.0
        if config["crop"][1] > 0:
            image[-config["crop"][1]:] = 0.0
        if config["crop"][2] > 0:
            image[:, :config["crop"][2]] = 0.0
        if config["crop"][3] > 0:
            image[:, -config["crop"][3]:] = 0.0
        DepthImage(image).save(os.path.join(config["out_dir"], f"in_{out_name}.png"), min_depth=config["min_depth"] , max_depth=config["max_depth"])
        image[image > 0] = (image[image > 0] - config["min_depth"]) * (np.iinfo(np.uint8).max / (config["max_depth"] - config["min_depth"]))
        masks, mask_info = model.detect(image[..., None].astype(np.uint8))
        SegmentationImage((masks * np.arange(1, len(masks) + 1)[:, None, None]).max(0).astype(np.uint8)).to_color().save(os.path.join(config["out_dir"], f"seg_{out_name}.png"))