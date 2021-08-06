import os
import time

import numpy as np
import skimage
import torchvision
from tqdm import tqdm

from autolab_core import YamlConfig
from . import utils


class SDMaskRCNNModel(object):
    def __init__(self, mode, config):

        self.mode = mode
        self.config = YamlConfig(config)

        if self.mode not in ["training", "inference"]:
            raise ValueError(
                "Can only create a model with mode inference or training"
            )
        if not os.path.exists(self.config["path"]):
            if self.mode == "inference":
                raise ValueError(
                    "No model located at {}".format(self.config["path"])
                )
            else:
                utils.mkdir_if_missing(self.config["path"])

        image_shape = self.config["settings"]["image_shape"]
        self.config["settings"]["image_min_dim"] = min(image_shape)
        self.config["settings"]["image_max_dim"] = max(image_shape)

        if self.mode == "inference":
            self.config["settings"]["gpu_count"] = 1
            self.config["settings"]["images_per_gpu"] = 1

        self._model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=2)
        exclude_layers = []
        if self.mode == "training":

            # Select weights file to load
            weights = self.config["weights"].lower()
            if weights == "coco":
                weights = os.path.join(
                    self.config["path"], "mask_rcnn_coco.h5"
                )
                # Download weights file
                if not os.path.exists(weights):
                    utilslib.download_trained_weights(weights)
                if self.config["settings"]["image_channel_count"] == 1:
                    exclude_layers = ["conv1"]
                # Exclude the last layers because they require a matching
                # number of classes
                exclude_layers += [
                    "mrcnn_class_logits",
                    "mrcnn_bbox_fc",
                    "mrcnn_bbox",
                    "mrcnn_mask",
                ]
            elif weights == "last":
                # Find last trained weights
                weights = self._model.find_last()
            elif weights == "imagenet":
                # Start from ImageNet trained weights
                weights = self._model.get_imagenet_weights()
                if self.config["settings"]["image_channel_count"] == 1:
                    exclude_layers = ["conv1"]
            elif weights != "new":
                weights = os.path.join(
                    self.config["path"], self.config["weights"]
                )
            else:
                weights = None

        elif self.mode == "inference":
            if self.config["weights"].lower() == "last":
                weights = self._model.find_last()
            else:
                weights = os.path.join(
                    self.config["path"], self.config["weights"]
                )

        # Load weights
        if weights is not None:
            print("Loading weights from {}".format(weights))
            self._model.load_weights(
                weights, by_name=True, exclude=exclude_layers
            )

    def detect(self, image, bin_mask=None, overlap_thresh=0.5):

        if self.mode != "inference":
            raise ValueError("Can only call detect in inference mode!")

        image, _, _, _, _ = utilslib.resize_image(
            image,
            min_dim=self.mask_config.IMAGE_MIN_DIM,
            min_scale=self.mask_config.IMAGE_MIN_SCALE,
            max_dim=self.mask_config.IMAGE_MAX_DIM,
            mode=self.mask_config.IMAGE_RESIZE_MODE,
        )

        # Run detection
        r = self._model.detect([image], verbose=0)[0]

        # If we choose to mask out bin pixels, load the bin masks and
        # transform them properly.
        # Then, delete the mask, score, class id, and bbox corresponding
        # to each mask that is entirely bin pixels.
        if bin_mask is not None:
            bin_mask, _, _, _, _ = utilslib.resize_image(
                bin_mask,
                min_dim=self.mask_config.IMAGE_MIN_DIM,
                min_scale=self.mask_config.IMAGE_MIN_SCALE,
                max_dim=self.mask_config.IMAGE_MAX_DIM,
                mode=self.mask_config.IMAGE_RESIZE_MODE,
            )

            bin_mask = bin_mask.squeeze()

            deleted_masks = []  # which segmasks are gonna be tossed?
            num_detects = r["masks"].shape[2]
            for k in range(num_detects):
                # compute the area of the overlap.
                inter = np.logical_and(bin_mask, r["masks"][:, :, k])
                frac_overlap = np.sum(inter) / np.sum(r["masks"][:, :, k])
                if frac_overlap <= overlap_thresh:
                    deleted_masks.append(k)

            r["masks"] = [
                r["masks"][:, :, k]
                for k in range(num_detects)
                if k not in deleted_masks
            ]
            r["masks"] = (
                np.stack(r["masks"], axis=2) if r["masks"] else np.array([])
            )
            r["rois"] = [
                r["rois"][k, :]
                for k in range(num_detects)
                if k not in deleted_masks
            ]
            r["rois"] = (
                np.stack(r["rois"], axis=0) if r["rois"] else np.array([])
            )
            r["class_ids"] = np.array(
                [
                    r["class_ids"][k]
                    for k in range(num_detects)
                    if k not in deleted_masks
                ]
            )
            r["scores"] = np.array(
                [
                    r["scores"][k]
                    for k in range(num_detects)
                    if k not in deleted_masks
                ]
            )

        masks = (
            np.stack([r["masks"][:, :, i] for i in range(r["masks"].shape[2])])
            if np.any(r["masks"])
            else np.array([])
        )
        mask_info = {
            "rois": r["rois"],
            "scores": r["scores"],
            "class_ids": r["class_ids"],
            "time": r["time"],
        }

        return masks, mask_info

    def detect_dataset(
        self, output_dir, dataset, bin_mask_dir=None, overlap_thresh=0.5
    ):

        # Create subdirectory for prediction masks
        pred_dir = os.path.join(output_dir, "pred_masks")
        utils.mkdir_if_missing(pred_dir)

        # Create subdirectory for prediction scores & bboxes
        pred_info_dir = os.path.join(output_dir, "pred_info")
        utils.mkdir_if_missing(pred_info_dir)

        # Create subdirectory for transformed GT segmasks
        resized_segmask_dir = os.path.join(
            output_dir, "modal_segmasks_processed"
        )
        utils.mkdir_if_missing(resized_segmask_dir)

        # Feed images into model one by one. For each image, predict and save.
        print("MAKING PREDICTIONS")
        times = []
        for image_id in tqdm(dataset.image_ids):
            # Load image and ground truth data and resize for net
            image, _, _, _, gt_mask = modellib.load_image_gt(
                dataset, self.mask_config, image_id, use_mini_mask=False
            )

            bin_mask = None
            if bin_mask_dir is not None:
                name = "image_{:06d}.png".format(
                    dataset.image_info[image_id]["id"]
                )
                bin_mask = skimage.io.imread(os.path.join(bin_mask_dir, name))[
                    ..., np.newaxis
                ]

            masks, mask_info = self.detect(image, bin_mask, overlap_thresh)

            # Save copy of transformed GT segmasks to disk in preparation for annotations
            mask_name = "image_{:06d}".format(image_id)
            mask_path = os.path.join(resized_segmask_dir, mask_name)

            # save the transpose so it's (n, h, w) instead of (h, w, n)
            np.save(mask_path, gt_mask.transpose(2, 0, 1))

            # Save masks
            save_masks_path = os.path.join(
                pred_dir, "image_{:06d}.npy".format(image_id)
            )
            np.save(save_masks_path, masks)

            # Save info
            r_info = {
                "rois": mask_info["rois"],
                "scores": mask_info["scores"],
                "class_ids": mask_info["class_ids"],
                "time": mask_info["time"],
            }
            times.append(mask_info["time"])
            r_info_path = os.path.join(
                pred_info_dir, "image_{:06d}.npy".format(image_id)
            )
            np.save(r_info_path, r_info)

        print("Took {} s".format(sum(times)))
        print("Saved prediction masks to:\t {}".format(pred_dir))
        print(
            "Saved prediction info (bboxes, scores, classes) to:\t {}".format(
                pred_info_dir
            )
        )
        print(
            "Saved transformed GT segmasks to:\t {}".format(
                resized_segmask_dir
            )
        )
        return pred_dir, pred_info_dir, resized_segmask_dir

    def train(self, train_dataset, val_dataset):

        if self.mode != "training":
            raise ValueError("Can only call train in training mode!")

        self.mask_config.display()
        self._model.train(
            train_dataset,
            val_dataset,
            learning_rate=self.mask_config.LEARNING_RATE,
            epochs=self.config["epochs"],
            layers="all",
        )

        # save in the models folder
        current_datetime = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(
            self.config["path"],
            "sd_mask_rcnn_{}_{}.h5".format(
                self.mask_config.NAME, current_datetime
            ),
        )
        self._model.keras_model.save_weights(model_path)
