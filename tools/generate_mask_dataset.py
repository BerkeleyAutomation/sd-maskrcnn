"""
Copyright ©2019. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Generates a dataset for training SD Mask R-CNN
Authors: Jeff Mahler, Mike Danielczuk
"""

import argparse
import gc
import json
import os
import shutil
import time
import traceback

import autolab_core.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from autolab_core import (
    BinaryImage,
    ColorImage,
    DepthImage,
    GrayscaleImage,
    Logger,
    TensorDataset,
    YamlConfig,
)

from sd_maskrcnn.envs import BinHeapEnv
from sd_maskrcnn.envs.constants import (
    ENVIRONMENT_KEY,
    JSON_INDENT,
    POINT_DIM,
    POSE_DIM,
    TRAIN_ID,
)

SEED = 744

# set up logger
logger = Logger.get_logger("tools/generate_segmask_dataset.py")


def generate_segmask_dataset(
    output_dataset_path, config, save_tensors=True, warm_start=False
):
    """Generate a segmentation training dataset

    Parameters
    ----------
    dataset_path : str
        path to store the dataset
    config : dict
        dictionary-like objects containing parameters of the simulator and visualization
    save_tensors : bool
        save tensor datasets (for recreating state)
    warm_start : bool
        restart dataset generation from a previous state
    """

    # read subconfigs
    dataset_config = config["dataset"]
    image_config = config["images"]
    vis_config = config["vis"]

    # debugging
    debug = config["debug"]
    if debug:
        np.random.seed(SEED)

    # read general parameters
    num_states = config["num_states"]
    num_images_per_state = config["num_images_per_state"]

    states_per_flush = config["states_per_flush"]
    states_per_garbage_collect = config["states_per_garbage_collect"]

    # set max obj per state
    max_objs_per_state = config["state_space"]["heap"]["max_objs"]

    # read image parameters
    im_height = config["state_space"]["camera"]["im_height"]
    im_width = config["state_space"]["camera"]["im_width"]
    segmask_channels = max_objs_per_state + 1

    # create the dataset path and all subfolders if they don't exist
    if not os.path.exists(output_dataset_path):
        os.mkdir(output_dataset_path)

    image_dir = os.path.join(output_dataset_path, "images")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    color_dir = os.path.join(image_dir, "color_ims")
    if image_config["color"] and not os.path.exists(color_dir):
        os.mkdir(color_dir)
    depth_dir = os.path.join(image_dir, "depth_ims")
    if image_config["depth"] and not os.path.exists(depth_dir):
        os.mkdir(depth_dir)
    amodal_dir = os.path.join(image_dir, "amodal_masks")
    if image_config["amodal"] and not os.path.exists(amodal_dir):
        os.mkdir(amodal_dir)
    modal_dir = os.path.join(image_dir, "modal_masks")
    if image_config["modal"] and not os.path.exists(modal_dir):
        os.mkdir(modal_dir)
    semantic_dir = os.path.join(image_dir, "semantic_masks")
    if image_config["semantic"] and not os.path.exists(semantic_dir):
        os.mkdir(semantic_dir)

    # setup logging
    experiment_log_filename = os.path.join(
        output_dataset_path, "dataset_generation.log"
    )
    if os.path.exists(experiment_log_filename) and not warm_start:
        os.remove(experiment_log_filename)
    Logger.add_log_file(logger, experiment_log_filename, global_log_file=True)
    config.save(
        os.path.join(output_dataset_path, "dataset_generation_params.yaml")
    )
    metadata = {}
    num_prev_states = 0

    # set dataset params
    if save_tensors:

        # read dataset subconfigs
        state_dataset_config = dataset_config["states"]
        image_dataset_config = dataset_config["images"]
        state_tensor_config = state_dataset_config["tensors"]
        image_tensor_config = image_dataset_config["tensors"]

        obj_pose_dim = POSE_DIM * max_objs_per_state
        obj_com_dim = POINT_DIM * max_objs_per_state
        state_tensor_config["fields"]["obj_poses"]["height"] = obj_pose_dim
        state_tensor_config["fields"]["obj_coms"]["height"] = obj_com_dim
        state_tensor_config["fields"]["obj_ids"]["height"] = max_objs_per_state

        image_tensor_config["fields"]["camera_pose"]["height"] = POSE_DIM

        if image_config["color"]:
            image_tensor_config["fields"]["color_im"] = {
                "dtype": "uint8",
                "channels": 3,
                "height": im_height,
                "width": im_width,
            }

        if image_config["depth"]:
            image_tensor_config["fields"]["depth_im"] = {
                "dtype": "float32",
                "channels": 1,
                "height": im_height,
                "width": im_width,
            }

        if image_config["modal"]:
            image_tensor_config["fields"]["modal_segmasks"] = {
                "dtype": "uint8",
                "channels": segmask_channels,
                "height": im_height,
                "width": im_width,
            }

        if image_config["amodal"]:
            image_tensor_config["fields"]["amodal_segmasks"] = {
                "dtype": "uint8",
                "channels": segmask_channels,
                "height": im_height,
                "width": im_width,
            }

        if image_config["semantic"]:
            image_tensor_config["fields"]["semantic_segmasks"] = {
                "dtype": "uint8",
                "channels": 1,
                "height": im_height,
                "width": im_width,
            }

        # create dataset filenames
        state_dataset_path = os.path.join(output_dataset_path, "state_tensors")
        image_dataset_path = os.path.join(output_dataset_path, "image_tensors")

        if warm_start:

            if not os.path.exists(state_dataset_path) or not os.path.exists(
                image_dataset_path
            ):
                logger.error(
                    "Attempting to warm start without saved tensor dataset"
                )
                exit(1)

            # open datasets
            logger.info("Opening state dataset")
            state_dataset = TensorDataset.open(
                state_dataset_path, access_mode="READ_WRITE"
            )
            logger.info("Opening image dataset")
            image_dataset = TensorDataset.open(
                image_dataset_path, access_mode="READ_WRITE"
            )

            # read configs
            state_tensor_config = state_dataset.config
            image_tensor_config = image_dataset.config

            # clean up datasets (there may be datapoints with indices corresponding to non-existent data)
            num_state_datapoints = state_dataset.num_datapoints
            num_image_datapoints = image_dataset.num_datapoints
            num_prev_states = num_state_datapoints

            # clean up images
            image_ind = num_image_datapoints - 1
            image_datapoint = image_dataset[image_ind]
            while (
                image_ind > 0
                and image_datapoint["state_ind"] >= num_state_datapoints
            ):
                image_ind -= 1
                image_datapoint = image_dataset[image_ind]
            images_to_remove = num_image_datapoints - 1 - image_ind
            logger.info("Deleting last %d image tensors" % (images_to_remove))
            if images_to_remove > 0:
                image_dataset.delete_last(images_to_remove)
                num_image_datapoints = image_dataset.num_datapoints
        else:
            # create datasets from scratch
            logger.info("Creating datasets")

            state_dataset = TensorDataset(
                state_dataset_path, state_tensor_config
            )
            image_dataset = TensorDataset(
                image_dataset_path, image_tensor_config
            )

        # read templates
        state_datapoint = state_dataset.datapoint_template
        image_datapoint = image_dataset.datapoint_template

    if warm_start:

        if not os.path.exists(
            os.path.join(output_dataset_path, "metadata.json")
        ):
            logger.error(
                "Attempting to warm start without previously created dataset"
            )
            exit(1)

        # Read metadata and indices
        metadata = json.load(
            open(os.path.join(output_dataset_path, "metadata.json"), "r")
        )
        test_inds = np.load(
            os.path.join(image_dir, "test_indices.npy")
        ).tolist()
        train_inds = np.load(
            os.path.join(image_dir, "train_indices.npy")
        ).tolist()

        # set obj ids and splits
        reverse_obj_ids = metadata["obj_ids"]
        obj_id_map = utils.reverse_dictionary(reverse_obj_ids)
        obj_splits = metadata["obj_splits"]
        obj_keys = obj_splits.keys()
        mesh_filenames = metadata["meshes"]

        # Get list of images generated so far
        generated_images = (
            sorted(os.listdir(color_dir))
            if image_config["color"]
            else sorted(os.listdir(depth_dir))
        )
        num_total_images = len(generated_images)

        # Do our own calculation if no saved tensors
        if num_prev_states == 0:
            num_prev_states = num_total_images // num_images_per_state

        # Find images to remove and remove them from all relevant places if they exist
        num_images_to_remove = num_total_images - (
            num_prev_states * num_images_per_state
        )
        logger.info(
            "Deleting last {} invalid images".format(num_images_to_remove)
        )
        for k in range(num_images_to_remove):
            im_name = generated_images[-(k + 1)]
            im_basename = os.path.splitext(im_name)[0]
            im_ind = int(im_basename.split("_")[1])
            if os.path.exists(os.path.join(depth_dir, im_name)):
                os.remove(os.path.join(depth_dir, im_name))
            if os.path.exists(os.path.join(color_dir, im_name)):
                os.remove(os.path.join(color_dir, im_name))
            if os.path.exists(os.path.join(semantic_dir, im_name)):
                os.remove(os.path.join(semantic_dir, im_name))
            if os.path.exists(os.path.join(modal_dir, im_basename)):
                shutil.rmtree(os.path.join(modal_dir, im_basename))
            if os.path.exists(os.path.join(amodal_dir, im_basename)):
                shutil.rmtree(os.path.join(amodal_dir, im_basename))
            if im_ind in train_inds:
                train_inds.remove(im_ind)
            elif im_ind in test_inds:
                test_inds.remove(im_ind)

    else:

        # Create initial env to generate metadata
        env = BinHeapEnv(config)
        obj_id_map = env.state_space.obj_id_map
        obj_keys = env.state_space.obj_keys
        obj_splits = env.state_space.obj_splits
        mesh_filenames = env.state_space.mesh_filenames
        save_obj_id_map = obj_id_map.copy()
        save_obj_id_map[ENVIRONMENT_KEY] = np.iinfo(np.uint32).max
        reverse_obj_ids = utils.reverse_dictionary(save_obj_id_map)
        metadata["obj_ids"] = reverse_obj_ids
        metadata["obj_splits"] = obj_splits
        metadata["meshes"] = mesh_filenames
        json.dump(
            metadata,
            open(os.path.join(output_dataset_path, "metadata.json"), "w"),
            indent=JSON_INDENT,
            sort_keys=True,
        )
        train_inds = []
        test_inds = []

    # generate states and images
    state_id = num_prev_states
    while state_id < num_states:

        # create env and set objects
        create_start = time.time()
        env = BinHeapEnv(config)
        env.state_space.obj_id_map = obj_id_map
        env.state_space.obj_keys = obj_keys
        env.state_space.set_splits(obj_splits)
        env.state_space.mesh_filenames = mesh_filenames
        create_stop = time.time()
        logger.info(
            "Creating env took %.3f sec" % (create_stop - create_start)
        )

        # sample states
        states_remaining = num_states - state_id
        for i in range(min(states_per_garbage_collect, states_remaining)):

            # log current rollout
            if state_id % config["log_rate"] == 0:
                logger.info("State: %06d" % (state_id))

            try:
                # reset env
                env.reset()
                state = env.state
                split = state.metadata["split"]

                # render state
                if vis_config["state"]:
                    env.view_3d_scene()

                # Save state if desired
                if save_tensors:

                    # set obj state variables
                    obj_pose_vec = np.zeros(obj_pose_dim)
                    obj_com_vec = np.zeros(obj_com_dim)
                    obj_id_vec = np.iinfo(np.uint32).max * np.ones(
                        max_objs_per_state
                    )
                    j = 0
                    for obj_state in state.obj_states:
                        obj_pose_vec[
                            j * POSE_DIM : (j + 1) * POSE_DIM
                        ] = obj_state.pose.vec
                        obj_com_vec[
                            j * POINT_DIM : (j + 1) * POINT_DIM
                        ] = obj_state.center_of_mass
                        obj_id_vec[j] = int(obj_id_map[obj_state.key])
                        j += 1

                    # store datapoint env params
                    state_datapoint["state_id"] = state_id
                    state_datapoint["obj_poses"] = obj_pose_vec
                    state_datapoint["obj_coms"] = obj_com_vec
                    state_datapoint["obj_ids"] = obj_id_vec
                    state_datapoint["split"] = split

                    # store state datapoint
                    image_start_ind = image_dataset.num_datapoints
                    image_end_ind = image_start_ind + num_images_per_state
                    state_datapoint["image_start_ind"] = image_start_ind
                    state_datapoint["image_end_ind"] = image_end_ind

                    # clean up
                    del obj_pose_vec
                    del obj_com_vec
                    del obj_id_vec

                    # add state
                    state_dataset.add(state_datapoint)

                # render images
                for k in range(num_images_per_state):

                    # reset the camera
                    if num_images_per_state > 1:
                        env.reset_camera()

                    obs = env.render_camera_image(color=image_config["color"])
                    if image_config["color"]:
                        color_obs, depth_obs = obs
                    else:
                        depth_obs = obs

                    # vis obs
                    if vis_config["obs"]:
                        if image_config["depth"]:
                            plt.figure()
                            plt.imshow(depth_obs)
                            plt.title("Depth Observation")
                        if image_config["color"]:
                            plt.figure()
                            plt.imshow(color_obs)
                            plt.title("Color Observation")
                        plt.show()

                    if (
                        image_config["modal"]
                        or image_config["amodal"]
                        or image_config["semantic"]
                    ):

                        # render segmasks
                        (
                            amodal_segmasks,
                            modal_segmasks,
                        ) = env.render_segmentation_images()

                        # retrieve segmask data
                        modal_segmask_arr = np.iinfo(np.uint8).max * np.ones(
                            [im_height, im_width, segmask_channels],
                            dtype=np.uint8,
                        )
                        amodal_segmask_arr = np.iinfo(np.uint8).max * np.ones(
                            [im_height, im_width, segmask_channels],
                            dtype=np.uint8,
                        )
                        stacked_segmask_arr = np.zeros(
                            [im_height, im_width, 1], dtype=np.uint8
                        )

                        modal_segmask_arr[
                            :, :, : env.num_objects
                        ] = modal_segmasks
                        amodal_segmask_arr[
                            :, :, : env.num_objects
                        ] = amodal_segmasks

                        if image_config["semantic"]:
                            for j in range(env.num_objects):
                                this_obj_px = np.where(
                                    modal_segmasks[:, :, j] > 0
                                )
                                stacked_segmask_arr[
                                    this_obj_px[0], this_obj_px[1], 0
                                ] = (j + 1)

                    # visualize
                    if vis_config["semantic"]:
                        plt.figure()
                        plt.imshow(stacked_segmask_arr.squeeze())
                        plt.show()

                    if save_tensors:
                        # save image data as tensors
                        if image_config["color"]:
                            image_datapoint["color_im"] = color_obs
                        if image_config["depth"]:
                            image_datapoint["depth_im"] = depth_obs[:, :, None]
                        if image_config["modal"]:
                            image_datapoint[
                                "modal_segmasks"
                            ] = modal_segmask_arr
                        if image_config["amodal"]:
                            image_datapoint[
                                "amodal_segmasks"
                            ] = amodal_segmask_arr
                        if image_config["semantic"]:
                            image_datapoint[
                                "semantic_segmasks"
                            ] = stacked_segmask_arr

                        image_datapoint["camera_pose"] = env.camera.pose.vec
                        image_datapoint[
                            "camera_intrs"
                        ] = env.camera.intrinsics.vec
                        image_datapoint["state_ind"] = state_id
                        image_datapoint["split"] = split

                        # add image
                        image_dataset.add(image_datapoint)

                    # Save depth image and semantic masks
                    if image_config["color"]:
                        ColorImage(color_obs).save(
                            os.path.join(
                                color_dir,
                                "image_{:06d}.png".format(
                                    num_images_per_state * state_id + k
                                ),
                            )
                        )
                    if image_config["depth"]:
                        DepthImage(depth_obs).save(
                            os.path.join(
                                depth_dir,
                                "image_{:06d}.png".format(
                                    num_images_per_state * state_id + k
                                ),
                            )
                        )
                    if image_config["modal"]:
                        modal_id_dir = os.path.join(
                            modal_dir,
                            "image_{:06d}".format(
                                num_images_per_state * state_id + k
                            ),
                        )
                        if not os.path.exists(modal_id_dir):
                            os.mkdir(modal_id_dir)
                        for i in range(env.num_objects):
                            BinaryImage(modal_segmask_arr[:, :, i]).save(
                                os.path.join(
                                    modal_id_dir,
                                    "channel_{:03d}.png".format(i),
                                )
                            )
                    if image_config["amodal"]:
                        amodal_id_dir = os.path.join(
                            amodal_dir,
                            "image_{:06d}".format(
                                num_images_per_state * state_id + k
                            ),
                        )
                        if not os.path.exists(amodal_id_dir):
                            os.mkdir(amodal_id_dir)
                        for i in range(env.num_objects):
                            BinaryImage(amodal_segmask_arr[:, :, i]).save(
                                os.path.join(
                                    amodal_id_dir,
                                    "channel_{:03d}.png".format(i),
                                )
                            )
                    if image_config["semantic"]:
                        GrayscaleImage(stacked_segmask_arr.squeeze()).save(
                            os.path.join(
                                semantic_dir,
                                "image_{:06d}.png".format(
                                    num_images_per_state * state_id + k
                                ),
                            )
                        )

                    # Save split
                    if split == TRAIN_ID:
                        train_inds.append(num_images_per_state * state_id + k)
                    else:
                        test_inds.append(num_images_per_state * state_id + k)

                # auto-flush after every so many timesteps
                if state_id % states_per_flush == 0:
                    np.save(
                        os.path.join(image_dir, "train_indices.npy"),
                        train_inds,
                    )
                    np.save(
                        os.path.join(image_dir, "test_indices.npy"), test_inds
                    )
                    if save_tensors:
                        state_dataset.flush()
                        image_dataset.flush()

                # delete action objects
                for obj_state in state.obj_states:
                    del obj_state
                del state
                gc.collect()

                # update state id
                state_id += 1

            except Exception as e:
                # log an error
                logger.warning("Heap failed!")
                logger.warning("%s" % (str(e)))
                logger.warning(traceback.print_exc())
                if debug:
                    raise

                del env
                gc.collect()
                env = BinHeapEnv(config)
                env.state_space.obj_id_map = obj_id_map
                env.state_space.obj_keys = obj_keys
                env.state_space.set_splits(obj_splits)
                env.state_space.mesh_filenames = mesh_filenames

        # garbage collect
        del env
        gc.collect()

    # write all datasets to file, save indices
    np.save(os.path.join(image_dir, "train_indices.npy"), train_inds)
    np.save(os.path.join(image_dir, "test_indices.npy"), test_inds)
    if save_tensors:
        state_dataset.flush()
        image_dataset.flush()

    logger.info(
        "Generated %d image datapoints" % (state_id * num_images_per_state)
    )


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(
        description="Generate a training dataset for a SD Mask R-CNN"
    )
    parser.add_argument(
        "output_dataset_path",
        type=str,
        default=None,
        help="directory to store a dataset containing the images",
    )
    parser.add_argument(
        "--config_filename",
        type=str,
        default=None,
        help="configuration file to use",
    )
    parser.add_argument(
        "--save_tensors",
        action="store_true",
        help="whether to save raw tensors to recreate the state",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="warm start system after crash",
    )

    args = parser.parse_args()
    output_dataset_path = args.output_dataset_path
    config_filename = args.config_filename
    save_tensors = args.save_tensors
    warm_start = args.warm_start

    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "cfg/generate_mask_dataset.yaml",
        )

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # open config file
    config = YamlConfig(config_filename)

    # generate dataset
    generation_start = time.time()
    generate_segmask_dataset(
        output_dataset_path,
        config,
        save_tensors=save_tensors,
        warm_start=warm_start,
    )

    # log time
    generation_stop = time.time()
    logger.info(
        "Mask dataset generation took %.3f sec"
        % (generation_stop - generation_start)
    )
