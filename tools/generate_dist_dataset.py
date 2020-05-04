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

Generates a distribution dataset for training X-Ray
Authors: Jeff Mahler, Mike Danielczuk
"""

import argparse
import gc
import numpy as np
import os
import json
import shutil
import time
import traceback
import matplotlib.pyplot as plt

from autolab_core import YamlConfig, Logger
import autolab_core.utils as utils
from perception import DepthImage, GrayscaleImage, BinaryImage, ColorImage

from sd_maskrcnn.envs import BinHeapEnv
from sd_maskrcnn.envs.constants import *

SEED = 744

# set up logger
logger = Logger.get_logger('tools/generate_distribution_dataset.py')

def generate_distribution_dataset(output_dataset_path, 
                                  config, warm_start=False):
    """ Generate a distribution training dataset

    Parameters
    ----------
    dataset_path : str
        path to store the dataset
    config : dict
        dictionary-like objects containing parameters of the 
        simulator and visualization
    warm_start : bool
        restart dataset generation from a previous state
    """
    
    # read subconfigs
    image_config = config['images']
    vis_config = config['vis']

    # debugging
    debug = config['debug']
    if debug:
        np.random.seed(SEED)
    
    # read general parameters
    num_states = config['num_states']
    num_images_per_state = config['num_images_per_state']
    states_per_flush = config['states_per_flush']
    states_per_garbage_collect = config['states_per_garbage_collect']
    max_objs_per_state = config['state_space']['heap']['max_objs']
    im_height = config['state_space']['camera']['im_height']
    im_width = config['state_space']['camera']['im_width']

    # setup target keys
    target_keys = []
    box_keys = []
    for ratio in config['distributions']['ratios']:
        target_keys.append('boxes~box_{}'.format(ratio))
        box_keys.append('box_{}'.format(ratio))
    config['state_space']['heap']['objects']['target_keys'] = target_keys
    config['state_space']['heap']['objects']['object_keys']['boxes'] = box_keys
    
    # create the dataset path and all subfolders if they don't exist
    if not os.path.exists(output_dataset_path):
        os.mkdir(output_dataset_path)

    image_dir = os.path.join(output_dataset_path, 'images')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    color_dir = os.path.join(image_dir, 'color_ims')
    if image_config['color'] and not os.path.exists(color_dir):
        os.mkdir(color_dir)
    depth_dir = os.path.join(image_dir, 'depth_ims')
    if image_config['depth'] and not os.path.exists(depth_dir):
        os.mkdir(depth_dir)
    combo_dir = os.path.join(image_dir, 'combo_ims')
    if image_config['combo'] and not os.path.exists(combo_dir):
        os.mkdir(combo_dir)
    dist_dir = os.path.join(image_dir, 'dist_ims')
    if image_config['dist'] and not os.path.exists(dist_dir):
        os.mkdir(dist_dir)

    # setup logging
    experiment_log_filename = os.path.join(output_dataset_path, 'dataset_generation.log')
    if os.path.exists(experiment_log_filename) and not warm_start:
        os.remove(experiment_log_filename)
    Logger.add_log_file(logger, experiment_log_filename, global_log_file=True)
    config.save(os.path.join(output_dataset_path, 'dataset_generation_params.yaml'))
    metadata = {}
    num_prev_states = 0

    if warm_start:

        if not os.path.exists(os.path.join(output_dataset_path, 'metadata.json')):
            logger.error('Attempting to warm start without previously created dataset')
            exit(1)

        # Read metadata and indices
        metadata = json.load(open(os.path.join(output_dataset_path, 'metadata.json'), 'r'))
        test_inds = np.load(os.path.join(image_dir, 'test_indices.npy')).tolist()
        train_inds = np.load(os.path.join(image_dir, 'train_indices.npy')).tolist()

        # set obj ids and splits
        reverse_obj_ids = metadata['obj_ids']
        obj_id_map = utils.reverse_dictionary(reverse_obj_ids)
        obj_splits = metadata['obj_splits']
        obj_keys = obj_splits.keys()
        mesh_filenames = metadata['meshes']

        # Get list of images generated so far
        generated_images = sorted(os.listdir(color_dir)) if image_config['color'] else sorted(os.listdir(depth_dir))
        num_total_images = len(generated_images)

        # Do our own calculation if no saved tensors
        if num_prev_states == 0:
            num_prev_states = num_total_images // num_images_per_state
        
        # Find images to remove and remove them from all relevant places if they exist
        num_images_to_remove = num_total_images - (num_prev_states * num_images_per_state)
        logger.info('Deleting last {} invalid images'.format(num_images_to_remove))
        for k in range(num_images_to_remove):
            im_name = generated_images[-(k+1)]
            im_basename = os.path.splitext(im_name)[0]
            im_ind = int(im_basename.split('_')[1])
            if os.path.exists(os.path.join(depth_dir, im_name)):
                os.remove(os.path.join(depth_dir, im_name))
            if os.path.exists(os.path.join(color_dir, im_name)):
                os.remove(os.path.join(color_dir, im_name))
            if os.path.exists(os.path.join(combo_dir, im_name)):
                os.remove(os.path.join(combo_dir, im_name))
            if os.path.exists(os.path.join(dist_dir, im_name)):
                os.remove(os.path.join(dist_dir, im_name))
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
        metadata['obj_ids'] = reverse_obj_ids
        metadata['obj_splits'] = obj_splits
        metadata['meshes'] = mesh_filenames
        json.dump(metadata, open(os.path.join(output_dataset_path, 'metadata.json'), 'w'),
                  indent=JSON_INDENT, sort_keys=True)
        train_inds = []
        test_inds = []
    
    # generate states and images
    state_id = num_prev_states
    gen_start = time.time()
    flush_start = time.time()
    while state_id < num_states:

        # create env and set objects
        create_start = time.time()
        env = BinHeapEnv(config)
        env.state_space.obj_id_map = obj_id_map
        env.state_space.obj_keys = obj_keys
        env.state_space.set_splits(obj_splits)
        env.state_space.mesh_filenames = mesh_filenames
        create_stop = time.time()
        logger.debug('Creating env took {:.3f} sec'.format(create_stop-create_start))           

        # sample states
        states_remaining = num_states - state_id
        for i in range(min(states_per_garbage_collect, states_remaining)):
            
            # log current rollout
            if state_id % config['log_rate'] == 0:
                logger.info('State: {:06d}'.format(state_id))

            try:    
                # reset env
                sample_start = time.time()
                env.reset()
                logger.info('Sampling heap took {:.3f} sec'.format(time.time() - 
                                                                   sample_start))
                state = env.state
                split = state.metadata['split']
                
                # render state
                if vis_config['state']:
                    env.view_3d_scene()

                # render images
                render_start = time.time()
                for k in range(num_images_per_state):
                    
                    # reset the camera
                    if k > 0:
                        env.reset_camera()
                    
                    obs = env.render_camera_image(color=image_config['color'], 
                                                  render_bin=False)
                    color_obs, depth_obs, combo_obs = obs
                    
                    dist_start = time.time()
                    dist_im = env.find_target_ar_distribution(stride=config['distributions']['stride'], 
                                                              rotations=config['distributions']['rotations'])
                    logger.debug('Computing distribution took {:.3f} sec'.format(time.time() - 
                                                                                 dist_start))

                    # vis obs
                    if vis_config['obs']:
                        if image_config['depth']:
                            plt.figure()
                            plt.imshow(depth_obs)
                            plt.title('Depth Observation')
                        if image_config['color']:
                            plt.figure()
                            plt.imshow(color_obs)
                            plt.title('Color Observation')
                        if image_config['combo']:
                            plt.figure()
                            plt.imshow(combo_obs)
                            plt.title('Combo Observation')
                        if image_config['dist']:
                            plt.figure()
                            plt.imshow(dist_im)
                            plt.title('Target Soft Distribution')
                            plt.figure()
                            plt.imshow(depth_obs)
                            plt.imshow(dist_im, alpha=0.5)
                        plt.show()

                    # Save depth image and semantic masks
                    if image_config['color']:
                        ColorImage(color_obs).save(os.path.join(color_dir, 
                                                                'image_{:06d}.png'.format(num_images_per_state * 
                                                                                          state_id + k)))
                    if image_config['depth']:
                        DepthImage(depth_obs).save(os.path.join(depth_dir, 
                                                                'image_{:06d}.png'.format(num_images_per_state * 
                                                                                          state_id + k)))
                    if image_config['combo']:
                        ColorImage(combo_obs).save(os.path.join(combo_dir, 
                                                                'image_{:06d}.png'.format(num_images_per_state * 
                                                                                          state_id + k)))
                    if image_config['dist']:
                        GrayscaleImage(dist_im).save(os.path.join(dist_dir, 
                                                                  'image_{:06d}.png'.format(num_images_per_state * 
                                                                                            state_id + k)))
                    
                    # Save split
                    if split == TRAIN_ID:
                        train_inds.append(num_images_per_state*state_id + k)
                    else:
                        test_inds.append(num_images_per_state*state_id + k)

                logger.info('Rendering and saving images took {:.3f} sec'.format(time.time()-render_start))
                
                # auto-flush after every so many timesteps
                if state_id > 0 and state_id % states_per_flush == 0:
                    logger.info('Last {} state(s) took {:.3f} sec'.format(states_per_flush, time.time()-flush_start))
                    flush_start = time.time()
                    expected_finish = (time.time() - gen_start) * (num_states - num_prev_states) / (state_id - num_prev_states + 1) + gen_start
                    logger.info('Expected finish time: {}'.format(time.strftime("%b %d %H:%M:%S", time.localtime(expected_finish))))
                    np.save(os.path.join(image_dir, 'train_indices.npy'), train_inds)
                    np.save(os.path.join(image_dir, 'test_indices.npy'), test_inds)
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
                logger.warning('Heap failed!')
                logger.warning('{}'.format(str(e)))
                if not isinstance(e, TargetMissingError):
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
    np.save(os.path.join(image_dir, 'train_indices.npy'), train_inds)
    np.save(os.path.join(image_dir, 'test_indices.npy'), test_inds)

    logger.info('Generated {:d} image datapoints'.format(state_id * num_images_per_state))

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Generate a training dataset for a SD Mask R-CNN')
    parser.add_argument('output_dataset_path', type=str, default=None, help='directory to store a dataset containing the images')
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    parser.add_argument('--warm_start', action='store_true', help='warm start system after crash')
    
    args = parser.parse_args()
    output_dataset_path = args.output_dataset_path
    config_filename = args.config_filename
    warm_start = args.warm_start
    
    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/generate_dist_dataset.yaml')

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # open config file
    config = YamlConfig(config_filename)

    # generate dataset
    generation_start = time.time()
    generate_distribution_dataset(output_dataset_path, config, warm_start=warm_start)

    # log time
    generation_stop = time.time()
    logger.info('Distribution dataset generation took {:.3f} sec'.format(generation_stop - 
                                                                         generation_start))
