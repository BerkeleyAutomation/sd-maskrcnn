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

Author: Mike Danielczuk
"""

import os
import time

import gym
import numpy as np
import scipy.stats as sstats
import trimesh
from autolab_core import Logger, RigidTransform

from .constants import KEY_SEP_TOKEN, TEST_ID, TRAIN_ID
from .random_variables import CameraRandomVariable
from .states import CameraState, HeapAndCameraState, HeapState, ObjectState


class CameraStateSpace(gym.Space):
    """State space for a camera."""

    def __init__(self, config):
        self._config = config

        # read params
        self.frame = config["name"]

        # random variable for pose of camera
        self.camera_rv = CameraRandomVariable(config)

    def sample(self):
        """Sample a camera state."""
        pose, intrinsics = self.camera_rv.sample(size=1)
        return CameraState(self.frame, pose, intrinsics)


class HeapStateSpace(gym.Space):
    """State space for object heaps."""

    def __init__(self, physics_engine, config):

        self._physics_engine = physics_engine
        self._config = config

        # set up logger
        self._logger = Logger.get_logger(self.__class__.__name__)

        # read subconfigs
        obj_config = config["objects"]
        workspace_config = config["workspace"]

        self.num_objs_rv = sstats.poisson(config["mean_objs"] - 1)
        self.max_objs = config["max_objs"]
        self.min_objs = 1
        if "min_objs" in config.keys():
            self.min_objs = config["min_objs"]
        self.replace = config["replace"]

        self.max_obj_diam = config["max_obj_diam"]
        self.drop_height = config["drop_height"]
        self.max_settle_steps = config["max_settle_steps"]
        self.mag_v_thresh = config["mag_v_thresh"]
        self.mag_w_thresh = config["mag_w_thresh"]

        # bounds of heap center in the table plane
        min_heap_center = np.array(config["center"]["min"])
        max_heap_center = np.array(config["center"]["max"])
        self.heap_center_space = gym.spaces.Box(
            min_heap_center, max_heap_center, dtype=np.float32
        )

        # Set up object configs
        # bounds of object drop pose in the table plane
        # organized as [tx, ty, theta] where theta is in degrees
        min_obj_pose = np.r_[obj_config["planar_translation"]["min"], 0]
        max_obj_pose = np.r_[
            obj_config["planar_translation"]["max"], 2 * np.pi
        ]
        self.obj_planar_pose_space = gym.spaces.Box(
            min_obj_pose, max_obj_pose, dtype=np.float32
        )

        # bounds of object drop orientation
        min_sph_coords = np.array([0.0, 0.0])
        max_sph_coords = np.array([2 * np.pi, np.pi])
        self.obj_orientation_space = gym.spaces.Box(
            min_sph_coords, max_sph_coords, dtype=np.float32
        )

        # bounds of center of mass
        delta_com_sigma = max(1e-6, obj_config["center_of_mass"]["sigma"])
        self.delta_com_rv = sstats.multivariate_normal(
            np.zeros(3), delta_com_sigma ** 2
        )

        self.obj_density = obj_config["density"]

        # bounds of workspace (for checking out of bounds)
        min_workspace_trans = np.array(workspace_config["min"])
        max_workspace_trans = np.array(workspace_config["max"])
        self.workspace_space = gym.spaces.Box(
            min_workspace_trans, max_workspace_trans, dtype=np.float32
        )

        # Setup object keys and directories
        object_keys = []
        mesh_filenames = []
        self._train_pct = obj_config["train_pct"]
        num_objects = obj_config["num_objects"]
        self._mesh_dir = obj_config["mesh_dir"]
        if not os.path.isabs(self._mesh_dir):
            self._mesh_dir = os.path.join(os.getcwd(), self._mesh_dir)
        for root, _, files in os.walk(self._mesh_dir):
            dataset_name = os.path.basename(root)
            if dataset_name in obj_config["object_keys"].keys():
                for f in files:
                    filename, ext = os.path.splitext(f)
                    if ext.split(".")[
                        1
                    ] in trimesh.exchange.load.mesh_formats() and (
                        filename in obj_config["object_keys"][dataset_name]
                        or obj_config["object_keys"][dataset_name] == "all"
                    ):
                        obj_key = "{}{}{}".format(
                            dataset_name, KEY_SEP_TOKEN, filename
                        )
                        object_keys.append(obj_key)
                        mesh_filenames.append(os.path.join(root, f))

        inds = np.arange(len(object_keys))
        np.random.shuffle(inds)
        self.all_object_keys = list(np.array(object_keys)[inds][:num_objects])
        all_mesh_filenames = list(np.array(mesh_filenames)[inds][:num_objects])
        self.train_keys = self.all_object_keys[
            : int(len(self.all_object_keys) * self._train_pct)
        ]
        self.test_keys = self.all_object_keys[
            int(len(self.all_object_keys) * self._train_pct) :
        ]
        self.obj_ids = dict(
            [(key, i + 1) for i, key in enumerate(self.all_object_keys)]
        )
        self.mesh_filenames = {}
        [
            self.mesh_filenames.update({k: v})
            for k, v in zip(self.all_object_keys, all_mesh_filenames)
        ]

        if (len(self.test_keys) == 0 and self._train_pct < 1.0) or (
            len(self.train_keys) == 0 and self._train_pct > 0.0
        ):
            raise ValueError("Not enough objects for train/test split!")

    @property
    def obj_keys(self):
        return self.all_object_keys

    @obj_keys.setter
    def obj_keys(self, keys):
        self.all_object_keys = keys

    @property
    def num_objects(self):
        return len(self.all_object_keys)

    @property
    def obj_id_map(self):
        return self.obj_ids

    @obj_id_map.setter
    def obj_id_map(self, id_map):
        self.obj_ids = id_map

    @property
    def obj_splits(self):
        obj_splits = {}
        for key in self.all_object_keys:
            if key in self.train_keys:
                obj_splits[key] = TRAIN_ID
            else:
                obj_splits[key] = TEST_ID
        return obj_splits

    def set_splits(self, obj_splits):
        self.train_keys = []
        self.test_keys = []
        for k in obj_splits.keys():
            if obj_splits[k] == TRAIN_ID:
                self.train_keys.append(k)
            else:
                self.test_keys.append(k)

    def in_workspace(self, pose):
        """Check whether a pose is in the workspace."""
        return self.workspace_space.contains(pose.translation)

    def sample(self):
        """Samples a state from the space
        Returns
        -------
        :obj:`HeapState`
            state of the object pile
        """

        # Start physics engine
        self._physics_engine.start()

        # setup workspace
        workspace_obj_states = []
        workspace_objs = self._config["workspace"]["objects"]
        for work_key, work_config in workspace_objs.items():

            # make paths absolute
            mesh_filename = work_config["mesh_filename"]
            pose_filename = work_config["pose_filename"]

            if not os.path.isabs(mesh_filename):
                mesh_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    mesh_filename,
                )
            if not os.path.isabs(pose_filename):
                pose_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    pose_filename,
                )

            # load mesh
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.density = self.obj_density
            pose = RigidTransform.load(pose_filename)
            workspace_obj = ObjectState(
                "{}{}0".format(work_key, KEY_SEP_TOKEN), mesh, pose
            )
            self._physics_engine.add(workspace_obj, static=True)
            workspace_obj_states.append(workspace_obj)

        # sample state
        train = True
        if np.random.rand() > self._train_pct:
            train = False
            sample_keys = self.test_keys
            self._logger.info("Sampling from test")
        else:
            sample_keys = self.train_keys
            self._logger.info("Sampling from train")

        total_num_objs = len(sample_keys)

        # sample object ids
        num_objs = self.num_objs_rv.rvs(size=1)[0]
        num_objs = min(num_objs, self.max_objs)
        num_objs = max(num_objs, self.min_objs)
        obj_inds = np.random.choice(
            np.arange(total_num_objs), size=2 * num_objs, replace=self.replace
        )
        self._logger.info("Sampling %d objects" % (num_objs))

        # sample pile center
        heap_center = self.heap_center_space.sample()
        t_heap_world = np.array([heap_center[0], heap_center[1], 0])
        self._logger.debug(
            "Sampled pile location: %.3f %.3f"
            % (t_heap_world[0], t_heap_world[1])
        )

        # sample object, center of mass, pose
        objs_in_heap = []
        total_drops = 0
        while total_drops < 2 * num_objs and len(objs_in_heap) < num_objs:
            obj_key = sample_keys[obj_inds[total_drops]]
            obj_mesh = trimesh.load_mesh(self.mesh_filenames[obj_key])
            obj_mesh.visual = trimesh.visual.ColorVisuals(
                obj_mesh, vertex_colors=(0.7, 0.7, 0.7, 1.0)
            )
            obj_mesh.density = self.obj_density
            obj_state_key = "{}{}{}".format(
                obj_key, KEY_SEP_TOKEN, total_drops
            )
            obj = ObjectState(obj_state_key, obj_mesh)
            self._logger.info(obj_state_key)
            _, radius = trimesh.nsphere.minimum_nsphere(obj.mesh)
            if 2 * radius > self.max_obj_diam:
                self._logger.warning("Obj too big, skipping .....")
                total_drops += 1
                continue

            # sample center of mass
            delta_com = self.delta_com_rv.rvs(size=1)
            center_of_mass = obj.mesh.center_mass + delta_com
            obj.mesh.center_mass = center_of_mass

            # sample obj drop pose
            obj_orientation = self.obj_orientation_space.sample()
            az = obj_orientation[0]
            elev = obj_orientation[1]
            T_obj_table = RigidTransform.sph_coords_to_pose(
                az, elev
            ).as_frames("obj", "world")

            # sample object planar pose
            obj_planar_pose = self.obj_planar_pose_space.sample()
            theta = obj_planar_pose[2]
            R_table_world = RigidTransform.z_axis_rotation(theta)
            R_obj_drop_world = R_table_world.dot(T_obj_table.rotation)
            t_obj_drop_heap = np.array(
                [obj_planar_pose[0], obj_planar_pose[1], self.drop_height]
            )
            t_obj_drop_world = t_obj_drop_heap + t_heap_world
            obj.pose = RigidTransform(
                rotation=R_obj_drop_world,
                translation=t_obj_drop_world,
                from_frame="obj",
                to_frame="world",
            )

            self._physics_engine.add(obj)
            try:
                v, w = self._physics_engine.get_velocity(obj.key)
            except:
                self._logger.warning(
                    "Could not get base velocity for object %s. Skipping ..."
                    % (obj.key)
                )
                self._physics_engine.remove(obj.key)
                total_drops += 1
                continue

            objs_in_heap.append(obj)

            # setup until approx zero velocity
            wait = time.time()
            objects_in_motion = True
            num_steps = 0
            while objects_in_motion and num_steps < self.max_settle_steps:

                # step simulation
                self._physics_engine.step()

                # check velocities
                max_mag_v = 0
                max_mag_w = 0
                objs_to_remove = set()
                for o in objs_in_heap:
                    try:
                        v, w = self._physics_engine.get_velocity(o.key)
                    except:
                        self._logger.warning(
                            "Could not get base velocity for object %s. Skipping ..."
                            % (o.key)
                        )
                        objs_to_remove.add(o)
                        continue
                    mag_v = np.linalg.norm(v)
                    mag_w = np.linalg.norm(w)
                    if mag_v > max_mag_v:
                        max_mag_v = mag_v
                    if mag_w > max_mag_w:
                        max_mag_w = mag_w

                # Remove invalid objects
                for o in objs_to_remove:
                    self._physics_engine.remove(o.key)
                    objs_in_heap.remove(o)

                # check objs in motion
                if (
                    max_mag_v < self.mag_v_thresh
                    and max_mag_w < self.mag_w_thresh
                ):
                    objects_in_motion = False

                num_steps += 1

            # read out object poses
            objs_to_remove = set()
            for o in objs_in_heap:
                obj_pose = self._physics_engine.get_pose(o.key)
                o.pose = obj_pose.copy()

                # remove the object if its outside of the workspace
                if not self.in_workspace(obj_pose):
                    self._logger.warning(
                        "Object {} fell out of the workspace!".format(o.key)
                    )
                    objs_to_remove.add(o)

            # remove invalid objects
            for o in objs_to_remove:
                self._physics_engine.remove(o.key)
                objs_in_heap.remove(o)

            total_drops += 1
            self._logger.debug(
                "Waiting for zero velocity took %.3f sec"
                % (time.time() - wait)
            )

        # Stop physics engine
        self._physics_engine.stop()

        # add metadata for heap state and return it
        metadata = {"split": TRAIN_ID}
        if not train:
            metadata["split"] = TEST_ID

        return HeapState(workspace_obj_states, objs_in_heap, metadata=metadata)


class HeapAndCameraStateSpace(gym.Space):
    """State space for environments."""

    def __init__(self, physics_engine, config):

        heap_config = config["heap"]
        cam_config = config["camera"]

        # individual state spaces
        self.heap = HeapStateSpace(physics_engine, heap_config)
        self.camera = CameraStateSpace(cam_config)

    @property
    def obj_id_map(self):
        return self.heap.obj_id_map

    @obj_id_map.setter
    def obj_id_map(self, id_map):
        self.heap.obj_ids = id_map

    @property
    def obj_keys(self):
        return self.heap.obj_keys

    @obj_keys.setter
    def obj_keys(self, keys):
        self.heap.all_object_keys = keys

    @property
    def obj_splits(self):
        return self.heap.obj_splits

    def set_splits(self, splits):
        self.heap.set_splits(splits)

    @property
    def mesh_filenames(self):
        return self.heap.mesh_filenames

    @mesh_filenames.setter
    def mesh_filenames(self, fns):
        self.heap.mesh_filenames = fns

    def sample(self):
        """Sample a state."""
        # sample individual states
        heap_state = self.heap.sample()
        cam_state = self.camera.sample()

        return HeapAndCameraState(heap_state, cam_state)
