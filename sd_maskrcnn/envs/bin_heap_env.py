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

import gym
import numpy as np
from pyrender import (
    DirectionalLight,
    IntrinsicsCamera,
    Mesh,
    MetallicRoughnessMaterial,
    Node,
    OffscreenRenderer,
    RenderFlags,
    Scene,
    Viewer,
)

from .physics_engine import PybulletPhysicsEngine
from .state_spaces import HeapAndCameraStateSpace


class BinHeapEnv(gym.Env):
    """OpenAI Gym-style environment for creating object heaps in a bin."""

    def __init__(self, config):

        self._config = config

        # read subconfigs
        self._state_space_config = self._config["state_space"]

        # initialize class variables
        self._state = None
        self._scene = None
        self._physics_engine = PybulletPhysicsEngine(
            urdf_cache_dir=config["urdf_cache_dir"], debug=config["debug"]
        )
        self._state_space = HeapAndCameraStateSpace(
            self._physics_engine, self._state_space_config
        )

    @property
    def config(self):
        return self._config

    @property
    def state(self):
        return self._state

    @property
    def camera(self):
        return self._camera

    @property
    def observation(self):
        return self.render_camera_image()

    @property
    def scene(self):
        return self._scene

    @property
    def num_objects(self):
        return self.state.num_objs

    @property
    def state_space(self):
        return self._state_space

    @property
    def obj_keys(self):
        return self.state.obj_keys

    def _reset_state_space(self):
        """Sample a new static and dynamic state."""
        state = self._state_space.sample()
        self._state = state.heap
        self._camera = state.camera

    def _update_scene(self):
        # update camera
        camera = IntrinsicsCamera(
            self.camera.intrinsics.fx,
            self.camera.intrinsics.fy,
            self.camera.intrinsics.cx,
            self.camera.intrinsics.cy,
        )
        cn = next(iter(self._scene.get_nodes(name=self.camera.frame)))
        cn.camera = camera
        pose_m = self.camera.pose.matrix.copy()
        pose_m[:, 1:3] *= -1.0
        cn.matrix = pose_m
        self._scene.main_camera_node = cn

        # update workspace
        for obj_key in self.state.workspace_keys:
            next(
                iter(self._scene.get_nodes(name=obj_key))
            ).matrix = self.state[obj_key].pose.matrix

        # update object
        for obj_key in self.state.obj_keys:
            next(
                iter(self._scene.get_nodes(name=obj_key))
            ).matrix = self.state[obj_key].pose.matrix

    def _reset_scene(self, scale_factor=1.0):
        """Resets the scene.

        Parameters
        ----------
        scale_factor : float
            optional scale factor to apply to the image dimensions
        """
        # delete scene
        if self._scene is not None:
            self._scene.clear()
            del self._scene

        # create scene
        scene = Scene()

        # setup camera
        camera = IntrinsicsCamera(
            self.camera.intrinsics.fx,
            self.camera.intrinsics.fy,
            self.camera.intrinsics.cx,
            self.camera.intrinsics.cy,
        )
        pose_m = self.camera.pose.matrix.copy()
        pose_m[:, 1:3] *= -1.0
        scene.add(camera, pose=pose_m, name=self.camera.frame)
        scene.main_camera_node = next(
            iter(scene.get_nodes(name=self.camera.frame))
        )

        material = MetallicRoughnessMaterial(
            baseColorFactor=np.array([1, 1, 1, 1.0]),
            metallicFactor=0.2,
            roughnessFactor=0.8,
        )

        # add workspace objects
        for obj_key in self.state.workspace_keys:
            obj_state = self.state[obj_key]
            obj_mesh = Mesh.from_trimesh(obj_state.mesh, material=material)
            T_obj_world = obj_state.pose.matrix
            scene.add(obj_mesh, pose=T_obj_world, name=obj_key)

        # add scene objects
        for obj_key in self.state.obj_keys:
            obj_state = self.state[obj_key]
            obj_mesh = Mesh.from_trimesh(obj_state.mesh, material=material)
            T_obj_world = obj_state.pose.matrix
            scene.add(obj_mesh, pose=T_obj_world, name=obj_key)

        # add light (for color rendering)
        light = DirectionalLight(color=np.ones(3), intensity=1.0)
        scene.add(light, pose=np.eye(4))
        ray_light_nodes = self._create_raymond_lights()
        [scene.add_node(rln) for rln in ray_light_nodes]

        self._scene = scene

    def reset_camera(self):
        """Resets only the camera.
        Useful for generating image data for multiple camera views
        """
        self._camera = self.state_space.camera.sample()
        self._update_scene()

    def reset(self):
        """Reset the environment."""

        # reset state space
        self._reset_state_space()

        # reset scene
        self._reset_scene()

    def view_3d_scene(self):
        """Render the scene in a 3D viewer."""
        if self.state is None or self.camera is None:
            raise ValueError(
                "Cannot render 3D scene before state is set! You can set the state with the reset() function"
            )

        Viewer(self.scene, use_raymond_lighting=True)

    def render_camera_image(self, color=True):
        """Render the camera image for the current scene."""
        renderer = OffscreenRenderer(self.camera.width, self.camera.height)
        flags = RenderFlags.NONE if color else RenderFlags.DEPTH_ONLY
        image = renderer.render(self._scene, flags=flags)
        renderer.delete()
        return image

    def render_segmentation_images(self):
        """Renders segmentation masks (modal and amodal) for each object in the state."""

        full_depth = self.render_camera_image(color=False)
        modal_data = np.zeros(
            (full_depth.shape[0], full_depth.shape[1], len(self.obj_keys)),
            dtype=np.uint8,
        )
        amodal_data = np.zeros(
            (full_depth.shape[0], full_depth.shape[1], len(self.obj_keys)),
            dtype=np.uint8,
        )
        renderer = OffscreenRenderer(self.camera.width, self.camera.height)
        flags = RenderFlags.DEPTH_ONLY

        # Hide all meshes
        obj_mesh_nodes = [
            next(iter(self._scene.get_nodes(name=k))) for k in self.obj_keys
        ]
        for mn in self._scene.mesh_nodes:
            mn.mesh.is_visible = False

        for i, node in enumerate(obj_mesh_nodes):
            node.mesh.is_visible = True

            depth = renderer.render(self._scene, flags=flags)
            amodal_mask = depth > 0.0
            modal_mask = np.logical_and(
                (np.abs(depth - full_depth) < 1e-6), full_depth > 0.0
            )
            amodal_data[amodal_mask, i] = np.iinfo(np.uint8).max
            modal_data[modal_mask, i] = np.iinfo(np.uint8).max
            node.mesh.is_visible = False

        renderer.delete()

        # Show all meshes
        for mn in self._scene.mesh_nodes:
            mn.mesh.is_visible = True

        return amodal_data, modal_data

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                Node(
                    light=DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )

        return nodes
