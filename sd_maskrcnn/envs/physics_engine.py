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

import abc
import os
import time
import trimesh
import pybullet
import numpy as np

from autolab_core import RigidTransform, Logger
from pyrender import Scene, Viewer, Mesh, Node, PerspectiveCamera

from .constants import GRAVITY_ACCEL

class PhysicsEngine(metaclass=abc.ABCMeta):
    """ Abstract Physics Engine class """
    def __init__(self):
        
        # set up logger
        self._logger = Logger.get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def reset(self):
        pass
    
    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass
    
    @abc.abstractmethod
    def stop(self):
        pass

class PybulletPhysicsEngine(PhysicsEngine):
    """ Wrapper for pybullet physics engine that is tied to a single ID """
    def __init__(self, urdf_cache_dir, debug=False):
        PhysicsEngine.__init__(self)
        self._physics_client = None
        self._debug = debug
        self._urdf_cache_dir = urdf_cache_dir
        if not os.path.isabs(self._urdf_cache_dir):
            self._urdf_cache_dir = os.path.join(os.getcwd(), self._urdf_cache_dir)
        if not os.path.exists(self._urdf_cache_dir):
            os.makedirs(self._urdf_cache_dir)

    def add(self, obj, static=False):

        # create URDF
        urdf_filename = os.path.join(self._urdf_cache_dir, obj.key, '{}.urdf'.format(obj.key))
        urdf_dir = os.path.dirname(urdf_filename)
        if not os.path.exists(urdf_filename):
            try:
                os.makedirs(urdf_dir)
            except:
                self._logger.warning('Failed to create dir %s. The object may have been created simultaneously by another process' %(urdf_dir))
            self._logger.info('Exporting URDF for object %s' %(obj.key))
            
            # Fix center of mass (for rendering) and density and export
            geometry = obj.mesh.copy()
            geometry.apply_translation(-obj.mesh.center_mass)
            trimesh.exchange.export.export_urdf(geometry, urdf_dir)
       
        com = obj.mesh.center_mass
        pose = self._convert_pose(obj.pose, com)
        obj_t = pose.translation
        obj_q_wxyz = pose.quaternion
        obj_q_xyzw = np.roll(obj_q_wxyz, -1)
        try:
            obj_id = pybullet.loadURDF(urdf_filename,
                                       obj_t,
                                       obj_q_xyzw,
                                       useFixedBase=static,
                                       physicsClientId=self._physics_client)
        except:
            raise Exception('Failed to load %s' %(filename))
        
        if self._debug:
            self._add_to_scene(obj)

        self._key_to_id[obj.key] = obj_id
        self._key_to_com[obj.key] = com

    def get_velocity(self, key):
        obj_id = self._key_to_id[key]
        return pybullet.getBaseVelocity(obj_id, physicsClientId=self._physics_client)

    def get_pose(self, key):
        obj_id = self._key_to_id[key]
        obj_t, obj_q_xyzw = pybullet.getBasePositionAndOrientation(obj_id, physicsClientId=self._physics_client)
        obj_q_wxyz = np.roll(obj_q_xyzw, 1)
        pose = RigidTransform(rotation=obj_q_wxyz,
                              translation=obj_t,
                              from_frame='obj',
                              to_frame='world')
        pose = self._deconvert_pose(pose, self._key_to_com[key])
        return pose

    def remove(self, key):
        obj_id = self._key_to_id[key]
        pybullet.removeBody(obj_id, physicsClientId=self._physics_client)
        self._key_to_id.pop(key)
        self._key_to_com.pop(key)
        if self._debug:
            self._remove_from_scene(key)

    def step(self):
        pybullet.stepSimulation(physicsClientId=self._physics_client)
        if self._debug:
            time.sleep(0.04)
            self._update_scene()

    def reset(self):
        if self._physics_client is not None:
            self.stop()
        self.start()

    def start(self):
        if self._physics_client is None:
            self._physics_client = pybullet.connect(pybullet.DIRECT)
            pybullet.setGravity(0, 0, -GRAVITY_ACCEL, physicsClientId=self._physics_client)
            self._key_to_id = {}
            self._key_to_com = {}
            if self._debug:
                self._create_scene()
                self._viewer = Viewer(self._scene, use_raymond_lighting=True, run_in_thread=True)

    def stop(self):
        if self._physics_client is not None:
            pybullet.disconnect(self._physics_client)
            self._physics_client = None
            if self._debug:
                self._scene = None
                self._viewer.close_external()
                while self._viewer.is_active:
                    pass

    def __del__(self):
        self.stop()
        del self

    def _convert_pose(self, pose, com):
        new_pose = pose.copy()
        new_pose.translation = pose.rotation.dot(com) + pose.translation
        return new_pose

    def _deconvert_pose(self, pose, com):
        new_pose = pose.copy()
        new_pose.translation = pose.rotation.dot(-com) + pose.translation
        return new_pose

    def _create_scene(self):
        self._scene = Scene()
        camera = PerspectiveCamera(yfov=0.833, znear=0.05,
                                    zfar=3.0, aspectRatio=1.0)
        cn = Node()
        cn.camera = camera
        pose_m = np.array([[0.0,1.0,0.0,0.0],
                        [1.0,0.0,0.0,0.0],
                        [0.0,0.0,-1.0,0.88],
                        [0.0,0.0,0.0,1.0]])
        pose_m[:,1:3] *= -1.0
        cn.matrix = pose_m
        self._scene.add_node(cn)
        self._scene.main_camera_node = cn
    
    def _add_to_scene(self, obj):
        self._viewer.render_lock.acquire()
        n = Node(mesh=Mesh.from_trimesh(obj.mesh), matrix=obj.pose.matrix, name=obj.key)
        self._scene.add_node(n)
        self._viewer.render_lock.release()

    def _remove_from_scene(self, key):
        self._viewer.render_lock.acquire()
        if self._scene.get_nodes(name=key):
            self._scene.remove_node(next(iter(self._scene.get_nodes(name=key))))
        self._viewer.render_lock.release()
    
    def _update_scene(self):
        self._viewer.render_lock.acquire()
        for key in self._key_to_id.keys():
            obj_pose = self.get_pose(key).matrix
            if self._scene.get_nodes(name=key):
                next(iter(self._scene.get_nodes(name=key))).matrix = obj_pose
        self._viewer.render_lock.release()
