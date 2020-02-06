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

Authors: Mike Danielczuk, Jeff Mahler

Random variables for sampling camera poses (adapted from BerkeleyAutomation/meshrender).
"""
import numpy as np
import scipy.stats as sstats

from autolab_core import RigidTransform, transformations
from autolab_core.utils import sph2cart, cart2sph
from perception import CameraIntrinsics

class CameraRandomVariable(object):
    """Uniform distribution over camera poses and intrinsics about a viewsphere over a planar worksurface.
    The camera is positioned pointing towards (0,0,0).
    """

    def __init__(self, config):
        """Initialize a CameraRandomVariable.
        Parameters
        ----------
        config : autolab_core.YamlConfig
            configuration containing parameters of random variable
        Notes
        -----
        Required parameters of config are specified in Other Parameters
        Other Parameters
        ----------
        focal_length :        Focal length of the camera
            min : float
            max : float
        delta_optical_center: Change in optical center from neutral.
            min : float
            max : float
        radius:               Distance from camera to world origin.
            min : float
            max : float
        azimuth:              Azimuth (angle from x-axis) of camera in degrees.
            min : float
            max : float
        elevation:            Elevation (angle from z-axis) of camera in degrees.
            min : float
            max : float
        roll:                 Roll (angle about view direction) of camera in degrees.
            min : float
            max : float
        x:                    Translation of world center in x axis.
            min : float
            max : float
        y:                    Translation of world center in y axis.
            min : float
            max : float
        im_height : float     Height of image in pixels.
        im_width : float      Width of image in pixels.
        """
        # read params
        self.config = config
        self._parse_config(config)

        self.frame = config['name']

        # setup random variables
        # camera
        self.focal_rv = sstats.uniform(loc=self.min_f, scale=self.max_f-self.min_f)
        self.cx_rv = sstats.uniform(loc=self.min_cx, scale=self.max_cx-self.min_cx)
        self.cy_rv = sstats.uniform(loc=self.min_cy, scale=self.max_cy-self.min_cy)

        # viewsphere
        self.rad_rv = sstats.uniform(loc=self.min_radius, scale=self.max_radius-self.min_radius)
        self.elev_rv = sstats.uniform(loc=self.min_elev, scale=self.max_elev-self.min_elev)
        self.az_rv = sstats.uniform(loc=self.min_az, scale=self.max_az-self.min_az)
        self.roll_rv = sstats.uniform(loc=self.min_roll, scale=self.max_roll-self.min_roll)

        # table translation
        self.tx_rv = sstats.uniform(loc=self.min_x, scale=self.max_x-self.min_x)
        self.ty_rv = sstats.uniform(loc=self.min_y, scale=self.max_y-self.min_y)

    def _parse_config(self, config):
        """Reads parameters from the config into class members.
        """
        # camera params
        self.min_f = config['focal_length']['min']
        self.max_f = config['focal_length']['max']
        self.min_delta_c = config['delta_optical_center']['min']
        self.max_delta_c = config['delta_optical_center']['max']
        self.im_height = config['im_height']
        self.im_width = config['im_width']

        self.mean_cx = float(self.im_width - 1) / 2
        self.mean_cy = float(self.im_height - 1) / 2
        self.min_cx = self.mean_cx + self.min_delta_c
        self.max_cx = self.mean_cx + self.max_delta_c
        self.min_cy = self.mean_cy + self.min_delta_c
        self.max_cy = self.mean_cy + self.max_delta_c

        # viewsphere params
        self.min_radius = config['radius']['min']
        self.max_radius = config['radius']['max']
        self.min_az = np.deg2rad(config['azimuth']['min'])
        self.max_az = np.deg2rad(config['azimuth']['max'])
        self.min_elev = np.deg2rad(config['elevation']['min'])
        self.max_elev = np.deg2rad(config['elevation']['max'])
        self.min_roll = np.deg2rad(config['roll']['min'])
        self.max_roll = np.deg2rad(config['roll']['max'])

        # params of translation in plane
        self.min_x = config['x']['min']
        self.max_x = config['x']['max']
        self.min_y = config['y']['min']
        self.max_y = config['y']['max']

    def camera_to_world_pose(self, radius, elev, az, roll, x, y):
        """Convert spherical coords to a camera pose in the world.
        """
        # generate camera center from spherical coords
        delta_t = np.array([x, y, 0])
        camera_z = np.array([sph2cart(radius, az, elev)]).squeeze()
        camera_center = camera_z + delta_t
        camera_z = -camera_z / np.linalg.norm(camera_z)

        # find the canonical camera x and y axes
        camera_x = np.array([camera_z[1], -camera_z[0], 0])
        x_norm = np.linalg.norm(camera_x)
        if x_norm == 0:
            camera_x = np.array([1, 0, 0])
        else:
            camera_x = camera_x / x_norm
        camera_y = np.cross(camera_z, camera_x)
        camera_y = camera_y / np.linalg.norm(camera_y)

        # Reverse the x direction if needed so that y points down
        if camera_y[2] > 0:
            camera_x = -camera_x
            camera_y = np.cross(camera_z, camera_x)
            camera_y = camera_y / np.linalg.norm(camera_y)

        # rotate by the roll
        R = np.vstack((camera_x, camera_y, camera_z)).T
        roll_rot_mat = transformations.rotation_matrix(roll, camera_z, np.zeros(3))[:3,:3]
        R = roll_rot_mat.dot(R)
        T_camera_world = RigidTransform(R, camera_center, from_frame=self.frame, to_frame='world')

        return T_camera_world

    def sample(self, size=1):
        """Sample random variables from the model.
        Parameters
        ----------
        size : int
            number of sample to take
        Returns
        -------
        :obj:`list` of :obj:`CameraSample`
            sampled camera intrinsics and poses
        """
        samples = []
        for i in range(size):
            # sample camera params
            focal = self.focal_rv.rvs(size=1)[0]
            cx = self.cx_rv.rvs(size=1)[0]
            cy = self.cy_rv.rvs(size=1)[0]

            # sample viewsphere params
            radius = self.rad_rv.rvs(size=1)[0]
            elev = self.elev_rv.rvs(size=1)[0]
            az = self.az_rv.rvs(size=1)[0]
            roll = self.roll_rv.rvs(size=1)[0]

            # sample plane translation
            tx = self.tx_rv.rvs(size=1)[0]
            ty = self.ty_rv.rvs(size=1)[0]

            # convert to pose and intrinsics
            pose = self.camera_to_world_pose(radius, elev, az, roll, tx, ty)
            intrinsics = CameraIntrinsics(self.frame, fx=focal, fy=focal,
                                          cx=cx, cy=cy, skew=0.0,
                                          height=self.im_height, width=self.im_width)

            # convert to camera pose
            samples.append((pose, intrinsics))

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples
