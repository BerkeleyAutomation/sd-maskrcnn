import trimesh
import os
import numpy as np
import argparse
from tqdm import tqdm
from pyrender import (Scene, Mesh, Node, IntrinsicsCamera, 
                      MetallicRoughnessMaterial, DirectionalLight, 
                      OffscreenRenderer)
from perception import ColorImage, DepthImage, BinaryImage

class MeshLoader(object):
    """A tool for loading meshes from a base directory.

    Attributes
    ----------
    basedir : str
        basedir containing mesh files
    """

    def __init__(self, basedir):
        self.basedir = basedir
        self._map = {}
        for root, dirs, fns in os.walk(basedir):
            for fn in fns:
                full_fn = os.path.join(root, fn)
                f, ext = os.path.splitext(fn)
                if ext[1:] not in trimesh.available_formats():
                    continue
                if f in self._map:
                    raise ValueError('Duplicate file named {}'.format(f))
                self._map[f] = full_fn
    
    def meshes(self):
        return self._map.keys()
    
    def get_path(self, name):
        if name in self._map:
            return self._map[name]
        raise ValueError('Could not find mesh with name {} in directory {}'.format(name, self.basedir))

    def load(self, name):
        m = trimesh.load(self.get_path(name))
        m.metadata['name'] = name
        return m


def create_scene():
    
    # create scene
    scene = Scene()

    # setup camera
    camera = IntrinsicsCamera(1122.0, 1122.0, 511.5, 384.0)
    
    pose_m = np.array([[0.0, -1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0, 0.80],
                       [0.0, 0.0, 0.0, 1.0]])
    pose_m[:,1:3] *= -1.0
    scene.add(camera, pose=pose_m, name='camera')
    scene.main_camera_node = next(iter(scene.get_nodes(name='camera')))

    # add light (for color rendering)
    light = DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(light, pose=np.eye(4))
    ray_light_nodes = create_raymond_lights()
    [scene.add_node(rln) for rln in ray_light_nodes]

    return scene

def create_raymond_lights():
    
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
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(Node(
            light=DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

def render_camera_image(scene):
    """ Render the camera image for the current scene. """
    renderer = OffscreenRenderer(1032, 772)
    image = renderer.render(scene)
    renderer.delete()
    return image


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate masks for meshes')
    parser.add_argument('mesh_folder', type=str, help='folder containing meshes')
    parser.add_argument('output_folder', type=str, help='where to output images')
    parser.add_argument('--num_stps', type=int, default=10, help='max stable poses for each object')
    parser.add_argument('--stp_threshold', type=float, default=0.01, help='threshold for probability of stable pose')

    args = parser.parse_args()

    mesh_loader = MeshLoader(args.mesh_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Setup scene
    scene = create_scene()
    mesh_material = MetallicRoughnessMaterial(
        baseColorFactor=np.array([1, 1, 1, 1.0]),
        metallicFactor=0.2,
        roughnessFactor=0.8
    )
    
    # Add plane (for depth images)
    plane_mesh = trimesh.load('sd_maskrcnn/data/plane/plane.obj')
    scene.add(Mesh.from_trimesh(plane_mesh, material=mesh_material), name='plane')
    _, plane_depth = render_camera_image(scene)

    mesh_node = None
    for m in tqdm(mesh_loader.meshes()):
        mesh = mesh_loader.load(m)
        mesh.visual.vertex_colors = (1.0, 0.0, 0.0)

        stps, _ = mesh.compute_stable_poses(threshold=args.stp_threshold)
        stps = stps[:args.num_stps]
        for i, stp in enumerate(stps):
            
            # add scene objects
            obj_mesh = Mesh.from_trimesh(mesh, material=mesh_material)
            if mesh_node is None:
                mesh_node = scene.add(obj_mesh, pose=stp, name=mesh.metadata['name'])
            else:
                mesh_node.mesh = obj_mesh
                mesh_node.name = mesh.metadata['name']
                scene.set_pose(mesh_node, stp)
            
            # render with mesh and make mask
            color, depth = render_camera_image(scene)
            obj_mask = (np.iinfo(np.uint8).max * (depth < plane_depth)).astype(np.uint8)
            
            # Save images to output folder
            ColorImage(color).resize((384, 512)).save(os.path.join(args.output_folder, '{}_{:02d}_color.png'.format(mesh.metadata['name'], i)))
            BinaryImage(obj_mask).resize((384, 512)).save(os.path.join(args.output_folder, '{}_{:02d}_mask.png'.format(mesh.metadata['name'], i)))
            DepthImage(depth).resize((384, 512)).save(os.path.join(args.output_folder, '{}_{:02d}_depth.png'.format(mesh.metadata['name'], i)))
