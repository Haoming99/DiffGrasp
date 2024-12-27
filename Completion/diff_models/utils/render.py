import os

import socket
# if socket.gethostname() not in ['enough-oryx', 'ee-3090-0']: # Some node don't need this
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import pyrender
from pyrender import Mesh
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as scipyR

color = np.array([31/255, 119/255, 180/255])
# wandb_R = np.array([[1, 0, 0.], [0, 0, 1], [0, 1, 0]])
pts2mesh = lambda x: pyrender.Mesh.from_points(x, colors=color)

class CanonRenderer:

    def __init__(self, img_w=256, img_h=256, fx=200, fy=200):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        self.fx = fx
        self.fy = fy
        self.img_h = img_h
        self.img_w = img_w

    def render(self, mesh):
        """
        input:
        @mesh: trimesh object

        return:
        @image: np.array(img_h, img_w, 3)
        """
        camera = pyrender.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.img_h//2, cy=self.img_w//2, zfar=1000)
        cam_pose = np.eye(4)
        r = scipyR.from_euler('xyz', [-30, 0, 0], degrees=True)
        cam_pose[:3, :3] = r.as_matrix()
        cam_pose[:3, 3] = np.array([0., 0.8, 1.5])

        material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=(0.5, 0.2, 0.2, 1.0))
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        if not isinstance(mesh, Mesh): # Use its own material if it has been PyMesh
            mesh = Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh)
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        image, depth = self.renderer.render(scene)
        return image

    def __call__(self, *args):
        return self.render(*args)

class ShapeNetRender:

    def __init__(self, img_w=137, img_h=137, fx=149.84375, fy=149.84375):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        self.fx = fx
        self.fy = fy
        self.img_h = img_h
        self.img_w = img_w

    def render(self, mesh):
        """
        input:
        @mesh: trimesh object
        return:
        @image: np.array(img_h, img_w, 3)
        """
        camera = pyrender.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.img_h//2, cy=self.img_w//2, zfar=1000)
        cam_pose = np.eye(4)
        r = scipyR.from_euler('xyz', [-35, 0, 0], degrees=True)
        cam_pose[:3, :3] = r.as_matrix()
        cam_pose[:3, 3] = np.array([0., 1.0, 1.5])

        material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=(0.5, 0.5, 0.5, 1.0))
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        if not isinstance(mesh, Mesh): # Use its own material if it has been PyMesh
            mesh = Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh)
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        image, depth = self.renderer.render(scene)
        return image

    def __call__(self, *args):
        return self.render(*args)

class SynRoomRender:

    def __init__(self, img_w=137, img_h=137, fx=149.84375, fy=149.84375):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        self.fx = fx
        self.fy = fy
        self.img_h = img_h
        self.img_w = img_w

    def render(self, mesh):
        """
        input:
        @mesh: trimesh object
        return:
        @image: np.array(img_h, img_w, 3)
        """
        camera = pyrender.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.img_h//2, cy=self.img_w//2, zfar=1000)
        cam_pose = np.eye(4)
        r = scipyR.from_euler('xyz', [-45, 0, 0], degrees=True)
        cam_pose[:3, :3] = r.as_matrix()
        cam_pose[:3, 3] = np.array([0., 0.8, 1.2])

        material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.5,
                    alphaMode='OPAQUE',
                    baseColorFactor=(0.5, 0.5, 0.5, 1.0))
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        if not isinstance(mesh, Mesh): # Use its own material if it has been PyMesh
            mesh = Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh)
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 0, 2])
        scene.add(light, pose=light_pose)

        image, depth = self.renderer.render(scene)
        return image

    def __call__(self, *args):
        return self.render(*args)


class BaseRenderer:
    """
    Render from the corrner of box
    """
    def __init__(self, img_w=256, img_h=256, fx=200, fy=200):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        self.fx = fx
        self.fy = fy
        self.img_h = img_h
        self.img_w = img_w

    def get_cam_pose(self):
        cam_pose = np.eye(4)
        r = scipyR.from_euler('xyz', [-40, 45, 0], degrees=True)
        cam_pose[:3, :3] = r.as_matrix()
        cam_pose[:3, 3] = np.array([1.2, 1.2, 1.2])
        return cam_pose


    def render(self, mesh):
        """
        input:
        @mesh: trimesh object

        return:
        @image: np.array(img_h, img_w, 3)
        """
        camera = pyrender.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.img_h//2, cy=self.img_w//2, zfar=1000)
        cam_pose = self.get_cam_pose()

        material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=(0.5, 0.2, 0.2, 1.0))
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        if not isinstance(mesh, Mesh): # Use its own material if it has been PyMesh
            mesh = Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh)
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        image, depth = self.renderer.render(scene)
        return image

    def __call__(self, *args):
        return self.render(*args)

class CornerRender(BaseRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_cam_pose(self):
        cam_pose = np.eye(4)
        r = scipyR.from_euler('xyz', [-40, 45, 0], degrees=True)
        cam_pose[:3, :3] = r.as_matrix()
        cam_pose[:3, 3] = np.array([1.0, 1.0, 1.0]) / 2

        return cam_pose