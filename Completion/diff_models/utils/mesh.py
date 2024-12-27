import time
import trimesh
import torch
import numpy as np
import os
import os.path as osp
from ..libs import libmcubes
from ..libs.libmise import MISE
from ..libs.libsimplify import simplify_mesh


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.
    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def extract_mesh(occ_hat, z, c=None, refinement_step=0, simplify_nfaces=None, with_normals=False, box_size = 2.1, threshold=0.2, stats_dict=dict()):
    ''' Extracts the mesh from the predicted occupancy grid.
    Args:
        occ_hat (tensor): value grid of occupancies
        z (tensor): latent code z
        c (tensor): latent conditioned code c
        stats_dict (dict): stats dictionary
    '''
    # Some short hands
    n_x, n_y, n_z = occ_hat.shape
    # Make sure that mesh is watertight
    t0 = time.time()
    occ_hat_padded = np.pad(
        occ_hat, 1, 'constant', constant_values=-1e6)
    vertices, triangles = libmcubes.marching_cubes(
        occ_hat_padded, threshold)
    stats_dict['time (marching cubes)'] = time.time() - t0
    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x-1, n_y-1, n_z-1])
    vertices = box_size * (vertices - 0.5)

    # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
    # mesh_pymesh = fix_pymesh(mesh_pymesh)

    # Estimate normals if needed
    if with_normals and not vertices.shape[0] == 0:
        raise NotImplementedError("Normal estimation not implemented yet")
        t0 = time.time()
        normals = self.estimate_normals(vertices, z, c)
        stats_dict['time (normals)'] = time.time() - t0

    else:
        normals = None

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles,
                           vertex_normals=normals,
                           process=False)

    # Directly return if mesh is empty
    if vertices.shape[0] == 0:
        return mesh

    # TODO: normals are lost here
    if simplify_nfaces is not None:
        t0 = time.time()
        mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
        stats_dict['time (simplify)'] = time.time() - t0

    # Refine mesh
    if refinement_step > 0:
        raise NotImplementedError("Refinement not implemented yet")
        t0 = time.time()
        self.refine_mesh(mesh, occ_hat, z, c)
        stats_dict['time (refine)'] = time.time() - t0

    return mesh

def generate_from_latent(eval_func, z, c=None, stats_dict={}, threshold=0.2, resolution0=32, padding=0.1, upsampling_steps=2, B_MIN=-1, B_MAX=1, device=None, **kwargs):
    ''' Generates mesh from latent.
    Args:
        z (tensor): latent code z
        c (tensor): latent conditioned code c
        stats_dict (dict): stats dictionary
    '''
    if device is None:
        device = c.device # to support old version

    threshold = np.log(threshold) - np.log(1. - threshold)

    t0 = time.time()
    # Compute bounding box size
    box_size = (B_MAX - B_MIN) + padding

    # Shortcut
    if upsampling_steps == 0:
        nx = resolution0
        pointsf = box_size * make_3d_grid(
            (B_MIN,)*3, (B_MAX,)*3, (nx,)*3
        )
        values = eval_func(pointsf, z=z, c=c, **kwargs).cpu().numpy()
        value_grid = values.reshape(nx, nx, nx)
    else:
        mesh_extractor = MISE(
            resolution0, upsampling_steps, threshold)

        points = mesh_extractor.query()
        # import ipdb; ipdb.set_trace()

        while points.shape[0] != 0:
            # Query points
            pointsf = torch.FloatTensor(points).to(device)
            # Normalize to bounding box
            pointsf = pointsf / mesh_extractor.resolution
            pointsf = box_size * (pointsf - 0.5)
            # Evaluate model and update
            values = eval_func(
                pointsf, z=z, c=c, **kwargs).cpu().numpy()
            values = values.astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        value_grid = mesh_extractor.to_dense()

    # Extract mesh
    stats_dict['time (eval points)'] = time.time() - t0

    mesh = extract_mesh(value_grid, z, c, stats_dict=stats_dict, threshold=threshold, box_size=box_size, )
    return mesh

def export_shapenet_samples(mesh_list, cat_list, model_list, input_list, out_dir: str, test_num_pts:int =2048) -> None:

    try:
        for meshes, cat_id, model_id, pcs_in in zip(mesh_list, cat_list, model_list, input_list):
            cat_dir = osp.join(out_dir, cat_id)
            cat_points_dir = osp.join(out_dir, '..', 'points', cat_id)
            os.makedirs(cat_dir, exist_ok=True)
            os.makedirs(cat_points_dir, exist_ok=True)
            np.savetxt(osp.join(cat_dir, f"{model_id}_input.txt"), pcs_in.cpu())
            np.savetxt(osp.join(cat_points_dir, f"{model_id}_input.txt"), pcs_in.cpu())

            for s_id, mesh in enumerate(meshes):
                model_path = osp.join(cat_dir, f"{model_id}_{s_id:02d}.obj")
                mesh.export(model_path)
                points_path = osp.join(cat_points_dir, f"{model_id}_{s_id:02d}.pts.npy")
                sampled_pts, _ = trimesh.sample.sample_surface(mesh, test_num_pts)
                np.save(points_path, sampled_pts)
    except Exception as e:
       print(e) 

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

