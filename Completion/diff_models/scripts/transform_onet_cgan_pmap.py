import os.path as osp
import pickle
import numpy as np
import os
import math
import trimesh
import sys
from tqdm import tqdm
import open3d as o3d
import argparse
from glob import glob

def preprocess_point_cloud(pcd, voxel_size=1./128, verbose=False):
    """
    Down sample and estimate normal and FPFH feature for pointcloud
    """
    if verbose:
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    
    if verbose:
        print(":: Estimate normal with search radius %.3f." % radius_normal)
    
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    
    if verbose:
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def find_transformation_icp(source_pcs, target_pcs, verbose=False):
    """
    source_pcs: np.array((N, 3))
    target_pcs: np.array((M, 3))
    """
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pcs)

    source = o3d.geometry.PointCloud()
    selected_idx = np.random.choice(np.arange(source_pcs.shape[0]), target_pcs.shape[0])
    source.points = o3d.utility.Vector3dVector(source_pcs[selected_idx])

    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Fisrt global transformation
    source_down, source_fpfh = preprocess_point_cloud(source, verbose=verbose)
    target_down, target_fpfh = preprocess_point_cloud(target, verbose=verbose)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=1./128))

    current_transformation = result.transformation

    if verbose:
        print("2. Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. Distance threshold 0.002.")
        
    # Solve ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, 0.01, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
    
    if verbose:
        print(result_icp)
        print(result_icp.transformation)

#     rotated_pcs = np.asarray(source.points) @ result_icp.transformation[:3, :3].T
    
    return result_icp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--dist_id", type=int, default=0)
    parser.add_argument("--dist_size", type=int, default=1)
    args = parser.parse_args()
    res_dict = {}

    task_list = []
    cat_ids, model_ids = [], []
    fnames = sorted(glob(f"/scratch/wen/ShapeNetPointCloud/*/*.ply"))
    chunk_size = int(math.ceil(len(fnames) / args.dist_size))
    chunked_names = fnames[args.dist_id * chunk_size: min((args.dist_id + 1) * chunk_size, len(fnames))]

    for fname in tqdm(chunked_names):
        cat_id, model_id = fname.split('/')[-2:]
        model_id = model_id.rstrip('.ply')
        if cat_id not in res_dict:
            res_dict[cat_id] = {}


        onet_data = dict(**np.load(f"/Datasets/ShapeNet/{cat_id}/{model_id}/pointcloud.npz"))
        onet_pcs = (onet_data['points'] + onet_data['loc']) * onet_data['scale']

        cgan_pcs = trimesh.load(fname).vertices

        icp_res = find_transformation_icp(onet_pcs, cgan_pcs)

        res_dict[cat_id][model_id] = icp_res.transformation.copy()

    with open(f'/scratch/wen/onet2cgan_trans/{args.dist_id}.pkl', 'wb') as f:
        pickle.dump(res_dict, f)

    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(res_dict, f)
