import torch
import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    as_tensor = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    occ1 = as_tensor(occ1)
    occ2 = as_tensor(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).float().sum(axis=-1)
    area_intersect = (occ1 & occ2).float().sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """
    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist


def nn_distance(query_points, ref_points):
    ref_points_kd_tree = KDTree(ref_points)
    one_distances, one_vertex_ids = ref_points_kd_tree.query(query_points)
    return one_distances


def completeness(query_points, ref_points, thres=0.03):
    a2b_nn_distance =  nn_distance(query_points, ref_points)
    percentage = np.sum(a2b_nn_distance < thres) / len(a2b_nn_distance)
    return percentage


