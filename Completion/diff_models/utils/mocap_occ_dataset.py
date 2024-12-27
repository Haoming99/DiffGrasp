import os
from optax import piecewise_interpolate_schedule
from smplx import SMPLLayer
import numpy as np
import torch
# from open3d import JVisualizer
from smplx import SMPL

import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from numpy.random import default_rng
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from typing import Optional
from libs.libmesh import check_mesh_contains
import trimesh

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def sample_mesh(mesh, num_pts, sigma=0.005, B_MIN=-0.55, B_MAX=0.55,):
    """
    Code form PiFU
    extend B_MIN and B_MAX a little bit to ensure boundary case
    """
    surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * num_pts)
    sample_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)

    # add random points within image space
    length = B_MAX - B_MIN
    random_points = np.random.rand(num_pts // 4, 3) * length + B_MIN
    sample_points = np.concatenate([sample_points, random_points], 0)
    np.random.shuffle(sample_points)

    inside = check_mesh_contains(mesh, sample_points)
    inside_points = sample_points[inside]
    outside_points = sample_points[np.logical_not(inside)]

    nin = inside_points.shape[0]
    inside_points = inside_points[
                    :num_pts // 2] if nin > num_pts // 2 else inside_points
    outside_points = outside_points[
                        :num_pts // 2] if nin > num_pts // 2 else outside_points[:(num_pts - nin)]

    samples = np.concatenate([inside_points, outside_points], 0).T
    labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

    # save_samples_truncted_prob('out.ply', samples.T, labels.T)
    # exit()

    samples = torch.Tensor(samples).float()
    labels = torch.Tensor(labels).float()
    
    del mesh

    return samples.t(), labels[0] # Change to Nx3 and N



class MocapOccDataset(torch.utils.data.Dataset):
    def __init__(self, stage, annot_path='', smpl_path='', n_sample_points=2048, n_partial_pcs=3000, n_full_pcs=6000,
         pcs_noise=0.005, scale=2.0, loc=0., sigma=0.005):
        self.annot_path = annot_path
        self.annot = dict(np.load(annot_path))

        rng = default_rng(42)
        dataset_size = len(self.annot['body_pose'])
        #test_indexs = rng.choice(np.arange(dataset_size), int(dataset_size * 0.3))
        test_mask = np.zeros(dataset_size)
        #test_mask[test_indexs] = 1
        test_mask[-int(0.2 * dataset_size):] = 1

        self.n_sample_points = n_sample_points
        self.n_partial_pcs = n_partial_pcs
        self.n_full_pcs = n_full_pcs

        self.loc = loc
        self.scale = scale
        self.pcs_noise = pcs_noise
        self.sigma = sigma

        self.stage = stage
        if stage == 'test':
            dataset_mask = test_mask > 0
        else:
            dataset_mask = test_mask == 0

        self.annot['body_pose'] = self.annot['body_pose'][dataset_mask]
#         self.annot['betas'] = self.annot['betas'][dataset_mask]

        self.smpl = SMPL(model_path=smpl_path, create_global_orient=False, create_body_pose=False, create_betas=False, create_transl=False)
        self.template_points = self.smpl(global_orient=torch.zeros(1, 3), betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 69)).vertices[0]

        points = self.template_points
        pivot = (points.max(0)[0] + points.min(0)[0]) /2

        tp = self.template_points
        idx_full2partial = torch.arange(tp.shape[0])
        idx_full2partial[tp[:, 1] >= pivot[1]] = -1
        idx_full2partial[tp[:, 1] < pivot[1]] = torch.arange(torch.sum(tp[:, 1] < pivot[1])) # Take points on the bottom half

        replaced_idx = idx_full2partial[self.smpl.faces.astype(int)]
        self.partial_faces = replaced_idx[torch.min(replaced_idx, 1)[0] != -1]
        self.partial_idx_mask = tp[:, 1] < pivot[1]
        partial_smpl_pts =  tp[self.partial_idx_mask]
        

        self.pivot = pivot



    def __getitem__(self, index):
        body_pose = torch.tensor(self.annot['body_pose'][index], dtype=torch.float32)

        body_pose[:3] = 0 # Cancel global orientation

        betas = torch.zeros(10)

        verts = (self.smpl(global_orient=body_pose[None, :3], body_pose=body_pose[None, 3:],
                     betas=torch.zeros(1, 10)).vertices[0] - self.loc) / self.scale
        partial_verts = verts[self.partial_idx_mask]

        full_mesh = trimesh.Trimesh(verts, self.smpl.faces)
        partial_mesh = trimesh.Trimesh(partial_verts, self.partial_faces)

        full_pcs, _ = trimesh.sample.sample_surface(full_mesh, self.n_full_pcs)
        partial_pcs, _ = trimesh.sample.sample_surface(partial_mesh, self.n_partial_pcs)

        full_pcs += self.pcs_noise * np.random.randn(*full_pcs.shape)
        partial_pcs += self.pcs_noise * np.random.randn(*partial_pcs.shape)

        query_pts, query_occ = sample_mesh(full_mesh, self.n_sample_points, sigma=self.sigma)        

        return {'body_pose': body_pose, 'betas': betas, 'query_pts': query_pts, 'query_occ': query_occ, 
                'full_pcs': full_pcs.astype(np.float32), 'partial_pcs': partial_pcs.astype(np.float32), 
                'scale': self.scale, 'loc': self.loc,
                }

    def __len__(self):
        return self.annot['body_pose'].shape[0]


class MocapOccDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, test_samples=4, val_trainset=False, subset=0, pcs_noise=0.005,
            annot_path='/scratch/wen/data/cmu_mocap.npz', smpl_path='/scratch/wen/smpl/', n_sample_points=2048, 
            n_partial_pcs=3000, n_full_pcs=6000, scale=2.0, loc=0., sigma=0.005):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_samples = test_samples
        self.val_trainset = val_trainset
        self.dataset_kwargs = dict(annot_path=annot_path, smpl_path=smpl_path, n_sample_points=n_sample_points,
           n_partial_pcs=n_partial_pcs, n_full_pcs=n_full_pcs, scale=scale, loc=loc, pcs_noise=pcs_noise, sigma=sigma)
        self.subset = subset

    def prepare_data(self):
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        train_val_dataset = MocapOccDataset(stage='train', **self.dataset_kwargs)
        test_dataset = MocapOccDataset(stage='test', **self.dataset_kwargs)

        # train/val split
        val_size = int(len(train_val_dataset) * 0.2)
        val_mask = np.zeros(len(train_val_dataset))
        #train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, len(train_val_dataset) - train_size])
        rng = default_rng(42)
        val_idxs = rng.choice(np.arange(len(train_val_dataset)), val_size)
        val_mask[val_idxs] = 1
        train_idxs = np.where(val_mask == 0)[0]
        train_dataset, val_dataset = torch.utils.data.Subset(train_val_dataset, torch.from_numpy(train_idxs)), torch.utils.data.Subset(train_val_dataset, torch.from_numpy(val_idxs))
        if self.subset > 0:
            train_dataset  = torch.utils.data.Subset(train_dataset, range(self.subset))

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers>0)

    def val_dataloader(self):
        if self.val_trainset:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers>0)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers>0)

    def test_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.train_dataset, batch_size=self.batch_size//self.test_samples, num_workers=self.num_workers)
        return DataLoader(self.test_dataset, batch_size=self.batch_size//self.test_samples, num_workers=self.num_workers)

