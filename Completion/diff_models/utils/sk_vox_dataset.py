import os
from smplx import SMPLLayer
import numpy as np
import torch
from smplx import SMPL

import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from numpy.random import default_rng
from torch.utils.data import DataLoader
from typing import Optional
import trimesh
from libs.libmesh import check_mesh_contains
from einops import rearrange

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

SMPL_LEFT_JOINTS = [23, 21, 19, 17, 14, 2, 5, 8, 11]
SMPL_MIDDLE_JOINTS = [0, 3, 6, 9, 12, 15]
SMPL_RIGHT_JOINTS = [13, 16, 18, 20, 22, 1, 4, 7, 10]

class SkVoxDataset(torch.utils.data.Dataset):
    def __init__(self, annot_path='', smpl_path='', VOXEL_RESOLUTION=32, canon=False, mask_out='left'):
        self.annot_path = annot_path
        self.annot = dict(np.load(annot_path))
        self.smpl = SMPL(model_path=smpl_path, create_global_orient=False, create_body_pose=False, create_betas=False, create_transl=False)
        smpl_output = self.smpl(global_orient=torch.zeros(1, 3),  body_pose=torch.zeros(1, 69), betas=torch.zeros(1, 10))
        template_points = smpl_output.vertices[0]

        if mask_out == 'left':
            self.half_idxs = template_points[:, 0] < 0
            self.part_joints = SMPL_LEFT_JOINTS
            print('Masking out left part')
        elif mask_out == 'right':
            self.half_idxs = template_points[:, 0] > 0
            self.part_joints = SMPL_RIGHT_JOINTS
            print('Masking out right part')
        self.canon = canon
        v_min = [-0.9296, -1.2696, -0.8471]
        v_max = [0.9601, 0.9930, 1.0547] # Obatined by randomly sampling 1% traning data
        x_ = np.linspace(v_min[0], v_max[0], VOXEL_RESOLUTION)
        y_ = np.linspace(v_min[1], v_max[1], VOXEL_RESOLUTION)
        z_ = np.linspace(v_min[2], v_max[2], VOXEL_RESOLUTION)
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

        self.pts3d = rearrange([x, y, z], 'c d h w -> (d h w) c')
        self.VOXEL_RESOLUTION = VOXEL_RESOLUTION



    def __getitem__(self, index):
        body_pose = torch.tensor(self.annot['body_pose'][index], dtype=torch.float32)
        if self.canon:
            smpl_output = self.smpl(global_orient=torch.zeros_like(body_pose[None, :3]), body_pose=body_pose[None, 3:], betas=torch.zeros(1, 10))
        else:
            smpl_output = self.smpl(global_orient=body_pose[None, :3], body_pose=body_pose[None, 3:], betas=torch.zeros(1, 10))
        joints = smpl_output.joints[0, :24]
        joints_mask = torch.ones(joints.shape[0])
        masked_joints = self.part_joints # Always mask out left part of joints for consistency
        joints_mask[masked_joints] = 0
        partial_joints = joints[joints_mask > 0]
        full_joints = joints

        full_pcs = smpl_output.vertices[0]
        partial_pcs = full_pcs[self.half_idxs]

        mesh = trimesh.Trimesh(smpl_output.vertices[0], self.smpl.faces, process=False)
        pts_occ = check_mesh_contains(mesh, self.pts3d)
        vox_occ = rearrange(pts_occ, '(d h w) -> d h w', w=self.VOXEL_RESOLUTION, h=self.VOXEL_RESOLUTION, d=self.VOXEL_RESOLUTION)

        return {'partial_joints': partial_joints, 'full_joints': full_joints, 'joints_mask': joints_mask,
                'vox_occ': torch.from_numpy(vox_occ).float(), 'index': index,
               }


    def __len__(self):
        return self.annot['body_pose'].shape[0]

class SkVoxDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, test_samples=4, val_trainset=False, gpu_smpl=False, subset=0, shuffle_train=True, **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_samples = test_samples
        self.val_trainset = val_trainset
        self.dataset_kwargs = dataset_kwargs
        self.subset = subset
        self.shuffle_train = shuffle_train

    def prepare_data(self):
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        train_val_dataset = SkVoxDataset(**self.dataset_kwargs)
        test_dataset = SkVoxDataset(**self.dataset_kwargs)

        # train/val split
        val_size = int(len(train_val_dataset) * 0.2)
        train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [len(train_val_dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))
        if self.subset > 0:
            train_dataset  = torch.utils.data.Subset(train_dataset, range(self.subset))

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers>0, shuffle=self.shuffle_train, drop_last=True)

    def val_dataloader(self):
        if self.val_trainset:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers>0)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers>0)

    def test_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.train_dataset, batch_size=self.batch_size//self.test_samples, num_workers=self.num_workers)
        return DataLoader(self.test_dataset, batch_size=self.batch_size//self.test_samples, num_workers=self.num_workers)
