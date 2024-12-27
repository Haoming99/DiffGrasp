import os
from smplx import SMPLLayer
import numpy as np
import torch
#from open3d import JVisualizer
from smplx import SMPL

import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from numpy.random import default_rng
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from typing import Optional

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

SMPL_LEFT_JOINTS = [23, 21, 19, 17, 14, 2, 5, 8, 11]
SMPL_MIDDLE_JOINTS = [0, 3, 6, 9, 12, 15]
SMPL_RIGHT_JOINTS = [13, 16, 18, 20, 22, 1, 4, 7, 10]

class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, annot_path='', smpl_path=''):
        self.annot_path = annot_path
        self.annot = dict(np.load(annot_path))
        self.smpl = SMPL(model_path=smpl_path, create_global_orient=False, create_body_pose=False, create_betas=False, create_transl=False)
        self.part_joints = [SMPL_LEFT_JOINTS, SMPL_RIGHT_JOINTS]


    def __getitem__(self, index):
        # TODO: Sample points here if that's not a bottleneck.
        body_pose = torch.tensor(self.annot['body_pose'][index], dtype=torch.float32)
        smpl_output = self.smpl(global_orient=body_pose[None, :3], body_pose=body_pose[None, 3:], betas=torch.zeros(1, 10))
        joints = smpl_output.joints[0, :24]
        joints_mask = torch.ones(joints.shape[0])
        #masked_joints = self.part_joints[torch.randint(0, 2, (1,))[0]] # This is wrong because we are encoding different joints
        masked_joints = self.part_joints[0] # Always mask out left part of joints for consistency
        joints_mask[masked_joints] = 0
        partial_joints = joints[joints_mask > 0]
        full_joints = joints
        return {'partial_joints': partial_joints, 'full_joints': full_joints, 'joints_mask': joints_mask}


    def __len__(self):
        return self.annot['body_pose'].shape[0]

class SkeletonDataModule(pl.LightningDataModule):
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
        train_val_dataset = SkeletonDataset(**self.dataset_kwargs)
        test_dataset = SkeletonDataset(**self.dataset_kwargs)

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers>0, shuffle=self.shuffle_train)

    def val_dataloader(self):
        if self.val_trainset:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers>0)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers>0)

    def test_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.train_dataset, batch_size=self.batch_size//self.test_samples, num_workers=self.num_workers)
        return DataLoader(self.test_dataset, batch_size=self.batch_size//self.test_samples, num_workers=self.num_workers)

