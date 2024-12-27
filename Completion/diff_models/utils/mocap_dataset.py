import os
from smplx import SMPLLayer
import numpy as np
import torch
from open3d import JVisualizer
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

class MocapDataset(torch.utils.data.Dataset):
    def __init__(self, stage, annot_path='', smpl_path='', query_percent=1., sample_bound=200, sample_zones=3, gpu_smpl=False, halfspace=False, est_res=False):
        self.annot_path = annot_path
        self.annot = dict(np.load(annot_path))
        self.sample_bound = sample_bound
        self.sample_zones = sample_zones

        rng = default_rng(42)
        dataset_size = len(self.annot['body_pose'])
        #test_indexs = rng.choice(np.arange(dataset_size), int(dataset_size * 0.3))
        test_mask = np.zeros(dataset_size)
        #test_mask[test_indexs] = 1
        test_mask[-int(0.2 * dataset_size):] = 1
        self.query_percent = query_percent
        self.halfspace = halfspace
        self.est_res = est_res

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
        parts = np.zeros(points.shape[0], dtype=np.int64)
        signs = [-1, 1]
        for s1 in signs:
            for s2 in signs:
                for s3 in signs:
                    pts_selected = (points[:, 0] * s1 > 0) & (points[:, 1] * s2 > 0) & (points[:, 2] * s3 > 0)
                    parts[pts_selected] = (s1 > 0) * 4 + (s2 > 0) * 2 + (s3 > 0) * 1
        part_idxs = list()
        for i in range(8):
            part_idxs.append(np.where(parts == i)[0])
        self.part_idxs = part_idxs
        self.gpu_smpl = gpu_smpl


    def __getitem__(self, index):
        # TODO: Sample points here if that's not a bottleneck.
        body_pose = torch.tensor(self.annot['body_pose'][index], dtype=torch.float32)
#         betas = torch.tensor(self.annot['betas'], dtype=torch.float32)
        if self.halfspace:
            selected_idxs = torch.arange(len(self.template_points))[self.template_points[:, 0] > 0]
        else:
            selected_zones = [self.part_idxs[i] for i in torch.randperm(len(self.part_idxs))[: self.sample_zones]]
            selected_idxs = torch.cat([torch.randperm(len(i))[:self.sample_bound] for i in selected_zones])
        query_num = int(self.query_percent * len(self.template_points))
        query_idxs = torch.randperm(len(self.template_points))[:query_num]

        betas = torch.zeros(10)
        if self.gpu_smpl:
            return {'body_pose': body_pose, 'betas': betas, 'selected_idxs': selected_idxs, 'query_idxs': query_idxs}

        verts = self.smpl(global_orient=body_pose[None, :3], body_pose=body_pose[None, 3:], betas=torch.zeros(1, 10)).vertices[0]
        sampled_points = verts[selected_idxs]
        targets = verts

        if self.est_res:
            targets -= self.template_points
            sampled_points -= self.template_points
        return {'body_pose': body_pose, 'betas': betas, "pts": sampled_points, 'query_pts': self.template_points[query_idxs],
                'targets': targets[query_idxs], 'selected_idxs': selected_idxs, 'verts': verts, 'query_idxs': query_idxs}

    def __len__(self):
        return self.annot['body_pose'].shape[0]


class MocapDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, test_samples=4, val_trainset=False, gpu_smpl=False, subset=0, **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_samples = test_samples
        self.val_trainset = val_trainset
        self.dataset_kwargs = dataset_kwargs
        self.subset = subset

    def prepare_data(self):
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        train_val_dataset = MocapDataset(stage='train', **self.dataset_kwargs)
        test_dataset = MocapDataset(stage='test', **self.dataset_kwargs)

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

