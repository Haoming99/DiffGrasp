from lib2to3.pgen2.token import OP
import torch
import numpy as np
import os
import os.path as osp
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from typing import Optional, Union
from torch.utils.data import DataLoader
import yaml
import math
# from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from copy import deepcopy
import warnings
from utils.syn_room_data import SubsamplePointcloud, PointCloudField, PointcloudNoise, PointsField, SubsamplePoints
import torchvision
from glob import glob
import open3d as o3d

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class SyntheticRoomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, n_partial_pcs=10000, n_full_pcs=20000, n_query_points=2048, 
                 obj_min_rate=0.2, pcs_noise=0.005, multi_files=10, overfit_factor: Optional[int]=None, partial_mode="cgca"):
        self.dataset_folder = dataset_folder
        self.split = split
        self.obj_min_rate = obj_min_rate
        self.overfit_factor = overfit_factor
        self.partial_mode = partial_mode
        self.n_partial_pcs = n_partial_pcs
        
        models = list()
        for room_path in sorted(glob(f"{dataset_folder}/room*/")):
            room_name = osp.basename(osp.dirname(room_path))
            with open(f"{room_path}/{split}.lst", 'r') as f:
                scene_ids = f.read().splitlines()
                models.extend([{'room_name': room_name, 'id': scene_id} for scene_id in scene_ids])            
        self.models = models

        if self.partial_mode == "cgca":
            transform = torchvision.transforms.Compose([
                SubsamplePointcloud(N=n_partial_pcs),
                PointcloudNoise(stddev=pcs_noise)
            ])
            self.partial_pcs_field = PointCloudField(
                'pointcloud', transform, multi_files=multi_files
            )

        full_pcs_transform = torchvision.transforms.Compose([
            SubsamplePointcloud(N=n_full_pcs),
            PointcloudNoise(stddev=pcs_noise)
        ])
        self.full_pcs_field = PointCloudField(
            'pointcloud', full_pcs_transform, multi_files=multi_files
        )
        
        points_transform = SubsamplePoints(N=n_query_points)
        self.points_field = PointsField('points_iou', transform=points_transform, unpackbits=True, multi_files=multi_files)
        
    def __getitem__(self, index) -> dict:
        if self.overfit_factor is not None:
            index = index % self.overfit_factor
            
        model = self.models[index]
        model_dir = osp.join(self.dataset_folder, model['room_name'], model['id'])
        if self.split == 'test':
            if self.partial_mode == "cgca":
                partial_pcs = dict(np.load(osp.join(self.dataset_folder, self.split, model['room_name'], model['id'], f"pointcloud_{self.obj_min_rate}.npz")))['points']
            elif self.partial_mode == "camera":
                full_pcs = self.full_pcs_field.load(model_dir, None, None, obj_min_rate=None)[None]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(full_pcs)
                camera = [0, 0, 0]
                radius = 100  # Borrowed from tutorial http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Hidden-point-removal
                _, pt_map = pcd.hidden_point_removal(camera, radius)
                partial_pcs = np.asarray(pcd.select_by_index(pt_map).points)
                idx = np.random.randint(partial_pcs.shape[0], size=self.n_partial_pcs)
                partial_pcs = partial_pcs[idx].astype("float32")

            # Don't need the following things in test set
            full_pcs = torch.empty(0)
            points_iou = torch.empty(0)
            query_pts = torch.empty(0)
            query_occ = torch.empty(0)
        else:
            full_pcs = self.full_pcs_field.load(model_dir, None, None, obj_min_rate=None)[None]
            if self.partial_mode == "cgca":
                partial_pcs = self.partial_pcs_field.load(model_dir, None, None, obj_min_rate=self.obj_min_rate)[None]
            elif self.partial_mode == "camera":
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(full_pcs)
                camera = [0, 0, 0]
                radius = 100  # Borrowed from tutorial http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Hidden-point-removal
                _, pt_map = pcd.hidden_point_removal(camera, radius)
                partial_pcs = np.asarray(pcd.select_by_index(pt_map).points)
                idx = np.random.randint(partial_pcs.shape[0], size=self.n_partial_pcs)
                partial_pcs = partial_pcs[idx].astype("float32")
            
            points_iou = self.points_field.load(model_dir, None, None)
            query_pts = points_iou[None]
            query_occ = points_iou['occ']
        
        return {'category': model['room_name'], 'model': model['id'], 'c_idx': model['room_name'], 'index':index,
               'full_pcs': full_pcs, 'partial_pcs': partial_pcs, 'query_pts': query_pts, 'query_occ': query_occ,
                'scale': torch.tensor(1.), 'loc': torch.tensor(0.),
               }
    
    def __len__(self):
        return len(self.models)

# @DATAMODULE_REGISTRY
class SyntheticRoomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=8, val_trainset=False, subset=0, shuffle_train=True,
            dataset_folder='/scratch/wen/synthetic_room_dataset/',  n_partial_pcs=10000, n_full_pcs=20000, n_query_points=2048, 
                 obj_min_rate=0.2, pcs_noise=0.005, multi_files=10, overfit_batch=0, partial_mode="cgca",
            **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_trainset = val_trainset
        self.dataset_kwargs = dict(
            dataset_folder=dataset_folder, n_partial_pcs=n_partial_pcs, n_full_pcs=n_full_pcs, 
            n_query_points=n_query_points, obj_min_rate=obj_min_rate, pcs_noise=pcs_noise, multi_files=multi_files,
            overfit_factor=batch_size * overfit_batch if overfit_batch > 0 else None, partial_mode=partial_mode
            )
        self.subset = subset
        self.shuffle_train = shuffle_train

    def prepare_data(self):
        # download only
        pass

    def setup(self, stage: Optional[str] = None):

        # assign to use in dataloaders
        self.train_dataset = SyntheticRoomDataset(**self.dataset_kwargs, split='train')
        self.val_dataset = SyntheticRoomDataset(**self.dataset_kwargs, split='val')
        self.test_dataset = SyntheticRoomDataset(**self.dataset_kwargs, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers>0, shuffle=self.shuffle_train, drop_last=True, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        if self.val_trainset:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers>0, worker_init_fn=worker_init_fn, shuffle=False)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers>0, worker_init_fn=worker_init_fn, shuffle=True)

    def test_dataloader(self):
        if self.val_trainset:
            return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers, worker_init_fn=worker_init_fn)
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, worker_init_fn=worker_init_fn, shuffle=False)

