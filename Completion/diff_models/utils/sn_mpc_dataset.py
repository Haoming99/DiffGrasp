from copyreg import pickle
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
import trimesh
import math
# from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from copy import deepcopy
import warnings

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

snc_synth_category_to_id = {
    'airplane' : '02691156' ,  'bag'       : '02773838' ,  'basket'        : '02801938' ,
    'bathtub'  : '02808440' ,  'bed'       : '02818832' ,  'bench'         : '02828884' ,
    'bicycle'  : '02834778' ,  'birdhouse' : '02843684' ,  'bookshelf'     : '02871439' ,
    'bottle'   : '02876657' ,  'bowl'      : '02880940' ,  'bus'           : '02924116' ,
    'cabinet'  : '02933112' ,  'can'       : '02747177' ,  'camera'        : '02942699' ,
    'cap'      : '02954340' ,  'car'       : '02958343' ,  'chair'         : '03001627' ,
    'clock'    : '03046257' ,  'dishwasher': '03207941' ,  'monitor'       : '03211117' ,
    'table'    : '04379243' ,  'telephone' : '04401088' ,  'tin_can'       : '02946921' ,
    'tower'    : '04460130' ,  'train'     : '04468005' ,  'keyboard'      : '03085013' ,
    'earphone' : '03261776' ,  'faucet'    : '03325088' ,  'file'          : '03337140' ,
    'guitar'   : '03467517' ,  'helmet'    : '03513137' ,  'jar'           : '03593526' ,
    'knife'    : '03624134' ,  'lamp'      : '03636649' ,  'laptop'        : '03642806' ,
    'speaker'  : '03691459' ,  'mailbox'   : '03710193' ,  'microphone'    : '03759954' ,
    'microwave': '03761084' ,  'motorcycle': '03790512' ,  'mug'           : '03797390' ,
    'piano'    : '03928116' ,  'pillow'    : '03938244' ,  'pistol'        : '03948459' ,
    'pot'      : '03991062' ,  'printer'   : '04004475' ,  'remote_control': '04074963' ,
    'rifle'    : '04090263' ,  'rocket'    : '04099429' ,  'skateboard'    : '04225987' ,
    'sofa'     : '04256520' ,  'stove'     : '04330267' ,  'vessel'        : '04530566' ,
    'washer'   : '04554684' ,  'boat'      : '02858304' ,  'cellphone'     : '02992529'
}
snc_synth_category_to_id['all'] = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459',
    '04090263', '04256520', '04379243', '04401088', '04530566'] # All evaluation category in onet
snc_synth_category_to_id['cgan_all'] = ['03001627', '02691156', '04379243']

split_prefix_dict = {'onet': '', 'cgan': 'cgan_'}

mask_op = [lambda x: x[:, 0]<0, lambda x: x[:, 0] >=0, 
           lambda x: x[:, 1]<0, lambda x: x[:, 1] >=0,
           lambda x: x[:, 2]<0, lambda x: x[:, 2] >=0,
           lambda x: np.logical_and(np.logical_and(x[:, 0]>0, x[:, 1]<0), x[:, 2]>0), # octant: front-left-bottom
           lambda x: np.logical_and(np.logical_and(x[:, 0]>=x.mean(0)[0], x[:, 1]<=x.mean(0)[1]), x[:, 2]>=x.mean(0)[2]), # octant: front-left-bottom
           ]

as_homo = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
de_homo = lambda x: x[..., :-1] / x[..., -1, None]

transform4x4 = lambda x, RT: de_homo(as_homo(x) @ RT.T)

class ShapeNetMPCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, cgan_partial_folder, split, split_type='onet', categories=None, test_cat='chair', n_pcs_pts=300, n_sampled_points=2048, pcs_noise=0.005,
                 overfit_factor: Optional[int]=None, **kwargs):
        ''' Initialization of the the 3D shape dataset.
        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.cgan_partial_folder = cgan_partial_folder
        self.n_pcs_pts = n_pcs_pts
        self.n_sampled_points = n_sampled_points
        self.pcs_noise = pcs_noise
        self.split_type = split_type
        self.kwargs = kwargs
        self.overfit_factor = overfit_factor # For debugging
        self.split = split

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]
        categories = sorted(categories)

        # Different strateigies for test set
        if split == 'test':
            categories = snc_synth_category_to_id[test_cat]
            if not isinstance(categories, list):
                categories = [categories]


        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                warnings.warn('Category %s does not exist in dataset.' % c)

            split_prefix = split_prefix_dict[self.split_type]
            split_file = os.path.join(subpath, split_prefix + split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.models += [
                {'category': c, 'model': m}
                for m in models_c if osp.exists(f"{cgan_partial_folder}/{c}/{m}__0__.ply")  # part of partial pcs for training set are missing
            ]

        # trans = dict()
        # for cat_id in categories:
        #     with open(f'/scratch/wen/onet2cgan_trans/sum-{cat_id}.pkl', 'rb') as f:
        #         d = pickle.load(f)
        #     trans[cat_id] = d
        # self.trans = trans #transformation between onet and cgan coordinates 
    
        if self.split == 'test' and False: # Don't need to run test yet.
            cgan_test_seqs = dict()
            for cat_id in categories:
                cgan_test_seqs[cat_id] = dict()
                with open(f"{cgan_partial_folder}/{cat_id}/test_seq.txt", 'r') as f:
                    test_seqs = f.read().split('\n')[:-1]
                for test_file in test_seqs:
                    cat_id, fname = test_file.split('/')[-2:]
                    model_id = fname.split('_')[0]
                    cgan_test_seqs[cat_id][model_id] = fname
            
            self.cgan_test_seqs = cgan_test_seqs
            


    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        if self.overfit_factor is not None:
            idx = idx % self.overfit_factor
        index = idx #because idx will be overrided later

        category = self.models[idx]['category'] # shapenet cat id like 03001627
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx'] # index of cat_id from 0 to 10+

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {'category': category, 'model': model, 'c_idx': c_idx, 'index':idx}

        points_dict = dict(**np.load(osp.join(model_path, 'points.npz')))
        points, occupancies = points_dict['points'], points_dict['occupancies']
        occupancies = np.unpackbits(occupancies)

        # n_sampled_points = 2048

        idx = np.random.randint(points.shape[0], size=self.n_sampled_points)

        points_sampled = points[idx]

        # points_sampled = transform4x4(points_sampled, self.trans[c_idx][model])

        occ_sampled = occupancies[idx]
        data['query_pts'] = points_sampled.astype(np.float32)
        data['query_occ'] = occ_sampled.astype(np.float32)
        data['scale'] = points_dict['scale']
        data['loc'] = points_dict['loc']

        pcs_dict = dict(**np.load(osp.join(model_path, 'pointcloud.npz')))
        pcs = pcs_dict['points']

        # pcs = transform4x4(pcs, self.trans[c_idx][model])

        if self.split == 'test':
            # Load cgan instead
            partial_scan_fn = self.cgan_test_seqs[category][model]
            partial_pcs = np.array(trimesh.load(f"{self.cgan_partial_folder}/{category}/{partial_scan_fn}").vertices)
            pcs = partial_pcs
        else:
            rand_idx = np.random.randint(0, 8)
            partial_pcs = np.array(trimesh.load(f"{self.cgan_partial_folder}/{category}/{model}__{rand_idx}__.ply").vertices)
            # TODO: create partial pcs using xgutils or from cgan training data

        idx = np.random.randint(partial_pcs.shape[0], size=self.n_pcs_pts)
        pcs_sampled = partial_pcs[idx].copy()
        pcs_sampled += self.pcs_noise * np.random.randn(*pcs_sampled.shape)
        data['partial_pcs'] = pcs_sampled.astype(np.float32)

        idx = np.random.randint(pcs.shape[0], size=self.n_pcs_pts * 2)
        full_pcs_sampled = pcs[idx].copy() # To avoid doubling noises on some points
        full_pcs_sampled += self.pcs_noise * np.random.randn(*full_pcs_sampled.shape)
        data['full_pcs'] = full_pcs_sampled.astype(np.float32)
        
        return data

# @DATAMODULE_REGISTRY
class ShapeNetMPCDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, val_trainset=False, subset=0, shuffle_train=True,
            dataset_folder='/Datasets/ShapeNet', cgan_partial_folder='/scratch/wen/shapenet_dim32_sdf_pc', split_type='onet', categories: Optional[list]=None, test_cat='chair', n_pcs_pts=300,
            n_sampled_points=2048, pcs_noise=0.005, overfit_batch=0, 
            **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_trainset = val_trainset
        self.dataset_kwargs = dict(
            dataset_folder=dataset_folder, cgan_partial_folder=cgan_partial_folder, split_type=split_type, categories=categories, test_cat=test_cat, n_pcs_pts=n_pcs_pts,
            n_sampled_points=n_sampled_points, pcs_noise=pcs_noise, overfit_factor=batch_size * overfit_batch if overfit_batch > 0 else None,
            )
        self.subset = subset
        self.shuffle_train = shuffle_train

    def prepare_data(self):
        # download only
        pass

    def setup(self, stage: Optional[str] = None):

        # assign to use in dataloaders
        self.train_dataset = ShapeNetMPCDataset(**self.dataset_kwargs, split='train')
        self.val_dataset = ShapeNetMPCDataset(**self.dataset_kwargs, split='val')
        self.test_dataset = ShapeNetMPCDataset(**self.dataset_kwargs, split='test')

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

