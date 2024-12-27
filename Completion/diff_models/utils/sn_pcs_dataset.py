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
import open3d as o3d
import random

# from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from copy import deepcopy
import warnings


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


snc_synth_category_to_id = {
    "airplane": "02691156",
    "bag": "02773838",
    "basket": "02801938",
    "bathtub": "02808440",
    "bed": "02818832",
    "bench": "02828884",
    "bicycle": "02834778",
    "birdhouse": "02843684",
    "bookshelf": "02871439",
    "bottle": "02876657",
    "bowl": "02880940",
    "bus": "02924116",
    "cabinet": "02933112",
    "can": "02747177",
    "camera": "02942699",
    "cap": "02954340",
    "car": "02958343",
    "chair": "03001627",
    "clock": "03046257",
    "dishwasher": "03207941",
    "monitor": "03211117",
    "table": "04379243",
    "telephone": "04401088",
    "tin_can": "02946921",
    "tower": "04460130",
    "train": "04468005",
    "keyboard": "03085013",
    "earphone": "03261776",
    "faucet": "03325088",
    "file": "03337140",
    "guitar": "03467517",
    "helmet": "03513137",
    "jar": "03593526",
    "knife": "03624134",
    "lamp": "03636649",
    "laptop": "03642806",
    "speaker": "03691459",
    "mailbox": "03710193",
    "microphone": "03759954",
    "microwave": "03761084",
    "motorcycle": "03790512",
    "mug": "03797390",
    "piano": "03928116",
    "pillow": "03938244",
    "pistol": "03948459",
    "pot": "03991062",
    "printer": "04004475",
    "remote_control": "04074963",
    "rifle": "04090263",
    "rocket": "04099429",
    "skateboard": "04225987",
    "sofa": "04256520",
    "stove": "04330267",
    "vessel": "04530566",
    "washer": "04554684",
    "boat": "02858304",
    "cellphone": "02992529",
}
snc_synth_category_to_id["all"] = [
    "02691156",
    "02828884",
    "02933112",
    "02958343",
    "03001627",
    "03211117",
    "03636649",
    "03691459",
    "04090263",
    "04256520",
    "04379243",
    "04401088",
    "04530566",
]  # All evaluation category in onet
snc_synth_category_to_id["cgan_all"] = ["03001627", "02691156", "04379243"]

split_prefix_dict = {"onet": "", "cgan": "cgan_"}

mask_op = [
    lambda x: x[:, 0] < 0,
    lambda x: x[:, 0] >= 0,
    lambda x: x[:, 1] < 0,
    lambda x: x[:, 1] >= 0,
    lambda x: x[:, 2] < 0,
    lambda x: x[:, 2] >= 0,
    lambda x: np.logical_and(
        np.logical_and(x[:, 0] > 0, x[:, 1] < 0), x[:, 2] > 0
    ),  # octant: front-left-bottom
    lambda x: np.logical_and(
        np.logical_and(x[:, 0] >= x.mean(0)[0], x[:, 1] <= x.mean(0)[1]), x[:, 2] >= x.mean(0)[2]
    ),  # octant: front-left-bottom
]

oct_mask_op = [
    lambda x: np.logical_and(np.logical_and(x[:, 0] > 0, x[:, 1] > 0), x[:, 2] > 0),
    lambda x: np.logical_and(np.logical_and(x[:, 0] > 0, x[:, 1] > 0), x[:, 2] < 0),
    lambda x: np.logical_and(np.logical_and(x[:, 0] > 0, x[:, 1] < 0), x[:, 2] > 0),
    lambda x: np.logical_and(np.logical_and(x[:, 0] > 0, x[:, 1] < 0), x[:, 2] < 0),
    lambda x: np.logical_and(np.logical_and(x[:, 0] < 0, x[:, 1] > 0), x[:, 2] > 0),
    lambda x: np.logical_and(np.logical_and(x[:, 0] < 0, x[:, 1] > 0), x[:, 2] < 0),
    lambda x: np.logical_and(np.logical_and(x[:, 0] < 0, x[:, 1] < 0), x[:, 2] > 0),
    lambda x: np.logical_and(np.logical_and(x[:, 0] < 0, x[:, 1] < 0), x[:, 2] < 0),
]


def fibonacci_sphere(samples=1000):
    rnd = 1.0
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(2 - np.power(y, 2))
        phi = ((i + rnd) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x, y, z])
    return points


camera_poses = fibonacci_sphere()


class ShapetNetTestSet(torch.utils.data.Dataset):
    def __init__(self, shapenet_folder, split="test"):
        self.shapenet_folder = shapenet_folder
        self.split = split
        self.models = list()
        categories = [
            i for i in os.listdir(shapenet_folder) if os.path.isdir(osp.join(shapenet_folder, i))
        ]
        # categories = ['03001627']
        # Read metadata file
        metadata_file = os.path.join(shapenet_folder, "metadata.yaml")

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {c: {"id": c, "name": "n/a"} for c in categories}

        for cat_id in categories:
            with open(f"/Datasets/ShapeNet/{cat_id}/{split}.lst", "r") as f:
                model_names = f.read().split("\n")
            self.models.extend([{"model": model, "category": cat_id} for model in model_names])

    def __getitem__(self, index):
        category, model = self.models[index]["category"], self.models[index]["model"]
        pointcloud = dict(
            **np.load(osp.join(f"{self.shapenet_folder}/{category}/{model}/pointcloud.npz"))
        )
        pointcloud_tgt = pointcloud["points"]
        normal_tgt = pointcloud["normals"]
        pointcloud_scale, pointcloud_loc = pointcloud["scale"], pointcloud["loc"]

        points = dict(**np.load(osp.join(f"{self.shapenet_folder}/{category}/{model}/points.npz")))
        points_tgt = points["points"]
        occ_tgt = np.unpackbits(points["occupancies"])[: points_tgt.shape[0]]
        points_scale, points_loc = points["scale"], points["loc"]

        assert math.isclose(
            pointcloud_scale, points_scale
        ), f"Scale between query points and pointcloud should be the same, {pointcloud_scale} != {points_scale}, diff: {pointcloud_scale - points_scale}"
        assert np.allclose(
            pointcloud_loc, points_loc
        ), "Scale between query points and pointcloud should be the same"

        return {
            "model": model,
            "category": category,
            "points_tgt": points_tgt,
            "occ_tgt": occ_tgt,
            "scale": points_scale,
            "loc": points_loc,
            "idx": index,
            "pointcloud_tgt": pointcloud_tgt,
            "normal_tgt": normal_tgt,
        }

    def __len__(self):
        return len(self.models)


class ShapeNetPcsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_folder,
        split,
        split_type="onet",
        categories=None,
        test_cat="chair",
        n_pcs_pts=300,
        n_sampled_points=2048,
        pcs_noise=0.005,
        partial_mode=None,
        with_query_mask=False,
        overfit_factor: Optional[int] = None,
        test_partial_mode: Optional[str] = None,
        num_complete_pcs: Optional[int] = None,
        **kwargs,
    ):
        """Initialization of the the 3D shape dataset.
        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        """
        # Attributes
        self.dataset_folder = dataset_folder
        self.n_pcs_pts = n_pcs_pts
        self.n_sampled_points = n_sampled_points
        self.partial_mode = partial_mode
        self.pcs_noise = pcs_noise
        self.split_type = split_type
        self.with_query_mask = with_query_mask
        self.kwargs = kwargs
        self.overfit_factor = overfit_factor  # For debugging
        self.split = split
        self.num_complete_pcs = num_complete_pcs if num_complete_pcs is not None else n_pcs_pts * 2 # x2 is default configurations for cGAN

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(dataset_folder, c))]
        categories = sorted(categories)

        # Different strateigies for test set
        if split == "test":
            categories = snc_synth_category_to_id[test_cat]
            if not isinstance(categories, list):
                categories = [categories]
            # if self.partial_mode is not None and self.partial_mode not in ['bottom', 'octant']:
            #     print(f"Note: changing partial mode from {self.partial_mode} to bottom for test set")
            #     self.partial_mode = 'bottom'
            if self.partial_mode == "rand_half":
                self.partial_mode = "bottom"
            elif self.partial_mode == "rand_oct":
                self.partial_mode = "octant"
            if test_partial_mode is not None:
                self.partial_mode = test_partial_mode

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, "metadata.yaml")

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {c: {"id": c, "name": "n/a"} for c in categories}

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]["idx"] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                warnings.warn("Category %s does not exist in dataset." % c)

            split_prefix = split_prefix_dict[self.split_type]
            split_file = os.path.join(subpath, split_prefix + split + ".lst")
            with open(split_file, "r") as f:
                models_c = f.read().split("\n")

            self.models += [{"category": c, "model": m} for m in models_c]

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.models)

    def __getitem__(self, idx):
        """Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        """
        if self.overfit_factor is not None:
            idx = idx % self.overfit_factor
        index = idx  # because idx will be overrided later

        category = self.models[idx]["category"]
        model = self.models[idx]["model"]
        c_idx = self.metadata[category]["idx"]

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {"category": category, "model": model, "c_idx": c_idx, "index": idx}

        try:
            points_dict = dict(**np.load(osp.join(model_path, "points.npz")))
        except:
            print(f"Load {osp.join(model_path, 'points.npz')} failed.")
            raise ValueError(f"Load {osp.join(model_path, 'points.npz')} failed.")
        points, occupancies = points_dict["points"], points_dict["occupancies"]
        occupancies = np.unpackbits(occupancies)

        idx = np.random.randint(points.shape[0], size=self.n_sampled_points)

        points_sampled = points[idx]
        occ_sampled = occupancies[idx]
        data["query_pts"] = points_sampled.astype(np.float32)
        data["query_occ"] = occ_sampled.astype(np.float32)
        data["scale"] = points_dict["scale"]
        data["loc"] = points_dict["loc"]

        try:
            pcs_dict = dict(**np.load(osp.join(model_path, "pointcloud.npz")))
        except:
            print(f"Load {osp.join(model_path, 'pointcloud.npz')} failed")
            raise
        pcs = pcs_dict["points"]

        if self.partial_mode == 'rand_all':
            partial_mode = np.random.choice(['rand_half', 'rand_oct', 'rand_camera'])
        else:
            partial_mode = self.partial_mode

        if partial_mode == None:
            partial_pcs = pcs
        elif partial_mode == "bottom":
            partial_pcs = pcs[pcs[:, 1] < 0]  # Bottom parts of the shape
            data["mask_type"] = 2  # hard code for bottom
        elif partial_mode == "octant":
            data["mask_type"] = 6  # hard code for octant
            partial_pcs = pcs[mask_op[data["mask_type"]](pcs)]
            if partial_pcs.shape[0] <= 0:
                data["mask_type"] = 7  # hard code for octant
                partial_pcs = pcs[mask_op[data["mask_type"]](pcs)]
                if self.split == "test":
                    warnings.warn(f"Use mass center to get center crop for {index} of {self.split}")
                if partial_pcs.shape[0] <= 0:  # For some lamps...
                    partial_pcs = pcs[pcs[:, 1] < 0]  # Bottom parts of the shape
                    data["mask_type"] = 2  # hard code for bottom
                    if self.split == "test":
                        warnings.warn(
                            f"Use half space to get partial observation for {index} of {self.split}"
                        )
        elif partial_mode == "rand_half":
            half_size = 0
            while half_size <= 0:  # Some half space doesn't have any points
                # mask_type = np.random.randint(len(mask_op))
                mask_type = np.random.randint(
                    8
                )  # This will include octant but is the original implementation of early experiments.
                # keep it for reproducibility
                partial_pcs = pcs[mask_op[mask_type](pcs)]
                half_size = partial_pcs.shape[0]
            data["mask_type"] = mask_type
        elif partial_mode == "rand_oct":
            half_size = 0
            while half_size <= 0:  # Some half space doesn't have any points
                # mask_type = np.random.randint(len(mask_op))
                mask_type = np.random.randint(len(oct_mask_op))
                partial_pcs = pcs[oct_mask_op[mask_type](pcs)]
                half_size = partial_pcs.shape[0]
            data["mask_type"] = mask_type
        elif partial_mode == "rand_part":
            half_size = 0
            combined_ops = mask_op + oct_mask_op
            while half_size <= 0:  # Some half space doesn't have any points
                # mask_type = np.random.randint(len(mask_op))
                mask_type = np.random.randint(len(combined_ops))
                partial_pcs = pcs[combined_ops[mask_type](pcs)]
                half_size = partial_pcs.shape[0]
            data["mask_type"] = mask_type
        elif partial_mode == "rand_camera":
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcs)
            camera = random.choice(camera_poses)
            radius = 100  # Borrowed from tutorial http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Hidden-point-removal
            _, pt_map = pcd.hidden_point_removal(camera, radius)
            partial_pcs = np.asarray(pcd.select_by_index(pt_map).points)
            data["mask_type"] = -1
        else:
            raise NotImplementedError()

        idx = np.random.randint(partial_pcs.shape[0], size=self.n_pcs_pts)
        pcs_sampled = partial_pcs[idx].copy()
        pcs_sampled += self.pcs_noise * np.random.randn(*pcs_sampled.shape)
        data["partial_pcs"] = pcs_sampled.astype(np.float32)

        idx = np.random.randint(pcs.shape[0], size=self.n_pcs_pts * 2)
        full_pcs_sampled = pcs[idx].copy()  # To avoid doubling noises on some points
        full_pcs_sampled += self.pcs_noise * np.random.randn(*full_pcs_sampled.shape)
        data["full_pcs"] = full_pcs_sampled.astype(np.float32)

        if self.with_query_mask:
            query_mask = np.zeros_like(data["query_occ"])
            query_mask[mask_op[data["mask_type"]](data["query_pts"])] = 1  # 1 for visible part
            data["query_mask"] = query_mask

        return data


# @DATAMODULE_REGISTRY
class ShapeNetPcsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=4,
        val_trainset=False,
        subset=0,
        shuffle_train=True,
        dataset_folder="/Datasets/ShapeNet",
        split_type="onet",
        categories: Optional[list] = None,
        test_cat="chair",
        n_pcs_pts=300,
        n_sampled_points=2048,
        pcs_noise=0.005,
        partial_mode: Optional[str] = None,
        overfit_batch=0,
        with_query_mask=False,
        test_partial_mode: Optional[str] = None,
        num_complete_pcs: Optional[int]=None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_trainset = val_trainset
        self.dataset_kwargs = dict(
            dataset_folder=dataset_folder,
            split_type=split_type,
            categories=categories,
            test_cat=test_cat,
            n_pcs_pts=n_pcs_pts,
            n_sampled_points=n_sampled_points,
            pcs_noise=pcs_noise,
            partial_mode=partial_mode,
            with_query_mask=with_query_mask,
            overfit_factor=batch_size * overfit_batch if overfit_batch > 0 else None,
            test_partial_mode=test_partial_mode,
            num_complete_pcs=num_complete_pcs,
        )
        self.subset = subset
        self.shuffle_train = shuffle_train

    def prepare_data(self):
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        # assign to use in dataloaders
        self.train_dataset = ShapeNetPcsDataset(**self.dataset_kwargs, split="train")
        self.val_dataset = ShapeNetPcsDataset(**self.dataset_kwargs, split="val")
        self.test_dataset = ShapeNetPcsDataset(**self.dataset_kwargs, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            shuffle=self.shuffle_train,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        if self.val_trainset:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                worker_init_fn=worker_init_fn,
                shuffle=False,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=worker_init_fn,
            shuffle=True,
        )

    def test_dataloader(self):
        if self.val_trainset:
            return DataLoader(
                self.train_dataset,
                batch_size=1,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn,
            )
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
        )
