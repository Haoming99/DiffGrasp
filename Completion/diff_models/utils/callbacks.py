from typing import Optional, Tuple
import os
import logging
import torch
import tensorboard

import torchvision
import cv2
from pytorch_lightning import Callback, LightningModule, Trainer
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import trimesh
from utils.render import CanonRenderer, ShapeNetRender, SynRoomRender, CornerRender, pts2mesh
from utils.viz import plot_pcs
# from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
import wandb
import warnings
from collections.abc import Iterable
from einops import repeat, rearrange
# import pyrender
import numpy as np

def log_image(loggers, tag, img, global_step):
    if not isinstance(loggers, Iterable):
        loggers = [loggers]
    for logger in loggers:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_image(tag, img, global_step=global_step)
        elif isinstance(logger, pl.loggers.WandbLogger):
            img_wb = rearrange(img, 'c h w -> h w c').numpy()
            logger.experiment.log({tag: wandb.Image(img_wb)}, step=global_step)
        else:
            warnings.warn(f"Unknown logger: {logger}")


class MeshGenerationCallback(Callback):
    def __init__(self, viz_dir, num_viz=2, num_samples=3):
        super().__init__()
        os.makedirs(viz_dir, exist_ok=True)
        self.viz_dir = None #viz_dir save visualization has been disabled to save space
        self.num_viz = num_viz
        self.num_samples = num_samples
        self.render = CanonRenderer()

    def export_mesh(self, trainer, pl_module, batch, batch_idx, dataloader_idx, stage):
        if batch_idx > 0:
            # Only visualize the first batch
            return

        for sample in [True, False]:
            mesh_list, img_list = pl_module.generate_mesh(batch, batch_idx, max_bs=self.num_viz, num_samples=self.num_samples, sample=sample)

            # Render images and write to tensorboard
            all_imgs = list()
            for b_i in range(self.num_viz):
                all_imgs.append(cv2.resize(img_list[b_i], (self.render.img_h, self.render.img_w)))
                rendered_meshes = list(map(self.render, mesh_list[b_i]))
                all_imgs.extend(rendered_meshes)
            all_imgs = [torch.tensor(i.copy()).permute([2, 0, 1]) for i in all_imgs]
            grid_img = torchvision.utils.make_grid(all_imgs, self.num_samples+1) # +1 for input

            trainer.logger.experiment.add_image(f"{stage}/{'partial' if sample else 'full'}", grid_img, global_step=trainer.global_step)
            # plt.imsave(f"{self.viz_dir}/{trainer.global_step}.jpg", grid_img.permute([1, 2, 0]).numpy())

            # for b_i in range(self.num_viz):
            #     img = img_list[b_i]
            #     index = batch['index'][b_i]
            #     plt.imsave(f"{self.viz_dir}/{trainer.global_step}-{index}.png", img)
            #     for sample_idx in range(self.num_samples):
            #         m = mesh_list[b_i][sample_idx]
            #         m.export(f"{self.viz_dir}/{trainer.global_step}-{index}-{stage}-{'partial' if sample else 'full'}-{sample_idx}.obj")


    # def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.export_mesh(trainer, pl_module, batch, batch_idx, dataloader_idx, stage='val')


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        pl_module.eval()
        self.export_mesh(trainer, pl_module, batch, batch_idx, dataloader_idx, stage='train')
        pl_module.train()

class FullMeshGenerationCallback(MeshGenerationCallback):
    def __init__(self, faces, **kwargs):
        super().__init__(**kwargs)
        self.faces = faces

    def export_mesh(self, trainer, pl_module, batch, batch_idx, dataloader_idx, stage):
        if batch_idx > 0:
            # Only visualize the first batch
            return

        for sample in [True, False]:
            mesh_list, input_list = pl_module.generate_mesh(batch, batch_idx, max_bs=self.num_viz, num_samples=self.num_samples, sample=sample)
            input_mesh_list = list()

            for b_i in range(self.num_viz):
                cur_in = input_list[b_i]
                index = batch['index'][b_i]

                m = trimesh.Trimesh(cur_in.cpu(), self.faces) # SMPL mesh
                # m.export(f"{self.viz_dir}/{trainer.global_step}-{index}.obj")
                input_mesh_list.append(m)

                # for sample_idx in range(self.num_samples):
                #     m = mesh_list[b_i][sample_idx]
                #     m.export(f"{self.viz_dir}/{trainer.global_step}-{index}-{stage}-{'partial' if sample else 'full'}-{sample_idx}.obj")


            # Render images and write to tensorboard
            all_imgs = list()
            for b_i in range(self.num_viz):
                all_imgs.append(self.render(input_mesh_list[b_i]))
                rendered_meshes = list(map(self.render, mesh_list[b_i]))
                all_imgs.extend(rendered_meshes)
            all_imgs = [torch.tensor(i.copy()).permute([2, 0, 1]) for i in all_imgs]
            grid_img = torchvision.utils.make_grid(all_imgs, self.num_samples+1)

            trainer.logger.experiment.add_image(f"{stage}/{'partial' if sample else 'full'}", grid_img, global_step=trainer.global_step)
            # plt.imsave(f"{self.viz_dir}/{trainer.global_step}-{stage}-{'partial' if sample else 'full'}.jpg", grid_img.permute([1, 2, 0]).numpy())

class KptVoxGenerationCallback(MeshGenerationCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def export_mesh(self, trainer, pl_module, batch, batch_idx, dataloader_idx, stage):
        if batch_idx > 0:
            # Only visualize the first batch
            return

        for sample in [True, False]:
            all_imgs = pl_module.generate_viz(batch, batch_idx, max_bs=self.num_viz, num_samples=self.num_samples, sample=sample)

            all_imgs = [torch.tensor(i.copy()).permute([2, 0, 1]) for i in all_imgs]
            grid_img = torchvision.utils.make_grid(all_imgs, 1)
            plt.imsave(f"{self.viz_dir}/{trainer.global_step}.jpg", grid_img.permute([1, 2, 0]).numpy())

            trainer.logger.experiment.add_image(f"{stage}/{'partial' if sample else 'full'}", grid_img, global_step=trainer.global_step)

# @CALLBACK_REGISTRY
class ShapeNetMeshGenerationCallback(MeshGenerationCallback):
    #def __init__(self, dataset_dir: str, **kwargs):
    def __init__(self, viz_dir: Optional[str]=None, num_viz=2, num_samples=3, dataset_dir: Optional[str]=None, train_p=0.01, val_p=0.2, sample_only=False):
        super().__init__(viz_dir=viz_dir, num_viz=num_viz, num_samples=num_samples)
        self.dataset_dir = dataset_dir
        self.render = ShapeNetRender()
        self.train_p = train_p
        self.val_p = val_p
        self.sample_only = sample_only

    def export_mesh(self, trainer, pl_module, batch, batch_idx, dataloader_idx, stage):
        if stage == 'train' and torch.rand(1).item() > self.train_p: # Visualize 1% batch of training set
            return
        if stage == 'val' and (batch['category'][0] == '02691156' or torch.rand(1).item() > self.val_p): # Visualize 10% batch of validation set
            return

        it = [True, False] if not self.sample_only else [True]
        for sample in it:
            try:
                mesh_list, input_list = pl_module.generate_mesh(batch, batch_idx, max_bs=self.num_viz, num_samples=self.num_samples, sample=sample)
            except Exception as e:
                logging.error(f"Failed to generate mesh with sample={sample}")
                print(e)
                continue

            for b_i in range(self.num_viz):
                index = batch['index'][b_i]

                # for sample_idx in range(self.num_samples):
                #     m = mesh_list[b_i][sample_idx]
                #     m.export(f"{self.viz_dir}/{trainer.global_step}-{index}-{stage}-{'partial' if sample else 'full'}-{sample_idx}.obj")


            # Render images and write to tensorboard
            all_imgs = list()
            for b_i in range(self.num_viz):
                img = plt.imread(f"{self.dataset_dir}/{batch['category'][b_i]}/{batch['model'][b_i]}/img_choy2016/000.jpg")
                if img.ndim == 2: #Gray scale image
                    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

                all_imgs.append(img)
                plotted_pcs = plot_pcs(input_list[b_i].cpu().numpy(), return_img=True)
                img_pcs = cv2.resize(plotted_pcs, (self.render.img_h, self.render.img_w))
                all_imgs.append(img_pcs)

                rendered_meshes = list(map(self.render, mesh_list[b_i]))
                all_imgs.extend(rendered_meshes)
            all_imgs = [torch.tensor(i.copy()).permute([2, 0, 1]) for i in all_imgs]
            grid_img = torchvision.utils.make_grid(all_imgs, self.num_samples+2)

            # trainer.logger.experiment.add_image(f"{stage}/{'partial' if sample else 'full'}", grid_img, global_step=trainer.global_step)
            log_image(trainer.loggers, f"{stage}/{'partial' if sample else 'full'}", grid_img, global_step=trainer.global_step)
            # plt.imsave(f"{self.viz_dir}/{trainer.global_step}.jpg", grid_img.permute([1, 2, 0]).numpy())


class SynRoomVisCallback(MeshGenerationCallback):
    #def __init__(self, dataset_dir: str, **kwargs):
    def __init__(self, viz_dir: Optional[str]=None, num_viz=2, num_samples=3, dataset_dir: Optional[str]=None, train_p=0.01, val_p=0.2, sample_only=False):
        super().__init__(viz_dir=viz_dir, num_viz=num_viz, num_samples=num_samples)
        self.dataset_dir = dataset_dir
        self.render = SynRoomRender()
        self.train_p = train_p
        self.val_p = val_p
        self.sample_list = [True] if sample_only else [True, False]

    def export_mesh(self, trainer, pl_module, batch, batch_idx, dataloader_idx, stage):
        if stage == 'train' and torch.rand(1).item() > self.train_p: # Visualize 1% batch of training set
            return
        if stage == 'val' and torch.rand(1).item() > self.val_p: # Visualize 10% batch of validation set
            return

        mesh_recons = {True: list(), False: list()}
        full_pcs = batch['full_pcs'][:self.num_viz].cpu().numpy()
        partial_pcs = batch['partial_pcs'][:self.num_viz].cpu().numpy()

        for sample in self.sample_list:
#             tbl = wandb.Table(columns=["gt", "input", "recon#0", "recon#1", "recon#2"])
            try:
                mesh_list, input_list = pl_module.generate_mesh(batch, batch_idx, max_bs=self.num_viz, num_samples=self.num_samples, sample=sample)
            except Exception as e:
                logging.error(f"Failed to generate mesh with sample={sample}")
                print(e)
                continue

            img_records = list()

            for b_idx in range(self.num_viz):
                index = batch['index'][b_idx]
                gt_img = self.render(pts2mesh(full_pcs[b_idx]))
                input_img = self.render(pts2mesh(partial_pcs[b_idx]))
                recon_imgs = [self.render(m) for m in mesh_list[b_idx]]
                # for s_idx, m in enumerate(mesh_list[b_idx]):
                    # mesh_save_path = f"{self.viz_dir}/{trainer.global_step}-{index}-{stage}-{'partial' if sample else 'full'}-{s_idx}.obj"
                    # m.export(mesh_save_path)

                cur_img_records = [gt_img, input_img] + recon_imgs
#                 tbl.add_data(*(wandb.Image(im) for im in cur_img_records))
                img_records.extend(cur_img_records)
            all_imgs = [torch.tensor(i.copy()).permute([2, 0, 1]) for i in img_records]
            grid_img = torchvision.utils.make_grid(all_imgs, len(all_imgs) // len(mesh_list))

            mode_str = 'partial' if sample else 'full'
            # trainer.logger.experiment.add_image(f"{stage}/{'partial' if sample else 'full'}", grid_img, global_step=trainer.global_step)
            log_image(trainer.loggers, f"{stage}/{mode_str}", grid_img, global_step=trainer.global_step)
            # plt.imsave(f"{self.viz_dir}/{trainer.global_step}-{mode_str}.jpg", grid_img.permute([1, 2, 0]).numpy())


# @CALLBACK_REGISTRY
class MocapMeshGenerationCallback(MeshGenerationCallback):
    #def __init__(self, dataset_dir: str, **kwargs):
    def __init__(self, viz_dir: Optional[str]=None, num_viz=2, num_samples=3, dataset_dir: Optional[str]=None, train_p=0.01, val_p=0.2, sample_only=False):
        super().__init__(viz_dir=viz_dir, num_viz=num_viz, num_samples=num_samples)
        self.dataset_dir = dataset_dir
        self.render = CornerRender()
        self.train_p = train_p
        self.val_p = val_p
        self.sample_only = sample_only

    def export_mesh(self, trainer, pl_module, batch, batch_idx, dataloader_idx, stage):
        if stage == 'train' and torch.rand(1).item() > self.train_p: # Visualize 1% batch of training set
            return
        if stage == 'val' and (torch.rand(1).item() > self.val_p): # Visualize 10% batch of validation set
            return

        it = [True, False] if not self.sample_only else [True]

        full_pcs = batch['full_pcs'][:self.num_viz].cpu().numpy()
        partial_pcs = batch['partial_pcs'][:self.num_viz].cpu().numpy()

        for sample in it:
            try:
                mesh_list, input_list = pl_module.generate_mesh(batch, batch_idx, max_bs=self.num_viz, num_samples=self.num_samples, sample=sample)
            except Exception as e:
                logging.error(f"Failed to generate mesh with sample={sample}")
                print(e)
                continue


            # Render images and write to tensorboard
            all_imgs = list()
            for b_i in range(self.num_viz):
                cur_mesh_list = [pts2mesh(full_pcs[b_i]), pts2mesh(partial_pcs[b_i])] + mesh_list[b_i]

                rendered_meshes = list(map(self.render, cur_mesh_list))
                all_imgs.extend(rendered_meshes)
            all_imgs = [torch.tensor(i.copy()).permute([2, 0, 1]) for i in all_imgs]
            grid_img = torchvision.utils.make_grid(all_imgs, self.num_samples+2)

            # trainer.logger.experiment.add_image(f"{stage}/{'partial' if sample else 'full'}", grid_img, global_step=trainer.global_step)
            log_image(trainer.loggers, f"{stage}/{'partial' if sample else 'full'}", grid_img, global_step=trainer.global_step)
            # plt.imsave(f"{self.viz_dir}/{trainer.global_step}.jpg", grid_img.permute([1, 2, 0]).numpy())

# @CALLBACK_REGISTRY
class LatentRegWeightCallback(Callback):
    def __init__(self, final_weight=1e-4):
        super().__init__()
        self.final_weight = final_weight
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pseudo_epoch = trainer.global_step * 64 / 30000
        pl_module.latent_reg_weight = min(1, pseudo_epoch / 100) * self.final_weight 


# @CALLBACK_REGISTRY
class ADConvWeightCallback(Callback):
    def __init__(self, latent_reg_weight=1e-4, recon_weight=1.):
        super().__init__()
        self.latent_reg_weight = latent_reg_weight 
        self.recon_weight = recon_weight
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pseudo_epoch = trainer.global_step * 64 / 30000
        pl_module.latent_reg_weight = min(1, pseudo_epoch / 100) * self.latent_reg_weight
        pl_module.recon_weight = min(1, pseudo_epoch / 200) * self.recon_weight

# @CALLBACK_REGISTRY
class VaeWeightCallback(Callback):
    def __init__(self, kl_weight=1e-4, recon_weight=1., anneal_epochs=100, epoch_multiplier=64/30000):
        super().__init__()
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.anneal_epochs = anneal_epochs
        self.epoch_multiplier = epoch_multiplier
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pseudo_epoch = trainer.global_step * self.epoch_multiplier
        pl_module.kl_weight = min(1, pseudo_epoch / self.anneal_epochs) * self.kl_weight

# @CALLBACK_REGISTRY
class VaeWeightStepCallback(Callback):
    def __init__(self, kl_weight=1e-4, recon_weight=1., anneal_steps=1e5):
        super().__init__()
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.anneal_steps = anneal_steps
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pl_module.kl_weight = min(1, trainer.global_step / self.anneal_steps) * self.kl_weight

# @CALLBACK_REGISTRY
class LocalVaeWeightStepCallback(Callback):
    def __init__(self, kl_weight=1e-4, local_kl_weight=1e-4, recon_weight=1., anneal_steps=1e5, local_anneal_steps=None):
        super().__init__()
        self.kl_weight = kl_weight
        self.local_kl_weight = local_kl_weight
        self.recon_weight = recon_weight
        self.anneal_steps = anneal_steps
        if local_anneal_steps is None:
            self.local_anneal_steps = anneal_steps
        else:
            self.local_anneal_steps = local_anneal_steps
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pl_module.kl_weight = min(1, trainer.global_step / self.anneal_steps) * self.kl_weight
        pl_module.local_kl_weight = min(1, trainer.global_step / self.local_anneal_steps) * self.local_kl_weight

# @CALLBACK_REGISTRY
class VaeWarmUpCallback(Callback):
    def __init__(self, kl_weight=1e-4, recon_weight=1., anneal_steps=1e5, warmup_steps=2e3):
        super().__init__()
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pl_module.kl_weight = min(1, max(0, (trainer.global_step - self.warmup_steps) / self.anneal_steps)) * self.kl_weight

# @CALLBACK_REGISTRY
class RegReduceCallback(Callback):
    def __init__(self, reg_prob_weight, reg_prior_weight, stay_steps=10e3, reduce_steps=50e3):
        super().__init__()
        self.reg_prob_weight = reg_prob_weight
        self.reg_prior_weight = reg_prior_weight
        self.stay_steps = stay_steps
        self.reduce_steps = reduce_steps
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        factor = 1 if trainer.global_step < self.stay_steps else max(0, 1 - (trainer.global_step - self.stay_steps) / self.reduce_steps)
        pl_module.reg_prob_weight = self.reg_prob_weight * factor
        pl_module.reg_prior_weight = self.reg_prior_weight * factor
