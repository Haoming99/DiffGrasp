from optparse import Option
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import pytorch_lightning as pl
from typing import Union, Optional
import itertools
import numpy as np
import logging
from models.unet import UNet
from utils.eval_utils import compute_iou
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from utils.mesh import generate_from_latent, export_shapenet_samples
from einops import repeat, reduce, rearrange
from models.kilo_base import KiloBase
from models.conv_decoder import ConvONetDecoder
from models.encoder import ResnetPointnet
from models.vaes import VAEEncoder, GroupVAEEncoder
from models.encoder import ResnetPointnet
from models import LocalPoolPointnet
from models.unet3d import UNet3DCLI
from models.fcresnet import GroupLinear, GroupFCBlock, GroupFCResBlock, GroupDoubleFCLayer
# from torch_ema import ExponentialMovingAverage

@MODEL_REGISTRY
class ImplicitFunc(KiloBase):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder: nn.Module, encoder: nn.Module, latent_dim=32,
        lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1,
        reduction='sum', test_num_pts=2048, interactive_debug=False):
        
        super().__init__(lr=lr, padding=padding, b_min=b_min, b_max=b_max, points_batch_size=points_batch_size, batch_size=batch_size, test_num_samples=test_num_samples,
                batchnorm=False, reduction=reduction, invis_loss_weight=1., interactive_debug=interactive_debug, test_num_pts=test_num_pts)

        self.decoder = decoder # ConvONet decoder
        self.encoder = encoder

        self.save_hyperparameters(ignore=["decoder", "encoder"])

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z: dict or tensor for decdoer
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        return logits


    def step(self, batch, batch_idx):
        full_pcs =  batch['full_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']

        c = self.encoder(full_pcs)

        pred_occ = self.decode(query_pts, c) 
        
        recon_loss = F.binary_cross_entropy_with_logits(
                pred_occ, query_occ, reduction='none')


        if self.reduction == 'sum':
            loss = recon_loss.sum(-1).mean()
        elif self.reduction == 'mean':
            loss = recon_loss.mean()

        if self.interactive_debug and (torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss))):
            import ipdb; ipdb.set_trace()

        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()

        logs = {
            "loss": loss,
            'lr': self.lr,
            'iou': iou_pts,
            "batch_size": float(self.batch_size)
        }

        return loss, logs

    def encode_inputs(self, pcs):
        return self.encoder(pcs)


    def generate_mesh(self, batch, batch_idx, max_bs=3, num_samples=1, sample=True, denormalize=False):
        pcs = batch['full_pcs'] # No difference between partial and complete pcs so far
        bs = pcs.shape[0]
        if max_bs < 0:
            max_bs = bs
        device = self.device
        assert num_samples == 1, f"This module only support num_samples=1, but got num_samples={num_samples}"

        if denormalize:
            loc = batch['loc'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()

        with torch.no_grad():
            c = self.encode_inputs(pcs)

            mesh_list = list()
            input_list = list()
            for b_i in range(min(max_bs, bs)):
                cur_mesh_list = list()
                input_list.append(batch['full_pcs'][b_i])

                state_dict = dict()
                for sample_idx in range(num_samples):
                    if isinstance(c, dict):
                        cur_c = {k: v[b_i, None] for k, v in c.items()}
                    else:
                        cur_c = c[b_i, None]
                    generated_mesh = generate_from_latent(self.eval_points, z=None, c=cur_c,
                                                          state_dict=state_dict, padding=self.padding,
                                                          B_MAX=self.b_max, B_MIN=self.b_min, device=self.device)
                    cur_mesh_list.append(generated_mesh)
                    if denormalize:
                        generated_mesh.vertices = (generated_mesh.vertices + loc[b_i]) * scale[b_i]
                mesh_list.append(cur_mesh_list)

        return mesh_list, input_list

    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        mesh_list, _ = self.generate_mesh(batch, batch_idx, max_bs=-1, num_samples=self.test_num_samples, sample=True, denormalize=True)
        denormalized_pcs = (batch['full_pcs'] + batch['loc']) * batch['scale']
        export_shapenet_samples(mesh_list, batch['category'], batch['model'], denormalized_pcs, self.out_dir, test_num_pts=self.test_num_pts)
    

    def predict_step(self, batch, batch_idx):
        mesh_list, _ = self.generate_mesh(batch, batch_idx, max_bs=-1, num_samples=self.test_num_samples, sample=True, denormalize=True)
        denormalized_pcs = (batch['full_pcs'] + batch['loc']) * batch['scale']
        export_shapenet_samples(mesh_list, batch['category'], batch['model'], denormalized_pcs, self.out_dir, test_num_pts=self.test_num_pts)