from functools import partial
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
from models.vaes import VAEEncoder, GroupVAEEncoder, GroupFCEncoder
from models.encoder import ResnetPointnet
from models import LocalPoolPointnet
from models.unet3d import UNet3DCLI
from models.fcresnet import GroupLinear, GroupFCBlock, GroupFCResBlock, GroupDoubleFCLayer
from models.vdvae_modules import VDVAE_Encoder, VDVAE_Decoder
from models.uvae import VAEUNet


@MODEL_REGISTRY
class UVAEShare(KiloBase):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder: nn.Module, partial_local_pointnet: nn.Module, full_local_pointnet: nn.Module, unet3d: UNet3DCLI = None,
                 feat_dim=16, vox_reso=16, kl_weight=0.01, recon_weight=1., global_latent_dim=16, vdvae_width=64, reg_prob_weight=0., reg_prior_weight=0.,
                 plane_types=['xy', 'xz', 'yz'], local_cond_dim=32, unet_depth=5, unet_filts=32, block_type='BlockID', activation_type='gelu',
                 lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1, filter_nan=True,
                 batchnorm=True, dropout=False, reduction='sum', invis_loss_weight=1., interactive_debug=False, freeze_decoder=False,
                 test_num_pts=2048, ed_pre_path=None, dec_pre_path: Optional[str] = None, ende_lr=None):

        super().__init__(lr=lr, padding=padding, b_min=b_min, b_max=b_max, points_batch_size=points_batch_size, batch_size=batch_size, test_num_samples=test_num_samples,
                         batchnorm=batchnorm, reduction=reduction, invis_loss_weight=invis_loss_weight, interactive_debug=interactive_debug, freeze_decoder=freeze_decoder, test_num_pts=test_num_pts)

        self.decoder = decoder  # ConvONet decoder
        self.unet3d = unet3d  # 3D UNet after VAE sampling

        # For backward compatibility
        global_latent_dim = feat_dim if global_latent_dim is None else global_latent_dim

        self.partial_local_pointnet = partial_local_pointnet
        self.full_local_pointnet = full_local_pointnet

        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.invis_loss_weight = invis_loss_weight
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        self.global_latent_dim = global_latent_dim
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.ed_pre_path = ed_pre_path
        self.ende_lr = ende_lr
        self.plane_types = plane_types
        self.filter_nan = filter_nan
        self.local_cond_dim = local_cond_dim # necessary for configuration file
        self.pretrained_keys = []  # Empty until we configure the optimizer
        # Note: None and 0 are different. None means training from scratch and 0 means freeze the model
        self.reg_prob_weight = reg_prob_weight
        self.reg_prior_weight = reg_prior_weight

        self.vae_unet = VAEUNet(feat_dim, feat_dim, in_reso=vox_reso, block_type=block_type, activation_type=activation_type, depth=unet_depth, start_filts=unet_filts, z_dim=global_latent_dim)

        if dec_pre_path:
            ori_state_dict = torch.load(dec_pre_path)
            self.decoder.load_state_dict({k.lstrip(
                'decoder.'): v for k, v in ori_state_dict['model'].items() if k.startswith('decoder.')})

        if ed_pre_path:
            state_dict = torch.load(ed_pre_path)['state_dict']
            self.full_local_pointnet.load_state_dict({k[len('full_local_pointnet.'):]: v for k, v in state_dict.items(
            ) if k.startswith('full_local_pointnet.')}, strict=True)
            self.partial_local_pointnet.load_state_dict({k[len('full_local_pointnet.'):]: v for k, v in state_dict.items(
            ) if k.startswith('full_local_pointnet.')}, strict=False)
            self.decoder.load_state_dict({k[len('decoder.'):]: v for k, v in state_dict.items(
            ) if k.startswith('decoder.')}, strict=False)

    def decode(self, p, z, c=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z: dict or tensor for decdoer
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c_plane=z, c=c, **kwargs)
        return logits

    def step(self, batch, batch_idx):
        partial_pcs, full_pcs = batch['partial_pcs'], batch['full_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']

        x_local = self.full_local_pointnet(full_pcs)
        c_local = self.partial_local_pointnet(partial_pcs)

        kl_losses = {}
        reg_probs = {}
        reg_priors = {}
        pred_feats = {f"{k}_c": v for k, v in c_local.items()}

        for plane_type in self.plane_types:

            px_zc, stats, reg_prob_stats, reg_prior_stats = self.vae_unet.forward(x_local[plane_type], c_local[plane_type])
            cur_kl = sum([i.sum(dim=(1, 2, 3)) for i in stats]) # get a (b, ) tensor
            cur_kl = cur_kl.mean()

            cur_reg_prob = sum(i.sum(dim=(1, 2, 3)) for i in reg_prob_stats).mean()
            cur_reg_prior = sum(i.sum(dim=(1, 2, 3)) for i in reg_prior_stats).mean()

            pred_feats[plane_type] = px_zc # p(x|z,c)
            kl_losses[plane_type] = cur_kl
            reg_probs[plane_type] = cur_reg_prob
            reg_priors[plane_type] = cur_reg_prior

        pred_occ = self.decode(query_pts, z=pred_feats, c=None)

        recon_loss = F.binary_cross_entropy_with_logits(
            pred_occ, query_occ, reduction='none')

        if self.invis_loss_weight != 1.:  # we can compare with 1. because 1 = 2^0
            query_weight = batch['query_mask']
            query_weight[query_weight == 0] = self.invis_loss_weight
            recon_loss = recon_loss * query_weight

        if self.reduction == 'sum':
            recon_loss = recon_loss.sum(-1).mean()
        elif self.reduction == 'mean':
            recon_loss = recon_loss.mean()

        kl_loss = sum(v for v in kl_losses.values()) / len(kl_losses)
        reg_prob_loss = sum(v for v in reg_probs.values()) / len(reg_probs)
        reg_prior_loss = sum(v for v in reg_priors.values()) / len(reg_priors)

        loss = self.kl_weight * kl_loss + self.recon_weight * recon_loss  + self.reg_prob_weight * reg_prob_loss + self.reg_prior_weight * reg_prior_loss

        # Filter loss when NaN was introduced by recon_loss
        if self.filter_nan and torch.any(torch.isnan(loss)):
            console_logger = logging.getLogger("pytorch_lightning")
            console_logger.error(f"NaN loss encountered: loss: {loss}, kl_loss: {kl_loss}, recon_loss: {recon_loss}")
            loss = self.kl_weight * kl_loss

        if self.interactive_debug and (torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss))):
            import ipdb
            ipdb.set_trace()

        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()

        logs = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "recon_weight": self.recon_weight,
            "kl_weight": self.kl_weight,
            "reg_prior_loss": reg_prior_loss,
            "reg_prior_weight": self.reg_prior_weight,
            "reg_prob_loss": reg_prob_loss,
            "reg_prob_weight": self.reg_prob_weight,
            "loss": loss,
            'lr': self.lr,
            'iou': iou_pts,
            "batch_size": float(self.batch_size)
        }
        logs.update({f"kl-{k}": v.mean() for k, v in kl_losses.items()})

        if self.interactive_debug:
            print(f"loss: {loss.item()}, kl: {kl_loss.item()}")

        return loss, logs

    def encode_inputs(self, partial_pcs, full_pcs=None):
        bs = partial_pcs.shape[0]
        c_local = self.partial_local_pointnet(partial_pcs)
        pred_feats = {f"{k}_c": v for k, v in c_local.items()}

        if full_pcs is not None:
            x_local = self.full_local_pointnet(full_pcs)

            for plane_type in self.plane_types:
                px_zc = self.vae_unet.forward(x_local[plane_type], c_local[plane_type])[0] # The first return value is feat

                pred_feats[plane_type] = px_zc # p(x|z,c)

        else:
            for plane_type in self.plane_types:
                px_zc = self.vae_unet.forward_cond(c_local[plane_type]) # will be repeatede in get_inputs_cond

                pred_feats[plane_type] = px_zc # p(x|z,c)

        return pred_feats, torch.empty((bs, 0), device=partial_pcs.device)

    def get_nondecoder_params(self) -> list:
        return [self.partial_global_pointnet.parameters(), self.full_global_pointnet.parameters(),
                self.full_local_pointnet.parameters(), self.partial_local_pointnet.parameters(
        ), self.global_vae_encoder.parameters(),
            self.global_prior_encoder.parameters(), self.local_prior_encoders.parameters(
        ), self.local_vae_encoder.parameters(),
        ]

    def split_ende_params(self) -> tuple:
        pretrained_keys = list()
        pretrained_params = list()
        other_params = list()
        for k, v in dict(self.named_parameters()).items():
            if k.startswith('full_local_pointnet') or k.startswith('partial_local_pointnet') or (k.startswith('decoder.') and not k.startswith('decoder.transformer_encoder.')):
                pretrained_keys.append(k)
                pretrained_params.append(v)
            else:
                other_params.append(v)
        return pretrained_params, other_params

    def configure_optimizers(self):
        if self.ende_lr is not None:
            pretrained_params, other_params = self.split_ende_params()
            optimizer = torch.optim.Adam([{'params': pretrained_params, 'lr': self.ende_lr},
                                          {'params': other_params}], lr=self.lr)

        elif self.freeze_decoder:
            optimizer = torch.optim.Adam([{'params': itertools.chain(
                self.get_nondecoder_params()
            )},
                {'params': self.decoder.parameters(), 'lr': 0}], lr=self.lr)

        else:  # Default behavior
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer