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


@MODEL_REGISTRY
class CondVDVAE3X(KiloBase):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder: nn.Module, local_pointnet: nn.Module, unet3d: UNet3DCLI = None,
                 feat_dim=16, vox_reso=16, kl_weight=0.01, recon_weight=1., Rc=10, latent_dim=24, vdvae_width=64,
                 plane_types=['xy', 'xz', 'yz'], local_cond_dim=32, attn_grid=False,
                 enc_blocks_str='', dec_blocks_str='', custom_width_str='', bottleneck_multiple=0.25, no_bias_above=32,
                 lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1,
                 batchnorm=True, dropout=False, reduction='sum', invis_loss_weight=0., interactive_debug=False, freeze_decoder=False,
                 test_num_pts=2048, ed_pre_path=None, dec_pre_path: Optional[str] = None, ende_lr=None):

        super().__init__(lr=lr, padding=padding, b_min=b_min, b_max=b_max, points_batch_size=points_batch_size, batch_size=batch_size, test_num_samples=test_num_samples,
                         batchnorm=batchnorm, reduction=reduction, invis_loss_weight=invis_loss_weight, interactive_debug=interactive_debug, freeze_decoder=freeze_decoder, test_num_pts=test_num_pts)

        self.decoder = decoder  # ConvONet decoder
        self.unet3d = unet3d  # 3D UNet after VAE sampling

        self.local_pointnet = local_pointnet

        self.save_hyperparameters(ignore=["decoder", "unet3d", "local_pointnet", ])

        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.invis_loss_weight = invis_loss_weight
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        self.Rc = Rc
        self.latent_dim = latent_dim
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.ed_pre_path = ed_pre_path
        self.ende_lr = ende_lr
        self.plane_types = plane_types
        self.attn_grid = attn_grid
        self.local_cond_dim = local_cond_dim # necessary for configuration file
        self.pretrained_keys = []  # Empty until we configure the optimizer
        # Note: None and 0 are different. None means training from scratch and 0 means freeze the model

        self.encoder = VDVAE_Encoder(feat_dim * len(plane_types), vdvae_width, enc_blocks_str, bottleneck_multiple=bottleneck_multiple, custom_width_str=custom_width_str)
        self.part_encoder = VDVAE_Encoder(feat_dim * len(plane_types), vdvae_width, enc_blocks_str, bottleneck_multiple=bottleneck_multiple, custom_width_str=custom_width_str)
        self.vdvae_decoder = VDVAE_Decoder(vdvae_width, dec_blocks_str, latent_dim * len(plane_types), bottleneck_multiple=bottleneck_multiple, no_bias_above=no_bias_above)

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

        x_local = self.local_pointnet(full_pcs)
        stacked_x_local = rearrange([x_local[k] for k in self.plane_types], 'p b f v1 v2 -> b (p f) v1 v2')

        c_local = self.local_pointnet(partial_pcs)
        stacked_c_local = rearrange([c_local[k] for k in self.plane_types], 'p b f v1 v2 -> b (p f) v1 v2') # NOTE: we use (p f) instead of (f p) here

        pred_feats = {f"{k}_c": v for k, v in c_local.items()}

        full_activations = self.encoder.forward(stacked_x_local)
        part_activations = self.part_encoder.forward(stacked_c_local)
        px_zc, stats = self.vdvae_decoder.forward(
            full_activations, part_activations)
        ndims = np.prod(c_local['xz'].shape[1:])
        cur_kl = sum([i['kl'].sum(dim=(1, 2, 3)) for i in stats]) # get a (b, ) tensor
        kl_loss = (cur_kl / ndims).mean()

        splited_preds = rearrange(px_zc, 'b (p f) v1 v2 -> p b f v1 v2', f=self.feat_dim, p=len(self.plane_types))
        if self.attn_grid:
            pred_feats.update({k: v for k, v in zip(self.plane_types, splited_preds)})
        else:
            # concate conditional features instead
            pred_feats.update({k: torch.cat([v, pred_feats[f"{k}_c"]], dim=1) for k, v in zip(self.plane_types, splited_preds)})

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

        loss = self.kl_weight * kl_loss + self.recon_weight * recon_loss 

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
            "loss": loss,
            'lr': self.lr,
            'iou': iou_pts,
            "batch_size": float(self.batch_size)
        }

        if self.interactive_debug:
            print(f"loss: {loss.item()}, kl: {kl_loss.item()}")

        return loss, logs

    def encode_inputs(self, partial_pcs, full_pcs=None):
        bs = partial_pcs.shape[0]
        c_local = self.local_pointnet(partial_pcs)
        stacked_c_local = rearrange([c_local[k] for k in self.plane_types], 'p b f v1 v2 -> b (p f) v1 v2')
        pred_feats = {f"{k}_c": v for k, v in c_local.items()}

        if full_pcs is not None:
            x_local = self.local_pointnet(full_pcs)
            stacked_x_local = rearrange([x_local[k] for k in self.plane_types], 'p b f v1 v2 -> b (p f) v1 v2')

            full_activations = self.encoder.forward(stacked_x_local)
            part_activations = self.part_encoder.forward(stacked_c_local)
            px_zc, _ = self.vdvae_decoder.forward(full_activations, part_activations)
        else:
            part_activations = self.part_encoder.forward(stacked_c_local)
            px_zc = self.vdvae_decoder.forward_cond(part_activations, n=1) # will be repeatede in get_inputs_cond

        splited_preds = rearrange(px_zc, 'b (p f) v1 v2 -> p b f v1 v2', f=self.feat_dim, p=len(self.plane_types))
        if self.attn_grid:
            pred_feats.update({k: v for k, v in zip(self.plane_types, splited_preds)})
        else:
            # concate conditional features instead
            pred_feats.update({k: torch.cat([v, pred_feats[f"{k}_c"]], dim=1) for k, v in zip(self.plane_types, splited_preds)})

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
