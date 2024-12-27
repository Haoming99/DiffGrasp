from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import logging

from models.vaes import VAEEncoder, VAEDecoder
from models.decoder import ConvT3DDecoder, ConvT3DDecoder256, ConvT3DDecoder128
from models.fcresnet import FCBlock, FCResBlock
from models.embedder import get_embedder
from utils.eval_utils import compute_iou
from utils.viz import plot_pose_voxels
from utils.mesh import generate_from_latent
from argparse import ArgumentParser
import math
import matplotlib.pyplot as plt
from einops import repeat, reduce, rearrange
from torchvision.ops.focal_loss import sigmoid_focal_loss

class SKVOXVAE(pl.LightningModule):
    def __init__(self, enc_in_dim=24*3, enc_c_dim=15*3, enc_hidden_dim=256, latent_dim=256, c_dim=32, dec_type='convt32',
                 query_dim=3, dec_hidden_dim=256, dec_out_dim=24*3,lr=1e-4, kl_coeff=0.1, test_samples=4,
                 points_batch_size=100000, query_multires=-1, input_multires=-1,
                 c_hidden_size=128, ff_dim=256, trans_nl=3, nhead=8, attn_idx=0,
                 focal_alpha: float = 0.25, focal_gamma: float = 2, recon_loss_fn='focal',
                 num_init_layers=0, train_num_samples=1, val_num_samples=4, bin_thres=0.5, vox_res=32, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.test_samples = test_samples
        self.train_num_samples =  train_num_samples
        self.val_num_samples = val_num_samples
        self.points_batch_size = points_batch_size
        self.query_multires = query_multires
        self.input_multires = input_multires
        self.input_embeder, self.input_embeded_ch = get_embedder(input_multires, input_multires)

        enc_c_dim = enc_c_dim // 3 * self.input_embeded_ch
        enc_in_dim = enc_in_dim // 3 * self.input_embeded_ch

        module_list = [nn.Linear(enc_c_dim, c_dim, nn.ReLU(inplace=False))]
        for i in range(num_init_layers):
            module_list.append(FCResBlock(c_dim, c_dim))
        self.initial_layer = nn.Sequential(*module_list)

        self.encoder = VAEEncoder(enc_in_dim, enc_hidden_dim, latent_dim, c_dim)

        decoder_dict = {'convt32': ConvT3DDecoder, 'convt256': ConvT3DDecoder256, 'convt128': ConvT3DDecoder128}
        self.decoder = decoder_dict[dec_type](z_dim=latent_dim, c_dim=c_dim, D=vox_res, H=vox_res, W=vox_res)

        recon_loss_dict = {'focal': lambda x, y: sigmoid_focal_loss(x, y, alpha=focal_alpha, gamma=focal_gamma, reduction='mean'),
                'bcel': torch.nn.BCEWithLogitsLoss(reduction='mean')}
        self.recon_loss_fn = recon_loss_dict[recon_loss_fn]
        self.bin_thres = bin_thres

        if self.encoder.fc_layers:
            nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)
            nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)
        self.kwargs = kwargs


    def forward(self, c, num_samples=1):
        bs = c.shape[0]
        c = self.input_embeder(c)
        c = self.initial_layer(c)
        c = repeat(c, 'b c -> (b s) c', s=num_samples)

        z = torch.randn(c.shape[0], self.encoder.latent_dim, device=c.device, dtype=c.dtype)
        pred_occ_vox= self.decoder(z, c)
        pred_occ_vox = rearrange(pred_occ_vox, '(b s) d h w -> b s d h w', b=bs, s=num_samples)
        return pred_occ_vox

    def _run_step(self, x, c):
        bs = c.shape[0]
        c = self.input_embeder(c)
        x = self.input_embeder(x)

        c = self.initial_layer(c)
        mu, log_var = self.encoder(x, c)
        p, q, z = self.sample(mu, log_var)
        dec_out = self.decoder(z, c)
        return z, dec_out, p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx, num_samples=1):
        c, x = batch['partial_joints'], batch['full_joints']
        gt_occ_vox = batch['vox_occ']
        bs = x.shape[0]
        c = rearrange(c, 'b n1 t -> b (n1 t)')
        x = rearrange(x, 'b n2 t -> b (n2 t)')

        z, pred_occ_vox, p, q = self._run_step(x, c)

        recon_loss = self.recon_loss_fn(pred_occ_vox, gt_occ_vox)
        with torch.no_grad():
            iou_pts = compute_iou(pred_occ_vox, gt_occ_vox)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()


        kl = torch.distributions.kl_divergence(q, p).sum(-1)
        kl = kl.mean()

        loss = kl * self.kl_coeff + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
            'lr': self.lr,
            'iou': iou_pts,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, num_samples=self.train_num_samples)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, logs = self.step(batch, batch_idx, num_samples=self.train_num_samples)

        c = batch['partial_joints']
        gt_occ_vox = batch['vox_occ']

        bs = c.shape[0]
        c = rearrange(c, 'b n1 t -> b (n1 t)')
        pred_occ_vox = self.forward(c, num_samples=self.val_num_samples)
        with torch.no_grad():
            y = repeat(gt_occ_vox, 'b d h w -> b s (d h w)', s=self.val_num_samples)
            res_var = y.var(1).mean()
            logs['var'] = res_var

            x_hat = rearrange(pred_occ_vox, 'b s d h w -> b s (d h w)')
            c_recon_loss = self.recon_loss_fn(x_hat, y)
            logs['c_recon_loss'] = c_recon_loss

            x_hat = rearrange(x_hat, 'b s dhw -> (b s) dhw')
            y = rearrange(y, 'b s dhw -> (b s) dhw')
            c_iou_pts = compute_iou(x_hat, y).mean()
            logs['c_iou'] = c_iou_pts


        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        return loss

    def encode_input(self, c, x=None):
        """
        c: partial information
        x: complete information
        sample: sample from normal dist
        """
        c = self.input_embeder(c)
        c = self.initial_layer(c)

        if x is not None:
            x = self.input_embeder(x)
            mu, log_var = self.encoder(x, c)
            p, q, z = self.sample(mu, log_var)
        else:
            z = torch.randn(c.shape[0], self.encoder.latent_dim, device=c.device, dtype=c.dtype)
        return z, c

    def generate_viz(self, batch, batch_idx, max_bs=3, num_samples=4, sample=True):
        full_joints = batch['full_joints']
        partial_joints = batch['partial_joints']
        bs = partial_joints.shape[0]
        device = self.device

        partial_joints = repeat(partial_joints, 'b n t -> (b s) (n t)', s=num_samples).to(device)
        x = None if sample else repeat(full_joints, 'b n t -> (b s) (n t)', s=num_samples).to(device)

        with torch.no_grad():
            z, c = self.encode_input(partial_joints, x)
            pred_occ_vox= self.decoder(z, c)
            pred_occ_vox_bin = torch.sigmoid(pred_occ_vox) > self.bin_thres
            pred_occ_vox_bin = rearrange(pred_occ_vox_bin, '(b s) d h w -> b s d h w', b=bs, s=num_samples).cpu().numpy()

            img_list = list()
            for b_i in range(min(max_bs, bs)):
                cur_voxels = [batch['vox_occ'][b_i]] + [i for i in pred_occ_vox_bin[b_i]]
                print('Plotting voxels')
                cur_img = plot_pose_voxels(full_joints[b_i].cpu().numpy(), cur_voxels, return_img=True,
                                           figsize=(4*(2 + num_samples), 5))
                print('Done')
                img_list.append(cur_img)

        return img_list

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--enc_in_dim", type=int, default=24*3)
        parser.add_argument("--enc_c_dim", type=int, default=15*3)
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)
        parser.add_argument("--enc_hidden_dim", type=int, default=256)
        parser.add_argument("--c_dim", type=int, default=32)
        parser.add_argument("--dec_hidden_dim", type=int, default=256)
        parser.add_argument("--dec_out_dim", type=int, default=24*3)
        parser.add_argument("--num_init_layers", type=int, default=0)
        parser.add_argument("--recon_loss_fn", type=str, default="focal")
        parser.add_argument("--bin_thres", type=float, default=0.5)


        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dec_type", type=str, default="convt32")

        parser.add_argument("--smpl_path", type=str, default="/scratch/wen/smpl/")
        parser.add_argument("--test_samples", type=int, default=4)
        parser.add_argument("--val_trainset", action='store_true')
        parser.add_argument("--subset", type=int, default=0)
        parser.add_argument("--train_num_samples", type=int, default=1)
        parser.add_argument("--val_num_samples", type=int, default=4)
        parser.add_argument("--canon", action='store_true')
        parser.add_argument("--input_multires", type=int, default=-1, help='use of positional encoding')
        parser.add_argument("--c_hidden_size", type=int, default=128, help='hidden size before feeding into transformer')
        parser.add_argument("--ff_dim", type=int, default=256, help='dim of feedforward network in transformer')
        parser.add_argument("--trans_nl", type=int, default=3, help='number of layers for transformer')
        parser.add_argument("--nhead", type=int, default=8, help='number of heads for multihead attention')
        parser.add_argument("--attn_idx", type=int, default=0, help='output to pick from trnasformer, 0: z, 1: c')
        parser.add_argument("--mask_out", type=str, default='left', help='part to be masked out for partial inputs')
        parser.add_argument("--vox_res", type=int, default=32)
        return parser
