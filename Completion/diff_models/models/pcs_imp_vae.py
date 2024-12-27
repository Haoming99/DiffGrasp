from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import logging

from models.vaes import VAEEncoder, VAEDecoder
from models.decoder import DecoderCBatchNormOcc, DecoderCBatchNormCatZ, DecoderTransformerCBatchNormOcc
from models.fcresnet import FCBlock, FCResBlock
from models.embedder import get_embedder
from utils.eval_utils import compute_iou
from utils.viz import plot_smpl
from utils.mesh import generate_from_latent
from models.encoder import ResnetPointnet
from argparse import ArgumentParser
import math
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

@MODEL_REGISTRY
class PCSIMPVAE(pl.LightningModule):
    def __init__(self, enc_in_dim=512, enc_c_dim=3, enc_hidden_dim=512, latent_dim=512, c_dim=512, dec_type='cbn',
                 query_dim=3, dec_hidden_dim=512, lr=1e-4, kl_coeff=0.1, test_samples=4,
                 points_batch_size=100000, query_multires=-1, input_multires=-1,
                 c_hidden_size=128, ff_dim=256, trans_nl=3, nhead=8, attn_idx=0,
                 num_init_layers=0, train_num_samples=1, val_num_samples=4, **kwargs):
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
        self.query_embeder, self.query_embeded_ch = get_embedder(query_multires, query_multires)
        self.input_embeder, self.input_embeded_ch = get_embedder(input_multires, input_multires)

        enc_c_dim = enc_c_dim // 3 * self.input_embeded_ch
        enc_in_dim = enc_in_dim #// 3 * self.input_embeded_ch # Do't need positional encoding because i'ts feature now
        query_dim = query_dim // 3 * self.query_embeded_ch

        self.partial_encoder = ResnetPointnet(c_dim=c_dim, dim=enc_c_dim, hidden_dim=enc_hidden_dim)
        self.full_encoder = ResnetPointnet(c_dim=enc_in_dim, dim=enc_c_dim, hidden_dim=enc_hidden_dim)

        self.encoder = VAEEncoder(enc_in_dim, enc_hidden_dim, latent_dim, c_dim)
        dec_dict = {'cbn': DecoderCBatchNormOcc, 'cbn_cat': DecoderCBatchNormCatZ}

        if dec_type in dec_dict:
            self.decoder = dec_dict[dec_type](dim=query_dim, z_dim=latent_dim, c_dim=c_dim, hidden_size=dec_hidden_dim)
        elif dec_type == 'trans_cbn':
            self.decoder = DecoderTransformerCBatchNormOcc(dim=query_dim, z_dim=latent_dim, c_dim=c_dim,
                    hidden_size=dec_hidden_dim, c_hidden_size=c_hidden_size, ff_dim=ff_dim, trans_nl=trans_nl, nhead=nhead, attn_idx=attn_idx)
        else:
            raise NotImplementedError(f"Unsupported dec_type: {dec_type}")

        if self.encoder.fc_layers:
            nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)
            nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)

        self.enc_c_dim = enc_c_dim
        self.kwargs = kwargs


    def forward(self, c, pts, num_samples=1):
        bs = c.shape[0]
        c = self.input_embeder(c)
        c = self.partial_encoder(c)
        c = c.repeat_interleave(num_samples, dim=0)

        pts = self.query_embeder(pts)
        pts = pts.repeat_interleave(num_samples, dim=0)
        z = torch.randn(c.shape[0], self.encoder.latent_dim, device=c.device, dtype=c.dtype)
        return self.decoder(pts, z, c).reshape(bs, num_samples, -1)

    def _run_step(self, x, c, pts, num_samples=1):
        bs = c.shape[0]
        c = self.input_embeder(c)
        x = self.input_embeder(x)
        pts = self.query_embeder(pts)

        c = self.partial_encoder(c)
        x = self.full_encoder(x)

        mu, log_var = self.encoder(x, c)
        p, q, z = self.sample(mu, log_var)
        dec_out = self.decoder(pts, z, c)
        dec_out = dec_out.repeat_interleave(num_samples, dim=0)
        # TODO: doesn't really support multiple samples for now
        return z, dec_out, p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx, num_samples=1):
        c, x = batch['partial_pcs'], batch['full_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']
        bs = x.shape[0]

        z, pred_occ, p, q = self._run_step(x, c, query_pts, num_samples=num_samples)

        recon_loss = F.binary_cross_entropy_with_logits(
            pred_occ, query_occ, reduction='none').sum(-1).mean()
        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
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

        c = batch['partial_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']
        bs = c.shape[0]

        pred_occ = self.forward(c, query_pts, num_samples=self.val_num_samples)
        with torch.no_grad():
            #TODO: Make the shape of occ appropriate
            y = query_occ.repeat_interleave(self.val_num_samples, dim=0).reshape(bs, self.val_num_samples, -1)
            res_var = pred_occ.var(1).mean()
            logs['var'] = res_var
            c_recon_loss = F.binary_cross_entropy_with_logits(pred_occ, y, reduction='none').sum(-1).mean()
            logs['c_recon_loss'] = c_recon_loss

            total_sampels = bs * self.val_num_samples
            c_iou_pts = compute_iou(pred_occ.reshape(total_sampels, -1), y.reshape(total_sampels, -1)).mean()
            logs['c_iou'] = c_iou_pts


        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        return loss

    def eval_points(self, p, z, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p = self.query_embeder(p)
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.decoder(pi, z, c)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def encode_input(self, c, x=None):
        """
        c: partial information
        x: complete information
        sample: sample from normal dist
        """
        c = self.input_embeder(c)
        c = self.partial_encoder(c)

        if x is not None:
            x = self.input_embeder(x)
            x = self.full_encoder(x)
            mu, log_var = self.encoder(x, c)
            p, q, z = self.sample(mu, log_var)
        else:
            z = torch.randn(c.shape[0], self.encoder.latent_dim, device=c.device, dtype=c.dtype)
        return z, c

    def generate_mesh(self, batch, batch_idx, max_bs=3, num_samples=4, sample=True):
        full_pcs = batch['full_pcs']
        partial_pcs = batch['partial_pcs']
        bs = partial_pcs.shape[0]
        device = self.device

        partial_pcs = partial_pcs.repeat_interleave(num_samples, dim=0).reshape(bs * num_samples, -1, self.enc_c_dim).to(device)
        x = None if sample else batch['full_pcs'].repeat_interleave(num_samples, dim=0).reshape(bs * num_samples, -1, self.enc_c_dim).to(device)

        with torch.no_grad():
            z, c = self.encode_input(partial_pcs, x)
            z = z.reshape(bs, num_samples, -1)
            c = c.reshape(bs, num_samples, -1)

            mesh_list = list()
            input_list = list()
            for b_i in range(min(max_bs, bs)):
                cur_mesh_list = list()
                input_list.append(batch['full_pcs'][b_i])

                state_dict = dict()
                for sample_idx in range(num_samples):
                    generated_mesh = generate_from_latent(self, z[b_i, sample_idx, None, ], c[b_i, sample_idx, None],
                                                          state_dict, B_MAX=self.kwargs['b_max'], B_MIN=self.kwargs['b_min'])
                    cur_mesh_list.append(generated_mesh)
                mesh_list.append(cur_mesh_list)

        return mesh_list, input_list

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--enc_in_dim", type=int, default=512)
        parser.add_argument("--enc_c_dim", type=int, default=3)
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=512)
        parser.add_argument("--enc_hidden_dim", type=int, default=512)
        parser.add_argument("--c_dim", type=int, default=512)
        parser.add_argument("--dec_hidden_dim", type=int, default=512)
        parser.add_argument("--num_init_layers", type=int, default=0)


        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dec_type", type=str, default="cbn")

        parser.add_argument("--smpl_path", type=str, default="/scratch/wen/smpl/")
        parser.add_argument("--test_samples", type=int, default=4)
        parser.add_argument("--val_trainset", action='store_true')
        parser.add_argument("--subset", type=int, default=0)
        parser.add_argument("--train_num_samples", type=int, default=1)
        parser.add_argument("--val_num_samples", type=int, default=4)
        parser.add_argument("--b_max", type=float, default=1.7)
        parser.add_argument("--b_min", type=float, default=-1.7)
        parser.add_argument("--points_uniform_ratio", type=float, default=1.0)
        parser.add_argument("--canon", action='store_true')
        parser.add_argument("--query_multires", type=int, default=-1, help='use of positional encoding')
        parser.add_argument("--input_multires", type=int, default=-1, help='use of positional encoding')
        parser.add_argument("--c_hidden_size", type=int, default=128, help='hidden size before feeding into transformer')
        parser.add_argument("--ff_dim", type=int, default=256, help='dim of feedforward network in transformer')
        parser.add_argument("--trans_nl", type=int, default=3, help='number of layers for transformer')
        parser.add_argument("--nhead", type=int, default=8, help='number of heads for multihead attention')
        parser.add_argument("--attn_idx", type=int, default=0, help='output to pick from trnasformer, 0: z, 1: c')
        return parser
