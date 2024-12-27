"""
This file contains definitions of layers used as building blocks in SMPLParamRegressor
"""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from models.vaes import VAEEncoder, VAEDecoder
from models.fcresnet import FCBlock, FCResBlock
from argparse import ArgumentParser
import math

class SKVAE(pl.LightningModule):
    def __init__(self, enc_in_dim=24*3, enc_c_dim=15*3, enc_hidden_dim=256, latent_dim=256, c_dim=32, dec_hidden_dim=256, dec_out_dim=24*3, lr=1e-4, kl_coeff=0.1, test_samples=4, num_init_layers=0,
                 train_num_samples=1, val_num_samples=4, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_in_dim = enc_in_dim
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.test_samples = test_samples
        self.train_num_samples =  train_num_samples
        self.val_num_samples = val_num_samples

        module_list = [nn.Linear(enc_c_dim, c_dim, nn.ReLU(inplace=False))]
        for i in range(num_init_layers):
            module_list.append(FCResBlock(c_dim, c_dim))
        self.initial_layer = nn.Sequential(*module_list)

        self.encoder = VAEEncoder(enc_in_dim, enc_hidden_dim, latent_dim, c_dim)
        self.decoder = VAEDecoder(latent_dim, dec_hidden_dim, dec_out_dim, c_dim)
        nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)
        self.recon_loss_fn = F.l1_loss
        self.kwargs = kwargs


    def forward(self, c, num_samples=1):
        bs = c.shape[0]
        c = self.initial_layer(c)
        c = c.repeat_interleave(num_samples, dim=0)
        z = torch.randn(c.shape[0], self.encoder.latent_dim, device=c.device, dtype=c.dtype)
        return self.decoder(z, c).reshape(bs, num_samples, -1)

    def _run_step(self, x, c, num_samples=1):
        bs = c.shape[0]
        c = self.initial_layer(c)
        mu, log_var = self.encoder(x, c)
        p, q, z = self.sample(mu, log_var)
        dec_out = self.decoder(z, c)
        dec_out = dec_out.repeat_interleave(num_samples, dim=0)
        return z, self.decoder(z, c), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx, num_samples=1):
        c, x = batch['partial_joints'], batch['full_joints']
        bs = x.shape[0]
        c = c.reshape(bs, -1)
        x = x.reshape(bs, -1)
        y = x.clone()

        z, x_hat, p, q = self._run_step(x, c, num_samples=num_samples)
        y = y.repeat_interleave(num_samples, dim=0)
        x_hat = x_hat.reshape(*y.shape)

        recon_loss = self.recon_loss_fn(x_hat, y, reduction="mean")


        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()

        loss = kl * self.kl_coeff + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
            'lr': self.lr,
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
        bs = c.shape[0]
        c = c.reshape(bs, -1)
        x_hat = self.forward(c, num_samples=self.val_num_samples)
        with torch.no_grad():
            y = batch['full_joints'].reshape(bs, -1)
            y = y.repeat_interleave(self.val_num_samples, dim=0).reshape(bs, self.val_num_samples, -1)
            res_var = x_hat.var(1).mean()
            logs['var'] = res_var
            c_recon_loss = self.recon_loss_fn(x_hat, y, reduction="mean")
            logs['c_recon_loss'] = c_recon_loss

            recon_joins = x_hat.reshape(bs, self.val_num_samples, -1, 3)
            full_joints = y.reshape(bs, self.val_num_samples, -1, 3)
            joints_mask = batch['joints_mask']
            total_mask = joints_mask[:, None, :, None].repeat(1, self.val_num_samples, 1, 3)
            vis_loss = torch.abs((full_joints - recon_joins) * total_mask).sum() / total_mask.sum()
            logs['vis_loss'] = vis_loss

        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        return loss

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


        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=4)

        parser.add_argument("--smpl_path", type=str, default="/scratch/wen/smpl/")
        parser.add_argument("--test_samples", type=int, default=4)
        parser.add_argument("--val_trainset", action='store_true')
        parser.add_argument("--subset", type=int, default=0)
        parser.add_argument("--train_num_samples", type=int, default=1)
        parser.add_argument("--val_num_samples", type=int, default=4)
        return parser
