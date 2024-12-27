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
from utils.mesh import generate_from_latent, export_shapenet_samples
from models.encoder import ResnetPointnet
from argparse import ArgumentParser
import math
import matplotlib.pyplot as plt
from einops import repeat, reduce, rearrange
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from models.decoder import DecoderBase

@MODEL_REGISTRY
class SNPCSIMPVAE(pl.LightningModule):
    def __init__(self, decoder: DecoderBase, enc_in_dim=512, enc_c_dim=3, enc_hidden_dim=512, latent_dim=512, c_dim=512, c_hidden_size=128,
                 lr=1e-4, batch_size=16, kl_coeff=0.1, test_samples=4, points_batch_size=100000, query_multires=-1, input_multires=-1,
                 num_init_layers=0, train_num_samples=1, val_num_samples=4, test_num_samples=10,
                 padding=0.1, b_min=-0.5, b_max=0.5,
                 **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.train_num_samples =  train_num_samples
        self.val_num_samples = val_num_samples
        self.test_num_samples = test_num_samples
        self.points_batch_size = points_batch_size
        self.query_multires = query_multires
        self.input_multires = input_multires
        self.query_embeder, self.query_embeded_ch = get_embedder(query_multires, query_multires)
        self.input_embeder, self.input_embeded_ch = get_embedder(input_multires, input_multires)

        enc_c_dim = enc_c_dim // 3 * self.input_embeded_ch
        enc_in_dim = enc_in_dim #// 3 * self.input_embeded_ch # Do't need positional encoding because i'ts feature now

        self.partial_encoder = ResnetPointnet(c_dim=c_dim, dim=enc_c_dim, hidden_dim=enc_hidden_dim)
        self.full_encoder = ResnetPointnet(c_dim=enc_in_dim, dim=enc_c_dim, hidden_dim=enc_hidden_dim)

        self.encoder = VAEEncoder(enc_in_dim, enc_hidden_dim, latent_dim, c_dim)
        self.decoder = decoder

        if self.encoder.fc_layers:
            nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)
            nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)

        self.enc_c_dim = enc_c_dim
        self.out_dir = None
        self.batch_size = batch_size
        self.padding = padding
        self.b_min = b_min
        self.b_max = b_max


    def forward(self, c, pts, num_samples=1):
        bs = c.shape[0]
        c = self.input_embeder(c)
        c = self.partial_encoder(c)
        c = repeat(c, 'b c -> (b s) c', s=num_samples)

        pts = self.query_embeder(pts)
        pts = repeat(pts, 'b p c -> (b s) p c', s=num_samples)
        z = torch.randn(c.shape[0], self.encoder.latent_dim, device=c.device, dtype=c.dtype)

        decoded_occ = self.decoder(pts, z, c)
        decoded_occ = rearrange(decoded_occ, '(b s) p -> b s p', b=bs, s=num_samples)
        return decoded_occ

    def _run_step(self, x, c, pts):
        bs = c.shape[0]
        c = self.input_embeder(c)
        x = self.input_embeder(x)
        pts = self.query_embeder(pts)

        c = self.partial_encoder(c)
        x = self.full_encoder(x)

        mu, log_var = self.encoder(x, c)
        p, q, z = self.sample(mu, log_var)
        dec_out = self.decoder(pts, z, c)
        return z, dec_out, p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        c, x = batch['partial_pcs'], batch['full_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']
        bs = x.shape[0]

        z, pred_occ, p, q = self._run_step(x, c, query_pts)

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
        loss, logs = self.step(batch, batch_idx)
        #self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, logs = self.step(batch, batch_idx)

        c = batch['partial_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']
        bs = c.shape[0]

        pred_occ = self.forward(c, query_pts, num_samples=self.val_num_samples)
        with torch.no_grad():
            #TODO: Make the shape of occ appropriate
            res_var = pred_occ.var(1).mean()
            logs['var'] = res_var

            y = repeat(query_occ, 'b p -> b s p', b=bs, s=self.val_num_samples)
            c_recon_loss = F.binary_cross_entropy_with_logits(pred_occ, y, reduction='none').sum(-1).mean()
            logs['c_recon_loss'] = c_recon_loss

            pred_occ = rearrange(pred_occ, 'b s p -> (b s) p')
            y = rearrange(y, 'b s p -> (b s) p')
            c_iou_pts = compute_iou(pred_occ, y)
            c_iou_pts = torch.nan_to_num(c_iou_pts, 0).mean()
            logs['c_iou'] = c_iou_pts


        self.log_dict({f"val/{k}": v for k, v in logs.items()}, batch_size=self.batch_size)
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

    def generate_mesh(self, batch, batch_idx, max_bs=3, num_samples=4, sample=True, denormalize=False):
        partial_pcs = batch['partial_pcs']
        bs = partial_pcs.shape[0]
        if max_bs < 0:
            max_bs = bs
        device = self.device

        if denormalize:
            loc = batch['loc'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()

        partial_pcs = repeat(partial_pcs, 'b p d -> (b s) p d', s=num_samples).to(device)
        x = None if sample else repeat(batch['full_pcs'], 'b p d -> (b s) p d', s=num_samples).to(device)

        with torch.no_grad():
            z, c = self.encode_input(partial_pcs, x)
            z = rearrange(z, '(b s) z -> b s z', b=bs, s=num_samples)
            c = rearrange(c, '(b s) c -> b s c', b=bs, s=num_samples)

            mesh_list = list()
            input_list = list()
            for b_i in range(min(max_bs, bs)):
                cur_mesh_list = list()
                input_list.append(batch['partial_pcs'][b_i])

                state_dict = dict()
                for sample_idx in range(num_samples):
                    generated_mesh = generate_from_latent(self.eval_points, z[b_i, sample_idx, None, ], c[b_i, sample_idx, None],
                                                          state_dict, padding=self.padding, B_MAX=self.b_max, B_MIN=self.b_min)
                    cur_mesh_list.append(generated_mesh)
                    if denormalize:
                        generated_mesh.vertices = (generated_mesh.vertices + loc[b_i]) * scale[b_i]
                mesh_list.append(cur_mesh_list)

        return mesh_list, input_list

    def test_step(self, batch, batch_idx):
        mesh_list, _ = self.generate_mesh(batch, batch_idx, max_bs=-1, num_samples=self.test_num_samples, sample=True, denormalize=True)
        denormalized_pcs = (batch['partial_pcs'] + batch['loc']) * batch['scale']
        export_shapenet_samples(mesh_list, batch['category'], batch['model'], denormalized_pcs, self.out_dir)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    @staticmethod
    def add_model_specific_args(parent_parser):
        from warnings import warn
        warn('This args has been depreacted. Please use CLI instead', DeprecationWarning, stacklevel=2)

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

        parser.add_argument("--padding", type=float, default=0.1, help="Padding used on the generation of mesh")
        parser.add_argument("--query_multires", type=int, default=-1, help='use of positional encoding')
        parser.add_argument("--input_multires", type=int, default=-1, help='use of positional encoding')
        parser.add_argument("--c_hidden_size", type=int, default=128, help='hidden size before feeding into transformer')
        parser.add_argument("--ff_dim", type=int, default=256, help='dim of feedforward network in transformer')
        parser.add_argument("--trans_nl", type=int, default=3, help='number of layers for transformer')
        parser.add_argument("--nhead", type=int, default=8, help='number of heads for multihead attention')
        parser.add_argument("--attn_idx", type=int, default=0, help='output to pick from trnasformer, 0: z, 1: c')

        # Datasets
        parser.add_argument("--smpl_path", type=str, default="/scratch/wen/smpl/")
        parser.add_argument("--val_trainset", action='store_true')
        parser.add_argument("--subset", type=int, default=0)
        parser.add_argument("--train_num_samples", type=int, default=1)
        parser.add_argument("--test_num_samples", type=int, default=10) #K=10 for cgan and AutoSDF
        parser.add_argument("--val_num_samples", type=int, default=4)
        parser.add_argument("--b_max", type=float, default=0.5) # In fact the input would be as large as 0.55
        parser.add_argument("--b_min", type=float, default=-0.5)
        parser.add_argument("--dataset_folder", type=str, default='/Datasets/ShapeNet/')
        parser.add_argument("--partial_mode", type=str, default='bottom')
        parser.add_argument("--test_cat", type=str, default='chair')
        parser.add_argument("--split_type", type=str, default='onet')
        return parser
