import pytorch_lightning as pl
import torch
from torch import nn
from smplx import SMPL
import torch.nn.functional as F
from argparse import ArgumentParser
from argparse import ArgumentParser
from .encoder import ResnetPointnet, NaiveMLP
from .decoder import DecoderCBatchNorm, DecoderMLP

class CVAE(pl.LightningModule):
    def __init__(self, template_points=None, smpl_path=None, enc_in_dim=3, enc_out_dim=256, enc_hidden_dim=256, latent_dim=256, lr=1e-4, kl_coeff=0.1, dec_type='implicit', dec_args=dict(), enc_type='res_pointnet', test_samples=4, gpu_smpl=False, split_feats=False, no_c=False, train_num_samples=1, val_num_samples=4, **kwargs):
        super().__init__()
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.enc_in_dim = enc_in_dim
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.test_samples = test_samples
        self.validate_vis = 10
        self.gpu_smpl = gpu_smpl
        self.split_feats = split_feats
        self.train_num_samples =  train_num_samples
        self.val_num_samples = val_num_samples

        if self.latent_dim > 0:
            if self.split_feats:
                self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
                self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)
            else:
                self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
                self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.smpl = SMPL(model_path=smpl_path, create_global_orient=False, create_body_pose=False, create_betas=False, create_transl=False)
        self.register_buffer('template_points', template_points)

        if enc_type == 'naive_mlp':
            self.encoder = NaiveMLP(c_dim=enc_out_dim, dim=enc_in_dim, hidden_dim=enc_out_dim, num_pts=torch.sum(template_points[0, :, 0] > 0))
        else:
            self.encoder = ResnetPointnet(c_dim=enc_out_dim, dim=enc_in_dim, hidden_dim=enc_out_dim)
        decoders = {'implicit': DecoderCBatchNorm, 'regress': DecoderMLP}

        if self.split_feats:
            c_dim = enc_out_dim - self.latent_dim
        else:
            c_dim = enc_out_dim
        self.decoder = decoders[dec_type](z_dim=latent_dim, c_dim=c_dim)

    def forward(self, x):
        x = self.encoder(x)
        if self.latent_dim > 0:
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
        else:
            z = torch.empty((x.shape[0], 0), device=x.device)
        return self.decoder(self.tempalte_points, z, x)

    def _run_step(self, x, query_pts, num_samples=1):
        x = self.encoder(x)
        if self.latent_dim > 0:
            if self.split_feats:
                mu = self.fc_mu(x[:, :self.latent_dim])
                log_var = self.fc_var(x[:, :self.latent_dim])
                x = x[:, self.latent_dim:]
            else:
                mu = self.fc_mu(x)
                log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
        else:
            z = torch.empty((x.shape[0], 0), device=x.device)
            device = x.device
            p = torch.distributions.Normal(torch.zeros(x.shape[0], 1, device=device), torch.ones(x.shape[0], 1, device=device))
            q = torch.distributions.Normal(torch.zeros(x.shape[0], 1, device=device), torch.ones(x.shape[0], 1, device=device))

        # Multi-samples during validation or training
        bs = x.shape[0]
        z = q.rsample([num_samples]).reshape(-1, z.shape[-1])
        x = x.repeat_interleave(num_samples, dim=0)
        query_pts = query_pts.repeat_interleave(num_samples, dim=0)
        return z, self.decoder(query_pts, z, x), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx, num_samples=1):
        body_pose, betas =  batch['body_pose'], batch['betas']
        selected_idxs, query_idxs = batch['selected_idxs'], batch['query_idxs']
        if self.gpu_smpl:
            verts = self.smpl(global_orient=body_pose[:, :3], body_pose=body_pose[:, 3:], betas=betas).vertices
            x = verts[torch.arange(verts.shape[0]).repeat_interleave(selected_idxs.shape[1]), selected_idxs.reshape(-1)].reshape(
                    selected_idxs.shape[0], selected_idxs.shape[1], 3)
            query_pts = self.template_points[:, query_idxs.reshape(-1)].reshape(query_idxs.shape[0], query_idxs.shape[1], -1)
            y = (verts - self.template_points)[torch.arange(verts.shape[0]).repeat_interleave(query_idxs.shape[1]), query_idxs.reshape(-1)].reshape(
                    query_idxs.shape[0], query_idxs.shape[1], 3)
        else:
            x, y = batch['pts'], batch['targets']
            query_pts = batch['query_pts']
        bs = x.shape[0]

        z, x_hat, p, q = self._run_step(x, query_pts, num_samples=num_samples)
        y = y.repeat_interleave(num_samples, dim=0)

        recon_loss = F.mse_loss(x_hat, y, reduction="none").reshape(bs, num_samples, -1).sum().mean()

        with torch.no_grad():
            multi_selected_idxs = selected_idxs.repeat_interleave(num_samples, dim=0)
            msk = torch.zeros(x_hat.shape[0], x_hat.shape[1]).to(selected_idxs.device)
            msk.scatter_(1, multi_selected_idxs, 1)
            vis_loss = (torch.square(x_hat - y) * msk.unsqueeze(-1)).sum() / (msk.sum() * x_hat.shape[-1]) # average over all elements
            res = x_hat.reshape(bs, num_samples, -1)
            res_var = res.var(1).sum(-1).mean()
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        #kl *= self.kl_coeff

        loss = kl * self.kl_coeff + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
            'lr': self.lr,
            'vis_loss': vis_loss,
            'var': res_var,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, num_samples=self.train_num_samples)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, num_samples=self.val_num_samples)
        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['pts'], batch['targets']

        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        z = q.rsample([self.test_samples]).reshape(-1, z.shape[-1])
        x = x.repeat_interleave(self.test_samples, dim=0)
        query_pts = batch['query_pts'].repeat_interleave(self.test_samples, dim=0)

        x_hat = self.decoder(query_pts, z, x)
        y = y.repeat_interleave(self.test_samples, dim=0)

        mpvpe = torch.sqrt((x_hat - y).square().sum(dim=-1)).mean(-1).reshape(-1, self.test_samples)
        recon_loss = F.mse_loss(x_hat, y, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        #kl *= self.kl_coeff

        loss = kl * self.kl_coeff + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
            "mpvpe_mean": mpvpe.mean(),
            "mpvpe_min": mpvpe.min(1)[0].mean(),
            "mpvpe_max": mpvpe.max(1)[0].mean(),
        }
        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

#         parser.add_argument("--enc_type", type=str, default="resnet18", help="resnet18/resnet50")
#         parser.add_argument("--first_conv", action="store_true")
#         parser.add_argument("--maxpool1", action="store_true")
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--enc_out_dim", type=int, default=256)
        parser.add_argument("--enc_in_dim", type=int, default=3)
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)
        parser.add_argument("--enc_hidden_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=4)

        parser.add_argument("--smpl_path", type=str, default="/scratch/wen/smpl/")
        parser.add_argument("--sample_bound", type=int, default=100)
        parser.add_argument("--sample_zones", type=int, default=3)
        parser.add_argument("--test_samples", type=int, default=4)
        parser.add_argument("--query_percent", type=float, default=1.)
        parser.add_argument("--dec_type", type=str, default='implicit')
        parser.add_argument("--enc_type", type=str, default='res_pointnet')
        parser.add_argument("--val_trainset", action='store_true')
        parser.add_argument("--gpu_smpl", action='store_true')
        parser.add_argument("--halfspace", action='store_true')
        parser.add_argument("--est_res", action='store_true')
        parser.add_argument("--subset", type=int, default=0)
        parser.add_argument("--split_feats", action='store_true')
        parser.add_argument("--train_num_samples", type=int, default=1)
        parser.add_argument("--val_num_samples", type=int, default=4)
        return parser
