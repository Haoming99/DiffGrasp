from optparse import Option
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import pytorch_lightning as pl
from typing import Union, Optional
import itertools
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

@MODEL_REGISTRY
class KiloLoF(KiloBase):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder: nn.Module, local_vae_encoders: nn.Module, local_prior_encoders: nn.Module,
        partial_global_pointnet: nn.Module, partial_local_pointnet: nn.Module, full_global_pointnet: nn.Module, full_local_pointnet: nn.Module,
        global_vae_encoder: nn.Module, global_prior_encoder: nn.Module,
        unet3d: UNet3DCLI=None, feat_dim=16, vox_reso=16, kl_weight=0.01,
        local_kl_weight=0.01, recon_weight=1., Rc=10, global_latent_dim=16, local_cond_dim=16,
        dec_pre_path: Optional[str]=None, dropout=False,
        lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1,
        batchnorm=True, reduction='sum', invis_loss_weight=1., interactive_debug=False, freeze_decoder=False):
        
        super().__init__(lr=lr, padding=padding, b_min=b_min, b_max=b_max, points_batch_size=points_batch_size, batch_size=batch_size, test_num_samples=test_num_samples,
                batchnorm=batchnorm, reduction=reduction, invis_loss_weight=invis_loss_weight, interactive_debug=interactive_debug, freeze_decoder=freeze_decoder)

        self.decoder = decoder # ConvONet decoder
        self.unet3d = unet3d #3D UNet after VAE sampling

        # For backward compatibility
        global_latent_dim = feat_dim if global_latent_dim is None else global_latent_dim

        self.partial_global_pointnet = partial_global_pointnet
        self.partial_local_pointnet = partial_local_pointnet
        self.full_global_pointnet = full_global_pointnet
        self.full_local_pointnet = full_local_pointnet

        # Init the global vae and local vaes, consider use index
        self.global_vae_encoder = global_vae_encoder
        self.global_prior_encoder = global_prior_encoder
        
        # Local VAEs, naive implementation
        # self.local_vae_encoders = GroupVAEEncoder(feat_dim, feat_dim*2, feat_dim, groups=Rc*3*feat_dim, conditioning_channels=global_latent_dim+local_cond_dim, use_conditioning=True, batchnorm=batchnorm)
        # self.local_prior_encoders = GroupVAEEncoder(global_latent_dim, feat_dim*2, feat_dim, groups=Rc*3*feat_dim, conditioning_channels=local_cond_dim, use_conditioning=False, batchnorm=batchnorm)
        self.local_vae_encoders = local_vae_encoders
        self.local_prior_encoders = local_prior_encoders

        self.kl_weight = kl_weight
        self.local_kl_weight = local_kl_weight
        self.recon_weight = recon_weight
        self.invis_loss_weight = invis_loss_weight 
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        self.Rc = Rc

        if dec_pre_path:
            ori_state_dict = torch.load(dec_pre_path)
            self.decoder.load_state_dict({k.lstrip('decoder.'): v for k, v in ori_state_dict['model'].items() if k.startswith('decoder.')})
        
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
        x_global = self.full_global_pointnet(full_pcs)

        c_global = self.partial_global_pointnet(partial_pcs)
        c_local = self.partial_local_pointnet(partial_pcs)

        grid_feats = x_local['grid']
        h_feats = reduce(grid_feats, 'b c h w d -> b c h', 'mean')
        w_feats = reduce(grid_feats, 'b c h w d -> b c w', 'mean')
        d_feats = reduce(grid_feats, 'b c h w d -> b c d', 'mean')
        
        stacked_feats = rearrange([h_feats, w_feats, d_feats], 's b f v -> s b f v')
        stacked_feats = repeat(stacked_feats, 's b f v -> s rc b f v', rc=self.Rc)
        flatten_x_local = rearrange(stacked_feats, 's rc b f v -> b f (s rc v)')

        grid_c = c_local['grid']
        h_c = reduce(grid_c, 'b c h w d -> b c h', 'mean')
        w_c = reduce(grid_c, 'b c h w d -> b c w', 'mean')
        d_c = reduce(grid_c, 'b c h w d -> b c d', 'mean')
        
        stacked_c = rearrange([h_c, w_c, d_c], 's b f v -> s b f v')
        stacked_c = repeat(stacked_c, 's b f v -> s rc b f v', rc=self.Rc)
        flatten_c_local = rearrange(stacked_c, 's rc b f v -> b f (s rc v)')

        mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        prior_mu, prior_log_var = self.global_prior_encoder(x=c_global, c=None)
        prior_std = torch.exp(prior_log_var / 2)
        p = torch.distributions.Normal(prior_mu, prior_std)

        repeated_z = repeat(z, 'b c -> b c (s rc v)', s=3, rc=self.Rc, v=self.vox_reso)
        combined_local_cond = torch.cat([repeated_z, flatten_c_local], dim=1)
        local_mus, local_log_vars = self.local_vae_encoders(x=flatten_x_local, c=combined_local_cond)
        local_stds = torch.exp(local_log_vars / 2)
        local_qs = torch.distributions.Normal(local_mus, local_stds)
        
        local_zs = local_qs.rsample()

        local_prior_mus, local_prior_log_vars = self.local_prior_encoders(x=repeated_z, c=flatten_c_local)
        local_prior_stds = torch.exp(local_prior_log_vars / 2)
        local_ps = torch.distributions.Normal(local_prior_mus, local_prior_stds)

        local_kl_losses = torch.distributions.kl_divergence(local_qs, local_ps) 
                      
        fvx, fvy, fvz = rearrange(local_zs, 'b f (s rc v) -> s b rc v f', s=3, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        sampled_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)

        if self.unet3d is not None:
            sampled_feat_grid = self.unet3d(sampled_feat_grid)

        pred_occ = self.decode(query_pts, z={'grid': sampled_feat_grid}, c=c_global) 

        
        recon_loss = F.binary_cross_entropy_with_logits(
                pred_occ, query_occ, reduction='none')

        if self.invis_loss_weight != 1.: # we can compare with 1. because 1 = 2^0
            query_weight = batch['query_mask']
            query_weight[query_weight == 0] = self.invis_loss_weight
            recon_loss = recon_loss * query_weight

        if self.reduction == 'sum':
            recon_loss = recon_loss.sum(-1).mean()
        elif self.reduction == 'mean':
            recon_loss = recon_loss.mean()
        
        kl_loss = torch.distributions.kl_divergence(q, p) 

        if self.reduction == 'sum':
            kl_loss = kl_loss.sum(-1).mean()
            local_kl_loss = local_kl_losses.sum(dim=[-1, -2]).mean()
        elif self.reduction == 'mean':
            kl_loss = kl_loss.mean()
            local_kl_loss = local_kl_losses.mean()


        loss = self.kl_weight * kl_loss + self.recon_weight * recon_loss + self.local_kl_weight * local_kl_loss

        if self.interactive_debug and (torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss))):
            import ipdb; ipdb.set_trace()

        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()

        logs = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "local_kl_loss": local_kl_loss,
            "recon_weight": self.recon_weight,
            "kl_weight": self.kl_weight,
            "local_kl_weight": self.local_kl_weight,
            "loss": loss,
            'lr': self.lr,
            'iou': iou_pts,
        }

        if self.interactive_debug:
            print(f"loss: {loss.item()}, kl: {kl_loss.item()}")

        return loss, logs

    def encode_inputs(self, partial_pcs, full_pcs=None):
        c_global = self.partial_global_pointnet(partial_pcs)
        c_local = self.partial_local_pointnet(partial_pcs)

        grid_c = c_local['grid']
        h_c = reduce(grid_c, 'b c h w d -> b c h', 'mean')
        w_c = reduce(grid_c, 'b c h w d -> b c w', 'mean')
        d_c = reduce(grid_c, 'b c h w d -> b c d', 'mean')
        
        stacked_c = rearrange([h_c, w_c, d_c], 's b f v -> s b f v')
        stacked_c = repeat(stacked_c, 's b f v -> s rc b f v', rc=self.Rc)
        flatten_c_local = rearrange(stacked_c, 's rc b f v -> b f (s rc v)')
        
        if full_pcs is not None:
            x_local = self.full_local_pointnet(full_pcs)
            x_global = self.full_global_pointnet(full_pcs)

            mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # Process local latents
            grid_feats = x_local['grid']
            h_feats = reduce(grid_feats, 'b c h w d -> b c h', 'mean')
            w_feats = reduce(grid_feats, 'b c h w d -> b c w', 'mean')
            d_feats = reduce(grid_feats, 'b c h w d -> b c d', 'mean')

            stacked_feats = rearrange([h_feats, w_feats, d_feats], 's b f v -> s b f v')
            stacked_feats = repeat(stacked_feats, 's b f v -> s rc b f v', rc=self.Rc)
            flatten_x_local = rearrange(stacked_feats, 's rc b f v -> b f (s rc v)')
            
            repeated_z = repeat(z, 'b c -> b c (s rc v)', s=3, rc=self.Rc, v=self.vox_reso)
            combined_local_cond = torch.cat([repeated_z, flatten_c_local], dim=1)
            local_mus, local_log_vars = self.local_vae_encoders(x=flatten_x_local, c=combined_local_cond)
            local_stds = torch.exp(local_log_vars / 2)
            local_qs = torch.distributions.Normal(local_mus, local_stds)
            
            local_zs = local_qs.rsample()
        else:
            prior_mu, prior_log_var = self.global_prior_encoder(x=c_global, c=None)
            prior_std = torch.exp(prior_log_var / 2)
            p = torch.distributions.Normal(prior_mu, prior_std)
            z = p.rsample()

            # process local latents
            repeated_z = repeat(z, 'b c -> b c (s rc v)', s=3, rc=self.Rc, v=self.vox_reso)
            local_prior_mus, local_prior_log_vars = self.local_prior_encoders(x=repeated_z, c=flatten_c_local)
            local_prior_stds = torch.exp(local_prior_log_vars / 2)
            local_ps = torch.distributions.Normal(local_prior_mus, local_prior_stds)
            local_zs = local_ps.rsample()
                      
        fvx, fvy, fvz = rearrange(local_zs, 'b f (s rc v) -> s b rc v f', s=3, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        sampled_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)

        if self.unet3d is not None:
            sampled_feat_grid = self.unet3d(sampled_feat_grid)
        return {'grid': sampled_feat_grid}, c_global

    
    def get_nondecoder_params(self) -> list:
        return [self.partial_global_pointnet.parameters(), self.full_global_pointnet.parameters(),
                self.full_local_pointnet.parameters(), self.partial_local_pointnet.parameters(), self.global_vae_encoder.parameters(),
                self.global_prior_encoder.parameters(), self.local_prior_encoders.parameters(), self.local_vae_encoder.parameters(),
            ]