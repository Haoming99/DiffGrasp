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
from models.vaes import VAEEncoder, GroupVAEEncoder
from models.encoder import ResnetPointnet
from models import LocalPoolPointnet
from models.unet3d import UNet3DCLI
from models.fcresnet import GroupLinear, GroupFCBlock, GroupFCResBlock, GroupDoubleFCLayer

@MODEL_REGISTRY
class KResLoF(KiloBase):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder: nn.Module, partial_global_pointnet: nn.Module, partial_local_pointnet: nn.Module,
                 full_global_pointnet: nn.Module, full_local_pointnet: nn.Module,
        global_vae_encoder: nn.Module, global_prior_encoder: nn.Module,
        unet3d: UNet3DCLI=None, with_proj=True, res_after_type=None,
        feat_dim=16, vox_reso=16, kl_weight=0.01, local_kl_weight=0.01, recon_weight=1., 
        Rc=10, global_latent_dim=16, local_cond_dim=16, vd_init=False,
        dec_pre_path: Optional[str]=None, residual_latents=True, resolutions=[1, 4, 8, 16],
        lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1,
        batchnorm=True, dropout=False, reduction='sum', invis_loss_weight=1., interactive_debug=False, freeze_decoder=False):
        
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

        self.kl_weight = kl_weight
        self.local_kl_weight = local_kl_weight
        self.recon_weight = recon_weight
        self.invis_loss_weight = invis_loss_weight 
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        self.Rc = Rc
        self.residual_latents = residual_latents
        self.resolutions = resolutions
        self.global_latent_dim = global_latent_dim
        self.batchnorm = batchnorm
        self.dropout = dropout
        
        assert res_after_type in [None, 'single', 'double'], f"Unsupported res_after_type: {res_after_type}"
        
        self.local_vae_encoders = nn.ModuleList([GroupVAEEncoder(in_channels=self.feat_dim,
                                      hidden_channels=self.global_latent_dim*2,
                                      latent_dim=self.global_latent_dim, groups=self.Rc * cur_reso * 3,
                                      use_conditioning=True,
                                      conditioning_channels=self.global_latent_dim + self.feat_dim,
                                      batchnorm=self.batchnorm,
                                      dropout=self.dropout,
                                    ) for cur_reso in resolutions[1:]])
        self.local_prior_encoders = nn.ModuleList([GroupVAEEncoder(in_channels=self.global_latent_dim,
                                      hidden_channels=self.global_latent_dim*2,
                                      latent_dim=self.global_latent_dim, groups=self.Rc * cur_reso * 3,
                                      use_conditioning=True,
                                      conditioning_channels=self.feat_dim,
                                      batchnorm=self.batchnorm,
                                      dropout=self.dropout,
                                    ) for cur_reso in resolutions[1:]])
        if with_proj:
            self.proj_blocks = nn.ModuleList([GroupLinear(self.global_latent_dim, self.global_latent_dim, 
                                                groups=self.Rc * cur_reso * 3) for cur_reso in resolutions[1:]])
        else:
            self.proj_blocks = None

        if res_after_type == 'single':
            self.res_after = nn.ModuleList([GroupFCBlock(self.global_latent_dim, self.global_latent_dim, 
                                                groups=self.Rc * cur_reso * 3, batchnorm=self.batchnorm, 
                                                dropout=self.dropout) for cur_reso in resolutions[1:]])
        elif res_after_type == 'double':
            self.res_after = nn.ModuleList([GroupDoubleFCLayer(self.global_latent_dim, self.global_latent_dim, 
                                                groups=self.Rc * cur_reso * 3, batchnorm=self.batchnorm, 
                                                dropout=self.dropout) for cur_reso in resolutions[1:]])
        else:
            self.res_after = None
        
        if vd_init:
            if self.proj_blocks is not None:
                for i, block in enumerate(self.proj_blocks):
                    block.weight.data *= np.sqrt(1 / (i + 1))
                    block.bias.data *= 0.
                    
            if self.res_after is not None:
                for i, block in enumerate(self.res_after):
                    block.fc_block[0].weight.data *= np.sqrt(1 / (i + 1)) # block is nn.Sequential
                    block.fc_block[0].bias.data *= 0
                    if res_after_type == 'double':
                        block.fc_block[4].weight.data *= np.sqrt(1 / (i + 1))
                        block.fc_block[4].bias.data *= 0.

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
        squeezed_x = rearrange(stacked_feats, 's b f v -> b (s f) v')

        layered_x = [F.avg_pool1d(squeezed_x, self.vox_reso // res) for res in self.resolutions[1:]] # The first reso will be provided by sepertate global pointnet
        layered_x = [repeat(cur_x, 'b (s f) vv -> b f (s rc vv)', s=3, rc=self.Rc, f=self.feat_dim) for cur_x in layered_x]

        grid_c = c_local['grid']
        h_c = reduce(grid_c, 'b c h w d -> b c h', 'mean')
        w_c = reduce(grid_c, 'b c h w d -> b c w', 'mean')
        d_c = reduce(grid_c, 'b c h w d -> b c d', 'mean')

        stacked_c = rearrange([h_c, w_c, d_c], 's b f v -> s b f v')
        squeezed_c = rearrange(stacked_c, 's b f v -> b (s f) v')

        layered_c = [F.avg_pool1d(squeezed_c, self.vox_reso // res) for res in self.resolutions[1:]] # The first reso will be provided by sepertate global pointnet
        layered_c = [repeat(cur_c, 'b (s f) vv -> b f (s rc vv)', s=3, rc=self.Rc, f=self.feat_dim) for cur_c in layered_c]

        mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        prior_mu, prior_log_var = self.global_prior_encoder(x=c_global, c=None)
        prior_std = torch.exp(prior_log_var / 2)
        p = torch.distributions.Normal(prior_mu, prior_std)

        repeated_latents = repeat(z, 'b c -> b c (s rc v)', s=3, rc=self.Rc, v=self.resolutions[0])
        local_kl_losses = list()

        for idx, (cur_reso, cur_x, cur_c, cur_vae_encoder, cur_prior_encoder) in enumerate(zip(self.resolutions[1:], layered_x, layered_c, self.local_vae_encoders, self.local_prior_encoders)):
            repeated_latents = repeat(repeated_latents, 'b c (s rc v) -> b c (s rc v ratio)', 
                                      s=3, rc=self.Rc, v=self.resolutions[idx], # last_reso
                                      ratio=cur_reso//self.resolutions[idx]) # next_reso / cur_reso

            combined_local_cond = torch.cat([repeated_latents, cur_c], dim=1)
            local_mus, local_log_vars = cur_vae_encoder(x=cur_x, c=combined_local_cond)
            local_stds = torch.exp(local_log_vars / 2)
            local_qs = torch.distributions.Normal(local_mus, local_stds)

            local_zs = local_qs.rsample()

            local_prior_mus, local_prior_log_vars = cur_prior_encoder(x=repeated_latents, c=cur_c)
            local_prior_stds = torch.exp(local_prior_log_vars / 2)
            local_ps = torch.distributions.Normal(local_prior_mus, local_prior_stds)

            local_kl_losses.append(torch.distributions.kl_divergence(local_qs, local_ps))

            
            if self.proj_blocks is not None:
                local_zs = self.proj_blocks[idx](local_zs)
                
            if self.residual_latents:
                repeated_latents = repeated_latents + local_zs
            else:
                repeated_latents = local_zs
            
            if self.res_after is not None:
                repeated_latents = repeated_latents + self.res_after[idx](repeated_latents)

        fvx, fvy, fvz = rearrange(repeated_latents, 'b f (s rc v) -> s b rc v f', s=3, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
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
            local_kl_loss = sum([cur_kl_loss.sum([-1, -2]) for cur_kl_loss in local_kl_losses]).mean()
            # Sum to get a (b,) tensor
        elif self.reduction == 'mean':
            kl_loss = kl_loss.mean()
            local_kl_loss = sum(map(torch.mean, local_kl_losses)) / len(local_kl_losses)


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
        squeezed_c = rearrange(stacked_c, 's b f v -> b (s f) v')

        layered_c = [F.avg_pool1d(squeezed_c, self.vox_reso // res) for res in self.resolutions[1:]] # The first reso will be provided by sepertate global pointnet
        layered_c = [repeat(cur_c, 'b (s f) vv -> b f (s rc vv)', s=3, rc=self.Rc, f=self.feat_dim) for cur_c in layered_c]
        
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
            stacked_feats = rearrange([h_feats, w_feats, d_feats], 's b f v -> s b f v')
            squeezed_x = rearrange(stacked_feats, 's b f v -> b (s f) v')

            layered_x = [F.avg_pool1d(squeezed_x, self.vox_reso // res) for res in self.resolutions[1:]] # The first reso will be provided by sepertate global pointnet
            layered_x = [repeat(cur_x, 'b (s f) vv -> b f (s rc vv)', s=3, rc=self.Rc, f=self.feat_dim) for cur_x in layered_x]

            repeated_latents = repeat(z, 'b c -> b c (s rc v)', s=3, rc=self.Rc, v=self.resolutions[0])

            for idx, (cur_reso, cur_x, cur_c, cur_vae_encoder) in enumerate(zip(self.resolutions[1:], layered_x, layered_c, self.local_vae_encoders)):
                repeated_latents = repeat(repeated_latents, 'b c (s rc v) -> b c (s rc v ratio)', 
                                          s=3, rc=self.Rc, v=self.resolutions[idx], # last_reso
                                          ratio=cur_reso//self.resolutions[idx]) # next_reso / cur_reso

                combined_local_cond = torch.cat([repeated_latents, cur_c], dim=1)
                local_mus, local_log_vars = cur_vae_encoder(x=cur_x, c=combined_local_cond)
                local_stds = torch.exp(local_log_vars / 2)
                local_qs = torch.distributions.Normal(local_mus, local_stds)

                local_zs = local_qs.rsample()

                if self.proj_blocks is not None:
                    local_zs = self.proj_blocks[idx](local_zs)

                if self.residual_latents:
                    repeated_latents = repeated_latents + local_zs
                else:
                    repeated_latents = local_zs

                if self.res_after is not None:
                    repeated_latents = repeated_latents + self.res_after[idx](repeated_latents)          
            
        else:
            prior_mu, prior_log_var = self.global_prior_encoder(x=c_global, c=None)
            prior_std = torch.exp(prior_log_var / 2)
            p = torch.distributions.Normal(prior_mu, prior_std)
            z = p.rsample()

            # process local latents
            
            repeated_latents = repeat(z, 'b c -> b c (s rc v)', s=3, rc=self.Rc, v=self.resolutions[0])

            for idx, (cur_reso, cur_c, cur_prior_encoder) in enumerate(zip(self.resolutions[1:], layered_c, self.local_prior_encoders)):
                repeated_latents = repeat(repeated_latents, 'b c (s rc v) -> b c (s rc v ratio)', 
                                          s=3, rc=self.Rc, v=self.resolutions[idx], # last_reso
                                          ratio=cur_reso//self.resolutions[idx]) # next_reso / cur_reso

                local_prior_mus, local_prior_log_vars = cur_prior_encoder(x=repeated_latents, c=cur_c)
                local_prior_stds = torch.exp(local_prior_log_vars / 2)
                local_ps = torch.distributions.Normal(local_prior_mus, local_prior_stds)
                
                local_zs = local_ps.rsample()

                if self.proj_blocks is not None:
                    local_zs = self.proj_blocks[idx](local_zs)

                if self.residual_latents:
                    repeated_latents = repeated_latents + local_zs
                else:
                    repeated_latents = local_zs

                if self.res_after is not None:
                    repeated_latents = repeated_latents + self.res_after[idx](repeated_latents)   
                
                      
        fvx, fvy, fvz = rearrange(repeated_latents, 'b f (s rc v) -> s b rc v f', s=3, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        sampled_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)

        if self.unet3d is not None:
            sampled_feat_grid = self.unet3d(sampled_feat_grid)
        return {'grid': sampled_feat_grid}, c_global

    
    def get_nondecoder_params(self) -> list:
        return [self.partial_global_pointnet.parameters(), self.full_global_pointnet.parameters(),
                self.full_local_pointnet.parameters(), self.partial_local_pointnet.parameters(), self.global_vae_encoder.parameters(),
                self.global_prior_encoder.parameters(), self.local_prior_encoders.parameters(), self.local_vae_encoder.parameters(),
            ]