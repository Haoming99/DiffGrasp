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
class KiloC(KiloBase):
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
        Rc=10, global_latent_dim=16, local_cond_dim=16, vd_init=False, fuse_feats=False,
        dec_pre_path: Optional[str]=None, residual_latents=True, resolutions=[1, 4, 8, 16],
        lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1,
        batchnorm=True, dropout=False, reduction='sum', invis_loss_weight=1., 
        interactive_debug=False, freeze_decoder=False, test_num_pts=2048, ed_pre_path=None, attn_grid=False, ende_lr=None):
        
        super().__init__(lr=lr, padding=padding, b_min=b_min, b_max=b_max, points_batch_size=points_batch_size, batch_size=batch_size, test_num_samples=test_num_samples,
                batchnorm=batchnorm, reduction=reduction, invis_loss_weight=invis_loss_weight, interactive_debug=interactive_debug, freeze_decoder=freeze_decoder, test_num_pts=test_num_pts)

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
        self.fuse_feats = fuse_feats
        self.ed_pre_path = ed_pre_path
        self.attn_grid = attn_grid
        self.ende_lr = ende_lr
        self.pretrained_keys = [] #Empty until we configure the optimizer
        # Note: None and 0 are different. None means training from scratch and 0 means freeze the model
        
        assert res_after_type in [None, 'single', 'double'], f"Unsupported res_after_type: {res_after_type}"
        
        self.local_vae_encoders = nn.ModuleList([GroupVAEEncoder(in_channels=self.feat_dim,
                                      hidden_channels=self.global_latent_dim*2,
                                      latent_dim=self.global_latent_dim, groups=cur_reso**3,
                                      use_conditioning=True,
                                      conditioning_channels=self.global_latent_dim,
                                      batchnorm=self.batchnorm,
                                      dropout=self.dropout,
                                      num_layers=1,
                                    ) for cur_reso in resolutions[1:]])
        self.local_prior_encoders = nn.ModuleList([GroupVAEEncoder(in_channels=self.global_latent_dim,
                                      hidden_channels=self.global_latent_dim*2,
                                      latent_dim=self.global_latent_dim, groups=cur_reso**3,
                                      use_conditioning=False,
                                      conditioning_channels=0,
                                      batchnorm=self.batchnorm,
                                      dropout=self.dropout,
                                      num_layers=1,
                                    ) for cur_reso in resolutions[1:]])
        if with_proj:
            self.proj_blocks = nn.ModuleList([GroupLinear(self.global_latent_dim, self.global_latent_dim, 
                                                groups=cur_reso**3) for cur_reso in resolutions[1:]])
        else:
            self.proj_blocks = None

        if res_after_type == 'single':
            self.res_after = nn.ModuleList([GroupFCBlock(self.global_latent_dim, self.global_latent_dim, 
                                                groups=cur_reso**3, batchnorm=self.batchnorm, 
                                                dropout=self.dropout) for cur_reso in resolutions[1:]])
        elif res_after_type == 'double':
            self.res_after = nn.ModuleList([GroupDoubleFCLayer(self.global_latent_dim, self.global_latent_dim, 
                                                groups=cur_reso**3, batchnorm=self.batchnorm, 
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
        
        if ed_pre_path:
            state_dict = torch.load(ed_pre_path)['state_dict']
            self.full_local_pointnet.load_state_dict({k[len('full_local_pointnet.'):] : v for k, v in state_dict.items() if k.startswith('full_local_pointnet.')}, strict=True)
            self.partial_local_pointnet.load_state_dict({k[len('full_local_pointnet.'):] : v for k, v in state_dict.items() if k.startswith('full_local_pointnet.')}, strict=False)
            self.decoder.load_state_dict({k[len('decoder.'):]: v for k, v in state_dict.items() if k.startswith('decoder.')}, strict=False)
            
        
        
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
        
        layered_x = [F.avg_pool3d(x_local['grid'], self.vox_reso // res) for res in self.resolutions[1:]]
        layered_x = [rearrange(cur_x, "b f h w d -> b f (h w d)") for cur_x in layered_x]

        c_global = self.partial_global_pointnet(partial_pcs)
        c_local = self.partial_local_pointnet(partial_pcs)

        grid_c = c_local['grid']

        mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        prior_mu, prior_log_var = self.global_prior_encoder(x=c_global, c=None)
        prior_std = torch.exp(prior_log_var / 2)
        p = torch.distributions.Normal(prior_mu, prior_std)

        repeated_latents = repeat(z, 'b c -> b c (h w d)', h=self.resolutions[0], w=self.resolutions[0], d=self.resolutions[0])
        local_kl_losses = list()

        for idx, (cur_reso, cur_x, cur_vae_encoder, cur_prior_encoder) in enumerate(zip(self.resolutions[1:], layered_x, self.local_vae_encoders, self.local_prior_encoders)):
            repeat_ratio = cur_reso//self.resolutions[idx]
            repeated_latents = repeat(repeated_latents, 'b c (h w d) -> b c (h r1 w r2 d r3)', 
                                     h=self.resolutions[idx], w=self.resolutions[idx], d=self.resolutions[idx], # last_reso
                                      r1=repeat_ratio, r2=repeat_ratio, r3=repeat_ratio) # next_reso / cur_reso

            local_mus, local_log_vars = cur_vae_encoder(x=cur_x, c=repeated_latents)
            local_stds = torch.exp(local_log_vars / 2)
            local_qs = torch.distributions.Normal(local_mus, local_stds)

            local_zs = local_qs.rsample()

            local_prior_mus, local_prior_log_vars = cur_prior_encoder(x=repeated_latents, c=None)
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

        sampled_feat_grid = rearrange(repeated_latents, 'b c (h w d) -> b c h w d', h=self.vox_reso, w=self.vox_reso, d=self.vox_reso)
        
        if self.unet3d is not None:
            if self.fuse_feats:
                sampled_feat_grid = rearrange([sampled_feat_grid, grid_c], 'u b c vx vy vz -> b (u c) vx vy vz')
            sampled_feat_grid = self.unet3d(sampled_feat_grid)
        
        if not self.fuse_feats and not self.attn_grid:
            sampled_feat_grid = rearrange([sampled_feat_grid, grid_c], 'u b c vx vy vz -> b (u c) vx vy vz')
        
        z_grid = {'grid': sampled_feat_grid}
        
        if self.attn_grid:
            z_grid['grid_c'] = c_local['grid']

        pred_occ = self.decode(query_pts, z=z_grid, c=c_global) 
        
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
            breakpoint()

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
            "batch_size": float(self.batch_size)
        }

        if self.interactive_debug:
            print(f"loss: {loss.item()}, kl: {kl_loss.item()}")

        return loss, logs

    def encode_inputs(self, partial_pcs, full_pcs=None):
        c_global = self.partial_global_pointnet(partial_pcs)
        c_local = self.partial_local_pointnet(partial_pcs)

        grid_c = c_local['grid']

        if full_pcs is not None:
            x_local = self.full_local_pointnet(full_pcs)
            x_global = self.full_global_pointnet(full_pcs)

            mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # Process local latents
            layered_x = [F.avg_pool3d(x_local['grid'], self.vox_reso // res) for res in self.resolutions[1:]]
            layered_x = [rearrange(cur_x, "b f h w d -> b f (h w d)") for cur_x in layered_x]
            
            repeated_latents = repeat(z, 'b c -> b c (h w d)', h=self.resolutions[0], w=self.resolutions[0], d=self.resolutions[0])

            for idx, (cur_reso, cur_x, cur_vae_encoder) in enumerate(zip(self.resolutions[1:], layered_x, self.local_vae_encoders)):
                repeat_ratio = cur_reso//self.resolutions[idx]
                repeated_latents = repeat(repeated_latents, 'b c (h w d) -> b c (h r1 w r2 d r3)', 
                                     h=self.resolutions[idx], w=self.resolutions[idx], d=self.resolutions[idx], # last_reso
                                      r1=repeat_ratio, r2=repeat_ratio, r3=repeat_ratio) # next_reso / cur_reso

                local_mus, local_log_vars = cur_vae_encoder(x=cur_x, c=repeated_latents)
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
            repeated_latents = repeat(z, 'b c -> b c (h w d)', h=self.resolutions[0], w=self.resolutions[0], d=self.resolutions[0])

            for idx, (cur_reso, cur_prior_encoder) in enumerate(zip(self.resolutions[1:], self.local_prior_encoders)):
                repeat_ratio = cur_reso//self.resolutions[idx]
                repeated_latents = repeat(repeated_latents, 'b c (h w d) -> b c (h r1 w r2 d r3)', 
                                     h=self.resolutions[idx], w=self.resolutions[idx], d=self.resolutions[idx], # last_reso
                                      r1=repeat_ratio, r2=repeat_ratio, r3=repeat_ratio) # next_reso / cur_reso

                local_prior_mus, local_prior_log_vars = cur_prior_encoder(x=repeated_latents, c=None)
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
                
                      
        sampled_feat_grid = rearrange(local_zs, 'b c (h w d) -> b c h w d', h=self.vox_reso, w=self.vox_reso, d=self.vox_reso)

        if self.unet3d is not None:
            if self.fuse_feats:
                sampled_feat_grid = rearrange([sampled_feat_grid, grid_c], 'u b c vx vy vz -> b (u c) vx vy vz')
            sampled_feat_grid = self.unet3d(sampled_feat_grid)
        
        if not self.fuse_feats and not self.attn_grid:
            sampled_feat_grid = rearrange([sampled_feat_grid, grid_c], 'u b c vx vy vz -> b (u c) vx vy vz')

        z_grid = {'grid': sampled_feat_grid}
        
        if self.attn_grid:
            z_grid['grid_c'] = c_local['grid']
        
        return z_grid, c_global

    
    def get_nondecoder_params(self) -> list:
        return [self.partial_global_pointnet.parameters(), self.full_global_pointnet.parameters(),
                self.full_local_pointnet.parameters(), self.partial_local_pointnet.parameters(), self.global_vae_encoder.parameters(),
                self.global_prior_encoder.parameters(), self.local_prior_encoders.parameters(), self.local_vae_encoder.parameters(),
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
            
        else: # Default behavior
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer