from http.client import PRECONDITION_FAILED
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
from models.vaes import VAEEncoder, GroupVAEEncoder
from models.encoder import ResnetPointnet
from models import LocalPoolPointnet
from models.unet3d import UNet3DCLI
from models.fcresnet import GroupLinear, GroupFCBlock, GroupFCResBlock, GroupDoubleFCLayer
# from torch_ema import ExponentialMovingAverage

@MODEL_REGISTRY
class KRLocalSemi(KiloBase):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder: nn.Module, full_global_pointnet: nn.Module, partial_global_pointnet: nn.Module, local_pointnet: nn.Module,
        global_vae_encoder: nn.Module, global_prior_encoder: nn.Module, factor_layer: nn.Module,
        unet3d: UNet3DCLI=None, with_proj=True, res_after_type=None, disable_prior_cond=False,
        feat_dim=16, vox_reso=16, kl_weight=0.01, local_kl_weight=0.01, recon_weight=1., 
        Rc=10, global_latent_dim=16, local_cond_dim=16, vd_init=False, fuse_feats=False, cat_grid=False,
        dec_pre_path: Optional[str]=None, residual_latents=True, resolutions=[1, 4, 8, 16],
        lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1,
        batchnorm=True, dropout=False, reduction='sum', invis_loss_weight=1., interactive_debug=False, freeze_decoder=False, 
        test_num_pts=2048, ed_pre_path=None, attn_grid=False, ende_lr=None, num_fc_layers=3):
        
        super().__init__(lr=lr, padding=padding, b_min=b_min, b_max=b_max, points_batch_size=points_batch_size, batch_size=batch_size, test_num_samples=test_num_samples,
                batchnorm=batchnorm, reduction=reduction, invis_loss_weight=invis_loss_weight, interactive_debug=interactive_debug, freeze_decoder=freeze_decoder, test_num_pts=test_num_pts)

        self.decoder = decoder # ConvONet decoder
        self.unet3d = unet3d #3D UNet after VAE sampling

        # For backward compatibility
        global_latent_dim = feat_dim if global_latent_dim is None else global_latent_dim

        self.full_global_pointnet = full_global_pointnet
        self.partial_global_pointnet = partial_global_pointnet
        self.local_pointnet = local_pointnet

        # Init the global vae and local vaes, consider use index
        self.global_vae_encoder = global_vae_encoder
        self.global_prior_encoder = global_prior_encoder
        self.factor_layer = factor_layer

        self.save_hyperparameters(ignore=["decoder", "unet3d", "global_vae_encoder", "global_prior_encoder",
          "full_global_pointnet", "partial_global_pointnet", "local_pointnet", "factor_layer"])

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
         
        assert res_after_type in [None, 'single', 'double', 'single_linear', 'single_relu'], f"Unsupported res_after_type: {res_after_type}"
        
        self.local_vae_encoders = nn.ModuleList([GroupVAEEncoder(in_channels=self.feat_dim,
                                      hidden_channels=self.global_latent_dim*2,
                                      latent_dim=self.global_latent_dim, groups=self.Rc * cur_reso * 3,
                                      use_conditioning=True,
                                      conditioning_channels=self.global_latent_dim + self.feat_dim,
                                      batchnorm=self.batchnorm,
                                      dropout=self.dropout,
                                      num_layers=num_fc_layers,
                                    ) for cur_reso in resolutions[1:]])
        self.local_prior_encoders = nn.ModuleList([GroupVAEEncoder(in_channels=self.global_latent_dim,
                                      hidden_channels=self.global_latent_dim*2,
                                      latent_dim=self.global_latent_dim, groups=self.Rc * cur_reso * 3,
                                      use_conditioning=not disable_prior_cond,
                                      conditioning_channels=self.feat_dim, # should we use self.global_feat_dim instead?
                                      batchnorm=self.batchnorm,
                                      dropout=self.dropout,
                                      num_layers=num_fc_layers,
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
        elif res_after_type == 'single_linear' or res_after_type == 'single_relu':
            self.res_after = nn.ModuleList([GroupLinear(self.global_latent_dim, self.global_latent_dim, 
                                                groups=self.Rc * cur_reso * 3) for cur_reso in resolutions[1:]])
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

        x_local = self.local_pointnet(full_pcs)
        x_global = self.full_global_pointnet(full_pcs)

        c_global = self.partial_global_pointnet(partial_pcs)
        c_local = self.local_pointnet(partial_pcs)

        layered_x = self.factor_layer(x_local)
        layered_c = self.factor_layer(c_local)

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
                if self.hparams.res_after_type == 'single_relu':
                    repeated_latents = F.relu(repeated_latents + self.res_after[idx](repeated_latents))
                else:
                    repeated_latents = repeated_latents + self.res_after[idx](repeated_latents)

        fvx, fvy, fvz = rearrange(repeated_latents, 'b f (s rc v) -> s b rc v f', s=3, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        sampled_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)

        if self.unet3d is not None:
            sampled_feat_grid = self.unet3d(sampled_feat_grid)

        if self.hparams.cat_grid:
            sampled_feat_grid =torch.cat([sampled_feat_grid, c_local['grid']], dim=1)

        pred_feats = {'grid': sampled_feat_grid}
        
        pred_occ = self.decode(query_pts, z=pred_feats, c=c_global) 
        
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

        if torch.any(torch.isnan(loss)):
            console_logger = logging.getLogger("pytorch_lightning")
            console_logger.error(f"NaN loss encountered: loss: {loss}, kl_loss: {kl_loss}, local_kl_loss: {local_kl_loss}")
            loss = self.kl_weight * kl_loss

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
            "batch_size": float(self.batch_size)
        }

        if self.interactive_debug:
            print(f"loss: {loss.item()}, kl: {kl_loss.item()}")

        return loss, logs

    def encode_inputs(self, partial_pcs, full_pcs=None):
        c_global = self.partial_global_pointnet(partial_pcs)
        c_local = self.local_pointnet(partial_pcs)

        layered_c = self.factor_layer(c_local)
        
        if full_pcs is not None:
            x_local = self.local_pointnet(full_pcs)
            x_global = self.full_global_pointnet(full_pcs)

            mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # Process local latents
            layered_x = self.factor_layer(x_local)

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
                    if self.hparams.res_after_type == 'single_relu':
                        repeated_latents = F.relu(repeated_latents + self.res_after[idx](repeated_latents))
                    else:
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
                    if self.hparams.res_after_type == 'single_relu':
                        repeated_latents = F.relu(repeated_latents + self.res_after[idx](repeated_latents))
                    else:
                        repeated_latents = repeated_latents + self.res_after[idx](repeated_latents)
                
                      
        fvx, fvy, fvz = rearrange(repeated_latents, 'b f (s rc v) -> s b rc v f', s=3, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        sampled_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)

        if self.unet3d is not None:
            sampled_feat_grid = self.unet3d(sampled_feat_grid)

        if self.hparams.cat_grid:
            sampled_feat_grid =torch.cat([sampled_feat_grid, c_local['grid']], dim=1)

        pred_feats = {'grid': sampled_feat_grid}
        # NOTE: Maybe we need an attention layer for this mixed feats
        
        return pred_feats, c_global
    
    def get_nondecoder_params(self) -> list:
        # DEPERACTED
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