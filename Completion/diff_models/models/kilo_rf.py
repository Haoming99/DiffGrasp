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
from models.conv_decoder import ConvONetDecoder
from models.encoder import ResnetPointnet
from models.vaes import VAEEncoder, GroupVAEEncoder
from models.encoder import ResnetPointnet
from models import LocalPoolPointnet
from models.unet3d import UNet3DCLI

@MODEL_REGISTRY
class KiloRF(pl.LightningModule):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder: nn.Module, unet3d: UNet3DCLI=None, lr=1e-4, padding=0.1, b_min=-0.5,
                 b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1, kl_weight=0.01,
                feat_dim=16, pts_dim=3, vox_reso=16, local_kl_weight=0.01, recon_weight=1., 
                batchnorm=True, reduction='sum', Rc=10, global_latent_dim=None,
                invis_loss_weight=1., interactive_debug=False, dec_pre_path: Optional[str]=None, freeze_decoder=False):
        
        super().__init__()

        self.decoder = decoder # ConvONet decoder
        self.unet3d = unet3d #3D UNet after VAE sampling

        # For backward compatibility
        global_latent_dim = feat_dim if global_latent_dim is None else global_latent_dim

        self.partial_global_pointnet = ResnetPointnet(c_dim=global_latent_dim, dim=3, hidden_dim=global_latent_dim*2)
        self.full_global_pointnet = ResnetPointnet(c_dim=global_latent_dim, dim=3, hidden_dim=global_latent_dim*2)
        self.full_local_pointnet = LocalPoolPointnet(c_dim=feat_dim, dim=3, hidden_dim=32, scatter_type='max',
                                                   unet=False,unet_kwargs=None, unet3d=True, 
                                                   unet3d_kwargs={'num_levels':3, 'f_maps': feat_dim, 'in_channels': feat_dim, 'out_channels': feat_dim},
        plane_resolution=None, grid_resolution=vox_reso, plane_type='grid', padding=0.1, n_blocks=5)
        
        # Init the global vae and local vaes, consider use index
        self.global_vae_encoder = VAEEncoder(global_latent_dim, global_latent_dim*2, global_latent_dim, conditioning_channels=global_latent_dim, batchnorm=batchnorm)
        self.global_prior_encoder = VAEEncoder(global_latent_dim, global_latent_dim*2, global_latent_dim, conditioning_channels=0, use_conditioning=False, batchnorm=batchnorm)
        
        # Local VAEs, naive implementation
        self.local_vae_encoders = GroupVAEEncoder(feat_dim, feat_dim*2, feat_dim, groups=Rc*3*feat_dim, conditioning_channels=global_latent_dim, use_conditioning=True, batchnorm=batchnorm)
        self.local_prior_encoders = GroupVAEEncoder(global_latent_dim, feat_dim*2, feat_dim, groups=Rc*3*feat_dim, conditioning_channels=0, use_conditioning=False, batchnorm=batchnorm)

        self.out_dir = None
        self.lr = lr
        self.padding = padding
        self.b_min = b_min
        self.b_max = b_max
        self.points_batch_size = points_batch_size
        self.batch_size = batch_size
        self.test_num_samples = test_num_samples
        self.kl_weight = kl_weight
        self.local_kl_weight = local_kl_weight
        self.recon_weight = recon_weight
        self.invis_loss_weight = invis_loss_weight 
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        self.Rc = Rc
        self.freeze_decoder = freeze_decoder

        assert reduction in ['sum', 'mean'], f"reduction {reduction} not supported"
        self.reduction = reduction
        self.interactive_debug = interactive_debug
        self._override_ckpt = None

        if dec_pre_path:
            ori_state_dict = torch.load(dec_pre_path)
            self.decoder.load_state_dict({k.lstrip('decoder.'): v for k, v in ori_state_dict['model'].items() if k.startswith('decoder.')})
        

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.
        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        # This is not working for now, prepare demo in the future
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c=c, **kwargs)
        return p_r

    
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

        grid_feats = x_local['grid']
        h_feats = reduce(grid_feats, 'b c h w d -> b c h', 'mean')
        w_feats = reduce(grid_feats, 'b c h w d -> b c w', 'mean')
        d_feats = reduce(grid_feats, 'b c h w d -> b c d', 'mean')
        
        stacked_feats = rearrange([h_feats, w_feats, d_feats], 's b f v -> s b f v')
        stacked_feats = repeat(stacked_feats, 's b f v -> s rc b f v', rc=self.Rc)
        flatten_x_local = rearrange(stacked_feats, 's rc b f v -> b f (s rc v)')

        mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        prior_mu, prior_log_var = self.global_prior_encoder(x=c_global, c=None)
        prior_std = torch.exp(prior_log_var / 2)
        p = torch.distributions.Normal(prior_mu, prior_std)

        repeated_z = repeat(z, 'b c -> b c (s rc v)', s=3, rc=self.Rc, v=self.vox_reso)
        local_mus, local_log_vars = self.local_vae_encoders(x=flatten_x_local, c=repeated_z)
        local_stds = torch.exp(local_log_vars / 2)
        local_qs = torch.distributions.Normal(local_mus, local_stds)
        
        local_zs = local_qs.rsample()

        local_prior_mus, local_prior_log_vars = self.local_prior_encoders(x=repeated_z, c=None)
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

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        #self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, logs = self.step(batch, batch_idx)

        self.log_dict({f"val/{k}": v for k, v in logs.items()}, batch_size=self.batch_size)
        return loss

    def eval_points(self, p, z=None, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        # p = self.query_embeder(p)
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.decode(pi, z=z, c=c)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def encode_inputs(self, partial_pcs, full_pcs=None):
        c_global = self.partial_global_pointnet(partial_pcs)
        
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
            local_mus, local_log_vars = self.local_vae_encoders(x=flatten_x_local, c=repeated_z)
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
            local_prior_mus, local_prior_log_vars = self.local_prior_encoders(x=repeated_z, c=None)
            local_prior_stds = torch.exp(local_prior_log_vars / 2)
            local_ps = torch.distributions.Normal(local_prior_mus, local_prior_stds)
            local_zs = local_ps.rsample()
                      
        fvx, fvy, fvz = rearrange(local_zs, 'b f (s rc v) -> s b rc v f', s=3, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        sampled_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)

        if self.unet3d is not None:
            sampled_feat_grid = self.unet3d(sampled_feat_grid)
        return {'grid': sampled_feat_grid}, c_global

    
    def generate_mesh(self, batch, batch_idx, max_bs=3, num_samples=1, sample=True, denormalize=False):
        partial_pcs = batch['partial_pcs'] # No difference between partial and complete pcs so far
        bs = partial_pcs.shape[0]
        if max_bs < 0:
            max_bs = bs
        device = self.device

        if denormalize:
            loc = batch['loc'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()

        partial_pcs = repeat(partial_pcs, 'b p d -> (b s) p d', s=num_samples).to(device)
        full_pcs = None if sample else repeat(batch['full_pcs'], 'b p d -> (b s) p d', s=num_samples).to(device)

        with torch.no_grad():
            decoded_latent, c = self.encode_inputs(partial_pcs, full_pcs)

            decoded_latent = {k: rearrange(v, '(b s) ... -> b s ...', b=bs, s=num_samples) for k, v in decoded_latent.items()}
            c = rearrange(c, '(b s) ... -> b s ...', b=bs, s=num_samples)

            mesh_list = list()
            input_list = list()
            for b_i in range(min(max_bs, bs)):
                cur_mesh_list = list()
                input_list.append(batch['partial_pcs'][b_i])

                state_dict = dict()
                for sample_idx in range(num_samples):
                    cur_decoded_latent = {k: v[b_i, sample_idx, None] for k, v in decoded_latent.items()}
                    cur_c = c[b_i, sample_idx, None]
                    generated_mesh = generate_from_latent(self.eval_points, z=cur_decoded_latent, c=cur_c,
                                                          state_dict=state_dict, padding=self.padding,
                                                          B_MAX=self.b_max, B_MIN=self.b_min, device=self.device)
                    cur_mesh_list.append(generated_mesh)
                    if denormalize:
                        generated_mesh.vertices = (generated_mesh.vertices + loc[b_i]) * scale[b_i]
                mesh_list.append(cur_mesh_list)

        return mesh_list, input_list

    def test_step(self, batch, batch_idx):
        mesh_list, _ = self.generate_mesh(batch, batch_idx, max_bs=-1, num_samples=self.test_num_samples, sample=True, denormalize=True)
        denormalized_pcs = (batch['partial_pcs'] + batch['loc']) * batch['scale']
        export_shapenet_samples(mesh_list, batch['category'], batch['model'], denormalized_pcs, self.out_dir)
    

    def predict_step(self, batch, batch_idx):
        mesh_list, _ = self.generate_mesh(batch, batch_idx, max_bs=-1, num_samples=self.test_num_samples, sample=True, denormalize=True)
        denormalized_pcs = (batch['partial_pcs'] + batch['loc']) * batch['scale']
        export_shapenet_samples(mesh_list, batch['category'], batch['model'], denormalized_pcs, self.out_dir)

    def configure_optimizers(self):
        if not self.freeze_decoder:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam([{'params': itertools.chain(
                self.partial_global_pointnet.parameters(), self.full_global_pointnet.parameters(),
                self.full_local_pointnet.parameters(), self.global_vae_encoder.parameters(),
                self.global_prior_encoder.parameters(), self.local_prior_encoders.parameters(), self.local_vae_encoder.parameters(),
            )},
                             {'params': self.decoder.parameters(), 'lr': 0}], lr=self.lr)
            return optimizer

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    @property
    def override_ckpt(self):
        return self._override_ckpt

    @override_ckpt.setter
    def override_ckpt(self, ckpt_path):
        print(f"[IMPORTANT]: Setting override ckpt path {self._override_ckpt} -> {ckpt_path}")
        self._override_ckpt = ckpt_path

    def on_predict_epoch_start(self):
        ckpt_path = self._override_ckpt
        if ckpt_path is not None:
            print(f"[IMPORTANT]: Loading {ckpt_path} before predict epoch start")
            ckpt = torch.load(self.override_ckpt)
            print(f"Loaded, global step: {ckpt['global_step']}")
            self.load_state_dict(ckpt['state_dict'])
