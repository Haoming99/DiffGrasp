import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import numpy as np
from einops import repeat, reduce, rearrange
from tqdm.auto import tqdm
from diffusers import UNet2DModel, AutoencoderKL, VQModel
from diffusers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers import SchedulerMixin
from diffusers.models.vae import DiagonalGaussianDistribution
from ..utils.mesh import generate_from_latent
from typing import Iterable, Union, Optional, Tuple, List
from ..models.diff_cli import UNet2DModelCLI, AutoencoderKLCLI, VQModelCLI
from ..models.unet3d import UNet3DCLI
from ..models.unet import UNet
import itertools
import inspect


class Diffusion3XC(nn.Module):
    """
    Diffusion3X Mono that takes both partial and complete point cloud as input and attempts
    to have a skip connection on partial stream to improve fidelity.
    Features will be appended with _c for attention decoder
    Compatible (c)VAE that supports empty z.
    """

    def __init__(self, decoder: nn.Module, local_pointnet: nn.Module, vae: nn.Module, unet: UNet2DModel,
                 scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
                 unet3d: nn.Module = None, feat_dim=16, vox_reso=64, latent_dim: int = 4,
                 num_inference_steps: int = 50, plane_types: Iterable[str] = ['xy', 'xz', 'yz'],
                 generator=None, eta=0., verbose=False, normalize_latent=False,
                 full_std=1 / 0.18215, partial_std=1 / 0.18215, mix_noise=False,
                 corrupt_partial=False, corrupt_mult=1.0, attend_partial=False,
                 b_min=-0.5, b_max=0.5, padding=0.1):

        super().__init__()

        self.decoder = decoder  # ConvONet decoder
        self.unet3d = unet3d  # 3D UNet after VAE sampling
        self.local_pointnet = local_pointnet
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler  # Noise scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        self.feat_dim = feat_dim
        self.vox_reso = vox_reso
        self.latent_dim = latent_dim  # Note for VQVAE this latent_dim is 3x of true embedding dim as they are concatenated.
        self.plane_types = plane_types
        self.eta = eta
        self.normalize_latent = normalize_latent

        self.mix_noise = mix_noise
        self.corrupt_partial = corrupt_partial
        self.corrupt_mult = corrupt_mult
        self.attend_partial = attend_partial
        self.verbose = verbose
        self.partial_std = partial_std
        self.full_std = full_std

        self.b_min = b_min
        self.b_max = b_max
        self.padding = padding

    def decode(self, p, z, c=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z: dict or tensor for decoder
            c (tensor): latent conditioned code c
        '''
        logits = self.decoder(p, c_plane=z, c=c, **kwargs)
        return logits

    def encode_vae_latents(self, feats: dict) -> torch.Tensor:
        stacked_feats = rearrange([feats[k] for k in self.plane_types], 'p b f v1 v2 -> b (f p) v1 v2')
        latent_dist = self.vae.encode(stacked_feats, return_dict=False)[0]
        if isinstance(latent_dist, DiagonalGaussianDistribution):
            latents = latent_dist.sample()
        else:
            latents = latent_dist  # VQ-VAE
        return latents

    def encode_inputs(self, partial_pcs, full_pcs=None):
        bs = partial_pcs.shape[0]
        pred_feats = {}
        latents_shape = (bs, self.latent_dim, self.vox_reso // 2 ** (len(self.vae.decoder.up_blocks) - 1),
                         self.vox_reso // 2 ** (len(self.vae.decoder.up_blocks) - 1))

        with torch.no_grad():
            c_local = self.local_pointnet(partial_pcs)

        latents = torch.randn(latents_shape, generator=None, device=partial_pcs.device)
        timesteps_tensor = self.scheduler.timesteps.to(partial_pcs.device)
        latents = latents * self.scheduler.init_noise_sigma

        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.eta

        partial_latents = self.encode_vae_latents(c_local)

        if self.normalize_latent:
            latents_std = partial_latents.std(dim=(1, 2, 3), keepdim=True)
            partial_latents = partial_latents / latents_std
        else:
            partial_latents = partial_latents / self.partial_std

        if self.corrupt_partial:
            partial_latents += self.corrupt_mult * torch.randn_like(partial_latents)

        it = tqdm(timesteps_tensor) if self.verbose else timesteps_tensor

        for t in it:
            if self.mix_noise:
                latent_model_input = latents + partial_latents
            else:
                latent_model_input = torch.cat([latents, partial_latents], dim=1)

            noise_pred = self.unet(latent_model_input, t).sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        if self.normalize_latent:
            latents = latents * latents_std
        else:
            latents = latents * self.partial_std

        decoded_feats = self.vae.decode(latents).sample
        splited_preds = rearrange(decoded_feats, 'b (f p) v1 v2 -> p b f v1 v2', f=self.feat_dim,
                                  p=len(self.plane_types))
        pred_feats.update({k: v for k, v in zip(self.plane_types, splited_preds)})

        pred_feats.update({f"{k}_c": v for k, v in zip(self.plane_types, splited_preds)})

        return pred_feats, torch.empty((bs, 0), device=partial_pcs.device)

    def forward(self, partial_pcs, full_pcs=None):
        return self.encode_inputs(partial_pcs, full_pcs)

    def eval_points(self, p, z=None, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, 100000)  # Using a fixed batch size for points
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(p.device)
            with torch.no_grad():
                occ_hat = self.decode(pi, z=z, c=c)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def generate_mesh(self, partial_pcs, num_samples=1, denormalize=False):
        bs = partial_pcs.shape[0]
        device = partial_pcs.device

        with torch.no_grad():
            partial_pcs = repeat(partial_pcs, 'b p d -> (b s) p d', s=num_samples).to(device)
            decoded_latent, c = self.encode_inputs(partial_pcs)

            decoded_latent = {k: rearrange(v, '(b s) ... -> b s ...', b=bs, s=num_samples) for k, v in
                              decoded_latent.items()}
            c = rearrange(c, '(b s) ... -> b s ...', b=bs, s=num_samples)

            mesh_list = list()
            for b_i in range(bs):
                cur_mesh_list = list()

                state_dict = dict()
                for sample_idx in range(num_samples):
                    cur_decoded_latent = {k: v[b_i, sample_idx, None] for k, v in decoded_latent.items()}
                    cur_c = c[b_i, sample_idx, None]
                    generated_mesh = generate_from_latent(self.eval_points, z=cur_decoded_latent, c=cur_c,
                                                          state_dict=state_dict, padding=self.padding,
                                                          B_MAX=self.b_max, B_MIN=self.b_min, device=device)
                    cur_mesh_list.append(generated_mesh)
                mesh_list.append(cur_mesh_list)

        return mesh_list
