from functools import partial
from optparse import Option
from tokenize import PseudoExtras
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import pytorch_lightning as pl
from typing import Iterable, Union, Optional, Tuple, List
import itertools
import numpy as np
import logging
from tqdm.auto import tqdm
from models.unet import UNet
from utils.eval_utils import compute_iou
# from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from utils.mesh import generate_from_latent, export_shapenet_samples
from einops import repeat, reduce, rearrange
from models.kilo_base import KiloBase
from models.unet3d import UNet3DCLI
from diffusers import UNet2DModel, AutoencoderKL, VQModel
from diffusers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers import SchedulerMixin
from diffusers.models.vae import DiagonalGaussianDistribution
from models.diff_cli import UNet2DModelCLI, AutoencoderKLCLI, VQModelCLI
import inspect


# @MODEL_REGISTRY
class Diffusion3X(KiloBase):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoder: nn.Module, local_pointnet: nn.Module, vae: nn.Module, unet: UNet2DModelCLI,
                 scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler], 
                 unet3d: UNet3DCLI = None, feat_dim=16, vox_reso=64, latent_dim: int=4, num_inference_steps: int=50, 
                 plane_types: Iterable[str]=['xy', 'xz', 'yz'], generator=None, eta=0., verbose=False,
                 normalize_latent=False, full_std=1/0.18215, partial_std=1/0.18215,
                 lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1, filter_nan=False,
                 batchnorm=True, dropout=False, reduction='sum', invis_loss_weight=1., interactive_debug=False, freeze_decoder=False, accumulate_grad_batches=1, automatic_optimization=True,
                 test_num_pts=2048, pretrained_path: Optional[str] = None, pretrained_lr=None, lr_decay_steps=1.0e5, pre_lr_freeze_steps=0, pre_lr_freeze_factor=0., lr_decay_factor=0.9, gradient_clip_val=None,
                 mix_noise=False, corrupt_partial=False, corrupt_mult=1.0, attend_partial=False,
                optimizer_name:str='Adam'):

        super().__init__(lr=lr, padding=padding, b_min=b_min, b_max=b_max, points_batch_size=points_batch_size, batch_size=batch_size, test_num_samples=test_num_samples,
                         batchnorm=batchnorm, reduction=reduction, invis_loss_weight=invis_loss_weight, interactive_debug=interactive_debug, freeze_decoder=freeze_decoder, test_num_pts=test_num_pts)

        self.automatic_optimization = automatic_optimization

        self.decoder = decoder  # ConvONet decoder
        self.unet3d = unet3d  # 3D UNet after VAE sampling

        self.local_pointnet = local_pointnet
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler # Noise scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        self.generator = generator

        self.save_hyperparameters(ignore=["decoder", "unet3d", "local_pointnet", "generator", "scheduler", "vae", "unet"])

        self.invis_loss_weight = invis_loss_weight
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim # Note for VQVAE this latent_dim is 3x of true embedding dim as they are concatenated.
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.plane_types = plane_types
        self.filter_nan = filter_nan
        self.eta = eta
        self.normalize_latent = normalize_latent

        self.pretrained_path = pretrained_path
        self.pretrained_lr = pretrained_lr
        # Note: None and 0 are different. None means training from scratch and 0 means freeze the model
        self.pretrained_param_names = []
        self.pretrained_keys = []  # Empty until we configure the optimizer
        
        self.lr_decay_steps = int(lr_decay_steps)
        self.pre_lr_freeze_steps = int(pre_lr_freeze_steps)
        self.pre_lr_freeze_factor = pre_lr_freeze_factor
        self.lr_decay_factor = lr_decay_factor
        compression_ratio = len(self.vae.decoder.up_blocks) - 1 # don't downsalce in the last block
        self.latent_plane_dim = vox_reso // 2**compression_ratio
        self.mix_noise = mix_noise
        self.corrupt_partial = corrupt_partial
        self.corrupt_mult = corrupt_mult
        self.attend_partial =  attend_partial
        self.optimizer_name = optimizer_name
        self.verbose = verbose

        if pretrained_path: # Depracted for now
            console_logger = logging.getLogger("pytorch_lightning")
            console_logger.info(f"Loading pretrained checkpoint from {pretrained_path}")

            state_dict = torch.load(pretrained_path)['state_dict']
            self.load_state_dict(state_dict, strict=False) 

    def decode(self, p, z, c=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z: dict or tensor for decdoer
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
            latents = latent_dist # VQ-VAE
        return latents

    def step(self, batch, batch_idx):
        full_pcs = batch['full_pcs']
        partial_pcs = batch['partial_pcs']

        with torch.no_grad():
            x_local = self.local_pointnet(full_pcs)
            c_local = self.local_pointnet(partial_pcs)
            
            partial_latents = self.encode_vae_latents(c_local)
            latents = self.encode_vae_latents(x_local)
            
            if self.normalize_latent:
                partial_latents = partial_latents / partial_latents.std(dim=(1, 2, 3), keepdim=True)
                latents = latents / latents.std(dim=(1, 2, 3), keepdim=True)
            else:
                partial_latents = partial_latents / self.hparams.partial_std
                latents = latents / self.hparams.full_std

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        if self.mix_noise:
            noisy_in = noisy_latents + latents # Maybe add a weight?
        elif self.corrupt_partial:
            partial_latents += self.corrupt_mult * torch.randn_like(partial_latents)
            noisy_in = torch.cat([noisy_latents, partial_latents], dim=1)
        else:
            noisy_in = torch.cat([noisy_latents, partial_latents], dim=1)

        noise_pred = self.unet(noisy_in, timesteps).sample
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        if self.interactive_debug and (torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss))):
            import ipdb
            ipdb.set_trace()


        logs = {
            "loss": loss,
            "batch_size": float(self.batch_size)
        }

        return loss, logs

    def encode_inputs(self, partial_pcs, full_pcs=None):
        bs = partial_pcs.shape[0] # just to have compatibality with previous callbacks
        pred_feats = {}
        latents_shape = (bs, self.latent_dim, self.latent_plane_dim, self.latent_plane_dim)
        
        with torch.no_grad():
            c_local = self.local_pointnet(partial_pcs)

        # set timesteps
        # self.scheduler.set_timesteps(num_inference_steps)

        latents = torch.randn(latents_shape, generator=self.generator, device=partial_pcs.device)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = self.eta
            
        partial_latents = self.encode_vae_latents(c_local)

        if self.normalize_latent:
            latents_std = partial_latents.std(dim=(1, 2, 3), keepdim=True)
            partial_latents = partial_latents / latents_std
        else:
            partial_latents = partial_latents / self.hparams.partial_std

        if self.corrupt_partial:
            partial_latents += self.corrupt_mult * torch.randn_like(partial_latents)
        
        it = tqdm(timesteps_tensor) if self.verbose else timesteps_tensor
        for i, t in enumerate(it):
            if self.mix_noise:
                latent_model_input = latents + partial_latents # Maybe add a weight?
            else:
                latent_model_input = torch.cat([latents, partial_latents], dim=1)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t).sample

        #     # perform guidance
        #     if do_classifier_free_guidance:
        #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        if self.normalize_latent:
            latents = latents * latents_std
        else:
            latents = latents * self.hparams.partial_std # Should we use partial of full std here?

        decoded_feats = self.vae.decode(latents).sample
        splited_preds = rearrange(decoded_feats, 'b (f p) v1 v2 -> p b f v1 v2', f=self.feat_dim, p=len(self.plane_types))
        pred_feats.update({k: v for k, v in zip(self.plane_types, splited_preds)})

        return pred_feats, torch.empty((bs, 0), device=partial_pcs.device)

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)

        if not self.automatic_optimization:
            self.manual_backward(loss)

            # accumulate gradients of N batches
            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
                opts = self.optimizers()
                for opt in opts:
                    self.clip_gradients(opt, gradient_clip_val=self.hparams.gradient_clip_val)
                    opt.step()
                    opt.zero_grad()

            scheds = self.lr_schedulers()
            for sched in scheds:
                sched.step()
        #self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        return loss

    def get_nondecoder_params(self) -> list:
        # Depreacted
        return None

    def split_ende_params(self) -> tuple:
        return None

    def configure_optimizers(self):
        optimizer_cls = torch.optim.__dict__[self.optimizer_name]
        optimizer = optimizer_cls(self.unet.parameters(), lr=self.lr)
        return optimizer