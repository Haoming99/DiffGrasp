import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.backbones.fcresnet import FCBlock, FCResBlock
from utils.geometry import rot6d_to_rotmat, batch_rodrigues, batch_svd
import math

class NPVAE(nn.Module):

    def __init__(self, cfg):
        super(NPVAE, self).__init__()
        self.vae = VAE(cfg)
        self.cfg = cfg
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.initial_layer = nn.Sequential(nn.Linear(cfg.MODEL.BACKBONE.OUT_CHANNELS, cfg.MODEL.SMPLX_HEAD.CONDITIONING_FEATURES), nn.ReLU(inplace=False))

    def forward(self, feats, num_samples=None, pose=None):

        assert( (num_samples is not None) or (pose is not None) )
        batch_size = feats.shape[0]
        if pose is not None:
            if num_samples is None:
                num_samples = pose.shape[1]
            # subtract mean pose to help learning

        # Transform features and replicate them num_samples times
        feats = self.initial_layer(feats)
        feats = feats.unsqueeze(1).aepeat(1, num_samples, 1).reshape(batch_size * num_samples, -1)

        # Decode only
        if pose is None:
            pred_pose = self.vae.sample(feats)
            mu = None
            log_sigma = None
        else:
            pose = pose.reshape(batch_size * num_samples, -1)
            pred_pose, mu, log_sigma = self.vae(pose, feats)
        pred_pose = pred_pose.reshape(batch_size, num_samples, -1)

        return pred_pose, mu, log_sigma


class VAE(nn.Module):

    def __init__(self, cfg):
        super(VAE, self).__init__()
        self.encoder = vae_encoder(cfg)
        nn.init.xavier_uniform_(self.encoder.fc_layers[-1].weight, gain=0.01)
        self.decoder = vae_decoder(cfg)
        nn.init.xavier_uniform_(self.decoder.fc_layers[-1].weight, gain=0.01)

    def encode(self, x, c):
        return self.encoder(x, c)

    def sample(self, c, z=None):
        if z is None:
            z = torch.randn(c.shape[0], self.encoder.latent_dim ,device=c.device, dtype=c.dtype)
        return self.decoder(z, c)

    def forward(self, x, c):
        mu, log_sigma = self.encode(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(log_sigma)
        samples = self.sample(c, z=z)
        return samples, mu, log_sigma

class VAEEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, latent_dim, conditioning_channels, use_conditioning=True, num_layers=3, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(VAEEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_conditioning = use_conditioning
        module_list = [FCBlock(in_channels + int(use_conditioning) * conditioning_channels, hidden_channels, batchnorm=batchnorm, activation=activation)]
        for i in range(num_layers):
            module_list.append(FCResBlock(hidden_channels, hidden_channels, batchnorm=batchnorm, activation=activation))
        module_list.append(nn.Linear(hidden_channels, 2*latent_dim))
        self.fc_layers = nn.Sequential(*module_list)

    def forward(self, x, c):
        if self.use_conditioning:
            x = torch.cat((x, c), dim=-1)
        out = self.fc_layers(x)
        mu = out[:, :self.latent_dim]
        log_sigma = out[:, self.latent_dim:]
        return mu, log_sigma

class VAEDecoder(nn.Module):

    def __init__(self, latent_dim, hidden_channels, out_channels, conditioning_channels, use_conditioning=True, num_layers=3, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(VAEDecoder, self).__init__()
        self.use_conditioning = use_conditioning
        module_list = [FCBlock(latent_dim + int(use_conditioning) * conditioning_channels, hidden_channels, batchnorm=batchnorm, activation=activation)]
        for i in range(num_layers):
            module_list.append(FCResBlock(hidden_channels, hidden_channels, batchnorm=batchnorm, activation=activation))
        module_list.append(nn.Linear(hidden_channels, out_channels))
        self.fc_layers = nn.Sequential(*module_list)

    def forward(self, x, c):
        if self.use_conditioning:
            x = torch.cat((x, c), dim=-1)
        out = self.fc_layers(x)
        return out

def vae_encoder(cfg):
    return VAEEncoder(cfg.MODEL.VAE.IN_DIM, cfg.MODEL.VAE.ENCODER.HIDDEN_CHANNELS,
                    cfg.MODEL.VAE.LATENT_DIM, cfg.MODEL.SMPLX_HEAD.CONDITIONING_FEATURES, cfg.MODEL.VAE.ENCODER.USE_CONDITIONING, cfg.MODEL.VAE.ENCODER.NUM_LAYERS)

def vae_decoder(cfg):
    return VAEDecoder(cfg.MODEL.VAE.LATENT_DIM, cfg.MODEL.VAE.DECODER.HIDDEN_CHANNELS,
                    cfg.MODEL.VAE.OUT_DIM, cfg.MODEL.SMPLX_HEAD.CONDITIONING_FEATURES, cfg.MODEL.VAE.DECODER.USE_CONDITIONING, cfg.MODEL.VAE.DECODER.NUM_LAYERS)
