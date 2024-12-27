import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from einops import repeat, reduce, rearrange
from models.embedder import PositionalEncoding

class GroupConv1x1Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(GroupConv1x1Block, self).__init__()
        module_list = [nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding='same', padding_mode='reflect', groups=groups)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_channels))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(nn.Dropout(inplace=True))
        self.conv_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.conv_block(x)

class GroupConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(GroupConvBlock, self).__init__()
        module_list = [nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', groups=groups)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_channels))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(nn.Dropout(inplace=True))
        self.conv_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.conv_block(x)

class GroupConvResBlock(nn.Module):
    """Residual block using conv layers with group settings."""
    def __init__(self, in_channels, out_channels, groups=1, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False, num_layers=2):
        super(GroupConvResBlock, self).__init__()
        module_list = list()
        for i in range(num_layers):
            module_list.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding='same', padding_mode='reflect', groups=groups))
            in_channels = out_channels
            if batchnorm:
                module_list.append(nn.BatchNorm1d(out_channels))
            if activation is not None:
                module_list.append(activation)
            if dropout:
                module_list.append(nn.Dropout(inplace=True))
            self.conv_block = nn.Sequential(*module_list)

    def forward(self, x):
        return F.relu(x + self.conv_block(x))

@MODEL_REGISTRY
class GroupConvVAEEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, latent_dim, groups, conditioning_channels, use_conditioning=True, spatial_dim=3, Rc=10, vox_reso=32, num_layers=3, batchnorm=True, dropout=False):
        super(GroupConvVAEEncoder, self).__init__()
        activation=nn.ReLU(inplace=True)
        self.latent_dim = latent_dim
        self.conditioning_channels = conditioning_channels
        self.use_conditioning = use_conditioning
        self.spatial_dim = spatial_dim
        self.Rc = Rc
        self.vox_reso = vox_reso
        self.proj_cond = GroupConv1x1Block(in_channels=conditioning_channels, out_channels=in_channels)
        if latent_dim > 0:
            actual_in = in_channels * self.spatial_dim * self.Rc
            actual_hidden = hidden_channels * self.spatial_dim * self.Rc
            module_list = [GroupConv1x1Block(actual_in, actual_hidden, groups=groups, batchnorm=batchnorm, activation=activation)]
            for i in range(num_layers):
                module_list.append(GroupConvResBlock(actual_hidden, actual_hidden, groups=groups, batchnorm=batchnorm, activation=activation))
            module_list.append(GroupConv1x1Block(actual_hidden, 2*latent_dim*self.spatial_dim*self.Rc, batchnorm=None, activation=None))
            self.conv_layers = nn.Sequential(*module_list)
        else:
            self.conv_layers = None

    def forward(self, x, c):
        if self.latent_dim > 0:
            if self.use_conditioning:
                x = x + self.proj_cond(c)

            x = rearrange(x, 'b c (s rc v) -> b (s rc c) v', s=self.spatial_dim, rc=self.Rc, v=self.vox_reso)
            out = self.conv_layers(x)
            mu = out[:, ::2]
            log_sigma = out[:, 1::2] # Features in group conv has to be neighbor
            
            mu = rearrange(mu, 'b (s rc f) v -> b f (s rc v)', s=self.spatial_dim, rc=self.Rc, v=self.vox_reso, f=self.latent_dim)
            log_sigma = rearrange(log_sigma, 'b (s rc f) v -> b f (s rc v)', s=self.spatial_dim, rc=self.Rc, v=self.vox_reso, f=self.latent_dim)
        else:
            mu = torch.empty(x.shape[0], 0, x.shape[-1])
            log_sigma = torch.empty(x.shape[0], 0, x.shape[-1])
        return mu, log_sigma