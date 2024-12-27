import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fcresnet import FCBlock, FCResBlock, GroupLinear, GroupFCBlock, GroupFCResBlock
# from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from einops import repeat, reduce, rearrange

class VAEEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, latent_dim, conditioning_channels, use_conditioning=True, num_layers=3, batchnorm=True, dropout=False):
        super(VAEEncoder, self).__init__()
        activation=nn.ReLU(inplace=True)
        self.latent_dim = latent_dim
        self.use_conditioning = use_conditioning
        if latent_dim > 0:
            module_list = [FCBlock(in_channels + int(use_conditioning) * conditioning_channels, hidden_channels, batchnorm=batchnorm, activation=activation, dropout=dropout)]
            for i in range(num_layers):
                module_list.append(FCResBlock(hidden_channels, hidden_channels, batchnorm=batchnorm, activation=activation, dropout=dropout))
            module_list.append(nn.Linear(hidden_channels, 2*latent_dim))
            self.fc_layers = nn.Sequential(*module_list)
        else:
            self.fc_layers = None

    def forward(self, x, c):
        if self.use_conditioning:
            x = torch.cat((x, c), dim=-1)
        if self.latent_dim > 0:
            out = self.fc_layers(x)
            mu = out[:, :self.latent_dim]
            log_sigma = out[:, self.latent_dim:]
        else:
            mu = torch.empty(x.shape[0], 0)
            log_sigma = torch.empty(x.shape[0], 0)
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

# @MODEL_REGISTRY
class GroupVAEEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, latent_dim, groups, conditioning_channels, use_conditioning=True, num_layers=3, batchnorm=True, dropout=False):
        super(GroupVAEEncoder, self).__init__()
        activation=nn.ReLU(inplace=True)
        self.latent_dim = latent_dim
        self.groups = groups
        self.use_conditioning = use_conditioning
        if latent_dim > 0:
            module_list = [GroupFCBlock(in_channels + int(use_conditioning) * conditioning_channels, hidden_channels, groups=groups, batchnorm=batchnorm, activation=activation, dropout=dropout)]
            for i in range(num_layers):
                module_list.append(GroupFCResBlock(hidden_channels, hidden_channels, groups=groups, batchnorm=batchnorm, activation=activation, dropout=dropout))
            module_list.append(GroupLinear(hidden_channels, 2*latent_dim, groups))
            self.fc_layers = nn.Sequential(*module_list)
        else:
            self.fc_layers = None

    def forward(self, x, c):
        if self.use_conditioning:
            x = torch.cat((x, c), dim=1)
        if self.latent_dim > 0:
            out = self.fc_layers(x)
            mu = out[:, :self.latent_dim]
            log_sigma = out[:, self.latent_dim:]
        else:
            mu = torch.empty(x.shape[0], 0, self.groups)
            log_sigma = torch.empty(x.shape[0], 0, self.groups)
        return mu, log_sigma

# @MODEL_REGISTRY
class GroupFCEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, groups, conditioning_channels, use_conditioning=True, num_layers=3, batchnorm=True, dropout=False):
        super(GroupFCEncoder, self).__init__()
        activation=nn.ReLU(inplace=True)
        self.out_channels = out_channels
        self.groups = groups
        self.use_conditioning = use_conditioning
        if out_channels > 0:
            module_list = [GroupFCBlock(in_channels + int(use_conditioning) * conditioning_channels, hidden_channels, groups=groups, batchnorm=batchnorm, activation=activation, dropout=dropout)]
            for i in range(num_layers):
                module_list.append(GroupFCResBlock(hidden_channels, hidden_channels, groups=groups, batchnorm=batchnorm, activation=activation, dropout=dropout))
            module_list.append(GroupLinear(hidden_channels, out_channels, groups))
            self.fc_layers = nn.Sequential(*module_list)
        else:
            self.fc_layers = None

    def forward(self, x, c):
        if self.use_conditioning:
            x = torch.cat((x, c), dim=1)
        return self.fc_layers(x)