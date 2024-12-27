import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from abc import ABC, abstractmethod
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

class DecoderBase(nn.Module):

    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
    '''

    def __init__(self, dim: int = 3, z_dim: int = 128, c_dim: int = 128,
            hidden_size: int = 256):
        super().__init__()

    def forward(self, p, z, c, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out

class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 3, 1) #Change to 3 as we output the deformation

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
#         batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.transpose(1, 2)

        return out


class DecoderMLP(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3*6890, z_dim=128, c_dim=128,
                 hidden_size=1024):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_c = nn.Linear(c_dim, hidden_size)
        self.actvn = F.relu

        self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, dim),
                )


    def forward(self, p, z, c, **kwargs):
#        p = p.transpose(1, 2)
#         batch_size, D, T = p.size()
        net = self.fc_c(c)

        if self.z_dim != 0:
            net_z = self.fc_z(z)
            net = net + net_z

        out = self.mlp(net)
        out = out.reshape(net.shape[0], -1, 3)

        return out

@MODEL_REGISTRY
class DecoderCBatchNormOcc(DecoderBase):
    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
#         batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out

@MODEL_REGISTRY
class DecoderCBatchNormCatZ(DecoderBase):
    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            c_dim = z_dim + c_dim

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2) # BxNx3 to Bx3xN
#         batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            c = torch.cat([z, c], dim=-1)

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out

class DecoderTransformerCBatchNormOcc(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, c_hidden_size=128, ff_dim=256, trans_nl=3, nhead=8, attn_idx=0, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, c_hidden_size)

        self.fc_c = nn.Linear(c_dim, c_hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(c_hidden_size, nhead, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_nl)
        self.attn_idx = attn_idx

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_hidden_size, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_hidden_size, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_hidden_size, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_hidden_size, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_hidden_size, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_hidden_size, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_hidden_size, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
#         batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            z_hat = self.fc_z(z)
            c_hat = self.fc_c(c)
            c = self.transformer_encoder(torch.stack([z_hat, c_hat]))[self.attn_idx]

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out

class ConvT3DDecoder(nn.Module):
    """Decoder from DMC but only estimates occupancy"""
    def __init__(self, z_dim=128, c_dim=32, W=32, H=32, D=32):
        super(ConvT3DDecoder, self).__init__()
        self.W = W
        self.H = H
        self.D = D
        self.in_dim = z_dim + c_dim

        self.actvn = nn.ReLU()

        # decoder
        self.deconv4 = nn.Conv3d(self.in_dim, 128, 1)
        self.deconv3_1 = nn.ConvTranspose3d(128, 128, 3, stride=2, padding=1)
        self.deconv3_2 = nn.ConvTranspose3d(128, 64, 3, padding=1)
        self.deconv2_occ_1 = nn.ConvTranspose3d(64, 64, 3, stride=2, padding=1)
        self.deconv2_occ_2 = nn.ConvTranspose3d(64, 32, 3, padding=1)
        self.deconv1_occ_1 = nn.ConvTranspose3d(32, 32, 3, stride=2, padding=1)
        self.deconv1_occ_2 = nn.ConvTranspose3d(32, 1, 3, padding=3)

        # batchnorm
        self.deconv4_bn = nn.BatchNorm3d(128)
        self.deconv3_1_bn = nn.BatchNorm3d(128)
        self.deconv3_2_bn = nn.BatchNorm3d(64)
        self.deconv2_occ_1_bn = nn.BatchNorm3d(64)
        self.deconv2_occ_2_bn = nn.BatchNorm3d(32)
        self.deconv1_occ_1_bn = nn.BatchNorm3d(32)

    def forward(self, z, c):
        x = torch.cat([z, c], dim=-1) # einops doesn't support this yet
        x = rearrange(x, 'b c -> b c 1 1 1')
        #
        x = self.actvn(self.deconv4_bn(self.deconv4(x)))
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        #
        x = self.actvn(self.deconv3_1_bn(self.deconv3_1(x)))
        x = self.actvn(self.deconv3_2_bn(self.deconv3_2(x)))

        #
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.actvn(self.deconv2_occ_1_bn(self.deconv2_occ_1(x)))
        x = self.actvn(
            self.deconv2_occ_2_bn(self.deconv2_occ_2(x)))

        #
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.actvn(
            self.deconv1_occ_1_bn(self.deconv1_occ_1(x)))
        x = self.deconv1_occ_2(x)

        x = F.interpolate(x, size=(self.D, self.H, self.W), mode='trilinear', align_corners = False)
        x = rearrange(x, 'b 1 d h w -> b d h w')

        return x

class ConvT3DDecoder256(nn.Module):
    """Modified from DMC but only estimates occupancy"""
    def __init__(self, z_dim=256, c_dim=32, D=64, H=64, W=64):
        super(ConvT3DDecoder256, self).__init__()
        self.D = D
        self.H = H
        self.W = W
        self.in_dim = z_dim + c_dim

        self.net = nn.Sequential(
            nn.Conv3d(self.in_dim, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.ConvTranspose3d(32, 32, 3, stride=2, padding=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 1, 3, padding=3),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=-1) # einops doesn't support this yet
        x = rearrange(x, 'b c -> b c 1 1 1')

        x = self.net(x)
#
        x = F.interpolate(x, size=(self.D, self.H, self.W), mode='trilinear', align_corners = False)
        x = rearrange(x, 'b 1 d h w -> b d h w')

        return x

class ConvT3DDecoder128(nn.Module):
    """Modified from DMC but only estimates occupancy"""
    def __init__(self, z_dim=256, c_dim=32, D=128, H=128, W=128):
        super(ConvT3DDecoder128, self).__init__()
        self.D = D
        self.H = H
        self.W = W
        self.in_dim = z_dim + c_dim

        self.net = nn.Sequential(
            nn.Conv3d(self.in_dim, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose3d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 1, 3, padding=3),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=-1) # einops doesn't support this yet
        x = rearrange(x, 'b c -> b c 1 1 1')

        x = self.net(x)
#
        x = F.interpolate(x, size=(self.D, self.H, self.W), mode='trilinear', align_corners = False)
        x = rearrange(x, 'b 1 d h w -> b d h w')

        return x


@MODEL_REGISTRY
class ONetDecoderConvFmt(DecoderCBatchNormCatZ):
    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__(dim=dim, z_dim=z_dim, c_dim=c_dim, hidden_size=hidden_size, leaky=leaky, legacy=legacy)
        
    def forward(self, p, c_plane, c, **kwargs):
        z = c_plane['global']
        return super().forward(p, z=z, c=c)


@MODEL_REGISTRY
class DetONetDecoder(nn.Module):
    '''
    Official ONet Decoder but with ConvONet interface
    Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out
