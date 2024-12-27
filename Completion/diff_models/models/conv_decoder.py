# Local Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from ..models.layers import ResnetBlockFC
from ..utils.convonet_common import normalize_coordinate, normalize_3d_coordinate, map2local
from einops import rearrange, repeat, reduce

class ConvONetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

# @MODEL_REGISTRY
class LocalDecoder(ConvONetDecoder):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out


# @MODEL_REGISTRY
class LocalGridDecoder(ConvONetDecoder):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, z, **kwargs):
        if self.c_dim != 0:
            c = self.sample_grid_feature(p, z) # standardize the name convetion
            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out


# @MODEL_REGISTRY
class LocalDecoderAttn(ConvONetDecoder):
    """ ConvONet Decoder with cross attention to fuse features.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, num_attn_layers=2, n_attn_head=8):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
        encoder_layer = nn.TransformerEncoderLayer(d_model=c_dim, nhead=n_attn_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane, **kwargs):
        """
        p: (B, N, dim)
        c_plane: {'grid': (b, c, v, v, v), 'grid'}
        """
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                sampled_z = self.sample_grid_feature(p, c_plane['grid'])
                sampled_c = self.sample_grid_feature(p, c_plane['grid_c'])
                attn_in = rearrange([sampled_z, sampled_c], "u b f p -> u (b p) f")
                attn_out = self.transformer_encoder(attn_in)
                c += rearrange(attn_out, 'u (b p) f -> u b f p', b=p.shape[0])[0]
                
            if 'xz' in plane_type:
                sampled_z = self.sample_plane_feature(p, c_plane['xz'])
                sampled_c = self.sample_plane_feature(p, c_plane['xz_c'])
                attn_in = rearrange([sampled_z, sampled_c], "u b f p -> u (b p) f")
                attn_out = self.transformer_encoder(attn_in)
                c += rearrange(attn_out, 'u (b p) f -> u b f p', b=p.shape[0])[0]
                
            if 'xy' in plane_type:
                sampled_z = self.sample_plane_feature(p, c_plane['xy'])
                sampled_c = self.sample_plane_feature(p, c_plane['xy_c'])
                attn_in = rearrange([sampled_z, sampled_c], "u b f p -> u (b p) f")
                attn_out = self.transformer_encoder(attn_in)
                c += rearrange(attn_out, 'u (b p) f -> u b f p', b=p.shape[0])[0]

            if 'yz' in plane_type:
                sampled_z = self.sample_plane_feature(p, c_plane['yz'])
                sampled_c = self.sample_plane_feature(p, c_plane['yz_c'])
                attn_in = rearrange([sampled_z, sampled_c], "u b f p -> u (b p) f")
                attn_out = self.transformer_encoder(attn_in)
                c += rearrange(attn_out, 'u (b p) f -> u b f p', b=p.shape[0])[0]

            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out


class LocalDecoderMultiAttn(ConvONetDecoder):
    """ ConvONet Decoder with cross attention to fuse features.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(self, dim=3, c_dim=128, hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1,
            num_attn_layers=2, n_attn_head=8, plane_type=['xy', 'xz', 'yz']):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
        encoder_layer = nn.TransformerEncoderLayer(d_model=c_dim, nhead=n_attn_head)
        self.transformer_encoders = nn.ModuleDict({k: nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers) for k in plane_type})


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane, **kwargs):
        """
        p: (B, N, dim)
        c_plane: {'grid': (b, c, v, v, v), 'grid'}
        """
        if self.c_dim != 0:
            
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                sampled_z = self.sample_grid_feature(p, c_plane['grid'])
                sampled_c = self.sample_grid_feature(p, c_plane['grid_c'])
                attn_in = rearrange([sampled_z, sampled_c], "u b f p -> u (b p) f")
                attn_out = self.transformer_encoders['grid'](attn_in)
                c += rearrange(attn_out, 'u (b p) f -> u b f p', b=p.shape[0])[0]
                
            if 'xz' in plane_type:
                sampled_z = self.sample_plane_feature(p, c_plane['xz'])
                sampled_c = self.sample_plane_feature(p, c_plane['xz_c'])
                attn_in = rearrange([sampled_z, sampled_c], "u b f p -> u (b p) f")
                attn_out = self.transformer_encoders['xz'](attn_in)
                c += rearrange(attn_out, 'u (b p) f -> u b f p', b=p.shape[0])[0]
                
            if 'xy' in plane_type:
                sampled_z = self.sample_plane_feature(p, c_plane['xy'])
                sampled_c = self.sample_plane_feature(p, c_plane['xy_c'])
                attn_in = rearrange([sampled_z, sampled_c], "u b f p -> u (b p) f")
                attn_out = self.transformer_encoders['xy'](attn_in)
                c += rearrange(attn_out, 'u (b p) f -> u b f p', b=p.shape[0])[0]
            if 'yz' in plane_type:
                sampled_z = self.sample_plane_feature(p, c_plane['yz'])
                sampled_c = self.sample_plane_feature(p, c_plane['yz_c'])
                attn_in = rearrange([sampled_z, sampled_c], "u b f p -> u (b p) f")
                attn_out = self.transformer_encoders['yz'](attn_in)
                c += rearrange(attn_out, 'u (b p) f -> u b f p', b=p.shape[0])[0]

            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out
        