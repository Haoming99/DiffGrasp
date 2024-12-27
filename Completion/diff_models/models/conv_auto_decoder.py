import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from models.embedder import PositionalEncoding


class ResBlock1D(nn.Module):
    def __init__(self, dim, batch_norm=True, weight_norm=True):
        super().__init__()
        layer_initer = nn.utils.weight_norm if weight_norm else lambda x: x
        self.batch_norm = batch_norm
        
#         self.conv1 = layer_initer(nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding='same', padding_mode='zeros'))
#         self.conv2 = layer_initer(nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding='same', padding_mode='zeros'))
        self.conv1 = layer_initer(nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding='same', padding_mode='reflect'))
        self.conv2 = layer_initer(nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding='same', padding_mode='reflect'))
        self.actvn = nn.ReLU()
        
        self.norm1 = nn.BatchNorm1d(dim) if self.batch_norm else nn.Identity()
        self.norm2 = nn.BatchNorm1d(dim) if self.batch_norm else nn.Identity()
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.norm1(x)   
        x = self.actvn(x)
        
        x = self.conv2(x)
        x = self.norm2(x)   
        x = self.actvn(x)
        
        return x + identity

class ProjBlock1D(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm=True, weight_norm=True):
        super().__init__()
        layer_initer = nn.utils.weight_norm if weight_norm else lambda x: x
        self.batch_norm = batch_norm
        
        self.conv = layer_initer(nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, padding='same'))
        self.actvn = nn.ReLU()
        
        self.norm = nn.BatchNorm1d(out_dim) if self.batch_norm else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.actvn(x)
        return x

@MODEL_REGISTRY
class ConvAutoDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=1024, num_blocks=2, num_bottlenecks=1, 
                 weight_norm=True, batch_norm=True,
            spatial_dim=3, Rc=10, vox_reso=32, feat_dim=32, pe_mode='add'):
        super().__init__()
        
        self.pe_layer = PositionalEncoding(d_model=latent_dim, dropout=0., max_len=128) # Acutally we just use 32 for now
        self.num_blocks = num_blocks
        self.num_bottlenecks = num_bottlenecks
        self.spatial_dim = spatial_dim
        self.Rc = Rc
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        
        assert pe_mode in ['add', 'cat'], f"pe_mode {pe_mode} not supported"
        self.pe_mode = pe_mode
        
        in_dim = spatial_dim  * Rc * latent_dim
        
        if pe_mode == 'cat':
            in_dim *= 2 # Positional encoding is in the same width
        
        self.proj_in1 = ProjBlock1D(in_dim, hidden_dim, batch_norm=batch_norm, weight_norm=weight_norm)
        self.proj_in2 = ProjBlock1D(hidden_dim, hidden_dim//2, batch_norm=batch_norm, weight_norm=weight_norm)
        


        
        in_dim = hidden_dim // 2
        
        for i in range(num_blocks):
            setattr(self, f'block{i}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))
            setattr(self, f'proj{i}', ProjBlock1D(in_dim, in_dim//2, batch_norm=batch_norm, weight_norm=weight_norm))
            in_dim = in_dim // 2
        
        for i in range(num_bottlenecks):
            setattr(self, f'bottleneck{i}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))
        
        for i in range(num_blocks):
            setattr(self, f'proj{i+num_blocks}', ProjBlock1D(in_dim, in_dim*2, batch_norm=batch_norm, weight_norm=weight_norm))
            in_dim = in_dim * 2
            setattr(self, f'block{i+num_blocks}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))            
        
        self.proj_final = nn.Conv1d(in_dim, spatial_dim * Rc * feat_dim, 
                                    kernel_size=1, stride=1, padding='same') 
        
    def forward(self, z, c=None):
        """
        z: latent code to decode
        c: c will not be used, just for back compatbility 
        """
        z_repeated = repeat(z, 'b f -> b v (s rc) f', rc=self.Rc, v=self.vox_reso, s=self.spatial_dim)
        
        if self.pe_mode == 'add':
            pe_batch = rearrange(self.pe_layer.pe[:self.vox_reso], 'v 1 f -> 1 v 1 f')
            z_pe = (z_repeated + pe_batch)
        else:
            bs = z_repeated.shape[0]
            pe_batch = repeat(self.pe_layer.pe[:self.vox_reso], 'v 1 f -> b v (s rc) f',
                              b=bs, v=self.vox_reso, s=self.spatial_dim, rc=self.Rc)
            z_pe = rearrange([z_repeated, pe_batch], 'pe b v (s rc) f -> b v (s rc) (pe f)',
                             s=self.spatial_dim, rc=self.Rc)

        flattened_z = rearrange(z_pe, 'b v (s rc) f -> b (s rc f) v',
                                s=self.spatial_dim, rc=self.Rc, v=self.vox_reso)
        x = flattened_z
        
        x = self.proj_in1(x)
        x = self.proj_in2(x)
        
        for i in range(self.num_blocks):
            block = getattr(self, f'block{i}')
            x = block(x)
            proj = getattr(self, f'proj{i}')
            x = proj(x)
        
        for i in range(self.num_bottlenecks):
            bottleneck = getattr(self, f'bottleneck{i}')
            x = bottleneck(x)
        
        for i in range(self.num_blocks):
            proj = getattr(self, f'proj{i+self.num_blocks}')
            x= proj(x)
            block = getattr(self, f'block{i+self.num_blocks}')
            x = block(x)
        
        x = self.proj_final(x)
        fvx, fvy, fvz = rearrange(x, 'b (s rc f) v -> s b rc v f', s=self.spatial_dim, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        pred_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)
        return {'grid': pred_feat_grid}

@MODEL_REGISTRY
class CondConvAutoDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=1024, num_blocks=2, num_bottlenecks=1, 
                 weight_norm=True, batch_norm=True, cond_dim=256,
            spatial_dim=3, Rc=10, vox_reso=32, feat_dim=32, pe_mode='add', cond_mode='cat'):
        super().__init__()
        
        self.pe_layer = PositionalEncoding(d_model=latent_dim, dropout=0., max_len=128) # Acutally we just use 32 for now
        self.num_blocks = num_blocks
        self.num_bottlenecks = num_bottlenecks
        self.spatial_dim = spatial_dim
        self.Rc = Rc
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        
        assert pe_mode in ['add'], f"pe_mode {pe_mode} not supported"
        assert cond_mode in ['add', 'cat'], f"cond_mode {cond_mode} not supported"
        self.pe_mode = pe_mode
        self.cond_mode = cond_mode

        
        if self.cond_mode == 'cat':
            in_dim = latent_dim + cond_dim

            self.fuse1 = ProjBlock1D(in_dim, in_dim, batch_norm=batch_norm, weight_norm=weight_norm)
            self.fuse2 = ProjBlock1D(in_dim, latent_dim, batch_norm=batch_norm, weight_norm=weight_norm)
        
        in_dim = spatial_dim  * Rc * latent_dim
        
        if pe_mode == 'cat':
            in_dim *= 2 # Positional encoding is in the same width
        
        self.proj_in1 = ProjBlock1D(in_dim, hidden_dim, batch_norm=batch_norm, weight_norm=weight_norm)
        self.proj_in2 = ProjBlock1D(hidden_dim, hidden_dim//2, batch_norm=batch_norm, weight_norm=weight_norm)
        
        
        in_dim = hidden_dim // 2
        
        for i in range(num_blocks):
            setattr(self, f'block{i}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))
            setattr(self, f'proj{i}', ProjBlock1D(in_dim, in_dim//2, batch_norm=batch_norm, weight_norm=weight_norm))
            in_dim = in_dim // 2
        
        for i in range(num_bottlenecks):
            setattr(self, f'bottleneck{i}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))
        
        for i in range(num_blocks):
            setattr(self, f'proj{i+num_blocks}', ProjBlock1D(in_dim, in_dim*2, batch_norm=batch_norm, weight_norm=weight_norm))
            in_dim = in_dim * 2
            setattr(self, f'block{i+num_blocks}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))            
        
        self.proj_final = nn.Conv1d(in_dim, spatial_dim * Rc * feat_dim, 
                                    kernel_size=1, stride=1, padding='same') 
        
    def forward(self, z, c=None):
        """
        z: latent code to decode
        c: c will not be used, just for back compatbility 
        """
        if self.cond_mode == 'cat':
            z = torch.cat([z,  c], dim=1) #can't use inops because num_channels might be diff
            z = rearrange(z, 'b f -> b f 1')
            z = self.fuse2(self.fuse1(z))
        elif self.cond_mode == 'add':
            z = z + c
            z = rearrange(z, 'b f -> b f 1')

        z_repeated = repeat(z, 'b f 1 -> b v (s rc) f', rc=self.Rc, v=self.vox_reso, s=self.spatial_dim)
        
        if self.pe_mode == 'add':
            pe_batch = rearrange(self.pe_layer.pe[:self.vox_reso], 'v 1 f -> 1 v 1 f')
            z_pe = (z_repeated + pe_batch)
        else:
            bs = z_repeated.shape[0]
            pe_batch = repeat(self.pe_layer.pe[:self.vox_reso], 'v 1 f -> b v (s rc) f',
                              b=bs, v=self.vox_reso, s=self.spatial_dim, rc=self.Rc)
            z_pe = rearrange([z_repeated, pe_batch], 'pe b v (s rc) f -> b v (s rc) (pe f)',
                             s=self.spatial_dim, rc=self.Rc)

        flattened_z = rearrange(z_pe, 'b v (s rc) f -> b (s rc f) v',
                                s=self.spatial_dim, rc=self.Rc, v=self.vox_reso)
        x = flattened_z
        
        x = self.proj_in1(x)
        x = self.proj_in2(x)
        
        for i in range(self.num_blocks):
            block = getattr(self, f'block{i}')
            x = block(x)
            proj = getattr(self, f'proj{i}')
            x = proj(x)
        
        for i in range(self.num_bottlenecks):
            bottleneck = getattr(self, f'bottleneck{i}')
            x = bottleneck(x)
        
        for i in range(self.num_blocks):
            proj = getattr(self, f'proj{i+self.num_blocks}')
            x= proj(x)
            block = getattr(self, f'block{i+self.num_blocks}')
            x = block(x)
        
        x = self.proj_final(x)
        fvx, fvy, fvz = rearrange(x, 'b (s rc f) v -> s b rc v f', s=self.spatial_dim, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        pred_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)
        return {'grid': pred_feat_grid}


@MODEL_REGISTRY
class SlimCCAutoDecoder(nn.Module):
    """
    Slimmed convolutional auto decoder for tensor factors estimation
    to use c only: keep latent_dim as default but set cond_mode c_only
    to use z only: set cond_mode to None
    """

    def __init__(self, latent_dim=256, num_down_blocks=3, num_up_blocks=3, num_bottlenecks=1, 
                 weight_norm=True, batch_norm=True, cond_dim=256, spatial_dim=3, Rc=10,
                  vox_reso=32, feat_dim=32, pe_mode='add', cond_mode=None, extra_res=False):
        super().__init__()
        
        self.pe_layer = PositionalEncoding(d_model=latent_dim, dropout=0., max_len=128) # Acutally we just use 32 for now
        self.num_down_blocks = num_down_blocks
        self.num_up_blocks = num_up_blocks
        self.num_bottlenecks = num_bottlenecks
        self.spatial_dim = spatial_dim
        self.Rc = Rc
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        self.extra_res = extra_res

        
        assert pe_mode in ['add'], f"pe_mode {pe_mode} not supported"
        assert cond_mode in ['add', 'cat', 'proj_add', 'c_only', None], f"cond_mode {cond_mode} not supported"
        self.pe_mode = pe_mode
        self.cond_mode = cond_mode

        
        if self.cond_mode == 'cat':
            in_dim = latent_dim + cond_dim

            self.fuse1 = ProjBlock1D(in_dim, in_dim, batch_norm=batch_norm, weight_norm=weight_norm)
            self.fuse2 = ProjBlock1D(in_dim, latent_dim, batch_norm=batch_norm, weight_norm=weight_norm)
            in_dim = latent_dim

        elif self.cond_mode == 'proj_add':
            in_dim = latent_dim
            self.proj_cond1 = ProjBlock1D(cond_dim, in_dim, batch_norm=batch_norm, weight_norm=weight_norm)
            self.proj_cond2 = ProjBlock1D(in_dim, in_dim, batch_norm=batch_norm, weight_norm=weight_norm)
            in_dim = latent_dim
        elif self.cond_mode == 'c_only':
            in_dim = cond_dim
            self.pe_layer = PositionalEncoding(d_model=cond_dim, dropout=0., max_len=128) # Acutally we just use 32 for now
        else:
            in_dim = latent_dim # cond_mode is add or z only
        
        if pe_mode == 'cat':
            in_dim *= 2 # Positional encoding is in the same width
        
        for i in range(self.num_down_blocks):
            setattr(self, f'block{i}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))
            if self.extra_res:
                setattr(self, f'block{i}_ext', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))

            setattr(self, f'proj{i}', ProjBlock1D(in_dim, in_dim//2, batch_norm=batch_norm, weight_norm=weight_norm))
            in_dim = in_dim // 2
        
        for i in range(self.num_bottlenecks):
            setattr(self, f'bottleneck{i}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))
        
        for i in range(self.num_up_blocks):
            setattr(self, f'proj{i+self.num_down_blocks}', ProjBlock1D(in_dim, in_dim*2, batch_norm=batch_norm, weight_norm=weight_norm))
            in_dim = in_dim * 2
            setattr(self, f'block{i+self.num_down_blocks}', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))            
            if self.extra_res:
                setattr(self, f'block{i+self.num_down_blocks}_ext', ResBlock1D(in_dim, batch_norm=batch_norm, weight_norm=weight_norm))
        
        self.proj_final = nn.Conv1d(in_dim, spatial_dim * Rc * feat_dim, 
                                    kernel_size=1, stride=1, padding='same') 
        
    def forward(self, z, c=None):
        """
        z: latent code to decode
        c: c will not be used, just for back compatbility 
        """
        if self.cond_mode == 'cat':
            z = torch.cat([z,  c], dim=1) #can't use inops because num_channels might be diff
            z = rearrange(z, 'b f -> b f 1')
            z = self.fuse2(self.fuse1(z))
        elif self.cond_mode == 'add':
            z = z + c
            z = rearrange(z, 'b f -> b f 1')
        elif self.cond_mode == 'proj_add':
            c = rearrange(c, 'b c_f -> b c_f 1')
            proj_c = self.proj_cond1(c)
            proj_c += self.proj_cond2(c)
            z = rearrange(z, 'b f -> b f 1')
            z = z + proj_c
        elif self.cond_mode == 'c_only':
            z = rearrange(c, 'b c_f -> b c_f 1')
        else:
            z = rearrange(z, 'b f -> b f 1')
            
        
        if self.pe_mode == 'add':
            pe_batch = rearrange(self.pe_layer.pe[:self.vox_reso], 'v 1 f -> 1 f v')
            z_pe = (z + pe_batch) # (b, f, v)
        else:
            bs = z_repeated.shape[0]
            z_repeated = repeat(z, 'b f 1 -> b f v', v=self.vox_reso)
            pe_batch = repeat(self.pe_layer.pe[:self.vox_reso], 'v 1 f -> b f v',
                              b=bs)
            z_pe = rearrange([z_repeated, pe_batch], 'pe b f v -> b (pe f) v',
                             s=self.spatial_dim, rc=self.Rc)
            
        x = z_pe
        
        for i in range(self.num_down_blocks):
            block = getattr(self, f'block{i}')
            x = block(x)
            if self.extra_res:
                block = getattr(self, f'block{i}_ext')
                x = block(x)

            proj = getattr(self, f'proj{i}')
            x = proj(x)
        
        for i in range(self.num_bottlenecks):
            bottleneck = getattr(self, f'bottleneck{i}')
            x = bottleneck(x)
        
        for i in range(self.num_up_blocks):
            proj = getattr(self, f'proj{i+self.num_down_blocks}')
            x= proj(x)
            block = getattr(self, f'block{i+self.num_down_blocks}')
            x = block(x)
            if self.extra_res:
                block = getattr(self, f'block{i+self.num_down_blocks}_ext')
                x = block(x)
        
        x = self.proj_final(x)
        fvx, fvy, fvz = rearrange(x, 'b (s rc f) v -> s b rc v f', s=self.spatial_dim, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        pred_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)
        return {'grid': pred_feat_grid}


@MODEL_REGISTRY
class IdentityLatentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z, c=None):
        return {'global': z}