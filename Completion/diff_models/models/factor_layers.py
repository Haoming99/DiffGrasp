import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from typing import Union, Optional
from einops import repeat, reduce, rearrange



# @MODEL_REGISTRY
class FactorLayerGridMean(nn.Module):
    def __init__(self, v, Rc, feat_dim, resolutions):
        """
        Same impelmentation as kresc
        """
        super().__init__()
        self.resolutions = resolutions
        self.Rc = Rc
        self.v = v # voxel resolution or plane_dim
        self.feat_dim = feat_dim
        # No params needed
        

    def forward(self, x: dict) -> torch.Tensor:
        grid_feats = x['grid']
        h_feats = reduce(grid_feats, 'b c h w d -> b c h', 'mean')
        w_feats = reduce(grid_feats, 'b c h w d -> b c w', 'mean')
        d_feats = reduce(grid_feats, 'b c h w d -> b c d', 'mean')

        stacked_feats = rearrange([h_feats, w_feats, d_feats], 's b f v -> s b f v')
        squeezed_x = rearrange(stacked_feats, 's b f v -> b (s f) v')

        layered_x = [F.avg_pool1d(squeezed_x, self.v // res) for res in self.resolutions[1:]] # The first reso will be provided by sepertate global pointnet
        layered_x = [repeat(cur_x, 'b (s f) vv -> b f (s rc vv)', s=3, rc=self.Rc, f=self.feat_dim) for cur_x in layered_x]

        return layered_x


# @MODEL_REGISTRY
class FactorLayerMean(nn.Module):
    def __init__(self, v, Rc, feat_dim, resolutions):
        super().__init__()
        self.resolutions = resolutions
        self.Rc = Rc
        self.v = v # voxel resolution or plane_dim
        self.feat_dim = feat_dim
        # No params needed
        

    def forward(self, x: dict) -> torch.Tensor:
        dx_feats = rearrange([x['xy'], x['xz']], 's b f x v -> b f x (s v)')
        dy_feats = rearrange([rearrange(x['xy'], 'b f x y -> b f y x'), x['yz']], 's b f y v -> b f y (s v)')
        dz_feats = rearrange([x['xz'], x['yz']], 's b f v z -> b f z (s v)')

        h_feats = reduce(dx_feats, 'b f x sv -> b f x', 'mean')
        w_feats = reduce(dy_feats, 'b f y sv -> b f y', 'mean')
        d_feats = reduce(dz_feats, 'b f z sv -> b f z', 'mean')

        squeezed_x = rearrange([h_feats, w_feats, d_feats], 's b f v -> b (s f) v')

        layered_x = [F.avg_pool1d(squeezed_x, self.v // res) for res in self.resolutions[1:]] # The first reso will be provided by sepertate global pointnet
        layered_x = [repeat(cur_x, 'b (s f) vv -> b f (s rc vv)', s=3, rc=self.Rc, f=self.feat_dim) for cur_x in layered_x]

        return layered_x


# @MODEL_REGISTRY
class FactorLayerLinear(nn.Module):
    def __init__(self, v, Rc, feat_dim, resolutions, num_layers=2):
        super().__init__()
        self.resolutions = resolutions
        self.Rc = Rc
        self.v = v # voxel resolution or plane_dim
        self.feat_dim = feat_dim
        
        in_channels = self.v * self.feat_dim * 2
        out_channels = self.Rc * self.feat_dim

        module_list = []
        for _ in range(num_layers):
            module_list.append(nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding='same'))
            module_list.append(nn.ReLU(inplace=True))
        module_list.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding='same'))
        self.fc_layers = nn.Sequential(*module_list)

    def forward(self, x: dict) -> torch.Tensor:
        bs = x['xy'].shape[0]
        
        dx_feats = rearrange([x['xy'], x['xz']], 's b f x v -> b (v s f) x') # feats on the x direction
        dy_feats = rearrange([rearrange(x['xy'], 'b f x y -> b f y x'), x['yz']], 's b f y v -> b (v s f) y')
        dz_feats = rearrange([x['xz'], x['yz']], 's b f v z -> b (v s f) z')

        batched_feats = rearrange([dx_feats, dy_feats, dz_feats],
                          'ss b (v1 s f) v2 -> (ss b) (v1 s f) v2', v1=self.v, s=2, f=self.feat_dim)
        projected_feats = self.fc_layers(batched_feats)
        
        squeezed_x = rearrange(projected_feats, '(s b) (rc f) v -> b (s rc f) v', s=3, b=bs, rc=self.Rc, f=self.feat_dim)

        layered_x = [F.avg_pool1d(squeezed_x, self.v // res) for res in self.resolutions[1:]] # The first reso will be provided by sepertate global pointnet
        layered_x = [rearrange(cur_x, 'b (s rc f) vv -> b f (s rc vv)', s=3, rc=self.Rc, f=self.feat_dim) for cur_x in layered_x]
        
        return layered_x