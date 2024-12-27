import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

@MODEL_REGISTRY
class AutoDecoder(nn.Module):
    
    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=8, append_latent=[4], dropout=True, weight_norm=True, dropout_p=0.2, layer_norm=True,
                spatial_dim=3, Rc=10, vox_reso=32, feat_dim=32):
        super().__init__()
        layer_initer = nn.utils.weight_norm if weight_norm else lambda x: x
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.append_latent = append_latent
        self.layer_norm = layer_norm
        self.spatial_dim = spatial_dim
        self.Rc = Rc
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        
        out_feat_dim = spatial_dim * vox_reso * feat_dim * Rc        
        
        for layer_idx in range(1, num_layers):
            in_dim = hidden_dim if layer_idx > 1 else latent_dim
            out_dim = hidden_dim if layer_idx not in append_latent else hidden_dim - latent_dim
            setattr(self, f"lin{layer_idx}", layer_initer(nn.Linear(in_dim, out_dim)))
            if layer_norm:
                setattr(self, f"bn{layer_idx}", nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout_p) if dropout else None
        self.relu = nn.ReLU()
        
        setattr(self, f"lin{num_layers}", layer_initer(nn.Linear(hidden_dim, out_feat_dim)))
        
    def forward(self, z):
        """
        z: (bn, f) tensor
        c: c will not be used
        """
        x = z
        for layer_idx in range(1, self.num_layers):
            lin = getattr(self, f"lin{layer_idx}")
            x = lin(x)
            
            if layer_idx in self.append_latent:
                x = torch.cat([x, z], dim=1)

            if self.layer_norm:
                bn = getattr(self, f"bn{layer_idx}")
                x = bn(x)
            x = self.relu(x)
            
            if self.dropout is not None:
                x = self.dropout(x)
        lin = getattr(self, f"lin{self.num_layers}")
        x = lin(x)
        fvx, fvy, fvz = rearrange(
            x, 'b (s rc v f) -> s b rc v f', s=self.spatial_dim, rc=self.Rc, v=self.vox_reso, f=self.feat_dim)
        pred_feat_grid = torch.einsum('brif, brjf, brkf -> bfijk', fvx, fvy, fvz)
        return {'grid': pred_feat_grid}