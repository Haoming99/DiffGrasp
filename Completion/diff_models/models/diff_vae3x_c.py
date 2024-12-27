import torch
import torch.nn as nn
from einops import rearrange, repeat
from diffusers import AutoencoderKL
from ..utils.mesh import generate_from_latent


class DiffVAE3XC(nn.Module):
    def __init__(self, decoder: nn.Module, local_pointnet: nn.Module, unet3d: nn.Module = None,
                 feat_dim=16, vox_reso=64, latent_dim: int = 4, plane_types=['xy', 'xz', 'yz'],
                 down_block_types: list = ["DownEncoderBlock2D", ],
                 up_block_types: list = ["UpDecoderBlock2D", ],
                 block_out_channels: list = [64, ], layers_per_block: int = 1,
                 act_fn: str = "silu", norm_num_groups: int = 32,
                 sample_size: int = 32, corrupt_mult=0.01, padding=0.1, b_min=-0.5, b_max=0.5):

        super().__init__()

        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.plane_types = plane_types
        self.corrupt_mult = corrupt_mult
        self.vox_reso = vox_reso
        self.padding = padding
        self.b_min = b_min
        self.b_max = b_max

        self.local_pointnet = local_pointnet
        self.num_planes = len(plane_types)
        self.vae = AutoencoderKL(in_channels=feat_dim * self.num_planes, out_channels=feat_dim * self.num_planes,
                                 down_block_types=down_block_types, up_block_types=up_block_types,
                                 block_out_channels=block_out_channels, layers_per_block=layers_per_block,
                                 act_fn=act_fn, latent_channels=latent_dim, norm_num_groups=norm_num_groups,
                                 sample_size=sample_size)
        self.decoder = decoder
        self.unet3d = unet3d

    def encode_inputs(self, partial_pcs, full_pcs=None):
        bs = partial_pcs.shape[0]
        pred_feats = {}
        latent_plane_dim = int(self.vox_reso / 2 ** (len(self.vae.down_block_types) - 1))

        c_local = self.local_pointnet(partial_pcs)
        pred_feats.update({f"{k}_c": v for k, v in c_local.items()})

        if full_pcs is not None:
            x_local = self.local_pointnet(full_pcs)
            stacked_x_local = rearrange([x_local[k] for k in self.plane_types], 'p b f v1 v2 -> b (f p) v1 v2')
            posterior = self.vae.encode(stacked_x_local).latent_dist
            px_z = posterior.sample()
            decoded_feats = self.vae.decode(px_z).sample
            splited_preds = rearrange(decoded_feats, 'b (f p) v1 v2 -> p b f v1 v2', f=self.feat_dim, p=self.num_planes)
            pred_feats.update({k: v for k, v in zip(self.plane_types, splited_preds)})
            for k in pred_feats.keys():
                pred_feats[k] = pred_feats[k] + self.corrupt_mult * torch.randn_like(pred_feats[k])
        else:
            px_z = torch.randn((bs, self.latent_dim, latent_plane_dim, latent_plane_dim)).to(partial_pcs.device)
            decoded_feats = self.vae.decode(px_z).sample
            splited_preds = rearrange(decoded_feats, 'b (f p) v1 v2 -> p b f v1 v2', f=self.feat_dim, p=self.num_planes)
            pred_feats.update({k: v for k, v in zip(self.plane_types, splited_preds)})

        return pred_feats, torch.empty((bs, 0), device=partial_pcs.device)

    def decode(self, p, z, c=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z: dict or tensor for decoder
            c (tensor): latent conditioned code c
        '''
        logits = self.decoder(p, c_plane=z, c=c, **kwargs)
        return logits

    def forward(self, partial_pcs, query_pts):
        pred_feats, _ = self.encode_inputs(partial_pcs)
        pred_occ = self.decode(query_pts, z=pred_feats)
        return pred_occ

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
