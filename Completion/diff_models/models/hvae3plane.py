from optparse import Option
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import pytorch_lightning as pl
from typing import Union, Optional
import itertools
import numpy as np
import logging
# from models.unet import UNet
from utils.eval_utils import compute_iou

# from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from utils.mesh import generate_from_latent, export_shapenet_samples
from einops import repeat, reduce, rearrange
from models.kilo_base import KiloBase
# from models.conv_decoder import ConvONetDecoder
# from models.encoder import ResnetPointnet
from models.vaes import VAEEncoder, GroupVAEEncoder
# from models.encoder import ResnetPointnet
# from models.local_encoder import LocalPoolPointnet
from models.unet3d import UNet3DCLI
from models.fcresnet import GroupLinear, GroupFCBlock, GroupFCResBlock, GroupDoubleFCLayer

# from torch_ema import ExponentialMovingAverage


# @MODEL_REGISTRY
class HVAE3Plane(KiloBase):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(
        self,
        decoder: nn.Module,
        global_pointnet: nn.Module,
        local_pointnet: nn.Module,
        global_vae_encoder: nn.Module,
        global_prior_encoder: nn.Module,
        unet3d: UNet3DCLI = None,
        with_proj=True,
        res_after_type=None,
        feat_dim=16,
        vox_reso=16,
        kl_weight=0.01,
        local_kl_weight=0.01,
        recon_weight=1.0,
        tv_weight=1.0,
        reg_weight=0.01,
        edr_weight=1.0,
        Rc=10,
        global_latent_dim=16,
        local_cond_dim=16,
        vd_init=False,
        fuse_feats=False,
        dec_pre_path: Optional[str] = None,
        residual_latents=True,
        resolutions=[1, 4, 8, 16],
        lr=1e-4,
        padding=0.1,
        b_min=-0.5,
        b_max=0.5,
        points_batch_size=100000,
        batch_size=12,
        test_num_samples=1,
        batchnorm=True,
        dropout=False,
        reduction="sum",
        invis_loss_weight=1.0,
        interactive_debug=False,
        freeze_decoder=False,
        test_num_pts=2048,
        ed_pre_path=None,
        attn_grid=False,
        ende_lr=None,
        num_fc_layers=3,
        impute_cond=False,
        kl_reg_loss_weight=0,
        temperature=1.0,
    ):
        super().__init__(
            lr=lr,
            padding=padding,
            b_min=b_min,
            b_max=b_max,
            points_batch_size=points_batch_size,
            batch_size=batch_size,
            test_num_samples=test_num_samples,
            batchnorm=batchnorm,
            reduction=reduction,
            invis_loss_weight=invis_loss_weight,
            interactive_debug=interactive_debug,
            freeze_decoder=freeze_decoder,
            test_num_pts=test_num_pts,
        )

        self.decoder = decoder  # ConvONet decoder
        self.unet3d = unet3d  # 3D UNet after VAE sampling

        # For backward compatibility
        global_latent_dim = feat_dim if global_latent_dim is None else global_latent_dim

        self.global_pointnet = global_pointnet
        self.local_pointnet = local_pointnet
        self.impute_cond = impute_cond

        # Init the global vae and local vaes, consider use index
        self.global_vae_encoder = global_vae_encoder
        self.global_prior_encoder = global_prior_encoder

        self.save_hyperparameters(
            ignore=[
                "decoder",
                "unet3d",
                "global_vae_encoder",
                "global_prior_encoder",
                "global_pointnet",
                "local_pointnet",
            ]
        )

        self.kl_weight = kl_weight
        self.local_kl_weight = local_kl_weight
        self.recon_weight = recon_weight
        self.invis_loss_weight = invis_loss_weight
        self.tv_weight = tv_weight
        self.reg_weight = reg_weight
        self.edr_weight = edr_weight
        self.vox_reso = vox_reso
        self.feat_dim = feat_dim
        self.Rc = Rc
        self.residual_latents = residual_latents
        self.resolutions = resolutions
        self.global_latent_dim = global_latent_dim
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.fuse_feats = fuse_feats
        self.ed_pre_path = ed_pre_path
        self.attn_grid = attn_grid
        self.ende_lr = ende_lr
        self.kl_reg_loss_weight = kl_reg_loss_weight
        self.register_buffer(
            "log_temperature",
            rearrange(torch.tensor([temperature]).log(), "1 -> 1 1 1"),
            persistent=False,
        )
        self.pretrained_keys = []  # Empty until we configure the optimizer
        # Note: None and 0 are different. None means training from scratch and 0 means freeze the model

        assert res_after_type in [
            None,
            "single",
            "double",
            "single_linear",
            "single_relu",
        ], f"Unsupported res_after_type: {res_after_type}"

        self.local_vae_encoders = nn.ModuleList(
            [
                GroupVAEEncoder(
                    in_channels=self.feat_dim,
                    hidden_channels=self.global_latent_dim * 2,
                    latent_dim=self.global_latent_dim,
                    groups=self.Rc * cur_reso * 3,
                    use_conditioning=True,
                    conditioning_channels=self.global_latent_dim,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    num_layers=num_fc_layers,
                )
                for cur_reso in resolutions[1:]
            ]
        )
        self.local_prior_encoders = nn.ModuleList(
            [
                GroupVAEEncoder(
                    in_channels=self.global_latent_dim,
                    hidden_channels=self.global_latent_dim * 2,
                    latent_dim=self.global_latent_dim,
                    groups=self.Rc * cur_reso * 3,
                    use_conditioning=False,
                    conditioning_channels=0,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    num_layers=num_fc_layers,
                )
                for cur_reso in resolutions[1:]
            ]
        )
        if with_proj:
            self.proj_blocks = nn.ModuleList(
                [
                    GroupLinear(
                        self.global_latent_dim,
                        self.global_latent_dim,
                        groups=self.Rc * cur_reso * 3,
                    )
                    for cur_reso in resolutions[1:]
                ]
            )
        else:
            self.proj_blocks = None

        if res_after_type == "single":
            self.res_after = nn.ModuleList(
                [
                    GroupFCBlock(
                        self.global_latent_dim,
                        self.global_latent_dim,
                        groups=self.Rc * cur_reso * 3,
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                    )
                    for cur_reso in resolutions[1:]
                ]
            )  # This is a bug as we introduced activation before addition
        elif res_after_type == "double":
            self.res_after = nn.ModuleList(
                [
                    GroupDoubleFCLayer(
                        self.global_latent_dim,
                        self.global_latent_dim,
                        groups=self.Rc * cur_reso * 3,
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                    )
                    for cur_reso in resolutions[1:]
                ]
            )
        elif res_after_type == "single_linear" or res_after_type == "single_relu":
            self.res_after = nn.ModuleList(
                [
                    GroupLinear(
                        self.global_latent_dim,
                        self.global_latent_dim,
                        groups=self.Rc * cur_reso * 3,
                    )
                    for cur_reso in resolutions[1:]
                ]
            )
        else:
            self.res_after = None

        if vd_init:
            if self.proj_blocks is not None:
                for i, block in enumerate(self.proj_blocks):
                    block.weight.data *= np.sqrt(1 / (i + 1))
                    block.bias.data *= 0.0

            if self.res_after is not None:
                for i, block in enumerate(self.res_after):
                    block.fc_block[0].weight.data *= np.sqrt(1 / (i + 1))  # block is nn.Sequential
                    block.fc_block[0].bias.data *= 0
                    if res_after_type == "double":
                        block.fc_block[4].weight.data *= np.sqrt(1 / (i + 1))
                        block.fc_block[4].bias.data *= 0.0

        if dec_pre_path:
            ori_state_dict = torch.load(dec_pre_path)
            self.decoder.load_state_dict(
                {
                    k.lstrip("decoder."): v
                    for k, v in ori_state_dict["model"].items()
                    if k.startswith("decoder.")
                }
            )

        if ed_pre_path:
            state_dict = torch.load(ed_pre_path)["state_dict"]
            self.full_local_pointnet.load_state_dict(
                {
                    k[len("full_local_pointnet.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("full_local_pointnet.")
                },
                strict=True,
            )
            self.partial_local_pointnet.load_state_dict(
                {
                    k[len("full_local_pointnet.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("full_local_pointnet.")
                },
                strict=False,
            )
            self.decoder.load_state_dict(
                {
                    k[len("decoder.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("decoder.")
                },
                strict=False,
            )

    def decode(self, p, z, c=None, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z: dict or tensor for decdoer
            c (tensor): latent conditioned code c
        """

        logits = self.decoder(p, c_plane=z, c=c, **kwargs)
        return logits

    def step(self, batch, batch_idx):
        # if self.interactive_debug:
        #     import ipdb; ipdb.set_trace()

        partial_pcs, full_pcs = batch["partial_pcs"], batch["full_pcs"]
        query_pts, query_occ = batch["query_pts"], batch["query_occ"]

        x_local = self.local_pointnet(full_pcs)
        x_global = self.global_pointnet(full_pcs)

        c_global = self.global_pointnet(partial_pcs)
        c_local = self.local_pointnet(partial_pcs)

        #  distribute features to basises with interpolation
        # we are reducing v*2 feature vectors to Rc vectors

        # h_feats = F.avg_pool2d(x_local['xy'], self.vox_reso // self.Rc)
        h_feats = rearrange([x_local['xy'], x_local['xz']], "t b f h w -> b (f h) (w t)")
        h_feats = F.interpolate(h_feats, size=self.Rc) # b (f h) Rc

        w_feats = rearrange([rearrange(x_local['xy'], "b f h w -> b f w h" ), x_local['yz']], "t b f w h -> b (f w) (h t)")
        w_feats = F.interpolate(w_feats, size=self.Rc) # b (f h) Rc

        d_feats = rearrange([rearrange(x_local['xz'], "b f h d -> b f d h" ), rearrange(x_local['yz'], 'b f w d -> b f d w')], "t b f d w -> b (f d) (w t)")
        d_feats = F.interpolate(d_feats, size=self.Rc) # b (f h) Rc

        squeezed_x = rearrange([h_feats, w_feats, d_feats], "s b (f v) rc -> b (s f rc) v", f=self.feat_dim, rc=self.Rc)

        layered_x = [
            F.avg_pool1d(squeezed_x, self.vox_reso // res) for res in self.resolutions[1:]
        ]  # The first reso will be provided by sepertate global pointnet
        layered_x = [
            rearrange(cur_x, "b (s f rc) vv -> b f (s rc vv)", s=3, rc=self.Rc, f=self.feat_dim)
            for cur_x in layered_x
        ]


        h_feats = rearrange([c_local['xy'], c_local['xz']], "t b f h w -> b (f h) (w t)")
        h_feats = F.interpolate(h_feats, size=self.Rc) # b (f h) Rc

        w_feats = rearrange([rearrange(c_local['xy'], "b f h w -> b f w h" ), c_local['yz']], "t b f w h -> b (f w) (h t)")
        w_feats = F.interpolate(w_feats, size=self.Rc) # b (f h) Rc

        d_feats = rearrange([rearrange(c_local['xz'], "b f h d -> b f d h" ), rearrange(c_local['yz'], 'b f w d -> b f d w')], "t b f d w -> b (f d) (w t)")
        d_feats = F.interpolate(d_feats, size=self.Rc) # b (f h) Rc

        squeezed_c = rearrange([h_feats, w_feats, d_feats], "s b (f v) rc -> b (s f rc) v", f=self.feat_dim, rc=self.Rc)

        layered_c = [
            F.avg_pool1d(squeezed_c, self.vox_reso // res) for res in self.resolutions[1:]
        ]  # The first reso will be provided by sepertate global pointnet
        layered_c = [
            rearrange(cur_c, "b (s f rc) vv -> b f (s rc vv)", s=3, rc=self.Rc, f=self.feat_dim)
            for cur_c in layered_c
        ]

        mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        prior_mu, prior_log_var = self.global_prior_encoder(x=c_global, c=None)
        prior_std = torch.exp(prior_log_var / 2)
        p = torch.distributions.Normal(prior_mu, prior_std)

        if self.kl_reg_loss_weight > 0:
            p0 = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            global_prior_kl_reg_loss = torch.distributions.kl_divergence(p, p0)
            global_posterior_kl_reg_loss = torch.distributions.kl_divergence(q, p0)
            if self.reduction == 'sum':
                global_prior_kl_reg_loss = global_prior_kl_reg_loss.mean()
                global_posterior_kl_reg_loss = global_posterior_kl_reg_loss.mean()
                # Sum to get a (b,) tensor
            elif self.reduction == 'mean':
                global_prior_kl_reg_loss = global_prior_kl_reg_loss.sum(-1).mean()
                global_posterior_kl_reg_loss = global_posterior_kl_reg_loss.sum(-1).mean()

        repeated_latents = repeat(z, "b c -> b c (s rc v)", s=3, rc=self.Rc, v=self.resolutions[0])
        local_kl_losses = list()
        local_prior_kl_reg_losses = list()
        local_posterior_kl_reg_losses = list()

        for idx, (cur_reso, cur_x, cur_c, cur_vae_encoder, cur_prior_encoder) in enumerate(
            zip(self.resolutions[1:], layered_x, layered_c, self.local_vae_encoders, self.local_prior_encoders)
        ):
            repeated_latents = repeat(
                repeated_latents,
                "b c (s rc v) -> b c (s rc v ratio)",
                s=3,
                rc=self.Rc,
                v=self.resolutions[idx],  # last_reso
                ratio=cur_reso // self.resolutions[idx],
            )  # next_reso / cur_reso

            local_mus, local_log_vars = cur_vae_encoder(x=cur_x, c=repeated_latents)
            local_stds = torch.exp(local_log_vars / 2)
            local_qs = torch.distributions.Normal(local_mus, local_stds)

            local_zs = local_qs.rsample()

            local_prior_mus, local_prior_log_vars = cur_prior_encoder(x=repeated_latents, c=cur_c)
            local_prior_stds = torch.exp(local_prior_log_vars / 2)
            local_ps = torch.distributions.Normal(local_prior_mus, local_prior_stds)

            local_kl_losses.append(torch.distributions.kl_divergence(local_qs, local_ps))

            # Add kl regularizations to avoid NaN
            if self.kl_reg_loss_weight > 0:
                p0 = torch.distributions.Normal(torch.zeros_like(local_mus), torch.ones_like(local_stds))
                local_prior_kl_reg_losses.append(torch.distributions.kl_divergence(local_ps, p0))
                local_posterior_kl_reg_losses.append(torch.distributions.kl_divergence(local_ps, p0))


            if self.proj_blocks is not None:
                local_zs = self.proj_blocks[idx](local_zs)

            if self.residual_latents:
                repeated_latents = repeated_latents + local_zs
            else:
                repeated_latents = local_zs

            if self.res_after is not None:
                if self.hparams.res_after_type == "single_relu":
                    repeated_latents = F.relu(
                        repeated_latents + self.res_after[idx](repeated_latents)
                    )
                else:
                    repeated_latents = repeated_latents + self.res_after[idx](repeated_latents)

        fvx, fvy, fvz = rearrange(
            repeated_latents,
            "b f (s rc v) -> s b rc v f",
            s=3,
            rc=self.Rc,
            v=self.vox_reso,
            f=self.feat_dim,
        )
        sampled_feat_grid = torch.einsum("brif, brjf, brkf -> bfijk", fvx, fvy, fvz)

        if self.unet3d is not None:
            sampled_feat_grid = self.unet3d(sampled_feat_grid)

        z_grid = {"grid": sampled_feat_grid}

        if self.impute_cond:
            for k in ['xy', 'xz', 'yz']:
                z_grid[k] = c_local[k]


        pred_occ = self.decode(query_pts, z=z_grid, c=c_global)

        recon_loss = F.binary_cross_entropy_with_logits(pred_occ, query_occ, reduction="none")

        if self.invis_loss_weight != 1.0:  # we can compare with 1. because 1 = 2^0
            query_weight = batch["query_mask"]
            query_weight[query_weight == 0] = self.invis_loss_weight
            recon_loss = recon_loss * query_weight

        if self.reduction == "sum":
            recon_loss = recon_loss.sum(-1).mean()
        elif self.reduction == "mean":
            recon_loss = recon_loss.mean()

        kl_loss = torch.distributions.kl_divergence(q, p)

        if self.reduction == "sum":
            kl_loss = kl_loss.sum(-1).mean()
            local_kl_loss = sum(
                [cur_kl_loss.sum([-1, -2]) for cur_kl_loss in local_kl_losses]
            ).mean()
            # Sum to get a (b,) tensor
        elif self.reduction == "mean":
            kl_loss = kl_loss.mean()
            local_kl_loss = sum(map(torch.mean, local_kl_losses)) / len(local_kl_losses)

        tv_x = ((sampled_feat_grid[:, :, 1:, :, :] - sampled_feat_grid[:, :, :-1, :, :]) ** 2).sum()
        tv_y = ((sampled_feat_grid[:, :, :, 1:, :] - sampled_feat_grid[:, :, :, :-1, :]) ** 2).sum()
        tv_z = ((sampled_feat_grid[:, :, :, :, 1:] - sampled_feat_grid[:, :, :, :, :-1]) ** 2).sum()
        tv_loss = (tv_x + tv_y + tv_z) / (
            sampled_feat_grid.shape[-1] * sampled_feat_grid.shape[-2] * sampled_feat_grid.shape[-3]
        )

        reg_loss = torch.norm(sampled_feat_grid)

        edr_num_points, edr_offset_distance = 10000, 0.01
        initial_coords = (torch.rand(query_pts.shape[0], edr_num_points, 3) * 2 - 1).to(self.device)
        offset_coords = (
            initial_coords + torch.randn_like(initial_coords) * edr_offset_distance
        ).to(self.device)
        edr_initial_pred_occ = self.decode(initial_coords, z=z_grid, c=c_global)
        edr_offset_pred_occ = self.decode(offset_coords, z=z_grid, c=c_global)
        edr_loss = F.mse_loss(edr_initial_pred_occ, edr_offset_pred_occ)

        if self.kl_reg_loss_weight > 0:
            if self.reduction == 'sum':
                kl_loss = kl_loss.sum(-1).mean()
                local_prior_kl_reg_loss = sum([cur_kl_loss.sum([-1, -2]) for cur_kl_loss in local_prior_kl_reg_losses]).mean()
                local_posterior_kl_reg_loss = sum([cur_kl_loss.sum([-1, -2]) for cur_kl_loss in local_posterior_kl_reg_losses]).mean()

                # Sum to get a (b,) tensor
            elif self.reduction == 'mean':
                kl_loss = kl_loss.mean()
                local_prior_kl_reg_loss = sum(map(torch.mean, local_prior_kl_reg_losses)) / len(local_prior_kl_reg_losses)
                local_posterior_kl_reg_loss = sum(map(torch.mean, local_posterior_kl_reg_losses)) / len(local_posterior_kl_reg_losses)
        else:
            local_prior_kl_reg_loss = 0.
            local_posterior_kl_reg_loss = 0.
            global_posterior_kl_reg_loss = 0.
            global_prior_kl_reg_loss = 0.
            

        loss = (
            self.kl_weight * kl_loss
            + self.recon_weight * recon_loss
            + self.local_kl_weight * local_kl_loss
            + self.tv_weight * tv_loss
            + self.reg_weight * reg_loss
            + self.edr_weight * edr_loss
            + self.kl_reg_loss_weight * global_prior_kl_reg_loss
            + self.kl_reg_loss_weight * global_posterior_kl_reg_loss
            + self.kl_reg_loss_weight * local_prior_kl_reg_loss
            + self.kl_reg_loss_weight * local_posterior_kl_reg_loss
        )

        if torch.any(torch.isnan(loss)):
            console_logger = logging.getLogger("pytorch_lightning")
            console_logger.error(
                f"NaN loss encountered: loss: {loss}, kl_loss: {kl_loss}, local_kl_loss: {local_kl_loss}"
            )
            loss = self.kl_weight * kl_loss

        if self.interactive_debug and (
            torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss))
        ):
            import ipdb

            ipdb.set_trace()

        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()

        logs = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "local_kl_loss": local_kl_loss,
            "tv_loss": tv_loss,
            "reg_loss": reg_loss,
            "edr_loss": edr_loss,
            "recon_weight": self.recon_weight,
            "kl_weight": self.kl_weight,
            "local_kl_weight": self.local_kl_weight,
            "tv_weight": self.tv_weight,
            "reg_weight": self.reg_weight,
            "edr_weight": self.edr_weight,
            "local_prior_kl_reg_loss": local_prior_kl_reg_loss,
            "local_posterior_kl_reg_loss":local_posterior_kl_reg_loss,
            "global_posterior_kl_reg_loss": global_posterior_kl_reg_loss,
            "global_prior_kl_reg_loss": global_prior_kl_reg_loss,
            "loss": loss,
            "lr": self.lr,
            "iou": iou_pts,
            "batch_size": float(self.batch_size),
        }

        if self.interactive_debug:
            print(f"loss: {loss.item()}, kl: {kl_loss.item()}")

        return loss, logs

    def encode_inputs(self, partial_pcs, full_pcs=None):
        c_global = self.global_pointnet(partial_pcs)
        c_local = self.local_pointnet(partial_pcs)

        h_feats = rearrange([c_local['xy'], c_local['xz']], "t b f h w -> b (f h) (w t)")
        h_feats = F.interpolate(h_feats, size=self.Rc) # b (f h) Rc

        w_feats = rearrange([rearrange(c_local['xy'], "b f h w -> b f w h" ), c_local['yz']], "t b f w h -> b (f w) (h t)")
        w_feats = F.interpolate(w_feats, size=self.Rc) # b (f h) Rc

        d_feats = rearrange([rearrange(c_local['xz'], "b f h d -> b f d h" ), rearrange(c_local['yz'], 'b f w d -> b f d w')], "t b f d w -> b (f d) (w t)")
        d_feats = F.interpolate(d_feats, size=self.Rc) # b (f h) Rc

        squeezed_c = rearrange([h_feats, w_feats, d_feats], "s b (f v) rc -> b (s f rc) v", f=self.feat_dim, rc=self.Rc)

        layered_c = [
            F.avg_pool1d(squeezed_c, self.vox_reso // res) for res in self.resolutions[1:]
        ]  # The first reso will be provided by sepertate global pointnet
        layered_c = [
            rearrange(cur_c, "b (s f rc) vv -> b f (s rc vv)", s=3, rc=self.Rc, f=self.feat_dim)
            for cur_c in layered_c
        ]



        if full_pcs is not None:
            x_local = self.local_pointnet(full_pcs)
            x_global = self.global_pointnet(full_pcs)

            mu, log_var = self.global_vae_encoder(x=x_global, c=c_global)
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # Process local latents
            h_feats = rearrange([x_local['xy'], x_local['xz']], "t b f h w -> b (f h) (w t)")
            h_feats = F.interpolate(h_feats, size=self.Rc) # b (f h) Rc

            w_feats = rearrange([rearrange(x_local['xy'], "b f h w -> b f w h" ), x_local['yz']], "t b f w h -> b (f w) (h t)")
            w_feats = F.interpolate(w_feats, size=self.Rc) # b (f h) Rc

            d_feats = rearrange([rearrange(x_local['xz'], "b f h d -> b f d h" ), rearrange(x_local['yz'], 'b f w d -> b f d w')], "t b f d w -> b (f d) (w t)")
            d_feats = F.interpolate(d_feats, size=self.Rc) # b (f h) Rc

            squeezed_x = rearrange([h_feats, w_feats, d_feats], "s b (f v) rc -> b (s f rc) v", f=self.feat_dim, rc=self.Rc)

            layered_x = [
                F.avg_pool1d(squeezed_x, self.vox_reso // res) for res in self.resolutions[1:]
            ]  # The first reso will be provided by sepertate global pointnet
            layered_x = [
                rearrange(cur_x, "b (s f rc) vv -> b f (s rc vv)", s=3, rc=self.Rc, f=self.feat_dim)
                for cur_x in layered_x
            ]

            repeated_latents = repeat(
                z, "b c -> b c (s rc v)", s=3, rc=self.Rc, v=self.resolutions[0]
            )

            for idx, (cur_reso, cur_x, cur_vae_encoder) in enumerate(
                zip(self.resolutions[1:], layered_x, self.local_vae_encoders)
            ):
                repeated_latents = repeat(
                    repeated_latents,
                    "b c (s rc v) -> b c (s rc v ratio)",
                    s=3,
                    rc=self.Rc,
                    v=self.resolutions[idx],  # last_reso
                    ratio=cur_reso // self.resolutions[idx],
                )  # next_reso / cur_reso

                local_mus, local_log_vars = cur_vae_encoder(x=cur_x, c=repeated_latents)
                local_stds = torch.exp(local_log_vars / 2)
                local_qs = torch.distributions.Normal(local_mus, local_stds)

                local_zs = local_qs.rsample()

                if self.proj_blocks is not None:
                    local_zs = self.proj_blocks[idx](local_zs)

                if self.residual_latents:
                    repeated_latents = repeated_latents + local_zs
                else:
                    repeated_latents = local_zs

                if self.res_after is not None:
                    if self.hparams.res_after_type == "single_relu":
                        repeated_latents = F.relu(
                            repeated_latents + self.res_after[idx](repeated_latents)
                        )
                    else:
                        repeated_latents = repeated_latents + self.res_after[idx](repeated_latents)

        else:
            prior_mu, prior_log_var = self.global_prior_encoder(x=c_global, c=None)
            prior_log_var += self.log_temperature[0]  # log_temp: (1, 1, 1)

            prior_std = torch.exp(prior_log_var / 2)
            p = torch.distributions.Normal(prior_mu, prior_std)
            z = p.rsample()

            # process local latents

            repeated_latents = repeat(
                z, "b c -> b c (s rc v)", s=3, rc=self.Rc, v=self.resolutions[0]
            )

            for idx, (cur_reso, cur_c, cur_prior_encoder) in enumerate(
                zip(self.resolutions[1:], layered_c, self.local_prior_encoders)
            ):
                repeated_latents = repeat(
                    repeated_latents,
                    "b c (s rc v) -> b c (s rc v ratio)",
                    s=3,
                    rc=self.Rc,
                    v=self.resolutions[idx],  # last_reso
                    ratio=cur_reso // self.resolutions[idx],
                )  # next_reso / cur_reso

                local_prior_mus, local_prior_log_vars = cur_prior_encoder(
                    x=repeated_latents, c=cur_c
                )
                local_prior_log_vars += self.log_temperature
                local_prior_stds = torch.exp(local_prior_log_vars / 2)
                local_ps = torch.distributions.Normal(local_prior_mus, local_prior_stds)

                local_zs = local_ps.rsample()

                if self.proj_blocks is not None:
                    local_zs = self.proj_blocks[idx](local_zs)

                if self.residual_latents:
                    repeated_latents = repeated_latents + local_zs
                else:
                    repeated_latents = local_zs

                if self.res_after is not None:
                    if self.hparams.res_after_type == "single_relu":
                        repeated_latents = F.relu(
                            repeated_latents + self.res_after[idx](repeated_latents)
                        )
                    else:
                        repeated_latents = repeated_latents + self.res_after[idx](repeated_latents)

        fvx, fvy, fvz = rearrange(
            repeated_latents,
            "b f (s rc v) -> s b rc v f",
            s=3,
            rc=self.Rc,
            v=self.vox_reso,
            f=self.feat_dim,
        )
        sampled_feat_grid = torch.einsum("brif, brjf, brkf -> bfijk", fvx, fvy, fvz)

        if self.unet3d is not None:
            sampled_feat_grid = self.unet3d(sampled_feat_grid)

        z_grid = {"grid": sampled_feat_grid}

        if self.impute_cond:
            for k in ['xy', 'xz', 'yz']:
                z_grid[k] = c_local[k]


        return z_grid, c_global

    def get_nondecoder_params(self) -> list:
        # DEPERACTED
        return [
            self.partial_global_pointnet.parameters(),
            self.full_global_pointnet.parameters(),
            self.full_local_pointnet.parameters(),
            self.partial_local_pointnet.parameters(),
            self.global_vae_encoder.parameters(),
            self.global_prior_encoder.parameters(),
            self.local_prior_encoders.parameters(),
            self.local_vae_encoder.parameters(),
        ]

    def split_ende_params(self) -> tuple:
        pretrained_keys = list()
        pretrained_params = list()
        other_params = list()
        for k, v in dict(self.named_parameters()).items():
            if (
                k.startswith("full_local_pointnet")
                or k.startswith("partial_local_pointnet")
                or (k.startswith("decoder.") and not k.startswith("decoder.transformer_encoder."))
            ):
                pretrained_keys.append(k)
                pretrained_params.append(v)
            else:
                other_params.append(v)
        return pretrained_params, other_params

    def configure_optimizers(self):
        if self.ende_lr is not None:
            pretrained_params, other_params = self.split_ende_params()
            optimizer = torch.optim.Adam(
                [{"params": pretrained_params, "lr": self.ende_lr}, {"params": other_params}],
                lr=self.lr,
            )

        elif self.freeze_decoder:
            optimizer = torch.optim.Adam(
                [
                    {"params": itertools.chain(self.get_nondecoder_params())},
                    {"params": self.decoder.parameters(), "lr": 0},
                ],
                lr=self.lr,
            )

        else:  # Default behavior
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
