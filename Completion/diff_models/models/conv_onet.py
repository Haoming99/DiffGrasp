import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import pytorch_lightning as pl
from typing import Union, Optional
from utils.eval_utils import compute_iou
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from utils.mesh import generate_from_latent, export_shapenet_samples
from einops import repeat, reduce, rearrange
from .conv_decoder import ConvONetDecoder

@MODEL_REGISTRY
class ConvONet(pl.LightningModule):
    ''' Occupancy Network class.
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder: ConvONetDecoder, encoder: Optional[nn.Module], lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5,points_batch_size=100000, batch_size=12, test_num_samples=1):
        super().__init__()

        self.decoder = decoder

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None
        self.out_dir = None
        self.lr = lr
        self.padding = padding
        self.b_min = b_min
        self.b_max = b_max
        self.points_batch_size = points_batch_size
        self.batch_size = batch_size
        self.test_num_samples = test_num_samples

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.
        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c=c, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.
        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c: dict, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (dict): latent conditioned code c
        '''

        logits = self.decoder(p, c_plane=c, **kwargs)
        return logits

    def _run_step(self, c, pts):
        c = self.encode_inputs(c)

        dec_out = self.decode(pts, c)
        return dec_out

    def step(self, batch, batch_idx):
        c, x = batch['partial_pcs'], batch['full_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']
        # Just use full pcs now to reproduce conv_onet
        bs = x.shape[0]

        pred_occ = self._run_step(c, query_pts)

        recon_loss = F.binary_cross_entropy_with_logits(
            pred_occ, query_occ, reduction='none').sum(-1).mean()
        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()

        loss = recon_loss

        logs = {
            "recon_loss": recon_loss,
            "loss": loss,
            'lr': self.lr,
            'iou': iou_pts,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        #self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, logs = self.step(batch, batch_idx)

        self.log_dict({f"val/{k}": v for k, v in logs.items()}, batch_size=self.batch_size)
        return loss

    def eval_points(self, p, z=None, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        # p = self.query_embeder(p)
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.decode(pi, z=z, c=c)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat


    def generate_mesh(self, batch, batch_idx, max_bs=3, num_samples=1, sample=True, denormalize=False):
        partial_pcs = batch['partial_pcs'] # No difference between partial and complete pcs so far
        bs = partial_pcs.shape[0]
        if max_bs < 0:
            max_bs = bs
        device = self.device

        if denormalize:
            loc = batch['loc'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()

        partial_pcs = repeat(partial_pcs, 'b p d -> (b s) p d', s=num_samples).to(device)
        x = None if sample else repeat(batch['full_pcs'], 'b p d -> (b s) p d', s=num_samples).to(device)

        with torch.no_grad():
            c = self.encode_inputs(partial_pcs)
#             import ipdb; ipdb.set_trace()
            # c is a dict of tensor now
            #c = rearrange(c, '(b s) c -> b s c', b=bs, s=num_samples)
            c = {k: rearrange(v, '(b s) ... -> b s ...', b=bs, s=num_samples) for k, v in c.items()}

            mesh_list = list()
            input_list = list()
            for b_i in range(min(max_bs, bs)):
                cur_mesh_list = list()
                input_list.append(batch['partial_pcs'][b_i])

                state_dict = dict()
                for sample_idx in range(num_samples):
                    cur_c = {k: v[b_i, sample_idx, None] for k, v in c.items()}
                    generated_mesh = generate_from_latent(self.eval_points, z=None, c=cur_c,
                                                          state_dict=state_dict, padding=self.padding,
                                                          B_MAX=self.b_max, B_MIN=self.b_min, device=self.device)
                    cur_mesh_list.append(generated_mesh)
                    if denormalize:
                        generated_mesh.vertices = (generated_mesh.vertices + loc[b_i]) * scale[b_i]
                mesh_list.append(cur_mesh_list)

        return mesh_list, input_list

    def test_step(self, batch, batch_idx):
        mesh_list, _ = self.generate_mesh(batch, batch_idx, max_bs=-1, num_samples=self.test_num_samples, sample=True, denormalize=True)
        denormalized_pcs = (batch['partial_pcs'] + batch['loc']) * batch['scale']
        export_shapenet_samples(mesh_list, batch['category'], batch['model'], denormalized_pcs, self.out_dir)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir
