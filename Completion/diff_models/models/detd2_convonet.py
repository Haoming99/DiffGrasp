import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import pytorch_lightning as pl
from typing import Union, Optional
import itertools
from utils.eval_utils import compute_iou
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from utils.mesh import generate_from_latent, export_shapenet_samples
from einops import repeat, reduce, rearrange
from models.conv_decoder import ConvONetDecoder
from models.encoder import ResnetPointnet
from models.vaes import VAEEncoder
from models.fcresnet import FCBlock, FCResBlock

@MODEL_REGISTRY
class DetD2ConvONet(pl.LightningModule):
    ''' Occupancy Network class.
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder: ConvONetDecoder, latent_decoder: nn.Module, lr=1e-4, padding=0.1, b_min=-0.5,
                 b_max=0.5,points_batch_size=100000, batch_size=12, test_num_samples=1,
                ptn_out_dim=256, pts_dim=3, ptn_hidden_dim=256,
                batchnorm=True, reduction='sum', interactive_debug=False,
                 dec_pre_path: Optional[str]=None, freeze_decoder=False):
        
        super().__init__()

        self.decoder = decoder # ConvONet decoder
        self.latent_decoder = latent_decoder
        
        self.pointnet = ResnetPointnet(c_dim=ptn_out_dim, dim=pts_dim, hidden_dim=ptn_hidden_dim)

        self.out_dir = None
        self.lr = lr
        self.padding = padding
        self.b_min = b_min
        self.b_max = b_max
        self.points_batch_size = points_batch_size
        self.freeze_decoder = freeze_decoder
        self.batch_size = batch_size
        self.test_num_samples = test_num_samples
        assert reduction in ['sum', 'mean'], f"reduction {reduction} not supported"
        self.reduction = reduction
        self.interactive_debug = interactive_debug
        if dec_pre_path:
            ori_state_dict = torch.load(dec_pre_path)
            self.decoder.load_state_dict({k.lstrip('decoder.'): v for k, v in ori_state_dict['model'].items() if k.startswith('decoder.')})
        

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.
        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        # This is not working for now, prepare demo in the future
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c=c, **kwargs)
        return p_r

    
    def decode(self, p, z: torch.Tensor, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c_plane=z, **kwargs)
        return logits


    def step(self, batch, batch_idx):
        full_pcs = batch['full_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']
                
        decoded_latent = self.encode_inputs(full_pcs) 
        
        pred_occ = self.decode(query_pts, z=decoded_latent) 
        
        recon_loss = F.binary_cross_entropy_with_logits(
                pred_occ, query_occ, reduction='none')

        if self.reduction == 'sum':
            recon_loss = recon_loss.sum(-1).mean()
        elif self.reduction == 'mean':
            recon_loss = recon_loss.mean()
        
        loss = recon_loss

        if self.interactive_debug and (torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss))):
            import ipdb; ipdb.set_trace()

        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()

        logs = {
            "recon_loss": recon_loss,
            "loss": loss,
            'lr': self.lr,
            'iou': iou_pts,
        }

        if self.interactive_debug:
            print(f"loss: {loss.item()}, kl: {kl_loss.item()}")

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
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
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.decode(pi, z=z, c=c)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def encode_inputs(self, full_pcs):
        x = self.pointnet(full_pcs)
        decoded_latent = self.latent_decoder(x)
        return decoded_latent
    
    def generate_mesh(self, batch, batch_idx, max_bs=3, num_samples=1, sample=True, denormalize=False):
        partial_pcs = batch['partial_pcs'] # No difference between partial and complete pcs so far
        bs = partial_pcs.shape[0]
        if max_bs < 0:
            max_bs = bs
        device = self.device

        if denormalize:
            loc = batch['loc'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()

        full_pcs = repeat(batch['full_pcs'], 'b p d -> (b s) p d', s=num_samples).to(device)

        with torch.no_grad():
            decoded_latent= self.encode_inputs(full_pcs)
#             import ipdb; ipdb.set_trace()
            decoded_latent = {k: rearrange(v, '(b s) ... -> b s ...', b=bs, s=num_samples) for k, v in decoded_latent.items()}

            mesh_list = list()
            input_list = list()
            for b_i in range(min(max_bs, bs)):
                cur_mesh_list = list()
                input_list.append(batch['partial_pcs'][b_i])

                state_dict = dict()
                for sample_idx in range(num_samples):
                    cur_decoded_latent = {k: v[b_i, sample_idx, None] for k, v in decoded_latent.items()}
                    generated_mesh = generate_from_latent(self.eval_points, z=cur_decoded_latent, c=None,
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
        if not self.freeze_decoder:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam([{'params': itertools.chain(self.pointnet.parameters(), self.latent_decoder.parameters())},
                             {'params': self.decoder.parameters(), 'lr': 0}], lr=self.lr)
        return optimizer

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir
