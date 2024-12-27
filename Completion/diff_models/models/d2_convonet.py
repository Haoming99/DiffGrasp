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
from models.conv_decoder import ConvONetDecoder
from models.encoder import ResnetPointnet
from models.vaes import VAEEncoder

@MODEL_REGISTRY
class D2ConvONet(pl.LightningModule):
    ''' Occupancy Network class.
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder: ConvONetDecoder, latent_decoder: nn.Module, lr=1e-4, padding=0.1, b_min=-0.5,
                 b_max=0.5,points_batch_size=100000, batch_size=12, test_num_samples=1, kl_weight=0.01,
                ptn_out_dim=256, pts_dim=3, ptn_hidden_dim=256, enc_hidden_dim=512, latent_dim=256, 
                recon_weight=1., share_ptn=False, batchnorm=True, reduction='sum',
                invis_loss_weight=1., 
                interactive_debug=False, std=False, bi_kl=False):
        
        super().__init__()

        self.decoder = decoder # ConvONet decoder
        self.latent_decoder = latent_decoder
        
        self.partial_pointnet = ResnetPointnet(c_dim=ptn_out_dim, dim=pts_dim, hidden_dim=ptn_hidden_dim)
        if share_ptn:
            self.full_pointnet = self.partial_pointnet
        else:
            self.full_pointnet = ResnetPointnet(c_dim=ptn_out_dim, dim=pts_dim, hidden_dim=ptn_hidden_dim)
        
        self.encoder = VAEEncoder(ptn_out_dim, enc_hidden_dim, latent_dim, conditioning_channels=ptn_out_dim, batchnorm=batchnorm)
        # x is c and c is None for prior encoder
        self.prior_encoder = VAEEncoder(ptn_out_dim, enc_hidden_dim, latent_dim, conditioning_channels=ptn_out_dim,
                                        use_conditioning=False, batchnorm=batchnorm)

        self.out_dir = None
        self.lr = lr
        self.padding = padding
        self.b_min = b_min
        self.b_max = b_max
        self.points_batch_size = points_batch_size
        self.batch_size = batch_size
        self.test_num_samples = test_num_samples
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.invis_loss_weight = invis_loss_weight 

        assert reduction in ['sum', 'mean'], f"reduction {reduction} not supported"
        self.reduction = reduction
        self.bi_kl = bi_kl # bi directional kl divergence loss ( no steop gradient)
        self.std = std # Use standard normal as prior distribution 
        self.interactive_debug = interactive_debug
        

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

    
    def decode(self, p, z: torch.Tensor, c=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c_plane=z, c=c, **kwargs)
        return logits


    def step(self, batch, batch_idx):
        partial_pcs, full_pcs = batch['partial_pcs'], batch['full_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']
        
        x = self.full_pointnet(full_pcs)
        c = self.partial_pointnet(partial_pcs)
        
        mu, log_var = self.encoder(x=x, c=c)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        
        prior_mu, prior_log_var = self.prior_encoder(x=c, c=None)
        prior_std = torch.exp(prior_log_var / 2)
        p = torch.distributions.Normal(prior_mu, prior_std)
        
        decoded_latent = self.latent_decoder(z, c) 
        # question: do we want to have conditional signal in conv_onet decoder?
        
        pred_occ = self.decode(query_pts, z=decoded_latent, c=c) 
        # c is not guranteed to be used, depending on the implementation of autodecoder
        
        recon_loss = F.binary_cross_entropy_with_logits(
                pred_occ, query_occ, reduction='none')

        if self.invis_loss_weight != 1.: # we can compare with 1. because 1 = 2^0
            query_weight = batch['query_mask']
            query_weight[query_weight == 0] = self.invis_loss_weight
            recon_loss = recon_loss * query_weight

        if self.reduction == 'sum':
            recon_loss = recon_loss.sum(-1).mean()
        elif self.reduction == 'mean':
            recon_loss = recon_loss.mean()
        
        if self.bi_kl:
            kl_loss = torch.distributions.kl_divergence(q, p) 
            # Here q means dist from encoder and p means dist from prior_encoder
            # which is discrepant to most paper
        else:
            q_detached = torch.distributions.Normal(mu.detach(), std.detach())
            if self.std:
                q_detached = torch.distributions.Normal(torch.zeros_like(prior_mu), torch.ones_like(prior_std))
                p  = q # just hack for debugging
            kl_loss = torch.distributions.kl_divergence(p, q_detached)


        if self.reduction == 'sum':
            kl_loss = kl_loss.sum(-1).mean()
        elif self.reduction == 'mean':
            kl_loss = kl_loss.mean()


        loss = self.kl_weight * kl_loss + self.recon_weight * recon_loss

        if self.interactive_debug and (torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss))):
            import ipdb; ipdb.set_trace()

        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()

        logs = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "recon_weight": self.recon_weight,
            "kl_weight": self.kl_weight,
            "loss": loss,
            'lr': self.lr,
            'iou': iou_pts,
        }

        if self.interactive_debug:
            print(f"loss: {loss.item()}, kl: {kl_loss.item()}")

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

    def encode_inputs(self, partial_pcs, full_pcs=None):
        # We might need to modify this in the future for JIT
        if full_pcs is not None:
            # test mode
            x = self.full_pointnet(full_pcs)
            c = self.partial_pointnet(partial_pcs)

            mu, log_var = self.encoder(x=x, c=c)
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

        else:
            # training mode
            c = self.partial_pointnet(partial_pcs)
            
            prior_mu, prior_log_var = self.prior_encoder(x=c, c=None)
            prior_std = torch.exp(prior_log_var / 2)
            p = torch.distributions.Normal(prior_mu, prior_std)
            
            z = p.rsample()
        decoded_latent = self.latent_decoder(z, c)
        return decoded_latent, c
    
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
        full_pcs = None if sample else repeat(batch['full_pcs'], 'b p d -> (b s) p d', s=num_samples).to(device)

        with torch.no_grad():
            decoded_latent, c = self.encode_inputs(partial_pcs, full_pcs)
#             import ipdb; ipdb.set_trace()
            decoded_latent = {k: rearrange(v, '(b s) ... -> b s ...', b=bs, s=num_samples) for k, v in decoded_latent.items()}
            c = rearrange(c, '(b s) ... -> b s ...', b=bs, s=num_samples)

            mesh_list = list()
            input_list = list()
            for b_i in range(min(max_bs, bs)):
                cur_mesh_list = list()
                input_list.append(batch['partial_pcs'][b_i])

                state_dict = dict()
                for sample_idx in range(num_samples):
                    cur_decoded_latent = {k: v[b_i, sample_idx, None] for k, v in decoded_latent.items()}
                    cur_c = c[b_i, sample_idx, None]
                    generated_mesh = generate_from_latent(self.eval_points, z=cur_decoded_latent, c=cur_c,
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
