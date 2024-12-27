from optax import linear_schedule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import pytorch_lightning as pl
import pl_bolts
from typing import Union, Optional
from utils.eval_utils import compute_iou
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from utils.mesh import generate_from_latent, export_shapenet_samples
from einops import repeat, reduce, rearrange
from models.conv_onet import ConvONet, ConvONetDecoder
from models.auto_decoder import AutoDecoder

@MODEL_REGISTRY
class ADConvONet(ConvONet):
    def __init__(self, decoder: ConvONetDecoder, encoder: Optional[nn.Module], auto_decoder: nn.Module, lr=5e-4, padding=0.1, b_min=-0.5, b_max=0.5,points_batch_size=100000, batch_size=12, test_num_samples=1,
                pretrained_path='/scratch/wen/shapenet_grid32.pt', latent_lr=1e-3, approx_weight=1., latent_reg_weight=1e-4, recon_weight=0., finetune_lr=0.):
        super().__init__(decoder, encoder, lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1)
        self.auto_decoder = auto_decoder
        self.latent_codes = nn.Parameter(torch.normal(0, 0.01, (30661, 256))) # Code book for training set, 306612 is shapenet training set size of ONet split
        
        vox_reso = self.encoder.reso_grid
        grid_mask = torch.zeros((6, vox_reso, vox_reso, vox_reso))
        grid_mask[0, :, :, :vox_reso//2] = 1 # z
        grid_mask[1, :, :, vox_reso//2:] = 1
        grid_mask[2, :, :vox_reso//2, :] = 1 # x
        grid_mask[3, :, vox_reso//2:, :] = 1
        grid_mask[4, :vox_reso//2, :, :] = 1 # y
        grid_mask[5, vox_reso//2:, :, :] = 1
        grid_mask = grid_mask.unsqueeze(1)
        self.register_buffer('grid_mask', grid_mask)

        self.approx_weight = approx_weight
        self.latent_reg_weight = latent_reg_weight
        self.recon_weight = recon_weight
        self.latent_lr = latent_lr
        self.finetune_lr = finetune_lr
        self.automatic_optimization = False
        
        if pretrained_path:
            ori_state_dict = torch.load(pretrained_path)
            self.encoder.load_state_dict({k.lstrip('encoder.'): v for k, v in ori_state_dict['model'].items() if k.startswith('encoder.')})
            self.decoder.load_state_dict({k.lstrip('decoder.'): v for k, v in ori_state_dict['model'].items() if k.startswith('decoder.')})
            
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
        # TODO: Introduce auto decoder here
        p_r = self.decode(p, c=c, **kwargs)
        return p_r
            

    def step(self, batch, batch_idx):
        c, x = batch['partial_pcs'], batch['full_pcs']
        query_pts, query_occ = batch['query_pts'], batch['query_occ']
        msk = self.grid_mask[batch['mask_type']]
        # Just use full pcs now to reproduce conv_onet
        bs = x.shape[0]

        with torch.no_grad(): # NOTE: This would lead to a potential bug that we didn't traing the encoder.
            target_c = self.encode_inputs(c)
        z = self.latent_codes[batch['index']]
        
        pred_c = self.auto_decoder(z)
        
        approx_loss = F.mse_loss(pred_c['grid'], target_c['grid'], reduction='none') * msk # TODO: Add support for other type
        approx_loss = torch.sum(approx_loss) / bs
        reg_loss =  torch.sum(torch.norm(z, dim=1)) / bs # TODO: add annealing here

        pred_occ = self.decode(query_pts, c=pred_c)
        recon_loss = F.binary_cross_entropy_with_logits(
            pred_occ, query_occ, reduction='none').sum(-1).mean()

        loss = self.approx_weight * approx_loss + self.latent_reg_weight * reg_loss + self.recon_weight * recon_loss
        
        with torch.no_grad():
            iou_pts = compute_iou(pred_occ, query_occ)
            iou_pts = torch.nan_to_num(iou_pts, 0).mean()

        logs = {
            "recon_loss": recon_loss,
            "loss": loss,
            "approx_loss": approx_loss,
            "reg_loss": reg_loss,
            'lr': self.lr,
            'iou': iou_pts,
            'latent_reg_weight': self.latent_reg_weight,
            'recon_weight': self.recon_weight,
            'approx_weight': self.approx_weight,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers() # NOTE: Should assume the optimizer is real net_optimizer
        lr_schedulers = self.lr_schedulers()
        
        loss, logs = self.step(batch, batch_idx)
        
        for optim in optimizers:
            optim.zero_grad()

        self.manual_backward(loss)

        for optim in optimizers:
            optim.step()

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

        #self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, batch_size=self.batch_size)
        # Fix the issue that no running loss is printed in progress bar according to issue #4295
        if self.trainer is not None:
            self.trainer.fit_loop.running_loss.append(loss)
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
#             c = self.encode_inputs(partial_pcs)
            z = self.latent_codes[batch['index']]
            c = self.auto_decoder(z)
            # TODO: Add support for evaluation
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
        net_optimizer = torch.optim.Adam(self.auto_decoder.parameters(), lr=self.lr)
        net_scheduler = torch.optim.lr_scheduler.StepLR(net_optimizer, 30000*500//64, gamma=0.5)
        latent_optimizer = torch.optim.Adam([self.latent_codes], lr=self.latent_lr)
        latent_scheduler = torch.optim.lr_scheduler.StepLR(latent_optimizer, 30000*500//64, gamma=0.5)

        finetune_optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}],
             lr=self.finetune_lr)
        finetune_scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            finetune_optimizer, warmup_epochs=30000*1000/64, max_epochs=30000*10000/64, eta_min=1e-8)

        return {'optimizer': net_optimizer,
             'lr_scheduler': {"scheduler": net_scheduler,
                "interval": "step",
                "name": 'auto_decoder',}}, {
            'optimizer': latent_optimizer,
             'lr_scheduler': {"scheduler": latent_scheduler,
                "interval": "step",
                "name": 'latent_codes',
        }},{
            'optimizer': finetune_optimizer, 
            'lr_scheduler': {"scheduler": finetune_scheduler,
               "interval": "step",
               "name": 'finetune',}
        }
#         return [torch.optim.Adam(self.auto_decoder.parameters(), lr=self.lr), ]