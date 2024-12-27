from optparse import Option
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import pytorch_lightning as pl
from typing import Union, Optional, List, Any
import itertools
from datetime import datetime
from models.unet import UNet
from utils.eval_utils import compute_iou
# from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from utils.mesh import generate_from_latent, export_shapenet_samples
from einops import repeat, reduce, rearrange
from abc import ABC, abstractclassmethod, abstractmethod
import logging
import traceback

class KiloBase(pl.LightningModule, ABC):
    """
    Compatiable (c)VAE that support empty z
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, lr=1e-4, padding=0.1, b_min=-0.5, b_max=0.5, points_batch_size=100000, batch_size=12, test_num_samples=1,
                batchnorm=True, reduction='sum', invis_loss_weight=1., interactive_debug=False, freeze_decoder=False, test_num_pts=2048):
        
        super().__init__()

        self.out_dir = None
        self.lr = lr
        self.padding = padding
        self.b_min = b_min
        self.b_max = b_max
        self.points_batch_size = points_batch_size
        self.batch_size = batch_size
        self.test_num_samples = test_num_samples
        self.invis_loss_weight = invis_loss_weight 
        self.freeze_decoder = freeze_decoder
        self.test_num_pts = test_num_pts 
        # num of points to sample from the generated points, 2048 for shapenet, 100000 for synthetic room

        assert reduction in ['sum', 'mean'], f"reduction {reduction} not supported"
        self.reduction = reduction
        self.interactive_debug = interactive_debug
        self._override_ckpt = None
        self._ema = False


    def forward(self, partial_pcs):
        bs = partial_pcs.shape[0]
        device = self.device
        num_samples = 1

        with torch.no_grad():
            # partial_pcs = repeat(partial_pcs, 'b p d -> (b s) p d', s=num_samples).to(device)
            decoded_latent, c = self.encode_inputs(partial_pcs.to(device), None)

            # decoded_latent = {k: rearrange(v, '(b s) ... -> b s ...', b=bs, s=num_samples) for k, v in decoded_latent.items()}
            # c = rearrange(c, '(b s) ... -> b s ...', b=bs, s=num_samples)
            rand_queris = 0.5 * (torch.rand(105781, 3).to(device) - 0.5)

            out = self.eval_points(rand_queris, decoded_latent, c)

            # self.eval_points()

            return decoded_latent, c, out

        #     mesh_list = list()
        #     for b_i in range(bs):
        #         cur_mesh_list = list()

        #         state_dict = dict()
        #         for sample_idx in range(num_samples):
        #             cur_decoded_latent = {k: v[b_i, sample_idx, None] for k, v in decoded_latent.items()}
        #             cur_c = c[b_i, sample_idx, None]
        #             generated_mesh = generate_from_latent(self.eval_points, z=cur_decoded_latent, c=cur_c,
        #                                                   state_dict=state_dict, padding=self.padding,
        #                                                   B_MAX=self.b_max, B_MIN=self.b_min, device=self.device)
        #             cur_mesh_list.append(generated_mesh)
        #         mesh_list.append(cur_mesh_list)

        # return mesh_list
    
    @abstractmethod
    def decode(self, p, z, c=None, **kwargs) -> torch.Tensor:
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z: dict or tensor for decdoer
            c (tensor): latent conditioned code c
        '''
        pass
        # logits = self.decoder(p, c_plane=z, c=c, **kwargs)
        # return logits


    @abstractmethod
    def step(self, batch, batch_idx):
        pass

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

    @abstractmethod
    def encode_inputs(self, partial_pcs, full_pcs=None) -> tuple:
        pass
    
    def generate_mesh(self, batch, batch_idx, max_bs=3, num_samples=1, sample=True, denormalize=False):
        partial_pcs = batch['partial_pcs'] # No difference between partial and complete pcs so far
        bs = partial_pcs.shape[0]
        if max_bs < 0:
            max_bs = bs
        device = self.device

        if denormalize:
            loc = batch['loc'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()

        with torch.no_grad():
            partial_pcs = repeat(partial_pcs, 'b p d -> (b s) p d', s=num_samples).to(device)
            full_pcs = None if sample else repeat(batch['full_pcs'], 'b p d -> (b s) p d', s=num_samples).to(device)
            decoded_latent, c = self.encode_inputs(partial_pcs, full_pcs)

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
        export_shapenet_samples(mesh_list, batch['category'], batch['model'], denormalized_pcs, self.out_dir, test_num_pts=self.test_num_pts)
    

    def predict_step(self, batch, batch_idx):
        try:
            mesh_list, _ = self.generate_mesh(batch, batch_idx, max_bs=-1, num_samples=self.test_num_samples, sample=True, denormalize=True)
            denormalized_pcs = (batch['partial_pcs'] + batch['loc']) * batch['scale']
            export_shapenet_samples(mesh_list, batch['category'], batch['model'], denormalized_pcs, self.out_dir, test_num_pts=self.test_num_pts)
        except Exception as e:
            traceback.print_exc()
            console_logger = logging.getLogger("pytorch_lightning")
            console_logger.error(f"failed to generate mesh on batch_idx={batch_idx}")
            console_logger.error(str(e))
            print(e)

    def get_nondecoder_params(self) -> list:
        pass

    def configure_optimizers(self):
        if not self.freeze_decoder:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam([{'params': itertools.chain(
            self.get_nondecoder_params()
            )},
            {'params': self.decoder.parameters(), 'lr': 0}], lr=self.lr)
            return optimizer

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    @property
    def override_ckpt(self):
        return self._override_ckpt

    @override_ckpt.setter
    def override_ckpt(self, ckpt_path):
        print(f"[IMPORTANT]: Setting override ckpt path {self._override_ckpt} -> {ckpt_path}")
        self._override_ckpt = ckpt_path

    @property
    def ema(self):
        return self._ema

    @ema.setter
    def ema(self, ema: bool) -> None:
        if self._ema != ema:
            print(f"[IMPORTANT]: setting module.ema {self._ema} -> {ema}")
            self._ema = ema
        
    def on_predict_epoch_start(self):
        ckpt_path = self._override_ckpt
        global_step = self.trainer.global_step
        job_id = os.environ.get('SLURM_JOB_ID', 0) 

        console_logger = logging.getLogger("pytorch_lightning")

        if ckpt_path is not None:
            console_logger.info(f"[IMPORTANT]: Loading {ckpt_path} before predict epoch start")
            ckpt = torch.load(self.override_ckpt)
            console_logger.info(f"Loaded, global step: {ckpt['global_step']}")
            global_step = ckpt['global_step']
            print(f"self.ema: {self.ema}")
            if self.ema:
                # Because EMA callbacks are not automatically loaded during validation and prediction https://github.com/Lightning-AI/lightning/issues/10914
                # callback_state = [v for k, v in ckpt['callbacks'].items() if k.startswith('EMA{')][0]
                # callback_state = ckpt['callbacks']['EMA']
                state_dict = ckpt['ema_state_dict']
                print('Loading ema parameters from the callback state')
            else:
                state_dict = ckpt['state_dict']
            self.load_state_dict(state_dict)

        # Write iteration info to the test directory
        print(f"on_predict_epoch_start global step: {self.trainer.global_step}")
        time_str = datetime.now().strftime("%m%d-%H:%M")
        name_str = f"{global_step}it-{time_str}-{job_id}.start"
        os.makedirs(f"{self.out_dir}/../", exist_ok=True)
        os.makedirs(f"{self.out_dir}/../points/", exist_ok=True)
        self.status_fn = f"{self.out_dir}/../{name_str}"
        with open(self.status_fn, "w") as f:
            f.write(name_str)

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        """
        Change record name to indicte the generation is done
        """
        done_fn = self.status_fn.replace('.start', '.done')
        os.system(f"mv {self.status_fn} {done_fn}")
        return super().on_predict_epoch_end(results)