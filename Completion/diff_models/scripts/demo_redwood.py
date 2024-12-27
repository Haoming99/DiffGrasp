import torch
import os
os.chdir('/home/wen/redwood-3dscan/')
import redwood_3dscan as rws
os.chdir('/home/wen/pts_exp/')

import os
import os.path as osp
import sys

import math
import torch
import numpy as np
import meshplot as mp
from cli_scratch import ClusterCLI
from utils.mesh import generate_from_latent
from utils import datamodules
from utils.sn_pcs_dataset import worker_init_fn

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from livelossplot import PlotLosses
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import cv2
from utils.sn_pcs_dataset import mask_op
from utils.viz import plot_pcs
from glob import glob
import seaborn as sns
from typing import Union, Optional, List, Any
import csv
import pickle

class NBCLI(ClusterCLI):
    def fit(self, *args, **kwargs):
        pass



def main():
    device = torch.device('cuda')
    CONFIG_FILE = './configs/krc-mpc-reb-allcat.yaml'

    sys.argv =['cli_scratch.py', 'fit', '--config', CONFIG_FILE, '--trainer.strategy=dp', '--trainer.logger=False']
    cli = NBCLI(subclass_mode_model=True, subclass_mode_data=True, auto_registry=True, parser_kwargs={"parser_mode": "omegaconf", "error_handler": None}, save_config_overwrite=True, run=True, env_parse=False)

    model = cli.model.to(device)
    model.verbose = True

    ckpt = torch.load('./logs/krc-mpc-reb-allcat/krc-mpc-reb-allcat_pts_exp/_krc-mpc-reb-allcat/checkpoints/epoch=1258-step=190000.ckpt')
    model.load_state_dict(ckpt['state_dict'])

    redwood_base = '/home/wen/redwood-3dscan/data/rgbd/00033/'

    depths = sorted(glob(f"{redwood_base}/depth/*.png"))

    matched_fns = list()
    for depth_fn in depths:
        timestamp = osp.basename(depth_fn).split('-')[0]
        matched_rgb_fns = glob(f"{redwood_base}/rgb/{timestamp}-*.jpg")
        
        if matched_rgb_fns:
            matched_rgb_fn = matched_rgb_fns[0]
            matched_fns.append((depth_fn, matched_rgb_fn))

    depth_rgb_pair = matched_fns[600]
    depth_raw = o3d.io.read_image(depth_rgb_pair[0])
    color_raw = o3d.io.read_image(depth_rgb_pair[1])

    im = np.array(color_raw)
    outputs = predictor(im)
    pred_mask = outputs['instances'].pred_masks.cpu().numpy()[0]

    masked_depth = np.asanyarray(depth_raw).copy().astype(np.float32)
    masked_depth[pred_mask==0] = -1

    mask_pcs = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(masked_depth),
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        project_valid_depth_only=True    
    )

    scan_pts = np.asarray(mask_pcs.points)

    normalzied_pts = (scan_pts - scan_pts.mean())/ (scan_pts.std() * 4)
    # emprically normalize the data to the range of -0.3 ~ 0.3 (roughly)

    idx = np.random.randint(scan_pts.shape[0], size=1024)
    pcs_sampled = normalzied_pts[idx].copy()
    partial_pcs = pcs_sampled.astype(np.float32)

    pseudo_batch = {'partial_pcs': torch.tensor(partial_pcs)[None,:].to(device)}

    mesh_list, _ = model.generate_mesh(pseudo_batch, 0, max_bs=-1, num_samples=2, sample=True, denormalize=False)

if __name__ == '__main__':
    main()