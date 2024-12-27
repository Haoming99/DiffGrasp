import torch
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from models.diffusion3x_c import Diffusion3XC
from models.local_encoder import LocalPoolPointnet
from models.conv_decoder import LocalDecoder
from models.diff_cli import UNet2DModelCLI, AutoencoderKLCLI
from diffusers import PNDMScheduler
from utils.callbacks import YcbMeshGenerationCallback
from utils.ema import EMA
from utils.ycb_pcs_dataset import YcbDataModule
from models.conv_decoder import LocalDecoderAttn
from utils.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


def save_point_cloud_xyz(point_cloud, filename):
    """
    Save point cloud to a .xyz file.
    Args:
        point_cloud (torch.Tensor or np.ndarray): Point cloud to save (N x 3).
        filename (str): File path to save the point cloud in .xyz format.
    """
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()


    np.savetxt(filename, point_cloud, fmt="%.6f", delimiter=" ")

def main():

    pl.seed_everything(42)

    # Initialize the model
    model = Diffusion3XC(
        decoder=LocalDecoderAttn(
            dim=3,
            c_dim=8,  # 8
            n_blocks=5,
            hidden_size=32
        ),
        local_pointnet=LocalPoolPointnet(
            c_dim=8, # 8
            dim=3,
            hidden_dim=32,
            scatter_type='max',
            unet=True,
            unet_kwargs={"depth": 4, "merge_mode": "concat", "start_filts": 32},
            unet3d=False,
            unet3d_kwargs=None,
            plane_resolution=64,
            grid_resolution=None,
            plane_type=["xy", "xz", "yz"],
            padding=0.1,
            n_blocks=5
        ),
        vae=AutoencoderKLCLI(
            in_channels=24,  # 24
            out_channels=24,  # 24
            latent_channels=12,  
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],   #["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"], #["UpDecoderBlock2D"] * 4,
            block_out_channels=[64, 128, 256, 256],
            norm_num_groups=32,
            layers_per_block=2,
            act_fn="silu"
        ),
        unet=UNet2DModelCLI(
            in_channels=24,  # 2 * latent_dim 
            out_channels=12,  
            attention_head_dim=8,
            block_out_channels=[224, 448, 672, 896],
            down_block_types=["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            up_block_types=["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
            layers_per_block=2,
            downsample_padding=1,
            flip_sin_to_cos=True,
            freq_shift=0,
            mid_block_scale_factor=1,
            norm_eps=1e-05,
            norm_num_groups=32,
            center_input_sample=False
        ),
        scheduler=PNDMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            set_alpha_to_one=False,
            trained_betas=None
        ),
        lr=0.0001, 
        feat_dim=8,  # 8
        latent_dim=12,  
        num_inference_steps=50,  
        vox_reso=64,  
        eta=0.,  
        padding=0.1, 
        b_min=-0.5,  
        b_max=0.5, 
        points_batch_size=100000, 
        batch_size=1,  
        batchnorm=False, 
        dropout=False,  
        filter_nan=False, 
        invis_loss_weight=1.0,  
        test_num_samples=10,  
        interactive_debug=False,  
        pretrained_path=None, 
        pretrained_lr=None,  
        lr_decay_steps=1.0e+4,
        full_std=1.073,
        partial_std=1.073, 
        lr_decay_factor=0.9,  
        pre_lr_freeze_steps=5.0e+4,  
        pre_lr_freeze_factor=0.1,  
        automatic_optimization=True
    )
    ckpt=torch.load('/home/reallhm/projects/pts_exp/logs/diffusion-ycb-new26000/version_5/checkpoints/last.ckpt')
    model.load_state_dict(ckpt['state_dict'])

    # Initialize the dataset
    data_module = YcbDataModule(
        train_root_dir='/mnt/kostas-graid/datasets/haoming/3dsgrasp_ycb_train_test_split/input/*/train',  
        test_root_dir='/mnt/kostas-graid/datasets/haoming/3dsgrasp_ycb_train_test_split/input/*/test',  
        pcd_dir='/mnt/kostas-graid/datasets/haoming/3dsgrasp_ycb_train_test_split/gt',  
        batch_size=1,  
        num_workers=4,  
        npoint=2048  # Number of points to sample from the ground truth point cloud
    )

    data_module.setup(stage="test")  

    
    test_dataloader = data_module.test_dataloader()

    sample_counter = 0  
    for batch in test_dataloader:
        full_point_clouds = batch['full_pcs']  

        
        with torch.no_grad():
            mesh_list, _ = model.generate_mesh(batch, sample_counter, max_bs=-1, num_samples=1, sample=True, denormalize=False)

        
        for i, (mesh, full_pc) in enumerate(zip(mesh_list, full_point_clouds)):
            first_mesh = mesh[0]  # Take only the first (and only) mesh variant

            sampled_points = first_mesh.sample(8192)
            pred = sampled_points.reshape(1, 8192, 3)
            pred = torch.tensor(pred).float().cuda()
            gt = full_pc
            ChamferDisL2 = ChamferDistanceL2()
            CDL2 = ChamferDisL2(pred, gt)
            print(f"Chamfer L2 Loss: {CDL2.item() * 1000}")

            save_point_cloud_xyz(sampled_points, f"results/sample_points/sample_pc_{sample_counter}.xyz")
           
            first_mesh.export(f"results/meshes/output_mesh_{sample_counter}.obj")
            
            save_point_cloud_xyz(full_pc, f"results/inputs/gt_pc_{sample_counter}.xyz")


            
            sample_counter += 1


if __name__ == "__main__":
    main()