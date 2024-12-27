import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from models.diffusion3x_c import Diffusion3XC
from models.local_encoder import LocalPoolPointnet
from models.conv_decoder import LocalDecoder
from models.diff_cli import UNet2DModelCLI, AutoencoderKLCLI
from diffusers import PNDMScheduler
from utils.callbacks import ShapeNetMeshGenerationCallback
from utils.ema import EMA
from utils.sn_pcs_dataset import ShapeNetPcsDataModule
from models.conv_decoder import LocalDecoderAttn

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
            in_channels=24, 
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
        lr=0.00001, 
        feat_dim=8,  
        latent_dim=12,  
        num_inference_steps=50, 
        vox_reso=64,  
        eta=0.,  
        padding=0.1,  
        b_min=-0.5,  
        b_max=0.5,  
        points_batch_size=100000,  
        batch_size=24,  
        batchnorm=False,  
        dropout=False,  
        filter_nan=False,  
        invis_loss_weight=1.0, 
        test_num_samples=10,  
        interactive_debug=False,  
        pretrained_path=None, 
        pretrained_lr=None, 
        lr_decay_steps=2.0e+4,
        full_std=1.073,
        partial_std=1.073,  
        lr_decay_factor=0.9, 
        pre_lr_freeze_steps=5.0e+4,  
        pre_lr_freeze_factor=0.1, 
        automatic_optimization=True
    )
    ckpt=torch.load('/home/reallhm/projects/pts_exp/logs/diffusion-all/version_2/checkpoints/last.ckpt')
    model.load_state_dict(ckpt['state_dict'])

    # Initialize the dataset
    data_module = ShapeNetPcsDataModule(
        batch_size=24, num_workers=12, val_trainset=False,
        subset=0, shuffle_train=True,
        dataset_folder="/mnt/kostas-graid/datasets/haoming/ShapeNet",
        split_type="onet", categories=None,
        test_cat="chair", n_pcs_pts=1024,
        n_sampled_points=2048, pcs_noise=0.005,
        partial_mode="rand_all", with_query_mask=False, test_partial_mode="octant"
    )

    # Initialize the logger
    tensorboard_logger = TensorBoardLogger(save_dir="./logs/", name="diffusion-all-3500")
    wandb_logger = WandbLogger(
        project="pts_exp", name="diffusion-all-3500", log_model=False,
        id="diffusion-all-3500"
    )

    # Initialize the callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", verbose=True,
        save_last=True, save_top_k=-1,
        save_weights_only=False,
        mode="min", auto_insert_metric_name=True,
        every_n_train_steps=10000,
        save_on_train_epoch_end=True,
        train_time_interval=None,
        every_n_epochs=None
    )

    shapenet_callback = ShapeNetMeshGenerationCallback(
        viz_dir="./visualizations/diffusion-all-3500",
        dataset_dir="/mnt/kostas-graid/datasets/haoming/ShapeNet",  
        num_viz=4, num_samples=3, train_p=0.001, sample_only=True
    )

    ema_callback = EMA(
        decay=0.999, ema_device="cuda",
        pin_memory=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=[tensorboard_logger, wandb_logger],
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, shapenet_callback, ema_callback, lr_monitor],
        default_root_dir="./logs/diffusion-all-3500",
        gpus=1, max_epochs=-1, max_steps=-1,
        limit_val_batches=10,
        val_check_interval=1.0,
        log_every_n_steps=50,
        strategy=DDPStrategy(find_unused_parameters=True),
        sync_batchnorm=False,
        precision=16,
        accumulate_grad_batches=2,
        gradient_clip_val=5.0
    )

    # Start training
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()

