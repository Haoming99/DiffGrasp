# CLI warpper for diffusers
from random import sample
from typing import Union, Optional, Tuple, List, Iterable
from diffusers import UNet2DModel, AutoencoderKL, VQModel
from diffusers.models.vae import VectorQuantizer, DecoderOutput, Encoder, Decoder
from diffusers.models.vq_model import VQEncoderOutput

# from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

# @MODEL_REGISTRY
class VQModelCLI(VQModel):
    r"""VQ-VAE model from the paper Neural Discrete Representation Learning by Aaron van den Oord, Oriol Vinyals and Koray
    Kavukcuoglu.
    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)
    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Iterable[str] = ["DownEncoderBlock2D",],
        up_block_types: Iterable[str] = ["UpDecoderBlock2D",],
        block_out_channels: Iterable[int] = [64,],
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels= out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            sample_size=sample_size,
            num_vq_embeddings=num_vq_embeddings,
            norm_num_groups=norm_num_groups,
        )


# @MODEL_REGISTRY
class AutoencoderKLCLI(AutoencoderKL):
    r"""Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.
    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)
    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `4`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Iterable[str] = ["DownEncoderBlock2D",],
        up_block_types: Iterable[str] = ["UpDecoderBlock2D",],
        block_out_channels: Iterable[int] = [64,],
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels= out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            sample_size=sample_size,
            norm_num_groups=norm_num_groups,
        )

# @MODEL_REGISTRY
class UNet2DModelCLI(UNet2DModel):
    r"""
    UNet2DModel is a 2D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.
    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)
    Parameters:
        sample_size (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Input sample size.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`False`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`): Tuple of downsample block
            types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(224, 448, 672, 896)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for the normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for the normalization.
    """


    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Iterable[str] = ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
        up_block_types: Iterable[str] = ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
        block_out_channels: Iterable[int] = [224, 448, 672, 896],
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=mid_block_scale_factor,
            downsample_padding=downsample_padding,
            act_fn=act_fn,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps
        )


# @MODEL_REGISTRY
class GroupVQModel(nn.Module):
    r"""VQ-VAE model from the paper Neural Discrete Representation Learning by Aaron van den Oord, Oriol Vinyals and Koray
    Kavukcuoglu.
    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)
    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels of embedding space
        sample_size (`int`, *optional*, defaults to `32`): TODO
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
        feat_split_group: (`int`): number of feat planes, encode_out_dim == latent_channels * feat_split_group
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Iterable[str] = ["DownEncoderBlock2D",],
        up_block_types: Iterable[str] = ["UpDecoderBlock2D",],
        block_out_channels: Iterable[int] = [64,],
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        feat_split_group: int = 3,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels * feat_split_group,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
        )

        self.quant_conv = torch.nn.Conv2d(latent_channels * feat_split_group, latent_channels * feat_split_group, 1)
        self.quantize = VectorQuantizer(
            num_vq_embeddings, latent_channels, beta=0.25, remap=None, sane_index_shape=False
        )
        self.post_quant_conv = torch.nn.Conv2d(latent_channels * feat_split_group, latent_channels * feat_split_group, 1)

        self.latent_channels = latent_channels
        self.feat_split_group = feat_split_group

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels * feat_split_group,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> VQEncoderOutput:
        h = self.encoder(x)
        h = self.quant_conv(h)

        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    def decode(
        self, h: torch.FloatTensor, force_not_quantize: bool = False, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        # also go through quantization layer
        if not force_not_quantize:
            # Split the feats channel and hide them into batch dimenstion
            h_ = rearrange(h, 'b (c g) h w -> (b g) c h w', c=self.latent_channels, g=self.feat_split_group)
            quant, emb_loss, info = self.quantize(h_)
            quant = rearrange(quant, '(b g) c h w -> b (c g) h w', c=self.latent_channels, g=self.feat_split_group)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(self, sample: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        h = self.encode(x).latents
        dec = self.decode(h).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)