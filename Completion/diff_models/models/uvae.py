import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

from models.vdvae_modules import gaussian_analytical_kl, draw_gaussian_diag_samples, Block, BlockBN, BlockLN, BlockID
from models.unet import conv3x3, conv1x1, upconv2x2, DownConv, UpConv

block_dict = {'BlockID': BlockID, 'BlockBN': BlockBN, 'BlockLN': BlockLN}

kl_regularization = lambda qm, qv: gaussian_analytical_kl(qm, torch.zeros_like(qm), qv, torch.zeros_like(qv)) # qv is log(sgima_1)


class Bottleneck(nn.Module):
    """
    A global VAE bottleneck module that maps a (B, C, W, W) feats to (B, C, W, W)
    """
    def __init__(self, in_channels, out_channels, z_dim, n_blocks, reso,
                 merge_mode='concat', block_type='BlockID', activation_type='gelu'):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_dim = z_dim

        reso = reso // 2 # A global pooling is performed for the bottleneck
        
        self.enc = block_dict[block_type](self.in_channels, self.out_channels, self.z_dim * 2, resolution=(reso, reso), residual=False, activation=activation_type)
        self.prior = block_dict[block_type](self.in_channels, self.out_channels, self.z_dim * 2 + self.out_channels, resolution=(reso, reso), residual=False, zero_last=True, activation=activation_type)

        self.z_proj = conv1x1(z_dim, self.out_channels)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = block_dict[block_type](out_channels, out_channels, out_channels, resolution=(reso*2, reso*2), residual=True, use_3x3=False, activation=activation_type)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

    def sample(self, full_vecs, part_vec):
        qm, qv = self.enc(full_vecs).chunk(2, dim=1)
        feats = self.prior(part_vec)
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        kl_reg_prob = kl_regularization(qm, qv)
        kl_reg_prior = kl_regularization(pm, pv)

        return z, x, kl, kl_reg_prior, kl_reg_prob

    def sample_cond(self, part_vec, t=None, lvs=None):
        feats = self.prior(part_vec)
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def forward(self, full_acts, part_acts):
        """ Forward pass
        Arguments:
            feats: current feature map in the upconv branch
            from_up: upconv'd tensor from the decoder pathway
        """
        scale_factor = part_acts.shape[2]
        
        full_vec = F.avg_pool2d(full_acts, scale_factor)
        part_vec = F.avg_pool2d(part_acts, scale_factor)
        
        z, x, kl, kl_reg_prob, kl_reg_prior = self.sample(full_vec, part_vec)
        
        x = x + self.z_proj(z)
        
        x = F.interpolate(x, scale_factor=scale_factor) + full_acts
        
        x = self.resnet(x)
        
        return z, x, kl, kl_reg_prob, kl_reg_prior
    
    def forward_cond(self, part_acts):
        scale_factor = part_acts.shape[2]
        
        part_vec = F.avg_pool2d(part_acts, scale_factor)
        
        z, x = self.sample_cond(part_vec)
        
        x = x + self.z_proj(z)
        
        x = F.interpolate(x, scale_factor=scale_factor) + part_acts
        
        x = self.resnet(x)
        
        return z, x


class UpConvVAE(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, z_dim, n_blocks, reso,
                 merge_mode='concat', up_mode='transpose', block_type='BlockID', activation_type='gelu'):
        super(UpConvVAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.z_dim = z_dim

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)
        
        self.enc = block_dict[block_type](self.out_channels + self.out_channels, self.out_channels, self.z_dim * 2, resolution=(reso, reso), residual=False, activation=activation_type)
        self.prior = block_dict[block_type](self.out_channels + self.out_channels, self.out_channels, self.z_dim * 2 + self.out_channels, resolution=(reso, reso), residual=False, zero_last=True, activation=activation_type)

        self.z_proj = conv1x1(z_dim, self.out_channels)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = block_dict[block_type](out_channels, out_channels, out_channels, resolution=(reso, reso), residual=True, use_3x3=True, activation=activation_type)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

    def sample(self, x, full_acts, part_acts):
        qm, qv = self.enc(torch.cat([x, full_acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(torch.cat([x, part_acts], dim=1))
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)

        kl_reg_prob = kl_regularization(qm, qv)
        kl_reg_prior = kl_regularization(pm, pv)

        return z, x, kl, kl_reg_prior, kl_reg_prob

    def sample_cond(self, x, part_acts, t=None, lvs=None):
        feats = self.prior(torch.cat([x, part_acts], dim=1))
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def forward(self, feats, full_acts, part_acts):
        """ Forward pass
        Arguments:
            feats: current feature map in the upconv branch
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(feats)
        
        z, x, kl, kl_reg_prob, kl_reg_prior = self.sample(from_up, full_acts, part_acts)
        
        x = x + self.z_proj(z)
        
        x = self.resnet(x)
        
        return z, x, kl, kl_reg_prob, kl_reg_prior
    
    def forward_cond(self, feats, part_acts,):
        from_up = self.upconv(feats)
        
        z, x = self.sample_cond(from_up, part_acts)
        
        x = x + self.z_proj(z)
        
        x = self.resnet(x)
        
        return z, x

class VAEUNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, z_dim=16, up_mode='transpose', in_reso=32, block_type='BlockID', activation_type='gelu', 
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(VAEUNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.in_reso = in_reso

        self.down_convs = []
        self.up_convs = []
        part_down_convs = []

        # create the encoder pathway and add to a list
        cur_reso = in_reso
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            cur_reso = cur_reso // 2 if pooling else cur_reso

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
            
            part_down_conv = DownConv(ins, outs, pooling=pooling)
            part_down_convs.append(part_down_conv)

        self.bottleneck = Bottleneck(outs, outs, z_dim, depth, reso=cur_reso, block_type=block_type, activation_type=activation_type)
        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks

        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            cur_reso = cur_reso * 2
            up_conv = UpConvVAE(ins, outs, z_dim, depth, reso=cur_reso, block_type=block_type, activation_type=activation_type, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.part_down_convs = nn.ModuleList(part_down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, full_feats, part_feats):
        encoder_outs = []
        part_encoder_outs = []
        kl_losses = []
        reg_probs = []
        reg_priors = []
        x = full_feats
        c = part_feats
        # encoder pathway, save outputs for merging
        for i, (module, part_module) in enumerate(zip(self.down_convs, self.part_down_convs)):
            x, before_pool = module(x)
            c, part_before_pool = part_module(c)

            encoder_outs.append(before_pool)
            part_encoder_outs.append(part_before_pool)

        _, x, kl, kl_reg_prob, kl_reg_prior = self.bottleneck(x, c)
        kl_losses.append(kl)
        reg_probs.append(kl_reg_prob)
        reg_priors.append(kl_reg_prior)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            part_before_pool = part_encoder_outs[-(i+2)]
            _, x, kl, kl_reg_prob, kl_reg_prior = module(x, before_pool, part_before_pool)
            kl_losses.append(kl)
            reg_probs.append(kl_reg_prob)
            reg_priors.append(kl_reg_prior)
        
        x = self.conv_final(x)
        return x, kl_losses, reg_probs, reg_priors

    
    def forward_cond(self, part_feats):
        
        part_encoder_outs = []
        c = part_feats
        # encoder pathway, save outputs for merging
        for i, part_module in enumerate(self.part_down_convs):
            c, part_before_pool = part_module(c)
            part_encoder_outs.append(part_before_pool)

        _, x = self.bottleneck.forward_cond(c)

        for i, module in enumerate(self.up_convs):
            
            part_before_pool = part_encoder_outs[-(i+2)]
            _, x = module.forward_cond(x, part_before_pool)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class BottleneckPlugged(nn.Module):
    """
    A global VAE bottleneck module that maps a (B, C, W, W) feats to (B, C, W, W)
    """
    def __init__(self, in_channels, out_channels, z_dim, n_blocks, reso,
                 merge_mode='concat', block_type='BlockID', activation_type='gelu'):
        super(BottleneckPlugged, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.z_dim = z_dim

        reso = reso // 2 # A global pooling is performed for the bottleneck
        
        self.enc = block_dict[block_type](self.in_channels, self.out_channels, self.z_dim * 2, resolution=(reso, reso), residual=False, activation=activation_type)
        self.prior = block_dict[block_type](self.in_channels, self.out_channels, self.z_dim * 2 + self.out_channels, resolution=(reso, reso), residual=False, zero_last=True, activation=activation_type)

        self.z_proj = conv1x1(z_dim, self.out_channels)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.mixer = conv1x1(2*self.out_channels, self.out_channels)

        self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def sample(self, full_vecs, part_vec):
        qm, qv = self.enc(full_vecs).chunk(2, dim=1)
        feats = self.prior(part_vec)
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        kl_reg_prob = kl_regularization(qm, qv)
        kl_reg_prior = kl_regularization(pm, pv)

        return z, x, kl, kl_reg_prior, kl_reg_prob

    def sample_cond(self, part_vec, t=None, lvs=None):
        feats = self.prior(part_vec)
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def forward(self, full_acts, part_acts):
        """ Forward pass
        Arguments:
            feats: current feature map in the upconv branch
            from_up: upconv'd tensor from the decoder pathway
        """
        scale_factor = part_acts.shape[2]
        
        full_vec = F.avg_pool2d(full_acts, scale_factor)
        part_vec = F.avg_pool2d(part_acts, scale_factor)
        
        z, x, kl, kl_reg_prob, kl_reg_prior = self.sample(full_vec, part_vec)

        if self.merge_mode == 'concat':
            x = torch.cat((x, self.z_proj(z)), 1)
        else:
            x = x + self.z_proj(z)
        
        x = self.mixer(x)
        
        x = F.interpolate(x, scale_factor=scale_factor) + full_acts

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))        
        
        return z, x, kl, kl_reg_prob, kl_reg_prior
    

    def forward_cond(self, part_acts):
        scale_factor = part_acts.shape[2]
        
        part_vec = F.avg_pool2d(part_acts, scale_factor)
        
        z, x = self.sample_cond(part_vec)

        if self.merge_mode == 'concat':
            x = torch.cat((x, self.z_proj(z)), 1)
        else:
            x = x + self.z_proj(z)
        
        x = self.mixer(x)
        
        x = F.interpolate(x, scale_factor=scale_factor) + part_acts
         
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))        
        
        return z, x


class UpConvPlugged(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, z_dim, n_blocks, reso,
                 merge_mode='concat', up_mode='transpose', block_type='BlockID', activation_type='gelu'):
        super(UpConvPlugged, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.z_dim = z_dim

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)
        
        self.enc = block_dict[block_type](self.out_channels + self.out_channels, self.out_channels, self.z_dim * 2, resolution=(reso, reso), residual=False, activation=activation_type)
        self.prior = block_dict[block_type](self.out_channels + self.out_channels, self.out_channels, self.z_dim * 2 + self.out_channels, resolution=(reso, reso), residual=False, zero_last=True, activation=activation_type)

        self.z_proj = conv1x1(z_dim, self.out_channels)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
#         self.resnet = block_dict[block_type](out_channels, out_channels, out_channels, resolution=(reso, reso), residual=True, use_3x3=True, activation=activation_type)
#         self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def sample(self, x, full_acts, part_acts):
        qm, qv = self.enc(torch.cat([x, full_acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(torch.cat([x, part_acts], dim=1))
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)

        kl_reg_prob = kl_regularization(qm, qv)
        kl_reg_prior = kl_regularization(pm, pv)

        return z, x, kl, kl_reg_prior, kl_reg_prob

    def sample_cond(self, x, part_acts, t=None, lvs=None):
        feats = self.prior(torch.cat([x, part_acts], dim=1))
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def forward(self, feats, full_acts, part_acts):
        """ Forward pass
        Arguments:
            feats: current feature map in the upconv branch
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(feats)
        
        z, x, kl, kl_reg_prob, kl_reg_prior = self.sample(from_up, full_acts, part_acts)
        
        if self.merge_mode == 'concat':
            x = torch.cat((x, self.z_proj(z)), 1)
        else:
            x = x + self.z_proj(z)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        return z, x, kl, kl_reg_prob, kl_reg_prior
    
    def forward_cond(self, feats, part_acts,):
        from_up = self.upconv(feats)
        
        z, x = self.sample_cond(from_up, part_acts)
        
        if self.merge_mode == 'concat':
            x = torch.cat((x, self.z_proj(z)), 1)
        else:
            x = x + self.z_proj(z)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        return z, x

class UNetPlugged(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, z_dim=16, up_mode='transpose', in_reso=32, block_type='BlockID', activation_type='gelu', 
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNetPlugged, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.in_reso = in_reso

        self.down_convs = []
        self.up_convs = []
        part_down_convs = []

        # create the encoder pathway and add to a list
        cur_reso = in_reso
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            cur_reso = cur_reso // 2 if pooling else cur_reso

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
            
            part_down_conv = DownConv(ins, outs, pooling=pooling)
            part_down_convs.append(part_down_conv)

        self.bottleneck = BottleneckPlugged(outs, outs, z_dim, depth, reso=cur_reso, block_type=block_type, activation_type=activation_type)
        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks

        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            cur_reso = cur_reso * 2
            up_conv = UpConvPlugged(ins, outs, z_dim, depth, reso=cur_reso, block_type=block_type, activation_type=activation_type, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.part_down_convs = nn.ModuleList(part_down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, full_feats, part_feats):
        encoder_outs = []
        part_encoder_outs = []
        kl_losses = []
        reg_probs = []
        reg_priors = []
        x = full_feats
        c = part_feats
        # encoder pathway, save outputs for merging
        for i, (module, part_module) in enumerate(zip(self.down_convs, self.part_down_convs)):
            x, before_pool = module(x)
            c, part_before_pool = part_module(c)

            encoder_outs.append(before_pool)
            part_encoder_outs.append(part_before_pool)

        _, x, kl, kl_reg_prob, kl_reg_prior = self.bottleneck(x, c)
        kl_losses.append(kl)
        reg_probs.append(kl_reg_prob)
        reg_priors.append(kl_reg_prior)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            part_before_pool = part_encoder_outs[-(i+2)]
            _, x, kl, kl_reg_prob, kl_reg_prior = module(x, before_pool, part_before_pool)
            kl_losses.append(kl)
            reg_probs.append(kl_reg_prob)
            reg_priors.append(kl_reg_prior)
        
        x = self.conv_final(x)
        return x, kl_losses, reg_probs, reg_priors

    
    def forward_cond(self, part_feats):
        
        part_encoder_outs = []
        c = part_feats
        # encoder pathway, save outputs for merging
        for i, part_module in enumerate(self.part_down_convs):
            c, part_before_pool = part_module(c)
            part_encoder_outs.append(part_before_pool)

        _, x = self.bottleneck.forward_cond(c)

        for i, module in enumerate(self.up_convs):
            
            part_before_pool = part_encoder_outs[-(i+2)]
            _, x = module.forward_cond(x, part_before_pool)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x