import torch
from torch import nn
from torch.nn import functional as F
# from vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, low_bit=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)




class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out

class BlockBN(nn.Module):
    def __init__(self, in_width, middle_width, out_width, resolution, down_rate=None, residual=False, use_3x3=True, zero_last=False, activation='gelu'):
        """
        resoltuion is not used for bn
        """
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)
        self.actvn = getattr(F, activation)

        self.norm1 = nn.BatchNorm2d(middle_width)
        self.norm2 = nn.BatchNorm2d(middle_width)
        self.norm3 = nn.BatchNorm2d(middle_width)


    def forward(self, x):
        xhat = self.norm1(self.actvn(self.c1(x)))
        xhat = self.norm2(self.actvn(self.c2(xhat)))
        xhat = self.norm3(self.actvn(self.c3(xhat)))
        xhat = self.c4(xhat)
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out

class BlockLN(nn.Module):
    def __init__(self, in_width, middle_width, out_width, resolution, down_rate=None, residual=False, use_3x3=True, zero_last=False, activation='gelu'):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)
        self.actvn = getattr(F, activation)

        self.norm1 = nn.LayerNorm(resolution)
        self.norm2 = nn.LayerNorm(resolution)
        self.norm3 = nn.LayerNorm(resolution)


    def forward(self, x):
        xhat = self.norm1(self.actvn(self.c1(x)))
        xhat = self.norm2(self.actvn(self.c2(xhat)))
        xhat = self.norm3(self.actvn(self.c3(xhat)))
        xhat = self.c4(xhat)
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out

class BlockID(nn.Module):
    def __init__(self, in_width, middle_width, out_width, resolution, down_rate=None, residual=False, use_3x3=True, zero_last=False, activation='gelu'):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)
        self.actvn = getattr(F, activation)

        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.norm3 = nn.Identity()


    def forward(self, x):
        xhat = self.norm1(self.actvn(self.c1(x)))
        xhat = self.norm2(self.actvn(self.c2(xhat)))
        xhat = self.norm3(self.actvn(self.c3(xhat)))
        xhat = self.c4(xhat)
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


class VDVAE_Encoder(nn.Module):
    def __init__(self, in_dim, width, enc_blocks_str, custom_width_str='', bottleneck_multiple=0.25):
        super().__init__()
        self.in_conv = get_3x3(in_dim, width)
        self.widths = get_width_settings(width, custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(enc_blocks_str)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, width, res, mixin, n_blocks, z_dim, bottleneck_multiple=0.25):
        super().__init__()
        self.base = res
        self.mixin = mixin
        
        use_3x3 = res > 2
        cond_width = int(width * bottleneck_multiple)
        self.z_dim = z_dim
        self.enc = Block(width * 2, cond_width, z_dim * 2, residual=False, use_3x3=use_3x3)
        self.prior = Block(width * 2, cond_width, z_dim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_proj = get_1x1(z_dim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, full_acts, part_acts):
        qm, qv = self.enc(torch.cat([x, full_acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(torch.cat([x, part_acts], dim=1))
        pm, pv, xpp = feats[:, :self.z_dim, ...], feats[:, self.z_dim:self.z_dim * 2, ...], feats[:, self.z_dim * 2:, ...]
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl

    def sample_cond(self, x, part_acts, t=None, lvs=None):
        n, c, h, w = x.shape
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

    def get_inputs(self, xs, full_activations, part_activations):
        full_acts = full_activations[self.base]
        part_acts = part_activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(part_acts)
        if part_acts.shape[0] != x.shape[0]:
            x = x.repeat(part_acts.shape[0], 1, 1, 1)
        return x, full_acts, part_acts

    def get_inputs_cond(self, xs, part_activations):
        part_acts = part_activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
#             print(f"can't find x with reso {self.base}, part_acts: {part_acts.shape}")
            x = torch.zeros_like(part_acts)
        if part_acts.shape[0] != x.shape[0]:
            x = x.repeat(part_acts.shape[0], 1, 1, 1)
        return x, part_acts

    def forward(self, xs, full_activations, part_activations, get_latents=False):
        x, full_acts, part_acts = self.get_inputs(xs, full_activations, part_activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl = self.sample(x, full_acts, part_acts)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl)
        return xs, dict(kl=kl)

    def forward_cond(self, xs, part_activations, t=None, lvs=None):
        x, part_acts = self.get_inputs_cond(xs, part_activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_cond(x, part_acts, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs


class VDVAE_Decoder(nn.Module):

    def __init__(self, width, dec_blocks_str, z_dim, custom_width_str='', no_bias_above=64, bottleneck_multiple=0.25):
        super().__init__()
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(width, custom_width_str)
        blocks = parse_layer_string(dec_blocks_str)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(self.widths[res], res, mixin, n_blocks=len(blocks), 
                                       z_dim=z_dim, bottleneck_multiple=bottleneck_multiple))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= no_bias_above])

        self.gain = nn.Parameter(torch.ones(1, width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias
        self.final_reso = self.resolutions[-1]
        self.out_net = get_1x1(width, z_dim)

    def forward(self, full_activations, part_activations, get_latents=False):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs, full_activations, part_activations, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.final_reso] = self.final_fn(xs[self.final_reso])
        out = self.out_net(xs[self.final_reso])
        return out, stats

    def forward_cond(self, part_activations, n, t=None, y=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1) 
            # n will always be the b*s for now because we repeat full and partial pcs outside
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_cond(xs, part_activations, temp)
        xs[self.final_reso] = self.final_fn(xs[self.final_reso])
        out = self.out_net(xs[self.final_reso])
        return out

#     def forward_manual_latents(self, n, latents, t=None):
#         raise Unim
#         xs = {}
#         for bias in self.bias_xs:
#             xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
#         for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
#             xs = block.forward_uncond(xs, t, lvs=lvs)
#         xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
#         return xs[self.H.image_size]
