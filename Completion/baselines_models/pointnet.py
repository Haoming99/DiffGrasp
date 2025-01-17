#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Pingping Cai

from asyncio import gather
from multiprocessing.connection import wait
import torch
from torch import nn, einsum
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation


# from knn_cuda import KNN
# knn = KNN(k=16, transpose_mode=False)

class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True,
                 activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = 0.5 * (self.conv_2(torch.relu(self.conv_1(x))) + shortcut)
        return out


def MLP_Stacks(channels, bn=None):
    '''
    function
        stack multiple layers of mlp based on input channel list
    input
        channels: [list]
    output
        layer of multiple mlps
    '''
    layers = []
    last_channel = channels[0]
    for out_channel in channels[1:-1]:
        layers.append(MLP_Res(last_channel, None, out_channel))
        if bn:
            layers.append(nn.BatchNorm1d(out_channel))
        layers.append(nn.LeakyReLU())
        last_channel = out_channel
    layers.append(MLP_Res(last_channel, None, channels[-1]))
    return nn.Sequential(*layers)


def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint))  # (B, 3, npoint)

    idx = ball_query(radius, nsample, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())  # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    # scale to center
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

    if points is not None:
        grouped_points = grouping_operation(points, idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self, npoint, nsample, radius, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(xyz, points, self.npoint, self.nsample,
                                                                     self.radius, self.use_xyz)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        return new_xyz, new_points


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:MLP_CONV
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous())
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0 / dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx, weight)  # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


def square_distance(src, dst):
    """
    Calculate Euclid's distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz):
    """Find k-NN of new_xyz (target) in xyz (ref)"""
    with torch.no_grad():
        sqrdists = square_distance(new_xyz, xyz)  # B, S, N
        # sorted_dist, indices = torch.sort(sqrdists, dim=-1, descending=False)
        # idx = indices[:, :, pad: nsample+pad]
        _, idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
        # sdist = sorted_dist[:,:,pad: nsample+pad]
    return idx.int()


def query_knn1(nsample, xyz, new_xyz, include_self=True):
    _, idx = knn(xyz, new_xyz)  # bs c n
    return idx.int()


def query_kfn(nsample, xyz, new_xyz, include_self=False):
    """Find k-Farest Neighbor of new_xyz (target) in xyz """
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    sorted_dist, indices = torch.sort(sqrdists, dim=-1, descending=True)
    idx = indices[:, :, pad: nsample + pad]
    # sdist = sorted_dist[:,:,pad: nsample+pad]
    return idx.int()


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint))  # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        '''
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        '''
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(xyz, points, self.npoint, self.nsample,
                                                                         self.use_xyz, idx=idx)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:MLP_CONV
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous())
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0 / dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx, weight)  # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class Extrapolation(nn.Module):
    def __init__(self, dim_feat, in_channel, out_channel, dim):
        super(Extrapolation, self).__init__()

        self.reverse_mapping = MLP_Res(in_dim=dim_feat + dim, hidden_dim=in_channel, out_dim=in_channel)
        self.mapping = MLP_Res(in_dim=dim_feat + in_channel, hidden_dim=dim, out_dim=dim)

        # self.mapping = nn.Sequential(
        #     nn.Conv1d(in_channel+dim_feat, dim_feat, 1),
        #     #nn.BatchNorm1d(dim),
        #     nn.ReLU(),
        #     nn.Conv1d(dim_feat, dim, 1))

        # self.reverse_mapping = nn.Sequential(
        #     nn.Conv1d(dim_feat+dim, dim_feat, 1),
        #     #nn.BatchNorm1d(in_channel),
        #     nn.ReLU(),
        #     nn.Conv1d(dim_feat, in_channel, 1))

        self.translate_mlp = MLP_CONV(in_channel=dim_feat, layer_dims=[dim, dim])
        self.conv_scale = nn.Conv1d(dim, 1, 1)
        self.conv_origin = nn.Conv1d(dim, dim, 1)

        self.ps_2 = nn.ConvTranspose1d(in_channel, out_channel, 2, 2, bias=True)
        # self.mlp_fea = MLP_CONV(in_channel=in_channel, layer_dims=[out_channel, out_channel])
        self.mlp_dxyz = MLP_Res(in_dim=out_channel, hidden_dim=32, out_dim=3)

    def forward(self, fea, glo_fea):
        """
        Inputs:
            pos: (B, 3, N)
            fea: (B, in_channel, N)
        """
        b, c, n = fea.shape
        # value = fea
        value = self.mapping(torch.cat([fea, glo_fea.repeat(1, 1, n)], 1))  # (B, dim, N)

        dup_value = self.reverse_mapping(torch.cat([value, glo_fea.repeat(1, 1, n)], 1))  # (B, out_dim, N*up_factor)

        temp = self.translate_mlp(glo_fea)
        scale = (self.conv_scale(temp))
        shift = self.conv_origin(temp)

        avg = torch.mean(value, dim=2, keepdim=True)
        # shiffting
        value = (1 + scale) * value - scale * shift - avg

        # upsample
        # up_value = self.ps(value)
        extrapolated = self.reverse_mapping(
            torch.cat([value, glo_fea.repeat(1, 1, n)], 1))  # (B, out_dim, N*up_factor)

        # output point cloud
        gen_fea = self.ps_2(torch.cat([fea, extrapolated], dim=-1))
        child_xyz = torch.tanh(self.mlp_dxyz(torch.relu(gen_fea)))  # (B, 3, N_prev * up_factor)

        constr = dup_value - fea

        return child_xyz, gen_fea, constr


class CrossTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=8, pos_hidden_dim=64, attn_hidden_multiplier=2):
        super(CrossTransformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1, bias=False)
        self.conv_query = nn.Conv1d(in_channel, dim, 1, bias=False)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.value_mlp = nn.Sequential(
            nn.Conv1d(in_channel * 2, in_channel, 1),
            # nn.BatchNorm1d(in_channel),
            nn.ReLU(),
            nn.Conv1d(in_channel, in_channel, 1)
        )

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, pos_flipped, fea, prev_fea, idx_knn=None):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        value = self.value_mlp(torch.cat([fea, prev_fea], dim=1))
        identity = value
        b, dim, n = pos.shape

        if idx_knn == None:
            # pos_flipped = pos.permute(0, 2, 1).contiguous()
            idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        query = self.conv_query(fea)
        key = self.conv_key(prev_fea)
        value = self.conv_value(value)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn
        value = grouping_operation(value,
                                   idx_knn) + pos_embedding  # value.reshape((b, -1, n, 1)) +pos_embedding #grouping_operation(value, idx_knn)

        weights = self.attn_mlp(qk_rel + pos_embedding)  # (B, dim, N*up_factor, k)
        weights = torch.softmax(weights, -1)

        agg = einsum('b c i j, b c i j -> b c i', weights, value)  # b, dim, n
        y = self.linear_end(agg)

        return 0.5 * (y + identity)


class FeaProp(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=2):
        super(FeaProp, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(4, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, pos_flipped, fea, seed, seed_fea):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = fea

        value = self.linear_start(seed_fea)
        b, dim, n = pos.shape

        # pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, seed, pos_flipped)

        key = self.conv_key(value)
        value = self.conv_value(value)
        query = self.conv_query(fea)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        # pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(seed, idx_knn)  # b, 3, n, n_knn
        # pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn
        value = grouping_operation(value, idx_knn)

        coord_xi = pos.view(b, -1, n, 1).repeat(1, 1, 1, self.n_knn)
        coord_xj = grouping_operation(seed.permute(0, 2, 1).contiguous(), idx_knn)
        pos_rel = coord_xi - coord_xj  # b, 3, n, n_knn
        dis_xi_xj = torch.norm(pos_rel, p=2, dim=1).unsqueeze(1)
        h_xi_xj = torch.cat((dis_xi_xj, pos_rel), dim=1)

        pos_embedding = self.pos_mlp(h_xi_xj)

        weights = self.attn_mlp(qk_rel + pos_embedding)  # (B, dim, N*up_factor, k)
        weights = torch.softmax(weights, -1)

        agg = einsum('b c i j, b c i j -> b c i', weights, value + pos_embedding)  # b, dim, n
        y = self.linear_end(agg)

        return (y + identity)


class Interpolation(nn.Module):
    def __init__(self, in_channel, dim=64, scale=0.2, n_knn=16, up_factor=2,
                 pos_hidden_dim=32, attn_hidden_multiplier=4):
        super(Interpolation, self).__init__()
        self.n_knn = n_knn
        self.scale = scale
        self.up_factor = up_factor  # -1 if up_factor >1 else 1

        self.CrossA = CrossTransformer(in_channel, dim=64, n_knn=8)
        self.propa = FeaProp(in_channel, dim=64, n_knn=8)
        self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[32, 64])
        # self.mlp = Conv2d(64, 64, if_bn=True)
        self.mlp_2 = MLP_CONV(in_channel=in_channel + 64 * 2, layer_dims=[64 * 2, in_channel])

        self.mapping = MLP_Res(in_dim=in_channel * 2, hidden_dim=in_channel, out_dim=in_channel)
        # self.mapping = nn.Sequential(
        #     nn.Conv1d(in_channel*2, in_channel*2, 1),
        #     #nn.BatchNorm1d(in_channel),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channel*2, in_channel, 1))

        self.rev_mapping = MLP_Res(in_dim=in_channel * 2, hidden_dim=in_channel, out_dim=in_channel)
        # self.rev_mapping = nn.Sequential(
        #     nn.Conv1d(in_channel*2, in_channel*2, 1),
        #     #nn.BatchNorm1d(in_channel),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channel*2, in_channel, 1)
        # )

        self.conv_start = nn.Conv1d(in_channel, dim, 1)
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

        self.conv_key = nn.Conv1d(in_channel, dim, 1, bias=False)
        self.conv_query = nn.Conv1d(in_channel, dim, 1, bias=False)

        # self.pos_mlp = nn.Sequential(
        #    nn.Conv2d(4, pos_hidden_dim, 1),
        #    nn.BatchNorm2d(pos_hidden_dim),
        #    nn.ReLU(),
        #    nn.Conv2d(pos_hidden_dim, dim, 1)
        # )
        # self.pos_emb = nn.Conv1d(3, dim, 1)
        self.pos_emb = nn.Sequential(
            nn.Conv1d(3, pos_hidden_dim, 1),
            # nn.BatchNorm1d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv1d(pos_hidden_dim, dim, 1)
        )

        # self.prox_mlp = nn.Conv2d(dim,dim,1)

        self.prox_mlp = nn.Sequential(
            nn.Conv2d(dim, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, dim, 1, bias=False)
        )

        # score layers
        self.weight_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            # nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
            nn.ConvTranspose2d(dim * attn_hidden_multiplier, dim, (self.up_factor, 1), (self.up_factor, 1), bias=True))

        self.mlp_dxyz = MLP_Res(in_dim=in_channel, hidden_dim=32, out_dim=3)

    def forward(self, pos, fea, seed_flipped, seed_feat):
        """
        Inputs:
            pos: (B, 3, N)
            fea: (B, in_channel, N)
            seed_flipped: (B,M,3)

        """

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        # ################# Seed Feature Propogation
        if fea == None:
            dist, idx = three_nn(pos_flipped, seed_flipped)
            dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
            recip_dist = 1.0 / dist
            norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
            weight = recip_dist / norm
            prop_fea = three_interpolate(seed_feat, idx, weight)  # B, in_channel, N
            fea = prop_fea

        else:
            prop_fea = self.propa(pos, pos_flipped, fea, seed_flipped, seed_feat)

        fea = self.CrossA(pos, pos_flipped, prop_fea, fea)
        ################ Query mlps
        feat_1 = self.mlp_1(pos)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            prop_fea], 1)
        Q = self.mlp_2(feat_1)

        ################ Mapping to linear addable space
        value = self.mapping(torch.cat([Q, fea], dim=1))

        dup_value = self.rev_mapping(torch.cat([Q, value], 1))  # (B, out_dim, N*up_factor)
        cons = value - dup_value

        identity = value
        key = self.conv_key(value)  # (B, dim, N)
        query = self.conv_query(Q)
        value = self.conv_start(value)

        b, dim, n = value.shape
        # geo_fea = geo_fea.view(b,-1,n,1).repeat(1,1,1,self.n_knn)

        ################ Query,Key relevance
        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        proxyfeat = self.conv_upfeat(fea) + self.pos_emb(pos)  # (B, dim, N)
        proxy_rel = grouping_operation(proxyfeat, idx_knn) - proxyfeat.reshape((b, -1, n, 1))  # (B, dim, N, k)
        proxy_rel = self.prox_mlp(proxy_rel)
        # upfeat = self.conv_upfeat(feat_upsample) # (B, dim, N)
        # upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn) # (B, dim, N, k)

        ############### Interpolation Weights
        score = self.weight_mlp(qk_rel + proxy_rel)  # (B, dim*up_factor, N, k)

        # softmax function
        score = torch.softmax(score, -1)

        ############### Linear Interoplation
        value = grouping_operation(value, idx_knn) - value.view(b, -1, n, 1) + proxy_rel  # (B, dim, N, k)
        value = torch.repeat_interleave(value, self.up_factor, dim=2)  # (B, dim, N*up_factor, k)

        agg = torch.einsum('b c i j, b c i j -> b c i', score, value)  # (B, dim, N*up_factor)
        # agg = agg.view(b,-1,n*self.up_factor)

        identity = torch.repeat_interleave(identity, self.up_factor, dim=2)  # (B, out_dim, N*up_factor)
        y = self.conv_end(agg) + identity

        # Reverse Mapping
        Q_up = torch.repeat_interleave(Q, self.up_factor, dim=-1)

        # Output point cloud
        delta = self.scale * torch.tanh(self.mlp_dxyz(torch.relu(y)))  # (B, 3, N_prev * up_factor)
        inter_pos = delta + torch.repeat_interleave(pos, self.up_factor, dim=-1)

        # inter_fea = y+torch.repeat_interleave(identity,self.up_factor,dim=-1)
        interpolated = self.rev_mapping(torch.cat([Q_up, y], dim=1))  # (B, out_dim, N*up_factor)
        # child_pos = torch.cat([inter_pos,pos],dim=-1).contiguous()
        # child_fea = torch.cat([inter_fea,fea],dim=-1).contiguous()
        return inter_pos, interpolated, cons




