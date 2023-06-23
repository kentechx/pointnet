import numpy as np
from collections import namedtuple
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from .taichi import ball_query
from .utils import farthest_point_sampling
from .pointnet import STN


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val


def _ball_query(src, query, radius, k):
    # conduct ball query on dim 1
    src = rearrange(src, 'b d n -> b n d')
    query = rearrange(query, 'b d m -> b m d')
    return ball_query(src, query, radius, k)


def gather(x, idx):
    # x: (b, d, n)
    # idx: (b, m, k)
    # output: (b, d, m, k)
    m = idx.shape[1]
    ind = repeat(idx, 'b m k -> b d (m k)', d=x.shape[1])
    out = x.gather(-1, ind)  # (b, d, (m k))
    out = rearrange(out, 'b d (m k) -> b d m k', m=m)
    return out


SampleResult = namedtuple('SampleResult', ['x', 'xyz', 'sample_idx', 'neighbor_idx'])


def downsample_fps(xyz, n_sample, start_idx=None):
    # xyz: (b, 3, n)
    _xyz = rearrange(xyz, 'b d n -> b n d')
    sample_ind = farthest_point_sampling(_xyz, n_sample, start_idx=start_idx)  # (b, k)
    sample_xyz = xyz.gather(-1, repeat(sample_ind, 'b k -> b d k', d=xyz.shape[1]))  # (b, 3, k)
    return SampleResult(None, sample_xyz, sample_ind, None)


class SABlock(nn.Module):
    """
    Set abstraction block without downsampling.
    """

    def __init__(
            self,
            in_dim,
            dims: Union[List[int], List[List[int]]] = [64, 64, 128],
            radius: Union[float, List[float]] = 0.2,
            k: Union[int, List[int]] = 32
    ):
        super().__init__()
        self.dims_list = dims if isinstance(dims[0], list) else [dims]
        self.radius_list = radius if isinstance(radius, list) else [radius]
        self.k_list = k if isinstance(k, list) else [k]

        self.conv_blocks = nn.ModuleList()
        self.last_norms = nn.ModuleList()
        for i, (dims, radius, k) in enumerate(zip(self.dims_list, self.radius_list, self.k_list)):
            dims = [in_dim + 3] + dims
            conv = nn.Sequential(*[
                nn.Sequential(nn.Conv2d(in_d, out_d, 1, bias=False),
                              nn.BatchNorm2d(out_d),
                              nn.GELU())
                for in_d, out_d in zip(dims[:-2], dims[1:-1])
            ])
            conv.append(nn.Conv2d(dims[-2], dims[-1], 1, bias=False))
            self.conv_blocks.append(conv)
            self.last_norms.append(nn.BatchNorm1d(dims[-1]))

    def route(self, src_x, src_xyz, xyz, radius, k, neighbor_idx=None):
        # src_x: (b, d, n)
        # src_xyz: (b, 3, n)
        # xyz: (b, 3, m)
        if not exists(neighbor_idx):
            neighbor_idx = _ball_query(src_xyz, xyz, radius, k)  # (b, m, k)
        neighbor_xyz = gather(src_xyz, neighbor_idx)  # (b, 3, m, k)
        neighbor_xyz -= xyz[..., None]
        x = gather(src_x, neighbor_idx)  # (b, d, m, k)
        x = torch.cat([x, neighbor_xyz], dim=1)  # (b, d+3, m, k)
        return SampleResult(x, xyz, None, neighbor_idx)

    def forward(self, src_x, src_xyz, xyz):
        # src_x: (b, d, n)
        # src_xyz: (b, 3, n)
        # xyz: (b, 3, m)
        # out: (b, d', m)
        xs = []
        for i, (conv_block, norm, radius, k) in enumerate(
                zip(self.conv_blocks, self.last_norms, self.radius_list, self.k_list)):
            out = self.route(src_x, src_xyz, xyz, radius, k)
            x = conv_block(out.x)
            x = x.max(-1)[0]
            x = F.gelu(norm(x))
            xs.append(x)

        return torch.cat(xs, dim=1)


class PointNet2SSGCls(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            *,
            downsampling_factors=(2, 4),
            radii=(0.2, 0.4),
            ks=(32, 64),
            head_norm=True,
            dropout=0.5,
    ):
        super().__init__()
        self.downsampling_factors = downsampling_factors

        self.sa1 = SABlock(in_dim, [64, 64, 128], radii[0], ks[0])
        self.sa2 = SABlock(128 + 3, [128, 128, 256], radii[1], ks[1])
        self.global_sa = nn.Sequential(
            nn.Conv1d(256 + 3, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 1024, 1, bias=False),
        )

        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(1024)
        self.act = nn.GELU()

        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            norm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )

    def forward(self, x, xyz):
        # x: (b, c, n)
        # xyz: (b, 3, n)
        xyz1 = downsample_fps(xyz, xyz.shape[2] // self.downsampling_factors[0]).xyz
        x1 = self.sa1(x, xyz, xyz1)
        x1 = torch.cat([x1, xyz1], dim=1)

        xyz2 = downsample_fps(xyz1, xyz1.shape[2] // self.downsampling_factors[1]).xyz
        x2 = self.sa2(x1, xyz1, xyz2)
        x2 = torch.cat([x2, xyz2], dim=1)

        x3 = self.global_sa(x2)
        out = x3.max(-1)[0]
        out = self.act(self.norm(out))
        out = self.head(out)
        return out


class PointNet2MSGCls(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            *,
            downsampling_factors=(2, 4),
            base_radius=0.1,
            base_k=16,
            head_norm=True,
            dropout=0.5,
    ):
        super().__init__()
        self.downsampling_factors = downsampling_factors

        radii1 = [base_radius, base_radius * 2, base_radius * 4]
        ks1 = [base_k, base_k * 2, base_k * 8]
        self.sa1 = SABlock(in_dim, [[32, 32, 64], [64, 64, 128], [64, 96, 128]], radii1, ks1)

        radii2 = [base_radius * 2, base_radius * 4, base_radius * 8]
        ks2 = [base_k * 2, base_k * 4, base_k * 8]
        self.sa2 = SABlock(320 + 3, [[64, 64, 128], [128, 128, 256], [128, 128, 256]], radii2, ks2)

        self.global_sa = nn.Sequential(
            nn.Conv1d(640 + 3, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 1024, 1, bias=False),
        )

        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(1024)
        self.act = nn.GELU()

        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            norm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )

    def forward(self, x, xyz):
        # x: (b, c, n)
        # xyz: (b, 3, n)
        xyz1 = downsample_fps(xyz, xyz.shape[2] // self.downsampling_factors[0]).xyz
        x1 = self.sa1(x, xyz, xyz1)
        x1 = torch.cat([x1, xyz1], dim=1)

        xyz2 = downsample_fps(xyz1, xyz1.shape[2] // self.downsampling_factors[1]).xyz
        x2 = self.sa2(x1, xyz1, xyz2)
        x2 = torch.cat([x2, xyz2], dim=1)

        x3 = self.global_sa(x2)
        out = x3.max(-1)[0]
        out = self.act(self.norm(out))
        out = self.head(out)
        return out
