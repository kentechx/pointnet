from collections import namedtuple
from typing import Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from .utils import farthest_point_sampling, ball_query_pytorch
from .pointnet import STN

# whether to use taichi for ball query
TAICHI = False


def enable_taichi():
    import taichi as ti
    global TAICHI
    TAICHI = True
    ti.init(ti.cuda)


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
    if TAICHI:
        from .taichi import ball_query
        return ball_query(src, query, radius, k)
    else:
        return ball_query_pytorch(src, query, radius, k)


def cdist(x, y=None):
    # perform cdist in dimension 1
    # x: (b, d, n)
    # y: (b, d, m)
    if exists(y):
        x = rearrange(x, 'b d n -> b n d')
        y = rearrange(y, 'b d m -> b m d')
        return torch.cdist(x, y)
    else:
        x = rearrange(x, 'b d n -> b n d')
        return torch.cdist(x, x)


def knn(src, query, k):
    dists = cdist(query, src)  # (b, m, n)
    idx = dists.topk(k, dim=-1, largest=False, sorted=False)[1]  # (b, m, k)
    dists = dists.gather(-1, idx)  # (b, m, k)
    return idx, dists


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


def downsample_fps(xyz, n_sample):
    # xyz: (b, 3, n)
    _xyz = rearrange(xyz, 'b d n -> b n d')
    sample_ind = farthest_point_sampling(_xyz, n_sample, start_idx=0)  # (b, k)
    sample_xyz = xyz.gather(-1, repeat(sample_ind, 'b k -> b d k', d=xyz.shape[1]))  # (b, 3, k)
    return SampleResult(None, sample_xyz, sample_ind, None)


class SABlock(nn.Module):
    """
    Set abstraction block without downsampling.
    """

    def __init__(
            self,
            in_dim,
            dims: Union[Iterable[int], Iterable[Iterable[int]]] = (64, 64, 128),
            radius: Union[float, Iterable[float]] = 0.2,
            k: Union[int, Iterable[int]] = 32
    ):
        super().__init__()
        self.dims_list = dims if isinstance(dims[0], Iterable) else [dims]
        self.radius_list = radius if isinstance(radius, Iterable) else [radius]
        self.k_list = k if isinstance(k, Iterable) else [k]

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
            neighbor_idx = _ball_query(src_xyz, xyz, radius, k)[0]  # (b, m, k)
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


class UpBlock(nn.Module):

    def __init__(self, in_dim, dims: Iterable[int] = (256, 256), k=3, eps=1e-5):
        super().__init__()
        self.k = k
        self.eps = eps
        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(in_d, out_d, 1, bias=False),
                          nn.BatchNorm1d(out_d),
                          nn.GELU())
            for in_d, out_d in zip([in_dim, *dims[:-1]], dims)
        ])

    def route(self, src_x, src_xyz, dst_x, dst_xyz, neighbor_idx=None, dists=None):
        # use knn and weighted average to get the features
        if not exists(neighbor_idx):
            neighbor_idx, dists = knn(src_xyz, dst_xyz, self.k)  # (b, m, k)

        weights = 1. / (dists + self.eps)  # (b, m, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (b, m, k)

        neighbor_x = gather(src_x, neighbor_idx)  # (b, d, m, k)
        neighbor_x = (weights[:, None] * neighbor_x).sum(dim=-1)  # (b, d, m)

        dst_x = torch.cat([dst_x, neighbor_x], dim=1)  # (b, d+d', m)
        return dst_x

    def forward(self, ori_x, ori_xyz, sub_x, sub_xyz):
        ori_x = self.route(sub_x, sub_xyz, ori_x, ori_xyz)
        ori_x = self.conv(ori_x)
        return ori_x


class PointNet2ClsSSG(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            *,
            downsample_points=(512, 128),
            radii=(0.2, 0.4),
            ks=(32, 64),
            head_norm=True,
            dropout=0.5,
    ):
        super().__init__()
        self.downsample_points = downsample_points

        self.sa1 = SABlock(in_dim, [64, 64, 128], radii[0], ks[0])
        self.sa2 = SABlock(128, [128, 128, 256], radii[1], ks[1])
        self.global_sa = nn.Sequential(
            nn.Conv1d(256, 256, 1, bias=False),
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
        xyz1 = downsample_fps(xyz, self.downsample_points[0]).xyz
        x1 = self.sa1(x, xyz, xyz1)

        xyz2 = downsample_fps(xyz1, self.downsample_points[1]).xyz
        x2 = self.sa2(x1, xyz1, xyz2)

        x3 = self.global_sa(x2)
        out = x3.max(-1)[0]
        out = self.act(self.norm(out))
        out = self.head(out)
        return out


class PointNet2ClsMSG(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            *,
            downsample_points=(512, 128),
            base_radius=0.1,
            base_k=16,
            head_norm=True,
            dropout=0.5,
    ):
        super().__init__()
        self.downsample_points = downsample_points

        radii1 = [base_radius, base_radius * 2, base_radius * 4]
        ks1 = [base_k, base_k * 2, base_k * 8]
        self.sa1 = SABlock(in_dim, [[32, 32, 64], [64, 64, 128], [64, 96, 128]], radii1, ks1)

        radii2 = [base_radius * 2, base_radius * 4, base_radius * 8]
        ks2 = [base_k * 2, base_k * 4, base_k * 8]
        self.sa2 = SABlock(320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]], radii2, ks2)

        self.global_sa = nn.Sequential(
            nn.Conv1d(640, 256, 1, bias=False),
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
        xyz1 = downsample_fps(xyz, self.downsample_points[0]).xyz
        x1 = self.sa1(x, xyz, xyz1)

        xyz2 = downsample_fps(xyz1, self.downsample_points[1]).xyz
        x2 = self.sa2(x1, xyz1, xyz2)

        x3 = self.global_sa(x2)
        out = x3.max(-1)[0]
        out = self.act(self.norm(out))
        out = self.head(out)
        return out


class PointNet2PartSegSSG(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            n_category=16,
            *,
            downsample_points=(512, 128),
            global_norm=True,
            dropout=0.5,
    ):
        super().__init__()
        self.downsample_points = downsample_points
        self.n_category = n_category

        self.sa_blocks = nn.ModuleList([
            SABlock(in_dim, [64, 64, 128], 0.2, 32),
            SABlock(128, [128, 128, 256], 0.4, 64)
        ])

        self.global_sa = nn.Sequential(
            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 1024, 1, bias=False),
        )

        norm = nn.BatchNorm1d if global_norm else nn.Identity
        self.norm = norm(1024)
        self.act = nn.GELU()

        self.up_blocks = nn.ModuleList([
            UpBlock(1024 + 256, [256, 256], k=1),
            UpBlock(256 + 128, [256, 128], k=3),
            UpBlock(128 + in_dim, [128, 128, 128], k=3)
        ])

        self.category_emb = nn.Embedding(n_category, 128)

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, out_dim, 1),
        )

    def forward(self, x, xyz, category):
        # x: (b, c, n)
        # xyz: (b, 3, n)
        # category: (b,)
        xs = [x]
        xyzs = [xyz]
        for i, sa_block in enumerate(self.sa_blocks):
            xyz = downsample_fps(xyzs[-1], self.downsample_points[i]).xyz
            x = sa_block(xs[-1], xyzs[-1], xyz)
            xs.append(x)
            xyzs.append(xyz)

        xyz = xyzs[-1].mean(-1, keepdim=True)
        x = self.global_sa(xs[-1])
        x = x.max(-1, keepdim=True)[0]
        x = self.act(self.norm(x))

        for up_block in self.up_blocks:
            ori_x, ori_xyz = xs.pop(), xyzs.pop()
            x = up_block(ori_x, ori_xyz, x, xyz)
            xyz = ori_xyz

        category_emb = repeat(self.category_emb(category), 'b c -> b c n', n=x.shape[2])
        x = x + category_emb
        out = self.head(x)
        return out


class PointNet2PartSegMSG(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            n_category=16,
            *,
            downsample_points=(512, 128),
            global_norm=True,
            dropout=0.5,
    ):
        super().__init__()
        self.downsample_points = downsample_points
        self.n_category = n_category

        self.sa_blocks = nn.ModuleList([
            SABlock(in_dim, [[32, 32, 64], [64, 64, 128], [64, 96, 128]], [0.1, 0.2, 0.4], [32, 64, 128]),
            SABlock(320, [[128, 128, 256], [128, 196, 256]], [0.4, 0.8], [64, 128])
        ])

        self.global_sa = nn.Sequential(
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 1024, 1, bias=False),
        )

        norm = nn.BatchNorm1d if global_norm else nn.Identity
        self.norm = norm(1024)
        self.act = nn.GELU()

        self.up_blocks = nn.ModuleList([
            UpBlock(1024 + 512, [256, 256], k=1),
            UpBlock(320 + 256, [256, 128], k=3),
            UpBlock(128 + in_dim, [128, 128], k=3)
        ])

        self.category_emb = nn.Embedding(n_category, 128)

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, out_dim, 1),
        )

    def forward(self, x, xyz, category):
        # x: (b, c, n)
        # xyz: (b, 3, n)
        # category: (b,)
        xs = [x]
        xyzs = [xyz]
        for i, sa_block in enumerate(self.sa_blocks):
            xyz = downsample_fps(xyzs[-1], self.downsample_points[i]).xyz
            x = sa_block(xs[-1], xyzs[-1], xyz)
            xs.append(x)
            xyzs.append(xyz)

        xyz = xyzs[-1].mean(-1, keepdim=True)
        x = self.global_sa(xs[-1])
        x = x.max(-1, keepdim=True)[0]
        x = self.act(self.norm(x))

        for up_block in self.up_blocks:
            ori_x, ori_xyz = xs.pop(), xyzs.pop()
            x = up_block(ori_x, ori_xyz, x, xyz)
            xyz = ori_xyz

        category_emb = repeat(self.category_emb(category), 'b c -> b c n', n=x.shape[2])
        x = x + category_emb
        out = self.head(x)
        return out


class PointNet2SegSSG(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            k=32,
            *,
            downsample_points=(1024, 256, 64, 16),
            base_radius=0.1,
            dropout=0.5
    ):
        super().__init__()
        self.downsample_points = downsample_points

        self.sa_blocks = nn.ModuleList([
            SABlock(in_dim, [32, 32, 64], base_radius, k),
            SABlock(64, [64, 64, 128], base_radius * 2, k),
            SABlock(128, [128, 128, 256], base_radius * 4, k),
            SABlock(256, [256, 256, 512], base_radius * 8, k)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock(512 + 256, [256, 256], k=3),
            UpBlock(256 + 128, [256, 256], k=3),
            UpBlock(256 + 64, [256, 128], k=3),
            UpBlock(128 + in_dim, [128, 128], k=3)
        ])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, out_dim, 1),
        )

    def forward(self, x, xyz):
        # x: (b, c, n)
        # xyz: (b, 3, n)
        xs = [x]
        xyzs = [xyz]
        for i, sa_block in enumerate(self.sa_blocks):
            xyz = downsample_fps(xyzs[-1], self.downsample_points[i]).xyz
            x = sa_block(xs[-1], xyzs[-1], xyz)
            xs.append(x)
            xyzs.append(xyz)

        x, xyz = xs.pop(), xyzs.pop()
        for up_block in self.up_blocks:
            ori_x, ori_xyz = xs.pop(), xyzs.pop()
            x = up_block(ori_x, ori_xyz, x, xyz)
            xyz = ori_xyz

        out = self.head(x)
        return out


class PointNet2SegMSG(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            ks=(16, 32),
            *,
            downsample_points=(1024, 256, 64, 16),
            base_radii=(0.05, 0.1),
            dropout=0.5
    ):
        super().__init__()
        self.downsample_points = downsample_points

        self.sa_blocks = nn.ModuleList([
            SABlock(in_dim, [[16, 16, 32], [32, 32, 64]], base_radii, ks),
            SABlock(32 + 64, [[64, 64, 128], [64, 96, 128]], [r * 2 for r in base_radii], ks),
            SABlock(128 + 128, [[128, 196, 256], [128, 196, 256]], [r * 4 for r in base_radii], ks),
            SABlock(256 + 256, [[256, 256, 512], [256, 384, 512]], [r * 8 for r in base_radii], ks)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock(512 + 512 + 256 + 256, [256, 256], k=3),
            UpBlock(256 + 128 + 128, [256, 256], k=3),
            UpBlock(256 + 64 + 32, [256, 128], k=3),
            UpBlock(128 + in_dim, [128, 128], k=3)
        ])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, out_dim, 1),
        )

    def forward(self, x, xyz):
        # x: (b, c, n)
        # xyz: (b, 3, n)
        xs = [x]
        xyzs = [xyz]
        for i, sa_block in enumerate(self.sa_blocks):
            xyz = downsample_fps(xyzs[-1], self.downsample_points[i]).xyz
            x = sa_block(xs[-1], xyzs[-1], xyz)
            xs.append(x)
            xyzs.append(xyz)

        x, xyz = xs.pop(), xyzs.pop()
        for up_block in self.up_blocks:
            ori_x, ori_xyz = xs.pop(), xyzs.pop()
            x = up_block(ori_x, ori_xyz, x, xyz)
            xyz = ori_xyz

        out = self.head(x)
        return out
