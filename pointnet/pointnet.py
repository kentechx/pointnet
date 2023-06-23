import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val


class STN(nn.Module):
    # perform spatial transformation in n-dimensional space

    def __init__(self, in_dim=3, out_nd=None, head_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_nd = default(out_nd, in_dim)

        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1024, 1, bias=False),
        )

        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(1024)
        self.act = nn.GELU()

        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            norm(512),
            nn.GELU(),
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Linear(256, self.out_nd ** 2),
        )

        nn.init.normal_(self.head[-1].weight, 0, 0.001)
        nn.init.eye_(self.head[-1].bias.view(in_dim, in_dim))

    def forward(self, x):
        # x: (b, d, n)
        x = self.net(x)
        x = torch.max(x, dim=-1, keepdim=False)[0]
        x = self.act(self.norm(x))

        x = self.head(x)
        x = rearrange(x, "b (x y) -> b x y", x=self.out_nd, y=self.out_nd)
        return x


class PointNetCls(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            *,
            stn_3d=STN(in_dim=3),  # if None, no stn_3d
            with_head=True,
            head_norm=True,
            dropout=0.3,
    ):
        super().__init__()
        self.with_head = with_head

        # if using stn, put other features behind xyz
        self.stn_3d = stn_3d

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.stn_nd = STN(in_dim=64, head_norm=head_norm)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1024, 1, bias=False),
        )

        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(1024)
        self.act = nn.GELU()

        if self.with_head:
            self.head = nn.Sequential(
                nn.Linear(1024, 512, bias=False),
                norm(512),
                nn.GELU(),
                nn.Linear(512, 256, bias=False),
                norm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, out_dim),
            )

    def forward(self, x):
        # x: (b, d, n)
        if exists(self.stn_3d):
            transform_3d = self.stn_3d(x)
            if x.size(1) == 3:
                x = torch.bmm(transform_3d, x)
            elif x.size(1) > 3:
                x = torch.cat([torch.bmm(transform_3d, x[:, :3]), x[:, 3:]], dim=1)
            else:
                raise ValueError(f"invalid input dimension: {x.size(1)}")

        x = self.conv1(x)
        transform_nd = self.stn_nd(x)
        x = torch.bmm(transform_nd, x)
        x = self.conv2(x)

        x = torch.max(x, dim=-1, keepdim=False)[0]
        x = self.act(self.norm(x))

        if self.with_head:
            x = self.head(x)
        return x


class PointNetSeg(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            *,
            stn_3d=STN(in_dim=3),  # if None, no stn_3d
            global_head_norm=True,  # if using normalization in the global head, disable it if batch size is 1
    ):
        super().__init__()

        self.backbone = PointNetCls(in_dim=in_dim,
                                    out_dim=out_dim,
                                    stn_3d=stn_3d,
                                    head_norm=global_head_norm,
                                    with_head=False)

        self.head = nn.Sequential(
            nn.Conv1d(1024 + 64, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, out_dim, 1),
        )

    def forward_backbone(self, x):
        # x: (b, d, n)
        if exists(self.backbone.stn_3d):
            transform_3d = self.backbone.stn_3d(x)
            if x.size(1) == 3:
                x = torch.bmm(transform_3d, x)
            elif x.size(1) > 3:
                x = torch.cat([torch.bmm(transform_3d, x[:, :3]), x[:, 3:]], dim=1)
            else:
                raise ValueError(f"invalid input dimension: {x.size(1)}")

        x = self.backbone.conv1(x)
        transform_nd = self.backbone.stn_nd(x)
        x = torch.bmm(transform_nd, x)

        global_feat = self.backbone.conv2(x)
        global_feat = torch.max(global_feat, dim=-1, keepdim=False)[0]
        global_feat = self.backbone.act(self.backbone.norm(global_feat))
        return x, global_feat

    def forward(self, x):
        # x: (b, d, n)
        x, global_feat = self.forward_backbone(x)
        global_feat = repeat(global_feat, "b d -> b d n", n=x.size(-1))
        x = torch.cat([x, global_feat], dim=1)
        x = self.head(x)
        return x
