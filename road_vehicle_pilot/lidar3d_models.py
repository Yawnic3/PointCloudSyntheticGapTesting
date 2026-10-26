"""Small raw-XYZ pillar CNN and window-attention hybrid, not official replicas."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CELL = 0.4
NX, NY = 400, 200


def conv(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU()
    )


class PointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

    def forward(self, clouds):
        device = clouds[0].device
        points = torch.cat(clouds)
        batch = torch.cat(
            [
                torch.full((len(p),), i, device=device, dtype=torch.long)
                for i, p in enumerate(clouds)
            ]
        )
        ix = (points[:, 0] / CELL).long().clamp(0, NX - 1)
        iy = ((points[:, 1] + 40) / CELL).long().clamp(0, NY - 1)
        index = batch * NX * NY + ix * NY + iy
        size = len(clouds) * NX * NY
        sums = torch.zeros((size, 3), device=device)
        sums.index_add_(0, index, points.float())
        counts = torch.bincount(index, minlength=size).float().clamp_min(1)
        means = sums / counts[:, None]
        center = torch.stack([(ix + 0.5) * CELL, (iy + 0.5) * CELL - 40], dim=1)
        scale = points.new_tensor([80, 40, 4])
        shift = points.new_tensor([80, 0, 0])
        features = torch.cat(
            [
                (points - shift) / scale,
                (points - means[index]) / points.new_tensor([CELL, CELL, 2]),
                (points[:, :2] - center) / CELL,
                torch.log1p(counts[index, None]) / 6,
            ],
            dim=1,
        )
        encoded = self.encoder(features)
        dense = torch.zeros((size, 32), device=device)
        dense.scatter_reduce_(
            0,
            index[:, None].expand(-1, 32),
            encoded.float(),
            reduce="amax",
            include_self=True,
        )
        return dense.view(len(clouds), NX, NY, 32).permute(0, 3, 1, 2).contiguous()


class WindowAttention(nn.Module):
    def __init__(self, channels=96, window=8, shift=0):
        super().__init__()
        self.window = window
        self.shift = shift
        self.heads = 4
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.bias = nn.Parameter(torch.zeros(4, (2 * window - 1) ** 2))
        xy = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(window), torch.arange(window), indexing="ij"
                )
            )
            .flatten(1)
            .T
        )
        delta = xy[:, None, :] - xy[None, :, :] + window - 1
        self.register_buffer(
            "relative_index", delta[:, :, 0] * (2 * window - 1) + delta[:, :, 1]
        )

    def forward(self, image):
        b, c, h, w = image.shape
        s = self.shift
        win = self.window
        bottom = (-(h + s)) % win
        right = (-(w + s)) % win
        padded = F.pad(image, (s, right, s, bottom))
        hp, wp = padded.shape[2:]
        tokens = (
            padded.permute(0, 2, 3, 1)
            .reshape(b, hp // win, win, wp // win, win, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, win * win, c)
        )
        valid = (
            F.pad(torch.ones((b, 1, h, w), device=image.device), (s, right, s, bottom))
            .reshape(b, 1, hp // win, win, wp // win, win)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, win * win)
            > 0
        )
        normalized = self.norm1(tokens)
        q, k, v = (
            self.qkv(normalized)
            .reshape(-1, win * win, 3, 4, c // 4)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        bias = self.bias[:, self.relative_index].unsqueeze(0).to(q.dtype)
        mask = bias.expand(len(tokens), -1, -1, -1).clone()
        mask = mask.masked_fill(~valid[:, None, None, :], -10000)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        tokens = tokens + self.proj(attention.transpose(1, 2).reshape(-1, win * win, c))
        tokens = tokens + self.mlp(self.norm2(tokens))
        result = (
            tokens.reshape(b, hp // win, wp // win, win, win, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, hp, wp)
        )
        return result[:, :, s : s + h, s : s + w]


class PillarDetector(nn.Module):
    def __init__(self, kind="pillar_cnn"):
        super().__init__()
        assert kind in ["pillar_cnn", "pillar_transformer"]
        self.kind = kind
        self.pillars = PointEncoder()
        self.stem = nn.Sequential(conv(32, 32), conv(32, 32))
        self.down = nn.Sequential(conv(32, 64, 2), conv(64, 64))
        self.bottom = nn.Sequential(
            conv(64, 96, 2),
            *(
                [conv(96, 96), conv(96, 96)]
                if kind == "pillar_cnn"
                else [WindowAttention(96, 8, 0), WindowAttention(96, 8, 4)]
            )
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.up4 = nn.ConvTranspose2d(96, 32, 4, 4)
        self.fuse = nn.Sequential(conv(96, 48), conv(48, 48))
        self.heat = nn.Conv2d(48, 1, 1)
        self.box = nn.Conv2d(48, 8, 1)
        nn.init.normal_(self.heat.weight, std=0.001)
        nn.init.constant_(self.heat.bias, -4.6)
        nn.init.normal_(self.box.weight, std=0.001)
        with torch.no_grad():
            self.box.bias.copy_(
                torch.tensor(
                    [0, 0, 0.5, math.log(4.3), math.log(1.8), math.log(1.7), 0, 1]
                )
            )

    def forward(self, clouds):
        first = self.stem(self.pillars(clouds))
        second = self.down(first)
        third = self.bottom(second)
        features = self.fuse(
            torch.cat([first, self.up2(second), self.up4(third)], dim=1)
        )
        return self.heat(features), self.box(features)


def targets(boxes, device):
    heat = np.zeros((len(boxes), 1, NX, NY), dtype=np.float32)
    records = []
    indices = []
    for batch, items in enumerate(boxes):
        occupied = set()
        for values in items:
            x, y, z, l, w, h, angle = map(float, values)
            gx, gy = x / CELL, (y + 40) / CELL
            ix, iy = int(gx), int(gy)
            if not (0 <= ix < NX and 0 <= iy < NY):
                continue
            if (ix, iy) in occupied:
                continue
            occupied.add((ix, iy))
            radius = max(1, min(6, int(min(l, w) / CELL / 3)))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if 0 <= ix + dx < NX and 0 <= iy + dy < NY:
                        value = math.exp(
                            -(dx * dx + dy * dy) / (2 * ((2 * radius + 1) / 6) ** 2)
                        )
                        heat[batch, 0, ix + dx, iy + dy] = max(
                            heat[batch, 0, ix + dx, iy + dy], value
                        )
            records.append(
                [
                    gx - ix,
                    gy - iy,
                    z / 2,
                    math.log(l),
                    math.log(w),
                    math.log(h),
                    math.sin(2 * angle),
                    math.cos(2 * angle),
                ]
            )
            indices.append([batch, ix, iy])
    return (
        torch.tensor(heat, device=device),
        torch.tensor(records, device=device, dtype=torch.float32).reshape(-1, 8),
        torch.tensor(indices, device=device, dtype=torch.long).reshape(-1, 3),
    )


def loss(outputs, truth):
    logits, reg = outputs
    heat, values, index = truth
    probability = logits.float().sigmoid().clamp(1e-5, 1 - 1e-5)
    pos = heat == 1
    neg = heat < 1
    focal = -(
        pos * (1 - probability) ** 2 * probability.log()
        + neg * (1 - heat) ** 4 * probability**2 * torch.log1p(-probability)
    ).sum() / max(1, len(values))
    if len(values):
        selected = reg.permute(0, 2, 3, 1)[
            index[:, 0], index[:, 1], index[:, 2]
        ].float()
        selected = torch.cat([selected[:, :2].sigmoid(), selected[:, 2:]], dim=1)
        weight = selected.new_tensor([2, 2, 1, 1, 1, 1, 0.5, 0.5])
        regression = (
            (F.smooth_l1_loss(selected, values, reduction="none", beta=0.1) * weight)
            .sum(dim=1)
            .mean()
        )
    else:
        regression = reg.sum() * 0
    return focal + 2 * regression, {
        "heat": float(focal.detach()),
        "box": float(regression.detach()),
    }


@torch.no_grad()
def decode(outputs, minimum=0.01, topk=100):
    logits, reg = outputs
    p = logits.float().sigmoid()
    p = p * (p == F.max_pool2d(p, 3, 1, 1))
    result = []
    for batch in range(len(p)):
        scores, index = p[batch, 0].flatten().topk(topk)
        mask = scores >= minimum
        scores = scores[mask]
        index = index[mask]
        ix = index // NY
        iy = index % NY
        v = reg[batch, :, ix, iy].T.float()
        offset = v[:, :2].sigmoid()
        dims = v[:, 3:6].clamp(math.log(0.3), math.log(25)).exp()
        boxes = torch.cat(
            [
                ((ix.float() + offset[:, 0]) * CELL)[:, None],
                ((iy.float() + offset[:, 1]) * CELL - 40)[:, None],
                (v[:, 2] * 2).clamp(-3, 8)[:, None],
                dims,
                (torch.atan2(v[:, 6], v[:, 7]) / 2)[:, None],
            ],
            dim=1,
        )
        result.append(
            [
                {"score": float(s), "box": b.tolist()}
                for s, b in zip(scores.cpu(), boxes.cpu())
            ]
        )
    return result
