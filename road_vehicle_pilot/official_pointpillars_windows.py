"""Windows runtime adapter for published MMDetection3D PointPillars Car weights.

Architecture/math follows the archived OpenMMLab v1.4.0 reference files.
Copyright (c) OpenMMLab; Apache-2.0 license is retained in
pretrained_pointpillars/official/LICENSE. Changes: dependency-free module
construction, NumPy hard voxelization and CPU rotated NMS. No learned layer is
replaced. Railway mode expands spatial support and preserves VFE z reference.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lidar_common import BASE
from lidar3d_geometry import Cuboid, bev_iou, from_seven

OUT = BASE / "pretrained_pointpillars"
CHECKPOINT = next((OUT / "weights").glob("*.pth"))


def voxelize(points, native=True, max_voxels=40000):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[1] == 3:
        points = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])
    xmin, ymin, zmin, xmax, ymax, zmax = (
        [0, -39.68, -3, 69.12, 39.68, 1] if native else [0, -40.32, -5, 160, 40.32, 8]
    )
    nx, ny = (432, 496) if native else (1000, 504)
    valid = (
        (points[:, 0] >= xmin)
        & (points[:, 0] < xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] < ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] < zmax)
    )
    points = points[valid]
    x = np.floor((points[:, 0] - xmin) / 0.16).astype(np.int64)
    y = np.floor((points[:, 1] - ymin) / 0.16).astype(np.int64)
    keys = y * nx + x
    unique, first, inverse, counts = np.unique(
        keys, return_index=True, return_inverse=True, return_counts=True
    )
    selected = np.argsort(first)[:max_voxels]
    mapping = np.full(len(unique), -1, dtype=np.int64)
    mapping[selected] = np.arange(len(selected))
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(np.r_[0, counts[:-1]])
    ranks = np.empty(len(points), dtype=np.int64)
    ranks[order] = np.arange(len(points)) - np.repeat(starts, counts)
    index = mapping[inverse]
    keep = (index >= 0) & (ranks < 32)
    voxels = np.zeros((len(selected), 32, 4), dtype=np.float32)
    voxels[index[keep], ranks[keep]] = points[keep]
    coords = np.zeros((len(selected), 4), dtype=np.int64)
    coords[:, 2] = unique[selected] // nx
    coords[:, 3] = unique[selected] % nx
    return voxels, np.minimum(counts[selected], 32), coords, (nx, ny), ymin


class PFNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(10, 64, bias=False)

    def forward(self, features):
        x = self.linear(features)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        return x.relu().max(dim=1).values


class PillarFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layers = nn.ModuleList([PFNLayer()])

    def forward(self, voxels, counts, coords, ymin):
        # Preserve the official legacy=True aliasing behavior: the first raw
        # XYZ channels become center offsets, and are duplicated at the end.
        features = voxels.clone()
        mean = features[:, :, :3].sum(dim=1, keepdim=True) / counts[:, None, None]
        cluster = features[:, :, :3] - mean
        center = features[:, :, :3]
        center[:, :, 0] -= (coords[:, 3].to(features.dtype) * 0.16 + 0.08)[:, None]
        center[:, :, 1] -= (coords[:, 2].to(features.dtype) * 0.16 + (ymin + 0.08))[
            :, None
        ]
        # Native reference is -1m. In railway mode keep this pretrained
        # feature reference while allowing taller vertical pillar support.
        center[:, :, 2] -= -1.0
        decorated = torch.cat([features, cluster, center], dim=-1)
        valid = torch.arange(32, device=features.device)[None, :] < counts[:, None]
        decorated *= valid[:, :, None]
        return self.pfn_layers[0](decorated)


class SECOND(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for cin, cout, n in zip([64, 64, 128], [64, 128, 256], [3, 5, 5]):
            modules = [
                nn.Conv2d(cin, cout, 3, 2, 1, bias=False),
                nn.BatchNorm2d(cout, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for _ in range(n):
                modules.extend(
                    [
                        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(cout, eps=0.001, momentum=0.01),
                        nn.ReLU(inplace=True),
                    ]
                )
            blocks.append(nn.Sequential(*modules))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        result = []
        for block in self.blocks:
            x = block(x)
            result.append(x)
        return result


class SECONDFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(cin, 128, stride, stride, bias=False),
                    nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
                for cin, stride in zip([64, 128, 256], [1, 2, 4])
            ]
        )

    def forward(self, features):
        return torch.cat(
            [module(x) for module, x in zip(self.deblocks, features)], dim=1
        )


class Anchor3DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_cls = nn.Conv2d(384, 2, 1)
        self.conv_reg = nn.Conv2d(384, 14, 1)
        self.conv_dir_cls = nn.Conv2d(384, 4, 1)

    def forward(self, x):
        return self.conv_cls(x), self.conv_reg(x), self.conv_dir_cls(x)


class OfficialPointPillars(nn.Module):
    def __init__(self, native=False):
        super().__init__()
        self.native = native
        self.voxel_encoder = PillarFeatureNet()
        self.backbone = SECOND()
        self.neck = SECONDFPN()
        self.bbox_head = Anchor3DHead()

    def load_official(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["state_dict"], strict=True)
        return checkpoint["meta"]

    def freeze_batchnorm(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

    def forward(self, points, max_voxels=None):
        data, count, coord, (nx, ny), ymin = voxelize(
            points, self.native, max_voxels or (16000 if self.training else 40000)
        )
        device = next(self.parameters()).device
        data = torch.from_numpy(data).to(device)
        count = torch.from_numpy(count).to(device)
        coord = torch.from_numpy(coord).to(device)
        encoded = self.voxel_encoder(data, count, coord, ymin)
        canvas = encoded.new_zeros((64, ny * nx))
        index = coord[:, 2] * nx + coord[:, 3]
        canvas[:, index] = encoded.T
        dense = canvas.reshape(1, 64, ny, nx)
        return self.bbox_head(self.neck(self.backbone(dense)))


def anchors(height, width, native=False, device="cuda"):
    xmax, ymin, ymax = (69.12, -39.68, 39.68) if native else (160, -40.32, 40.32)
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (xmax / width)
    y = ymin + (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (
        (ymax - ymin) / height
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    a = torch.zeros((height, width, 2, 7), device=device)
    a[:, :, :, 0] = xx[:, :, None]
    a[:, :, :, 1] = yy[:, :, None]
    a[:, :, :, 2] = -1.78
    a[:, :, :, 3:6] = a.new_tensor([3.9, 1.6, 1.56])
    a[:, :, :, 6] = a.new_tensor([0, 1.57])
    return a.reshape(-1, 7)


def flatten(outputs):
    scores, reg, direction = outputs
    return (
        scores.permute(0, 2, 3, 1).reshape(-1),
        reg.permute(0, 2, 3, 1).reshape(-1, 7),
        direction.permute(0, 2, 3, 1).reshape(-1, 2),
    )


def encode_boxes(a, gt):
    diagonal = (a[:, 3] ** 2 + a[:, 4] ** 2).sqrt()
    # Ground truth is center-Z. Anchors in the official format are bottom-Z.
    target = torch.empty_like(gt)
    target[:, 0] = (gt[:, 0] - a[:, 0]) / diagonal
    target[:, 1] = (gt[:, 1] - a[:, 1]) / diagonal
    target[:, 2] = (gt[:, 2] - (a[:, 2] + a[:, 5] / 2)) / a[:, 5]
    target[:, 3:6] = torch.log(gt[:, 3:6] / a[:, 3:6])
    target[:, 6] = gt[:, 6] - a[:, 6]
    return target


def decode_boxes(a, delta, direction):
    diagonal = (a[:, 3] ** 2 + a[:, 4] ** 2).sqrt()
    box = torch.empty_like(delta)
    box[:, 0] = delta[:, 0] * diagonal + a[:, 0]
    box[:, 1] = delta[:, 1] * diagonal + a[:, 1]
    box[:, 2] = delta[:, 2] * a[:, 5] + a[:, 2] + a[:, 5] / 2
    box[:, 3:6] = delta[:, 3:6].clamp(-4, 4).exp() * a[:, 3:6]
    angle = delta[:, 6] + a[:, 6]
    offset = -math.pi / 2
    period = angle - offset - torch.floor((angle - offset) / math.pi) * math.pi
    box[:, 6] = period + offset + math.pi * direction.argmax(dim=1)
    return box


@torch.no_grad()
def predict(model, points, z_restore=0.0, minimum=0.1, pre=100, nms=0.01, maximum=50):
    outputs = model(points)
    score, delta, direction = flatten(outputs)
    score = score.float().sigmoid()
    chosen = score.topk(min(pre, len(score))).indices
    chosen = chosen[score[chosen] >= minimum]
    a = anchors(outputs[0].shape[2], outputs[0].shape[3], model.native, score.device)[
        chosen
    ]
    boxes = (
        decode_boxes(a, delta[chosen].float(), direction[chosen].float()).cpu().numpy()
    )
    boxes[:, 2] += z_restore
    scores = score[chosen].cpu().numpy()
    result = []
    geometry = []
    for values, value in zip(boxes, scores):
        if not model.native and not (0 <= values[0] < 160 and abs(values[1]) < 40):
            continue
        if not np.isfinite(values).all() or min(values[3:6]) <= 0:
            continue
        candidate = Cuboid(from_seven(values))
        duplicate = False
        for other in geometry:
            if (
                np.all(
                    np.minimum(candidate.high[:2], other.high[:2])
                    > np.maximum(candidate.low[:2], other.low[:2])
                )
                and bev_iou(candidate, other) > nms
            ):
                duplicate = True
                break
        if duplicate:
            continue
        result.append({"score": float(value), "box": values.tolist()})
        geometry.append(candidate)
        if len(result) >= maximum:
            break
    return result


if __name__ == "__main__":
    import json

    torch.set_num_threads(4)
    model = OfficialPointPillars(native=True).cuda().eval()
    meta = model.load_official()
    data = np.fromfile(
        OUT / "official/demo/data/kitti/000008.bin", dtype=np.float32
    ).reshape(-1, 4)
    with torch.no_grad():
        predictions = predict(model, data)
    report = {
        "checkpoint": str(CHECKPOINT),
        "strict_checkpoint_load": True,
        "learned_parameters": sum(p.numel() for p in model.parameters()),
        "native_demo_points": len(data),
        "predictions": predictions,
    }
    (OUT / "native_demo_predictions.json").write_text(json.dumps(report, indent=2))
    print("OFFICIAL_POINTPILLARS_NATIVE_DEMO", json.dumps(report), flush=True)
