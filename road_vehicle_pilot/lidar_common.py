"""Shared, label-free XYZ-to-BEV preprocessing and cuboid geometry.

Coordinates: X forward, Y left, Z up, metres. BEV rows run far to near;
columns run left to right. The detector sees only XYZ-derived channels.
"""

from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE / "lidar_vehicle_v2"
XMAX = 160.0
YMAX = 40.0
WIDTH = 480
HEIGHT = 960
ZMIN = -3.0
ZMAX = 8.0
MIN_POINTS = 5


def rotation(quaternion):
    x, y, z, w = np.asarray(quaternion) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def corners(box):
    signs = np.array(
        [[x, y, z] for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]]
    )
    return (signs * box[7:10]) @ rotation(box[3:7]).T + box[:3]


def bounds(box):
    points = corners(box)
    return np.r_[points[:, :2].min(axis=0), points[:, :2].max(axis=0)]


def eligible(box):
    b = bounds(box)
    return bool(b[0] >= 0 and b[1] >= -YMAX and b[2] < XMAX and b[3] < YMAX)


def point_count(xyz, box):
    local = (xyz - box[:3]) @ rotation(box[3:7])
    return int(np.all(np.abs(local) <= np.array(box[7:10]) / 2 + 0.03, axis=1).sum())


def bev(xyz):
    mask = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= 0)
        & (xyz[:, 0] < XMAX)
        & (np.abs(xyz[:, 1]) < YMAX)
        & (xyz[:, 2] >= ZMIN)
        & (xyz[:, 2] <= ZMAX)
    )
    p = xyz[mask]
    row = np.minimum(((XMAX - p[:, 0]) / XMAX * HEIGHT).astype(int), HEIGHT - 1)
    col = np.minimum(((YMAX - p[:, 1]) / (2 * YMAX) * WIDTH).astype(int), WIDTH - 1)
    index = row * WIDTH + col
    size = HEIGHT * WIDTH
    count = np.bincount(index, minlength=size)
    highest = np.full(size, ZMIN, dtype=np.float32)
    lowest = np.full(size, ZMAX, dtype=np.float32)
    np.maximum.at(highest, index, p[:, 2])
    np.minimum.at(lowest, index, p[:, 2])
    occupied = count > 0
    # Fixed physical scales for both domains; no per-frame auto-contrast.
    channels = np.stack(
        [
            np.where(occupied, (highest - ZMIN) / (ZMAX - ZMIN), 0),
            np.clip(np.log1p(count) / np.log(65), 0, 1),
            np.where(occupied, np.clip((highest - lowest) / 4, 0, 1), 0),
        ],
        axis=-1,
    )
    return (channels.reshape(HEIGHT, WIDTH, 3) * 255).round().astype(np.uint8)


def label(box):
    x0, y0, x1, y1 = bounds(box)
    return [
        0,
        (YMAX - (y0 + y1) / 2) / (2 * YMAX),
        (XMAX - (x0 + x1) / 2) / XMAX,
        (y1 - y0) / (2 * YMAX),
        (x1 - x0) / XMAX,
    ]


def export_sample(
    domain,
    sample_id,
    xyz,
    objects,
    metadata,
    instances=None,
    sensor_indices=None,
    write_points=True,
):
    from PIL import Image

    folder = ROOT / domain
    for name in ["points", "images", "labels", "metadata"]:
        (folder / name).mkdir(parents=True, exist_ok=True)
    arrays = {"xyz": np.asarray(xyz, dtype=np.float32)}
    if instances is not None:
        arrays["instance_id_ground_truth_only"] = instances
    if sensor_indices is not None:
        arrays["sensor_index_ground_truth_only"] = sensor_indices
    if write_points:
        np.savez_compressed(folder / "points" / (sample_id + ".npz"), **arrays)
    Image.fromarray(bev(xyz)).save(folder / "images" / (sample_id + ".png"))
    kept = []
    for obj in objects:
        obj["points_in_cuboid"] = point_count(xyz, obj["box"])
        obj["inside_roi"] = eligible(obj["box"])
        obj["evaluated"] = obj["inside_roi"] and obj["points_in_cuboid"] >= MIN_POINTS
        if obj["evaluated"]:
            kept.append(label(obj["box"]))
    # Low-support and boundary objects remain in metadata for explicit exclusions.
    text = "".join("0 " + " ".join(f"{v:.8f}" for v in row[1:]) + "\n" for row in kept)
    (folder / "labels" / (sample_id + ".txt")).write_text(text)
    import json

    metadata.update(
        sample_id=sample_id,
        domain=domain,
        objects=objects,
        point_count=len(xyz),
        label_count=len(kept),
    )
    (folder / "metadata" / (sample_id + ".json")).write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata
