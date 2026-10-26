"""Source-only measured-return copy/paste, range thinning and partial visibility.

This is a point-level training augmentation, not physical ray tracing or paired
RGB scene generation. In-cuboid returns may include ground/clutter. Pasted
returns retain the source viewpoint's visible surfaces.
"""

import math

import numpy as np

from lidar_common import rotation


def matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def inside(points, box, margin=0.03):
    b = np.asarray(box)
    radius = np.linalg.norm(b[3:6]) / 2 + margin
    near = np.flatnonzero(np.all(np.abs(points - b[:3]) <= radius, axis=1))
    local = (points[near] - b[:3]) @ matrix(b[6])
    return near[np.all(np.abs(local) <= b[3:6] / 2 + margin, axis=1)]


def overlaps(a, b, clearance=0.4):
    axes = [matrix(a[6])[:2, j] for j in [0, 1]] + [matrix(b[6])[:2, j] for j in [0, 1]]
    delta = np.asarray(a[:2]) - b[:2]
    ra, rb = matrix(a[6])[:2, :2], matrix(b[6])[:2, :2]
    for axis in axes:
        extent_a = np.sum(
            (np.asarray(a[3:5]) / 2 + clearance / 2) * np.abs(ra.T @ axis)
        )
        extent_b = np.sum(
            (np.asarray(b[3:5]) / 2 + clearance / 2) * np.abs(rb.T @ axis)
        )
        if abs(delta @ axis) > extent_a + extent_b:
            return False
    return True


def database(data):
    records = {}
    for sample in data.samples:
        xyz = sample["xyz"]
        for obj in sample["objects"]:
            if not obj["evaluated"]:
                continue
            b = np.array(obj["box"], dtype=float)
            r = rotation(b[3:7])
            radius = np.linalg.norm(b[7:]) / 2 + 0.03
            near = xyz[np.all(np.abs(xyz - b[:3]) <= radius, axis=1)]
            local = (near - b[:3]) @ r
            local = local[np.all(np.abs(local) <= b[7:] / 2 + 0.03, axis=1)]
            if len(local) < 10:
                continue
            size = b[7:].copy()
            if size[0] < size[1]:
                size[[0, 1]] = size[[1, 0]]
                local = local @ matrix(math.pi / 2)
            # Stable subsampling before stochastic training sampling.
            if len(local) > 3000:
                local = local[np.linspace(0, len(local) - 1, 3000).astype(int)]
            records.setdefault(obj["instance_id"], []).append(
                {
                    "sample": sample["id"],
                    "instance": obj["instance_id"],
                    "local": local.astype(np.float32),
                    "size": size,
                    "range": float(np.linalg.norm(b[:2])),
                }
            )
    assert records
    return records


def augment(points, boxes, rng, donors, fallback_ground):
    points = points.copy()
    boxes = [list(b) for b in boxes]
    log = {"pasted": [], "attempts": 0, "z_translation": 0.0}
    if rng.random() < 0.75:
        identities = list(donors)
        for _ in range(int(rng.integers(1, 4))):
            records = donors[identities[int(rng.integers(len(identities)))]]
            donor = records[int(rng.integers(len(records)))]
            size = donor["size"] * float(rng.uniform(0.9, 1.1))
            for attempt in range(20):
                log["attempts"] += 1
                x = float(rng.uniform(8, 145))
                y = float(rng.choice([-1, 1]) * rng.uniform(3, 32))
                yaw = float(
                    rng.choice([0, math.pi / 2, math.pi, -math.pi / 2])
                    + rng.uniform(-0.35, 0.35)
                )
                nearby = points[
                    (np.abs(points[:, 0] - x) < 3) & (np.abs(points[:, 1] - y) < 3)
                ]
                ground = (
                    float(np.quantile(nearby[:, 2], 0.1))
                    if len(nearby) >= 5
                    else fallback_ground
                )
                ground = float(np.clip(ground, -1.5, 3.0))
                box = [
                    x,
                    y,
                    ground + size[2] / 2 + float(rng.uniform(-0.1, 0.1)),
                    *size,
                    yaw,
                ]
                corners = (
                    np.array(
                        [
                            [a, b, 0]
                            for a in [-size[0] / 2, size[0] / 2]
                            for b in [-size[1] / 2, size[1] / 2]
                        ]
                    )
                    @ matrix(yaw).T
                    + box[:3]
                )
                if not (
                    (corners[:, 0] >= 0).all()
                    and (corners[:, 0] < 160).all()
                    and (np.abs(corners[:, 1]) < 40).all()
                ):
                    continue
                if any(overlaps(box, old) for old in boxes):
                    continue
                local = donor["local"].copy() * (size / donor["size"])
                structured = bool(rng.random() < 0.5)
                if structured:
                    axis = int(rng.integers(2))
                    threshold = float(rng.uniform(-0.2, 0.2)) * size[axis]
                    partial = local[(local[:, axis] * rng.choice([-1, 1])) >= threshold]
                    if len(partial) >= 5:
                        local = partial
                    else:
                        structured = False
                fraction = min(1.0, (donor["range"] / max(1, math.hypot(x, y))) ** 2)
                count = min(len(local), max(5, int(len(local) * fraction)))
                local = local[rng.choice(len(local), count, replace=False)]
                pasted = local @ matrix(yaw).T + np.array(box[:3])
                mask = np.ones(len(points), dtype=bool)
                mask[inside(points, box, 0.05)] = False
                points = np.concatenate([points[mask], pasted.astype(np.float32)])
                boxes.append(box)
                log["pasted"].append(
                    {
                        "source_sample": donor["sample"],
                        "source_instance": donor["instance"],
                        "returns": count,
                        "partial_visibility": structured,
                        "box_before_global_z": box.copy(),
                    }
                )
                break
    # Expose different sensor/road heights without using validation/test priors.
    delta_z = float(rng.uniform(-1, 1))
    log["z_translation"] = delta_z
    points[:, 2] += delta_z
    for box in boxes:
        box[2] += delta_z
    points = points[(points[:, 2] >= -3) & (points[:, 2] <= 8)]
    if len(points) > 160000:
        points = points[rng.choice(len(points), 160000, replace=False)]
    boxes = [box for box in boxes if len(inside(points, box)) >= 5]
    return np.ascontiguousarray(points, dtype=np.float32), boxes, log


def prepare(data, index, base_rng, source, priors, donors=None, aug_rng=None):
    tensor, boxes = data.prepare(index, base_rng, True)
    points = tensor.cpu().numpy()
    del tensor
    log = {"pasted": [], "z_translation": 0.0}
    if donors is not None:
        points, boxes, log = augment(
            points,
            boxes,
            aug_rng,
            donors,
            priors["domains"][source]["median_training_trackbed_z"],
        )
    # Common rule: only supervise boxes still supported after point operations.
    boxes = [box for box in boxes if len(inside(points, box)) >= 5]
    points = points.copy()
    points[:, 2] += priors["domains"][source]["z_shift"]
    target = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    if len(target):
        target[:, 2] += priors["domains"][source]["z_shift"]
    base_rng.shuffle(points)
    return points, target, log
