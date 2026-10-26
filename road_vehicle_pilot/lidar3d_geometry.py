"""Exact convex cuboid overlap, including source roll/pitch; diagnostic 3D AP."""

import math

import numpy as np

from lidar_common import corners, rotation

SIGNS = np.array(
    [[x, y, z] for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]]
)
EDGES = np.array(
    [(a, b) for a in range(8) for b in range(a + 1, 8) if (a ^ b).bit_count() == 1]
)


def from_seven(box):
    x, y, z, l, w, h, yaw = map(float, box)
    return [x, y, z, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2), l, w, h]


class Cuboid:
    def __init__(self, box):
        self.box = np.asarray(box, dtype=np.float64)
        self.center = self.box[:3]
        self.rotation = rotation(self.box[3:7])
        self.size = self.box[7:]
        assert np.isfinite(self.box).all() and (self.size > 0).all()
        self.corners = corners(self.box)
        self.low = self.corners.min(axis=0)
        self.high = self.corners.max(axis=0)
        self.volume = float(np.prod(self.size))

    def inside(self, points):
        return np.all(
            np.abs((points - self.center) @ self.rotation) <= self.size / 2 + 1e-8,
            axis=1,
        )


def edge_intersections(a, b):
    start = a.corners[EDGES[:, 0]]
    delta = a.corners[EDGES[:, 1]] - start
    local = (start - b.center) @ b.rotation
    direction = delta @ b.rotation
    points = []
    for axis in range(3):
        for sign in [-1, 1]:
            denom = direction[:, axis]
            valid = np.abs(denom) > 1e-12
            t = np.zeros(len(start))
            t[valid] = (sign * b.size[axis] / 2 - local[valid, axis]) / denom[valid]
            valid &= (t >= -1e-9) & (t <= 1 + 1e-9)
            candidate = start[valid] + t[valid, None] * delta[valid]
            if len(candidate):
                points.extend(candidate[b.inside(candidate)])
    return np.asarray(points).reshape(-1, 3)


def iou3d(a, b):
    if np.any(np.minimum(a.high, b.high) - np.maximum(a.low, b.low) <= 0):
        return 0.0
    from scipy.spatial import ConvexHull

    vertices = np.concatenate(
        [
            a.corners[b.inside(a.corners)],
            b.corners[a.inside(b.corners)],
            edge_intersections(a, b),
            edge_intersections(b, a),
        ]
    )
    vertices = np.unique(vertices.round(8), axis=0)
    if len(vertices) < 4 or np.linalg.matrix_rank(vertices - vertices[0], tol=1e-8) < 3:
        return 0.0
    intersection = min(float(ConvexHull(vertices).volume), a.volume, b.volume)
    return max(0.0, min(1.0, intersection / (a.volume + b.volume - intersection)))


def polygon_area(p):
    if len(p) < 3:
        return 0.0
    return (
        abs(
            float(
                np.dot(p[:, 0], np.roll(p[:, 1], -1))
                - np.dot(p[:, 1], np.roll(p[:, 0], -1))
            )
        )
        / 2
    )


def bev_iou(a, b):
    from scipy.spatial import ConvexHull

    pa = a.corners[ConvexHull(a.corners[:, :2]).vertices, :2]
    pb = b.corners[ConvexHull(b.corners[:, :2]).vertices, :2]
    polygon = pa.copy()

    def cross(x, y):
        return x[0] * y[1] - x[1] * y[0]

    for start, end in zip(pb, np.roll(pb, -1, axis=0)):
        if not len(polygon):
            return 0.0
        edge = end - start
        result = []
        previous = polygon[-1]
        prev_inside = cross(edge, previous - start) >= -1e-9
        for point in polygon:
            inside = cross(edge, point - start) >= -1e-9
            if inside != prev_inside:
                delta = point - previous
                denom = cross(edge, delta)
                if abs(denom) > 1e-12:
                    t = -cross(edge, previous - start) / denom
                    result.append(previous + t * delta)
            if inside:
                result.append(point)
            previous = point
            prev_inside = inside
        polygon = np.asarray(result).reshape(-1, 2)
    area = polygon_area(polygon)
    return area / max(1e-12, polygon_area(pa) + polygon_area(pb) - area)


def suppress(predictions, threshold=0.15, limit=100):
    kept = []
    boxes = []
    for pred in sorted(predictions, key=lambda p: -p["score"]):
        box = Cuboid(from_seven(pred["box"]))
        if any(iou3d(box, other) > threshold for other in boxes):
            continue
        kept.append(pred)
        boxes.append(box)
        if len(kept) >= limit:
            break
    return kept


def evaluate(predictions, truth, threshold=0.5, mode="3d", confidence=0.25):
    gt = {name: [Cuboid(b) for b in boxes] for name, boxes in truth.items()}
    used = {name: set() for name in truth}
    total = sum(map(len, gt.values()))
    ordered = sorted(
        [
            (p["score"], name, p["box"])
            for name, items in predictions.items()
            for p in items
        ],
        key=lambda p: -p[0],
    )
    hits = []
    scores = []
    tp = fp = 0
    for score, name, values in ordered:
        box = Cuboid(from_seven(values))
        candidates = []
        for j, other in enumerate(gt[name]):
            if j not in used[name]:
                candidates.append(((iou3d if mode == "3d" else bev_iou)(box, other), j))
        hit = bool(candidates and max(candidates)[0] >= threshold)
        if hit:
            used[name].add(max(candidates)[1])
        hits.append(int(hit))
        scores.append(score)
        if score >= confidence:
            tp += int(hit)
            fp += int(not hit)
    if not hits:
        ap = 0.0
    else:
        cumulative = np.cumsum(hits)
        recall = cumulative / max(1, total)
        precision = cumulative / np.arange(1, len(hits) + 1)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        ap = float(
            np.mean(
                [
                    (
                        envelope[np.searchsorted(recall, r, side="left")]
                        if np.searchsorted(recall, r, side="left") < len(envelope)
                        else 0
                    )
                    for r in np.linspace(0, 1, 101)
                ]
            )
        )
    return {
        "AP101": ap,
        "IoU": threshold,
        "mode": mode,
        "ground_truth_boxes": total,
        "confidence": confidence,
        "tp": tp,
        "fp": fp,
        "fn": total - tp,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, total),
    }


def verify_geometry():
    unit = from_seven([0, 0, 0, 2, 2, 2, 0])
    a = Cuboid(unit)
    assert abs(iou3d(a, a) - 1) < 1e-9
    assert iou3d(a, Cuboid(from_seven([3, 0, 0, 2, 2, 2, 0]))) == 0
    assert abs(iou3d(a, Cuboid(from_seven([1, 0, 0, 2, 2, 2, 0]))) - 1 / 3) < 1e-8
    assert (
        abs(
            iou3d(a, Cuboid(from_seven([0, 0, 0, 2, 2, 2, math.pi / 4])))
            - 1 / math.sqrt(2)
        )
        < 1e-7
    )
    tilted = unit.copy()
    tilted[3:7] = [math.sin(0.3), 0, 0, math.cos(0.3)]
    b = Cuboid(tilted)
    assert abs(iou3d(a, b) - iou3d(b, a)) < 1e-9 and 0 < iou3d(a, b) < 1
    assert abs(bev_iou(a, a) - 1) < 1e-8
    perfect = evaluate(
        {"frame": [{"score": 0.9, "box": [0, 0, 0, 2, 2, 2, 0]}]}, {"frame": [unit]}
    )
    assert perfect["AP101"] == 1 and perfect["tp"] == 1
    return {
        "analytic_cube_cases": True,
        "rotated_cube_45_degree_IoU": 1 / math.sqrt(2),
        "tilted_box_symmetry": True,
        "perfect_detection_AP": perfect["AP101"],
    }
