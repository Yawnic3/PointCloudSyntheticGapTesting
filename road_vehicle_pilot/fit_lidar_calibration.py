"""Fit empirical sampling and placement priors using real TRAINING data only.

Return directions are not emitted-beam calibration. Sensor origins and range
noise remain assumptions. Real XYZ is never copied into synthetic scans: the
direction templates will be raycast against synthetic Blender geometry.
"""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from lidar_common import BASE, ROOT, rotation

OUT = BASE / "lidar_calibrated"
EDGES = np.array([0, 10, 20, 40, 60, 80, 120, 180], dtype=np.float32)
ORIGINS = np.array(
    [
        [0, 0, 2.6],
        [0.1, -0.5, 2.8],
        [0.1, 0, 2.8],
        [0.1, 0.5, 2.8],
        [0, -0.7, 1.1],
        [0, 0.7, 1.1],
    ],
    dtype=np.float32,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roi(xyz):
    return (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= 0)
        & (xyz[:, 0] < 160)
        & (np.abs(xyz[:, 1]) < 40)
        & (xyz[:, 2] >= -3)
        & (xyz[:, 2] <= 8)
    )


def bed_mode(xyz):
    near = xyz[(xyz[:, 0] > 5) & (xyz[:, 0] < 25) & (np.abs(xyz[:, 1]) < 1.5), 2]
    bins = np.arange(-1, 0.51, 0.02)
    hist, edges = np.histogram(near, bins=bins)
    assert hist.max() > 0
    i = int(hist.argmax())
    return float((edges[i] + edges[i + 1]) / 2)


def raw_pcd(path):
    with path.open("rb") as handle:
        header = {}
        while True:
            line = handle.readline().decode("ascii").strip()
            if line.startswith("#"):
                continue
            key, *values = line.split()
            header[key] = values
            if key == "DATA":
                break
        assert header["DATA"] == ["ascii"]
        fields = header["FIELDS"]
        data = np.fromstring(handle.read().decode("ascii"), sep=" ").reshape(
            -1, len(fields)
        )
        assert len(data) == int(header["POINTS"][0])
    return data[:, :3].astype(np.float32), data[:, fields.index("sensor_index")].astype(
        np.int8
    )


def main():
    OUT.mkdir(exist_ok=True)
    destination = OUT / "calibration.json"
    assert not destination.exists(), "Calibration is frozen; use existing files"
    training = (ROOT / "real_train.txt").read_text().splitlines()
    assert len(training) == 320
    metadata = []
    total_points, ground, poses = [], [], []
    grouped = defaultdict(list)
    for image in training:
        sample = Path(image).stem
        meta_path = ROOT / "real" / "metadata" / (sample + ".json")
        meta = json.loads(meta_path.read_text())
        assert meta["split"] == "train"
        with np.load(ROOT / "real" / "points" / (sample + ".npz")) as data:
            xyz = data["xyz"]
            total_points.append(int(roi(xyz).sum()))
            ground.append(bed_mode(xyz))
        metadata.append(meta)
        grouped[meta["sequence"]].append(meta)
        for obj in meta["objects"]:
            if not obj["evaluated"]:
                continue
            box = obj["box"]
            matrix = rotation(box[3:7])
            poses.append(
                {
                    "sample_id": sample,
                    "instance_id": obj["instance_id"],
                    "x": box[0],
                    "y": box[1],
                    "bottom_z": box[2] - box[9] / 2,
                    "yaw": math.atan2(matrix[1, 0], matrix[0, 0]),
                    "box_dimensions": box[7:10],
                    "center_z": box[2],
                }
            )
    # Every source is a training member. Spread the angular fit over sequences.
    selected = []
    queues = {}
    for sequence, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["frame_id"]))
        indices = np.linspace(0, len(rows) - 1, min(8, len(rows))).round().astype(int)
        queues[sequence] = [rows[i] for i in indices]
    while len(selected) < 40 and any(queues.values()):
        for sequence in sorted(queues):
            if queues[sequence] and len(selected) < 40:
                selected.append(queues[sequence].pop(0))
    directions = defaultdict(list)
    profiles = []
    sources = []
    for meta in selected:
        path = BASE / "real_raw" / meta["sequence"] / meta["pcd_member"]
        assert sha(path) == meta["pcd_sha256"]
        xyz, sensor = raw_pcd(path)
        mask = roi(xyz)
        xyz, sensor = xyz[mask], sensor[mask]
        assert set(np.unique(sensor)).issubset(set(range(6)))
        radius = np.linalg.norm(xyz[:, :2], axis=1)
        range_bin = np.clip(
            np.searchsorted(EDGES, radius, side="right") - 1, 0, len(EDGES) - 2
        )
        profile = np.zeros((6, len(EDGES) - 1), dtype=int)
        for s in range(6):
            for b in range(len(EDGES) - 1):
                chosen = (sensor == s) & (range_bin == b)
                p = xyz[chosen] - ORIGINS[s]
                norm = np.linalg.norm(p, axis=1)
                p = p[norm > 0.5] / norm[norm > 0.5, None]
                directions[(s, b)].append(p)
                profile[s, b] = len(p)
        profiles.append(profile.tolist())
        sources.append(
            {
                "sample_id": meta["sample_id"],
                "split": "train",
                "pcd_sha256": meta["pcd_sha256"],
            }
        )
        print("FIT_TRAIN_TEMPLATE", meta["sample_id"], len(xyz), flush=True)
    rng = np.random.default_rng(2301)
    arrays = {}
    for key, chunks in directions.items():
        pool = np.concatenate(chunks).astype(np.float32)
        if len(pool) > 250000:
            pool = pool[rng.choice(len(pool), 250000, replace=False)]
        arrays[f"s{key[0]}_b{key[1]}"] = pool
    np.savez_compressed(OUT / "direction_templates.npz", **arrays)
    synthetic_ground = []
    for route in range(8):
        with np.load(
            ROOT / "synthetic" / "points" / f"route_{route:02}_0000.npz"
        ) as data:
            synthetic_ground.append(bed_mode(data["xyz"]))
    report = {
        "source_split": "real_train_only",
        "real_train_list_sha256": sha(ROOT / "real_train.txt"),
        "source_script_sha256": sha(Path(__file__)),
        "source_frames": sources,
        "training_point_totals": total_points,
        "target_mean_roi_points": float(np.mean(total_points)),
        "range_edges_m": EDGES.tolist(),
        "assumed_sensor_origins_m": ORIGINS.tolist(),
        "frame_histograms_sensor_range": profiles,
        "real_trackbed_mode_median_m": float(np.median(ground)),
        "synthetic_trackbed_mode_median_m": float(np.median(synthetic_ground)),
        "frame_origin_world_z": float(np.median(synthetic_ground) - np.median(ground)),
        "placement_records": poses,
        "direction_templates_sha256": sha(OUT / "direction_templates.npz"),
        "limitations": "Observed return directions with assumed sensor origins, not calibrated emitted beams. No real XYZ is copied to synthetic clouds. Noise remains 0.015 m XYZ; material intensity and sensor timing are not fitted. Placement priors use real TRAIN annotations, so this is real-training-informed synthetic generation, not zero-real-label simulation.",
    }
    destination.write_text(json.dumps(report, indent=2))
    print(
        "CALIBRATION_FIT_COMPLETE",
        len(selected),
        len(poses),
        report["target_mean_roi_points"],
        report["frame_origin_world_z"],
        flush=True,
    )


if __name__ == "__main__":
    main()
