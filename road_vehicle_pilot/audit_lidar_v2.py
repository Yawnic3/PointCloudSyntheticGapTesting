"""Independently validate point clouds, paired scenes, split isolation and BEV labels."""

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

from lidar_common import (
    BASE,
    ROOT,
    HEIGHT,
    WIDTH,
    MIN_POINTS,
    bounds,
    bev,
    eligible,
    label,
    point_count,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((ROOT / "real_selection.json").read_text())
    all_metadata = {}
    stats = defaultdict(
        lambda: {
            "images": 0,
            "positive_images": 0,
            "boxes": 0,
            "excluded_boxes": 0,
            "points": 0,
            "feature_roi_points": 0,
        }
    )
    seen_hashes = {}
    instance_splits = defaultdict(set)
    lists = defaultdict(list)
    family_support = Counter()
    real_instances = defaultdict(set)
    for domain, expected in [("synthetic", 480), ("real", 400)]:
        paths = sorted((ROOT / domain / "metadata").glob("*.json"))
        assert len(paths) == expected, (domain, len(paths), expected)
        for path in paths:
            meta = json.loads(path.read_text())
            sample = path.stem
            split = meta["split"]
            key = domain + "_" + split
            image_path = ROOT / domain / "images" / (sample + ".png")
            labels_path = ROOT / domain / "labels" / (sample + ".txt")
            with np.load(ROOT / domain / "points" / (sample + ".npz")) as data:
                xyz = data["xyz"]
                assert xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all()
                assert len(xyz) == meta["point_count"] and len(xyz) > 1000
                roi = (
                    (xyz[:, 0] >= 0)
                    & (xyz[:, 0] < 160)
                    & (np.abs(xyz[:, 1]) < 40)
                    & (xyz[:, 2] >= -3)
                    & (xyz[:, 2] <= 8)
                )
                stats[key]["feature_roi_points"] += int(roi.sum())
                assert np.array_equal(
                    np.asarray(Image.open(image_path).convert("RGB")), bev(xyz)
                ), sample
                expected_labels = []
                for obj in meta["objects"]:
                    box = obj["box"]
                    assert len(box) == 10 and min(box[7:]) > 0
                    assert abs(np.linalg.norm(box[3:7]) - 1) < 0.01
                    count = point_count(xyz, box)
                    assert count == obj["points_in_cuboid"]
                    supported = eligible(box) and count >= MIN_POINTS
                    assert supported == obj["evaluated"]
                    if supported:
                        expected_labels.append(label(box))
                        if domain == "synthetic" and split == "train":
                            family_support[obj["family"]] += 1
                        if domain == "real":
                            real_instances[split].add(obj["instance_id"])
                    else:
                        stats[key]["excluded_boxes"] += 1
                    if domain == "real":
                        instance_splits[obj["instance_id"]].add(split)
                actual = (
                    np.loadtxt(labels_path, ndmin=2)
                    if labels_path.stat().st_size
                    else np.empty((0, 5))
                )
                expected_array = np.array(expected_labels).reshape(-1, 5)
                assert actual.shape == expected_array.shape and np.allclose(
                    actual, expected_array, atol=1e-7
                )
                if len(actual):
                    assert (
                        (actual[:, 0] == 0).all()
                        and (actual[:, 1:] > 0).all()
                        and (actual[:, 1:] < 1).all()
                    )
                if domain == "synthetic":
                    rgb_path = BASE / "synthetic_v2" / "metadata" / (sample + ".json")
                    assert (
                        meta["rgb_scene_match_verified"]
                        and digest(rgb_path) == meta["rgb_metadata_sha256"]
                    )
                    assert np.allclose(
                        np.array(meta["camera_matrix_world"])
                        @ np.array(meta["lidar_to_blender_camera_matrix"]),
                        meta["lidar_to_world_matrix"],
                        atol=1e-6,
                    )
                    assert split == (
                        "train"
                        if meta["route_id"] < 8
                        else "val" if meta["route_id"] < 10 else "test"
                    )
                else:
                    assert meta["group"] in frozen["groups"][split]
                # Excludes repeat identical scans crossing splits, not just reused filenames.
                cloud_hash = hashlib.sha256(xyz.tobytes()).hexdigest()
                assert cloud_hash not in seen_hashes or seen_hashes[cloud_hash] == key
                seen_hashes[cloud_hash] = key
            stats[key]["images"] += 1
            stats[key]["positive_images"] += bool(len(actual))
            stats[key]["boxes"] += len(actual)
            stats[key]["points"] += len(xyz)
            lists[key].append(str(image_path.resolve()))
            all_metadata[domain + "/" + sample] = meta
    assert all(
        len(splits) == 1 for splits in instance_splits.values()
    ), "Real annotation UUID crossed splits"
    for name, paths in lists.items():
        (ROOT / (name + ".txt")).write_text("\n".join(paths) + "\n")
    assert stats["synthetic_train"]["images"] == stats["real_train"]["images"] == 320
    for domain in ["synthetic", "real"]:
        text = (
            f"path: {ROOT.as_posix()}\n"
            f"train: {domain}_train.txt\nval: real_val.txt\ntest: real_test.txt\n"
            "names:\n  0: road_vehicle\n"
        )
        (ROOT / (domain + ".yaml")).write_text(text)
    hashes = {name: digest(ROOT / (name + ".txt")) for name in lists}
    sources = {
        name: digest(BASE / name)
        for name in [
            "generate_lidar_v2.py",
            "generate_vehicle_v2.py",
            "lidar_common.py",
            "prepare_lidar_v2.py",
            "audit_lidar_v2.py",
        ]
    }
    report = {
        "complete": True,
        "synthetic_training_family_support": dict(family_support),
        "real_unique_annotation_ids_by_split": {
            split: len(ids) for split, ids in real_instances.items()
        },
        "stats": dict(stats),
        "list_sha256": hashes,
        "selection_sha256": digest(ROOT / "real_selection.json"),
        "source_sha256": sources,
        "policy": {
            "class": "road_vehicle",
            "roi_m": {"x": [0, 160], "y": [-40, 40], "z_features": [-3, 8]},
            "grid": [HEIGHT, WIDTH],
            "channels": ["max_height", "log_point_density", "vertical_spread"],
            "minimum_cuboid_points": MIN_POINTS,
            "ground_truth": "Full cuboid XY AABB, wholly inside ROI and >=5 points; low-support/boundary objects excluded and retained in metadata. Predictions on excluded objects are not ignored by the baseline evaluator.",
            "task": "LiDAR-only axis-aligned BEV detection, not oriented BEV or full 3D AP",
            "sensor_limit": "Synthetic six-fan first-return idealization, not calibrated to six real OSDaR23 sensors; no real intensity input.",
        },
        "test_limit": frozen["test_limit"],
    }
    (ROOT / "audit.json").write_text(json.dumps(report, indent=2))
    # A labelled overview for visual checking of both domains.
    sheet = Image.new("RGB", (960, 1060), "#182027")
    draw = ImageDraw.Draw(sheet)
    for column, domain in enumerate(["synthetic", "real"]):
        rows = [
            m
            for k, m in all_metadata.items()
            if k.startswith(domain + "/")
            and m["split"] == "train"
            and m["label_count"] > 0
        ][:2]
        draw.text(
            (column * 480 + 10, 10),
            domain + " LiDAR BEV - green = vehicle GT",
            fill="white",
        )
        for row, meta in enumerate(rows):
            image = Image.open(
                ROOT / domain / "images" / (meta["sample_id"] + ".png")
            ).convert("RGB")
            painter = ImageDraw.Draw(image)
            for obj in meta["objects"]:
                if not obj["evaluated"]:
                    continue
                _, cx, cy, w, h = label(obj["box"])
                painter.rectangle(
                    (
                        (cx - w / 2) * WIDTH,
                        (cy - h / 2) * HEIGHT,
                        (cx + w / 2) * WIDTH,
                        (cy + h / 2) * HEIGHT,
                    ),
                    outline="lime",
                    width=2,
                )
            image.thumbnail((240, 480))
            sheet.paste(image, (column * 480 + 120, 40 + row * 500))
            draw.text(
                (column * 480 + 10, 25 + row * 500), meta["sample_id"], fill="white"
            )
    sheet.save(ROOT / "bev_review.jpg")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
