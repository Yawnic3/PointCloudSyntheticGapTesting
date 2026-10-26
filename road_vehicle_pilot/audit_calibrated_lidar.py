"""Export and validate both calibrated TRAIN splits without changing legacy data."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lidar_common import (
    BASE,
    ROOT,
    HEIGHT,
    WIDTH,
    bev,
    eligible,
    label,
    point_count,
    rotation,
)

OUT = BASE / "lidar_calibrated"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(xyz):
    mask = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= 0)
        & (xyz[:, 0] < 160)
        & (np.abs(xyz[:, 1]) < 40)
        & (xyz[:, 2] >= -3)
        & (xyz[:, 2] <= 8)
    )
    selected = xyz[mask]
    edges = np.array([0, 10, 20, 40, 60, 80, 120, 180])
    return (
        len(selected),
        np.histogram(np.linalg.norm(selected[:, :2], axis=1), edges)[0],
    )


def summarize_reference(domain, list_name):
    totals, histogram, boxes = [], np.zeros(7, dtype=np.int64), []
    for image in (ROOT / list_name).read_text().splitlines():
        name = Path(image).stem
        with np.load(ROOT / domain / "points" / (name + ".npz")) as data:
            count, bins = distribution(data["xyz"])
        totals.append(count)
        histogram += bins
        meta = json.loads((ROOT / domain / "metadata" / (name + ".json")).read_text())
        boxes.extend(o["box"] for o in meta["objects"] if o["evaluated"])
    return summary(totals, histogram, boxes)


def summary(totals, histogram, boxes):
    a = np.array(boxes).reshape(-1, 10)
    return {
        "frames": len(totals),
        "mean_roi_points": float(np.mean(totals)),
        "median_roi_points": float(np.median(totals)),
        "range_histogram": histogram.tolist(),
        "range_fractions": (histogram / max(1, histogram.sum())).tolist(),
        "supported_boxes": len(boxes),
        "median_supported_center_xyz": (
            np.median(a[:, :3], axis=0).tolist() if len(a) else None
        ),
        "median_supported_bottom_z": (
            float(np.median(a[:, 2] - a[:, 9] / 2)) if len(a) else None
        ),
    }


def main():
    fit = json.loads((OUT / "calibration.json").read_text())
    training_ids = {
        Path(p).stem for p in (ROOT / "real_train.txt").read_text().splitlines()
    }
    assert len(training_ids) == 320
    assert fit["source_split"] == "real_train_only"
    assert sha(ROOT / "real_train.txt") == fit["real_train_list_sha256"]
    assert sha(OUT / "direction_templates.npz") == fit["direction_templates_sha256"]
    assert all(
        row["split"] == "train" and row["sample_id"] in training_ids
        for row in fit["source_frames"]
    )
    assert all(row["sample_id"] in training_ids for row in fit["placement_records"])
    report = {
        "complete": False,
        "real_test_scored": False,
        "variants": {},
        "reference": {
            "real_train": summarize_reference("real", "real_train.txt"),
            "original_synthetic_train": summarize_reference(
                "synthetic", "synthetic_train.txt"
            ),
        },
        "input_sha256": {},
        "calibration_sha256": sha(OUT / "calibration.json"),
        "generator_sha256": sha(BASE / "generate_calibrated_lidar.py"),
        "audit_script_sha256": sha(Path(__file__)),
        "calibration_sources_train_only_verified": True,
    }
    required = {
        f"route_{route:02}_{frame:04}" for route in range(8) for frame in range(40)
    }
    metadata_by_variant = {}
    for variant in ["sensor_only", "sensor_pose"]:
        folder = OUT / variant
        for name in ["images", "labels"]:
            (folder / name).mkdir(parents=True, exist_ok=True)
        paths = sorted((folder / "metadata").glob("*.json"))
        assert {p.stem for p in paths} == required
        totals, histogram, boxes, positives = [], np.zeros(7, dtype=np.int64), [], 0
        families, fallback, target_returns = Counter(), 0, 0
        by_sample, image_paths = {}, []
        for index, path in enumerate(paths):
            meta = json.loads(path.read_text())
            assert meta["split"] == "train" and meta["route_id"] < 8
            assert (
                meta["variant"] == variant
                and meta["generator_sha256"] == report["generator_sha256"]
            )
            assert meta["calibration_sha256"] == report["calibration_sha256"]
            with np.load(folder / "points" / (path.stem + ".npz")) as data:
                xyz = data["xyz"]
                instances = data["instance_id_ground_truth_only"]
                sensors = data["sensor_index_ground_truth_only"]
                assert len(xyz) == len(instances) == len(sensors) == meta["point_count"]
                assert np.isfinite(xyz).all() and len(xyz) > 1000
                assert set(np.unique(sensors)).issubset(set(range(6)))
                assert set(np.unique(instances)).issubset(
                    {0} | {o["mask_id"] for o in meta["objects"]}
                )
                count, bins = distribution(xyz)
                assert count == len(xyz)
                totals.append(count)
                histogram += bins
                labels = []
                legacy = json.loads(
                    (
                        ROOT / "synthetic" / "metadata" / (path.stem + ".json")
                    ).read_text()
                )
                assert len(legacy["objects"]) == len(meta["objects"])
                for obj, old in zip(meta["objects"], legacy["objects"]):
                    assert (
                        obj["instance_id"] == old["instance_id"]
                        and obj["family"] == old["family"]
                    )
                    # Blender's float32 world/inverse transforms introduce sub-mm
                    # rounding when repositioning meshes at large route stations.
                    assert np.allclose(
                        obj["box"][7:], old["box"][7:], atol=0.0005, rtol=0
                    )
                    if variant == "sensor_only":
                        expected = np.array(old["box"])
                        expected[2] -= fit["frame_origin_world_z"]
                        assert np.allclose(obj["box"], expected, atol=1e-4)
                    box = obj["box"]
                    selected = xyz[instances == obj["mask_id"]]
                    local = (selected - box[:3]) @ rotation(box[3:7])
                    assert np.all(np.abs(local) <= np.array(box[7:]) / 2 + 0.1), (
                        variant,
                        path.stem,
                        obj["mask_id"],
                    )
                    target_returns += len(selected)
                    support = point_count(xyz, box)
                    obj.update(points_in_cuboid=support, inside_roi=eligible(box))
                    obj["evaluated"] = obj["inside_roi"] and support >= 5
                    if obj["evaluated"]:
                        labels.append(label(box))
                        boxes.append(box)
                        families[obj["family"]] += 1
                image_path = folder / "images" / (path.stem + ".png")
                Image.fromarray(bev(xyz)).save(image_path)
                assert np.array_equal(
                    np.asarray(Image.open(image_path).convert("RGB")), bev(xyz)
                )
            label_path = folder / "labels" / (path.stem + ".txt")
            label_path.write_text(
                "".join(
                    "0 " + " ".join(f"{v:.8f}" for v in row[1:]) + "\n"
                    for row in labels
                )
            )
            positives += bool(labels)
            if variant == "sensor_pose":
                assert not meta["rgb_scene_match_verified"]
                assert len(meta["placement"]) == len(meta["objects"])
                for placed in meta["placement"]:
                    fallback += placed["fallback_original_pose"]
                    if not placed["fallback_original_pose"]:
                        assert placed["source_training_sample"] in training_ids
            else:
                assert meta["rgb_scene_match_verified"]
            assert np.allclose(
                np.array(meta["camera_matrix_world"])
                @ np.array(meta["lidar_to_blender_camera_matrix"]),
                meta["lidar_to_world_matrix"],
                atol=1e-6,
            )
            meta.update(label_count=len(labels), positive=bool(labels))
            path.write_text(json.dumps(meta, indent=2))
            by_sample[path.stem] = meta
            image_paths.append(str(image_path.resolve()))
            for input_path in [image_path, label_path]:
                report["input_sha256"][input_path.relative_to(OUT).as_posix()] = sha(
                    input_path
                )
            if (index + 1) % 80 == 0:
                print("CALIBRATED_AUDIT", variant, index + 1, flush=True)
        assert len(families) == 8 and min(families.values()) >= 20
        assert len(list((folder / "scenes").glob("*.blend"))) == 8
        stats = summary(totals, histogram, boxes)
        stats.update(
            positive_frames=positives,
            family_support=dict(families),
            fallback_placements=fallback,
            assigned_target_returns_verified=target_returns,
        )
        report["variants"][variant] = stats
        (OUT / (variant + "_train.txt")).write_text("\n".join(image_paths) + "\n")
        config = (
            f"path: {OUT.as_posix()}\ntrain: {variant}_train.txt\n"
            f"val: {(ROOT / 'real_val.txt').as_posix()}\n"
            "names:\n  0: road_vehicle\n"
        )
        (OUT / (variant + ".yaml")).write_text(config)
        report["input_sha256"][variant + "_train.txt"] = sha(
            OUT / (variant + "_train.txt")
        )
        report["input_sha256"][variant + ".yaml"] = sha(OUT / (variant + ".yaml"))
        metadata_by_variant[variant] = by_sample
    for sample in required:
        a, b = (metadata_by_variant[v][sample] for v in ["sensor_only", "sensor_pose"])
        assert a["ray_plan_sha256"] == b["ray_plan_sha256"]
        assert a["quota_sensor_range"] == b["quota_sensor_range"]
    target = report["reference"]["real_train"]["mean_roi_points"]
    old_error = abs(
        report["reference"]["original_synthetic_train"]["mean_roi_points"] - target
    )
    target_range = np.array(report["reference"]["real_train"]["range_fractions"])
    for stats in report["variants"].values():
        stats["mean_point_count_error_vs_real_train"] = abs(
            stats["mean_roi_points"] - target
        )
        stats["range_total_variation_vs_real_train"] = float(
            np.abs(np.array(stats["range_fractions"]) - target_range).sum() / 2
        )
        assert stats["mean_point_count_error_vs_real_train"] < old_error
    report["complete"] = True
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    # Representative training maps; gamma is display-only and never used by the detector.
    canvas = Image.new("RGB", (1200, 1020), "#20242a")
    draw = ImageDraw.Draw(canvas)
    for column, variant in enumerate(["sensor_only", "sensor_pose"]):
        samples = [m for m in metadata_by_variant[variant].values() if m["positive"]][
            :2
        ]
        draw.text(
            (column * 600 + 10, 10),
            variant + " calibrated TRAIN (green GT)",
            fill="white",
        )
        for row, meta in enumerate(samples):
            image = Image.open(
                OUT / variant / "images" / (meta["sample_id"] + ".png")
            ).convert("RGB")
            image = Image.fromarray(
                ((np.asarray(image) / 255.0) ** 0.6 * 255).astype(np.uint8)
            )
            painter = ImageDraw.Draw(image)
            for obj in meta["objects"]:
                if obj["evaluated"]:
                    _, x, y, w, h = label(obj["box"])
                    painter.rectangle(
                        (
                            (x - w / 2) * WIDTH,
                            (y - h / 2) * HEIGHT,
                            (x + w / 2) * WIDTH,
                            (y + h / 2) * HEIGHT,
                        ),
                        outline="lime",
                        width=2,
                    )
            image.thumbnail((240, 480))
            canvas.paste(image, (column * 600 + 180, 35 + row * 490))
    canvas.save(OUT / "calibrated_review.jpg")
    print(
        "CALIBRATED_AUDIT_COMPLETE",
        json.dumps({k: v for k, v in report.items() if k != "input_sha256"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
