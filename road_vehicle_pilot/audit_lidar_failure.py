"""Read-only source/geometry audit and post-test failure diagnostics."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lidar_common import (
    BASE,
    ROOT,
    bev,
    bounds,
    corners,
    eligible,
    label,
    point_count,
    rotation,
)

OUT = BASE / "lidar_calibrated" / "failure_audit"
os.environ["YOLO_CONFIG_DIR"] = str(BASE / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    return np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist() if values else []


def attr(obj, key):
    return next(
        (
            v["val"]
            for v in obj.get("attributes", {}).get("text", [])
            if v["name"] == key
        ),
        "unknown",
    )


def visual_overlays(metas, predictions):
    folder = OUT / "overlays"
    folder.mkdir(exist_ok=True)
    sheets = []
    for meta in metas:
        name = meta["sample_id"]
        raw = np.asarray(Image.open(ROOT / "real" / "images" / (name + ".png")))
        # Display contrast only. The model inputs remain untouched.
        bright = (255 * (raw.astype(float) / 255) ** 0.55).round().astype(np.uint8)
        canvas = Image.new("RGB", (960, 1010), "#171b22")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 5), name, fill="white")
        draw.text(
            (10, 23),
            "GT green | prediction red (.25) | brighter display only",
            fill="white",
        )
        for col, source in enumerate(["real_reference", "sensor_pose"]):
            image = Image.fromarray(bright.copy())
            d = ImageDraw.Draw(image)
            for index, obj in enumerate(o for o in meta["objects"] if o["evaluated"]):
                _, x, y, w, h = label(obj["box"])
                rect = [
                    (x - w / 2) * 480,
                    (y - h / 2) * 960,
                    (x + w / 2) * 480,
                    (y + h / 2) * 960,
                ]
                d.rectangle(rect, outline="lime", width=2)
                d.text((rect[0], max(0, rect[1] - 11)), str(index + 1), fill="lime")
            for pred in predictions[source].get(name, []):
                d.rectangle(pred[:4], outline="red", width=2)
            canvas.paste(image, (col * 480, 50))
            draw.text((col * 480 + 10, 38), source, fill="white")
        path = folder / (name + ".jpg")
        canvas.save(path, quality=94)
        if len(sheets) < 4:
            thumbnail = canvas.copy()
            thumbnail.thumbnail((720, 758))
            sheets.append(thumbnail)
    contact = Image.new("RGB", (1440, 1516), "#171b22")
    for i, image in enumerate(sheets):
        contact.paste(image, ((i % 2) * 720, (i // 2) * 758))
    contact.save(OUT / "test_overview.jpg", quality=94)


def plot_diagnostics(rows, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = [
        ("range_m", "Vehicle center range (m)"),
        ("y_m", "Lateral position: +left (m)"),
        ("support", "Points in full cuboid"),
        ("pixel_short_side", "BEV box short side (pixels)"),
        ("occupancy", "Occupied BEV cells / box area"),
        ("visible_fraction", "Cuboid points inside feature Z range"),
    ]
    for ax, (key, title) in zip(axes.flat, fields):
        data = [
            [r[key] for r in rows if r["split"] == split]
            for split in ["train", "val", "test"]
        ]
        ax.boxplot(data, tick_labels=["Train", "Val", "Test"], showfliers=False)
        ax.set_title(title)
        if key == "support":
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Real supported vehicles: correlated frame observations, not independent vehicles"
    )
    fig.tight_layout()
    fig.savefig(OUT / "distribution_comparison.png", dpi=160)
    plt.close(fig)
    tests = [r for r in rows if r["split"] == "test"]
    near = min(tests, key=lambda r: r["range_m"])
    sparse = min(tests, key=lambda r: r["support"])
    train = min(
        (r for r in rows if r["split"] == "train"),
        key=lambda r: abs(r["range_m"] - sparse["range_m"]),
    )
    cases = [
        ("Training, similar range", train),
        ("Test, nearest vehicle", near),
        ("Test, sparse vehicle", sparse),
    ]
    fig = plt.figure(figsize=(15, 12))
    rng = np.random.default_rng(23)
    edges = [
        (a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1
    ]
    for i, (title, row) in enumerate(cases):
        xyz = np.load(ROOT / "real" / "points" / (row["sample_id"] + ".npz"))["xyz"]
        box = row["box"]
        local = (xyz - box[:3]) @ rotation(box[3:7])
        context_mask = np.all(
            np.abs(local) <= np.array(box[7:]) / 2 + [3, 3, 1.5], axis=1
        )
        points = xyz[context_mask]
        if len(points) > 8000:
            points = points[rng.choice(len(points), 8000, replace=False)]
        inside = np.all(
            np.abs((points - box[:3]) @ rotation(box[3:7]))
            <= np.array(box[7:]) / 2 + 0.03,
            axis=1,
        )
        vertices = corners(box)
        for col, dims in enumerate([(0, 1), (0, 2), (0, 1, 2)]):
            ax = fig.add_subplot(
                3, 3, i * 3 + col + 1, projection="3d" if col == 2 else None
            )
            color = np.where(inside, "#239b45", "#999999")
            if col == 2:
                ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2], c=color, s=1, alpha=0.6
                )
                for a, b in edges:
                    ax.plot(*vertices[[a, b]].T, color="#e78012", linewidth=1)
                ax.set_zlabel("Z up (m)")
                ax.view_init(elev=23, azim=-65)
                ax.set_box_aspect((2, 2, 1))
            else:
                ax.scatter(
                    points[:, dims[0]], points[:, dims[1]], c=color, s=2, alpha=0.6
                )
                for a, b in edges:
                    ax.plot(
                        vertices[[a, b], dims[0]],
                        vertices[[a, b], dims[1]],
                        color="#e78012",
                        linewidth=1,
                    )
                ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("X forward (m)")
            ax.set_ylabel("Z up (m)" if col == 1 else "Y left (m)")
            ax.set_title(
                f"{title}: {row['range_m']:.1f} m, {row['support']} points"
                if col == 0
                else ["", "Side view", "3D cloud and cuboid"][col]
            )
    fig.suptitle(
        "Cloud / cuboid alignment: orange GT edges, green enclosed points, gray context"
    )
    fig.tight_layout()
    fig.savefig(OUT / "cloud_box_alignment.png", dpi=160)
    plt.close(fig)
    (OUT / "visual_cases.json").write_text(json.dumps(cases, indent=2))


def main():
    from fetch_real import RAW
    from prepare_lidar_v2 import read_pcd
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    OUT.mkdir(exist_ok=True)
    source_audit = json.loads((ROOT / "audit.json").read_text())
    test = json.loads(
        (BASE / "lidar_calibrated/test_comparison/comparison.json").read_text()
    )
    validation = json.loads(
        (BASE / "lidar_calibrated/runs/validation_comparison.json").read_text()
    )
    annotations = {}
    rows = []
    metas = {}
    summaries = {}
    checked = 0
    raw_checked = 0
    all_ids = {}
    all_cloud_hashes = {}
    all_sequences = {}
    integrity = {}
    for split in ["train", "val", "test"]:
        paths = (ROOT / f"real_{split}.txt").read_text().splitlines()
        assert (
            sha(ROOT / f"real_{split}.txt")
            == source_audit["list_sha256"][f"real_{split}"]
        )
        ids = set()
        sequences = set()
        positive = []
        excluded = Counter()
        roi_counts = []
        z_clipped = 0
        for name in paths:
            sample = Path(name).stem
            m = json.loads((ROOT / "real/metadata" / (sample + ".json")).read_text())
            assert m["sample_id"] == sample and m["split"] == split
            sequence = m["sequence"]
            sequences.add(sequence)
            if sequence not in annotations:
                path = next((RAW / sequence).glob("*labels.json"))
                annotations[sequence] = json.loads(path.read_text())["openlabel"]
            source = annotations[sequence]
            frame = source["frames"][m["frame_id"]]
            uri = frame["frame_properties"]["streams"]["lidar"]["uri"].lstrip("/")
            assert uri == m["pcd_member"]
            with np.load(ROOT / "real/points" / (sample + ".npz")) as data:
                xyz = data["xyz"]
            assert np.array_equal(bev(xyz), np.asarray(Image.open(name).convert("RGB")))
            cloud_hash = hashlib.sha256(xyz.tobytes()).hexdigest()
            assert (
                cloud_hash not in all_cloud_hashes
                or all_cloud_hashes[cloud_hash] == split
            )
            all_cloud_hashes[cloud_hash] = split
            feature = (
                (xyz[:, 0] >= 0)
                & (xyz[:, 0] < 160)
                & (np.abs(xyz[:, 1]) < 40)
                & (xyz[:, 2] >= -3)
                & (xyz[:, 2] <= 8)
            )
            roi_counts.append(int(feature.sum()))
            expected = []
            pixels = np.asarray(Image.open(name))
            for index, obj in enumerate(m["objects"]):
                ids.add(obj["instance_id"])
                cuboids = frame["objects"][obj["instance_id"]]["object_data"]["cuboid"]
                assert source["objects"][obj["instance_id"]]["type"] == "road_vehicle"
                assert any(
                    c["coordinate_system"] == "lidar"
                    and np.allclose(c["val"], obj["box"], rtol=0, atol=1e-10)
                    for c in cuboids
                )
                box = obj["box"]
                support = point_count(xyz, box)
                assert support == obj["points_in_cuboid"]
                evaluated = eligible(box) and support >= 5
                assert evaluated == obj["evaluated"]
                if not evaluated:
                    excluded[
                        "outside_roi" if not eligible(box) else "under_5_points"
                    ] += 1
                    continue
                expected.append(label(box))
                local = (xyz - box[:3]) @ rotation(box[3:7])
                inside = np.all(np.abs(local) <= np.array(box[7:]) / 2 + 0.03, axis=1)
                visible = int((inside & feature).sum())
                z_clipped += visible < support
                _, x, y, w, h = label(box)
                x0 = max(0, int((x - w / 2) * 480))
                x1 = min(480, int(np.ceil((x + w / 2) * 480)))
                y0 = max(0, int((y - h / 2) * 960))
                y1 = min(960, int(np.ceil((y + h / 2) * 960)))
                patch = pixels[y0:y1, x0:x1]
                rows.append(
                    {
                        "split": split,
                        "sample_id": sample,
                        "sequence": sequence,
                        "instance_id": obj["instance_id"],
                        "box": box,
                        "range_m": float(np.linalg.norm(box[:2])),
                        "y_m": box[1],
                        "support": support,
                        "visible_fraction": visible / max(1, support),
                        "pixel_short_side": min(w * 480, h * 960),
                        "occupancy": float((patch[:, :, 1] > 0).mean()),
                        "occlusion": attr(obj, "occlusion"),
                        "subtype": attr(obj, "type"),
                    }
                )
            actual = (
                np.loadtxt(
                    Path(name).parent.parent / "labels" / (sample + ".txt"), ndmin=2
                )
                if expected
                else np.empty((0, 5))
            )
            assert np.allclose(actual, np.array(expected).reshape(-1, 5), atol=1e-7)
            if expected:
                positive.append(m)
            # Raw PCD reparse all positive test scans and one scan per other sequence.
            if (split == "test" and expected) or sequence not in integrity:
                raw = RAW / sequence / m["pcd_member"]
                assert sha(raw) == m["pcd_sha256"]
                assert np.array_equal(read_pcd(raw), xyz)
                integrity[sequence] = True
                raw_checked += 1
            checked += 1
            if checked % 80 == 0:
                print("FAILURE_AUDIT_GEOMETRY", checked, flush=True)
        selected = [r for r in rows if r["split"] == split]
        all_ids[split] = ids
        all_sequences[split] = sequences
        metas[split] = positive
        summaries[split] = {
            "frames": len(paths),
            "positive_frames": len(positive),
            "positive_sequences": dict(Counter(m["sequence"] for m in positive)),
            "supported_boxes": len(selected),
            "unique_supported_ids": len(set(r["instance_id"] for r in selected)),
            "all_unique_annotation_ids": len(ids),
            "mean_roi_points": float(np.mean(roi_counts)),
            "excluded": dict(excluded),
            "feature_clipped_objects": z_clipped,
            "occlusion": dict(Counter(r["occlusion"] for r in selected)),
            "subtypes": dict(Counter(r["subtype"] for r in selected)),
            "quantiles_min_q1_median_q3_max": {
                key: quantiles([r[key] for r in selected])
                for key in [
                    "range_m",
                    "y_m",
                    "support",
                    "pixel_short_side",
                    "occupancy",
                    "visible_fraction",
                ]
            },
            "lateral_left_fraction": float(np.mean([r["y_m"] > 0 for r in selected])),
            "under_50_points_fraction": float(
                np.mean([r["support"] < 50 for r in selected])
            ),
        }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not all_ids[a] & all_ids[b] and not all_sequences[a] & all_sequences[b]
    report = {
        "purpose": "Post-test diagnostic audit; test set now explicitly used for diagnosis, not future tuning claims",
        "weights_changed": False,
        "source_and_geometry_checks": {
            "real_frames_checked": checked,
            "raw_pcd_reparse_checks": raw_checked,
            "official_cuboids_match": True,
            "png_matches_xyz_bev": True,
            "point_support_and_labels_match": True,
            "no_cross_split_UUID_sequence_or_exact_cloud": True,
        },
        "real_split_statistics": summaries,
        "prior_final_epoch_training_fit": json.loads(
            (ROOT / "runs/matched30/source_fit_diagnostics.json").read_text()
        ),
        "fresh_real_training_fit": {},
    }
    predictions = {
        source: {
            f["sample_id"]: f["predictions_xyxy_conf_class"]
            for f in row["fixed_confidence_0_25"]["frames"]
        }
        for source, row in test["results"].items()
    }
    visual_overlays(metas["test"], predictions)
    plot_diagnostics(rows, summaries)
    (OUT / "object_observations.json").write_text(json.dumps(rows, indent=2))
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    for policy in ["best", "last"]:
        checkpoint = Path(validation["results"]["real_reference"][policy]["checkpoint"])
        before = sha(checkpoint)
        model = YOLO(str(checkpoint))
        metric = model.val(
            data=str(ROOT / "real.yaml"),
            split="train",
            imgsz=960,
            batch=4,
            device="0",
            workers=0,
            conf=0.001,
            iou=0.7,
            plots=False,
            save_json=False,
            project=str(OUT),
            name="real_train_" + policy,
        )
        assert sha(checkpoint) == before
        report["fresh_real_training_fit"][policy] = {
            "checkpoint_sha256": before,
            "BEV_AP50": float(metric.box.map50),
            "BEV_AP50_95": float(metric.box.map),
            "usage": "Training fit only; not generalization",
        }
        (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    print("FAILURE_AUDIT_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
