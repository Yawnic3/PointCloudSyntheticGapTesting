"""Audit expanded geometry/BEV labels and freeze matched per-fold training lists."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lidar_common import BASE, ROOT, bev, eligible, label, point_count, rotation

OUT = BASE / "lidar_grouped_v3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys(obj):
    b = obj["box"]
    return [
        "family:" + obj["family"],
        f"range_side:{min(int(b[0]//40),3)}:{'left' if b[1]>0 else 'right'}",
        f"lateral:{'left' if b[1]>0 else 'right'}:{'outer' if abs(b[1])>=20 else 'inner'}",
        f"support:{'sparse' if obj['points_in_cuboid']<50 else 'dense'}",
    ]


def main():
    protocol = json.loads((OUT / "benchmark_protocol.json").read_text())
    config = json.loads((OUT / "calibration.json").read_text())
    folder = OUT / "expanded"
    paths = sorted((folder / "metadata").glob("*.json"))
    assert len(paths) == 320
    expected = {f"route_{r:02d}_{i:04d}" for r in range(8) for i in range(40)}
    assert {p.stem for p in paths} == expected
    for sub in ["images", "labels"]:
        (folder / sub).mkdir(exist_ok=True)
    samples = []
    families = Counter()
    coverage = Counter()
    totals = []
    returns = 0
    fallback = 0
    inputs = {}
    for i, path in enumerate(paths):
        m = json.loads(path.read_text())
        assert (
            m["generator_sha256"] == config["generator_sha256"]
            and m["expanded_postprocess_complete"]
        )
        assert (
            not m["real_placement_records_used"]
            and not m["empirical_return_templates_used"]
        )
        with np.load(folder / "points" / (path.stem + ".npz")) as a:
            xyz = a["xyz"]
            ids = a["instance_id_ground_truth_only"]
        assert len(xyz) == m["point_count"] and np.isfinite(xyz).all()
        assert (
            (xyz[:, 0] >= 0)
            & (xyz[:, 0] < 160)
            & (np.abs(xyz[:, 1]) < 40)
            & (xyz[:, 2] >= -3)
            & (xyz[:, 2] <= 8)
        ).all()
        totals.append(len(xyz))
        rows = []
        features = []
        for obj in m["objects"]:
            b = obj["box"]
            assigned = xyz[ids == obj["mask_id"]]
            local = (assigned - b[:3]) @ rotation(b[3:7])
            assert np.all(np.abs(local) <= np.array(b[7:]) / 2 + 0.15), (
                path.stem,
                obj["mask_id"],
            )
            returns += len(assigned)
            support = point_count(xyz, b)
            obj.update(
                points_in_cuboid=support,
                inside_roi=eligible(b),
                evaluated=eligible(b) and support >= 5,
            )
            if obj["evaluated"]:
                rows.append(label(b))
                families[obj["family"]] += 1
                ks = keys(obj)
                features.extend(ks)
                coverage.update(ks)
        fallback += sum(v["fallback_original_pose"] for v in m["placement"])
        image = folder / "images" / (path.stem + ".png")
        Image.fromarray(bev(xyz)).save(image)
        assert np.array_equal(np.asarray(Image.open(image)), bev(xyz))
        labels = folder / "labels" / (path.stem + ".txt")
        labels.write_text(
            "".join("0 " + " ".join(f"{v:.8f}" for v in r[1:]) + "\n" for r in rows)
        )
        m["label_count"] = len(rows)
        path.write_text(json.dumps(m, indent=2))
        for file in [image, labels, path]:
            inputs[str(file)] = sha(file)
        samples.append(
            {
                "image": str(image),
                "sample_id": path.stem,
                "positive": bool(rows),
                "features": features,
                "boxes": len(rows),
            }
        )
        if (i + 1) % 80 == 0:
            print("EXPANDED_AUDIT", i + 1, flush=True)
    assert len(families) == 8 and min(families.values()) >= 40
    assert fallback == 0
    assert all(
        coverage[f"range_side:{r}:{side}"] >= 20
        for r in range(4)
        for side in ["left", "right"]
    )
    assert all(coverage[f"lateral:{side}:outer"] >= 40 for side in ["left", "right"])
    assert len(list((folder / "scenes").glob("*.blend"))) == 8
    positive = [s for s in samples if s["positive"]]
    negative = [s for s in samples if not s["positive"]]
    details = []
    for fold in protocol["folds"]:
        i = fold["outer_fold"]
        dest = OUT / f"fold_{i}"
        target = fold["splits"]["train"]
        n_pos = target["positive_frames"]
        n_neg = target["frames"] - n_pos
        assert len(positive) >= n_pos and len(negative) >= n_neg
        rng = random.Random(20261025 + i)
        pool = positive.copy()
        rng.shuffle(pool)
        chosen = []
        counts = Counter()
        for _ in range(n_pos):
            index = max(
                range(len(pool)),
                key=lambda j: sum(
                    1 / (1 + counts[k]) for k in set(pool[j]["features"])
                ),
            )
            sample = pool.pop(index)
            chosen.append(sample)
            counts.update(sample["features"])
        chosen.extend(rng.sample(negative, n_neg))
        rng.shuffle(chosen)
        assert (
            sum(s["positive"] for s in chosen) == n_pos
            and len(chosen) == target["frames"]
        )
        assert len({s["image"] for s in chosen}) == len(chosen)
        assert all(counts[f"family:{f}"] > 0 for f in families)
        assert all(
            counts[f"range_side:{r}:{side}"] > 0
            for r in range(4)
            for side in ["left", "right"]
        )
        listing = dest / "synthetic_train.txt"
        listing.write_text("\n".join(s["image"] for s in chosen) + "\n")
        yaml = dest / "synthetic.yaml"
        yaml.write_text(
            f"path: {dest.as_posix()}\ntrain: synthetic_train.txt\nval: real_val.txt\ntest: real_test.txt\nnames:\n  0: road_vehicle\n"
        )
        details.append(
            {
                "fold": i,
                "training_frames": len(chosen),
                "positive_frames": n_pos,
                "negative_frames": n_neg,
                "supported_synthetic_boxes": sum(s["boxes"] for s in chosen),
                "supported_real_boxes": target["supported_boxes"],
                "coverage": dict(counts),
                "list_sha256": sha(listing),
                "yaml_sha256": sha(yaml),
            }
        )
    # Validate that original real inputs and the frozen lists are unchanged.
    for name, expected_sha in protocol["input_sha256"].items():
        assert sha(Path(name)) == expected_sha
    for fold in protocol["folds"]:
        for split, stats in fold["splits"].items():
            assert (
                sha(OUT / f"fold_{fold['outer_fold']}" / f"real_{split}.txt")
                == stats["list_sha256"]
            )
    report = {
        "complete": True,
        "benchmark_protocol_sha256": sha(OUT / "benchmark_protocol.json"),
        "generator_sha256": config["generator_sha256"],
        "auditor_sha256": sha(Path(__file__)),
        "frames": 320,
        "positive_frames": len(positive),
        "negative_frames": len(negative),
        "mean_roi_points": float(np.mean(totals)),
        "point_count_min_max": [min(totals), max(totals)],
        "family_supported_boxes": dict(families),
        "coverage": dict(coverage),
        "verified_assigned_object_returns": returns,
        "placement_fallbacks": fallback,
        "input_sha256": inputs,
        "folds": details,
        "limits": config["limit"],
        "fresh_holdout_available": False,
    }
    (OUT / "expanded_audit.json").write_text(json.dumps(report, indent=2))
    canvas = Image.new("RGB", (1440, 1000), "#171b22")
    draw = ImageDraw.Draw(canvas)
    picks = [positive[j] for j in [0, len(positive) // 3, 2 * len(positive) // 3]]
    for col, sample in enumerate(picks):
        m = json.loads(
            (folder / "metadata" / (sample["sample_id"] + ".json")).read_text()
        )
        raw = np.asarray(Image.open(sample["image"]))
        image = Image.fromarray(
            (255 * (raw.astype(float) / 255) ** 0.55).round().astype(np.uint8)
        )
        d = ImageDraw.Draw(image)
        for obj in m["objects"]:
            if obj["evaluated"]:
                _, x, y, w, h = label(obj["box"])
                d.rectangle(
                    [
                        (x - w / 2) * 480,
                        (y - h / 2) * 960,
                        (x + w / 2) * 480,
                        (y + h / 2) * 960,
                    ],
                    outline="lime",
                    width=2,
                )
        canvas.paste(image, (480 * col, 40))
        draw.text(
            (480 * col + 10, 10), sample["sample_id"] + " | green GT", fill="white"
        )
    canvas.save(OUT / "expanded_review.jpg", quality=94)
    print(
        "EXPANDED_AUDIT_COMPLETE",
        json.dumps({k: v for k, v in report.items() if k != "input_sha256"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
