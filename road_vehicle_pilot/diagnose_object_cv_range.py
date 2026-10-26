"""Secondary range/support recall diagnostic; never selects models or recipes."""

import argparse
import json
import math
from pathlib import Path

from build_pointpillars_object_cv import OUT
from lidar3d_geometry import Cuboid, from_seven, iou3d
from train_pointpillars_object_cv import ARMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visibility-aware", action="store_true")
    args = parser.parse_args()
    comparison = json.loads((OUT / "comparison.json").read_text())
    assert (OUT / "final_verification.json").exists()
    keys = ["pretrained"] + [arm + "_best" for arm in ARMS]
    predictions = {key: {} for key in keys}
    objects = {}
    for i in range(5):
        root = OUT / f"fold_{i}"
        for name in (root / "real_test.txt").read_text().splitlines():
            image = Path(name)
            meta = json.loads(
                (image.parent.parent / "metadata" / (image.stem + ".json")).read_text()
            )
            assert image.stem not in objects
            objects[image.stem] = [o for o in meta["objects"] if o["evaluated"]]
        for key in keys:
            predictions[key].update(
                json.loads((root / (key + "_test_predictions.json")).read_text())
            )
    if args.visibility_aware:
        from evaluate_object_cv_visibility import filtered, hidden

        comparison = json.loads((OUT / "visibility_aware_comparison.json").read_text())
        predictions = {
            key: filtered(pred, objects)[0] for key, pred in predictions.items()
        }
        objects = {
            name: [o for o in items if not hidden(o)] for name, items in objects.items()
        }
    truth = {name: [Cuboid(o["box"]) for o in items] for name, items in objects.items()}
    ranges = [
        (0, 30, "<30m"),
        (30, 60, "30-60m"),
        (60, 100, "60-100m"),
        (100, math.inf, "100m+"),
    ]
    supports = [
        (5, 20, "5-19 returns"),
        (20, 50, "20-49 returns"),
        (50, 200, "50-199 returns"),
        (200, math.inf, "200+ returns"),
    ]
    rows = [
        (
            name,
            j,
            math.hypot(*obj["box"][:2]),
            obj["points_in_cuboid"],
            obj["instance_id"],
        )
        for name, items in objects.items()
        for j, obj in enumerate(items)
    ]
    result = {
        "policy": "Secondary diagnostic from already-frozen selected-checkpoint predictions. Recall at confidence .25 and exact 3D IoU .5, not AP. Fixed range and raw in-cuboid-return bins. No model, threshold or recipe selection. Box instances are correlated frames, not independent vehicles.",
        "frames": len(objects),
        "ground_truth_boxes": len(rows),
        "models": {},
    }
    for key in keys:
        used = {name: set() for name in objects}
        tp = fp = 0
        ordered = sorted(
            [
                (p["score"], name, p["box"])
                for name, items in predictions[key].items()
                for p in items
                if p["score"] >= 0.25
            ],
            key=lambda item: -item[0],
        )
        for score, name, values in ordered:
            box = Cuboid(from_seven(values))
            candidates = [
                (iou3d(box, other), j)
                for j, other in enumerate(truth[name])
                if j not in used[name]
            ]
            if candidates and max(candidates)[0] >= 0.5:
                used[name].add(max(candidates)[1])
                tp += 1
            else:
                fp += 1
        expected = comparison["summary"][key]["pooled_3D_AP50"]
        assert (tp, fp, len(rows) - tp) == (
            expected["tp"],
            expected["fp"],
            expected["fn"],
        )
        model = {"tp": tp, "fp": fp, "fn": len(rows) - tp, "range": {}, "support": {}}
        for kind, bins, position in [("range", ranges, 2), ("support", supports, 3)]:
            for low, high, label in bins:
                subset = [row for row in rows if low <= row[position] < high]
                count = sum(j in used[name] for name, j, *_ in subset)
                model[kind][label] = {
                    "ground_truth_boxes": len(subset),
                    "tp": count,
                    "recall": count / len(subset) if subset else None,
                }
        result["models"][key] = model
        print("OBJECT_CV_RANGE_DIAGNOSTIC", key, tp, fp, len(rows) - tp, flush=True)
    filename = (
        "range_support_visible_diagnostic.json"
        if args.visibility_aware
        else "range_support_diagnostic.json"
    )
    heading = (
        "## Secondary visible-only distance and point-support diagnostic"
        if args.visibility_aware
        else "## Secondary distance and point-support diagnostic"
    )
    result["visibility_aware"] = args.visibility_aware
    (OUT / filename).write_text(json.dumps(result, indent=2))
    lines = [
        "",
        heading,
        "",
        result["policy"],
    ]
    for kind, title in [
        ("range", "Horizontal distance"),
        ("support", "Raw returns inside the GT cuboid"),
    ]:
        lines += [
            "",
            f"### {title}",
            "",
            "| Bin | GT boxes | Pretrained recall | Real control | Real augmented | Synthetic control | Synthetic augmented |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, row in result["models"]["pretrained"][kind].items():
            values = []
            for key in keys:
                value = result["models"][key][kind][label]["recall"]
                values.append("n/a" if value is None else f"{value:.2%}")
            lines.append(
                f"| {label} | {row['ground_truth_boxes']} | "
                + " | ".join(values)
                + " |"
            )
    path = OUT / "REPORT.md"
    path.write_text(path.read_text().split("\n" + heading)[0] + "\n".join(lines) + "\n")
    print("OBJECT_CV_RANGE_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
