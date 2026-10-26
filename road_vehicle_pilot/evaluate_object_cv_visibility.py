"""Secondary visibility-aware evaluation of frozen models; no relabeling in place."""

import json
from pathlib import Path

import numpy as np

from build_pointpillars_object_cv import OUT
from lidar3d_geometry import Cuboid, evaluate, from_seven, iou3d
from train_pointpillars_object_cv import ARMS


def hidden(obj):
    attributes = {
        row["name"]: row["val"] for row in obj.get("attributes", {}).get("text", [])
    }
    return str(attributes.get("occlusion", "")).replace(" ", "") == "100%"


def load_objects(listing):
    result = {}
    for name in listing.read_text().splitlines():
        image = Path(name)
        meta = json.loads(
            (image.parent.parent / "metadata" / (image.stem + ".json")).read_text()
        )
        result[image.stem] = [obj for obj in meta["objects"] if obj["evaluated"]]
    return result


def filtered(predictions, objects):
    result = {}
    ignored = 0
    for name, items in predictions.items():
        visible = [Cuboid(o["box"]) for o in objects[name] if not hidden(o)]
        invisible = [Cuboid(o["box"]) for o in objects[name] if hidden(o)]
        used = set()
        result[name] = []
        for p in sorted(items, key=lambda p: -p["score"]):
            box = Cuboid(from_seven(p["box"]))
            candidates = [
                (iou3d(box, gt), j) for j, gt in enumerate(visible) if j not in used
            ]
            if candidates and max(candidates)[0] >= 0.5:
                used.add(max(candidates)[1])
                result[name].append(p)
            elif any(iou3d(box, gt) >= 0.5 for gt in invisible):
                ignored += 1
            else:
                result[name].append(p)
    return result, ignored


def main():
    assert (OUT / "final_verification.json").exists()
    original = json.loads((OUT / "comparison.json").read_text())
    keys = ["pretrained"] + [
        arm + "_" + policy for arm in ARMS for policy in ["best", "last"]
    ]
    report = {
        "policy": "Secondary post-hoc diagnostic motivated by annotation documentation and metadata, not model scores. Retain original ROI/support eligibility, but exclude GT explicitly marked LiDAR occlusion100%. Give visible-GT matches priority; otherwise ignore predictions with 3D IoU>=.5 to a fully-occluded eligible GT. Other original exclusions remain unchanged. Same frozen predictions/weights. Models were still TRAINED and SELECTED using the original eligibility; this does not fix their training targets or checkpoint selection.",
        "original_ground_truth_boxes": 0,
        "visible_ground_truth_boxes": 0,
        "folds": [],
    }
    pooled = {key: {} for key in keys}
    pooled_truth = {}
    for index in range(5):
        root = OUT / f"fold_{index}"
        objects = load_objects(root / "real_test.txt")
        truth = {
            name: [o["box"] for o in items if not hidden(o)]
            for name, items in objects.items()
        }
        before = sum(map(len, objects.values()))
        after = sum(map(len, truth.values()))
        report["original_ground_truth_boxes"] += before
        report["visible_ground_truth_boxes"] += after
        pooled_truth.update(truth)
        row = {
            "fold": index,
            "original_GT": before,
            "visible_GT": after,
            "results": {},
            "split_fully_occluded_counts": {},
        }
        for split in ["real_train", "real_val", "real_test"]:
            data = load_objects(root / (split + ".txt"))
            targets = [o for items in data.values() for o in items if hidden(o)]
            row["split_fully_occluded_counts"][split] = {
                "box_instances": len(targets),
                "annotation_IDs": len({o["instance_id"] for o in targets}),
            }
        for key in keys:
            predictions = json.loads(
                (root / (key + "_test_predictions.json")).read_text()
            )
            selected, ignored = filtered(predictions, objects)
            metrics = evaluate(selected, truth, 0.5)
            if before == after:
                assert (
                    metrics
                    == original["folds"][index]["results"][key]["metrics"]["3D_AP50"]
                )
            row["results"][key] = {"metrics": metrics, "ignored_predictions": ignored}
            pooled[key].update(selected)
        report["folds"].append(row)
        print("VISIBILITY_DIAGNOSTIC_FOLD", index, before, after, flush=True)
    assert (
        report["original_ground_truth_boxes"] - report["visible_ground_truth_boxes"]
        == 24
    )
    report["summary"] = {
        key: {
            "macro_3D_AP50": float(
                np.mean(
                    [
                        fold["results"][key]["metrics"]["AP101"]
                        for fold in report["folds"]
                    ]
                )
            ),
            "pooled_3D_AP50": evaluate(predictions, pooled_truth, 0.5),
        }
        for key, predictions in pooled.items()
    }
    (OUT / "visibility_aware_comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        "",
        "## Secondary visibility-aware diagnostic",
        "",
        report["policy"],
        "",
        f"Original eligible GT: {report['original_ground_truth_boxes']}; excluding 100%-occluded GT: {report['visible_ground_truth_boxes']}. The 24 excluded box instances belong to one annotation track in Bergedorf. They contain 12-89 raw in-cuboid returns despite their fully-occluded annotation. Such returns do not prove that the sensor observed the vehicle itself.",
        "",
        "| Arm / checkpoint | Visibility-aware macro AP50 | Visibility-aware pooled AP50 | TP / FP / FN at .25 |",
        "|---|---:|---:|---|",
    ]
    for key, row in report["summary"].items():
        m = row["pooled_3D_AP50"]
        lines.append(
            f"| {key} | {row['macro_3D_AP50']:.2%} | {m['AP101']:.2%} | {m['tp']} / {m['fp']} / {m['fn']} |"
        )
    lines += [
        "",
        "The official research report, section2.4.3 (page21), specifies estimated true-size boxes for occluded objects. Section2.6.2 (page43) explains that camera objects invisible in LiDAR can receive projected LiDAR annotations with occlusion100%. This confirms that the raw in-cuboid point-count gate alone is insufficient to establish visibility. It does not establish that every other annotation is correct or that dimensions should be changed to visible-point extents. Source: [DZSF research report44/2023](https://edocs.tib.eu/files/e01fn23/1868657744.pdf), retained locally as osdar23_research_report.pdf; relevant pages were extracted and visually inspected.",
        "",
        "A clean follow-up must remove fully-occluded targets from TRAIN supervision/donor crops and apply ignore handling to validation/test before selection, then rerun matched training. The tables here cannot substitute for that retraining.",
    ]
    path = OUT / "REPORT.md"
    path.write_text(
        path.read_text().split("\n## Secondary visibility-aware diagnostic")[0]
        + "\n".join(lines)
        + "\n"
    )
    print("VISIBILITY_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
