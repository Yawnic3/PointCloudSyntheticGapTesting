"""Index the synthetic evaluation inputs without inventing model scores."""

import json, csv
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "moving_dataset"
OUT = BASE / "comparison"
OUT.mkdir(exist_ok=True)
manifest = json.loads((DATA / "manifest.json").read_text())
verification = json.loads((DATA / "verification.json").read_text())
rows = []
for annotation in sorted((DATA / "captures" / "annotations").glob("*.json")):
    record = json.loads(annotation.read_text())
    stem = annotation.stem
    rgb = DATA / "captures" / "rgb" / (stem + ".png")
    lidar = DATA / "captures" / "lidar" / (stem + ".npz")
    assert rgb.is_file() and lidar.is_file()
    rows.append(
        {
            "sample_id": stem,
            "time_s": record["time_s"],
            "route_m": record["route_m"],
            "rgb": str(rgb.resolve()),
            "lidar": str(lidar.resolve()),
            "annotations": str(annotation.resolve()),
        }
    )
assert len(rows) == 63
with (OUT / "synthetic_samples.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (OUT / "class_mapping.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "synthetic_class_id",
            "synthetic_class_name",
            "scene_assemblies",
            "lidar_observed_instances",
            "model_class_id",
            "real_dataset_class_name",
            "include_in_comparison",
        ],
    )
    writer.writeheader()
    for i, name in manifest["classes"].items():
        writer.writerow(
            {
                "synthetic_class_id": i,
                "synthetic_class_name": name,
                "scene_assemblies": manifest["target_assembly_counts"][name],
                "lidar_observed_instances": verification[
                    "unique_lidar_observed_instances"
                ][name],
                "model_class_id": "",
                "real_dataset_class_name": "",
                "include_in_comparison": "unresolved",
            }
        )
config = {
    "status": "awaiting_model_and_real_data; ground_truth_alignment_required",
    "model": {
        "rgb_checkpoint": None,
        "lidar_checkpoint": None,
        "inference_repository": None,
        "model_config": None,
        "preprocessing": None,
        "class_mapping": "class_mapping.csv",
    },
    "datasets": {
        "synthetic_index": "synthetic_samples.csv",
        "real_osdar23_root": None,
        "real_evaluation_split": None,
        "synthetic_usage": "evaluation_only; one route with correlated frames",
    },
    "protocol": {
        "primary_question": "Performance of the same frozen checkpoint on real and synthetic inputs.",
        "freeze_on_both": [
            "checkpoint weights",
            "input preprocessing and image resize",
            "point features and point sampling",
            "confidence and NMS settings",
            "class mapping",
            "matching thresholds",
            "evaluator implementation",
        ],
        "report": [
            "per-class support and AP",
            "macro average over explicitly shared classes with ground-truth support in both datasets",
            "precision and recall at the same fixed operating point",
            "results by distance and visibility where comparable labels exist",
            "synthetic minus real metric differences",
        ],
        "metric_selection": "Use the existing model benchmark/evaluator. RGB box AP for matched 2D detection policies, 3D/BEV AP only for matched cuboid policies; segmentation/polyline metrics for corresponding tasks. Do not combine these into a single mAP.",
        "no_retraining_for_primary_comparison": True,
    },
    "required_alignment": [
        "Synthetic RGB labels are projected amodal world AABBs; can include fully occluded objects. Generate or review ground truth to match real-data visible/amodal conventions before scoring.",
        "Synthetic 3D boxes are world AABBs; recover oriented full-assembly cuboids and transform to the detector coordinate frame if required by its real-data benchmark.",
        "Track labels are artificial 10m mesh segments, not official continuous track polylines. Exclude track from box AP unless a shared valid task definition is established.",
        "Synthetic transition/switch geometry and grouped-object annotation granularity need alignment with the chosen real-data labels.",
        "Resolve wheelchair and group_of_animals against actual real split support; absent-class AP is undefined, not zero.",
        "Synthetic NPZ provides XYZ and labels, with no measured LiDAR intensity/ring/time features. Do not pass labels to the model or invent physical features silently.",
        "Keep ground truth and instance IDs out of model inputs.",
        "Synthetic cameras, mounting heights and geometric LiDAR sampling are assumed, not OSDaR23 calibration. Report this domain difference.",
        "The 63 frames share one route and repeated objects. Do not treat frames as independent sequences or split adjacent frames for training/testing.",
    ],
}
(OUT / "comparison_config.json").write_text(json.dumps(config, indent=2))
(OUT / "readiness.json").write_text(
    json.dumps(
        {
            "synthetic_samples": len(rows),
            "synthetic_classes": len(manifest["classes"]),
            "synthetic_lidar_points": verification["total_lidar_points"],
            "scores_computed": False,
            "missing": [
                "trained checkpoint(s)",
                "model inference/evaluation code and configuration",
                "real OSDaR23 evaluation data and split or existing predictions plus ground truth",
                "resolved model/real/synthetic class mapping",
            ],
            "ground_truth_alignment_required": True,
        },
        indent=2,
    )
)
print(
    json.dumps(
        {
            "indexed_samples": len(rows),
            "classes": len(manifest["classes"]),
            "output": str(OUT.resolve()),
            "scores_computed": False,
        }
    )
)
