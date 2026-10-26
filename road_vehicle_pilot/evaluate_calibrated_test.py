"""Freeze validation-selected checkpoints, then evaluate one shared real test."""

import hashlib
import json
import os
from pathlib import Path

from lidar_common import BASE, ROOT

os.environ["YOLO_CONFIG_DIR"] = str(BASE / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    from compare_lidar_v2 import fixed_threshold
    from train_calibrated_lidar import architecture

    settings.update({"sync": False})
    calibration = BASE / "lidar_calibrated"
    validation_path = calibration / "runs" / "validation_comparison.json"
    validation = json.loads(validation_path.read_text())
    audit = json.loads((ROOT / "audit.json").read_text())
    assert audit["complete"] and validation["identical_architecture_verified"]
    paths = (ROOT / "real_test.txt").read_text().splitlines()
    assert len(paths) == len(set(paths)) == 40
    assert sha(ROOT / "real_test.txt") == audit["list_sha256"]["real_test"]
    for split in ["real_train.txt", "real_val.txt"]:
        assert not set(paths).intersection((ROOT / split).read_text().splitlines())
    inputs, objects, positive = {}, 0, 0
    for name in paths:
        image = Path(name)
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        lines = label.read_text().splitlines()
        objects += len(lines)
        positive += bool(lines)
        for path in [image, label]:
            inputs[str(path)] = sha(path)
    assert objects == 86 and positive == 10
    checkpoints = {}
    for source in ["real_reference", "sensor_pose"]:
        row = validation["results"][source]["best"]
        path = Path(row["checkpoint"])
        assert path.name == "best.pt" and sha(path) == row["checkpoint_sha256"]
        checkpoints[source] = {
            "path": str(path),
            "sha256": sha(path),
            "architecture_sha256": row["architecture_sha256"],
        }
    output = calibration / "test_comparison"
    output.mkdir(exist_ok=True)
    plan = {
        "checkpoint_policy": "Frozen best.pt selected by real validation AP50:95 for both; no test selection or tuning",
        "checkpoints": checkpoints,
        "validation_report_sha256": sha(validation_path),
        "test_list_sha256": sha(ROOT / "real_test.txt"),
        "test_input_sha256": inputs,
        "frames": len(paths),
        "supported_boxes": objects,
        "positive_frames": positive,
        "native_evaluation": {
            "imgsz": 960,
            "batch": 4,
            "device": "0",
            "workers": 0,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
        },
        "fixed_confidence": 0.25,
        "fixed_matching_IoU": 0.5,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "runner_sha256": sha(Path(__file__)),
        "test_limit": audit["test_limit"],
        "prior_exposure": "This existing held-out split was scored for the original fixed-epoch baseline; it is not a fresh blind test.",
        "synthetic_training_limit": "Synthetic generator fitted real TRAIN statistics and annotated placement priors. Equal 320-frame budgets contain different supported-object counts.",
        "task": "Single road_vehicle class, LiDAR-only axis-aligned BEV; not full 3D AP",
    }
    plan_path = output / "frozen_protocol.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        plan_path.write_text(json.dumps(plan, indent=2))
    print("TEST_PROTOCOL_FROZEN", plan_path, flush=True)
    result_path = output / "comparison.json"
    if result_path.exists():
        print("TEST_ALREADY_COMPLETE", result_path, flush=True)
        return
    report = {"protocol": plan, "results": {}}
    signatures = set()
    for source, row in checkpoints.items():
        model = YOLO(row["path"])
        signature = architecture(model)
        assert signature == row["architecture_sha256"]
        signatures.add(signature)
        metrics = model.val(
            data=str(ROOT / "real.yaml"),
            split="test",
            project=str(output),
            name=source,
            plots=True,
            save_json=True,
            exist_ok=False,
            **plan["native_evaluation"],
        )
        fixed = fixed_threshold(model, paths, 960, "0")
        assert fixed["tp"] + fixed["fn"] == objects
        assert {f["sample_id"] for f in fixed["frames"]} == {
            Path(p).stem for p in paths
        }
        report["results"][source] = {
            "BEV_mAP50": float(metrics.box.map50),
            "BEV_mAP50_95": float(metrics.box.map),
            "architecture_sha256": signature,
            "fixed_confidence_0_25": fixed,
        }
        print(
            "FROZEN_TEST_RESULT",
            source,
            metrics.box.map50,
            {k: v for k, v in fixed.items() if k != "frames"},
            flush=True,
        )
    assert len(signatures) == 1
    report["identical_architecture_verified"] = True
    report["synthetic_minus_real_AP50_percentage_points"] = 100 * (
        report["results"]["sensor_pose"]["BEV_mAP50"]
        - report["results"]["real_reference"]["BEV_mAP50"]
    )
    result_path.write_text(json.dumps(report, indent=2))
    lines = [
        "# Frozen-checkpoint real OSDaR23 test comparison",
        "",
        "Both models use the same architecture, initialization, seed, 320 training frames and 30-epoch budget. Checkpoints were selected on real validation AP50:95 and frozen before this test evaluation. No training, threshold tuning or checkpoint selection used these new test results.",
        "",
        "Evaluated on the identical 40 held-out real frames: 10 positive frames, 30 negative frames, 86 supported vehicle boxes. Metrics are single-class LiDAR BEV scores, not full 3D AP.",
        "",
        "| Training source | BEV mAP@50 | BEV mAP@50:95 | Precision at .25 | Recall at .25 | TP / FP / FN |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source, row in report["results"].items():
        f = row["fixed_confidence_0_25"]
        lines.append(
            f"| {source} | {row['BEV_mAP50']:.2%} | {row['BEV_mAP50_95']:.2%} | {f['precision']:.2%} | {f['recall']:.2%} | {f['tp']} / {f['fp']} / {f['fn']} |"
        )
    lines.extend(
        [
            "",
            f"Synthetic minus real mAP@50: {report['synthetic_minus_real_AP50_percentage_points']:+.2f} percentage points.",
            "",
            plan["prior_exposure"],
            "",
            plan["test_limit"],
            "",
            plan["synthetic_training_limit"],
            "",
            "The original validation advantage is a development result and must be assessed separately from this held-out result. One seed and correlated vehicles limit conclusions. Predictions on excluded low-support/boundary objects are not ignored by the evaluator.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("FROZEN_TEST_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
