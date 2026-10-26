"""Train two independent, identical YOLO26s BEV detectors and score one real holdout.

This is a LiDAR-only BEV baseline. It does not predict cuboid height or yaw and
must not be reported as official OSDaR23 3D AP. Both starts load the same weights.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
from pathlib import Path

from lidar_common import BASE, ROOT

os.environ["YOLO_CONFIG_DIR"] = str(BASE / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_threshold(model, paths, imgsz, device):
    from run_pilot import iou

    tp = fp = fn = 0
    frames = []
    for result in model.predict(
        paths,
        conf=0.25,
        iou=0.7,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False,
        max_det=300,
    ):
        h, w = result.orig_shape
        truth = []
        labels = (
            Path(result.path).parent.parent
            / "labels"
            / (Path(result.path).stem + ".txt")
        )
        for line in labels.read_text().splitlines():
            _, x, y, bw, bh = map(float, line.split())
            truth.append(
                ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
            )
        matched = set()
        frame_fp = 0
        predictions = sorted(result.boxes.data.cpu().tolist(), key=lambda p: -p[4])
        for prediction in predictions:
            matches = [
                (iou(prediction[:4], box), j)
                for j, box in enumerate(truth)
                if j not in matched
            ]
            if matches and max(matches)[0] >= 0.5:
                matched.add(max(matches)[1])
            else:
                frame_fp += 1
        tp += len(matched)
        fp += frame_fp
        fn += len(truth) - len(matched)
        frames.append(
            {
                "sample_id": Path(result.path).stem,
                "tp": len(matched),
                "fp": frame_fp,
                "fn": len(truth) - len(matched),
                "predictions_xyxy_conf_class": predictions,
            }
        )
    return {
        "confidence": 0.25,
        "matching_BEV_IoU": 0.5,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tag", default="matched30")
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.utils.torch_utils import init_seeds

    settings.update({"sync": False})
    audit = json.loads((ROOT / "audit.json").read_text())
    assert audit["complete"]
    assert (
        audit["stats"]["real_train"]["images"]
        == audit["stats"]["synthetic_train"]["images"]
        == 320
    )
    for name, expected in audit["list_sha256"].items():
        assert sha(ROOT / (name + ".txt")) == expected
    for name, expected in audit["source_sha256"].items():
        assert sha(BASE / name) == expected, (name, "Changed after audit")
    if args.device != "cpu":
        assert torch.cuda.is_available()
    weights = BASE / "models" / "yolo26s.pt"
    output = ROOT / "runs" / args.tag
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": 4,
        "nbs": 4,
        "device": args.device,
        "workers": 0,
        "seed": 23,
        "deterministic": True,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "patience": 0,
        "amp": False,
        "cache": False,
        "close_mosaic": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "bgr": 0.0,
        "plots": True,
        "save": True,
        "exist_ok": False,
    }
    protocol = {
        "task": "LiDAR-only axis-aligned vehicle BEV detection",
        "model": "YOLO26s detect",
        "initial_checkpoint_sha256": sha(weights),
        "training_settings_identical_for_both": common,
        "same_real_test_list_sha256": sha(ROOT / "real_test.txt"),
        "same_real_validation_list_sha256": sha(ROOT / "real_val.txt"),
        "audit_sha256": sha(ROOT / "audit.json"),
        "runner_sha256": sha(Path(__file__)),
        "checkpoint_policy": "last.pt after the fixed epoch budget; no test-based selection",
        "validation": "Same real validation for both, monitoring only",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "important_limit": audit["test_limit"],
        "scope": audit["policy"],
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        assert (
            json.loads(protocol_path.read_text()) == protocol
        ), "Frozen protocol changed"
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    checkpoints = {}
    for domain in ["real", "synthetic"]:
        run_path = output / domain
        checkpoint = run_path / "weights" / "last.pt"
        if checkpoint.exists():
            with (run_path / "results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert int(rows[-1]["epoch"]) == args.epochs, "Existing run is incomplete"
            print("REUSE_COMPLETED", domain, flush=True)
        else:
            init_seeds(23, deterministic=True)
            model = YOLO(str(weights))
            model.train(
                data=str(ROOT / (domain + ".yaml")),
                project=str(output),
                name=domain,
                **common,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assert checkpoint.exists()
        checkpoints[domain] = checkpoint
    # The shared real test is scored only after BOTH fixed-budget models exist.
    test_paths = (ROOT / "real_test.txt").read_text().splitlines()
    report = {"protocol": protocol, "results": {}}
    architecture_hashes = set()
    for domain, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint))
        architecture = [
            (name, list(tensor.shape))
            for name, tensor in model.model.state_dict().items()
        ]
        architecture_hash = hashlib.sha256(
            json.dumps(architecture).encode()
        ).hexdigest()
        architecture_hashes.add(architecture_hash)
        metrics = model.val(
            data=str(ROOT / "real.yaml"),
            split="test",
            imgsz=args.imgsz,
            batch=4,
            device=args.device,
            workers=0,
            conf=0.001,
            iou=0.7,
            project=str(output),
            name=domain + "_real_test",
            plots=True,
            save_json=True,
            exist_ok=False,
        )
        fixed = fixed_threshold(model, test_paths, args.imgsz, args.device)
        row = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "architecture_sha256": architecture_hash,
            "train_images": 320,
            "test_images": len(test_paths),
            "BEV_AP50": float(metrics.box.map50),
            "BEV_AP50_95": float(metrics.box.map),
            "fixed_confidence_0_25": fixed,
        }
        report["results"][domain + "_trained"] = row
        print(
            "LIDAR_TEST_RESULT",
            domain,
            row["BEV_AP50"],
            {k: v for k, v in fixed.items() if k != "frames"},
            flush=True,
        )
    assert len(architecture_hashes) == 1, "Trained model architectures differ"
    report["identical_architecture_verified"] = True
    report["synthetic_minus_real_AP50_percentage_points"] = 100 * (
        report["results"]["synthetic_trained"]["BEV_AP50"]
        - report["results"]["real_trained"]["BEV_AP50"]
    )
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Matched LiDAR-only BEV comparison",
        "",
        "Both YOLO26s models started from the same checkpoint and trained for the same budget on 320 frames. Both were evaluated on the identical 40 real OSDaR23 frames.",
        "",
        "| Training source | BEV AP50 | BEV AP50:95 | TP / FP / FN at confidence 0.25 |",
        "|---|---:|---:|---|",
    ]
    for name, row in report["results"].items():
        fixed = row["fixed_confidence_0_25"]
        lines.append(
            f"| {name} | {row['BEV_AP50']:.2%} | {row['BEV_AP50_95']:.2%} | {fixed['tp']} / {fixed['fp']} / {fixed['fn']} |"
        )
    lines.extend(
        [
            "",
            "These are axis-aligned BEV scores from XYZ-derived height/density/spread images, not official 3D AP. Cars, buses, trucks and other annotated road vehicles share one class. Five-point support and full-ROI inclusion define evaluated objects; excluded objects are retained in metadata but not ignored for false positives.",
            "",
            audit["test_limit"],
            "",
            "One seed and one fixed training budget; the real frames contain repeated physical vehicles. Equal frame counts do not imply equal instance counts or diversity. Synthetic sensor coverage is approximate; no intensity, material response or motion distortion is simulated.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("LIDAR_COMPARISON_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
