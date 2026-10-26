"""Check source fit after the frozen comparison; never retrain or rescore real test."""

import json
import os
import hashlib
from pathlib import Path

from lidar_common import BASE, ROOT

os.environ["YOLO_CONFIG_DIR"] = str(BASE / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"


def main():
    from ultralytics import YOLO

    output = ROOT / "runs" / "matched30"
    comparison = json.loads((output / "comparison.json").read_text())
    report = {
        "purpose": "Post-comparison source-fit diagnostics; not held-out real performance",
        "weights_changed": False,
        "real_test_rescored": False,
    }
    for domain in ["real", "synthetic"]:
        checkpoint = comparison["results"][domain + "_trained"]["checkpoint"]
        model = YOLO(checkpoint)
        metrics = model.val(
            data=str(ROOT / (domain + ".yaml")),
            split="train",
            imgsz=960,
            batch=4,
            device="0",
            workers=0,
            conf=0.001,
            iou=0.7,
            project=str(output / "fit_diagnostics"),
            name=domain + "_training_fit",
            plots=False,
        )
        report[domain + "_training_fit"] = {
            "AP50": float(metrics.box.map50),
            "AP50_95": float(metrics.box.map),
            "images": 320,
            "usage": "Training-data fit only; not generalization",
        }
    config = ROOT / "synthetic_diagnostics.yaml"
    config.write_text(
        f"path: {ROOT.as_posix()}\n"
        "train: synthetic_train.txt\nval: synthetic_val.txt\ntest: synthetic_test.txt\n"
        "names:\n  0: road_vehicle\n"
    )
    model = YOLO(comparison["results"]["synthetic_trained"]["checkpoint"])
    metrics = model.val(
        data=str(config),
        split="val",
        imgsz=960,
        batch=4,
        device="0",
        workers=0,
        conf=0.001,
        iou=0.7,
        project=str(output / "fit_diagnostics"),
        name="synthetic_unseen_routes",
        plots=False,
    )
    report["synthetic_unseen_routes"] = {
        "AP50": float(metrics.box.map50),
        "AP50_95": float(metrics.box.map),
        "images": 80,
        "usage": "Synthetic routes 8 and 9; not real-data performance",
    }
    for row in comparison["results"].values():
        assert (
            hashlib.sha256(Path(row["checkpoint"]).read_bytes()).hexdigest()
            == row["checkpoint_sha256"]
        )
    (output / "source_fit_diagnostics.json").write_text(json.dumps(report, indent=2))
    report_path = output / "REPORT.md"
    text = report_path.read_text().split("\n## Source fit diagnostics")[0]
    lines = [
        "",
        "## Source fit diagnostics",
        "",
        "These checks ran after the frozen comparison, without changing weights or rescoring real test.",
        "",
        "| Diagnostic split | BEV AP50 |",
        "|---|---:|",
    ]
    for name in [
        "real_training_fit",
        "synthetic_training_fit",
        "synthetic_unseen_routes",
    ]:
        lines.append(f"| {name} | {report[name]['AP50']:.2%} |")
    lines.extend(
        [
            "",
            "Training-fit scores use training frames and do not measure generalization. Unseen synthetic routes measure synthetic-domain generalization only.",
        ]
    )
    report_path.write_text(text + "\n".join(lines) + "\n")
    print("LIDAR_FIT_DIAGNOSTICS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
