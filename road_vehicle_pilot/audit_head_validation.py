"""Locate proposal failures on validation without changing training or selection."""

import json
from pathlib import Path

import numpy as np

from lidar3d_geometry import Cuboid, bev_iou, from_seven, iou3d
from train_lidar3d import FOLD


def main():
    out = Path(__file__).parent / "pointpillars_head_transfer"
    truth = {}
    for image in (FOLD / "real_val.txt").read_text().splitlines():
        image = Path(image)
        meta = json.loads(
            (image.parent.parent / "metadata" / (image.stem + ".json")).read_text()
        )
        truth[image.stem] = [o["box"] for o in meta["objects"] if o["evaluated"]]
    paths = {"pretrained": out / "pretrained_validation.json"}
    for name in ["real_lr1e5", "real_lr1e4"]:
        if (out / name / "completed.json").exists():
            paths[name] = out / name / "best_validation_predictions.json"
    report = {
        "scope": "Validation only. Independent per-GT maximum overlaps among saved post-NMS proposals; not AP or one-to-one detection recall. Does not change checkpoints or hyperparameters.",
        "results": {},
    }
    for name, path in paths.items():
        predictions = json.loads(path.read_text())
        if name == "pretrained":
            predictions = predictions["predictions"]
        rows = []
        for sample, boxes in truth.items():
            proposals = predictions[sample]
            candidates = [Cuboid(from_seven(p["box"])) for p in proposals]
            for values in boxes:
                gt = Cuboid(values)
                overlaps = np.array([bev_iou(gt, p) for p in candidates])
                spatial = np.array([iou3d(gt, p) for p in candidates])
                row = {
                    "sample": sample,
                    "maximum_BEV_IoU": float(overlaps.max()) if len(overlaps) else 0,
                    "maximum_3D_IoU": float(spatial.max()) if len(spatial) else 0,
                }
                if len(overlaps):
                    best = int(overlaps.argmax())
                    p = candidates[best]
                    row.update(
                        {
                            "best_BEV_proposal_score": proposals[best]["score"],
                            "best_BEV_height_error_m": float(
                                p.center[2] - gt.center[2]
                            ),
                            "best_BEV_horizontal_error_m": float(
                                np.linalg.norm(p.center[:2] - gt.center[:2])
                            ),
                        }
                    )
                rows.append(row)
        summary = {
            "ground_truth_boxes": len(rows),
            "has_BEV_IoU_at_least_025": sum(r["maximum_BEV_IoU"] >= 0.25 for r in rows),
            "has_BEV_IoU_at_least_050": sum(r["maximum_BEV_IoU"] >= 0.5 for r in rows),
            "has_3D_IoU_at_least_025": sum(r["maximum_3D_IoU"] >= 0.25 for r in rows),
            "has_3D_IoU_at_least_050": sum(r["maximum_3D_IoU"] >= 0.5 for r in rows),
        }
        report["results"][name] = {"summary": summary, "objects": rows}
        print("VALIDATION_PROPOSAL_AUDIT", name, json.dumps(summary), flush=True)
    (out / "validation_proposal_audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
