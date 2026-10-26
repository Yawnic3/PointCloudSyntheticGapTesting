"""TRAIN-only geometry and input-retention audit; never changes labels."""

import json
from pathlib import Path

import numpy as np

from lidar_common import rotation
from lidar3d_geometry import Cuboid, from_seven, iou3d
from official_pointpillars_windows import voxelize
from train_lidar3d import FOLD, canonical


def support(points, box):
    center = np.array(box[:3])
    size = np.array(box[7:])
    near = points[
        np.all(np.abs(points - center) <= np.linalg.norm(size) / 2 + 0.03, axis=1)
    ]
    return int(
        np.all(
            np.abs((near - center) @ rotation(box[3:7])) <= size / 2 + 0.03, axis=1
        ).sum()
    )


def main():
    base = Path(__file__).parent
    out = base / "pointpillars_head_transfer"
    priors = json.loads(
        (base / "pretrained_pointpillars/training_priors.json").read_text()
    )
    rng = np.random.default_rng(9223)
    report = {
        "policy": "TRAIN only. Unaugmented deterministic input-retention replay, not all stochastic training transforms.",
        "domains": {},
    }
    for source in ["real", "synthetic"]:
        records = []
        frame_stats = []
        for name in (FOLD / (source + "_train.txt")).read_text().splitlines():
            image = Path(name)
            root = image.parent.parent
            meta = json.loads((root / "metadata" / (image.stem + ".json")).read_text())
            with np.load(root / "points" / (image.stem + ".npz")) as data:
                points = data["xyz"]
            valid = (
                (points[:, 0] >= 0)
                & (points[:, 0] < 160)
                & (np.abs(points[:, 1]) < 40)
                & (points[:, 2] >= -3)
                & (points[:, 2] <= 8)
            )
            points = points[valid].copy()
            if len(points) > 160000:
                points = points[rng.choice(len(points), 160000, replace=False)]
            shift = priors["domains"][source]["z_shift"]
            points[:, 2] += shift
            rng.shuffle(points)
            voxels, counts, _, _, _ = voxelize(points, native=False, max_voxels=32000)
            kept = voxels[np.arange(32)[None, :] < counts[:, None], :3].copy()
            kept[:, 2] -= shift
            points[:, 2] -= shift
            frame_stats.append(
                {
                    "id": image.stem,
                    "input_points": len(points),
                    "kept_points": len(kept),
                    "voxels": len(voxels),
                }
            )
            for obj in meta["objects"]:
                if not obj["evaluated"]:
                    continue
                box = obj["box"]
                r = rotation(box[3:7])
                yaw = float(np.arctan2(r[1, 0], r[0, 0]))
                yaw_box = canonical(box[:3], box[7:], yaw)
                records.append(
                    {
                        "sample": image.stem,
                        "instance": obj["instance_id"],
                        "stored_support": obj["points_in_cuboid"],
                        "after_point_cap": support(points, box),
                        "after_voxel_cap": support(kept, box),
                        "yaw_projection_3d_iou": iou3d(
                            Cuboid(box), Cuboid(from_seven(yaw_box))
                        ),
                    }
                )
        report["domains"][source] = {
            "frames": frame_stats,
            "objects": records,
            "summary": {
                "boxes": len(records),
                "fewer_than_five_after_point_cap": sum(
                    r["after_point_cap"] < 5 for r in records
                ),
                "fewer_than_five_after_voxel_cap": sum(
                    r["after_voxel_cap"] < 5 for r in records
                ),
                "yaw_projection_iou_min_median": [
                    float(np.min([r["yaw_projection_3d_iou"] for r in records])),
                    float(np.median([r["yaw_projection_3d_iou"] for r in records])),
                ],
                "frames_hitting_voxel_cap": sum(
                    r["voxels"] == 32000 for r in frame_stats
                ),
            },
        }
        print(
            "TRAIN_AUDIT",
            source,
            json.dumps(report["domains"][source]["summary"]),
            flush=True,
        )
    (out / "training_input_audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
