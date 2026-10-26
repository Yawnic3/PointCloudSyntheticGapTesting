"""Audit source isolation, geometry and replay before multi-location training."""

import gc
import hashlib
import json

import numpy as np
import torch

from build_pointpillars_object_cv import OUT, group
from lidar_object_augmentation import (
    augment,
    database,
    inside,
    matrix,
    overlaps,
    prepare,
)
from train_lidar3d import Samples
from train_pretrained_pointpillars import sha


def main():
    torch.set_num_threads(4)
    splits = json.loads((OUT / "splits.json").read_text())
    report = {
        "source_hashes": {
            name: sha(OUT.parent / name)
            for name in [
                "lidar_object_augmentation.py",
                "build_pointpillars_object_cv.py",
            ]
        },
        "folds": [],
    }
    # Independent analytic checks for oriented support and collision rejection.
    box = [20, -10, 1, 4, 2, 2, 0.7]
    local = np.array([[0, 0, 0], [1.9, 0.9, 0.9], [2.2, 0, 0]])
    xyz = local @ matrix(0.7).T + box[:3]
    assert inside(xyz, box).tolist() == [0, 1]
    assert overlaps(box, box)
    assert not overlaps(box, [50, 20, 1, 4, 2, 2, -0.4])
    for fold in splits["folds"]:
        root = OUT / f"fold_{fold['fold']}"
        for source in ["real", "synthetic"]:
            data = Samples(root / (source + "_train.txt"))
            donors = database(data)
            train_names = {sample["id"] for sample in data.samples}
            assert all(
                r["sample"] in train_names
                for records in donors.values()
                for r in records
            )
            if source == "real":
                assert not {group(name) for name in train_names} & set(
                    fold["positive_validation_groups"] + [fold["positive_test_group"]]
                )
            positive = [i for i, sample in enumerate(data.samples) if sample["truth"]]
            negative = [
                i for i, sample in enumerate(data.samples) if not sample["truth"]
            ]
            selected = positive[:3] + negative[:3]
            pasted = 0
            rows = []
            for index in selected:
                sample = data.samples[index]
                before = hashlib.sha256(sample["xyz"].tobytes()).hexdigest()
                rng = np.random.default_rng(1000 + index + fold["fold"])
                tensor, original_boxes = data.prepare(index, rng, True)
                original = tensor.cpu().numpy()
                del tensor
                points, boxes, log = augment(
                    original,
                    original_boxes,
                    np.random.default_rng(80 + index + fold["fold"]),
                    donors,
                    fold["priors"]["domains"][source]["median_training_trackbed_z"],
                )
                assert np.isfinite(points).all() and len(points) <= 160000
                assert all(len(inside(points, b)) >= 5 for b in boxes)
                assert before == hashlib.sha256(sample["xyz"].tobytes()).hexdigest()
                for record in log["pasted"]:
                    b = record["box_before_global_z"]
                    assert not any(overlaps(b, old) for old in original_boxes)
                    assert record["source_sample"] in train_names
                pasted += len(log["pasted"])
                rows.append(
                    {
                        "sample": sample["id"],
                        "original_boxes": len(original_boxes),
                        "output_boxes": len(boxes),
                        "points": len(points),
                        "augmentation": log,
                    }
                )
                if (
                    fold["fold"] == 0
                    and source == "real"
                    and log["pasted"]
                    and not (OUT / "TRAIN_augmentation_example.npz").exists()
                ):
                    np.savez_compressed(
                        OUT / "TRAIN_augmentation_example.npz",
                        before=original,
                        after=points,
                        boxes=np.array(boxes),
                        original_boxes=np.array(original_boxes),
                    )
            assert pasted > 0
            row = {
                "fold": fold["fold"],
                "source": source,
                "donor_instance_ids": len(donors),
                "donor_observations": sum(map(len, donors.values())),
                "minimum_donor_returns": min(
                    len(r["local"]) for records in donors.values() for r in records
                ),
                "replay": rows,
                "pasted_instances": pasted,
            }
            report["folds"].append(row)
            print(
                "OBJECT_AUG_PREFLIGHT",
                fold["fold"],
                source,
                {
                    k: row[k]
                    for k in [
                        "donor_instance_ids",
                        "donor_observations",
                        "pasted_instances",
                    ]
                },
                flush=True,
            )
            del data, donors
            gc.collect()
    for name, expected in splits["inputs"].items():
        from pathlib import Path

        assert sha(Path(name)) == expected
    report["passed"] = True
    (OUT / "preflight.json").write_text(json.dumps(report, indent=2))
    print("OBJECT_AUG_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
