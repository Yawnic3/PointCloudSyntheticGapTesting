"""Build retrospective, location-separated folds without altering prior datasets."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from lidar_common import BASE, ROOT, eligible

OUT = BASE / "lidar_grouped_v3"
POSITIVE_GROUPS = [1, 4, 5, 6, 7]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    inventory = defaultdict(
        lambda: {
            "lidar_frames": 0,
            "positive_frames_in_roi": 0,
            "roi_boxes": 0,
            "ids": set(),
            "sequences": [],
        }
    )
    for path in sorted((BASE / "real_raw").glob("*/*labels.json")):
        group = int(path.parent.name.split("_")[0])
        row = inventory[group]
        row["sequences"].append(path.parent.name)
        data = json.loads(path.read_text())["openlabel"]
        for frame in data["frames"].values():
            if "lidar" not in frame.get("frame_properties", {}).get("streams", {}):
                continue
            row["lidar_frames"] += 1
            positive = False
            for oid, obj in frame.get("objects", {}).items():
                if data["objects"][oid]["type"] != "road_vehicle":
                    continue
                for cuboid in obj.get("object_data", {}).get("cuboid", []):
                    if cuboid.get("coordinate_system") == "lidar" and eligible(
                        cuboid["val"]
                    ):
                        row["roi_boxes"] += 1
                        row["ids"].add(oid)
                        positive = True
            row["positive_frames_in_roi"] += positive
    assert [
        g for g in sorted(inventory) if inventory[g]["positive_frames_in_roi"]
    ] == POSITIVE_GROUPS
    for row in inventory.values():
        row["unique_roi_annotation_ids"] = len(row.pop("ids"))
    rows = []
    hashes = {}
    ids_to_group = defaultdict(set)
    for path in sorted((ROOT / "real" / "metadata").glob("*.json")):
        meta = json.loads(path.read_text())
        image = ROOT / "real" / "images" / (path.stem + ".png")
        label = ROOT / "real" / "labels" / (path.stem + ".txt")
        for file in [path, image, label]:
            hashes[str(file)] = sha(file)
        evaluated = [o for o in meta["objects"] if o["evaluated"]]
        for obj in meta["objects"]:
            ids_to_group[obj["instance_id"]].add(meta["group"])
        rows.append(
            {
                "sample_id": path.stem,
                "image": str(image),
                "group": meta["group"],
                "sequence": meta["sequence"],
                "positive": bool(evaluated),
                "supported_ids": [o["instance_id"] for o in evaluated],
                "all_ids": [o["instance_id"] for o in meta["objects"]],
                "boxes": len(evaluated),
            }
        )
    assert len(rows) == 400 and all(
        len(groups) == 1 for groups in ids_to_group.values()
    )
    assignment = {group: i for i, group in enumerate(POSITIVE_GROUPS)}
    # Balance negatives independently so every outer fold receives them.
    loads = [0] * 5
    negative_groups = sorted(
        {r["group"] for r in rows} - set(POSITIVE_GROUPS),
        key=lambda g: (-sum(r["group"] == g for r in rows), g),
    )
    for group in negative_groups:
        fold = min(range(5), key=lambda i: (loads[i], i))
        assignment[group] = fold
        loads[fold] += sum(r["group"] == group for r in rows)
    plan = {
        "purpose": "Retrospective five-fold location-separated benchmark; not a fresh blind holdout",
        "seed": 20261025,
        "positive_location_groups": POSITIVE_GROUPS,
        "real_frame_pool": 400,
        "all_source_annotation_inventory": dict(inventory),
        "group_to_outer_fold": assignment,
        "input_sha256": hashes,
        "dataset_limit": "Only five location groups have road_vehicle cuboids fully inside the 160m x +/-40m ROI. All have prior project exposure. Creating train/validation/test each with multiple positive locations would need at least six disjoint positive locations. A genuinely fresh external holdout is unavailable locally.",
        "evaluation_policy": "Outer test location rotates across all five groups; validation uses the next group, training the other three. Select checkpoints by validation AP50:95 only. Report each outer location, equal-weight macro location AP, and pooled out-of-fold predictions; do not treat correlated frame boxes as independent vehicles.",
        "training_selection": "At most 40 positive frames per training location, retaining annotation-ID diversity, plus at most 60 training-group negatives spread across groups. Validation/test retain all frames in assigned groups. Equal synthetic/real frame counts and positive/negative frame counts within each fold; 30 epochs, same initial YOLO26s, seed, optimizer and preprocessing.",
        "synthetic_policy": "New procedural directions and full-ROI pose randomization; no empirical real return-direction templates or real placement records are used. Original asset family meshes and environmental recipes are reused. Prior project knowledge still informs this retrospective experiment.",
        "fresh_holdout_status": "NOT AVAILABLE: additional unseen sequences required; no fresh-blind-performance claim permitted",
        "folds": [],
    }
    for outer in range(5):
        folder = OUT / f"fold_{outer}"
        folder.mkdir(exist_ok=True)
        split_rows = {"train": [], "val": [], "test": []}
        for row in rows:
            assigned = assignment[row["group"]]
            split = (
                "test"
                if assigned == outer
                else "val" if assigned == (outer + 1) % 5 else "train"
            )
            split_rows[split].append(row)
        by_group = defaultdict(list)
        for row in split_rows["train"]:
            if row["positive"]:
                by_group[row["group"]].append(row)
        negatives = defaultdict(list)
        for row in split_rows["train"]:
            if not row["positive"]:
                negatives[row["group"]].append(row)
        rng = random.Random(plan["seed"] + outer)
        for pool in negatives.values():
            rng.shuffle(pool)
        train = []
        while len(train) < 60 and any(negatives.values()):
            for group in sorted(negatives):
                if negatives[group] and len(train) < 60:
                    train.append(negatives[group].pop())
        for group, pool in sorted(by_group.items()):
            rng = random.Random(plan["seed"] + group)
            pool = pool.copy()
            rng.shuffle(pool)
            picked, seen = [], set()
            while pool and len(picked) < 40:
                # Cover all available IDs before adding repeated observations.
                best = max(
                    range(len(pool)),
                    key=lambda i: len(set(pool[i]["supported_ids"]) - seen),
                )
                row = pool.pop(best)
                picked.append(row)
                seen.update(row["supported_ids"])
            train.extend(picked)
        split_rows["train"] = sorted(train, key=lambda r: r["sample_id"])
        details = {}
        for split, selected in split_rows.items():
            paths = [r["image"] for r in selected]
            target = folder / ("real_" + split + ".txt")
            target.write_text("\n".join(paths) + "\n")
            details[split] = {
                "frames": len(selected),
                "positive_frames": sum(r["positive"] for r in selected),
                "supported_boxes": sum(r["boxes"] for r in selected),
                "unique_supported_ids": len(
                    {v for r in selected for v in r["supported_ids"]}
                ),
                "groups": sorted({r["group"] for r in selected}),
                "positive_groups": sorted(
                    {r["group"] for r in selected if r["positive"]}
                ),
                "list_sha256": sha(target),
            }
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not {r["group"] for r in split_rows[a]} & {
                r["group"] for r in split_rows[b]
            }
            assert not {v for r in split_rows[a] for v in r["all_ids"]} & {
                v for r in split_rows[b] for v in r["all_ids"]
            }
        assert len(details["train"]["positive_groups"]) == 3
        (folder / "real.yaml").write_text(
            f"path: {folder.as_posix()}\ntrain: real_train.txt\nval: real_val.txt\ntest: real_test.txt\nnames:\n  0: road_vehicle\n"
        )
        plan["folds"].append(
            {
                "outer_fold": outer,
                "positive_test_group": POSITIVE_GROUPS[outer],
                "positive_validation_group": POSITIVE_GROUPS[(outer + 1) % 5],
                "splits": details,
            }
        )
    all_tests = [
        p
        for i in range(5)
        for p in (OUT / f"fold_{i}" / "real_test.txt").read_text().splitlines()
    ]
    assert len(all_tests) == len(set(all_tests)) == 400
    target = OUT / "benchmark_protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == json.loads(
            json.dumps(plan)
        ), "Frozen benchmark changed"
    else:
        target.write_text(json.dumps(plan, indent=2))
    print("GROUPED_BENCHMARK_FROZEN", json.dumps(plan["folds"]), flush=True)


if __name__ == "__main__":
    main()
