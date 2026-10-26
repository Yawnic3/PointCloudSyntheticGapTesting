"""Build five disjoint location folds with two positive validation locations."""

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_common import BASE
from train_pretrained_pointpillars import bed_mode, sha

OUT = BASE / "pointpillars_object_aug_cv"
OLD = BASE / "lidar_grouped_v3"


def group(name):
    return int(Path(name).stem.split("_")[0])


def metadata(name):
    image = Path(name)
    return json.loads(
        (image.parent.parent / "metadata" / (image.stem + ".json")).read_text()
    )


def details(names):
    objects = [metadata(name)["objects"] for name in names]
    return {
        "frames": len(names),
        "positive_frames": sum(any(o["evaluated"] for o in items) for items in objects),
        "supported_boxes": sum(o["evaluated"] for items in objects for o in items),
        "supported_ids": sorted(
            {o["instance_id"] for items in objects for o in items if o["evaluated"]}
        ),
    }


def spread(names, maximum):
    if len(names) <= maximum:
        return names
    chosen = set()
    uncovered = {
        o["instance_id"]
        for name in names
        for o in metadata(name)["objects"]
        if o["evaluated"]
    }
    tracks = [
        {o["instance_id"] for o in metadata(name)["objects"] if o["evaluated"]}
        for name in names
    ]
    while uncovered:
        i = max(range(len(names)), key=lambda j: len(tracks[j] & uncovered))
        chosen.add(i)
        uncovered -= tracks[i]
    assert len(chosen) <= maximum
    candidates = list(np.linspace(0, len(names) - 1, maximum).astype(int)) + list(
        range(len(names))
    )
    for i in candidates:
        if len(chosen) >= maximum:
            break
        chosen.add(int(i))
    return [names[i] for i in sorted(chosen)]


def capped_validation(names, positive_groups):
    selected = []
    for location in positive_groups:
        local = [name for name in names if group(name) == location]
        positive = [
            name
            for name in local
            if any(o["evaluated"] for o in metadata(name)["objects"])
        ]
        negative = [name for name in local if name not in positive]
        selected += spread(positive, 20) + spread(negative, 4)
    other = [name for name in names if group(name) not in positive_groups]
    # Keep every available negative location represented before filling the cap.
    first = {
        next(name for name in other if group(name) == g)
        for g in {group(n) for n in other}
    }
    assert len(first) <= 20
    candidates = (
        [other[i] for i in np.linspace(0, len(other) - 1, len(other)).astype(int)]
        if other
        else []
    )
    for name in candidates:
        if len(first) >= 20:
            break
        first.add(name)
    return sorted(set(selected) | first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.rebuild:
        assert not (
            OUT / "protocol.json"
        ).exists(), "Frozen training protocol must be preserved"
    if (OUT / "splits.json").exists() and not args.rebuild:
        print("OBJECT_CV_SPLITS_EXIST", flush=True)
        return
    old = json.loads((OLD / "benchmark_protocol.json").read_text())
    positive = old["positive_location_groups"]
    pool = sorted(
        {
            name
            for i in range(5)
            for name in (OLD / f"fold_{i}/real_test.txt").read_text().splitlines()
        }
    )
    assert len(pool) == 400
    result = {
        "validation_cap": "Per positive location: up to 20 positive frames retaining all supported annotation IDs plus four negative frames. Up to 20 additional negative-location frames, retaining each negative location. All outer-test frames retained.",
        "policy": "Retrospective known corpus. Outer test membership unchanged; each outer test appears once. Add the next second positive location to validation, moving its entire group out of training. Two positive train / two validation / one test locations. Match real/synthetic total and positive/negative training-frame counts.",
        "folds": [],
        "inputs": {},
    }
    all_tests = []
    for i in range(5):
        dest = OUT / f"fold_{i}"
        dest.mkdir(exist_ok=True)
        extra = positive[(i + 2) % 5]
        val_groups = [positive[(i + 1) % 5], extra]
        original = OLD / f"fold_{i}"
        real_train = [
            name
            for name in (original / "real_train.txt").read_text().splitlines()
            if group(name) != extra
        ]
        val = sorted(
            set((original / "real_val.txt").read_text().splitlines())
            | {name for name in pool if group(name) == extra}
        )
        val = capped_validation(val, val_groups)
        test = (original / "real_test.txt").read_text().splitlines()
        groups = [{group(name) for name in names} for names in [real_train, val, test]]
        ids = [
            set(details(names)["supported_ids"]) for names in [real_train, val, test]
        ]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            assert not groups[a] & groups[b]
            assert not ids[a] & ids[b]
        real_info = details(real_train)
        source = (original / "synthetic_train.txt").read_text().splitlines()
        buckets = [
            [
                name
                for name in source
                if bool(any(o["evaluated"] for o in metadata(name)["objects"])) == flag
            ]
            for flag in [True, False]
        ]
        rng = np.random.default_rng(23 + i)
        counts = [
            real_info["positive_frames"],
            len(real_train) - real_info["positive_frames"],
        ]
        synthetic = sorted(
            name
            for bucket, count in zip(buckets, counts)
            for name in rng.choice(bucket, count, replace=False).tolist()
        )
        assert details(synthetic)["positive_frames"] == real_info["positive_frames"]
        fold = {
            "fold": i,
            "positive_validation_groups": val_groups,
            "positive_test_group": positive[i],
            "splits": {},
            "priors": {"domains": {}},
        }
        for name, names in {
            "real_train": real_train,
            "synthetic_train": synthetic,
            "real_val": val,
            "real_test": test,
        }.items():
            listing = dest / (name + ".txt")
            listing.write_text("\n".join(names) + "\n")
            fold["splits"][name] = {
                **details(names),
                "groups": (
                    sorted({group(n) for n in names})
                    if not name.startswith("synthetic")
                    else []
                ),
                "list_sha256": sha(listing),
            }
            for image_name in names:
                image = Path(image_name)
                root = image.parent.parent
                for path in [
                    root / "points" / (image.stem + ".npz"),
                    root / "metadata" / (image.stem + ".json"),
                ]:
                    if str(path) not in result["inputs"]:
                        result["inputs"][str(path)] = sha(path)
        for source_name, names in [("real", real_train), ("synthetic", synthetic)]:
            modes = []
            for image_name in names:
                image = Path(image_name)
                with np.load(
                    image.parent.parent / "points" / (image.stem + ".npz")
                ) as data:
                    mode = bed_mode(data["xyz"])
                if mode is not None:
                    modes.append(mode)
            median = float(np.median(modes))
            fold["priors"]["domains"][source_name] = {
                "median_training_trackbed_z": median,
                "z_shift": -1.6 - median,
            }
        (dest / "priors.json").write_text(json.dumps(fold["priors"], indent=2))
        result["folds"].append(fold)
        all_tests.extend(test)
        print(
            "OBJECT_CV_SPLIT",
            i,
            {
                name: (r["frames"], r["positive_frames"], len(r["supported_ids"]))
                for name, r in fold["splits"].items()
            },
            flush=True,
        )
    assert len(all_tests) == len(set(all_tests)) == 400
    (OUT / "splits.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
