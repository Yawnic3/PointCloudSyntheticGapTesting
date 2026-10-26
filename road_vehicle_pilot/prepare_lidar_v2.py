"""Freeze real LiDAR splits, fetch official frames, and build identical BEV inputs."""

import argparse
import hashlib
import io
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from fetch_real import RAW, archive
from lidar_common import BASE, ROOT, eligible, export_sample

GROUPS = {
    "train": [1, 3, 4, 7, 12, 15, 20],
    "val": [5, 10, 14, 18],
    "test": [2, 6, 8, 9, 11, 13, 16, 17, 19, 21],
}
TARGETS = {"train": [251, 69], "val": [30, 10], "test": [10, 30]}


def inventory():
    candidates = {split: [[], []] for split in GROUPS}
    for path in sorted(RAW.glob("*/*labels.json")):
        group = int(path.parent.name.split("_")[0])
        split = next(s for s, groups in GROUPS.items() if group in groups)
        data = json.loads(path.read_text())["openlabel"]
        for key, frame in sorted(data["frames"].items(), key=lambda kv: int(kv[0])):
            stream = frame["frame_properties"]["streams"].get("lidar")
            if not stream:
                continue
            objects = []
            for oid, obj in frame.get("objects", {}).items():
                if data["objects"][oid]["type"] != "road_vehicle":
                    continue
                for cuboid in obj.get("object_data", {}).get("cuboid", []):
                    assert cuboid.get("coordinate_system") == "lidar"
                    objects.append(
                        {
                            "instance_id": oid,
                            "box": cuboid["val"],
                            "attributes": cuboid.get("attributes", {}),
                        }
                    )
            record = {
                "sample_id": path.parent.name + "__" + key,
                "sequence": path.parent.name,
                "group": group,
                "frame_id": key,
                "split": split,
                "pcd_member": stream["uri"].lstrip("/"),
                "objects": objects,
            }
            positive = any(eligible(obj["box"]) for obj in objects)
            candidates[split][0 if positive else 1].append(record)
    selected = []
    rng = random.Random(2301)
    for split, pools in candidates.items():
        for pool, wanted in zip(pools, TARGETS[split]):
            assert len(pool) >= wanted, (split, len(pool), wanted)
            rng.shuffle(pool)
            # Spread selected frames round-robin over sequences before taking a prefix.
            by_sequence = {}
            for row in pool:
                by_sequence.setdefault(row["sequence"], []).append(row)
            spread = []
            while any(by_sequence.values()):
                for sequence in sorted(by_sequence):
                    if by_sequence[sequence]:
                        spread.append(by_sequence[sequence].pop())
            selected.extend(spread[:wanted])
    ROOT.mkdir(exist_ok=True)
    manifest = {
        "seed": 2301,
        "groups": GROUPS,
        "selection_targets_before_point_support_filter": TARGETS,
        "samples": selected,
        "source": "Official OSDaR23 v1.1 annotations and PCD, DOI 10.57806/9mv146r0",
        "scope": "road_vehicle, XYZ-only, axis-aligned BEV boxes; not official full 3D AP",
        "test_limit": "Only location group 6 has positive in-ROI LiDAR cuboids in this holdout; frames share physical vehicles. Group 6 was RGB validation earlier. This is a limited LiDAR pilot, not a fresh project-wide blind benchmark.",
    }
    destination = ROOT / "real_selection.json"
    if destination.exists():
        assert (
            json.loads(destination.read_text()) == manifest
        ), "Frozen selection changed"
    else:
        destination.write_text(json.dumps(manifest, indent=2))
    print(
        "FROZEN_SELECTION",
        {s: sum(r["split"] == s for r in selected) for s in GROUPS},
        flush=True,
    )
    return selected


def read_pcd(path):
    with path.open("rb") as handle:
        header = {}
        while True:
            line = handle.readline().decode("ascii").strip()
            if not line:
                raise ValueError("Incomplete PCD header")
            if line.startswith("#"):
                continue
            key, *values = line.split()
            header[key] = values
            if key == "DATA":
                break
        fields = header["FIELDS"]
        assert header["COUNT"] == ["1"] * len(fields)
        if header["DATA"] == ["ascii"]:
            data = np.fromstring(handle.read().decode("ascii"), sep=" ").reshape(
                -1, len(fields)
            )
            assert len(data) == int(header["POINTS"][0])
            xyz = data[:, [fields.index(k) for k in ["x", "y", "z"]]].astype(np.float32)
        else:
            raise ValueError(f"Unsupported PCD encoding: {header['DATA']}")
    return xyz[np.isfinite(xyz).all(axis=1)]


def download_sequence(sequence, records):
    todo = [
        r
        for r in records
        if not (ROOT / "real" / "metadata" / (r["sample_id"] + ".json")).exists()
    ]
    if not todo:
        return len(records)
    with archive(sequence) as zipped:
        for record in todo:
            path = RAW / sequence / record["pcd_member"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                data = zipped.read(record["pcd_member"])
                temporary = path.with_suffix(".partial")
                temporary.write_bytes(data)
                temporary.replace(path)
            xyz = read_pcd(path)
            metadata = {k: v for k, v in record.items() if k != "objects"}
            metadata["pcd_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            result = export_sample(
                "real", record["sample_id"], xyz, record["objects"], metadata
            )
            print(
                "REAL_LIDAR",
                record["sample_id"],
                len(xyz),
                result["label_count"],
                flush=True,
            )
            # Avoid retaining gigabytes of downloaded range bytes in the archive cache.
            zipped.fp.cache.clear()
    return len(records)


def fetch():
    records = inventory()
    grouped = {}
    for row in records:
        grouped.setdefault(row["sequence"], []).append(row)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(download_sequence, name, rows) for name, rows in grouped.items()
        ]
        for job in as_completed(jobs):
            job.result()
    print("REAL_LIDAR_READY", len(records), flush=True)


def synthetic():
    paths = sorted((ROOT / "synthetic" / "metadata").glob("*.json"))
    assert len(paths) == 480, f"Capture incomplete: {len(paths)}/480"
    for path in paths:
        meta = json.loads(path.read_text())
        yaw = meta["frame_yaw_world"]
        transform = np.eye(4)
        transform[:3, :3] = [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
        transform[:3, 3] = meta["frame_origin_world"]
        meta["lidar_to_world_matrix"] = transform.tolist()
        meta["lidar_to_blender_camera_matrix"] = (
            np.linalg.inv(np.array(meta["camera_matrix_world"])) @ transform
        ).tolist()
        meta["camera_axes"] = "Blender camera: x right, y up, -z forward"
        with np.load(ROOT / "synthetic" / "points" / (path.stem + ".npz")) as data:
            xyz = data["xyz"]
            instances = data["instance_id_ground_truth_only"]
            sensor_indices = data["sensor_index_ground_truth_only"]
        export_sample(
            "synthetic",
            path.stem,
            xyz,
            meta.pop("objects"),
            meta,
            instances,
            sensor_indices,
            write_points=False,
        )
    print("SYNTHETIC_BEV_READY", len(paths), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inventory", "fetch", "synthetic"])
    args = parser.parse_args()
    {"inventory": inventory, "fetch": fetch, "synthetic": synthetic}[args.mode]()
