"""Regenerate the exact RGB-v2 arrangements and capture first-return geometric LiDAR.

Run with Blender --background --python this_file -- --routes 0 1 ... .
The six idealized scan fans approximate coverage, not calibrated device response.
No RGB pixels or annotation IDs enter the model features.
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import generate_vehicle_v2 as geometry
from lidar_common import ROOT


def scan_pattern():
    fans = []
    azimuth, elevation = np.meshgrid(
        np.deg2rad(np.arange(-180, 180, 0.2)), np.deg2rad(np.linspace(-25, 15, 64))
    )
    fans.append((azimuth.ravel(), elevation.ravel(), (0, 0, 2.6)))
    rng = np.random.default_rng(2301)
    for yaw in [-15, 0, 15]:
        # Nonrepeating rectangular fan; not the proprietary Livox scan pattern.
        a = np.deg2rad(rng.uniform(yaw - 7.25, yaw + 7.25, 20000))
        e = np.deg2rad(rng.uniform(-9.1, 7.1, 20000))
        fans.append((a, e, (0.1, yaw / 30, 2.8)))
    for yaw in [-70, 70]:
        a = np.deg2rad(rng.uniform(yaw - 90, yaw + 90, 15000))
        e = np.deg2rad(rng.uniform(-70, 15, 15000))
        fans.append((a, e, (0, np.sign(yaw) * 0.7, 1.1)))
    return [
        (
            np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=1),
            Vector(origin),
            160 if i < 4 else 20,
        )
        for i, (a, e, origin) in enumerate(fans)
    ]


PATTERN = scan_pattern()


def capture(scene, camera, records, sample, split, route):
    output = ROOT / "synthetic"
    for name in ["points", "metadata"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    if (output / "metadata" / (sample + ".json")).exists():
        print("LIDAR_ALREADY_DONE", sample, flush=True)
        return
    started = time.monotonic()
    original = json.loads(
        (BASE / "synthetic_v2" / "metadata" / (sample + ".json")).read_text()
    )
    assert len(records) == len(original["objects"])
    for regenerated, saved in zip(records, original["objects"]):
        for key in [
            "mask_id",
            "instance_id",
            "family",
            "dimensions_m",
            "paint",
            "location_world",
            "heading_rad",
        ]:
            assert regenerated[key] == saved[key], (sample, key)
    assert np.allclose(
        np.array(camera.matrix_world), original["camera_matrix_world"], atol=1e-6
    )
    station = original["station_m"]
    bend = random.Random(original["route_seed"]).uniform(-0.000035, 0.000035)
    origin = Vector((station, bend * station * station, 0))
    yaw = math.atan(2 * bend * station)
    orient = Matrix.Rotation(yaw, 3, "Z")
    # All world-space mesh surfaces contribute to occlusion, including hard negatives.
    vertices, polygons, ids = [], [], []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        offset = len(vertices)
        vertices.extend(obj.matrix_world @ v.co for v in obj.data.vertices)
        polygons.extend(
            tuple(offset + i for i in p.vertices) for p in obj.data.polygons
        )
        ids.extend([obj.pass_index] * len(obj.data.polygons))
    tree = BVHTree.FromPolygons(vertices, polygons)
    points, instance_ids, sensor_ids = [], [], []
    noise = np.random.default_rng(original["sample_seed"] + 771)
    for sensor, (directions, local_origin, max_range) in enumerate(PATTERN):
        ray_origin = origin + orient @ local_origin
        for direction in directions:
            vector = orient @ Vector(direction)
            hit, normal, face, distance = tree.ray_cast(ray_origin, vector, max_range)
            if hit is None or distance < 0.5:
                continue
            p = orient.transposed() @ (hit - origin)
            if 0 <= p.x < 160 and abs(p.y) < 40 and -3 <= p.z <= 8:
                points.append(tuple(p))
                instance_ids.append(ids[face])
                sensor_ids.append(sensor)
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    # Modest range noise; no material response, multipath, weather or time distortion.
    xyz += noise.normal(0, 0.015, xyz.shape).astype(np.float32)
    objects = []
    for record in records:
        root = bpy.data.objects[record["root"]]
        inverse = root.matrix_world.inverted()
        local = np.array(
            [
                tuple(inverse @ child.matrix_world @ Vector(corner))
                for child in root.children
                if child.type == "MESH"
                for corner in child.bound_box
            ]
        )
        lo, hi = local.min(axis=0), local.max(axis=0)
        center_world = root.matrix_world @ Vector((lo + hi) / 2)
        center = orient.transposed() @ (center_world - origin)
        angle = record["heading_rad"] - yaw
        box = (
            list(center)
            + [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
            + list(hi - lo)
        )
        objects.append(
            {
                "instance_id": record["instance_id"],
                "mask_id": record["mask_id"],
                "family": record["family"],
                "box": box,
            }
        )
    np.savez_compressed(
        output / "points" / (sample + ".npz"),
        xyz=xyz,
        instance_id_ground_truth_only=np.asarray(instance_ids, dtype=np.int16),
        sensor_index_ground_truth_only=np.asarray(sensor_ids, dtype=np.int8),
    )
    meta = {
        "sample_id": sample,
        "split": split,
        "route_id": route,
        "objects": objects,
        "point_count": len(xyz),
        "rgb_scene_match_verified": True,
        "rgb_metadata_sha256": hashlib.sha256(
            (BASE / "synthetic_v2" / "metadata" / (sample + ".json")).read_bytes()
        ).hexdigest(),
        "frame_origin_world": list(origin),
        "frame_yaw_world": yaw,
        "camera_matrix_world": original["camera_matrix_world"],
        "sampling": "Six idealized first-return scan fans, 205200 rays; XYZ noise sigma 0.015 m; approximate coverage only.",
        "coordinate_system": "x forward along local track tangent, y left, z up; origin at route station world z=0",
        "ray_count": sum(len(fan[0]) for fan in PATTERN),
        "seconds": time.monotonic() - started,
    }
    (output / "metadata" / (sample + ".json")).write_text(json.dumps(meta, indent=2))
    print("LIDAR_SAMPLE", sample, len(xyz), round(meta["seconds"], 2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("--stop", type=int, default=40)
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    for route in args.routes:
        geometry.MATERIALS.clear()
        geometry.CUBES.clear()
        geometry.render_route(route, 40, 0, args.stop, capture=capture)
    print("LIDAR_CAPTURE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
