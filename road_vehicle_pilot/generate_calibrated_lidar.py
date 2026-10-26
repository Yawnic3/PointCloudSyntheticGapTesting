"""Raycast two training-informed LiDAR ablations from the preserved Blender scenes.

sensor_only keeps the RGB-v2 world poses. sensor_pose also samples TRAIN-derived
vehicle placement priors, with supporting service-road surfaces and rigid-object
collision rejection. No real point coordinates are inserted into either cloud.
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

OUT = BASE / "lidar_calibrated"
FIT = json.loads((OUT / "calibration.json").read_text())
assert (
    hashlib.sha256((OUT / "direction_templates.npz").read_bytes()).hexdigest()
    == FIT["direction_templates_sha256"]
)
GENERATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with np.load(OUT / "direction_templates.npz") as archive:
    POOLS = {name: archive[name] for name in archive.files}
EDGES = np.array(FIT["range_edges_m"])
NBINS = len(EDGES) - 1
MOUNTS = np.array(FIT["assumed_sensor_origins_m"])


def plan(sample_seed):
    rng = np.random.default_rng(sample_seed + 19231)
    profile_index = int(rng.integers(len(FIT["frame_histograms_sensor_range"])))
    profile = np.array(FIT["frame_histograms_sensor_range"][profile_index], dtype=float)
    desired = int(rng.choice(FIT["training_point_totals"]))
    quotas = rng.multinomial(desired, profile.ravel() / profile.sum()).reshape(6, NBINS)
    directions = []
    for sensor in range(6):
        chunks = []
        for b in range(NBINS):
            pool = POOLS[f"s{sensor}_b{b}"]
            count = min(len(pool), int(quotas[sensor, b] * 3))
            if count:
                chunks.append(pool[rng.choice(len(pool), count, replace=False)])
        directions.append(np.concatenate(chunks) if chunks else np.empty((0, 3)))
    fingerprint = hashlib.sha256()
    for rows in directions:
        fingerprint.update(rows.tobytes())
    return {
        "directions": directions,
        "quotas": quotas,
        "profile_index": profile_index,
        "desired": desired,
        "sha256": fingerprint.hexdigest(),
    }


def frame(original):
    station = original["station_m"]
    bend = random.Random(original["route_seed"]).uniform(-0.000035, 0.000035)
    yaw = math.atan(2 * bend * station)
    origin = Vector((station, bend * station * station, FIT["frame_origin_world_z"]))
    return origin, Matrix.Rotation(yaw, 3, "Z"), yaw


def object_box(root, origin, orient, yaw):
    inverse = root.matrix_world.inverted()
    local = np.array(
        [
            tuple(inverse @ child.matrix_world @ Vector(v))
            for child in root.children
            if child.type == "MESH"
            for v in child.bound_box
        ]
    )
    lo, hi = local.min(axis=0), local.max(axis=0)
    center = orient.transposed() @ (root.matrix_world @ Vector((lo + hi) / 2) - origin)
    angle = root.rotation_euler.z - yaw
    return (
        list(center) + [0, 0, math.sin(angle / 2), math.cos(angle / 2)] + list(hi - lo)
    )


def bvh(scene):
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
    return BVHTree.FromPolygons(vertices, polygons), ids


def capture(scene, records, original, variant, rays, placement=None):
    destination = OUT / variant
    sample = original["sample_id"]
    meta_path = destination / "metadata" / (sample + ".json")
    if meta_path.exists():
        assert json.loads(meta_path.read_text())["generator_sha256"] == GENERATOR_SHA256
        print("CALIBRATED_ALREADY_DONE", variant, sample, flush=True)
        return
    for folder in ["metadata", "points", "scenes"]:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    origin, orient, yaw = frame(original)
    transform = np.array(orient)
    tree, face_ids = bvh(scene)
    points, identifiers, sensor_ids = [], [], []
    cosine, sine = math.cos(yaw), math.sin(yaw)
    for sensor, directions in enumerate(rays["directions"]):
        ray_origin = origin + orient @ Vector(MOUNTS[sensor])
        world_directions = directions @ transform.T
        for direction in world_directions:
            hit, normal, face_index, distance = tree.ray_cast(
                ray_origin, Vector(direction), 180
            )
            if hit is None or distance < 0.5:
                continue
            x, y, z = hit.x - origin.x, hit.y - origin.y, hit.z - origin.z
            px, py = cosine * x + sine * y, -sine * x + cosine * y
            if 0 <= px < 160 and abs(py) < 40 and -3 <= z <= 8:
                points.append((px, py, z))
                identifiers.append(face_ids[face_index])
                sensor_ids.append(sensor)
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    ids = np.asarray(identifiers, dtype=np.int16)
    sensors = np.asarray(sensor_ids, dtype=np.int8)
    rng = np.random.default_rng(original["sample_seed"] + 30231)
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    bins = np.clip(np.searchsorted(EDGES, radius, side="right") - 1, 0, NBINS - 1)
    kept = []
    achieved = np.zeros((6, NBINS), dtype=int)
    for sensor in range(6):
        for b in range(NBINS):
            candidates = np.flatnonzero((sensors == sensor) & (bins == b))
            count = min(len(candidates), rays["quotas"][sensor, b])
            if count:
                kept.append(rng.choice(candidates, count, replace=False))
            achieved[sensor, b] = count
    indices = np.concatenate(kept) if kept else np.array([], dtype=int)
    xyz, ids, sensors = xyz[indices], ids[indices], sensors[indices]
    xyz += rng.normal(0, 0.015, xyz.shape).astype(np.float32)
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= 0)
        & (xyz[:, 0] < 160)
        & (np.abs(xyz[:, 1]) < 40)
        & (xyz[:, 2] >= -3)
        & (xyz[:, 2] <= 8)
    )
    xyz, ids, sensors = xyz[valid], ids[valid], sensors[valid]
    objects = []
    for record in records:
        root = bpy.data.objects[record["root"]]
        objects.append(
            {
                "mask_id": record["mask_id"],
                "instance_id": record["instance_id"],
                "family": record["family"],
                "box": object_box(root, origin, orient, yaw),
            }
        )
    np.savez_compressed(
        destination / "points" / (sample + ".npz"),
        xyz=xyz,
        instance_id_ground_truth_only=ids,
        sensor_index_ground_truth_only=sensors,
    )
    lidar_world = np.eye(4)
    lidar_world[:3, :3] = transform
    lidar_world[:3, 3] = origin
    metadata = {
        "sample_id": sample,
        "split": "train",
        "route_id": original["route_id"],
        "variant": variant,
        "generator_sha256": GENERATOR_SHA256,
        "objects": objects,
        "point_count": len(xyz),
        "rgb_scene_match_verified": variant == "sensor_only",
        "rgb_metadata_sha256": hashlib.sha256(
            (BASE / "synthetic_v2" / "metadata" / (sample + ".json")).read_bytes()
        ).hexdigest(),
        "lidar_to_world_matrix": lidar_world.tolist(),
        "lidar_to_blender_camera_matrix": (
            np.linalg.inv(original["camera_matrix_world"]) @ lidar_world
        ).tolist(),
        "camera_matrix_world": original["camera_matrix_world"],
        "ray_plan_sha256": rays["sha256"],
        "ray_count": sum(len(row) for row in rays["directions"]),
        "desired_roi_point_count": rays["desired"],
        "quota_sensor_range": rays["quotas"].tolist(),
        "achieved_sensor_range_before_noise": achieved.tolist(),
        "profile_index": rays["profile_index"],
        "placement": placement,
        "calibration_sha256": hashlib.sha256(
            (OUT / "calibration.json").read_bytes()
        ).hexdigest(),
        "limitations": FIT["limitations"],
        "seconds": time.monotonic() - started,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    if sample.endswith("_0000"):
        bpy.ops.wm.save_as_mainfile(
            filepath=str(destination / "scenes" / (sample + ".blend"))
        )
    print(
        "CALIBRATED_SAMPLE",
        variant,
        sample,
        len(xyz),
        "target",
        rays["desired"],
        "seconds",
        round(metadata["seconds"], 2),
        flush=True,
    )


def aabb_vertices(points):
    points = np.asarray(points)
    return np.r_[points.min(axis=0), points.max(axis=0)]


def overlaps(box, others, margin=0.02):
    if not len(others):
        return False
    others = np.asarray(others)
    return bool(
        np.any(
            np.all(box[:3] < others[:, 3:] - margin, axis=1)
            & np.all(box[3:] > others[:, :3] + margin, axis=1)
        )
    )


def reposition(scene, records, original):
    origin, orient, yaw = frame(original)
    inverse = orient.transposed()
    rng = np.random.default_rng(original["sample_seed"] + 41231)
    saved = {
        record["root"]: bpy.data.objects[record["root"]].matrix_world.copy()
        for record in records
    }
    static = []
    prefixes = (
        "Railcar",
        "Background building",
        "Platform",
        "Canopy",
        "Bridge pier",
        "Overpass",
        "Foreground low wall",
        "Catenary support",
    )
    for obj in scene.objects:
        if obj.type == "MESH" and obj.pass_index == 0 and obj.name.startswith(prefixes):
            static.append(
                aabb_vertices(
                    [
                        tuple(inverse @ (obj.matrix_world @ Vector(v) - origin))
                        for v in obj.bound_box
                    ]
                )
            )
    occupied, roads, road_solids, placement = [], [], [], []
    ground = -0.25 - origin.z

    def candidate(length, width, height, is_vehicle):
        for attempt in range(200):
            source = FIT["placement_records"][
                int(rng.integers(len(FIT["placement_records"])))
            ]
            x = source["x"] + rng.normal(0, 4)
            y = source["y"] + rng.normal(0, 0.6)
            angle = source["yaw"] + rng.normal(0, math.radians(5))
            bottom = max(ground + 0.03, source["bottom_z"] + rng.normal(0, 0.03))
            if abs(y) < 2.5 or not (6 < x < 150 and abs(y) < 35):
                continue
            r = np.array(
                [
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            size = np.array([length, width, height])
            center = np.array([x, y, bottom + height / 2])
            corners = np.array(
                [
                    [xx, yy, zz]
                    for xx in [-0.5, 0.5]
                    for yy in [-0.5, 0.5]
                    for zz in [-0.5, 0.5]
                ]
            )
            box = aabb_vertices((corners * size) @ r.T + center)
            road_center = np.array([x, y, (ground + bottom) / 2])
            road_size = np.array(
                [max(length + 4, 12), max(width + 2, 4.5), max(bottom - ground, 0.04)]
            )
            road_box = aabb_vertices((corners * road_size) @ r.T + road_center)
            if box[0] < 0 or box[3] >= 160 or box[1] <= -40 or box[4] >= 40:
                continue
            if is_vehicle and (
                overlaps(box, static)
                or overlaps(box, occupied, margin=-0.3)
                or overlaps(box, road_solids, margin=0.04)
            ):
                continue
            if overlaps(road_box, static) or overlaps(road_box, occupied):
                continue
            return (
                source,
                angle,
                bottom,
                center,
                box,
                road_center,
                road_size,
                attempt + 1,
            )
        return None

    def add_road(angle, center, size):
        r = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        corners = np.array(
            [[x, y, z] for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]]
        )
        road_solids.append(aabb_vertices((corners * size) @ r.T + center))
        world = origin + orient @ Vector(center)
        road = geometry.box("Calibration service-road support", world, size, "concrete")
        road.rotation_euler.z = yaw + angle
        top = Vector(center)
        top.z += size[2] / 2 + 0.012
        asphalt = geometry.box(
            "Calibration service-road surface",
            origin + orient @ top,
            (size[0], size[1], 0.024),
            "asphalt",
        )
        asphalt.rotation_euler.z = yaw + angle
        roads.extend([road, asphalt])

    for record in records:
        root = bpy.data.objects[record["root"]]
        local = np.array(
            [
                tuple(root.matrix_world.inverted() @ child.matrix_world @ Vector(v))
                for child in root.children
                if child.type == "MESH"
                for v in child.bound_box
            ]
        )
        lo, hi = local.min(axis=0), local.max(axis=0)
        size = hi - lo
        sampled = candidate(*size, True)
        if sampled is None:
            placement.append(
                {"mask_id": record["mask_id"], "fallback_original_pose": True}
            )
            current = object_box(root, origin, orient, yaw)
            from lidar_common import corners

            occupied.append(aabb_vertices(corners(current)))
            continue
        source, angle, bottom, center, box, road_center, road_size, attempts = sampled
        r = Matrix.Rotation(angle, 3, "Z")
        root_center = Vector(center) - r @ Vector((lo + hi) / 2)
        root.location = origin + orient @ root_center
        root.rotation_euler.z = yaw + angle
        occupied.append(box)
        add_road(angle, road_center, road_size)
        placement.append(
            {
                "mask_id": record["mask_id"],
                "source_training_sample": source["sample_id"],
                "source_training_instance": source["instance_id"],
                "target_center": center.tolist(),
                "target_yaw": angle,
                "target_bottom_z": bottom,
                "attempts": attempts,
                "fallback_original_pose": False,
            }
        )
    # Empty service-road patches also occur in negative frames; road presence is not a target label.
    for _ in range(max(0, 8 - len(roads) // 2)):
        sampled = candidate(8, 2, 2, False)
        if sampled is not None:
            _, angle, _, _, _, center, size, _ = sampled
            add_road(angle, center, size)
    bpy.context.view_layer.update()
    return saved, roads, placement


def callback(scene, camera, records, sample, split, route):
    assert split == "train" and route < 8
    original = json.loads(
        (BASE / "synthetic_v2" / "metadata" / (sample + ".json")).read_text()
    )
    assert len(records) == len(original["objects"])
    for record, saved in zip(records, original["objects"]):
        for key in [
            "instance_id",
            "family",
            "dimensions_m",
            "location_world",
            "heading_rad",
        ]:
            assert record[key] == saved[key]
    rays = plan(original["sample_seed"])
    capture(scene, records, original, "sensor_only", rays)
    saved, roads, placement = reposition(scene, records, original)
    try:
        capture(scene, records, original, "sensor_pose", rays, placement)
    finally:
        for name, matrix in saved.items():
            bpy.data.objects[name].matrix_world = matrix
        for obj in roads:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.context.view_layer.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--stop", type=int, default=40)
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    for route in args.routes:
        assert 0 <= route < 8
        geometry.MATERIALS.clear()
        geometry.CUBES.clear()
        geometry.render_route(route, 40, 0, args.stop, capture=callback)
    print("CALIBRATED_CAPTURE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
