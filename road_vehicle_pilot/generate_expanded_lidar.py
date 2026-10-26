"""Broaden geometric scene coverage using procedural rays and pose priors.

Reuses mesh/collision/capture utilities, but replaces empirical ray templates
and real placement records. This is retrospective development, not blind data.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import generate_calibrated_lidar as engine
import generate_vehicle_v2 as geometry

OUT = BASE / "lidar_grouped_v3"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
engine.OUT = OUT
engine.GENERATOR_SHA256 = SCRIPT_SHA
engine.FIT = {
    "frame_origin_world_z": 0.0,
    "placement_records": [],
    "limitations": "Procedural six-fan directions and assumed mounts; no real intensity/material response/timing. Broad pose and dropout ranges were designed after prior project diagnostics, so this is retrospective development. Eight existing mesh families and route backgrounds reused; no matching RGB renders.",
}
for x in [15, 45, 85, 125]:
    for y in [-30, -22, -14, -6, 6, 14, 22, 30]:
        for yaw in np.linspace(-math.pi, math.pi, 8, endpoint=False):
            for bottom in [0.0, 0.5, 1.5, 3.0]:
                engine.FIT["placement_records"].append(
                    {
                        "x": x,
                        "y": y,
                        "yaw": float(yaw),
                        "bottom_z": bottom,
                        "sample_id": "procedural_full_roi_grid",
                        "instance_id": "procedural",
                    }
                )


def rays(seed):
    rng = np.random.default_rng(seed + 51251)
    directions = []
    levels = int(rng.choice([64, 96, 128]))
    az, el = np.meshgrid(
        np.deg2rad(np.arange(-85, 85, 0.1)), np.deg2rad(np.linspace(-25, 15, levels))
    )
    az = az.ravel() + rng.normal(0, math.radians(0.025), az.size)
    el = el.ravel() + rng.normal(0, math.radians(0.025), el.size)
    directions.append(
        np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    )
    for yaw in [-15, 0, 15, -70, 70]:
        side = abs(yaw) > 20
        n = 30000 if side else 40000
        az = np.deg2rad(
            rng.uniform(yaw - (90 if side else 7.25), yaw + (90 if side else 7.25), n)
        )
        el = np.deg2rad(rng.uniform(-70 if side else -9.1, 15 if side else 7.1, n))
        directions.append(
            np.stack(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1
            )
        )
    fingerprint = hashlib.sha256()
    for d in directions:
        fingerprint.update(d.tobytes())
    return {
        "directions": directions,
        "quotas": np.full((6, engine.NBINS), 10**8, dtype=np.int64),
        "profile_index": -1,
        "desired": None,
        "sha256": fingerprint.hexdigest(),
        "vertical_levels": levels,
    }


def callback(scene, camera, records, sample, split, route):
    assert split == "train" and route < 8
    path = OUT / "expanded/metadata" / (sample + ".json")
    if path.exists():
        meta = json.loads(path.read_text())
        assert (
            meta["generator_sha256"] == SCRIPT_SHA
            and meta["expanded_postprocess_complete"]
        )
        print("EXPANDED_ALREADY_DONE", sample, flush=True)
        return
    original = json.loads(
        (BASE / "synthetic_v2/metadata" / (sample + ".json")).read_text()
    )
    plan = rays(original["sample_seed"])
    saved, roads, placements = engine.reposition(scene, records, original)
    try:
        engine.capture(scene, records, original, "expanded", plan, placements)
        point_path = OUT / "expanded/points" / (sample + ".npz")
        with np.load(point_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        xyz = arrays["xyz"]
        rng = np.random.default_rng(original["sample_seed"] + 62551)
        retention = float(rng.uniform(0.45, 1.0))
        exponent = float(rng.uniform(0, 0.65))
        radial = np.linalg.norm(xyz[:, :2], axis=1)
        probability = retention * np.clip(
            (45 / np.maximum(radial, 45)) ** exponent, 0.25, 1
        )
        chosen = rng.random(len(xyz)) < probability
        arrays = {key: val[chosen] for key, val in arrays.items()}
        xyz = arrays["xyz"]
        sigma = 0.003 + np.linalg.norm(xyz[:, :2], axis=1) * 0.00008
        xyz += rng.normal(size=xyz.shape).astype(np.float32) * sigma[:, None]
        valid = (
            (xyz[:, 0] >= 0)
            & (xyz[:, 0] < 160)
            & (np.abs(xyz[:, 1]) < 40)
            & (xyz[:, 2] >= -3)
            & (xyz[:, 2] <= 8)
        )
        arrays = {key: val[valid] for key, val in arrays.items()}
        np.savez_compressed(point_path, **arrays)
        meta = json.loads(path.read_text())
        for placed in meta["placement"]:
            if not placed["fallback_original_pose"]:
                placed["procedural_prior"] = placed.pop("source_training_sample")
                placed.pop("source_training_instance")
        meta.update(
            point_count=len(arrays["xyz"]),
            expanded_postprocess_complete=True,
            ray_design="Procedural scan fans, no real direction templates",
            vertical_levels=plan["vertical_levels"],
            retention_probability_near=retention,
            range_dropout_exponent=exponent,
            noise_extra_sigma_m="0.003 + radial_range*0.00008",
            real_placement_records_used=False,
            empirical_return_templates_used=False,
        )
        path.write_text(json.dumps(meta, indent=2))
        print(
            "EXPANDED_CAPTURE",
            sample,
            meta["point_count"],
            "levels",
            plan["vertical_levels"],
            flush=True,
        )
    finally:
        for name, matrix in saved.items():
            bpy.data.objects[name].matrix_world = matrix
        for road in roads:
            bpy.data.objects.remove(road, do_unlink=True)
        bpy.context.view_layer.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--stop", type=int, default=40)
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    spec = {
        "generator_sha256": SCRIPT_SHA,
        "sensor_mounts_m": engine.MOUNTS.tolist(),
        "placement_grid_x_m": [15, 45, 85, 125],
        "placement_grid_y_m": [-30, -22, -14, -6, 6, 14, 22, 30],
        "bottom_z_m": [0, 0.5, 1.5, 3],
        "headings": "8 uniformly spaced headings, plus 5-degree jitter",
        "range_x_jitter_sigma_m": 4,
        "lateral_jitter_sigma_m": 0.6,
        "vertical_beam_counts": [64, 96, 128],
        "real_sensor_templates_used": False,
        "real_placement_records_used": False,
        "limit": engine.FIT["limitations"],
    }
    OUT.mkdir(exist_ok=True)
    config = OUT / "calibration.json"
    if config.exists():
        assert json.loads(config.read_text()) == spec
    else:
        config.write_text(json.dumps(spec, indent=2))
    for route in args.routes:
        assert 0 <= route < 8
        geometry.MATERIALS.clear()
        geometry.CUBES.clear()
        geometry.render_route(route, 40, 0, args.stop, capture=callback)
    print("EXPANDED_GENERATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
