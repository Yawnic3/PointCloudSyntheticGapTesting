"""Render an ordered onboard RGB sequence and paired geometric LiDAR scans."""

import bpy, math, json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view

OUT = Path(__file__).parent / "moving_dataset"
bpy.ops.wm.open_mainfile(filepath=str(OUT / "moving_railway.blend"))
s = bpy.context.scene
manifest = json.loads((OUT / "manifest.json").read_text())
classes = list(manifest["classes"].values())
capture = OUT / "captures"
capture.mkdir(exist_ok=True)
for d in ["rgb", "lidar", "annotations"]:
    (capture / d).mkdir(exist_ok=True)
roots = {int(o["instance_id"]): o for o in s.objects if "class_id" in o}


def owner(o):
    while o:
        if "instance_id" in o:
            return o
        o = o.parent
    return None


def ego(o):
    while o:
        if o.get("ego_vehicle"):
            return True
        o = o.parent
    return False


static = []
dynamic = []
for o in s.objects:
    if o.type == "MESH":
        (dynamic if ego(o) else static).append(o)
print("BUILDING BVH", len(static), len(dynamic), flush=True)


def build_bvh(objects):
    vs = []
    fs = []
    ids = []
    for o in objects:
        # Mesh modifiers only soften manufactured edges; use evaluated geometry for alignment.
        ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        start = len(vs)
        vs.extend(tuple(o.matrix_world @ v.co) for v in me.vertices)
        own = owner(o)
        pair = (
            (int(own["instance_id"]), int(own["class_id"]))
            if own and not ego(o)
            else (0, -1)
        )
        me.calc_loop_triangles()
        for tri in me.loop_triangles:
            fs.append(tuple(start + i for i in tri.vertices))
            ids.append(pair)
        ev.to_mesh_clear()
    tree = BVHTree.FromPolygons(vs, fs, all_triangles=True)
    print("BVH triangles", len(fs), flush=True)
    return tree, ids


s.frame_set(1)
bpy.context.view_layer.update()
tree, ids = build_bvh(static)
# Cached world-space boxes remain valid for stationary targets.
corners = defaultdict(list)
for o in static:
    r = owner(o)
    if r:
        corners[int(r["instance_id"])].extend(
            o.matrix_world @ Vector(v) for v in o.bound_box
        )
boxes = {
    i: (
        [min(p[j] for p in ps) for j in range(3)],
        [max(p[j] for p in ps) for j in range(3)],
    )
    for i, ps in corners.items()
}
camera = bpy.data.objects["RGB_CENTER"]
lidar = bpy.data.objects["LIDAR_ORIGIN"]
s.camera = camera
s.cycles.samples = 8
s.render.use_persistent_data = True
s.cycles.volume_bounces = 1
s.cycles.volume_step_rate = 4
observed = defaultdict(set)
records = []
az = np.linspace(-math.pi / 2, math.pi / 2, 1536)
el = np.radians(np.linspace(-30, 12, 64))
directions = [
    Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    for e in el
    for a in az
]
# 2 m/s capture displacement per frame is unnecessary for this static-scene comparison:
# take 63 paired viewpoints spaced 20m along the continuously animated journey.
for idx, t in enumerate(range(0, 1260, 20)):
    frame = round(t / 12 * 24) + 1
    s.frame_set(frame)
    bpy.context.view_layer.update()
    origin = lidar.matrix_world.translation.copy()
    rot = lidar.matrix_world.to_3x3()
    inv = rot.inverted()
    moving, moving_ids = build_bvh(dynamic)
    points = []
    counts = Counter()
    for local in directions:
        d = rot @ local
        hit, n, fi, distance = tree.ray_cast(origin, d, 200)
        mh, mn, mi, md = moving.ray_cast(origin, d, 200)
        if mh is not None and (hit is None or md < distance):
            hit, fi, distance = mh, mi, md
            pair = moving_ids[mi]
        elif hit is not None:
            pair = ids[fi]
        else:
            continue
        if distance < 0.3:
            continue
        inst, cl = pair
        p = inv @ (hit - origin)
        points.append((*p, inst, cl))
        counts[inst] += 1
        if cl >= 0:
            observed[cl].add(inst)
    arr = np.asarray(points, dtype=np.float32)
    np.savez_compressed(
        capture / "lidar" / f"{idx:04}.npz",
        xyz=arr[:, :3],
        instance_id=arr[:, 3].astype(np.int32),
        class_id=arr[:, 4].astype(np.int16),
    )
    if idx in [0, 5, 15, 25, 35, 45, 55, 62]:
        with (capture / "lidar" / f"{idx:04}.ply").open("w") as f:
            f.write(
                f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty int instance_id\nproperty int class_id\nend_header\n"
            )
            for x, y, z, i, c in points:
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {i} {c}\n")
    annotations = []
    for i, (lo, hi) in boxes.items():
        center = Vector([(lo[j] + hi[j]) / 2 for j in range(3)])
        if (center - origin).length > 220:
            continue
        uv = [
            world_to_camera_view(s, camera, Vector((x, y, z)))
            for x in [lo[0], hi[0]]
            for y in [lo[1], hi[1]]
            for z in [lo[2], hi[2]]
        ]
        if min(p.z for p in uv) <= 0:
            continue
        x0 = max(0, min(p.x for p in uv))
        x1 = min(1, max(p.x for p in uv))
        y0 = max(0, 1 - max(p.y for p in uv))
        y1 = min(1, 1 - min(p.y for p in uv))
        if x1 > x0 and y1 > y0:
            annotations.append(
                {
                    "instance_id": i,
                    "class_id": int(roots[i]["class_id"]),
                    "class_name": roots[i]["class_name"],
                    "bbox_xyxy_normalized": [x0, y0, x1, y1],
                    "world_aabb": [lo, hi],
                    "lidar_returns": counts[i],
                }
            )
    record = {
        "capture_id": idx,
        "animation_frame": frame,
        "time_s": (frame - 1) / 24,
        "route_m": t,
        "lidar_origin_world": list(origin),
        "T_world_from_lidar": [list(row) for row in lidar.matrix_world],
        "T_world_from_camera": [list(row) for row in camera.matrix_world],
        "K": [[26 / 36 * 1280, 0, 640], [0, 26 / 36 * 1280, 360], [0, 0, 1]],
        "boxes_policy": "Projected amodal world AABBs, clipped to image; includes occluded boxes. Not OSDaR23 annotation format.",
        "boxes": annotations,
    }
    (capture / "annotations" / f"{idx:04}.json").write_text(json.dumps(record))
    s.render.filepath = str(capture / "rgb" / f"{idx:04}.png")
    bpy.ops.render.render(write_still=True)
    records.append({"id": idx, "frame": frame, "route_m": t, "points": len(points)})
    (capture / "progress.json").write_text(
        json.dumps({"completed": len(records), "total": 63})
    )
    print("CAPTURE", idx, "points", len(points), "classes", len(observed), flush=True)
report = {
    "frames": records,
    "unique_observed_instances_per_class": {
        classes[c]: len(observed[c]) for c in range(len(classes))
    },
    "sampling": "63 synchronized poses at 20m intervals; 12m/s train, 24fps animation. RGB and LiDAR instantaneous at each pose.",
    "lidar": "64x1536, azimuth +/-90deg, elevation -30..12deg, nearest hit 0.3..200m; ego self-returns included as background. No sensor physics, noise or motion distortion. Smoke returns use mesh proxy.",
}
(capture / "validation.json").write_text(json.dumps(report, indent=2))
print("CAPTURES COMPLETE", report["unique_observed_instances_per_class"], flush=True)
