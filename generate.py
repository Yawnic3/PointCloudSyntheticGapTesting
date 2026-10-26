"""Procedural OSDaR23-inspired scenes. Run with Blender 4.2 --background --python."""

import bpy, math, json, random
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view

OUT = Path(__file__).parent / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = [
    "person",
    "signal",
    "catenary_pole",
    "track",
    "signal_pole",
    "road_vehicle",
    "train",
    "buffer_stop",
    "animal",
    "switch",
    "bicycle",
    "crowd",
    "group_of_bicycles",
    "transition",
    "flame",
    "signal_bridge",
    "smoke",
    "wagons",
    "drag_shoe",
    "motorcycle",
    "wheelchair",
    "group_of_animals",
]
(OUT / "classes.json").write_text(
    json.dumps(
        {
            "classes": dict(enumerate(CLASSES)),
            "note": "Local zero-based IDs; not official dataset IDs. Last two are specification extensions absent from the paper count table.",
        },
        indent=2,
    )
)
random.seed(23)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene
s.unit_settings.system = "METRIC"
s.render.resolution_x = 1232
s.render.resolution_y = 800
s.render.resolution_percentage = 100
try:
    s.render.engine = "CYCLES"
except TypeError:
    pass
s.cycles.samples = 12
s.render.image_settings.file_format = "PNG"
s.world = bpy.data.worlds.new("Daylight")
s.world.use_nodes = True
s.world.node_tree.nodes.get("Background").inputs[0].default_value = (0.48, 0.62, 0.8, 1)
s.world.node_tree.nodes.get("Background").inputs[1].default_value = 0.5
bpy.ops.object.light_add(type="SUN", location=(0, 0, 30))
bpy.context.object.rotation_euler = (0.45, -0.55, -0.3)
bpy.context.object.data.energy = 3


def mat(n, c, metal=0):
    m = bpy.data.materials.new(n)
    m.diffuse_color = (*c, 1)
    m.use_nodes = True
    p = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    p.inputs["Base Color"].default_value = (*c, 1)
    p.inputs["Metallic"].default_value = metal
    p.inputs["Roughness"].default_value = 0.72
    return m


M = {
    n: mat(n, c, m)
    for n, c, m in [
        ("steel", (0.36, 0.39, 0.4), 0.7),
        ("dark", (0.035, 0.04, 0.045), 0.1),
        ("red", (0.55, 0.025, 0.025), 0.15),
        ("white", (0.78, 0.8, 0.76), 0.1),
        ("glass", (0.025, 0.095, 0.14), 0.5),
        ("wood", (0.18, 0.12, 0.08), 0),
        ("ballast", (0.25, 0.25, 0.23), 0),
        ("grass", (0.17, 0.23, 0.085), 0),
        ("skin", (0.57, 0.35, 0.22), 0),
        ("orange", (0.95, 0.21, 0.015), 0),
        ("blue", (0.025, 0.12, 0.32), 0),
        ("leaf", (0.12, 0.24, 0.055), 0),
        ("smoke", (0.23, 0.25, 0.27), 0),
    ]
}
roots = []
current = None


def root(cls, loc):
    global current
    o = bpy.data.objects.new(f"{cls}_{len(roots):04}", None)
    s.collection.objects.link(o)
    o.location = loc
    o["class_name"] = cls
    o["class_id"] = CLASSES.index(cls)
    o["instance_id"] = len(roots) + 1
    roots.append(o)
    current = o
    return o


def finish(o, n, loc, scale, material):
    o.name = n
    o.location = loc
    o.scale = scale
    o.data.materials.append(M[material])
    o.parent = current
    if current:
        o.pass_index = current["instance_id"]
    return o


def box(loc, size, material="steel"):
    mesh = bpy.data.meshes.new("box")
    mesh.from_pydata(
        [(x, y, z) for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]],
        [],
        [
            (0, 4, 6, 2),
            (1, 3, 7, 5),
            (0, 1, 5, 4),
            (2, 6, 7, 3),
            (0, 2, 3, 1),
            (4, 5, 7, 6),
        ],
    )
    mesh.update()
    o = bpy.data.objects.new("part", mesh)
    s.collection.objects.link(o)
    return finish(o, "part", loc, tuple(v / 2 for v in size), material)


def ball(loc, size, material="skin"):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8)
    o = finish(bpy.context.object, "rounded", loc, size, material)
    for p in o.data.polygons:
        p.use_smooth = True
    return o


def rod(a, b, r=0.04, material="steel"):
    a, b = Vector(a), Vector(b)
    d = b - a
    n = 10
    mesh = bpy.data.meshes.new("rod")
    mesh.from_pydata(
        [
            (r * math.cos(i * 2 * math.pi / n), r * math.sin(i * 2 * math.pi / n), z)
            for z in [-d.length / 2, d.length / 2]
            for i in range(n)
        ],
        [],
        [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
        + [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)],
    )
    mesh.update()
    o = bpy.data.objects.new("tube", mesh)
    s.collection.objects.link(o)
    finish(o, "tube", (a + b) / 2, (1, 1, 1), material)
    o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    return o


def wheel(x, y, z, r=0.33):
    bpy.ops.mesh.primitive_torus_add(
        major_segments=20, minor_segments=8, major_radius=r, minor_radius=0.045
    )
    o = finish(bpy.context.object, "wheel", (x, y, z), (1, 1, 1), "dark")
    o.rotation_euler.x = math.pi / 2
    rod((x, y - 0.04, z), (x, y + 0.04, z), 0.07)
    for a in range(0, 360, 60):
        t = math.radians(a)
        rod((x, y, z), (x + r * math.cos(t), y, z + r * math.sin(t)), 0.009)


def human(dx=0, dy=0):
    ball((dx, dy, 1.59), (0.12, 0.12, 0.16))
    box(
        (dx, dy, 1.15), (0.38, 0.24, 0.6), "orange" if random.random() < 0.4 else "blue"
    )
    for side in [-1, 1]:
        rod(
            (dx + side * 0.11, dy, 0.9),
            (dx + side * 0.14, dy + 0.05, 0.13),
            0.075,
            "dark",
        )
        rod(
            (dx + side * 0.23, dy, 1.4),
            (dx + side * 0.29, dy + 0.03, 0.87),
            0.055,
            "blue",
        )


def bike(dx=0, dy=0, motor=False):
    for x in [-0.6, 0.6]:
        wheel(dx + x, dy, 0.37, 0.32)
    for a, b in [
        ((-0.6, 0, 0.37), (0, 0, 0.8)),
        ((0, 0, 0.8), (0.6, 0, 0.37)),
        ((-0.6, 0, 0.37), (0.1, 0, 0.37)),
        ((0.1, 0, 0.37), (0, 0, 0.8)),
        ((0.1, 0, 0.37), (0.6, 0, 0.37)),
    ]:
        rod(
            tuple(a[i] + (dx, dy, 0)[i] for i in range(3)),
            tuple(b[i] + (dx, dy, 0)[i] for i in range(3)),
            0.035,
            "red",
        )
    box((dx - 0.05, dy, 0.94), (0.3, 0.22, 0.08), "dark")
    rod((dx + 0.45, dy, 0.5), (dx + 0.4, dy, 1.15), 0.04)
    rod((dx + 0.4, dy - 0.25, 1.15), (dx + 0.4, dy + 0.25, 1.15), 0.035)
    if motor:
        ball((dx, dy, 0.65), (0.37, 0.22, 0.25), "red")
        box((dx - 0.15, dy, 0.45), (0.5, 0.35, 0.35), "steel")


def animal(dx=0, dy=0):
    ball((dx, dy, 0.62), (0.47, 0.19, 0.23), "wood")
    ball((dx + 0.48, dy, 0.84), (0.2, 0.15, 0.19), "wood")
    for x in [-0.3, 0.3]:
        for y in [-0.13, 0.13]:
            rod((dx + x, dy + y, 0.55), (dx + x, dy + y, 0.05), 0.045, "wood")
    rod((dx - 0.4, dy, 0.66), (dx - 0.65, dy, 0.85), 0.035, "wood")


def signal_head(x, y, z):
    box((x, y, z), (0.25, 0.55, 1.3), "dark")
    for i in range(3):
        ball(
            (x - 0.14, y, z - 0.42 + i * 0.42),
            (0.045, 0.13, 0.13),
            "red" if i == 2 else "dark",
        )


def make(cls, loc):
    r = root(cls, loc)
    if cls == "person":
        human()
    elif cls == "crowd":
        for x, y in [(-0.6, 0), (0, 0.25), (0.6, -0.1), (1, 0.4)]:
            human(x, y)
    elif cls in ["bicycle", "motorcycle", "group_of_bicycles"]:
        for y in ([0, 0.65, 1.3] if cls == "group_of_bicycles" else [0]):
            bike(0, y, cls == "motorcycle")
    elif cls in ["animal", "group_of_animals"]:
        for y in ([0, 0.65, 1.3] if cls == "group_of_animals" else [0]):
            animal(0, y)
    elif cls == "wheelchair":
        for y in [-0.32, 0.32]:
            wheel(0, y, 0.36, 0.32)
            wheel(0.5, y, 0.13, 0.09)
        box((0, 0, 0.57), (0.52, 0.6, 0.1), "blue")
        box((-0.25, 0, 0.86), (0.08, 0.6, 0.65), "blue")
        rod((-0.3, -0.3, 0.3), (0.5, -0.3, 0.2))
        rod((-0.3, 0.3, 0.3), (0.5, 0.3, 0.2))
    elif cls in ["train", "wagons", "road_vehicle"]:
        l, w, h = (
            (12, 2.85, 3.5)
            if cls == "train"
            else ((9, 2.7, 2.7) if cls == "wagons" else (4.2, 1.8, 1.5))
        )
        box((0, 0, h / 2 + 0.4), (l, w, h), "red" if cls == "train" else "blue")
        box((0, 0, h + 0.45), (l, w, 0.15), "white")
        if cls != "wagons":
            box((-l / 2 - 0.015, 0, h * 0.75), (0.035, w * 0.8, h * 0.28), "glass")
            for x in range(int(-l / 2 + 1), int(l / 2), 2):
                for side in [-1, 1]:
                    box(
                        (x, side * (w / 2 + 0.015), h * 0.8),
                        (1, 0.035, h * 0.25),
                        "glass",
                    )
        for x in [-l * 0.32, l * 0.32]:
            for y in [-w * 0.42, w * 0.42]:
                wheel(x, y, 0.45, 0.4)
    elif cls == "drag_shoe":
        box((0, 0, 0.04), (0.45, 0.14, 0.08), "orange")
        box((0.15, 0, 0.15), (0.12, 0.15, 0.25), "orange")
        rod((0.15, 0, 0.25), (0.35, 0, 0.35), 0.025, "orange")
    elif cls == "buffer_stop":
        for y in [-0.75, 0.75]:
            rod((0, y, 0.1), (-0.8, y, 1.1), 0.13)
            rod((-1.5, y, 0.1), (-0.8, y, 1.1), 0.13)
        box((-0.8, 0, 1.1), (0.3, 2.4, 0.35), "red")
        box((-0.98, 0, 1.12), (0.05, 0.4, 0.25), "white")
    elif cls in ["signal", "signal_pole", "catenary_pole", "signal_bridge"]:
        if cls == "signal":
            signal_head(0, 0, 4.4)
        elif cls == "signal_pole":
            box((0, 0, 2.1), (0.22, 0.22, 4.2))
            box((0, 0, 0.1), (0.7, 0.7, 0.2), "white")
        elif cls == "catenary_pole":
            box((0, 0, 3.7), (0.25, 0.3, 7.4))
            rod((0, 0, 6.7), (0, 3.7, 5.7), 0.06)
            rod((0, 0, 5.7), (0, 3.7, 5.7), 0.05)
            box((0, 0, 0.1), (0.8, 0.8, 0.2), "white")
        else:
            for y in [-6, 6]:
                box((0, y, 3.6), (0.4, 0.4, 7.2))
            for z in [6.4, 7.2]:
                rod((0, -6, z), (0, 6, z), 0.1)
            for y in range(-6, 6):
                rod((0, y, 6.4), (0, y + 1, 7.2), 0.055)
            signal_head(0, -2, 5.5)
            signal_head(0, 2, 5.5)
    elif cls in ["track", "switch", "transition"]:
        for y in [-0.7525, 0.7525]:
            rod((-8, y, 0.15), (8, y, 0.15), 0.055)
        for x in range(-8, 9):
            box((x, 0, 0), (0.22, 2.6, 0.16), "wood")
        if cls != "track":
            for side in [-1, 1]:
                for i in range(16):
                    rod(
                        (-8 + i, side * 0.7525 + (i / 16) ** 2 * 3, 0.16),
                        (-7 + i, side * 0.7525 + ((i + 1) / 16) ** 2 * 3, 0.16),
                        0.05,
                    )
    elif cls == "flame":
        for i in range(9):
            bpy.ops.mesh.primitive_cone_add(
                vertices=9,
                radius1=random.uniform(0.12, 0.3),
                radius2=0,
                depth=random.uniform(0.6, 1.8),
            )
            finish(
                bpy.context.object,
                "flame",
                (random.uniform(-0.4, 0.4), random.uniform(-0.3, 0.3), 0.7),
                (1, 1, 1),
                "orange",
            )
    elif cls == "smoke":
        for i in range(9):
            ball(
                (math.sin(i) * 0.25, i * 0.08, 0.5 + i * 0.4),
                (0.4 + i * 0.06, 0.35 + i * 0.07, 0.5),
                "smoke",
            )
    return r


# Three geographically separated, individually named scenarios in one editable file.
scenarios = [("station", 0), ("yard", 300), ("hazards", 600)]
for name, offset in scenarios:
    print("BUILD", name, flush=True)
    current = None
    box((65, offset, -0.4), (200, 65, 0.6), "grass")
    for lane in [-4, 0, 4]:
        current = None
        box((65, offset + lane, -0.13), (200, 3.5, 0.25), "ballast")
        for y in [-0.7525, 0.7525]:
            box((65, offset + lane + y, 0.12), (200, 0.07, 0.16), "steel")
        for x in range(-30, 165):
            box((x, offset + lane, 0), (0.22, 2.5, 0.13), "wood")
        make("track", (20, offset + lane, 0))
    for x in range(-10, 160, 25):
        make("catenary_pole", (x, offset - 7, 0))
    current = None
    for z in [5.8, 6.5]:
        rod((-30, offset, z), (165, offset, z), 0.015)
    for x in range(0, 160, 12):
        box((x, offset + 20, 2.8), (9, 8, 5.6), "white")
        box((x, offset + 15.95, 2.6), (5, 0.03, 2), "glass")
    if name == "station":
        box((50, offset + 8, 0.4), (125, 4, 0.8), "white")
        for x in range(5, 110, 14):
            rod((x, offset + 10, 0.8), (x, offset + 10, 4), 0.1)
        box((50, offset + 10, 4.1), (125, 5, 0.18), "steel")
        layout = [
            ("train", 65, -4),
            ("person", 12, 6),
            ("crowd", 27, 7),
            ("bicycle", 17, 7),
            ("group_of_bicycles", 40, 7),
            ("wheelchair", 21, 6),
            ("signal_pole", 38, -2),
            ("signal", 38, -2),
            ("signal_bridge", 90, 0),
            ("road_vehicle", 32, -11),
        ]
    elif name == "yard":
        layout = [
            ("wagons", 65, 4),
            ("train", 100, -4),
            ("buffer_stop", 28, 4),
            ("switch", 44, 0),
            ("transition", 72, 0),
            ("drag_shoe", 13, 0.75),
            ("motorcycle", 18, 7),
            ("road_vehicle", 32, 10),
            ("person", 25, -3),
            ("signal_pole", 55, -2),
            ("signal", 55, -2),
        ]
    else:
        layout = [
            ("flame", 18, -3),
            ("smoke", 19, -3),
            ("animal", 16, 2),
            ("group_of_animals", 28, 5),
            ("road_vehicle", 40, 0),
            ("person", 10, -3),
            ("bicycle", 24, -5),
            ("drag_shoe", 8, 0.75),
            ("switch", 60, 0),
        ]
    for cls, x, y in layout:
        make(
            cls,
            (
                x,
                offset + y,
                (
                    0.18
                    if cls not in ["person", "crowd", "wheelchair"]
                    else 0.8 if name == "station" else 0.18
                ),
            ),
        )


def camera(n, loc, target, lens=24):
    bpy.ops.object.camera_add(location=loc)
    o = bpy.context.object
    o.name = n
    o.rotation_euler = (Vector(target) - o.location).to_track_quat("-Z", "Y").to_euler()
    o.data.lens = lens
    o.data.clip_end = 600
    return o


cameras = []
for name, y in scenarios:
    for side, angle in [("left", 0.25), ("center", 0), ("right", -0.25)]:
        c = camera(name + "_rgb_" + side, (0, y, 3.2), (60, y + 60 * angle, 1.5))
        c["calibration"] = "Assumed; not original OSDaR23 calibration"
        cameras.append(c)
s.camera = cameras[1]
bpy.context.view_layer.update()


# Instance geometry gives world boxes, projected amodal boxes and true nearest-hit scans.
def bounds(r):
    pts = [
        o.matrix_world @ Vector(v)
        for o in r.children
        if o.type == "MESH"
        for v in o.bound_box
    ]
    return [
        [min(p[i] for p in pts) for i in range(3)],
        [max(p[i] for p in pts) for i in range(3)],
    ]


instances = [
    {
        "id": r["instance_id"],
        "class_id": r["class_id"],
        "class_name": r["class_name"],
        "world_aabb": bounds(r),
    }
    for r in roots
]
(OUT / "instances.json").write_text(json.dumps(instances, indent=2))
verts = []
faces = []
face_ids = []
for o in s.objects:
    if o.type != "MESH":
        continue
    start = len(verts)
    verts.extend(o.matrix_world @ v.co for v in o.data.vertices)
    for p in o.data.polygons:
        faces.append(tuple(start + i for i in p.vertices))
        face_ids.append(
            (o.parent["instance_id"], o.parent["class_id"])
            if o.parent and "class_id" in o.parent
            else (0, -1)
        )
bvh = BVHTree.FromPolygons(verts, faces)
summary = []
for name, y in scenarios:
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    origin = Vector((0, y, 3.2))
    points = []
    for row in range(64):
        el = math.radians(-25 + row * 35 / 63)
        for col in range(1024):
            az = math.radians(-90 + col * 180 / 1023)
            d = Vector(
                (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
            )
            hit, normal, index, dist = bvh.ray_cast(origin, d, 200)
            if hit is not None:
                inst, cl = face_ids[index]
                p = hit - origin
                points.append((*p, inst, cl))
    with (folder / "lidar.ply").open("w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex "
            + str(len(points))
            + "\nproperty float x\nproperty float y\nproperty float z\nproperty int instance_id\nproperty int class_id\nend_header\n"
        )
        for x, z, h, i, c in points:
            f.write(f"{x:.5f} {z:.5f} {h:.5f} {i} {c}\n")
    for cam in [c for c in cameras if c.name.startswith(name + "_")]:
        labels = []
        for item, r in zip(instances, roots):
            uv = [
                world_to_camera_view(s, cam, o.matrix_world @ Vector(v))
                for o in r.children
                if o.type == "MESH"
                for v in o.bound_box
            ]
            if not uv or min(p.z for p in uv) <= 0:
                continue
            x0 = max(0, min(p.x for p in uv))
            x1 = min(1, max(p.x for p in uv))
            y0 = max(0, 1 - max(p.y for p in uv))
            y1 = min(1, 1 - min(p.y for p in uv))
            if x1 > x0 and y1 > y0:
                labels.append(
                    {
                        "instance_id": item["id"],
                        "class_id": item["class_id"],
                        "bbox_xyxy_normalized": [x0, y0, x1, y1],
                        "policy": "amodal clipped projection; may be fully occluded",
                    }
                )
        (folder / (cam.name + ".json")).write_text(
            json.dumps(
                {
                    "camera": cam.name,
                    "T_world_from_camera": [list(row) for row in cam.matrix_world],
                    "K": [
                        [cam.data.lens / cam.data.sensor_width * 1232, 0, 616],
                        [0, cam.data.lens / cam.data.sensor_width * 1232, 400],
                        [0, 0, 1],
                    ],
                    "boxes": labels,
                },
                indent=2,
            )
        )
        if cam.name.endswith("center"):
            s.camera = cam
            s.render.filepath = str(folder / "rgb_center.png")
            bpy.ops.render.render(write_still=True)
    summary.append(
        {
            "scene": name,
            "lidar_points": len(points),
            "classes_in_scan": sorted({p[4] for p in points if p[4] >= 0}),
        }
    )
s.camera = cameras[1]
for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        area.spaces.active.region_3d.view_perspective = "CAMERA"
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "osdar23_scenes.blend"))
assert set(i["class_name"] for i in instances) == set(CLASSES)
(OUT / "validation.json").write_text(
    json.dumps(
        {
            "all_22_classes_present": True,
            "instance_count": len(instances),
            "captures": summary,
            "sensor_model": "Idealized 64x1024 angular nearest-hit LiDAR, +/-90deg azimuth, -25..10deg elevation, 200m range. No physical IR/radar simulation.",
            "source": "https://arxiv.org/html/2305.03001v2",
            "replica_scope": "Procedural inspired scenes, not reconstruction of recorded sequences.",
        },
        indent=2,
    )
)
print("OSDAR23_GENERATION_COMPLETE", json.dumps(summary))
