"""Render independent road-vehicle arrangements from train-mounted railway views.

Run inside Blender. RGB and visible-instance labels come from the same render.
Each route is assigned wholly to one split. No real images or test scores are used.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

BASE = Path(__file__).resolve().parent
OUT = BASE / "synthetic_v2"
FAMILIES = ["hatchback", "sedan", "estate", "suv", "pickup", "van", "box_truck", "bus"]
CONTEXTS = [
    "station",
    "crossing",
    "woodland",
    "industrial",
    "suburban",
    "bridge",
    "cutting",
    "rural",
]
MATERIALS = {}
CUBES = {}
CURRENT = None


def mesh(name, vertices, faces, material, index=0, parent=None, smooth=False):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.materials.append(MATERIALS[material])
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent or CURRENT
    obj.pass_index = index
    if smooth:
        for polygon in data.polygons:
            polygon.use_smooth = True
    return obj


def box(name, location, size, material, index=0, parent=None):
    if material not in CUBES:
        data = bpy.data.meshes.new("Unit box " + material)
        data.from_pydata(
            [(x, y, z) for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]],
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
        data.materials.append(MATERIALS[material])
        CUBES[material] = data
    obj = bpy.data.objects.new(name, CUBES[material])
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.scale = size
    obj.parent = parent or CURRENT
    obj.pass_index = index
    return obj


def cylinder(name, a, b, radius, material, index=0, parent=None, sides=16):
    a, b = Vector(a), Vector(b)
    delta = b - a
    vertices = [
        (
            radius * math.cos(i * math.tau / sides),
            radius * math.sin(i * math.tau / sides),
            z,
        )
        for z in [-delta.length / 2, delta.length / 2]
        for i in range(sides)
    ]
    faces = [tuple(reversed(range(sides))), tuple(range(sides, 2 * sides))]
    faces += [
        (i, (i + 1) % sides, (i + 1) % sides + sides, i + sides) for i in range(sides)
    ]
    obj = mesh(name, vertices, faces, material, index, parent)
    obj.location = (a + b) / 2
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    return obj


def material(name, color, metallic=0, roughness=0.6, texture=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    if metallic > 0.1:
        shader.inputs["Coat Weight"].default_value = 0.3
    if texture:
        noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = texture
        coord = mat.node_tree.nodes.new("ShaderNodeTexCoord")
        mat.node_tree.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        ramp = mat.node_tree.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*[v * 0.55 for v in color], 1)
        ramp.color_ramp.elements[1].color = (*[min(1, v * 1.35) for v in color], 1)
        mat.node_tree.links.new(noise.outputs["Fac"], ramp.inputs[0])
        mat.node_tree.links.new(ramp.outputs[0], shader.inputs["Base Color"])
        bump = mat.node_tree.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.25
        bump.inputs["Distance"].default_value = 0.025
        mat.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        mat.node_tree.links.new(bump.outputs[0], shader.inputs["Normal"])
    MATERIALS[name] = mat


def setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs["Image"], composite.inputs[0])
    mask = nodes.new("CompositorNodeOutputFile")
    mask.name = "Visible instance IDs"
    mask.base_path = str(OUT / "masks")
    mask.format.file_format = "OPEN_EXR"
    mask.format.color_depth = "32"
    mask.format.color_mode = "RGB"
    scene.node_tree.links.new(layers.outputs["IndexOB"], mask.inputs[0])
    world = bpy.data.worlds.new("Daylight")
    world.use_nodes = True
    scene.world = world
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    background = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    world.node_tree.links.new(sky.outputs[0], background.inputs[0])
    background.inputs[1].default_value = 0.15
    backend = "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ["OPTIX", "CUDA"]:
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            found = [d for d in prefs.devices if d.type == kind]
            if found:
                for device in prefs.devices:
                    device.use = device.type == kind
                scene.cycles.device = "GPU"
                backend = kind
                break
        except (TypeError, RuntimeError):
            continue
    print("RENDER BACKEND", backend, flush=True)
    colors = [
        (0.7, 0.025, 0.015),
        (0.025, 0.07, 0.2),
        (0.5, 0.52, 0.53),
        (0.8, 0.78, 0.73),
        (0.018, 0.02, 0.024),
        (0.035, 0.16, 0.08),
        (0.55, 0.31, 0.035),
        (0.18, 0.03, 0.07),
        (0.07, 0.14, 0.18),
        (0.3, 0.28, 0.22),
        (0.26, 0.25, 0.24),
        (0.48, 0.54, 0.58),
    ]
    for i, color in enumerate(colors):
        material("paint_" + str(i), color, 0.55, 0.25)
    for name, color, metallic, rough, tex in [
        ("rubber", (0.015, 0.018, 0.02), 0, 0.85, 30),
        ("glass", (0.025, 0.055, 0.075), 0.5, 0.12, 0),
        ("chrome", (0.55, 0.57, 0.6), 0.8, 0.24, 0),
        ("lamp", (0.9, 0.88, 0.7), 0.2, 0.2, 0),
        ("rear_lamp", (0.55, 0.008, 0.003), 0.2, 0.22, 0),
        ("ballast", (0.22, 0.2, 0.18), 0, 0.9, 32),
        ("concrete", (0.32, 0.31, 0.28), 0, 0.85, 18),
        ("grass", (0.08, 0.14, 0.035), 0, 0.95, 3),
        ("dry_grass", (0.19, 0.16, 0.06), 0, 0.95, 5),
        ("steel", (0.33, 0.36, 0.39), 0.7, 0.4, 0),
        ("rust", (0.16, 0.065, 0.025), 0.4, 0.7, 12),
        ("asphalt", (0.075, 0.08, 0.085), 0, 0.85, 35),
        ("brick", (0.26, 0.09, 0.045), 0, 0.85, 8),
        ("wood", (0.13, 0.07, 0.025), 0, 0.95, 16),
        ("leaf", (0.04, 0.12, 0.018), 0, 0.85, 8),
        ("white", (0.75, 0.74, 0.7), 0, 0.65, 0),
        ("yellow", (0.8, 0.5, 0.015), 0, 0.7, 0),
    ]:
        material(name, color, metallic, rough, tex)
    return scene, sky, mask, backend


def loft(name, rings, material_name, index):
    vertices = []
    for x, width, bottom, waist in rings:
        vertices += [
            (x, y * width, z)
            for y, z in [
                (-0.44, bottom),
                (0.44, bottom),
                (0.5, bottom + 0.16),
                (0.5, waist - 0.12),
                (0.43, waist),
                (-0.43, waist),
                (-0.5, waist - 0.12),
                (-0.5, bottom + 0.16),
            ]
        ]
    faces = [tuple(reversed(range(8))), tuple(range(len(vertices) - 8, len(vertices)))]
    for ring in range(len(rings) - 1):
        for i in range(8):
            faces.append(
                (
                    ring * 8 + i,
                    ring * 8 + (i + 1) % 8,
                    (ring + 1) * 8 + (i + 1) % 8,
                    (ring + 1) * 8 + i,
                )
            )
    return mesh(name, vertices, faces, material_name, index)


def vehicle(family, rng, index, location, heading, seed):
    global CURRENT
    root = bpy.data.objects.new(family + "_" + str(index), None)
    bpy.context.scene.collection.objects.link(root)
    root.location = location
    root.rotation_euler.z = heading
    CURRENT = root
    dimensions = {
        "hatchback": (3.9, 1.72, 1.5),
        "sedan": (4.6, 1.82, 1.48),
        "estate": (4.8, 1.83, 1.5),
        "suv": (4.65, 1.95, 1.82),
        "pickup": (5.25, 1.95, 1.9),
        "van": (5.2, 1.98, 2.45),
        "box_truck": (7.0, 2.35, 3.2),
        "bus": (10.5, 2.5, 3.15),
    }
    length, width, height = [v * rng.uniform(0.92, 1.08) for v in dimensions[family]]
    paint = "paint_" + str(rng.randrange(12))
    waist = min(1.1, height * 0.65)
    radius = 0.32 if height < 2 else 0.42
    if family in ["box_truck", "bus"]:
        radius = 0.48
    loft(
        "Contoured body",
        [
            (-length / 2, width * 0.88, radius * 0.55, waist * 0.9),
            (-length * 0.36, width, radius * 0.5, waist),
            (length * 0.32, width, radius * 0.5, waist),
            (length / 2, width * 0.9, radius * 0.65, waist * 0.84),
        ],
        paint,
        index,
    )
    if family in ["van", "box_truck", "bus"]:
        rear = -length * 0.45
        front = length * (0.28 if family == "van" else 0.38)
        loft(
            "Tall body",
            [
                (rear, width * 0.9, waist - 0.05, height - 0.08),
                (front, width * 0.9, waist - 0.05, height - 0.08),
            ],
            paint,
            index,
        )
        box(
            "Front windshield",
            (front + 0.015, 0, height * 0.75),
            (0.035, width * 0.78, height * 0.32),
            "glass",
            index,
        )
        if family == "box_truck":
            box(
                "Cargo box",
                (-length * 0.14, 0, height * 0.6),
                (length * 0.65, width, height * 0.7),
                "white",
                index,
            )
        for side in [-1, 1]:
            if family == "bus":
                for x in np.arange(-length * 0.36, length * 0.35, 1.15):
                    box(
                        "Passenger window",
                        (x, side * width * 0.455, height * 0.73),
                        (0.95, 0.035, height * 0.32),
                        "glass",
                        index,
                    )
            else:
                box(
                    "Cab window",
                    (length * 0.27, side * width * 0.455, height * 0.74),
                    (length * 0.18, 0.035, height * 0.3),
                    "glass",
                    index,
                )
        roof_x = (rear + front) / 2
        box(
            "Roof",
            (roof_x, 0, height),
            (front - rear, width * 0.88, 0.08),
            paint,
            index,
        )
    else:
        rear_bottom = -length * 0.42
        rear_top = -length * (0.30 if family in ["estate", "suv"] else 0.24)
        front_top = length * 0.12
        front_bottom = length * 0.32
        if family == "pickup":
            rear_bottom, rear_top = -length * 0.08, -length * 0.06
        bottom_y, top_y = width * 0.425, width * rng.uniform(0.32, 0.37)
        vertices = [
            (rear_bottom, -bottom_y, waist),
            (rear_bottom, bottom_y, waist),
            (rear_top, -top_y, height),
            (rear_top, top_y, height),
            (front_top, -top_y, height),
            (front_top, top_y, height),
            (front_bottom, -bottom_y, waist),
            (front_bottom, bottom_y, waist),
        ]
        mesh(
            "Cabin glazing",
            vertices,
            [(0, 1, 3, 2), (2, 3, 5, 4), (4, 5, 7, 6), (0, 2, 4, 6), (1, 7, 5, 3)],
            "glass",
            index,
        )
        box(
            "Roof panel",
            ((rear_top + front_top) / 2, 0, height + 0.015),
            (front_top - rear_top, top_y * 2, 0.07),
            paint,
            index,
        )
        for side in [-1, 1]:
            for a, b in [
                (
                    (rear_bottom, side * bottom_y, waist),
                    (rear_top, side * top_y, height),
                ),
                (
                    (front_bottom, side * bottom_y, waist),
                    (front_top, side * top_y, height),
                ),
                (
                    (-length * 0.06, side * bottom_y, waist),
                    (-length * 0.06, side * top_y, height),
                ),
            ]:
                cylinder("Window pillar", a, b, 0.025, paint, index, sides=8)
            box(
                "Door handle",
                (-length * 0.1, side * width * 0.502, waist * 0.88),
                (0.16, 0.025, 0.035),
                "chrome",
                index,
            )
            box(
                "Wing mirror",
                (length * 0.26, side * width * 0.57, waist + 0.12),
                (0.18, 0.16, 0.10),
                paint,
                index,
            )
        if family == "pickup":
            box(
                "Pickup bed",
                (-length * 0.3, 0, waist - 0.08),
                (length * 0.35, width * 0.8, 0.08),
                "rubber",
                index,
            )
    wheelbase = length * (0.31 if family != "bus" else 0.34)
    for x in [-wheelbase, wheelbase]:
        for side in [-1, 1]:
            y = side * width * 0.47
            cylinder(
                "Tyre",
                (x, y - 0.11, radius),
                (x, y + 0.11, radius),
                radius,
                "rubber",
                index,
                sides=24,
            )
            cylinder(
                "Alloy wheel",
                (x, y + side * 0.115, radius),
                (x, y + side * 0.13, radius),
                radius * 0.64,
                "chrome",
                index,
                sides=20,
            )
            for spoke in range(5):
                a = spoke * math.tau / 5
                cylinder(
                    "Rim spoke",
                    (x, y + side * 0.145, radius),
                    (
                        x + radius * 0.57 * math.cos(a),
                        y + side * 0.145,
                        radius + radius * 0.57 * math.sin(a),
                    ),
                    0.025,
                    "rubber",
                    index,
                    sides=6,
                )
    for side in [-1, 1]:
        box(
            "Headlight",
            (length / 2 + 0.012, side * width * 0.31, waist * 0.73),
            (0.025, width * 0.22, 0.15),
            "lamp",
            index,
        )
        box(
            "Taillight",
            (-length / 2 - 0.012, side * width * 0.34, waist * 0.75),
            (0.025, width * 0.16, 0.18),
            "rear_lamp",
            index,
        )
    box(
        "Grille",
        (length / 2 + 0.018, 0, waist * 0.47),
        (0.03, width * 0.55, 0.16),
        "rubber",
        index,
    )
    for sign in [-1, 1]:
        box(
            "Number plate",
            (sign * (length / 2 + 0.035), 0, waist * 0.59),
            (0.025, 0.43, 0.11),
            "white",
            index,
        )
        box(
            "Bumper",
            (sign * length * 0.495, 0, waist * 0.29),
            (0.09, width * 0.86, 0.09),
            "rubber",
            index,
        )
    CURRENT = None
    return {
        "mask_id": index,
        "instance_id": seed,
        "family": family,
        "dimensions_m": [length, width, height],
        "paint": paint,
        "location_world": list(location),
        "heading_rad": heading,
        "root": root.name,
    }


def background(route, rng):
    global CURRENT
    CURRENT = None
    context = CONTEXTS[route % len(CONTEXTS)]
    bend = rng.uniform(-0.000035, 0.000035)

    def pos(x, y=0, z=0):
        return Vector((x, y + bend * x * x, z))

    material_name = "dry_grass" if route % 3 == 0 else "grass"
    box("Terrain", (400, 0, -0.5), (1200, 240, 0.5), material_name)
    for x in range(-50, 1000, 10):
        heading = math.atan(2 * bend * x)
        for lane in [0, 4.5]:
            obj = box("Ballast", pos(x, lane, -0.13), (10.1, 3.7, 0.24), "ballast")
            obj.rotation_euler.z = heading
            for side in [-1, 1]:
                for z, w, h, mat in [
                    (0.025, 0.14, 0.035, "rust"),
                    (0.1, 0.025, 0.13, "rust"),
                    (0.17, 0.07, 0.04, "steel"),
                ]:
                    obj = box(
                        "Rail section",
                        pos(x, lane + side * 0.7525, z),
                        (10.02, w, h),
                        mat,
                    )
                    obj.rotation_euler.z = heading
            for offset in [-4, -2.8, -1.6, -0.4, 0.8, 2, 3.2, 4.4]:
                obj = box(
                    "Sleeper",
                    pos(x + offset, lane, -0.01),
                    (0.24, 2.6, 0.16),
                    "concrete",
                )
                obj.rotation_euler.z = heading
        road_y = -10.5 if route % 2 == 0 else 13
        obj = box("Parallel road", pos(x, road_y, -0.06), (10.1, 7, 0.12), "asphalt")
        obj.rotation_euler.z = heading
        obj = box("Road marking", pos(x, road_y, 0.009), (4, 0.10, 0.012), "white")
        obj.rotation_euler.z = heading
    for x in range(-30, 1000, 35):
        for side in [-1, 1]:
            y = -5 if side < 0 else 10
            cylinder(
                "Catenary support", pos(x, y, 0), pos(x, y, 7), 0.12, "steel", sides=8
            )
            cylinder(
                "Cantilever",
                pos(x, y, 6.4),
                pos(x, side * 1.2 + 2.2, 5.8),
                0.035,
                "steel",
                sides=6,
            )
        for lane in [0, 4.5]:
            cylinder(
                "Contact wire",
                pos(x, lane, 5.8),
                pos(x + 35, lane, 5.8),
                0.012,
                "steel",
                sides=4,
            )
            cylinder(
                "Messenger wire",
                pos(x, lane, 6.5),
                pos(x + 35, lane, 6.5),
                0.012,
                "steel",
                sides=4,
            )
    for i in range(25):
        x = rng.uniform(-30, 950)
        y = rng.choice([-1, 1]) * rng.uniform(24, 70)
        w = rng.uniform(7, 20)
        d = rng.uniform(7, 12)
        h = rng.uniform(4, 11)
        box(
            "Background building",
            pos(x, y, h / 2),
            (w, d, h),
            rng.choice(["brick", "concrete", "white"]),
        )
        box("Building roof", pos(x, y, h + 0.1), (w + 0.4, d + 0.4, 0.22), "steel")
        for xx in np.arange(-w / 2 + 1, w / 2 - 1, 2):
            for z in np.arange(1.5, h - 1, 2.6):
                box(
                    "Window frame",
                    pos(x + xx, y - d / 2 - 0.03, z),
                    (1, 0.08, 1.3),
                    "white",
                )
                box(
                    "Window",
                    pos(x + xx, y - d / 2 - 0.08, z),
                    (0.85, 0.03, 1.15),
                    "glass",
                )
    for i in range(90):
        x = rng.uniform(-40, 1000)
        y = rng.choice([-1, 1]) * rng.uniform(20, 100)
        h = rng.uniform(5, 12)
        cylinder(
            "Tree trunk", pos(x, y, -0.25), pos(x, y, h * 0.8), 0.14, "wood", sides=7
        )
        for j in range(3):
            center = pos(
                x + rng.uniform(-1, 1), y + rng.uniform(-1, 1), h * 0.7 + j * 0.45
            )
            vertices = []
            faces = []
            for k in range(90):
                p = center + Vector(
                    (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-1.6, 1.6))
                )
                a = rng.uniform(0, math.tau)
                size = rng.uniform(0.15, 0.4)
                n = len(vertices)
                vertices.extend(
                    [
                        p + Vector((size * math.cos(a), size * math.sin(a), 0)),
                        p
                        + Vector((-size * math.sin(a), size * math.cos(a), size * 0.6)),
                        p - Vector((size * math.cos(a), size * math.sin(a), 0)),
                    ]
                )
                faces.append((n, n + 1, n + 2))
            mesh("Leaf cluster", vertices, faces, "leaf")
    if context == "station":
        for x in range(0, 900, 100):
            box("Platform", pos(x, 8, 0.35), (75, 3.3, 0.7), "concrete")
            box("Platform safety edge", pos(x, 6.5, 0.715), (75, 0.14, 0.025), "yellow")
            for xx in range(-30, 40, 15):
                cylinder(
                    "Canopy support",
                    pos(x + xx, 9, 0.7),
                    pos(x + xx, 9, 4),
                    0.07,
                    "steel",
                )
            box("Canopy", pos(x, 9, 4), (65, 4, 0.15), "steel")
    if context in ["crossing", "industrial"]:
        for x in range(50, 900, 130):
            box("Crossing road", pos(x, 0, -0.01), (7, 50, 0.12), "asphalt")
    if context in ["bridge", "cutting"]:
        for x in range(80, 900, 180):
            box("Overpass", pos(x, 0, 7.4), (7, 45, 1.1), "concrete")
            for y in [-8, 18]:
                box("Bridge pier", pos(x, y, 3.4), (2, 2, 6.8), "concrete")
    # Railway rolling stock is a hard negative, never a road-vehicle target.
    for station in [260, 520, 780]:
        for offset in [-10, 10]:
            x = station + offset
            anchor = bpy.data.objects.new("Parked rail carriage", None)
            bpy.context.scene.collection.objects.link(anchor)
            anchor.location = pos(x, 4.5, 0)
            anchor.rotation_euler.z = math.atan(2 * bend * x)
            box("Railcar body", (0, 0, 2), (19, 2.9, 2.8), "paint_0", parent=anchor)
            box("Railcar roof", (0, 0, 3.5), (19, 3, 0.2), "steel", parent=anchor)
            for side in [-1, 1]:
                box(
                    "Railcar white band",
                    (0, side * 1.46, 2.5),
                    (18.5, 0.025, 1),
                    "white",
                    parent=anchor,
                )
                for xx in range(-8, 9, 2):
                    box(
                        "Railcar window",
                        (xx, side * 1.49, 2.5),
                        (1.4, 0.025, 0.7),
                        "glass",
                        parent=anchor,
                    )
            for xx in [-6, 6]:
                for side in [-1, 1]:
                    cylinder(
                        "Railcar wheel",
                        (xx, side * 1.05 - 0.1, 0.45),
                        (xx, side * 1.05 + 0.1, 0.45),
                        0.42,
                        "rubber",
                        parent=anchor,
                    )
    return pos, context, bend


def remove_actors(names):
    for name in names:
        root = bpy.data.objects.get(name)
        if root:
            for obj in list(root.children):
                data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if data.users == 0 and data.name not in {
                    m.name for m in CUBES.values()
                }:
                    bpy.data.meshes.remove(data)
            bpy.data.objects.remove(root, do_unlink=True)


def render_route(route, frames, start, stop, capture=None):
    global CURRENT
    rng = random.Random(80231 + route * 1031)
    scene, sky, mask_output, backend = setup()
    CUBES.clear()
    pos, context, bend = background(route, rng)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "TRAIN_FRONT_RGB"
    camera.data.clip_end = 1500
    scene.camera = camera
    camera.data.sensor_width = 36
    actors = []
    occluders = []
    split = "train" if route < 8 else "val" if route < 10 else "test"
    for index in range(start, min(frames, stop)):
        sample = f"route_{route:02}_{index:04}"
        meta_path = OUT / "metadata" / (sample + ".json")
        if capture is None and meta_path.exists():
            continue
        remove_actors(actors)
        actors = []
        for obj in occluders:
            bpy.data.objects.remove(obj, do_unlink=True)
        occluders = []
        sample_seed = route * 100000 + index + 9900000
        shot = random.Random(sample_seed)
        station = 20 + index * 20
        camera.location = pos(
            station, shot.uniform(-0.15, 0.15), shot.uniform(3.1, 4.4)
        )
        target = pos(station + 70, shot.uniform(-2, 2), shot.uniform(1.0, 2.8))
        camera.rotation_euler = (
            (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        camera.data.lens = shot.uniform(24, 32)
        sky.sun_elevation = math.radians(shot.uniform(10, 60))
        sky.sun_rotation = shot.uniform(0, math.tau)
        sky.air_density = shot.uniform(0.8, 1.5)
        sky.dust_density = shot.uniform(0.3, 2.0)
        scene.view_settings.exposure = shot.uniform(-1.8, -0.8)
        negative = index % 5 == 4
        records = []
        occupied = []
        if not negative:
            for local_id in range(1, shot.randint(3, 6) + 1):
                family = FAMILIES[(route * 7 + index * 3 + local_id) % len(FAMILIES)]
                band = shot.choices(["near", "medium", "far"], [0.55, 0.35, 0.10])[0]
                distance = shot.uniform(
                    *{"near": (18, 36), "medium": (36, 70), "far": (70, 130)}[band]
                )
                road_y = -10.5 if route % 2 == 0 else 13
                lateral = road_y + shot.choice([-1.6, 1.6])
                # Prevent neighboring cars at almost identical poses from overlapping.
                distance += local_id * 4
                distance = max(distance, abs(lateral) * 2.1)
                while any(
                    abs(distance - old_x) < 13 and abs(lateral - old_y) < 3.4
                    for old_x, old_y in occupied
                ):
                    distance += 13
                occupied.append((distance, lateral))
                heading = (
                    math.atan(2 * bend * (station + distance))
                    + shot.choice([0, math.pi, math.pi / 4, -math.pi / 4])
                    + shot.uniform(-0.25, 0.25)
                )
                location = pos(station + distance, lateral, 0.025)
                record = vehicle(
                    family,
                    shot,
                    local_id,
                    location,
                    heading,
                    sample_seed * 10 + local_id,
                )
                record["distance_band"] = band
                record["range_m"] = (location - camera.location).length
                records.append(record)
                actors.append(record["root"])
                if local_id == 1 and index % 4 == 1:
                    # Low roadside wall in front of a vehicle; true occlusion is reflected in IndexOB.
                    wall = box(
                        "Foreground low wall",
                        pos(station + distance * 0.8, lateral * 0.8, 0.45),
                        (shot.uniform(2.5, 4.5), 0.25, 0.9),
                        "concrete",
                    )
                    wall.rotation_euler.z = math.atan(2 * bend * (station + distance))
                    occluders.append(wall)
        scene.frame_set(index + 1)
        scene.render.filepath = str(OUT / "images" / (sample + ".png"))
        mask_output.file_slots[0].path = sample + "_"
        bpy.context.view_layer.update()
        if capture is not None:
            capture(scene, camera, records, sample, split, route)
            continue
        bpy.ops.render.render(write_still=True)
        exr = OUT / "masks" / (sample + f"_{index+1:04}.exr")
        image = bpy.data.images.load(str(exr), check_existing=False)
        width, height = image.size
        values = np.empty(width * height * image.channels, dtype=np.float32)
        image.pixels.foreach_get(values)
        ids = (
            np.flipud(values.reshape(height, width, image.channels)[:, :, 0])
            .round()
            .astype(np.int16)
        )
        bpy.data.images.remove(image)
        labels = []
        for record in records:
            yy, xx = np.where(ids == record["mask_id"])
            record["visible_pixels"] = int(len(xx))
            record["bbox_xyxy_pixels"] = None
            if len(xx) >= 3:
                x0, x1 = int(xx.min()), int(xx.max()) + 1
                y0, y1 = int(yy.min()), int(yy.max()) + 1
                if x1 - x0 >= 2 and y1 - y0 >= 2:
                    record["bbox_xyxy_pixels"] = [x0, y0, x1, y1]
                    labels.append(
                        f"0 {(x0+x1)/2/width:.8f} {(y0+y1)/2/height:.8f} {(x1-x0)/width:.8f} {(y1-y0)/height:.8f}"
                    )
        (OUT / "labels" / (sample + ".txt")).write_text(
            "\n".join(labels) + ("\n" if labels else "")
        )
        meta = {
            "sample_id": sample,
            "split": split,
            "route_id": route,
            "context": context,
            "route_seed": 80231 + route * 1031,
            "sample_seed": sample_seed,
            "station_m": station,
            "width": width,
            "height": height,
            "negative_requested": negative,
            "objects": records,
            "lens_mm": camera.data.lens,
            "camera_matrix_world": [list(row) for row in camera.matrix_world],
            "backend": backend,
            "exposure": scene.view_settings.exposure,
            "sun_elevation_rad": sky.sun_elevation,
            "annotation_policy": "Visible object-index pass from the same RGB render. >=3 pixels and >=2 pixels per dimension.",
            "scene_policy": "Independent traffic arrangements, not a temporally coherent tracking sequence.",
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        if index in [0, frames - 1]:
            bpy.ops.wm.save_as_mainfile(
                filepath=str(OUT / "scenes" / (sample + ".blend"))
            )
        print(
            "V2_SAMPLE", sample, "labels", len(labels), "negative", negative, flush=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=40)
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    for folder in ["images", "labels", "masks", "metadata", "scenes"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    for route in args.routes:
        MATERIALS.clear()
        CUBES.clear()
        render_route(route, args.frames, args.start, args.stop)
    print("V2_RENDER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
