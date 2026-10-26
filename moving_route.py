"""Build a continuous railway with an animated sensor train and repeated class instances."""

from pathlib import Path

BASE = Path(__file__).parent
# Reuse primitive/class helpers only; do not run the previous gallery generator.
exec(
    compile(
        (BASE / "generate.py").read_text().split("# Three geographically separated")[0],
        str(BASE / "generate.py"),
        "exec",
    )
)
from collections import Counter
import sys

OUT = BASE / "moving_dataset"
OUT.mkdir(exist_ok=True)
random.seed(23024)
LENGTH = 1260.0
FPS = 24
SPEED = 12.0
END = int(LENGTH / SPEED * FPS) + 1
s.frame_start = 1
s.frame_end = END
s.render.fps = FPS
s.render.resolution_x = 1280
s.render.resolution_y = 720
s.cycles.samples = 20
s.cycles.use_denoising = True
s["project"] = "OSDaR23-inspired continuous railway, moving sensor train"
s["instructions"] = (
    "Space: play the 105-second train journey. Numpad 0: onboard view. RGB_CENTER is active. Capture script exports paired RGB and LiDAR."
)
s["limitations"] = (
    "Procedural reconstruction of railway content types; assumed geometry/calibration. Not the original dataset."
)
# Physical daylight sky, warm low-angle sun and procedural surface finishes.
world = s.world.node_tree
world.nodes.clear()
sky = world.nodes.new("ShaderNodeTexSky")
sky.sky_type = "NISHITA"
sky.sun_elevation = math.radians(24)
sky.sun_rotation = math.radians(135)
sky.altitude = 0.1
sky.air_density = 1.1
bg = world.nodes.new("ShaderNodeBackground")
bg.inputs[1].default_value = 0.3
wo = world.nodes.new("ShaderNodeOutputWorld")
world.links.new(sky.outputs[0], bg.inputs[0])
world.links.new(bg.outputs[0], wo.inputs[0])
sun = next(o for o in s.objects if o.type == "LIGHT")
sun.data.energy = 2.2
sun.data.angle = 0.045
sun.rotation_euler = (0.5, -0.8, -0.6)


def textured(name, c, scale, rough=0.8, metal=0):
    m = mat(name, c, metal)
    p = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    p.inputs["Roughness"].default_value = rough
    no = m.node_tree.nodes.new("ShaderNodeTexNoise")
    no.inputs["Scale"].default_value = scale
    no.inputs["Detail"].default_value = 3
    coord = m.node_tree.nodes.new("ShaderNodeTexCoord")
    m.node_tree.links.new(coord.outputs["Object"], no.inputs["Vector"])
    ramp = m.node_tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*[v * 0.55 for v in c], 1)
    ramp.color_ramp.elements[1].color = (*[min(1, v * 1.35) for v in c], 1)
    m.node_tree.links.new(no.outputs["Fac"], ramp.inputs[0])
    m.node_tree.links.new(ramp.outputs[0], p.inputs["Base Color"])
    bump = m.node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    bump.inputs["Distance"].default_value = 0.045
    m.node_tree.links.new(no.outputs["Fac"], bump.inputs["Height"])
    m.node_tree.links.new(bump.outputs[0], p.inputs["Normal"])
    return m


M["ballast"] = textured("Angular grey ballast", (0.32, 0.30, 0.27), 28)
M["grass"] = textured("Mixed dry grass", (0.22, 0.29, 0.10), 1.8)
M["wood"] = textured("Creosote and bark", (0.14, 0.085, 0.045), 12)
M["concrete"] = textured("Weathered concrete", (0.48, 0.47, 0.43), 16)
M["brick"] = textured("Red brown masonry", (0.32, 0.13, 0.075), 10)
M["asphalt"] = textured("Road asphalt", (0.10, 0.105, 0.11), 35)
M["rust"] = textured("Oxidized rail web", (0.24, 0.105, 0.045), 19, 0.7, 0.5)
M["roof"] = textured("Slate roofing", (0.12, 0.14, 0.16), 20)
M["leaf"] = textured("Summer leaves", (0.105, 0.23, 0.045), 3)
M["yellow"] = mat("Safety yellow", (0.95, 0.65, 0.025))
M["silver"] = mat("Brushed aluminium", (0.62, 0.65, 0.68), 0.75)
M["navy"] = mat("Denim", (0.045, 0.065, 0.11))
M["coat"] = mat("Jacket", (0.20, 0.22, 0.19))
M["skin"] = mat("Natural skin", (0.58, 0.36, 0.24))
M["red"] = mat("Rail red enamel", (0.49, 0.022, 0.018), 0.35)
M["glass"] = mat("Tinted glazing", (0.035, 0.075, 0.09), 0.55)
# Cache spheres as shared mesh data to make repeated detailed assemblies efficient.
_sphere = {}


def ball(loc, size, material="skin"):
    if material not in _sphere:
        n, m = 16, 10
        vs = [
            (
                math.sin(j * math.pi / m) * math.cos(i * 2 * math.pi / n),
                math.sin(j * math.pi / m) * math.sin(i * 2 * math.pi / n),
                math.cos(j * math.pi / m),
            )
            for j in range(m + 1)
            for i in range(n)
        ]
        fs = [
            (j * n + i, j * n + (i + 1) % n, (j + 1) * n + (i + 1) % n, (j + 1) * n + i)
            for j in range(m)
            for i in range(n)
        ]
        mesh = bpy.data.meshes.new("Smooth sphere")
        mesh.from_pydata(vs, [], fs)
        mesh.update()
        mesh.materials.append(M[material])
        _sphere[material] = mesh
        for p in mesh.polygons:
            p.use_smooth = True
    o = bpy.data.objects.new("rounded", _sphere[material])
    s.collection.objects.link(o)
    o.location = loc
    o.scale = size
    o.parent = current
    if current:
        o.pass_index = current["instance_id"]
    return o


def human(dx=0, dy=0):
    coat = random.choice(["coat", "blue", "orange", "navy"])
    stride = random.uniform(-0.1, 0.1)
    ball((dx, dy, 1.62), (0.105, 0.10, 0.135))
    ball((dx, dy, 1.70), (0.108, 0.102, 0.075), "wood")
    rod((dx, dy, 1.39), (dx, dy, 1.53), 0.055, "skin")
    ball((dx, dy, 1.18), (0.21, 0.135, 0.29), coat)
    ball((dx, dy, 0.91), (0.16, 0.12, 0.15), "navy")
    for side in [-1, 1]:
        hip = (dx + side * 0.10, dy, 0.92)
        knee = (dx + side * 0.11, dy + side * stride, 0.52)
        ankle = (dx + side * 0.12, dy + side * stride * 1.5, 0.11)
        rod(hip, knee, 0.075, "navy")
        rod(knee, ankle, 0.058, "navy")
        ball((ankle[0], ankle[1] + 0.055, 0.07), (0.08, 0.15, 0.065), "dark")
        shoulder = (dx + side * 0.22, dy, 1.4)
        elbow = (dx + side * 0.27, dy - 0.02, 1.12)
        hand = (dx + side * 0.27, dy + 0.08, 0.91)
        rod(shoulder, elbow, 0.059, coat)
        rod(elbow, hand, 0.047, coat)
        ball(hand, (0.045, 0.04, 0.065))
    if coat == "orange":
        for z in [1.1, 1.25]:
            box((dx, dy - 0.137, z), (0.36, 0.012, 0.035), "yellow")


def wheel(x, y, z, r=0.33):
    n, m = 24, 8
    vs = []
    fs = []
    for i in range(n):
        a = i * 2 * math.pi / n
        for j in range(m):
            b = j * 2 * math.pi / m
            vs.append(
                (
                    (r + 0.04 * math.cos(b)) * math.cos(a),
                    0.04 * math.sin(b),
                    (r + 0.04 * math.cos(b)) * math.sin(a),
                )
            )
    for i in range(n):
        for j in range(m):
            fs.append(
                (
                    i * m + j,
                    ((i + 1) % n) * m + j,
                    ((i + 1) % n) * m + (j + 1) % m,
                    i * m + (j + 1) % m,
                )
            )
    me = bpy.data.meshes.new("Rubber tyre")
    me.from_pydata(vs, [], fs)
    o = bpy.data.objects.new("Wheel", me)
    s.collection.objects.link(o)
    finish(o, "Wheel", (x, y, z), (1, 1, 1), "dark")
    for a in range(0, 360, 45):
        t = math.radians(a)
        rod((x, y, z), (x + r * math.cos(t), y, z + r * math.sin(t)), 0.008)
    rod((x, y - 0.05, z), (x, y + 0.05, z), 0.055)


def rounded_box(loc, size, material, r=0.04):
    o = box(loc, size, material)
    o.data = o.data.copy()
    for v in o.data.vertices:
        for i in range(3):
            v.co[i] *= o.scale[i]
    o.scale = (1, 1, 1)
    mod = o.modifiers.new("Soft manufactured edges", "BEVEL")
    mod.width = r
    mod.segments = 2
    o.modifiers.new("Weighted corner normals", "WEIGHTED_NORMAL")
    return o


def rolling_stock(cls):
    l, w, h = (19, 2.9, 3.45) if cls == "train" else (13, 2.8, 2.7)
    rounded_box(
        (0, 0, h / 2 + 0.65), (l, w, h), "red" if cls == "train" else "rust", 0.12
    )
    box((0, 0, 0.7), (l, 2.6, 0.32), "dark")
    rounded_box((0, 0, h + 0.7), (l, 2.95, 0.25), "silver", 0.12)
    if cls == "train":
        for side in [-1, 1]:
            box((0, side * 1.455, 2.35), (l - 0.6, 0.035, 0.95), "white")
            for x in [-7.5, -5.5, -3.5, -1.5, 0.5, 2.5, 4.5, 6.5]:
                rounded_box((x, side * 1.49, 2.48), (1.35, 0.04, 0.67), "glass", 0.045)
            for x in [-8.4, 7.7]:
                box((x, side * 1.51, 1.9), (0.85, 0.04, 2.1), "silver")
                box((x, side * 1.54, 2.45), (0.6, 0.03, 0.65), "glass")
        rounded_box((9.51, 0, 2.85), (0.035, 2.4, 0.85), "glass", 0.07)
        for y in [-1, 1]:
            ball((9.54, y, 1.45), (0.04, 0.15, 0.12), "white")
        for x in [-4, 4]:
            rounded_box((x, 0, 4.25), (2.7, 1.8, 0.4), "silver", 0.08)
    else:
        for x in range(-6, 7):
            for side in [-1, 1]:
                box((x, side * 1.42, 2), (0.07, 0.08, 2.3), "steel")
    for x in [-l * 0.34, l * 0.34]:
        box((x, 0, 0.55), (2.5, 2.2, 0.35), "dark")
        for xx in [-0.65, 0.65]:
            for y in [-1.05, 1.05]:
                # Train wheels: disk, flange and axle, not bicycle spokes.
                o = rod(
                    (x + xx, y - 0.09, 0.44), (x + xx, y + 0.09, 0.44), 0.42, "dark"
                )
                rod((x + xx, -1.1, 0.44), (x + xx, 1.1, 0.44), 0.08, "steel")
    rod((-l / 2 - 0.6, 0, 0.8), (-l / 2, 0, 0.8), 0.09, "dark")
    rod((l / 2, 0, 0.8), (l / 2 + 0.6, 0, 0.8), 0.09, "dark")


def car():
    color = random.choice(["blue", "red", "silver", "white"])
    rounded_box((0, 0, 0.72), (4.35, 1.82, 0.65), color, 0.16)
    rounded_box((-0.25, 0, 1.26), (2.35, 1.64, 0.68), color, 0.18)
    for side in [-1, 1]:
        for x in [-0.85, 0.25]:
            rounded_box((x, side * 0.835, 1.32), (0.9, 0.025, 0.43), "glass", 0.05)
        for x in [-1.4, 1.4]:
            rod((x, side * 0.83, 0.37), (x, side * 1.0, 0.37), 0.33, "dark")
            rod((x, side * 1.001, 0.37), (x, side * 1.015, 0.37), 0.20, "silver")
    for x in [-1.45, 0.96]:
        box((x, 0, 1.37), (0.035, 1.45, 0.43), "glass")
    for y in [-0.64, 0.64]:
        box((2.19, y, 0.85), (0.025, 0.32, 0.16), "white")
        box((-2.19, y, 0.85), (0.025, 0.32, 0.16), "red")


old_make = make


def make(cls, loc):
    if cls == "flame":
        r = root(cls, loc)
        for j in range(9):
            h = random.uniform(0.6, 1.5)
            radius = random.uniform(0.12, 0.25)
            n = 9
            vs = [
                (
                    radius * math.cos(i * 2 * math.pi / n),
                    radius * math.sin(i * 2 * math.pi / n),
                    0,
                )
                for i in range(n)
            ] + [(random.uniform(-0.15, 0.15), 0, h)]
            me = bpy.data.meshes.new("Flame tongues")
            me.from_pydata(vs, [], [(i, (i + 1) % n, n) for i in range(n)])
            o = bpy.data.objects.new("flame", me)
            s.collection.objects.link(o)
            finish(
                o,
                "flame",
                (random.uniform(-0.35, 0.35), random.uniform(-0.25, 0.25), 0),
                (1, 1, 1),
                "orange",
            )
        return r
    if cls not in ["train", "wagons", "road_vehicle"]:
        return old_make(cls, loc)
    r = root(cls, loc)
    if cls == "road_vehicle":
        car()
    else:
        rolling_stock(cls)
    return r


def path(t):
    return Vector((t, 8 * math.sin(t / 240), 0))


def yaw(t):
    return math.atan2(8 / 240 * math.cos(t / 240), 1)


def point(t, y=0, z=0):
    p = path(t)
    a = yaw(t)
    return p + Vector((-math.sin(a) * y, math.cos(a) * y, z))


def place(cls, t, y=0, z=0.17, angle=0):
    r = make(cls, point(t, y, z))
    r.rotation_euler.z = yaw(t) + angle
    r["route_station_m"] = t
    return r


def text(txt, loc, size=0.35, rot=(math.pi / 2, 0, 0)):
    cu = bpy.data.curves.new("Sign", "FONT")
    cu.body = txt
    cu.size = size
    cu.extrude = 0.001
    o = bpy.data.objects.new(txt, cu)
    s.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = rot
    cu.materials.append(M["white"])
    return o


def terrain_strip(y0, y1, material, z=-0.25, step=10):
    vs = []
    for t in range(-100, 1451, step):
        for y in [y0, y1]:
            vs.append(
                point(t, y, z + (0.06 * math.sin(t * 0.075 + y) if abs(y) > 12 else 0))
            )
    fs = [(i, i + 1, i + 3, i + 2) for i in range(0, len(vs) - 2, 2)]
    me = bpy.data.meshes.new("ground")
    me.from_pydata(vs, [], fs)
    o = bpy.data.objects.new("Continuous " + material, me)
    s.collection.objects.link(o)
    me.materials.append(M[material])


current = None
terrain_strip(-110, 110, "grass", -0.33)
for lane in [0, 4.5, -4.5]:
    terrain_strip(lane - 1.8, lane + 1.8, "ballast", -0.035)


# Continuous correctly gauged track: rust web, polished head, base and sleepers.
def rail_section(t, lane, length=10):
    r = root("track", point(t, lane, 0))
    r.rotation_euler.z = yaw(t)
    r["route_station_m"] = t
    for side in [-1, 1]:
        y = side * 0.7525
        box((0, y, 0.025), (length, 0.145, 0.025), "rust")
        box((0, y, 0.09), (length, 0.025, 0.12), "rust")
        box((0, y, 0.157), (length, 0.07, 0.045), "steel")
    for i in range(16):
        x = -4.7 + i * 0.6
        box((x, 0, -0.035), (0.23, 2.6, 0.17), "concrete")
        for y in [-0.7525, 0.7525]:
            box((x, y, 0.065), (0.26, 0.21, 0.035), "dark")
    return r


for t in range(-75, 1390, 10):
    for lane in [0, 4.5, -4.5]:
        rail_section(t, lane)
print("TRACK COMPLETE", len(s.objects), flush=True)
current = None
# Catenary poles with insulators, paired wires, droppers and alternating cantilevers.
for i, t in enumerate(range(-50, 1400, 30)):
    r = place("catenary_pole", t, -8.3, 0)
    current = r
    for z in [5.7, 6.3]:
        for y in [0.5, 0.8, 1.1]:
            ball((0, y, z), (0.10, 0.045, 0.10), "white")
    rod((0, 0, 7), (0, 3.8, 6.5), 0.035)
current = None
for t in range(-90, 1420, 5):
    for lane in [-4.5, 0, 4.5]:
        for z in [5.75, 6.5]:
            rod(point(t, lane, z), point(t + 5, lane, z), 0.011, "steel")
        if t % 10 == 0:
            rod(point(t, lane, 5.75), point(t, lane, 6.5), 0.008, "steel")
# Embankment fencing, drainage and a parallel service road.
terrain_strip(-16, -12, "asphalt", -0.13)
for t in range(-30, 1390, 8):
    for y in [-10.5, 12.5]:
        rod(point(t, y, 0), point(t, y, 1.5), 0.035, "steel")
        for z in [0.45, 1.2]:
            rod(point(t, y, z), point(t + 8, y, z), 0.016, "steel")
for t in range(-30, 1390, 8):
    a = point(t, -14, -0.115)
    b = point(t + 3, -14, -0.115)
    rod(a, b, 0.025, "white")
print("INFRASTRUCTURE COMPLETE", flush=True)


# Platform stations; architecture is at human scale with door/window frames and gabled roofs.
def house(t, y, width=12, depth=8, height=6):
    global current
    current = None
    p = point(t, y, 0)
    anchor = bpy.data.objects.new("Station architecture", None)
    s.collection.objects.link(anchor)
    anchor.location = p
    anchor.rotation_euler.z = yaw(t)
    # Temporarily use an unlabeled anchor; finish requires ID, so background parts are parented after construction.
    before = set(s.objects)
    box((0, 0, height / 2), (width, depth, height), "brick")
    box((0, 0, 0.2), (width + 0.3, depth + 0.3, 0.4), "concrete")
    for side in [-1, 1]:
        roof = box(
            (0, side * depth * 0.25, height + 0.65),
            (width + 0.9, depth * 0.57, 0.18),
            "roof",
        )
        roof.rotation_euler.x = -side * 0.3
        for x in range(int(-width / 2 + 1), int(width / 2), 2):
            for z in [1.7, 4.3]:
                box((x, side * (depth / 2 + 0.02), z), (1.1, 0.08, 1.45), "white")
                box((x, side * (depth / 2 + 0.07), z), (0.95, 0.03, 1.3), "glass")
                box((x, side * (depth / 2 + 0.09), z), (0.04, 0.03, 1.3), "white")
    for o in set(s.objects) - before:
        o.parent = anchor
    return anchor


stations = [(145, "NORDHAFEN"), (785, "WALDPARK")]
for t, name in stations:
    current = None
    # individual slabs, yellow edge line, tactile strips, benches and canopy.
    for xx in range(-65, 66, 5):
        p = point(t + xx, 8.4, 0.35)
        o = box(p, (5, 3.8, 0.7), "concrete")
        o.rotation_euler.z = yaw(t + xx)
        for yy in [6.75, 7.05]:
            o = box(
                point(t + xx, yy, 0.715),
                (5, 0.13, 0.025),
                "yellow" if yy < 7 else "white",
            )
            o.rotation_euler.z = yaw(t + xx)
    for xx in range(-55, 56, 11):
        rod(point(t + xx, 9.3, 0.7), point(t + xx, 9.3, 4), 0.09)
        o = box(point(t + xx, 9, 4), (11.2, 4.5, 0.17), "silver")
        o.rotation_euler.z = yaw(t + xx)
        o = box(point(t + xx, 9, 0.95), (2, 0.55, 0.12), "wood")
        o.rotation_euler.z = yaw(t + xx)
        for k in [-0.75, 0.75]:
            rod(point(t + xx + k, 9, 0.7), point(t + xx + k, 9, 0.95), 0.035)
        box(point(t + xx, 10, 1), (0.5, 0.5, 0.9), "steel")
    house(t, 18, 30, 9, 6)
    text(name, point(t - 10, 13.4, 3), 0.55)
    for xx in [-45, 5, 45]:
        o = box(point(t + xx, 10, 2.6), (3, 0.12, 0.7), "blue")
        text(name, point(t + xx - 1.3, 9.92, 2.4), 0.25)
for t in range(100, 1250, 100):
    house(
        t,
        random.choice([-30, 32]),
        random.choice([10, 14, 18]),
        8,
        random.choice([5, 7]),
    )
print("BUILDINGS COMPLETE", flush=True)

# Repeated labeled populations. At least 12 independent assemblies per class.
# Six locations contain two staggered groups each; no repeated world overlaps.
for zone, center in enumerate([120, 320, 520, 740, 940, 1140]):
    for rep in range(2):
        t = center + rep * 42
        platform = zone in [0, 3]
        py = 8 if platform else 9
        for j in range(4):
            place(
                "person",
                t + j * 5,
                py + random.uniform(-0.4, 0.5),
                0.73 if platform else 0.05,
                random.uniform(-1, 1),
            )
        place("crowd", t + 13, py + 1.3, 0.73 if platform else 0.05, math.pi / 2)
        place("bicycle", t - 9, py + 0.3, 0.73 if platform else 0.05, math.pi / 2)
        place(
            "group_of_bicycles",
            t + 23,
            py + 0.4,
            0.73 if platform else 0.05,
            math.pi / 2,
        )
        place("wheelchair", t + 5, py - 0.5, 0.73 if platform else 0.05, math.pi / 2)
        place("motorcycle", t + 30, -12.8, 0.05)
        place("road_vehicle", t - 6, -14, 0.02, math.pi if rep else 0)
        place("animal", t - 16, -8, 0.05)
        place("group_of_animals", t + 26, 11, 0.05)
        place("drag_shoe", t + 6, -4.5 + 0.7525, 0.18)
        place("buffer_stop", t + 34, 4.5, 0.17)
        place("switch", t + 10, 4.5, 0.02)
        place("transition", t + 30, -4.5, 0.02)
        place("signal_pole", t + 18, -2.5, 0)
        place("signal", t + 18, -2.5, 0)
        place("signal_bridge", t + 38, 0, 0)
        # Adjacent parked trains and freight cuts, off the ego track.
        place("train", t + 17, -4.5, 0.1)
        place("wagons", t + 11, 19, 0.1)
        current = None
        # Yard siding under freight stock, outside the main running line.
        for y in [18.2475, 19.7525]:
            rod(point(t - 4, y, 0.15), point(t + 28, y, 0.15), 0.06)
        # Staged small controlled fires at a maintenance apron, away from public platforms.
        place("flame", t + 34, -8, 0.04)
        place("smoke", t + 35, -8, 0.07)
        current = None
        box(point(t + 34, -8, -0.05), (3, 2, 0.12), "concrete")
print("CLASS POPULATIONS COMPLETE", len(roots), flush=True)

# Individual ballast stones near the track; shared mesh gives affordable detail.
current = None
stone_mesh = None
for i in range(4500):
    t = random.uniform(-30, 1350)
    lane = random.choice([-4.5, 0, 4.5])
    yy = lane + random.choice([-1, 1]) * random.uniform(1.15, 1.7)
    o = ball(
        point(t, yy, 0.015),
        (
            random.uniform(0.025, 0.065),
            random.uniform(0.02, 0.05),
            random.uniform(0.02, 0.05),
        ),
        "ballast",
    )

# Porous branching trees reused from the existing railway asset pipeline.
sys.path.insert(0, str(BASE.parent / "railway_lidar"))
import foliage


class Geo:
    def __init__(self):
        self.verts = []
        self.faces = []

    def polygon(self, vs, fs):
        n = len(self.verts)
        self.verts.extend(vs)
        self.faces.extend(tuple(n + i for i in f) for f in fs)


def tree_mesh(seed):
    rng = random.Random(seed)
    wood, leaves = Geo(), Geo()
    foliage.add_tree(
        wood,
        leaves,
        (0, 0, 0),
        rng.uniform(7, 11),
        rng.uniform(2.8, 4.2),
        rng,
        "broadleaf_spreading",
        "full",
    )
    meshes = []
    for g, material in [(wood, "wood"), (leaves, "leaf")]:
        me = bpy.data.meshes.new("Branched tree " + material)
        me.from_pydata(g.verts, [], g.faces)
        me.materials.append(M[material])
        me.update()
        meshes.append(me)
    return meshes


prototypes = [tree_mesh(i + 50) for i in range(10)]
for i in range(500):
    t = random.uniform(-70, 1400)
    y = random.choice([-1, 1]) * random.uniform(22, 85)
    for me in random.choice(prototypes):
        o = bpy.data.objects.new("Porous woodland", me)
        s.collection.objects.link(o)
        o.location = point(t, y, -0.25)
        o.rotation_euler.z = random.random() * 6.28
print("WOODLAND COMPLETE", len(s.objects), flush=True)

# Moving four-car sensor train. Ego train is excluded from target coverage counts.
ego = make("train", (0, 0, 0.12))
ego.name = "EGO_SENSOR_TRAIN"
ego["ego_vehicle"] = True
for r in list(roots):
    if r == ego:
        continue
ego_parts = list(ego.children)
for car_index in range(1, 4):
    r = make("train", (-car_index * 20.5, 0, 0.12))
    r["ego_vehicle"] = True
    r.name = "EGO_CARRIAGE_" + str(car_index)
    r.parent = ego
    r.location = (-car_index * 20.5, 0, 0)
current = None


def background_part(o):
    o.parent = ego
    return o


background_part(box((9, 0, 4.5), (1.1, 0.9, 0.12), "dark"))
background_part(rod((9, 0, 4.5), (9, 0, 4.85), 0.055))
background_part(rod((9, 0, 4.85), (9, 0, 5.05), 0.14, "dark"))
background_part(box((9.4, 0, 4.73), (0.3, 0.25, 0.2), "dark"))


def camera(name, loc, target, lens=28, parent=None):
    cu = bpy.data.cameras.new(name)
    o = bpy.data.objects.new(name, cu)
    s.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (
        (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    )
    cu.lens = lens
    cu.clip_end = 1800
    cu.sensor_width = 36
    if parent:
        o.parent = parent
    return o


for side, angle in [("LEFT", 0.36), ("CENTER", 0), ("RIGHT", -0.36)]:
    camera("RGB_" + side, (9.65, 0, 4.8), (90, 90 * angle, 2.1), 26, ego)
lidar = bpy.data.objects.new("LIDAR_ORIGIN", None)
s.collection.objects.link(lidar)
lidar.parent = ego
lidar.location = (9.25, 0, 5.08)
lidar["model"] = (
    "Instantaneous geometric nearest hit, 64 channels, 2048 azimuth rays, 200 m; uncalibrated"
)
external = camera("EXTERIOR_FOLLOW", (32, -32, 16), (-15, 0, 2), 34, ego)
for frame in list(range(1, END, 12)) + [END]:
    t = (frame - 1) / FPS * SPEED
    ego.location = path(t) + Vector((0, 0, 0.12))
    ego.rotation_euler.z = yaw(t)
    ego.keyframe_insert(data_path="location", frame=frame)
    ego.keyframe_insert(data_path="rotation_euler", frame=frame)
for fc in ego.animation_data.action.fcurves:
    for k in fc.keyframe_points:
        k.interpolation = "LINEAR"
s.camera = bpy.data.objects["RGB_CENTER"]
s.frame_set(1)
for a in bpy.context.screen.areas:
    if a.type == "VIEW_3D":
        a.spaces.active.region_3d.view_perspective = "CAMERA"
        a.spaces.active.clip_end = 2500
counts = Counter(r["class_name"] for r in roots if not r.get("ego_vehicle", False))
assert all(counts[c] >= 12 for c in CLASSES), counts
manifest = {
    "classes": dict(enumerate(CLASSES)),
    "target_assembly_counts": dict(counts),
    "route_length_m": LENGTH,
    "duration_s": LENGTH / SPEED,
    "speed_m_s": SPEED,
    "fps": FPS,
    "frames": [1, END],
    "camera_names": ["RGB_LEFT", "RGB_CENTER", "RGB_RIGHT", "EXTERIOR_FOLLOW"],
    "note": "Local class IDs, not official IDs. Counts are scene assemblies, not guaranteed sensor observations. Train carriages are rigid relative to lead vehicle.",
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "moving_railway.blend"))
# Three review views from the actual train journey.
for name, t, cam in [
    ("station", 85, "RGB_CENTER"),
    ("yard", 285, "RGB_CENTER"),
    ("train_exterior", 90, "EXTERIOR_FOLLOW"),
]:
    s.frame_set(round(t / SPEED * FPS) + 1)
    s.camera = bpy.data.objects[cam]
    s.render.filepath = str(OUT / (name + ".png"))
    bpy.ops.render.render(write_still=True)
print("MOVING_ROUTE_COMPLETE", json.dumps(dict(counts)), flush=True)
