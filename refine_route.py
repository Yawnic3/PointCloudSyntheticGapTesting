"""Final lighting, vegetation and hazard appearance pass on the generated route."""

import bpy, math, random
from pathlib import Path
from mathutils import Vector

OUT = Path(__file__).parent / "moving_dataset"
bpy.ops.wm.open_mainfile(filepath=str(OUT / "moving_railway.blend"))
s = bpy.context.scene
s.view_settings.exposure = -1.7
# Enlarge individual leaf surfaces around their own centers; retain actual canopy gaps.
for me in bpy.data.meshes:
    if me.name.startswith("Branched tree leaf"):
        for start in range(0, len(me.vertices), 4):
            group = list(me.vertices[start : start + 4])
            center = sum((v.co.copy() for v in group), Vector()) / len(group)
            for v in group:
                v.co = center + (v.co - center) * 3.5
        me.update()
# More natural shaded volumetric smoke, confined to existing labeled proxy volumes.
smoke = bpy.data.materials.new("Volumetric grey smoke")
smoke.use_nodes = True
nodes = smoke.node_tree.nodes
nodes.clear()
out = nodes.new("ShaderNodeOutputMaterial")
vol = nodes.new("ShaderNodeVolumePrincipled")
vol.inputs["Color"].default_value = (0.16, 0.18, 0.2, 1)
noise = nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 3
noise.inputs["Detail"].default_value = 3
mul = nodes.new("ShaderNodeMath")
mul.operation = "MULTIPLY"
mul.inputs[1].default_value = 2
smoke.node_tree.links.new(noise.outputs["Fac"], mul.inputs[0])
smoke.node_tree.links.new(mul.outputs[0], vol.inputs["Density"])
smoke.node_tree.links.new(vol.outputs["Volume"], out.inputs["Volume"])
flame = bpy.data.materials.new("Flame light")
flame.use_nodes = True
p = next(n for n in flame.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
p.inputs["Base Color"].default_value = (1, 0.11, 0.002, 1)
p.inputs["Emission Color"].default_value = (1, 0.18, 0.005, 1)
p.inputs["Emission Strength"].default_value = 4
for o in s.objects:
    if (
        o.type == "MESH"
        and o.parent
        and o.parent.get("class_name") in ["smoke", "flame"]
    ):
        # Smoke mesh data were shared, so material replacement propagates correctly to all smoke parts.
        o.data.materials.clear()
        o.data.materials.append(smoke if o.parent["class_name"] == "smoke" else flame)
# Grass blades on the verges and beneath the woodland are real mesh surfaces.
rng = random.Random(77)
vs = []
fs = []
for i in range(33000):
    t = rng.uniform(-50, 1380)
    y = rng.choice([-1, 1]) * rng.uniform(17, 95)
    x = t
    yy = 8 * math.sin(t / 240) + y
    z = -0.27
    h = rng.uniform(0.12, 0.42)
    w = rng.uniform(0.02, 0.055)
    a = rng.random() * 6.28
    dx, dy = w * math.cos(a), w * math.sin(a)
    n = len(vs)
    vs.extend(
        [
            (x - dx, yy - dy, z),
            (x + dx, yy + dy, z),
            (x + rng.uniform(-0.10, 0.10), yy + rng.uniform(-0.10, 0.10), z + h),
        ]
    )
    fs.append((n, n + 1, n + 2))
me = bpy.data.meshes.new("Verge grass")
me.from_pydata(vs, [], fs)
me.materials.append(bpy.data.materials["Summer leaves"])
o = bpy.data.objects.new("Verge grasses", me)
s.collection.objects.link(o)
s.frame_set(1)
s.camera = bpy.data.objects["RGB_CENTER"]
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "moving_railway.blend"))
for name, t, cam in [
    ("station", 85, "RGB_CENTER"),
    ("yard", 285, "RGB_CENTER"),
    ("train_exterior", 90, "EXTERIOR_FOLLOW"),
]:
    s.frame_set(round(t / 12 * 24) + 1)
    s.camera = bpy.data.objects[cam]
    s.render.filepath = str(OUT / (name + ".png"))
    bpy.ops.render.render(write_still=True)
print("REFINEMENT COMPLETE", flush=True)
