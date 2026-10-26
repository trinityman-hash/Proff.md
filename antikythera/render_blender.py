"""
Render the OpenUSD stage with Cycles (stand-in for RenderMan on this machine).

Geometry, transforms, animation come from usd/antikythera.usda (imported by
Blender's USD importer).  Materials/lights/world are rebuilt here from the
per-frame cue sidecar (usd/shots.json) that build_usd.py wrote from the SAME
functions that drive the USD, so everything stays locked.

usage:
  python3 render_blender.py --frames 1,193,300 --w 960 --h 540 --spp 24 --out raw_test
  python3 render_blender.py --range 1 1440 1 --out raw
"""
import sys, os, json, math, argparse, time
sys.path.insert(0, "/content/antikythera")
import numpy as np
import bpy
from mathutils import Vector, Matrix

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="")
ap.add_argument("--range", nargs=3, type=int, default=None)
ap.add_argument("--w", type=int, default=960)
ap.add_argument("--h", type=int, default=540)
ap.add_argument("--spp", type=int, default=24)
ap.add_argument("--out", default="raw")
ap.add_argument("--noblur", action="store_true")
args = ap.parse_args()

if args.frames:
    FRAMES = [int(x) for x in args.frames.split(",")]
elif args.range:
    FRAMES = list(range(args.range[0], args.range[1] + 1, args.range[2]))
else:
    FRAMES = [1]

SIDE = json.load(open("/content/antikythera/usd/shots.json"))["frames"]
os.makedirs(args.out, exist_ok=True)

# ----------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = 24
sc.frame_start, sc.frame_end = 1, 1440
bpy.ops.wm.usd_import(filepath="/content/antikythera/usd/antikythera.usda",
                      import_cameras=False, import_lights=False, import_materials=True,
                      set_frame_range=True, create_collection=False,
                      import_subdivision=False, merge_parent_xform=False)

r = sc.render
r.engine = "CYCLES"
r.resolution_x, r.resolution_y, r.resolution_percentage = args.w, args.h, 100
r.image_settings.file_format = "HDR"
r.image_settings.color_mode = "RGB"
r.use_motion_blur = not args.noblur
r.motion_blur_shutter = 0.5
cy = sc.cycles
cy.device = "CPU"
try:
    _p = bpy.context.preferences.addons["cycles"].preferences
    for _dt in ("OPTIX", "CUDA"):
        try:
            _p.compute_device_type = _dt
            _p.get_devices()
            _gpus = [d for d in _p.devices if d.type != "CPU"]
            if _gpus:
                for d in _p.devices:
                    d.use = d.type != "CPU"
                cy.device = "GPU"
                print("RENDER DEVICE:", _dt, [d.name for d in _gpus])
                break
        except Exception as e:
            print("device try failed", _dt, e)
except Exception as e:
    print("GPU setup failed, using CPU:", e)
if cy.device != "GPU":
    print("RENDER DEVICE: CPU")
cy.samples = args.spp
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.03
cy.use_denoising = True
cy.denoiser = "OPENIMAGEDENOISE"
cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces = 6, 3, 4
cy.transmission_bounces, cy.volume_bounces, cy.transparent_max_bounces = 2, 0, 4
cy.sample_clamp_indirect = 4.0
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.volume_step_rate = 2.0
cy.volume_max_steps = 48
cy.use_light_tree = True
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.display_settings.display_device = "sRGB"
sc.render.threads_mode = "AUTO"

# ----------------------------------------------------------------- materials
def rebuild(mat, builder):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    surf = builder(nt)
    nt.links.new(surf, out.inputs["Surface"])


def N(nt, kind, loc=(0, 0), **kw):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in kw.items():
        setattr(n, k, v)
    return n


def principled(nt, color, metal, rough, bump=None, spec=0.5, noise_rough=0.0, ior=None):
    bsdf = N(nt, "ShaderNodeBsdfPrincipled", (600, 0))
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular IOR Level"].default_value = spec
    if ior:
        bsdf.inputs["IOR"].default_value = ior
    if bump:
        tc = N(nt, "ShaderNodeTexCoord", (-600, 0))
        nz = N(nt, "ShaderNodeTexNoise", (-400, 0))
        nz.inputs["Scale"].default_value = bump["scale"]
        nz.inputs["Detail"].default_value = bump.get("detail", 8)
        nz.inputs["Roughness"].default_value = 0.6
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        bp = N(nt, "ShaderNodeBump", (200, -200))
        bp.inputs["Strength"].default_value = bump["strength"]
        bp.inputs["Distance"].default_value = bump.get("dist", 0.1)
        nt.links.new(nz.outputs["Fac"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], bsdf.inputs["Normal"])
        if noise_rough:
            mr = N(nt, "ShaderNodeMapRange", (200, 200))
            mr.inputs["To Min"].default_value = max(0.0, rough - noise_rough)
            mr.inputs["To Max"].default_value = min(1.0, rough + noise_rough)
            nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
            nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    return bsdf


EM = {}   # name -> emission strength socket


def emitter(nt, color, strength, name):
    e = N(nt, "ShaderNodeEmission", (400, 0))
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    EM[name] = e.inputs["Strength"]
    return e.outputs["Emission"]


def crust(nt):
    tc = N(nt, "ShaderNodeTexCoord", (-900, 0))
    n1 = N(nt, "ShaderNodeTexNoise", (-700, 100))
    n1.inputs["Scale"].default_value = 0.35
    n1.inputs["Detail"].default_value = 10
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    n2 = N(nt, "ShaderNodeTexNoise", (-700, -150))
    n2.inputs["Scale"].default_value = 3.0
    n2.inputs["Detail"].default_value = 12
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    ramp = N(nt, "ShaderNodeValToRGB", (-450, 100))
    ramp.color_ramp.elements[0].color = (0.05, 0.075, 0.06, 1)
    ramp.color_ramp.elements[1].color = (0.30, 0.34, 0.24, 1)
    e = ramp.color_ramp.elements.new(0.55)
    e.color = (0.22, 0.17, 0.10, 1)
    nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    bsdf = N(nt, "ShaderNodeBsdfPrincipled", (500, 0))
    bsdf.inputs["Roughness"].default_value = 0.92
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bp = N(nt, "ShaderNodeBump", (200, -150))
    bp.inputs["Strength"].default_value = 1.0
    bp.inputs["Distance"].default_value = 0.35
    nt.links.new(n2.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], bsdf.inputs["Normal"])
    return bsdf.outputs["BSDF"]


def fogged(nt, bsdf_out):
    lp = N(nt, "ShaderNodeLightPath", (100, 300))
    mr = N(nt, "ShaderNodeMapRange", (300, 300))
    mr.inputs["From Min"].default_value = 25.0
    mr.inputs["From Max"].default_value = 160.0
    nt.links.new(lp.outputs["Camera Ray Length"] if "Camera Ray Length" in lp.outputs else lp.outputs["Ray Length"], mr.inputs["Value"])
    fog = N(nt, "ShaderNodeEmission", (300, 150))
    fog.inputs["Color"].default_value = (0.010, 0.040, 0.046, 1)
    fog.inputs["Strength"].default_value = 1.0
    mix = N(nt, "ShaderNodeMixShader", (700, 100))
    nt.links.new(mr.outputs["Result"], mix.inputs[0])
    nt.links.new(bsdf_out, mix.inputs[1])
    nt.links.new(fog.outputs["Emission"], mix.inputs[2])
    return mix.outputs["Shader"]


def corona(nt):
    SUNR = 4000.0 * math.tan(math.radians(1.6))
    tc = N(nt, "ShaderNodeTexCoord", (-1200, 0))
    sep = N(nt, "ShaderNodeSeparateXYZ", (-1000, 0))
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    comb = N(nt, "ShaderNodeCombineXYZ", (-800, 100))
    nt.links.new(sep.outputs["X"], comb.inputs["X"]); nt.links.new(sep.outputs["Y"], comb.inputs["Y"])
    ln = N(nt, "ShaderNodeVectorMath", (-600, 100), operation="LENGTH")
    nt.links.new(comb.outputs["Vector"], ln.inputs[0])
    u = N(nt, "ShaderNodeMath", (-400, 100), operation="DIVIDE")
    nt.links.new(ln.outputs["Value"], u.inputs[0]); u.inputs[1].default_value = SUNR
    # radial falloff  (1/u)^2.4, faded to zero by u = 3.3
    inv = N(nt, "ShaderNodeMath", (-200, 100), operation="POWER")
    u2 = N(nt, "ShaderNodeMath", (-300, 250), operation="MAXIMUM"); u2.inputs[1].default_value = 0.9
    nt.links.new(u.outputs["Value"], u2.inputs[0])
    one = N(nt, "ShaderNodeMath", (-300, 350), operation="DIVIDE"); one.inputs[0].default_value = 1.0
    nt.links.new(u2.outputs["Value"], one.inputs[1])
    nt.links.new(one.outputs["Value"], inv.inputs[0]); inv.inputs[1].default_value = 2.6
    edge = N(nt, "ShaderNodeMapRange", (-200, -100))
    edge.inputs["From Min"].default_value = 2.2; edge.inputs["From Max"].default_value = 3.25
    edge.inputs["To Min"].default_value = 1.0; edge.inputs["To Max"].default_value = 0.0
    nt.links.new(u.outputs["Value"], edge.inputs["Value"])
    # angular streamers
    nx = N(nt, "ShaderNodeVectorMath", (-600, -200), operation="NORMALIZE")
    nt.links.new(comb.outputs["Vector"], nx.inputs[0])
    nz = N(nt, "ShaderNodeTexNoise", (-400, -250))
    nz.inputs["Scale"].default_value = 2.4; nz.inputs["Detail"].default_value = 4
    nt.links.new(nx.outputs["Vector"], nz.inputs["Vector"])
    st = N(nt, "ShaderNodeMapRange", (-200, -300))
    st.inputs["From Min"].default_value = 0.3; st.inputs["From Max"].default_value = 0.8
    st.inputs["To Min"].default_value = 0.35; st.inputs["To Max"].default_value = 1.7
    nt.links.new(nz.outputs["Fac"], st.inputs["Value"])
    m1 = N(nt, "ShaderNodeMath", (0, 100), operation="MULTIPLY")
    nt.links.new(inv.outputs["Value"], m1.inputs[0]); nt.links.new(edge.outputs["Result"], m1.inputs[1])
    m2 = N(nt, "ShaderNodeMath", (200, 100), operation="MULTIPLY")
    nt.links.new(m1.outputs["Value"], m2.inputs[0]); nt.links.new(st.outputs["Result"], m2.inputs[1])
    gain = N(nt, "ShaderNodeMath", (400, 100), operation="MULTIPLY")
    nt.links.new(m2.outputs["Value"], gain.inputs[0]); gain.inputs[1].default_value = 0.0
    EM["corona"] = gain.inputs[1]
    e = N(nt, "ShaderNodeEmission", (600, 0))
    e.inputs["Color"].default_value = (0.92, 0.9, 1.0, 1)
    nt.links.new(gain.outputs["Value"], e.inputs["Strength"])
    # transparent outside / behind: add holdout-free transparency via mix with transparent by falloff
    tr = N(nt, "ShaderNodeBsdfTransparent", (600, -150))
    mx = N(nt, "ShaderNodeMixShader", (800, 0))
    cl = N(nt, "ShaderNodeMath", (600, 250), operation="MINIMUM"); cl.inputs[1].default_value = 1.0
    nt.links.new(edge.outputs["Result"], mx.inputs[0])
    nt.links.new(tr.outputs["BSDF"], mx.inputs[1]); nt.links.new(e.outputs["Emission"], mx.inputs[2])
    return mx.outputs["Shader"]


BUILD = {
    "Bronze":     lambda nt: principled(nt, (0.80, 0.48, 0.19), 1.0, 0.30, dict(scale=60, strength=0.12, dist=0.01), noise_rough=0.12).outputs["BSDF"],
    "BronzeDark": lambda nt: principled(nt, (0.20, 0.135, 0.07), 1.0, 0.52, dict(scale=18, strength=0.25, dist=0.02), noise_rough=0.15).outputs["BSDF"],
    "Gold":       lambda nt: principled(nt, (1.0, 0.70, 0.26), 1.0, 0.16).outputs["BSDF"],
    "Crust":      crust,
    "Sand":       lambda nt: fogged(nt, principled(nt, (0.10, 0.09, 0.065), 0.0, 0.95, dict(scale=1.4, strength=0.5, dist=0.5)).outputs["BSDF"]),
    "Silver":     lambda nt: principled(nt, (0.88, 0.9, 0.93), 1.0, 0.16).outputs["BSDF"],
    "MoonBlack":  lambda nt: principled(nt, (0.012, 0.012, 0.018), 0.0, 0.22).outputs["BSDF"],
    "MoonDisc":   lambda nt: principled(nt, (0.01, 0.01, 0.014), 0.0, 0.85).outputs["BSDF"],
    "Stone":      lambda nt: principled(nt, (0.06, 0.055, 0.05), 0.0, 0.8, dict(scale=3, strength=0.4, dist=0.1)).outputs["BSDF"],
    "Sea":        lambda nt: principled(nt, (0.012, 0.03, 0.05), 0.0, 0.045, dict(scale=0.9, strength=0.25, dist=0.3), ior=1.33).outputs["BSDF"],
    "SigmaGlow":  lambda nt: emitter(nt, (1.0, 0.2, 0.07), 6.0, "sigma"),
    "BeadGlow":   lambda nt: emitter(nt, (1.0, 0.86, 0.55), 8.0, "bead"),
    "SunDisc":    lambda nt: emitter(nt, (1.0, 0.82, 0.55), 40.0, "sun"),
    "Corona":     corona,
    "Star":       lambda nt: emitter(nt, (0.85, 0.9, 1.0), 0.0, "stars"),
    "Mote":       lambda nt: emitter(nt, (1.0, 0.92, 0.72), 1.4, "mote"),
}
for m in list(bpy.data.materials):
    base = m.name.split(".")[0]
    if base in BUILD:
        rebuild(m, BUILD[base])
    else:
        print("!! no builder for", m.name)

# ----------------------------------------------------------------- lights
def mk_light(name, kind, **kw):
    ld = bpy.data.lights.new(name, kind)
    for k, v in kw.items():
        setattr(ld, k, v)
    ob = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(ob)
    return ob


def aim(obj, direction):
    d = Vector(direction).normalized()
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


L_shaft = mk_light("Shaft", "SUN", angle=math.radians(4), color=(0.5, 0.85, 0.9))
aim(L_shaft, (0.35, 0.2, -0.9))
L_glow = mk_light("CrackGlow", "POINT", shadow_soft_size=1.2, color=(1.0, 0.72, 0.38))
L_glow.location = (1.0, 1.0, 6.0)
L_sig = mk_light("SigmaLight", "POINT", shadow_soft_size=0.6, color=(1.0, 0.25, 0.1))
import shots as S
sx, sy = S.spiral_pos(S.SIGMA_CELL)[0]
L_sig.location = (sx, sy, 1.2)
L_rimA = mk_light("RimWarm", "AREA", shape="RECTANGLE", size=30, size_y=8, color=(1.0, 0.6, 0.3))
L_rimA.location = (-34, -30, 22); aim(L_rimA, (0.6, 0.5, -0.35))
L_rimB = mk_light("RimCool", "AREA", shape="RECTANGLE", size=30, size_y=8, color=(0.4, 0.7, 1.0))
L_rimB.location = (34, 30, 22); aim(L_rimB, (-0.6, -0.5, -0.4))
L_sun = mk_light("EclipseSun", "SUN", angle=math.radians(0.6), color=(1.0, 0.82, 0.62))
aim(L_sun, -S.sky_dir(0, S.SUN_EL) if hasattr(S, "sky_dir") else
    -np.array([0, math.cos(S.SUN_EL), math.sin(S.SUN_EL)]))
L_sun.rotation_euler = Vector((0, -math.cos(S.SUN_EL), -math.sin(S.SUN_EL))).to_track_quat("-Z", "Y").to_euler()

# ----------------------------------------------------------------- world
world = bpy.data.worlds.new("W")
sc.world = world
world.use_nodes = True
wt = world.node_tree
wt.nodes.clear()
wout = wt.nodes.new("ShaderNodeOutputWorld")
bgA = wt.nodes.new("ShaderNodeBackground"); bgA.inputs["Color"].default_value = (0.010, 0.040, 0.046, 1)
bgA.inputs["Strength"].default_value = 1.0
sky_day = wt.nodes.new("ShaderNodeTexSky"); sky_day.sky_type = "MULTIPLE_SCATTERING"
sky_day.sun_disc = False; sky_day.sun_elevation = math.radians(22); sky_day.sun_rotation = math.radians(180)
sky_day.air_density = 1.0; sky_day.aerosol_density = 1.5
sky_dusk = wt.nodes.new("ShaderNodeTexSky"); sky_dusk.sky_type = "MULTIPLE_SCATTERING"
sky_dusk.sun_disc = False; sky_dusk.sun_elevation = math.radians(-1.2); sky_dusk.sun_rotation = math.radians(180)
sky_dusk.air_density = 1.2; sky_dusk.aerosol_density = 4.0
mixSky = wt.nodes.new("ShaderNodeMix"); mixSky.data_type = "RGBA"
wt.links.new(sky_day.outputs["Color"], mixSky.inputs[6])
wt.links.new(sky_dusk.outputs["Color"], mixSky.inputs[7])
bgB = wt.nodes.new("ShaderNodeBackground")
wt.links.new(mixSky.outputs[2], bgB.inputs["Color"])
mixAB = wt.nodes.new("ShaderNodeMixShader")
wt.links.new(bgA.outputs["Background"], mixAB.inputs[1])
wt.links.new(bgB.outputs["Background"], mixAB.inputs[2])
wt.links.new(mixAB.outputs["Shader"], wout.inputs["Surface"])
# underwater scattering (scene A only) - gives the god-ray shaft depth
vol = wt.nodes.new("ShaderNodeVolumeScatter")
vol.inputs["Color"].default_value = (0.45, 0.8, 0.8, 1)
vol.inputs["Anisotropy"].default_value = 0.55
vol.inputs["Density"].default_value = 0.0035
# (volume disabled for render speed)

# ----------------------------------------------------------------- camera
camd = bpy.data.cameras.new("Cam")
camd.sensor_width = 36.0
camd.sensor_fit = "HORIZONTAL"
camd.clip_start, camd.clip_end = 0.1, 20000.0
camd.dof.use_dof = True
cam = bpy.data.objects.new("Cam", camd)
sc.collection.objects.link(cam)
sc.camera = cam


def pose_camera(P, Tg, roll_deg):
    P = Vector(P); fwd = (Vector(Tg) - P).normalized()
    right = fwd.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(fwd)
    a = math.radians(roll_deg)
    right, up = right * math.cos(a) + up * math.sin(a), -right * math.sin(a) + up * math.cos(a)
    M = Matrix(((right.x, up.x, -fwd.x, P.x),
                (right.y, up.y, -fwd.y, P.y),
                (right.z, up.z, -fwd.z, P.z),
                (0, 0, 0, 1)))
    cam.matrix_world = M


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def apply_state(f):
    d = SIDE[f - 1]
    Lc = d["L"]
    t = d["t"]
    B = d["scene"] == "B"
    set_visibility(B)
    pose_camera(d["pos"], d["tgt"], d["roll"])
    camd.lens = d["focal"]
    camd.dof.focus_distance = d["focus"]
    camd.dof.aperture_fstop = d["fstop"]
    # lights
    L_shaft.data.energy = 0.0 if B else 3.2
    L_glow.data.energy = 0.0 if B or t < 7.6 else Lc["glow"] * 5200 + Lc["crack_flash"] * 15000
    L_sig.data.energy = 0.0 if B or t > 36 else Lc["sigma"] * 1800 + Lc["sigma_flare"] * 160000
    L_rimA.data.energy = 0.0 if B else Lc["rim"] * 26000
    L_rimB.data.energy = 0.0 if B else Lc["rim"] * 15000
    L_sun.data.energy = (Lc["sun_light"] * 2.6) if B else 0.0
    # emissives
    EM["sigma"].default_value = 6.0 + 260.0 * Lc["sigma_flare"]
    EM["bead"].default_value = 8.0 + 500.0 * Lc["sigma_flare"] + 220.0 * Lc["diamond"]
    EM["sun"].default_value = 18.0
    EM["corona"].default_value = 2.6 * Lc["tot"] + 0.3 * Lc["diamond"]
    EM["stars"].default_value = 9.0 * Lc["stars"]
    EM["mote"].default_value = 1.4
    for o in [o for o in SKY_OBJS if o.name.startswith("Stars")]:
        o.hide_render = (not B) or Lc["stars"] < 0.02
    # world
    mixAB.inputs[0].default_value = 1.0 if B else 0.0
    bgB.inputs["Strength"].default_value = 0.004 + 0.075 * Lc["ambient"]
    mixSky.inputs[0].default_value = float(min(1.0, Lc["horizon_glow"] * 1.15))


SKY_OBJS = []
A_OBJS = []
def _collect():
    for ob in bpy.data.objects:
        p = ob
        chain = []
        while p:
            chain.append(p.name); p = p.parent
        wx = ob.matrix_world.translation.x
        if ob.type in ("MESH", "EMPTY") and (wx > 500 or any(n.startswith("Sky") for n in chain)):
            SKY_OBJS.append(ob)
        elif ob.type in ("MESH", "EMPTY") and any(n.startswith(("Mechanism", "Lump", "Seabed", "Motes")) for n in chain):
            A_OBJS.append(ob)
_collect()
print("sky objs", len(SKY_OBJS), "A objs", len(A_OBJS))
_last = {"B": None}
def set_visibility(B):
    if _last["B"] == B:
        return
    for ob in SKY_OBJS:
        ob.hide_render = not B
    for ob in A_OBJS:
        ob.hide_render = B
    _last["B"] = B

# ----------------------------------------------------------------- go
print("rendering", len(FRAMES), "frame(s) at %dx%d, %d spp" % (args.w, args.h, args.spp))
t_all = time.time()
for n, f in enumerate(FRAMES):
    path = os.path.join(args.out, "%04d.hdr" % f)
    if os.path.exists(path):
        continue
    sc.frame_set(f)
    apply_state(f)
    bpy.context.view_layer.update()
    r.filepath = path
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print("frame %4d  (%d/%d)  %.1fs   elapsed %.0fs" % (f, n + 1, len(FRAMES), time.time() - t0, time.time() - t_all), flush=True)
