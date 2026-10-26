"""
Author the ANTIKYTHERA stage in OpenUSD.

Z-up, 24 fps, frames 1..1440 (t = (frame-1)/24 s).
Materials carry BOTH a UsdPreviewSurface network (universal) and a PxrSurface
network on `outputs:ri:surface` (Pixar RenderMan).  Lights are UsdLux.
"""
import sys, json
sys.path.insert(0, "/content/antikythera")
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf, Vt, UsdRender

import timeline as T
import geometry as G
import shots as S

OUT = "/content/antikythera/usd/antikythera.usda"
rng = np.random.default_rng(1901)           # the year the wreck was found

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetTimeCodesPerSecond(24)
stage.SetFramesPerSecond(24)
stage.SetStartTimeCode(1)
stage.SetEndTimeCode(T.NFRAMES)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
stage.GetRootLayer().documentation = (
    "ANTIKYTHERA - 60 s film. Gear train, tooth phases and every light/camera cue "
    "are procedural; see timeline.py / shots.py.")

FR = lambda t: 1 + t * T.FPS           # seconds -> timecode


# ------------------------------------------------------------ materials
MATS = {}

def make_material(name, color, metallic=0.0, rough=0.5, emit=None, emit_gain=0.0,
                  spec_edge=None, two_sided=False):
    p = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, p)
    # universal
    pv = UsdShade.Shader.Define(stage, p + "/Preview")
    pv.CreateIdAttr("UsdPreviewSurface")
    pv.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    pv.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    pv.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    if emit is not None:
        pv.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[c * emit_gain for c in emit]))
    pv.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(pv.ConnectableAPI(), "surface")
    # RenderMan
    px = UsdShade.Shader.Define(stage, p + "/Pxr")
    px.CreateIdAttr("PxrSurface")
    px.CreateInput("diffuseGain", Sdf.ValueTypeNames.Float).Set(0.0 if metallic > 0.5 else 0.8)
    px.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    px.CreateInput("specularModelType", Sdf.ValueTypeNames.Int).Set(1)
    px.CreateInput("specularFresnelMode", Sdf.ValueTypeNames.Int).Set(1)
    px.CreateInput("specularFaceColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(color if metallic > 0.5 else (0.04, 0.04, 0.04))))
    edge = spec_edge or (tuple(min(1.0, c * 1.25) for c in color) if metallic > 0.5 else (1, 1, 1))
    px.CreateInput("specularEdgeColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*edge))
    px.CreateInput("specularRoughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    if emit is not None:
        px.CreateInput("glowColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emit))
        px.CreateInput("glowGain", Sdf.ValueTypeNames.Float).Set(float(emit_gain))
    px.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateOutput("ri:surface", Sdf.ValueTypeNames.Token).ConnectToSource(
        px.ConnectableAPI(), "out")
    MATS[name] = mat
    return mat


UsdGeom.Scope.Define(stage, "/World/Looks")
make_material("Bronze",      (0.78, 0.52, 0.24), 1.0, 0.28)
make_material("BronzeDark",  (0.30, 0.22, 0.12), 1.0, 0.45)
make_material("Gold",        (1.00, 0.76, 0.33), 1.0, 0.18)
make_material("Crust",       (0.20, 0.27, 0.22), 0.0, 0.95)
make_material("Sand",        (0.36, 0.33, 0.26), 0.0, 0.95)
make_material("Silver",      (0.85, 0.87, 0.90), 1.0, 0.20)
make_material("MoonBlack",   (0.015, 0.015, 0.02), 0.0, 0.30)
make_material("SigmaGlow",   (1.0, 0.18, 0.06), 0.0, 0.4, emit=(1.0, 0.25, 0.08), emit_gain=6.0)
make_material("BeadGlow",    (1.0, 0.85, 0.5), 0.0, 0.4, emit=(1.0, 0.85, 0.55), emit_gain=8.0)
make_material("SunDisc",     (1.0, 0.9, 0.7), 0.0, 1.0, emit=(1.0, 0.82, 0.55), emit_gain=60.0)
make_material("MoonDisc",    (0.012, 0.012, 0.015), 0.0, 0.9)
make_material("Corona",      (1.0, 0.92, 0.8), 0.0, 1.0, emit=(0.95, 0.9, 1.0), emit_gain=9.0)
make_material("Sea",         (0.015, 0.03, 0.05), 0.0, 0.06)
make_material("Stone",       (0.07, 0.065, 0.06), 0.0, 0.8)
make_material("Star",        (1, 1, 1), 0.0, 1.0, emit=(0.85, 0.9, 1.0), emit_gain=4.0)
make_material("Mote",        (1, 0.95, 0.8), 0.0, 1.0, emit=(1.0, 0.9, 0.7), emit_gain=1.2)


def vertex_normals(m):
    n = np.zeros_like(m.pts)
    starts = np.concatenate([[0], np.cumsum(m.counts)])
    for f in range(len(m.counts)):
        ids = m.idx[starts[f]:starts[f + 1]]
        p = m.pts[ids]
        fn = np.cross(p[1] - p[0], p[2] - p[0])
        n[ids] += fn
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


def add_mesh(path, m, mat, smooth=False, double=False, parent_xform=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(m.pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(m.counts.astype(np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m.idx.astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(bool(double))
    lo, hi = m.pts.min(0), m.pts.max(0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    if smooth:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vertex_normals(m).astype(np.float32)))
        mesh.SetNormalsInterpolation("vertex")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(MATS[mat])
    return mesh


def xform(path):
    return UsdGeom.Xform.Define(stage, path)


def set_t(attr, val, t):
    attr.Set(val, Usd.TimeCode(FR(t)))


# =====================================================================
# 1.  THE MECHANISM
# =====================================================================
mech = xform("/World/Mechanism")
LAST_MECH_FRAME = int(FR(36.2))

# base plate + rim
plate = G.cylinder(25.0, 0.35, seg=180, z0=S.PLATE_Z - 0.35).translate((S.PLATE_C[0], S.PLATE_C[1], 0))
add_mesh("/World/Mechanism/Plate", plate, "BronzeDark")
rimring = G.annulus(24.1, 25.0, 0.18, seg=180, z0=S.PLATE_Z).translate((S.PLATE_C[0], S.PLATE_C[1], 0))
add_mesh("/World/Mechanism/PlateRim", rimring, "Bronze")

# dial ring around the main drive gear (zodiac / calendar)
dial = G.annulus(9.25, 10.9, 0.06, seg=240, z0=-0.40)
add_mesh("/World/Mechanism/DialRing", dial, "BronzeDark")
ticks = G.tick_marks(9.45, 10.15, 360, 30, 0.35, 0.03, 0.05, z0=-0.34)
add_mesh("/World/Mechanism/DialTicks", ticks, "Gold")
ticks2 = G.tick_marks(10.35, 10.7, 12, 1, 0.0, 0.14, 0.05, z0=-0.34)
add_mesh("/World/Mechanism/DialZodiac", ticks2, "Gold")

# arbors
arb = G.Mesh()
for gid in T.ORDER:
    g = T.TRAIN[gid]
    arb.merge(G.cylinder(0.12, g["z"] + 1.0 - (S.PLATE_Z), seg=12, z0=S.PLATE_Z)
              .translate((g["center"][0], g["center"][1], 0)))
add_mesh("/World/Mechanism/Arbors", arb, "Gold")

# gears
gear_xf = {}
for gid in T.ORDER:
    g = T.TRAIN[gid]
    gm = G.gear_mesh(g["N"], T.R_of(g["N"]), T.THICK, g["phi"])
    xf = xform(f"/World/Mechanism/Gear_{gid}")
    xf.AddTranslateOp().Set(Gf.Vec3d(g["center"][0], g["center"][1], g["z"]))
    rz = xf.AddRotateZOp()
    gear_xf[gid] = rz
    add_mesh(f"/World/Mechanism/Gear_{gid}/Mesh", gm, "Bronze" if gid not in ("G4", "G6") else "Gold")
    for f in range(1, LAST_MECH_FRAME + 1):
        t = (f - 1) / T.FPS
        rz.Set(float(np.degrees(T.gear_angle(gid, t))), Usd.TimeCode(f))

# drive key (star) riding on G1
key = G.Mesh()
key.merge(G.cylinder(0.9, 0.5, seg=32, z0=T.THICK))
for k in range(4):
    a = k * np.pi / 2
    key.merge(G.box((5.5, 0.55, 0.28), center=(3.1, 0, T.THICK + 0.25)).rotz(a))
    key.merge(G.uv_sphere(0.55, 16, 10).translate((5.9, 0, T.THICK + 0.25)).rotz(a))
add_mesh("/World/Mechanism/Gear_G1/DriveKey", key, "Gold")

# pointer on G6 axis (tied to the moon-phase ball)
ptr = G.Mesh()
ptr.merge(G.box((5.0, 0.18, 0.08), center=(2.1, 0, T.THICK + 0.18)))
ptr.merge(G.cylinder(0.32, 0.14, seg=20, z0=T.THICK + 0.1))
add_mesh("/World/Mechanism/Gear_G6/Pointer", ptr, "Gold")

# moon-phase ball: half silver / half black, driven by G6
BALL_C = (-9.5, 11.0, 0.7)
ball = xform("/World/Mechanism/MoonBall")
ball.AddTranslateOp().Set(Gf.Vec3d(*BALL_C))
bry = ball.AddRotateYOp()
add_mesh("/World/Mechanism/MoonBall/Silver", G.uv_sphere(1.5, 48, 24, hemi="+x"), "Silver", smooth=True)
add_mesh("/World/Mechanism/MoonBall/Black", G.uv_sphere(1.5, 48, 24, hemi="-x"), "MoonBlack", smooth=True)
ring = G.annulus(1.6, 1.95, 0.12, seg=64, z0=-0.06)
add_mesh("/World/Mechanism/MoonBallRing", ring.translate(BALL_C), "Gold")
for f in range(1, LAST_MECH_FRAME + 1):
    t = (f - 1) / T.FPS
    bry.Set(float(np.degrees(T.gear_angle("G6", t))), Usd.TimeCode(f))

# Saros spiral: 223 cells
sp = G.Mesh()
sig = G.Mesh()
for k in range(S.N_CELLS):
    xy, th, r = S.spiral_pos(k)
    cell = G.box((0.34, 0.2, 0.06), center=(0, 0, 0)).rotz(th + np.pi / 2)
    cell.translate((xy[0], xy[1], S.SPIRAL_Z))
    (sig if k == S.SIGMA_CELL else sp).merge(cell)
add_mesh("/World/Mechanism/SarosSpiral", sp, "Gold")
add_mesh("/World/Mechanism/SarosSigma", sig, "SigmaGlow")
disc = G.annulus(1.2, 8.6, 0.05, seg=160, z0=S.SPIRAL_Z - 0.08).translate((S.SPIRAL_C[0], S.SPIRAL_C[1], 0))
add_mesh("/World/Mechanism/SarosPlate", disc, "BronzeDark")

# slider arm + bead, driven by G5; hits the sigma cell exactly at t = 35.0
th5_end = abs(T.gear_angle("G5", 35.0) - T.gear_angle("G5", T.EV["first_tick"]))
arm = xform("/World/Mechanism/SarosArm")
arm_t = arm.AddTranslateOp()
arm_r = arm.AddRotateZOp()
arm_s = arm.AddScaleOp()
add_mesh("/World/Mechanism/SarosArm/Bar", G.box((1.0, 0.07, 0.05), center=(0.5, 0, 0)), "Gold")
bead = xform("/World/Mechanism/SarosBead")
bead_t = bead.AddTranslateOp()
bead_s = bead.AddScaleOp()
add_mesh("/World/Mechanism/SarosBead/Sphere", G.uv_sphere(0.17, 20, 12), "BeadGlow", smooth=True)

arm_t.Set(Gf.Vec3d(S.SPIRAL_C[0], S.SPIRAL_C[1], S.SPIRAL_Z + 0.12))
for f in range(1, LAST_MECH_FRAME + 1):
    t = (f - 1) / T.FPS
    prog = abs(T.gear_angle("G5", t) - T.gear_angle("G5", T.EV["first_tick"])) / th5_end if t > T.EV["first_tick"] else 0.0
    prog = float(np.clip(prog, 0, 1))
    k = S.SIGMA_CELL * prog
    xy, th, r = S.spiral_pos(k)
    d = xy - S.SPIRAL_C
    L = float(np.linalg.norm(d)) + 1e-6
    arm_r.Set(float(np.degrees(np.arctan2(d[1], d[0]))), Usd.TimeCode(f))
    arm_s.Set(Gf.Vec3d(L, 1, 1), Usd.TimeCode(f))
    bead_t.Set(Gf.Vec3d(xy[0], xy[1], S.SPIRAL_Z + 0.16), Usd.TimeCode(f))
    # bead swells as it approaches the eclipse cell
    sc = 1.0 + 1.8 * float(S.smooth((t - 33.0) / 3.0))
    bead_s.Set(Gf.Vec3d(sc, sc, sc), Usd.TimeCode(f))

# =====================================================================
# 2.  THE ENCRUSTED LUMP (1901) + SHARDS + SEABED + DUST
# =====================================================================
lump_mesh, dirs = G.lump(S.LUMP_C, S.LUMP_R, nu=160, nv=80, bump=3.2, seed=7)
fc = G.face_centroids(lump_mesh)
rel = (fc - S.LUMP_C) / S.LUMP_R
upper = rel[:, 2] > -0.05
add_mesh("/World/Lump/Base", G.submesh(lump_mesh, np.where(~upper)[0]), "Crust", smooth=True, double=True)
xform("/World/Lump")

N_SH = 18
seeds = []
while len(seeds) < N_SH:
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    if v[2] > 0.0:
        seeds.append(v)
seeds = np.array(seeds)
up_ids = np.where(upper)[0]
dirn = rel[up_ids] / np.linalg.norm(rel[up_ids], axis=1, keepdims=True)
owner = np.argmax(dirn @ seeds.T, axis=1)
shard_info = []
for i in range(N_SH):
    ids = up_ids[owner == i]
    if len(ids) == 0:
        continue
    sm = G.submesh(lump_mesh, ids)
    cen = sm.pts.mean(0)
    sm.pts = sm.pts - cen
    xf = xform(f"/World/Lump/Shard_{i:02d}")
    to = xf.AddTranslateOp()
    ro = xf.AddRotateXYZOp()
    so = xf.AddScaleOp()
    add_mesh(f"/World/Lump/Shard_{i:02d}/Mesh", sm, "Crust", smooth=True, double=True)
    out = (cen - S.LUMP_C)
    out = out / np.linalg.norm(out)
    # horizontal-biased burst, slightly up, with per-shard timing jitter
    vel = out * rng.uniform(9, 17) + np.array([0, 0, rng.uniform(4, 9)])
    spin = rng.normal(size=3) * rng.uniform(40, 160)
    t0 = T.EV["crack"] + rng.uniform(0.0, 0.18)
    for f in range(1, int(FR(10.4)) + 1):
        t = (f - 1) / T.FPS
        tau = max(t - t0, 0.0)
        pos = cen + vel * tau * np.exp(-tau * 0.35) - np.array([0, 0, 0.5 * 9.8 * tau * tau * 0.35])
        to.Set(Gf.Vec3d(*pos), Usd.TimeCode(f))
        ro.Set(Gf.Vec3f(*(spin * tau)), Usd.TimeCode(f))
        s = 1.0 if t < T.EV["shards_end"] else float(max(0.0, 1.0 - (t - T.EV["shards_end"]) / 0.8))
        so.Set(Gf.Vec3d(s, s, s), Usd.TimeCode(f))
    shard_info.append(dict(i=i, t0=float(t0), n=int(len(ids)),
                           speed=float(np.linalg.norm(vel))))

# seabed
xs = np.linspace(-240, 240, 120)
ys = np.linspace(-240, 240, 120)
def ripple(X, Y):
    r = np.hypot(X - S.LUMP_C[0], Y - S.LUMP_C[1])
    return S.GROUND_Z + 0.35 * np.sin(X * 0.45 + 0.3 * np.sin(Y * 0.11)) * np.exp(-r / 220) \
        + 0.25 * np.sin(Y * 0.31 + X * 0.07)
sea_floor = G.grid_surface(xs, ys, ripple)
add_mesh("/World/Seabed/Floor", sea_floor, "Sand", smooth=True, double=True)
rocks = G.Mesh()
for k in range(40):
    a = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(34, 140)
    sx = rng.uniform(1.2, 4.5)
    rk = G.uv_sphere(1.0, 14, 8)
    rk.pts = rk.pts * np.array([sx, sx * rng.uniform(.6, 1.2), sx * rng.uniform(.4, .8)])
    rk.translate((r * np.cos(a) + S.LUMP_C[0], r * np.sin(a) + S.LUMP_C[1], S.GROUND_Z + 0.2))
    rocks.merge(rk)
add_mesh("/World/Seabed/Rocks", rocks, "Crust", smooth=False, double=True)

# dust motes / suspended particles
for k in range(90):
    p0 = rng.uniform(-1, 1, 3) * np.array([46, 46, 20]) + np.array([0, 0, 6])
    drift = rng.normal(size=3) * np.array([0.25, 0.25, 0.18]) + np.array([0, 0, 0.12])
    wob = rng.uniform(0.2, 1.1)
    ph = rng.uniform(0, 6.28)
    size = rng.uniform(0.05, 0.16)
    xf = xform(f"/World/Motes/Mote_{k:02d}")
    to = xf.AddTranslateOp()
    xf.AddScaleOp().Set(Gf.Vec3d(size, size, size))
    add_mesh(f"/World/Motes/Mote_{k:02d}/Mesh", G.uv_sphere(1.0, 8, 6), "Mote", smooth=True)
    for f in range(1, LAST_MECH_FRAME + 1):
        t = (f - 1) / T.FPS
        pos = p0 + drift * t + wob * np.array([np.sin(t * .9 + ph), np.cos(t * .7 + ph), np.sin(t * .5 + 2 * ph)])
        to.Set(Gf.Vec3d(*pos), Usd.TimeCode(f))

# =====================================================================
# 3.  THE SKY / ECLIPSE SET   (origin x = 1000)
# =====================================================================
O = S.SKY_O
sky = xform("/World/Sky")
# aegean sea
sx = np.linspace(-3500, 3500, 200)
sy = np.linspace(40, 6000, 200)
sea = G.grid_surface(sx, sy, lambda X, Y: -1.2 + 0.06 * np.sin(X * 0.21 + Y * 0.05) + 0.04 * np.sin(Y * 0.37 - X * 0.09))
sea.translate((O[0], 0, 0))
add_mesh("/World/Sky/Sea", sea, "Sea", smooth=True, double=True)
# stone terrace + two columns (silhouettes framing the eclipse)
add_mesh("/World/Sky/Terrace", G.box((90, 70, 3.2), center=(O[0], 38, -1.8)), "Stone")
for sgn in (-1, 1):
    shaft = G.cylinder(1.5, 15.0, seg=28, z0=-0.2, r_top=1.2)
    shaft.translate((O[0] + sgn * 17.0, 46.0, 0))
    add_mesh(f"/World/Sky/Column_{'L' if sgn < 0 else 'R'}", shaft, "Stone", smooth=True)
    cap = G.box((4.4, 4.4, 1.0), center=(O[0] + sgn * 17.0, 46.0, 15.4))
    add_mesh(f"/World/Sky/Capital_{'L' if sgn < 0 else 'R'}", cap, "Stone")
    base = G.box((4.6, 4.6, 0.8), center=(O[0] + sgn * 17.0, 46.0, 0.2))
    add_mesh(f"/World/Sky/Base_{'L' if sgn < 0 else 'R'}", base, "Stone")

# stars
st = G.Mesh()
for k in range(260):
    az = rng.uniform(-1.3, 1.3)
    el = rng.uniform(np.radians(6), np.radians(80))
    d = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    pos = S.SKY_CAM + d * 5600
    rr = rng.uniform(3.5, 11.0)
    st.merge(G.uv_sphere(rr, 6, 4).translate(pos))
add_mesh("/World/Sky/Stars", st, "Star")


def sky_dir(az, el):
    return np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])

# sun
sun_c = S.SKY_CAM + S.SUN_DIST * sky_dir(0, S.SUN_EL)
sun_r = S.SUN_DIST * np.tan(S.SUN_ANG_R)
sunx = xform("/World/Sky/Sun")
sunx.AddTranslateOp().Set(Gf.Vec3d(*sun_c))
add_mesh("/World/Sky/Sun/Disc", G.uv_sphere(sun_r, 64, 32), "SunDisc", smooth=True)

# corona = a 223-tooth gear, emissive, spinning behind the moon
Rc = sun_r * S.CORONA_SCALE
n_cam = -sky_dir(0, S.SUN_EL)                  # faces the camera
rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*n_cam))
M = Gf.Matrix4d().SetRotate(rot)
M.SetTranslateOnly(Gf.Vec3d(*(sun_c - 30 * n_cam * -1 * 0)))     # centred on the sun
cor = xform("/World/Sky/Corona")
cor.AddTransformOp().Set(M)
cor_spin = cor.AddRotateZOp()
tooth = Rc * 0.045
cm = G.Mesh()
Nc = 223
outl, ang, _ = G.toothed_outline(Nc, Rc, 0.0, add=tooth, ded=tooth)
Mo = len(outl)
r_in = sun_r * 0.97
inn = np.stack([r_in * np.cos(ang), r_in * np.sin(ang)], 1)
thk = sun_r * 0.05
Pc = np.vstack([np.c_[outl, np.full(Mo, 0.0)], np.c_[outl, np.full(Mo, thk)],
                np.c_[inn, np.full(Mo, thk)], np.c_[inn, np.full(Mo, 0.0)]])
ii = np.arange(Mo); nn = (ii + 1) % Mo
cnt = []; idx = []
for qa, qb, qc, qd in [(ii, nn, nn + Mo, ii + Mo), (ii + Mo, nn + Mo, nn + 2 * Mo, ii + 2 * Mo),
                       (ii + 2 * Mo, nn + 2 * Mo, nn + 3 * Mo, ii + 3 * Mo), (ii + 3 * Mo, nn + 3 * Mo, nn, ii)]:
    cnt.append(np.full(len(qa), 4)); idx.append(np.stack([qa, qb, qc, qd], 1).reshape(-1))
cor_mesh = G.Mesh(Pc, np.concatenate(cnt), np.concatenate(idx)).translate((0, 0, -thk * 4))
add_mesh("/World/Sky/Corona/Gear223", cor_mesh, "Corona")

# moon (slightly nearer & larger than the sun)
moon_d = S.SUN_DIST * 0.995
moon_r = moon_d * np.tan(S.MOON_ANG_R)
moonx = xform("/World/Sky/Moon")
moon_t = moonx.AddTranslateOp()
add_mesh("/World/Sky/Moon/Sphere", G.uv_sphere(moon_r, 64, 32), "MoonDisc", smooth=True)

# diamond-ring bead
beadx = xform("/World/Sky/DiamondBead")
bd_t = beadx.AddTranslateOp()
bd_s = beadx.AddScaleOp()
add_mesh("/World/Sky/DiamondBead/Sphere", G.uv_sphere(sun_r * 0.12, 16, 10), "BeadGlow", smooth=True)

right = np.array([1.0, 0, 0])
for f in range(int(FR(36.0)), T.NFRAMES + 1):
    t = (f - 1) / T.FPS
    off = float(T.moon_offset(t))
    az_m = off * S.SUN_ANG_R * 1.0
    el_m = S.SUN_EL + off * S.SUN_ANG_R * 0.35
    mpos = S.SKY_CAM + moon_d * sky_dir(az_m, el_m)
    moon_t.Set(Gf.Vec3d(*mpos), Usd.TimeCode(f))
    cor_spin.Set(float(np.degrees(0.35 * max(0.0, t - 52.0) + 0.05 * (t - 36))), Usd.TimeCode(f))
    # bead sits on the limb opposite the approaching/receding moon
    side = 1.0 if t < 53.75 else -1.0
    L = S.lights(t)
    ba = np.radians(-14.0) if side > 0 else np.radians(194.0)
    # bead on a circle (in the view plane) of radius sun_r, around the sun
    plane_r = np.cross(n_cam, [0, 0, 1.0]); plane_r /= np.linalg.norm(plane_r)
    plane_u = np.cross(plane_r, n_cam)
    bpos = sun_c + sun_r * (np.cos(ba) * plane_r + np.sin(ba) * plane_u) + n_cam * sun_r * 0.5
    bd_t.Set(Gf.Vec3d(*bpos), Usd.TimeCode(f))
    sc = 0.001 + 1.5 * L["diamond"]
    bd_s.Set(Gf.Vec3d(sc, sc, sc), Usd.TimeCode(f))

# =====================================================================
# 4.  LIGHTS (UsdLux)
# =====================================================================
UsdGeom.Scope.Define(stage, "/World/Lights")
# water-filtered shaft of light (teal)
shaft_l = UsdLux.DistantLight.Define(stage, "/World/Lights/Shaft")
shaft_l.CreateColorAttr(Gf.Vec3f(0.55, 0.85, 0.85))
shaft_l.CreateIntensityAttr(2.2)
shaft_l.CreateAngleAttr(4.0)
xf = UsdGeom.Xformable(shaft_l)
xf.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(0.35, 0.2, -0.9).GetNormalized())))
# warm crack glow: lives inside the lump
glow = UsdLux.SphereLight.Define(stage, "/World/Lights/CrackGlow")
glow.CreateRadiusAttr(1.2)
glow.CreateColorAttr(Gf.Vec3f(1.0, 0.72, 0.38))
glow.AddTranslateOp().Set(Gf.Vec3d(1.0, 1.0, 6.0))
inten = glow.CreateIntensityAttr()
sig_l = UsdLux.SphereLight.Define(stage, "/World/Lights/SigmaLight")
sig_l.CreateRadiusAttr(0.6)
sig_l.CreateColorAttr(Gf.Vec3f(1.0, 0.25, 0.1))
sig_l.AddTranslateOp().Set(Gf.Vec3d(S.spiral_pos(S.SIGMA_CELL)[0][0], S.spiral_pos(S.SIGMA_CELL)[0][1], 1.2))
sinten = sig_l.CreateIntensityAttr()
rim_a = UsdLux.RectLight.Define(stage, "/World/Lights/RimWarm")
rim_a.CreateWidthAttr(30); rim_a.CreateHeightAttr(8)
rim_a.CreateColorAttr(Gf.Vec3f(1.0, 0.6, 0.3))
rim_a.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(0.6, 0.5, -0.35).GetNormalized())).SetTranslateOnly(Gf.Vec3d(-34, -30, 22)))
rim_b = UsdLux.RectLight.Define(stage, "/World/Lights/RimCool")
rim_b.CreateWidthAttr(30); rim_b.CreateHeightAttr(8)
rim_b.CreateColorAttr(Gf.Vec3f(0.4, 0.7, 1.0))
rim_b.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(-0.6, -0.5, -0.4).GetNormalized())).SetTranslateOnly(Gf.Vec3d(34, 30, 22)))
rinten = rim_b.CreateIntensityAttr(); ainten = rim_a.CreateIntensityAttr()
# eclipse sun light + dome
sunl = UsdLux.DistantLight.Define(stage, "/World/Lights/SunLight")
sunl.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.6))
sunl.CreateAngleAttr(0.5)
sunl.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*(-sky_dir(0, S.SUN_EL))))))
sl_i = sunl.CreateIntensityAttr()
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
dome.CreateColorAttr(Gf.Vec3f(0.12, 0.2, 0.4))
d_i = dome.CreateIntensityAttr()

for f in range(1, T.NFRAMES + 1):
    t = (f - 1) / T.FPS
    L = S.lights(t)
    tc = Usd.TimeCode(f)
    inten.Set(float(L["glow"] * 4000 + L["crack_flash"] * 60000), tc)
    sinten.Set(float((L["sigma"] * 900 + L["sigma_flare"] * 90000) if t < 36 else 0), tc)
    rinten.Set(float(L["rim"] * 0.7), tc)
    ainten.Set(float(L["rim"] * 0.9), tc)
    sl_i.Set(float(L["sun_light"] * 6.0 if t >= 36 else 0.0), tc)
    d_i.Set(float(L["ambient"] * 0.7 if t >= 36 else 0.04), tc)

# =====================================================================
# 5.  CAMERA
# =====================================================================
cam = UsdGeom.Camera.Define(stage, "/World/Camera")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(20.25)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 20000.0))
cop = cam.AddTransformOp()
fl = cam.CreateFocalLengthAttr(); fs = cam.CreateFStopAttr(); fd = cam.CreateFocusDistanceAttr()
sidecar = dict(fps=T.FPS, nframes=T.NFRAMES, frames=[])
for f in range(1, T.NFRAMES + 1):
    t = (f - 1) / T.FPS
    c = S.camera_state(t)
    P, Tg = c["pos"], c["tgt"]
    fwd = Tg - P; fwd /= np.linalg.norm(fwd)
    upw = np.array([0, 0, 1.0])
    r = np.cross(fwd, upw); r /= np.linalg.norm(r)
    u = np.cross(r, fwd)
    a = np.radians(c["roll"])
    r, u = r * np.cos(a) + u * np.sin(a), -r * np.sin(a) + u * np.cos(a)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(*r, 0)); m.SetRow(1, Gf.Vec4d(*u, 0))
    m.SetRow(2, Gf.Vec4d(*(-fwd), 0)); m.SetRow(3, Gf.Vec4d(*P, 1))
    tc = Usd.TimeCode(f)
    cop.Set(m, tc); fl.Set(float(c["f"]), tc); fs.Set(float(c["fstop"]), tc); fd.Set(float(c["focus"]), tc)
    L = S.lights(t)
    sidecar["frames"].append(dict(
        f=f, t=round(t, 5), scene=c["scene"],
        pos=[round(float(x), 4) for x in P], tgt=[round(float(x), 4) for x in Tg],
        focal=round(c["f"], 3), fstop=round(c["fstop"], 3), focus=round(c["focus"], 3),
        roll=round(c["roll"], 4), L={k: round(float(v), 5) for k, v in L.items()}))

# =====================================================================
# 6.  RENDER SETTINGS (RenderMan-ready)
# =====================================================================
rs = UsdRender.Settings.Define(stage, "/Render/Settings")
rs.CreateResolutionAttr(Gf.Vec2i(1920, 1080))
rs.CreateCameraRel().SetTargets(["/World/Camera"])
rs.CreatePixelAspectRatioAttr(1.0)
rs.GetPrim().CreateAttribute("ri:hider:maxsamples", Sdf.ValueTypeNames.Int).Set(256)
rs.GetPrim().CreateAttribute("ri:hider:minsamples", Sdf.ValueTypeNames.Int).Set(32)
rs.GetPrim().CreateAttribute("ri:hider:incremental", Sdf.ValueTypeNames.Int).Set(1)
rs.GetPrim().CreateAttribute("ri:Ri:Shutter", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(-0.25, 0.25))
rs.GetPrim().CreateAttribute("ri:integrator:name", Sdf.ValueTypeNames.String).Set("PxrPathTracer")
rs.GetPrim().CreateAttribute("ri:integrator:PxrPathTracer:maxPathLength", Sdf.ValueTypeNames.Int).Set(8)

stage.GetRootLayer().Save()
json.dump(sidecar, open("/content/antikythera/usd/shots.json", "w"))
json.dump(dict(shards=shard_info, train={k: dict(N=v["N"], center=list(map(float, v["center"])),
          z=v["z"], ratio=float(v["ratio"])) for k, v in T.TRAIN.items()}),
          open("/content/antikythera/usd/info.json", "w"), indent=1)
print("saved", OUT)
