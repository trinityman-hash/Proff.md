"""Camera, lighting & eclipse state as pure functions of time.
Used by build_usd.py (authoring) and render_blender.py (via the sidecar JSON)."""
import numpy as np
from scipy.interpolate import PchipInterpolator
import timeline as T

# ----------------------------------------------------------- world layout
LUMP_C = np.array([0.0, 0.5, 0.0])
LUMP_R = np.array([28.0, 28.0, 9.0])
GROUND_Z = -9.4
PLATE_C = np.array([0.0, 0.5])
PLATE_Z = -0.70
SPIRAL_C = np.array([14.0, -10.0])
SPIRAL_Z = -0.45
SIGMA_CELL = 168
N_CELLS = 223
SKY_O = np.array([1000.0, 0.0, 0.0])      # origin of the sky/eclipse set
SKY_CAM = SKY_O + np.array([0.0, 0.0, 1.7])
SUN_DIST = 4000.0
SUN_EL = np.radians(20.0)
SUN_ANG_R = np.radians(1.6)               # artistic: larger than reality
MOON_ANG_R = SUN_ANG_R * 1.035
CORONA_SCALE = 3.3


def spiral_pos(k):
    th = k * (4 * 2 * np.pi / N_CELLS)
    r = 2.0 + 5.0 * k / N_CELLS
    return SPIRAL_C + r * np.array([np.cos(th), np.sin(th)]), th, r


# ----------------------------------------------------------- camera nodes
# t, az(deg), el(deg), dist, target(x,y,z), focal(mm), f-stop
_SIGMA_XY = spiral_pos(SIGMA_CELL)[0]
A_NODES = [
    (0.00, -128, 6.0, 112, (0, .5, 3.0), 35, 8.0),
    (4.00, -124, 7.0, 78, (0, .5, 3.0), 38, 8.0),
    (7.50, -120, 8.0, 62, (0, .5, 3.0), 42, 8.0),
    (8.00, -120, 8.0, 61, (0, .5, 3.0), 42, 8.0),
    (8.70, -100, 38.0, 58, (0, .5, 1.5), 30, 5.6),
    (9.80, -80, 62.0, 56, (0, .5, 0.8), 30, 5.6),
    (11.0, -75, 66.0, 50, (0, .5, 0.8), 32, 5.6),
    (14.0, -40, 58.0, 44, (0, .5, 0.8), 34, 5.6),
    (18.0, 10, 52.0, 40, (1, 1, 0.8), 36, 4.0),
    (19.9, 38, 40.0, 30, (4, 2, 0.6), 40, 3.2),
    # macro: the G1/G2 tooth mesh
    (20.0, 62, 22.0, 15.0, (8.4, 3.1, 0.5), 60, 1.8),
    (23.5, 85, 18.0, 11.5, (8.4, 3.1, 0.5), 60, 1.8),
    (26.9, 108, 14.0, 9.5, (8.4, 3.1, 0.5), 60, 1.8),
    (27.0, 125, 48.0, 48, (2, 0, 0.6), 32, 5.6),
    (30.0, 150, 58.0, 40, (3, -1, 0.6), 34, 5.6),
    # dive towards the Saros spiral
    (32.0, 172, 66.0, 30, (9, -6, 0.0), 36, 4.0),
    (34.0, 190, 74.0, 16, (_SIGMA_XY[0] * .6 + 14 * .4, _SIGMA_XY[1] * .6 - 10 * .4, -0.2), 40, 2.8),
    (35.0, 200, 80.0, 8.5, (_SIGMA_XY[0], _SIGMA_XY[1], -0.4), 45, 2.2),
    (35.9, 205, 86.0, 2.6, (_SIGMA_XY[0], _SIGMA_XY[1], -0.4), 50, 2.0),
]
# sky camera: t, pitch(deg), focal
B_NODES = [
    (36.0, 11.0, 26),
    (44.0, 14.0, 40),
    (51.0, 17.0, 70),
    (52.0, 17.3, 78),
    (55.5, 18.0, 80),
    (58.0, 30.0, 42),
    (60.0, 34.0, 36),
]


def _interp(nodes, col):
    ts = [n[0] for n in nodes]
    return PchipInterpolator(ts, [n[col] for n in nodes])


_ia = {k: _interp(A_NODES, c) for k, c in dict(az=1, el=2, dist=3, f=5, fs=6).items()}
_iat = [PchipInterpolator([n[0] for n in A_NODES], [n[4][i] for n in A_NODES]) for i in range(3)]
_ib = {k: _interp(B_NODES, c) for k, c in dict(pitch=1, f=2).items()}


def _shake(t, base, bursts):
    """returns (dyaw, dpitch, droll) in degrees."""
    amp = base
    for t0, a, tau in bursts:
        if t >= t0:
            amp += a * np.exp(-(t - t0) / tau)
    ramp = np.clip((t - 7.0) / 1.0, 0, 1) * np.clip((8.0 - t) / 0.0001, 0, 1) * 0.7 if t < 8.0 else 0
    amp += ramp
    s = lambda f, p: np.sin(2 * np.pi * f * t + p)
    return (amp * (s(7.3, 0.1) + 0.6 * s(13.1, 1.3)),
            amp * (s(8.9, 2.1) + 0.5 * s(15.7, 0.4)),
            amp * 0.6 * s(6.1, 0.9))


def camera_state(t):
    """Returns dict(pos, tgt, f, fstop, focus, roll_deg, scene)."""
    if t < 36.0:
        az = np.radians(float(_ia["az"](t)))
        el = np.radians(float(_ia["el"](t)))
        d = float(_ia["dist"](t))
        tgt = np.array([float(_iat[i](t)) for i in range(3)])
        pos = tgt + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        f = float(_ia["f"](t))
        fs = float(_ia["fs"](t))
        dy, dp, dr = _shake(t, 0.035, [(8.0, 2.6, 0.45), (10.0, 0.25, 0.8)])
        scene = "A"
    else:
        pitch = np.radians(float(_ib["pitch"](t)))
        pos = SKY_CAM.copy()
        fwd = np.array([0, np.cos(pitch), np.sin(pitch)])
        d = 200.0
        tgt = pos + d * fwd
        f = float(_ib["f"](t))
        fs = 11.0
        dy, dp, dr = _shake(t, 0.025, [(52.0, 1.3, 0.55), (55.5, 0.6, 0.5), (36.0, 0.9, 0.3)])
        scene = "B"
    # apply shake as small angular offset of the aim point
    fwd = tgt - pos
    L = np.linalg.norm(fwd)
    fwd /= L
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    tgt = pos + L * (fwd + np.tan(np.radians(dy)) * right + np.tan(np.radians(dp)) * up)
    focus = np.linalg.norm(tgt - pos) if scene == "A" else 3500.0
    return dict(pos=pos, tgt=tgt, f=f, fstop=fs, focus=float(focus), roll=float(dr), scene=scene)


# ----------------------------------------------------------- lighting cues
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def circle_overlap_cov(d, rs=1.0, rm=T.MOON_R if hasattr(T, "MOON_R") else 1.035):
    """Fraction of the sun's disc covered by the moon (d = centre distance)."""
    d = abs(d)
    if d >= rs + rm:
        return 0.0
    if d <= abs(rm - rs):
        return 1.0 if rm >= rs else (rm / rs) ** 2
    a1 = np.arccos((d * d + rs * rs - rm * rm) / (2 * d * rs))
    a2 = np.arccos((d * d + rm * rm - rs * rs) / (2 * d * rm))
    area = rs * rs * a1 + rm * rm * a2 - 0.5 * np.sqrt(
        (-d + rs + rm) * (d + rs - rm) * (d - rs + rm) * (d + rs + rm))
    return float(area / (np.pi * rs * rs))


def eclipse_state(t):
    off = float(T.moon_offset(t))
    cov = circle_overlap_cov(off, 1.0, 1.035)
    return off, cov


def lights(t):
    """Normalised cues (0..1+) consumed by the Blender render script."""
    D = {}
    # --- mechanism set ---
    D["shaft"] = 1.0                                   # underwater god-ray key
    D["glow"] = float(                                  # warm light released by the crack
        smooth((t - 7.6) / 0.4) * (1.0 if t < 8.0 else 1.0) * (6.0 * np.exp(-(max(t, 8.0) - 8.0) / 0.5) + 0.9)
    ) if t >= 7.6 else 0.0
    D["rim"] = float(smooth((t - 9.0) / 2.0))           # cool/warm rim lights come up after reveal
    # sigma cell + bead glow builds to the whiteout
    D["sigma"] = float(0.25 + 0.75 * smooth((t - 33.0) / 2.5)) if t > 10 else 0.0
    D["sigma_flare"] = float(np.clip(smooth((t - 35.0) / 0.9) ** 2.2, 0, 1)) if 35 <= t < 36.0 else 0.0
    # --- whiteout ---
    flash = 0.0
    if 35.55 <= t < 36.0:
        flash = float(smooth((t - 35.55) / 0.45))
    elif 36.0 <= t < 37.6:
        flash = float(1.0 - smooth((t - 36.0) / 1.6))
    D["flash"] = flash
    # crack flash (brief)
    D["crack_flash"] = float(np.exp(-abs(t - 8.0) / 0.06)) if abs(t - 8.0) < 0.5 else 0.0
    # --- sky set ---
    off, cov = eclipse_state(max(t, 36.0))
    D["moon_off"] = off
    D["cov"] = cov
    tot = float(smooth((cov - 0.985) / 0.015)) if t >= 36 else 0.0     # ~1 in totality
    D["tot"] = tot
    D["sun_light"] = float((1.0 - cov) ** 1.1) if t >= 36 else 1.0
    D["ambient"] = float(0.18 + 0.82 * (1.0 - smooth(cov * 1.05)))
    D["horizon_glow"] = float(smooth((cov - 0.55) / 0.4) * 0.9 + tot * 0.4)
    D["stars"] = float(smooth((cov - 0.80) / 0.2))
    D["corona"] = float(tot)
    # diamond ring beads
    b = 0.0
    if 50.2 <= t <= 52.0:
        b = float(smooth((t - 50.2) / 0.8) * (1 - smooth((t - 51.6) / 0.4)))
    if 55.5 <= t <= 57.0:
        b = max(b, float(smooth((t - 55.5) / 0.08) * (1 - smooth((t - 55.7) / 1.3))))
    D["diamond"] = b
    D["fade"] = float(smooth((t - 57.6) / 1.4)) if t >= 57.6 else 0.0   # fade to black
    return D


if __name__ == "__main__":
    for tt in [0, 7.9, 8.0, 12, 24, 35.9, 36.0, 44, 51, 52, 54, 55.5, 58]:
        c = camera_state(tt)
        L = lights(tt)
        print("t=%5.1f  %s pos=(%7.1f,%7.1f,%6.1f) f=%4.0f  cov=%.3f tot=%.2f sun=%.2f flash=%.2f"
              % (tt, c["scene"], *c["pos"], c["f"], L["cov"], L["tot"], L["sun_light"], L["flash"]))
