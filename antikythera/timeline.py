"""
Shared timeline for ANTIKYTHERA.
Both the OpenUSD scene builder and the audio engine import this file, so every
SFX/music hit is computed from the same numbers that drive the animation.

Tempo: 120 BPM, 4/4  ->  beat = 0.5 s, bar = 2.0 s.  Every major picture event
below lands exactly on the beat grid.
"""
import numpy as np

FPS = 24
DUR = 60.0
NFRAMES = int(FPS * DUR)          # 1440
BPM = 120
BEAT = 60.0 / BPM                 # 0.5
BAR = 4 * BEAT                    # 2.0

# ---------------------------------------------------------------- key events
EV = dict(
    sonar1=1.0, sonar2=5.0,
    riser_start=6.0,
    crack_pre=7.5,
    crack=8.0,            # lump bursts open
    shards_end=9.5,
    camera_swing=8.0,     # camera whips around to reveal the mechanism
    first_tick=10.0,      # crank starts, G1 starts to turn
    pulse_in=16.0,        # bass pulse / full rhythm
    build=30.0,           # snare build
    sigma_lock=35.0,      # slider reaches the eclipse cell
    flash=36.0,           # whiteout, cut to sky
    sky_in=36.0,
    sun_pad=37.0,
    totality_riser=44.0,
    diamond_in=51.0,
    totality=52.0,
    diamond_out=55.5,
    fade=58.0,
    end=60.0,
)

# ---------------------------------------------------------------- gear train
MODULE = 0.08
THICK = 0.22

def R_of(N):
    return N * MODULE / 2.0

# id: (teeth, meshed_with, mesh_angle_deg, coaxial_with, z)
GEARS = {
    "G1": dict(N=223, mesh=None,  ang=0,   coax=None, z=0.00, center=(0.0, 0.0)),
    "G2": dict(N=64,  mesh="G1",  ang=20,  coax=None, z=0.00),
    "G3": dict(N=38,  mesh=None,  ang=0,   coax="G2", z=0.55),
    "G4": dict(N=48,  mesh="G3",  ang=100, coax=None, z=0.55),
    "G5": dict(N=24,  mesh=None,  ang=0,   coax="G4", z=1.10),
    "G6": dict(N=127, mesh="G5",  ang=150, coax=None, z=1.10),
    "G7": dict(N=96,  mesh="G1",  ang=205, coax=None, z=0.00),
    "G8": dict(N=30,  mesh=None,  ang=0,   coax="G7", z=0.55),
    "G9": dict(N=100, mesh="G8",  ang=255, coax=None, z=0.55),
}
ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]


def solve_train():
    """Positions, mesh phases and angular-velocity ratios (w / w1)."""
    out = {}
    for gid in ORDER:
        g = GEARS[gid]
        N = g["N"]
        if g["coax"]:
            p = out[g["coax"]]
            out[gid] = dict(N=N, center=p["center"], z=g["z"], ratio=p["ratio"],
                            phi=p["phi"])
        elif g["mesh"]:
            p = out[g["mesh"]]
            th = np.radians(g["ang"])
            d = R_of(p["N"]) + R_of(N)
            cx = p["center"][0] + d * np.cos(th)
            cy = p["center"][1] + d * np.sin(th)
            pA = 2 * np.pi / p["N"]
            pB = 2 * np.pi / N
            # nearest tooth offset of parent from the line of centres
            eps = ((p["phi"] - th + pA / 2) % pA) - pA / 2
            phi = (th + np.pi - eps * p["N"] / N + pB / 2) % pB
            out[gid] = dict(N=N, center=(cx, cy), z=g["z"],
                            ratio=-p["ratio"] * p["N"] / N, phi=phi)
        else:
            out[gid] = dict(N=N, center=g["center"], z=g["z"], ratio=1.0, phi=0.0)
    return out


TRAIN = solve_train()

# ------------------------------------------------- driving angular velocity
def omega1(t):
    """rad/s of the main drive gear G1 as a function of time (s)."""
    t = np.asarray(t, dtype=float)
    pts_t = [0.0, 10.0, 10.0, 16.0, 24.0, 30.0, 36.0, 60.0]
    pts_w = [0.0, 0.0,  0.0,  0.9,  2.0,  3.2,  5.0,  5.0]
    # smooth-ish: monotone cubic through the knots
    from scipy.interpolate import PchipInterpolator
    # remove duplicate x
    xs = [0.0, 10.0, 16.0, 24.0, 30.0, 36.0, 60.0]
    ys = [0.0, 0.0,  0.45, 1.1,  1.9,  3.4,  3.4]
    f = PchipInterpolator(xs, ys)
    return f(np.clip(t, 0, 60))


_HI = 4800  # samples / s for accurate integration & tick times
_t_hi = np.arange(0, int(DUR * _HI) + 1) / _HI
_w_hi = omega1(_t_hi)
THETA1_HI = np.concatenate([[0], np.cumsum((_w_hi[1:] + _w_hi[:-1]) * 0.5 / _HI)])


def theta1(t):
    return np.interp(t, _t_hi, THETA1_HI)


def gear_angle(gid, t):
    return TRAIN[gid]["ratio"] * theta1(t)


def tooth_ticks(gid="G1", t_max=36.0):
    """Exact times at which a tooth of `gid` passes the contact line."""
    th = np.abs(TRAIN[gid]["ratio"]) * THETA1_HI
    pitch = 2 * np.pi / TRAIN[gid]["N"]
    k = np.floor(th / pitch).astype(int)
    idx = np.where(np.diff(k) > 0)[0] + 1
    ts = _t_hi[idx]
    return ts[ts <= t_max]


# ---------------------------------------------------------------- eclipse
SUN_FLASH = EV["flash"]
def moon_offset(t):
    """Signed lateral offset of the moon centre from the sun centre, in units
    of the sun's angular radius.  |offset| <= 0.035 means the moon fully covers
    the sun (moon is 3.5% larger).  Totality is exactly 52.0 s -> 55.5 s."""
    t = np.asarray(t, dtype=float)
    from scipy.interpolate import PchipInterpolator
    xs = [36.0, 44.0, 50.0, 51.0, 51.6, 52.0, 55.5, 55.9, 56.5, 58.0, 60.0]
    ys = [-6.0, -3.4, -1.6, -0.75, -0.2, -0.034, 0.034, 0.2, 0.7, 1.5, 2.2]
    return PchipInterpolator(xs, ys)(np.clip(t, 36, 60))


if __name__ == "__main__":
    for k, v in TRAIN.items():
        print(k, v["N"], "center=(%.2f, %.2f)" % v["center"], "z=%.2f" % v["z"],
              "w/w1=%.4f" % v["ratio"])
    print("G1 revs at 36s: %.2f" % (theta1(36.0) / (2 * np.pi)))
    print("G1 tooth ticks <=36s:", len(tooth_ticks("G1")))
