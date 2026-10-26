"""Procedural mesh helpers.  Every builder returns a Mesh (points, counts, indices)."""
import numpy as np


class Mesh:
    def __init__(self, pts=None, counts=None, idx=None):
        self.pts = np.zeros((0, 3)) if pts is None else np.asarray(pts, float)
        self.counts = np.zeros(0, int) if counts is None else np.asarray(counts, int)
        self.idx = np.zeros(0, int) if idx is None else np.asarray(idx, int)

    def merge(self, other):
        off = len(self.pts)
        self.pts = np.vstack([self.pts, other.pts])
        self.counts = np.concatenate([self.counts, other.counts])
        self.idx = np.concatenate([self.idx, other.idx + off])
        return self

    def translate(self, v):
        self.pts = self.pts + np.asarray(v, float)
        return self

    def rotz(self, a):
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.pts = self.pts @ R.T
        return self

    def copy(self):
        return Mesh(self.pts.copy(), self.counts.copy(), self.idx.copy())


def _quads(a, b, c, d):
    """a,b,c,d : arrays of vertex ids -> counts, idx"""
    n = len(a)
    return np.full(n, 4), np.stack([a, b, c, d], 1).reshape(-1)


def box(size, center=(0, 0, 0)):
    sx, sy, sz = [s / 2 for s in size]
    p = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                  [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]]) + center
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return Mesh(p, [4] * 6, np.array(f).reshape(-1))


def cylinder(r, h, seg=32, z0=0.0, r_top=None):
    r_top = r if r_top is None else r_top
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    bot = np.stack([r * np.cos(a), r * np.sin(a), np.full(seg, z0)], 1)
    top = np.stack([r_top * np.cos(a), r_top * np.sin(a), np.full(seg, z0 + h)], 1)
    pts = np.vstack([bot, top, [[0, 0, z0], [0, 0, z0 + h]]])
    ib = np.arange(seg)
    nb = (ib + 1) % seg
    cnt, idx = _quads(ib, nb, nb + seg, ib + seg)
    cB = [3] * seg
    iB = np.stack([np.full(seg, 2 * seg), nb, ib], 1).reshape(-1)
    cT = [3] * seg
    iT = np.stack([np.full(seg, 2 * seg + 1), ib + seg, nb + seg], 1).reshape(-1)
    return Mesh(pts, np.concatenate([cnt, cB, cT]), np.concatenate([idx, iB, iT]))


def annulus(r_in, r_out, thick, seg=128, z0=0.0):
    """Flat ring plate."""
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    pts = np.vstack([
        np.stack([r_out * c, r_out * s, np.full(seg, z0)], 1),
        np.stack([r_out * c, r_out * s, np.full(seg, z0 + thick)], 1),
        np.stack([r_in * c, r_in * s, np.full(seg, z0 + thick)], 1),
        np.stack([r_in * c, r_in * s, np.full(seg, z0)], 1)])
    i = np.arange(seg)
    n = (i + 1) % seg
    parts = [
        _quads(i, n, n + seg, i + seg),                         # outer wall
        _quads(i + seg, n + seg, n + 2 * seg, i + 2 * seg),     # top
        _quads(i + 2 * seg, n + 2 * seg, n + 3 * seg, i + 3 * seg),  # inner wall
        _quads(i + 3 * seg, n + 3 * seg, n, i)]                 # bottom
    cnt = np.concatenate([p[0] for p in parts])
    idx = np.concatenate([p[1] for p in parts])
    return Mesh(pts, cnt, idx)


def uv_sphere(r, nu=48, nv=24, hemi=None):
    """hemi: None | '+x' | '-x' selects which half (split by plane x=0)."""
    v = np.linspace(0, np.pi, nv + 1)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    P = []
    for vi in v:
        for ui in u:
            P.append([r * np.sin(vi) * np.cos(ui), r * np.sin(vi) * np.sin(ui), r * np.cos(vi)])
    P = np.array(P)
    def vid(i, j):
        return i * nu + (j % nu)
    cnt, idx = [], []
    for i in range(nv):
        for j in range(nu):
            q = [vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)]
            if hemi:
                cx = P[q, 0].mean()
                if (hemi == "+x" and cx < 0) or (hemi == "-x" and cx >= 0):
                    continue
            cnt.append(4)
            idx += q
    m = Mesh(P, cnt, idx)
    if hemi:   # cap the cut with a disc so the half is closed
        cap_n = nv * 2
        ring = []
        for i in range(nv + 1):
            ring.append([0, r * np.sin(v[i]), r * np.cos(v[i])])
        ring = np.array(ring)
        # fan from centre
        base = len(m.pts)
        m.pts = np.vstack([m.pts, [[0, 0, 0]], ring])
        c = []
        ii = []
        for i in range(nv):
            c.append(3)
            tri = [base, base + 1 + i, base + 2 + i] if hemi == "+x" else [base, base + 2 + i, base + 1 + i]
            ii += tri
        m.counts = np.concatenate([m.counts, c])
        m.idx = np.concatenate([m.idx, ii])
    return m


# ------------------------------------------------------------------ gears
def toothed_outline(N, R, phi, add=None, ded=None, ang_off=0.0):
    """4 points per tooth (root-left, tip-left, tip-right, root-right)."""
    from timeline import MODULE
    add = MODULE * 1.0 if add is None else add
    ded = MODULE * 1.25 if ded is None else ded
    p = 2 * np.pi / N
    i = np.arange(N)
    a = phi + i * p + ang_off
    ang = np.stack([a - 0.30 * p, a - 0.12 * p, a + 0.12 * p, a + 0.30 * p], 1).reshape(-1)
    rad = np.tile([R - ded, R + add, R + add, R - ded], N)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), ang, rad


def gear_mesh(N, R, thick, phi, spokes=None, rim=None, hub_r=None):
    """A spoked, toothed gear lying in the XY plane, z in [0, thick]."""
    from timeline import MODULE
    out, ang, _ = toothed_outline(N, R, phi)
    M = len(out)
    solid = R < 1.9
    rim = (max(0.28, 0.10 * R) if rim is None else rim)
    r_in = 0.12 if solid else max(R - 0.1 - rim, 0.5)
    hub_r = (0.28 + 0.02 * R) if hub_r is None else hub_r
    inn = np.stack([r_in * np.cos(ang), r_in * np.sin(ang)], 1)

    z0, z1 = 0.0, thick
    P = np.vstack([
        np.c_[out, np.full(M, z0)],     # 0 outer bottom
        np.c_[out, np.full(M, z1)],     # 1 outer top
        np.c_[inn, np.full(M, z1)],     # 2 inner top
        np.c_[inn, np.full(M, z0)]])    # 3 inner bottom
    i = np.arange(M)
    n = (i + 1) % M
    parts = [
        _quads(i, n, n + M, i + M),                          # outer wall
        _quads(i + M, n + M, n + 2 * M, i + 2 * M),          # top face (CCW from +z)
        _quads(i + 2 * M, n + 2 * M, n + 3 * M, i + 3 * M),  # inner wall
        _quads(i + 3 * M, n + 3 * M, n, i)]                  # bottom face
    m = Mesh(P, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    # hub
    m.merge(cylinder(hub_r, thick * 1.35, seg=24, z0=-thick * 0.175))
    if not solid:
        ns = spokes or (4 if R < 4 else 5 if R < 7 else 6)
        sw = 0.22 + 0.025 * R
        for k in range(ns):
            a = phi * 0 + 2 * np.pi * k / ns + 0.5
            L = r_in - hub_r * 0.5
            b = box((L, sw, thick * 0.62), center=(hub_r * 0.5 + L / 2, 0, thick / 2))
            b.rotz(a)
            m.merge(b)
    return m


# ------------------------------------------------------------------ noise
def _hash3(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 1274126177) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFF) / 65535.0


def vnoise(p, seed=0):
    """3-D value noise in [0,1], p: (N,3)."""
    ip = np.floor(p).astype(np.int64)
    f = p - ip
    f = f * f * (3 - 2 * f)
    acc = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * \
                    (f[:, 2] if dz else 1 - f[:, 2])
                acc = acc + w * _hash3(ip[:, 0] + dx, ip[:, 1] + dy, ip[:, 2] + dz, seed)
    return acc


def fbm(p, octaves=5, seed=0):
    amp, freq, tot, norm = 1.0, 1.0, 0.0, 0.0
    for o in range(octaves):
        tot += amp * vnoise(p * freq, seed + o * 17)
        norm += amp
        amp *= 0.5
        freq *= 2.03
    return tot / norm


def lump(center, radii, nu=160, nv=80, bump=3.2, seed=7):
    """Encrusted, corroded ellipsoid: returns points (nv+1,nu,3) as a Mesh plus
    per-face centroid array & unit outward directions for shard splitting."""
    v = np.linspace(0, np.pi, nv + 1)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    U, V = np.meshgrid(u, v)
    d = np.stack([np.sin(V) * np.cos(U), np.sin(V) * np.sin(U), np.cos(V)], -1)
    flat = d.reshape(-1, 3)
    n1 = fbm(flat * 2.2 + 11, 5, seed)
    n2 = fbm(flat * 7.0 + 3, 4, seed + 5)
    # crater/barnacle ridges
    disp = bump * (n1 - 0.4) + 0.9 * (n2 - 0.5)
    disp = np.maximum(disp, -0.2)
    rad = np.array(radii)[None, :]
    P = flat * rad + flat * disp[:, None] + np.asarray(center)
    cnt, idx = [], []
    for i in range(nv):
        for j in range(nu):
            a = i * nu + j
            b = i * nu + (j + 1) % nu
            c = (i + 1) * nu + (j + 1) % nu
            dd = (i + 1) * nu + j
            cnt.append(4)
            idx += [a, b, c, dd]
    return Mesh(P, cnt, idx), flat


def submesh(mesh, face_ids):
    """Extract faces (by index) into a compact Mesh."""
    starts = np.concatenate([[0], np.cumsum(mesh.counts)])
    chosen = []
    cnts = []
    for f in face_ids:
        chosen.append(mesh.idx[starts[f]:starts[f + 1]])
        cnts.append(mesh.counts[f])
    allv = np.concatenate(chosen)
    uniq, inv = np.unique(allv, return_inverse=True)
    return Mesh(mesh.pts[uniq], cnts, inv)


def face_centroids(mesh):
    starts = np.concatenate([[0], np.cumsum(mesh.counts)])
    cs = np.zeros((len(mesh.counts), 3))
    for f in range(len(mesh.counts)):
        cs[f] = mesh.pts[mesh.idx[starts[f]:starts[f + 1]]].mean(0)
    return cs


def tick_marks(r0, r1, count, long_every, long_extra, width, thick, z0=0.0):
    """Radial engraved marks on a dial, merged into one mesh."""
    m = Mesh()
    for k in range(count):
        a = 2 * np.pi * k / count
        L = (r1 - r0) + (long_extra if k % long_every == 0 else 0.0)
        b = box((L, width * (1.8 if k % long_every == 0 else 1.0), thick),
                center=(r0 + L / 2, 0, z0 + thick / 2)).rotz(a)
        m.merge(b)
    return m


def grid_surface(xs, ys, zfun):
    X, Y = np.meshgrid(xs, ys)
    Z = zfun(X, Y)
    nx, ny = len(xs), len(ys)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    cnt, idx = [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            idx += [a, a + 1, a + nx + 1, a + nx]
            cnt.append(4)
    return Mesh(P, cnt, idx)
