"""Voxel grid state + SDF helpers (LihFolk brick conventions: kind 0 empty / 1 solid / 2+ glow; col rgb; pid part id; lvl seam level)."""
import os, json
import numpy as np
import config as cfg


class Vox:
    def __init__(self, lo=cfg.LO, hi=cfg.HI):
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float)
        self.n = tuple((self.hi - self.lo).astype(int))
        self.c = [self.lo[k] + .5 + np.arange(self.n[k]) for k in range(3)]
        self.X, self.Y, self.Z = np.meshgrid(*[c.astype(np.float32) for c in self.c], indexing='ij')
        self.kind = np.zeros(self.n, np.int8); self.col = np.zeros((*self.n, 3), np.float32)
        self.pid = np.zeros(self.n, np.int32); self.lvl = np.zeros(self.n, np.float32)
        self.extra = {}                        # named auxiliary fields (body coords, masks ...)
        self.meta = {'next_id': 10}

    def newid(self):
        self.meta['next_id'] += 1; return self.meta['next_id']

    def put(self, m, rgb, k=1, p=None, l=None):
        self.col[m] = rgb; self.kind[m] = k
        if p is not None: self.pid[m] = p
        if l is not None: self.lvl[m] = l

    def ix(self, val, a): return int(round(val - self.lo[a] - .5))

    def sub(self, p0, p1, pad=0):
        """slices of the grid covering the world-space box p0..p1 (+pad)."""
        return tuple(slice(max(0, self.ix(min(p0[k], p1[k]) - pad, k)), min(self.n[k], self.ix(max(p0[k], p1[k]) + pad, k) + 1)) for k in range(3))

    def mask(self, name):
        if name not in self.extra: self.extra[name] = np.zeros(self.n, bool)
        return self.extra[name]

    # ---- checkpoints
    def save(self, stage):
        d = os.path.join(cfg.CKPT, stage); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, 'grid.npz'), lo=self.lo, hi=self.hi, kind=self.kind, col=self.col, pid=self.pid, lvl=self.lvl,
                            **{'x_' + k: v for k, v in self.extra.items()})
        json.dump(self.meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
        print('CKPT_SAVED', stage, 'solid', int((self.kind > 0).sum()))

    @classmethod
    def load(cls, stage):
        d = os.path.join(cfg.CKPT, stage); z = np.load(os.path.join(d, 'grid.npz'))
        v = cls(z['lo'], z['hi'])
        v.kind = z['kind']; v.col = z['col']; v.pid = z['pid']; v.lvl = z['lvl']
        v.extra = {k[2:]: z[k] for k in z.files if k.startswith('x_')}
        v.meta = json.load(open(os.path.join(d, 'meta.json')))
        return v


# ---------------------------------------------------------------- signed distances
def sd_ell(X, Y, Z, c, r):
    px, py, pz = (X - c[0]) / r[0], (Y - c[1]) / r[1], (Z - c[2]) / r[2]
    k0 = np.sqrt(px * px + py * py + pz * pz)
    k1 = np.sqrt((px / r[0]) ** 2 + (py / r[1]) ** 2 + (pz / r[2]) ** 2) + 1e-9
    return k0 * (k0 - 1) / k1


def smin(a, b, k):
    h = np.clip(.5 + .5 * (b - a) / k, 0, 1); return b + (a - b) * h - k * h * (1 - h)


def sd_cap(X, Y, Z, a, b, ra, rb):
    """signed distance to a tapered capsule a->b. Returns (d, t)."""
    a = np.array(a, float); b = np.array(b, float); ab = b - a
    t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2]) / (ab.dot(ab) + 1e-9), 0, 1)
    d = np.sqrt((X - a[0] - ab[0] * t) ** 2 + (Y - a[1] - ab[1] * t) ** 2 + (Z - a[2] - ab[2] * t) ** 2)
    return d - (ra + (rb - ra) * t), t


def sd_chain(X, Y, Z, pts, radii):
    d = np.full(np.shape(X), 1e9, np.float32)
    for i in range(len(pts) - 1): d = np.minimum(d, sd_cap(X, Y, Z, pts[i], pts[i + 1], radii[i], radii[i + 1])[0])
    return d


# ---------------------------------------------------------------- curves
def catmull(P, step=.4):
    P = np.array(P, float); P = np.vstack([P[0] * 2 - P[1], P, P[-1] * 2 - P[-2]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        m = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, m, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2]); return np.array(out)


def arclen(C): return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])


def resample(C, step=None, n=None):
    """re-sample a polyline at uniform arc-length spacing (by step, or to n points)."""
    s = arclen(C)
    t = np.arange(0, s[-1], step) if n is None else np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, C[:, k]) for k in range(C.shape[1])], 1)


def curve(P, n=200):
    """smooth curve through control points, n samples, uniform in arc length."""
    return resample(catmull(P, .25), n=n)


def frames(C, ref):
    """tangent T, normal N (as close to `ref` as possible, smoothly), binormal B along polyline C. ref: (3,) or (len(C), 3)."""
    T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.broadcast_to(np.asarray(ref, float), C.shape).copy()
    N = np.zeros_like(C); n0 = ref[0] - T[0] * ref[0].dot(T[0]); N[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(C)):                                       # parallel transport (Bishop frame): no sudden flips
        w = N[i - 1] - T[i] * N[i - 1].dot(T[i]); N[i] = w / np.linalg.norm(w)
    r = ref - T * (ref * T).sum(1, keepdims=True); r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
    Bn = np.cross(T, N)
    tw = np.unwrap(np.arctan2((r * Bn).sum(1), (r * N).sum(1)))       # twist that brings N onto ref ...
    for _ in range(200): tw[1:-1] = (tw[:-2] + tw[1:-1] + tw[2:]) / 3  # ... smoothed = no sudden roll
    N = N * np.cos(tw)[:, None] + Bn * np.sin(tw)[:, None]; Bn = np.cross(T, N)
    return T, N, Bn


def tube_fields(v, C, R, T, N, Bn, pad=14.0, open_ends=(True, True)):
    """per-voxel tube coordinates in the sub-box around polyline C: returns (slices, S, TH, D, RB).
    S = arc length, TH = angle around (0 = N side), D = distance - radius (99 outside the shell), RB = local radius."""
    from scipy.spatial import cKDTree
    s = arclen(C); rm = float(R.max())
    sl = v.sub(C.min(0) - rm - pad, C.max(0) + rm + pad)
    X, Y, Z = v.X[sl], v.Y[sl], v.Z[sl]; sh = X.shape
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float64)
    dist, idx = cKDTree(C).query(P, distance_upper_bound=rm + pad, workers=-1)
    ok = np.isfinite(dist); idx = np.where(ok, idx, 0)
    D = np.where(ok, dist - R[idx], 99.0)
    vec = P - C[idx]
    TH = np.arctan2((vec * Bn[idx]).sum(1), (vec * N[idx]).sum(1))
    along = (vec * T[idx]).sum(1)
    if open_ends[0]: D = np.where((idx == 0) & (along < -.5), 99.0, D)
    if open_ends[1]: D = np.where((idx == len(C) - 1) & (along > .5), 99.0, D)
    return (sl, np.where(ok, s[idx], -1).astype(np.float32).reshape(sh), TH.astype(np.float32).reshape(sh),
            D.astype(np.float32).reshape(sh), R[idx].astype(np.float32).reshape(sh))


# ---------------------------------------------------------------- topology
def components(solid):
    """26-neighbour connected components. Returns (labels, [(label, size)] sorted by size desc)."""
    from scipy import ndimage
    lab, nlab = ndimage.label(solid, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(-sizes)
    return lab, [(int(i + 1), int(sizes[i])) for i in order]


def drop_floating(v):
    """skill rule: drop every brick not 26-connected to the main body. Returns (n_components_before, dropped_bricks)."""
    lab, comps = components(v.kind > 0)
    if len(comps) <= 1: return len(comps), 0
    drop = (lab > 0) & (lab != comps[0][0])
    v.kind[drop] = 0
    return len(comps), int(drop.sum())


def hash3(a, b=0, c=0):
    """deterministic integer hash -> float in [0,1) (no RNG state; stable across stages)."""
    h = (np.asarray(a, np.int64) * 73856093) ^ (np.asarray(b, np.int64) * 19349663) ^ (np.asarray(c, np.int64) * 83492791) ^ cfg.SEED
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 65536.0


def edt_in(mask):
    from scipy import ndimage
    return ndimage.distance_transform_edt(mask)


def edt_out(mask):
    from scipy import ndimage
    return ndimage.distance_transform_edt(~mask)


def up_open(kind):
    """solid cells whose upper neighbour is empty (they catch the snow)."""
    m = np.zeros(kind.shape, bool); m[:, :-1, :] = (kind[:, :-1, :] > 0) & (kind[:, 1:, :] == 0); m[:, -1, :] = kind[:, -1, :] > 0
    return m


# ---------------------------------------------------------------- local frames
def basis(f, u):
    f = np.array(f, float); f /= np.linalg.norm(f)
    u = np.array(u, float); u = u - f * u.dot(f); u /= np.linalg.norm(u)
    return np.cross(f, u), u, f                                       # right, up, forward


class Frame:
    """A sub-box of the grid seen in a local frame (origin o, forward f, up u, right r = f x u), optionally scaled.
    Coordinates: x lateral, ax = |x| (mirror symmetry), y up, z forward. Writes go straight into the grid (views)."""

    def __init__(self, v, o, f, u, p0, p1, scale=1.0):
        self.v = v; self.sl = v.sub(p0, p1); self.o = np.array(o, float); self.s = scale
        self.r, self.u, self.f = basis(f, u)
        X, Y, Z = v.X[self.sl], v.Y[self.sl], v.Z[self.sl]
        q = (X - self.o[0], Y - self.o[1], Z - self.o[2])
        self.x = (q[0] * self.r[0] + q[1] * self.r[1] + q[2] * self.r[2]) / scale; self.ax = np.abs(self.x)
        self.y = (q[0] * self.u[0] + q[1] * self.u[1] + q[2] * self.u[2]) / scale
        self.z = (q[0] * self.f[0] + q[1] * self.f[1] + q[2] * self.f[2]) / scale
        self.kind = v.kind[self.sl]; self.col = v.col[self.sl]; self.pid = v.pid[self.sl]; self.lvl = v.lvl[self.sl]

    def put(self, m, rgb, k=1, p=None, l=None):
        self.col[m] = rgb; self.kind[m] = k
        if p is not None: self.pid[m] = p
        if l is not None: self.lvl[m] = l

    def cap(self, a, b, ra, rb, sym=True):
        d, _ = sd_cap(self.ax if sym else self.x, self.y, self.z, a, b, ra, rb); return d < 0

    def chain(self, pts, radii, sym=True):
        m = np.zeros(self.kind.shape, bool)
        for i in range(len(pts) - 1): m |= self.cap(pts[i], pts[i + 1], radii[i], radii[i + 1], sym)
        return m

    def world(self, p):
        return self.o + (self.r * p[0] + self.u * p[1] + self.f * p[2]) * self.s

    def full(self, m):
        """lift a sub-box mask to a full-grid mask."""
        g = np.zeros(self.v.n, bool); g[self.sl] = m; return g


# ---------------------------------------------------------------- ice crystals
LIGHT = np.array((-.45, .78, -.44)); LIGHT = LIGHT / np.linalg.norm(LIGHT)      # key light direction baked into the facet tones


def crystal(v, p0, d, H, R, tip=.42, roll=0.0, flat=1.0, base=2.0, tones=None, pid=None, lvl=8, only_empty=False, sides=6):
    """faceted ice crystal: hexagonal prism from p0 along d (length H, radius R) ending in a pyramid over the last `tip` fraction.
    `flat` < 1 squashes one cross axis (blade). Facets are coloured by how much they face the key light = cut-gem look in bricks.
    Returns the full-grid mask of the crystal."""
    C = cfg.C
    tones = tones or (C['ICEDD'], C['ICED'], C['ICE'], C['FROST'])
    p0 = np.array(p0, float); d = np.array(d, float); d /= np.linalg.norm(d)
    e1 = np.cross(d, (0, 1, 0) if abs(d[1]) < .9 else (1, 0, 0)); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    e1, e2 = e1 * np.cos(roll) + e2 * np.sin(roll), e2 * np.cos(roll) - e1 * np.sin(roll)
    sl = v.sub(p0 - R - base - 2, p0 + d * H, pad=R + 2)
    X, Y, Z = v.X[sl], v.Y[sl], v.Z[sl]
    q = (X - p0[0], Y - p0[1], Z - p0[2])
    a = q[0] * d[0] + q[1] * d[1] + q[2] * d[2]
    b = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2]; c = (q[0] * e2[0] + q[1] * e2[1] + q[2] * e2[2]) / flat
    hs = []; ns = []
    for k in range(sides // 2):
        an = np.pi * k / (sides // 2); hs.append(b * np.cos(an) + c * np.sin(an)); ns.append(e1 * np.cos(an) + e2 * np.sin(an) * flat)
    hs = np.stack(hs); ah = np.abs(hs); h = ah.max(0); fi = ah.argmax(0); sg = np.sign(np.take_along_axis(hs, fi[None], 0)[0])
    rr = R * np.minimum(1.0, np.clip((H - a) / (tip * H), 0, 1)) * (1 - .12 * np.clip(a / H, 0, 1))
    m = (a > -base) & (a < H) & (h < rr)
    if only_empty: m &= v.kind[sl] == 0
    nrm = np.array(ns)[fi] * sg[..., None]                                             # facet normal (world)
    tipz = a > H * (1 - tip)
    nrm = np.where(tipz[..., None], nrm * .8 + d * .6, nrm)
    lit = (nrm * LIGHT).sum(-1) / (np.linalg.norm(nrm, axis=-1) + 1e-9)
    ti = np.digitize(lit, (-.25, .25, .68))                                           # 4 tones by facet orientation
    ti = np.where(a < .12 * H, np.minimum(ti, 1), ti)                                 # darker root where it grows out of the body
    ti = np.where(a > H * (1 - .16), np.maximum(ti, 2), ti)                           # bright point
    rgb = np.array(tones)[np.clip(ti, 0, len(tones) - 1)]
    v.col[sl][m] = rgb[m]; v.kind[sl][m] = 1
    v.pid[sl][m] = pid if pid is not None else v.newid(); v.lvl[sl][m] = lvl
    g = np.zeros(v.n, bool); g[sl] = m
    return g
