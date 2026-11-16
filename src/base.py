"""Stage 5a - the perch: a terraced ice-rock floe (dark strata on the cliff faces, wind-packed snow and blue ice sheets on the
tops), a boulder under the raised fore foot, and clusters of ice crystals growing out of the rim. One static piece (rig: root)."""
import numpy as np
import config as cfg, geometry as geo
C = cfg.C

# footprint lobes (x, z, rx, rz) blended into one outline; (x, z, rx, rz, top height) terraces
LOBES = [(0, 6, 50, 56), (-46, 86, 46, 26), (-22, -30, 30, 26), (16, -34, 30, 22), (26, 44, 34, 30)]
# crystal clusters on the rim: (x, z, [(dx, dz, tilt_x, tilt_z, height, radius)])
CLUSTERS = [
    (52, 40, [(0, 0, .32, .12, 34, 6.2), (7, -6, .62, -.2, 20, 4.0), (-6, 6, -.05, .5, 24, 4.4), (5, 9, .5, .55, 13, 2.8), (-9, -3, -.3, -.1, 12, 2.6)]),
    (46, -38, [(0, 0, .5, -.42, 19, 3.8), (-6, -4, .05, -.62, 13, 2.8), (5, 5, .7, .1, 11, 2.4)]),
    (-47, 28, [(0, 0, -.55, -.05, 22, 4.2), (-3, 9, -.62, .4, 15, 3.0), (6, -6, -.18, -.4, 12, 2.6)]),
    (-14, 99, [(0, 0, .1, .5, 21, 4.0), (9, -2, .5, .42, 13, 2.8), (-10, 0, -.4, .5, 14, 2.9)]),
    (-74, 99, [(0, 0, -.45, .35, 16, 3.2), (7, 3, .2, .5, 10, 2.3)]),
    (-40, -45, [(0, 0, -.5, -.5, 13, 2.8), (6, -3, .1, -.7, 9, 2.1)]),
    (8, -52, [(0, 0, .15, -.7, 10, 2.3), (-7, 1, -.35, -.6, 8, 2.0)]),
]


def outline(X, Z):
    """signed 'footprint' field: < 0 inside. Smooth union of ellipses + a low-frequency wobble = natural floe edge."""
    d = None
    for x, z, rx, rz in LOBES:
        e = (np.sqrt(((X - x) / rx) ** 2 + ((Z - z) / rz) ** 2) - 1) * min(rx, rz)
        d = e if d is None else geo.smin(d, e, 14)
    return d + 2.6 * np.sin(X * .19 + 1.3) * np.sin(Z * .16 + .4) + 1.5 * np.sin(X * .41 + Z * .37)


def run(v):
    X, Y, Z = v.X, v.Y, v.Z
    d = outline(X, Z)
    G = cfg.GROUND
    # terraced height: full height well inside, stepping down toward the rim (three ledges with wobbling edges)
    wob = 2.2 * np.sin(X * .23 - Z * .17) + 1.4 * np.sin(Z * .31 + 2.0)
    h = np.where(d < -17 + wob, G, np.where(d < -9 + wob * .7, G - 3, np.where(d < -3.5 + wob * .4, G - 6, G - 9)))
    h = h + np.where(np.sin(X * .13 + 2.2) * np.sin(Z * .11 - .7) > .55, 1.0, 0.0) * (d < -22)        # low snow drifts on the plateau
    rock = (d < 0) & (Y < h)
    # boulder under the raised fore foot (flat top at the foot's ground level) + a step beside it
    fl = cfg.LEGS['FL']; gy = fl['ground']; fx, fz = fl['P'][3][0], fl['P'][3][2]
    b = geo.sd_ell(X, Y, Z, (fx - 2, gy - 15, fz - 8), (20, 16.5, 22))
    b = geo.smin(b, geo.sd_ell(X, Y, Z, (fx - 15, 8, fz + 12), (13, 12, 14)), 4)
    b = geo.smin(b, geo.sd_ell(X, Y, Z, (fx + 14, 7, fz - 16), (12, 10, 12)), 4)
    b += .9 * np.sin(X * .7) * np.sin(Y * .8 + 1.0) * np.sin(Z * .6)                                  # chipped facets
    boulder = (b < 0) & (Y < gy) & (Y > 0)
    base = (rock | boulder) & (v.kind == 0)
    bid = v.newid()
    # cliff faces: strata bands that dip gently; tops get snow / ice below
    strata = np.floor((Y + .10 * X - .07 * Z + 1.2 * np.sin(X * .09 + Z * .05)) / 2.6).astype(int)
    v.put(base, C['ROCK'], p=bid, l=0)
    v.put(base & (strata % 3 == 0), C['ROCKD'])
    v.pid[base] = bid + (strata[base] % 3 == 0)
    v.extra['BASE'] = base.copy()
    # crystal clusters on the rim
    cm = np.zeros(v.n, bool)
    for cx, cz, shards in CLUSTERS:
        for j, (dx, dz, tx, tz, H, R) in enumerate(shards):
            x, z = cx + dx, cz + dz
            ix = v.ix(x, 0), v.ix(z, 2)
            col_ = base[ix[0], :, ix[1]]; top = (np.flatnonzero(col_).max() + .5 + v.lo[1]) if col_.any() else 2.0
            cm |= geo.crystal(v, (x, top - 3.0, z), (tx, 1.0, tz), H + 3.0, R, roll=.7 * j + cx * .1, tip=.45, base=1.0)
    v.extra['BASE'] |= cm; v.extra['BASEX'] = cm
    # snow and ice on every upward face of the rock: packed snow (frost white) with pale-cyan wind bands, blue ice sheets near the rim
    up = geo.up_open(v.kind) & base
    n1 = np.sin(X * .085 + Z * .06 + .6) + .6 * np.sin(X * .05 - Z * .14 + 2.0) + .35 * np.sin(X * .21 + Z * .19)
    snow = np.where(n1 > .55, 2, np.where(n1 > -.45, 1, 0))
    tone = np.array([C['GLAC'], C['CYANP'], C['FROST']])[snow]
    v.col[up] = tone[up]
    sheet = up & (np.sin(X * .11 - Z * .09 + 1.0) * np.sin(X * .07 + Z * .12) > .62) & (Y < G + 1.5)   # glassy blue ice sheets
    v.col[sheet] = C['ICED']
    crack = sheet & ((np.floor((X + .6 * Z) / 1.0) % 7 == 0) | (np.floor((Z - .5 * X) / 1.0) % 9 == 0))
    v.col[crack] = C['ICE']
    v.pid[up] = bid + 2 + snow[up]; v.lvl[up] = .5
    # icicle fringe under each ledge lip: top two bricks of every cliff face are snow-capped
    cap = base & ~up & np.roll(up, -1, axis=1) & (geo.hash3(X.astype(np.int64) // 2, Z.astype(np.int64) // 2, 11) < .5)
    v.col[cap] = C['CYANP']
    print('BASE rock %d bricks, rim crystals %d bricks' % (int(base.sum()), int(cm.sum())))
    return v
