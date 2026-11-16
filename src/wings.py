"""Stage 3c - wings: muscular arm (humerus + forearm), thumb claw, four long finger struts with knuckles, and the membrane as
ruled surfaces between neighbouring struts (two bricks thick, cupped, scalloped trailing edge). Membrane cells keep their panel
coordinates so the paint follows the anatomy:
  WPAN panel number (1-3 hand, 4-5 arm-to-flank, 6 leading-edge skin) with sign = side (+ front/top layer, - back/under layer)
  WU   0..1 across the panel (strut to strut)      WT   0..1 from the arm out to the trailing edge (already divided by the scallop)"""
import numpy as np
import config as cfg, geometry as geo
C = cfg.C
SC = {1: .13, 2: .15, 3: .16, 4: .12, 5: .05, 6: .0}             # scallop depth of each panel's trailing edge
BILL = {1: 2.2, 2: 3.0, 3: 3.4, 4: 3.0, 5: 1.2, 6: 1.2}          # cupping (bricks) of each panel


def mir(p, sx): return np.array((p[0] * sx, p[1], p[2]), float)


def line(a, b, n): return np.linspace(np.array(a, float), np.array(b, float), n)


def panel(v, A, B, pan, sx):
    """stamp the ruled surface between curves A(t) and B(t) (same sample count) into the grid; returns the number of new bricks."""
    nt = len(A); width = np.linalg.norm(A - B, axis=1).max()
    nu = max(8, int(width / .32)); u = np.linspace(0, 1, nu)[:, None, None]
    # resample along t so samples are <= .32 bricks apart on the longer curve
    ln = max(geo.arclen(A)[-1], geo.arclen(B)[-1]); m = max(8, int(ln / .32))
    ti = np.linspace(0, nt - 1, m); i0 = np.floor(ti).astype(int).clip(0, nt - 2); fr = (ti - i0)[:, None]
    A2 = A[i0] * (1 - fr) + A[i0 + 1] * fr; B2 = B[i0] * (1 - fr) + B[i0 + 1] * fr
    S = A2[None] * (1 - u) + B2[None] * u                                 # (nu, m, 3)
    t = np.linspace(0, 1, m)[None, :, None]
    du = np.gradient(S, axis=0); dt = np.gradient(S, axis=1)
    nrm = np.cross(du, dt); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
    if np.mean(nrm[..., 2]) > 0: nrm = -nrm                               # front/top side faces forward (-z)
    S = S - nrm * (BILL[pan] * np.sin(np.pi * u) * (.25 + .75 * np.sin(np.pi * np.clip(t, 0, 1) * .85)))   # cupped like a sail (bulging to the back)
    tmax = 1 - SC[pan] * np.sin(np.pi * u) ** .85
    ok = (t <= tmax)[..., 0]
    U = np.broadcast_to(u[..., 0], ok.shape)[ok]; Tn = (t / tmax)[..., 0][ok]; P = S[ok]; Nn = nrm[ok]
    new = 0
    WP, WU, WT, MEMB = v.extra['WPAN'], v.extra['WU'], v.extra['WT'], v.extra['MEMB']
    for k, sd in ((.55, -1), (-.55, 1)):                                  # back layer first, front layer wins overlaps
        Q = P + Nn * (-k)
        ix = np.floor(Q - v.lo).astype(int)
        inb = ((ix >= 0) & (ix < np.array(v.n))).all(1); ix = ix[inb]; i = tuple(ix.T)
        free = (v.kind[i] == 0) | MEMB[i]
        i = tuple(ix[free].T)
        new += int((v.kind[i] == 0).sum())
        v.kind[i] = 1; MEMB[i] = True; WP[i] = pan * sd; WU[i] = U[inb][free]; WT[i] = Tn[inb][free]
        v.pid[i] = 3000 + pan + (10 if sx < 0 else 0); v.lvl[i] = 1
    return new


def paint_membrane(v):
    """frosted membrane: four tone bands that follow the scalloped trailing edge (dark at the arm -> frost-white rim) with icicle
    zigzag borders, thin radial veins, the underside one tone deeper."""
    M = v.extra['MEMB'] & (v.kind == 1); WP, U, T = v.extra['WPAN'], v.extra['WU'], v.extra['WT']
    pan = np.abs(WP); front = WP > 0
    FR = [C['STEELL'], C['GLAC'], C['CYANP'], C['FROST']]; BK = [C['STEEL'], C['STEELL'], C['GLAC'], C['CYANP']]
    zig = .035 * np.abs(((U * 9.0) % 1.0) * 2 - 1)                         # icicle teeth on the band borders
    band = (T > .30 + zig).astype(int) + (T > .58 + zig) + (T > .88 + zig * .6)
    band = np.where(pan == 6, np.minimum(band, 1), band)                   # leading-edge skin stays dark
    band = np.where(pan == 5, np.minimum(band + 0, 2), band)
    rgb = np.where(front[..., None], np.array(FR)[band], np.array(BK)[band])
    vein = (np.abs(((U * 4.0) % 1.0) - .5) > .455) & (U > .1) & (U < .9) & (pan <= 4) & (T > .12) & (T < .93)
    rgb = np.where(vein[..., None], np.where(front[..., None], np.array(FR)[np.maximum(band - 1, 0)], np.array(BK)[np.maximum(band - 1, 0)]), rgb)
    v.col[M] = rgb[M]


def run(v):
    for k_, dt in (('WPAN', np.int8), ('WU', np.float32), ('WT', np.float32), ('MEMB', bool)): v.extra[k_] = np.zeros(v.n, dt)
    v.extra['WBONE'] = np.zeros(v.n, np.int8)                             # 1 arm, 2 forearm, 3 hand (fingers, thumb); sign = side
    g = cfg.WING; N = 160; stats = {}
    for sx in (1, -1):
        S, E, W, Hh = (mir(g[k_], sx) for k_ in 'SEWH')
        F = [geo.curve([mir(p, sx) for p in f], N) for f in g['FING']]
        nb = 0
        # --- membrane panels (stamped first; bones are written over them)
        for i in range(3): nb += panel(v, F[i], F[i + 1], i + 1, sx)
        EH = geo.curve([E, (E + Hh) / 2 + np.array((sx * 3.0, -6, 6)), Hh], N)
        nb += panel(v, F[3], EH, 4, sx)                                   # last finger -> elbow-to-hip line
        nb += panel(v, EH, geo.curve([S, (S + Hh) / 2 + np.array((sx * 1.0, 2, 0)), Hh], N), 5, sx)   # ... -> flank
        arm = np.vstack([line(S, E, N // 2), line(E, W, N // 2)])
        lead = geo.curve([S, (S + W) / 2 + np.array((sx * -2.0, 3.0, -9.0)), W], N)
        nb += panel(v, arm, lead, 6, sx)                                  # leading-edge skin in front of the elbow
        # --- arm: humerus + forearm, elbow and wrist knobs
        X, Y, Z = v.X, v.Y, v.Z
        sl = v.sub(np.minimum(S, np.minimum(E, W)) - 12, np.maximum(S, np.maximum(E, W)) + 12)
        x, y, z = X[sl], Y[sl], Z[sl]
        d1, t1 = geo.sd_cap(x, y, z, S, E, 6.2, 4.6); d2, t2 = geo.sd_cap(x, y, z, E, W, 4.6, 3.3)
        d1 = geo.smin(d1, geo.sd_ell(x, y, z, S * .55 + E * .45, (5.6, 7.5, 6.5)), 3)             # wing muscle
        kn = geo.sd_ell(x, y, z, E, (5.4, 5.4, 5.4)); wr = geo.sd_ell(x, y, z, W, (4.3, 4.3, 4.3))
        a1 = (d1 < 0) | (kn < 0); a2 = ((d2 < 0) | (wr < 0)) & ~a1
        ida, idf = v.newid(), v.newid()
        for m, pid, bn in ((a1, ida, 1), (a2, idf, 2)):
            v.col[sl][m] = C['STEEL']; v.kind[sl][m] = 1; v.pid[sl][m] = pid; v.lvl[sl][m] = 5; v.extra['WBONE'][sl][m] = bn * sx; v.extra['MEMB'][sl][m] = False
        # arm scales: banded rings down each bone (lighter lip toward the wrist)
        ring = np.where(a1, np.floor(t1 * 7), np.floor(t2 * 8))
        fr = np.where(a1, t1 * 7 % 1, t2 * 8 % 1)
        lip = (a1 | a2) & (fr > .68); v.col[sl][lip] = C['STEELL']
        v.pid[sl][a1 | a2] += (ring[a1 | a2].astype(np.int32) + 1) * 100; v.lvl[sl][a1 | a2] = 5 + (ring[a1 | a2] % 2) * .5
        v.col[sl][(kn < -1.2)] = C['SILVD']; v.col[sl][(wr < -.6)] = C['SILVD']                 # bony elbow / wrist caps
        # --- fingers: tapered struts with two knuckles and an ice-tipped claw
        for i, Fc in enumerate(F):
            fs = v.sub(Fc.min(0) - 5, Fc.max(0) + 5); x, y, z = X[fs], Y[fs], Z[fs]
            r0, r1 = (3.0, 1.15) if i else (2.8, 1.1)
            from scipy.spatial import cKDTree
            P = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
            dist, idx = cKDTree(Fc).query(P, distance_upper_bound=5.0, workers=-1); ok = np.isfinite(dist); idx = np.where(ok, idx, 0)
            f = (idx / (N - 1.0)).reshape(x.shape); dist = np.where(ok, dist, 99).reshape(x.shape)
            rad = r0 + (r1 - r0) * f + .9 * np.exp(-((f - .36) / .035) ** 2) + .8 * np.exp(-((f - .68) / .035) ** 2)
            m = dist < rad
            tone = np.where(f > .93, 3, np.where((np.abs(f - .36) < .045) | (np.abs(f - .68) < .045), 2, np.where(f < .1, 0, 1)))
            rgb = np.array([C['SILVD'], C['SILV'], C['FROST'], C['ICE']])[tone]
            under = m & (dist > rad - 1.0) & (np.floor(f * 22) % 2 == 0) & (tone == 1)
            rgb = np.where(under[..., None], C['SILVD'], rgb)                                    # bone segments
            v.col[fs][m] = rgb[m]; v.kind[fs][m] = 1; v.pid[fs][m] = v.newid(); v.lvl[fs][m] = 6
            v.extra['WBONE'][fs][m] = 3 * sx; v.extra['MEMB'][fs][m] = False
        # --- thumb claw at the wrist: hooks forward and up
        th = [W + np.array((sx * 1.0, 2.0, -3.0)), W + np.array((sx * 3.0, 8.0, -9.0)), W + np.array((sx * 3.5, 9.0, -18.0))]
        ts = v.sub(W - 24, W + 24); x, y, z = X[ts], Y[ts], Z[ts]
        d1, t1 = geo.sd_cap(x, y, z, th[0], th[1], 3.0, 2.2); d2, t2 = geo.sd_cap(x, y, z, th[1], th[2], 2.2, .5)
        m = (d1 < 0) | (d2 < 0)
        v.col[ts][m] = C['SILV']; v.kind[ts][m] = 1; v.pid[ts][m] = v.newid(); v.lvl[ts][m] = 7; v.extra['WBONE'][ts][m] = 3 * sx; v.extra['MEMB'][ts][m] = False
        v.col[ts][(d1 < 0) & (t1 < .5)] = C['SILVD']; v.col[ts][(d2 < 0) & (t2 > .5)] = C['FROST']
        v.mask('CLAW')[ts] |= m
        stats[sx] = nb
    paint_membrane(v)
    v.extra['WING'] = v.extra['MEMB'] | (v.extra['WBONE'] != 0)
    v.meta['wing'] = {k_: list(map(float, g[k_])) for k_ in 'SEWH'}
    print('WINGS membrane bricks R %d / L %d, total wing bricks %d' % (stats[1], stats[-1], int(v.extra['WING'].sum())))
    return v
