"""Stage 6 - frost signature features on top of anatomy + armour + materials:
  Frost Throat  : recessed glowing vents between the throat plates from the jaw down to the chest + the lit gullet in the open mouth
  Frost Heart   : a recessed diamond of glowing ice set into the breast plates, framed by crystal facets
  Ice Pauldrons : crystal clusters grown on the shoulders and haunches, spurs on elbows, hocks and wing elbows
  Blade Vein    : a thin glowing vein up both faces of the great tail crystal
Genuine emission (kind 3) only for the throat core, the heart and the vein; everything else is solid brick (skill glow budget)."""
import os
import numpy as np
import config as cfg, geometry as geo, head, armor
C = cfg.C


def frost_throat(v):
    S, TH, DB, FR, Z, RB = (v.extra[k] for k in ('S', 'TH', 'DB', 'FR', 'ZONE', 'RB')); L = v.meta['L']
    s0 = v.meta['s_chest'] + 22
    t = np.clip((S - s0) / (L - s0), 0, 1); ath = np.abs(TH)
    half = (2.2 + 5.2 * t) / np.maximum(RB, 1)                          # vent half-width (bricks -> angle): wide under the skull, closing down the neck
    slot = (Z == 3) & (t > 0) & (FR > .50) & (FR < .86) & (ath > np.pi - half) & (S < L - 2) & (v.kind == 1)
    v.kind[slot & (DB > -1.2)] = 0                                      # recess: cut the vent into the plates
    floor = slot & (DB <= -1.2)
    v.put(floor, C['ICED'], p=v.newid(), l=0)
    v.put(floor & (ath > np.pi - half * .8), C['ICE'])
    core = floor & (ath > np.pi - half * .55) & (t > .30)
    v.put(core, cfg.GLOW[3], k=3)
    v.mask('SOLIDGLOW')[floor] = True
    # gullet: the throat core seen through the open mouth
    g = v.extra['GULLET'] & (v.kind > 0)
    v.put(g, cfg.GLOW[3], k=3)
    # under the jaw: three chevron vents in jaw coordinates
    H, J = head.frames(v)
    dep = geo.edt_in(H.kind > 0)
    under = (J.y < -4.5) & (J.z > 5) & (J.z < 30) & (J.ax < 5.0 - (J.z - 5) * .09) & (np.mod(J.z + J.ax * .9, 6.5) < 2.3) & (H.kind == 1) & ~v.extra['HORN'][H.sl]
    H.kind[under & (dep <= 1.2)] = 0
    fl = under & (dep > 1.2) & (dep <= 2.5)
    H.put(fl, C['ICE'], p=v.newid(), l=0); H.put(fl & (J.ax > 3.2 - (J.z - 5) * .07), C['ICED']); H.put(fl & (J.ax < 1.9), cfg.GLOW[3], k=3)
    v.mask('SOLIDGLOW')[H.sl] |= fl
    print('THROAT vents: solid ice %d bricks, emissive %d bricks' % (int(v.extra['SOLIDGLOW'].sum()), int((v.kind == 3).sum())))


def frost_heart(v):
    """recessed glowing diamond in the breast, between the pectoral plates."""
    DB = v.extra['DB']
    d = np.array((0, -.22, -1.0)); d /= np.linalg.norm(d); o = armor.surface_point(v, np.array((0, 66.0, -16.0)), d, DB, 50)
    u = np.array((0, 1.0, 0)); u -= d * u.dot(d); u /= np.linalg.norm(u)
    sl = v.sub(o - 16, o + 16); X, Y, Z = v.X[sl], v.Y[sl], v.Z[sl]
    q = (X - o[0], Y - o[1], Z - o[2]); a = q[0] * d[0] + q[1] * d[1] + q[2] * d[2]; y = q[0] * u[0] + q[1] * u[1] + q[2] * u[2]; x = np.abs(q[0])
    dia = x / 5.6 + np.abs(y) / 8.2
    k = v.kind[sl]; dep = geo.edt_in(k > 0); out = geo.edt_out(k > 0)
    near = (a > -6) & (a < 5)
    frame = near & (dia < 1.42) & (dia >= 1.0) & (((k > 0) & (dep <= 2.0)) | ((k == 0) & (out <= 1.3)))
    fid = v.newid()
    lit = y > 0
    v.col[sl][frame] = np.where(lit[frame][:, None], C['ICE'], C['ICED']); v.kind[sl][frame] = 1; v.pid[sl][frame] = fid; v.lvl[sl][frame] = 6
    v.col[sl][frame & (dia > 1.25)] = C['SILV']
    v.col[sl][frame & (dia > 1.25) & ~lit] = C['SILVD']
    inside = near & (dia < 1.0)
    v.kind[sl][inside & ((dep <= 1.3) | (k == 0))] = 0                                         # recess
    core = inside & (k > 0) & (dep > 1.3) & (dep <= 2.8)
    v.col[sl][core] = cfg.GLOW[3]; v.kind[sl][core] = 3
    spar = core & ((np.abs(q[0]) < .6) | (np.abs(y) < .6))                                    # solid facet lines across the glow
    v.col[sl][spar] = C['ICE']; v.kind[sl][spar] = 1; v.pid[sl][spar] = v.newid(); v.lvl[sl][spar] = 0
    v.mask('SOLIDGLOW')[sl] |= spar
    v.mask('HEART')[sl] |= frame | core
    v.meta['heart'] = o.tolist()
    print('HEART glow %d bricks' % int(core.sum() - spar.sum()))


def pauldrons(v):
    DB = v.extra['DB']; n = 0; M = np.zeros(v.n, bool)
    def cluster(p, d, specs, seed):
        nonlocal n, M
        d = np.array(d, float); d /= np.linalg.norm(d)
        e1 = np.cross(d, (0, 0, 1.0)); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
        for j, (o1, o2, t1, t2, H, R) in enumerate(specs):
            dj = d + e1 * t1 + e2 * t2; dj /= np.linalg.norm(dj)
            b = armor.surface_point(v, np.array(p, float) + e1 * o1 + e2 * o2, dj, DB, 40) - dj * 2.0
            M |= geo.crystal(v, b, dj, H + 2.0, R, roll=seed + j * .8, tip=.5); n += 1
    for name in ('FL', 'FR', 'HL', 'HR'):
        P = np.array(cfg.LEGS[name]['P'], float); sx = np.sign(P[0, 0])
        if name[0] == 'F':
            cluster(P[0] + (sx * 2, 4, 1), (sx * .62, .7, .28), [(0, 0, 0, 0, 16, 3.4), (3.5, 2, .45, .1, 10.5, 2.4), (-3.5, 1, -.45, .25, 9, 2.2), (0, -4, 0, -.5, 7.5, 1.9)], sx)
            cluster(P[1], (sx * .25, .3, 1.0), [(0, 0, 0, 0, 9.5, 2.1)], sx + 2)                  # elbow spur
        else:
            cluster(P[0] + (sx * 2, 5, 2), (sx * .62, .62, .45), [(0, 0, 0, 0, 12.5, 2.9), (3, 2, .5, .1, 8, 2.0), (-3, 1, -.4, .3, 7, 1.9)], sx + 4)
            cluster(P[2] + (0, 2, 2), (sx * .1, .5, 1.0), [(0, 0, 0, 0, 8.5, 2.0)], sx + 6)        # hock spur
    g = cfg.WING
    for sx in (1, -1):                                                                          # wing-elbow crystal
        E = np.array(g['E'], float) * (sx, 1, 1)
        M |= geo.crystal(v, E - np.array((0, 1, 0)), (sx * -.25, .8, .55), 13.0, 2.6, roll=sx, tip=.5); n += 1
    v.extra['PAULDRON'] = M
    print('PAULDRONS %d crystals, %d bricks' % (n, int(M.sum())))


def blade_vein(v):
    b = v.meta['blade']; o, d, e, n = (np.array(b[k]) for k in 'oden')
    sl = v.sub(o - 12, o + d * 50, pad=12); X, Y, Z = v.X[sl], v.Y[sl], v.Z[sl]
    q = (X - o[0], Y - o[1], Z - o[2])
    a = q[0] * d[0] + q[1] * d[1] + q[2] * d[2]; bb = q[0] * e[0] + q[1] * e[1] + q[2] * e[2]
    dep = geo.edt_in(v.kind[sl] > 0)
    vein = v.extra['BLADE'][sl] & (np.abs(bb - .06 * a) < .85) & (a > 7) & (a < 31) & (dep <= 1.0)
    v.col[sl][vein] = cfg.GLOW[3]; v.kind[sl][vein] = 3
    print('BLADE vein glow %d bricks' % int(vein.sum()))


def run(v):
    pauldrons(v); frost_throat(v); frost_heart(v); blade_vein(v)
    return v


if __name__ == '__main__':
    import materials
    v = geo.Vox.load('05_materials'); run(v); v.save('06_features'); materials.finalize(v, '06_features_fin')
