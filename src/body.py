"""Stage 2 - body blockout: one smooth spine (tail tip -> tail -> torso -> S-neck -> skull base) voxelised as a tube with
blended muscle masses (deep chest, shoulders, haunches), four legs as tapered tubes with their own muscle masses, and feet.
Per-voxel coordinates are stored for the armour stage:
  S / TH / D / RB   spine tube: arc length from the tail tip, angle around (0 = dorsal ridge, +-pi = belly), distance above the tube skin, radius
  DB                signed distance to the finished body skin (spine + masses + legs, smooth union)
  LEG / LS / LTH    leg number 1..4 (0 = torso owns the voxel), arc length down the leg, angle around the leg (0 = outer side)"""
import os
import numpy as np
import config as cfg, geometry as geo
C = cfg.C
LEG_NAMES = ('FL', 'FR', 'HL', 'HR')


def spine():
    P = np.array(cfg.TAIL + cfg.TORSO + cfg.NECK, float)
    Cc = geo.catmull(P[:, :3], .2)
    for _ in range(30): Cc[1:-1] = (Cc[:-2] + 2 * Cc[1:-1] + Cc[2:]) / 4
    Cc = geo.resample(Cc, .3); s = geo.arclen(Cc); L = s[-1]
    # arc position of every control point -> radius profile + section boundaries
    sp = np.array([s[np.argmin(np.linalg.norm(Cc - p, axis=1))] for p in P[:, :3]]); sp = np.maximum.accumulate(sp)
    R = np.interp(s, sp, P[:, 3])
    for _ in range(120): R[1:-1] = (R[:-2] + R[1:-1] + R[2:]) / 3
    nt, nb = len(cfg.TAIL), len(cfg.TAIL) + len(cfg.TORSO)
    s_hip, s_chest, s_neck2, s_neck4 = sp[nt], sp[nb - 1], sp[nb + 2], sp[nb + 4]
    # dorsal reference: up on tail + torso, toward the back (+z) where the neck (and the raised tail tip) climb
    k = np.interp(s, [0, sp[1], sp[3], s_chest, s_neck2, s_neck4, L], [1.0, 1.0, 0, 0, 1.3, 1.0, .25])
    ref = np.stack([np.zeros_like(s), np.ones_like(s), k], 1)
    T, N, Bn = geo.frames(Cc, ref)
    return dict(C=Cc, s=s, T=T, N=N, B=Bn, R=R, sp=sp, s_hip=s_hip, s_chest=s_chest, L=L)


def leg_curve(name):
    g = cfg.LEGS[name]; P = np.array(g['P'], float)
    Cc = geo.resample(geo.catmull(P, .2), .3); s = geo.arclen(Cc)
    sp = np.array([s[np.argmin(np.linalg.norm(Cc - p, axis=1))] for p in P])
    R = np.interp(s, sp, g['R'])
    for _ in range(40): R[1:-1] = (R[:-2] + R[1:-1] + R[2:]) / 3
    out = np.array((np.sign(P[0, 0]), 0, 0.0))                           # TH = 0 on the outer side of the leg
    T, N, Bn = geo.frames(Cc, out)
    return dict(C=Cc, s=s, T=T, N=N, B=Bn, R=R, sp=sp)


def leg_masses(name, X, Y, Z):
    """extra muscle on top of the leg tube: thigh / drumstick / forearm."""
    P = np.array(cfg.LEGS[name]['P'], float); sx = np.sign(P[0, 0])
    if name[0] == 'H':
        th = (P[0] + P[1]) / 2 + np.array((sx * .5, 1.5, 3.0))
        d = geo.sd_ell(X, Y, Z, th, (8.4, 14.5, 13.5))                    # big flat thigh
        d = np.minimum(d, geo.sd_ell(X, Y, Z, (P[1] * .62 + P[2] * .38) + np.array((0, 0, 1.5)), (5.6, 8.5, 6.6)))   # drumstick
    else:
        d = geo.sd_ell(X, Y, Z, (P[0] * .6 + P[1] * .4) + np.array((sx * .5, 0, 0)), (7.4, 12.5, 9.8))                 # upper arm
        d = np.minimum(d, geo.sd_ell(X, Y, Z, (P[1] * .6 + P[2] * .4) + np.array((0, 0, -1.0)), (5.2, 8.5, 6.0)))    # forearm
    return d


def foot(v, name, lid):
    """paw + 4 splayed toes + curved claws + heel spur, in a ground-level frame (x lateral, y up from the ground, z forward)."""
    g = cfg.LEGS[name]; P = np.array(g['P'], float); yaw = np.radians(g['yaw']); gy = g['ground']
    f = np.array((-np.sin(yaw), 0, -np.cos(yaw)))
    o = np.array((P[3, 0], gy, P[3, 2])); k = 1.12 if name[0] == 'H' else 1.0
    F = geo.Frame(v, o, f, (0, 1, 0), o - 30, o + 30, scale=k)
    x, y, z = F.x, F.y, F.z
    pid = v.newid(); M = np.zeros(F.kind.shape, bool)
    paw = (geo.sd_ell(x, y, z, (0, 3.4, .5), (6.6, 4.4, 6.6)) < 0) & (y > 0)
    F.put(paw, C['STEEL'], p=pid, l=1); M |= paw
    side = -np.sign(P[0, 0])                                              # inner toe is a bit shorter
    for i, an in enumerate((-33, -11, 11, 33)):
        a = np.radians(an); dx, dz = np.sin(a), np.cos(a); ln = 1.0 - .1 * (an * side > 20) - .04 * (abs(an) > 20)
        k0 = (dx * 3.2, 3.2, dz * 3.2 + 1.0); k1 = (dx * 8.5 * ln, 4.4, dz * 8.5 * ln + 1.2); k2 = (dx * 12.6 * ln, 2.6, dz * 12.6 * ln + 1.2)
        toe = (geo.sd_chain(x, y, z, [k0, k1, k2], [2.9, 2.7, 2.15]) < 0) & (y > 0)
        tp = v.newid(); F.put(toe, C['STEEL'], p=tp, l=2 + i % 2); M |= toe
        F.put(toe & (geo.sd_ell(x, y, z, (k1[0], k1[1] + 1.5, k1[2]), (2.3, 1.7, 2.6)) < 0), C['STEELL'])            # knuckle scale
        c0 = (dx * 12.4 * ln, 3.2, dz * 12.4 * ln + 1.2); c1 = (dx * 15.6 * ln, 3.7, dz * 15.6 * ln + 1.2); c2 = (dx * 18.6 * ln, .5, dz * 18.6 * ln + 1.2)
        d1, t1 = geo.sd_cap(x, y, z, c0, c1, 1.9, 1.45); d2, t2 = geo.sd_cap(x, y, z, c1, c2, 1.45, .45)
        claw = ((d1 < 0) | (d2 < 0)) & (y > 0)
        cp = v.newid(); F.put(claw, C['SILV'], p=cp, l=4); F.put(claw & (d1 < 0) & (t1 < .45), C['SILVD']); F.put(claw & (d2 < 0) & (t2 > .45), C['FROST'])
        M |= claw; v.mask('CLAW')[F.sl] |= claw
    d1, t1 = geo.sd_cap(x, y, z, (side * 2.0, 4.2, -5.0), (side * 3.4, 4.8, -9.5), 1.9, .5)                           # heel spur (dewclaw)
    sp = (d1 < 0) & (y > 0); F.put(sp, C['SILV'], p=v.newid(), l=4); F.put(sp & (t1 > .6), C['FROST']); M |= sp
    v.mask('CLAW')[F.sl] |= sp
    v.mask('FOOT')[F.sl] |= M
    v.extra['LEG'][F.sl][M] = lid
    return o, f


def run():
    v = geo.Vox(); sp = spine()
    sl, S, TH, D, RB = geo.tube_fields(v, sp['C'], sp['R'], sp['T'], sp['N'], sp['B'], pad=16.0)
    for k_, a in (('S', S), ('TH', TH), ('D', D), ('RB', RB)):
        f = np.full(v.n, -1 if k_ == 'S' else (99 if k_ == 'D' else 0), np.float32); f[sl] = a; v.extra[k_] = f
    X, Y, Z = v.X, v.Y, v.Z; AX = np.abs(X)
    d = v.extra['D'].copy()
    # torso masses (mirror-symmetric): deep chest keel, shoulder blocks, wing-root muscles, haunches, belly tuck
    for c, r, k in (((0, 65, -13), (15.5, 18.5, 15.5), 7), ((14.5, 69, -12), (9.5, 13, 12.5), 6), ((9.5, 81.5, -8), (7.5, 6.5, 10), 5),
                    ((13.5, 59, 33), (10, 14.5, 14.5), 6), ((0, 60, 12), (14.5, 14.5, 20), 7), ((0, 87, -24), (9.5, 9, 8), 6)):
        d = geo.smin(d, geo.sd_ell(AX, Y, Z, c, r), k)
    body_d = d.copy()
    # legs
    v.extra['LEG'] = np.zeros(v.n, np.int8); v.extra['LS'] = np.zeros(v.n, np.float32); v.extra['LTH'] = np.zeros(v.n, np.float32)
    leg_d = np.full(v.n, 99.0, np.float32); legs = {}
    for i, name in enumerate(LEG_NAMES):
        lc = leg_curve(name); legs[name] = lc
        lsl, LS, LTH, LD, LRB = geo.tube_fields(v, lc['C'], lc['R'], lc['T'], lc['N'], lc['B'], pad=12.0, open_ends=(False, False))
        LD = geo.smin(LD, leg_masses(name, X[lsl], Y[lsl], Z[lsl]), 4.0)
        LD = np.where(Y[lsl] < cfg.LEGS[name]['ground'], 99.0, LD)
        own = LD < leg_d[lsl]
        leg_d[lsl] = np.where(own, LD, leg_d[lsl])
        v.extra['LEG'][lsl] = np.where(own & (LD < 6), i + 1, v.extra['LEG'][lsl])
        v.extra['LS'][lsl] = np.where(own, LS, v.extra['LS'][lsl]); v.extra['LTH'][lsl] = np.where(own, LTH, v.extra['LTH'][lsl])
    d = geo.smin(body_d, leg_d, 4.5)
    v.extra['LEG'] = np.where(leg_d < body_d + 1.5, v.extra['LEG'], 0).astype(np.int8)       # the torso owns everything the legs do not clearly stick out of
    v.extra['DB'] = d.astype(np.float32)
    body = d < 0
    v.put(body, C['NAVY'], p=1, l=0)
    feet = {}
    for i, name in enumerate(LEG_NAMES):
        o, f = foot(v, name, i + 1); feet[name] = dict(o=o.tolist(), f=f.tolist())
    v.meta.update(L=float(sp['L']), s_hip=float(sp['s_hip']), s_chest=float(sp['s_chest']), feet=feet)
    os.makedirs(os.path.join(cfg.CKPT, '02_blockout'), exist_ok=True)
    np.savez(os.path.join(cfg.CKPT, '02_blockout', 'spine.npz'), **sp)
    for name, lc in legs.items(): np.savez(os.path.join(cfg.CKPT, '02_blockout', 'leg_%s.npz' % name), **lc)
    nc = geo.components(v.kind > 0)[1]
    print('BLOCKOUT spine length %.1f (hip %.1f chest %.1f)  body bricks %d  components %d %s' % (sp['L'], sp['s_hip'], sp['s_chest'], int(body.sum()), len(nc), nc[:4]))
    v.save('02_blockout')


if __name__ == '__main__':
    run()
