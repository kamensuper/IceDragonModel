"""Stage 4 - armour as real geometry, laid out in body coordinates (S along the spine, TH around it) and raised on the true
body skin (DB = signed distance to the blended body):
  dorsal   : overlapping chevron plates along the ridge (they carry the ice-crystal spines)
  lateral  : staggered rows of pointed scales (shingled, free edge toward the tail)
  ventral  : broad banded belly plates from the throat down the chest and belly to the tail (heavier on the chest)
  legs     : smaller shingled scales in each leg's own coordinates (free edge toward the foot), shin / forearm ridge plates
  spines   : ice crystals standing on the dorsal ridge - tall shards on the neck, the largest cluster over the shoulders,
             shrinking down the back and tail - each with smaller companions.
Scale size is constant in bricks, so thin parts simply carry fewer rows (consistent voxel scale)."""
import os
import numpy as np
import config as cfg, geometry as geo
C = cfg.C
TH_D = .60                         # dorsal band |TH| < TH_D
W_V = 11.5                         # ventral plates: half-width in bricks (arc distance from the belly line)
LEN_D, LEN_L, LEN_V, ROW_H = 9.0, 7.0, 6.5, 5.0
LEN_G, ROW_G = 5.6, 4.2            # leg scales


def fields(v):
    S, TH, RB, DB = (v.extra[k] for k in ('S', 'TH', 'RB', 'DB'))
    L = v.meta['L']
    free = (v.kind == 0) | (v.pid == 1)
    for k in ('HEAD', 'BLADE', 'WING', 'FOOT'):
        if k in v.extra: free &= ~v.extra[k]
    shell = (DB < 4.0) & (DB > -3.0) & free
    ath = np.abs(TH); w = ath * RB
    fade = np.clip((L - 2 - S) / 9, 0, 1) * np.clip((S - 3) / 10, 0, 1)      # relief fades into the skull and the tail collar
    return S, TH, RB, DB, L, shell, ath, w, fade


def layout(v):
    S, TH, RB, DB, L, shell, ath, w, fade = fields(v)
    LEG, LS, LTH = v.extra['LEG'], v.extra['LS'], v.extra['LTH']
    zone = np.zeros(v.n, np.int8); fr = np.zeros(v.n, np.float32); sidx = np.zeros(v.n, np.int32); row = np.zeros(v.n, np.int8)
    swell = np.zeros(v.n, np.float32)
    torso = shell & (S >= 0) & (LEG == 0)
    chest = np.clip(1 - np.abs(S - (v.meta['s_chest'] + 6)) / 34, 0, 1)        # 1 on the chest, 0 on belly / upper neck
    # dorsal chevron plates (V points to the tail)
    dz = torso & (ath < TH_D)
    u = (S + .55 * w) / LEN_D; k = np.floor(u)
    zone[dz] = 1; fr[dz] = (u - k)[dz]; sidx[dz] = k[dz]
    swell[dz] = (.7 + 1.8 * (1 - (u - k)) + 1.0 * (w < .15 * RB))[dz]          # + raised keel along the ridge
    # ventral belly plates: straight bands across the underside, wider + heavier on the chest
    wv = (np.pi - ath) * RB
    vz = torso & (wv < W_V + 3.5 * chest) & (ath > 1.9)
    u = S / LEN_V; k = np.floor(u)
    zone[vz] = 3; fr[vz] = (u - k)[vz]; sidx[vz] = k[vz]
    swell[vz] = (.5 + (1.5 + 1.1 * chest) * (1 - (u - k)))[vz]
    # lateral rows of pointed scales, staggered half a scale per row
    lz = torso & ~dz & ~vz
    wl = (ath - TH_D) * RB; r = np.floor(wl / ROW_H); wc = wl - (r + .5) * ROW_H
    u = (S + .5 * np.abs(wc)) / LEN_L + .5 * r; k = np.floor(u)
    zone[lz] = 2; fr[lz] = (u - k)[lz]; sidx[lz] = k[lz]; row[lz] = np.clip((r + 1) * np.sign(TH), -120, 120)[lz]
    keel = geo.hash3(k.astype(np.int64), (r * np.sign(TH)).astype(np.int64), 3) < .08       # controlled asymmetry: a few keeled scales
    swell[lz] = (.15 + 1.3 * (1 - (u - k)) + .5 * keel)[lz]
    swell *= np.where(torso, fade, 1)
    # legs: shingled scales in leg coordinates (free edge toward the foot); the front ridge carries bigger plates
    gz = shell & (LEG > 0)
    rl = 6.5                                                                   # nominal leg radius for the row layout
    wg = (LTH + np.pi) * rl; r = np.floor(wg / ROW_G); wc = wg - (r + .5) * ROW_G
    u = (LS - .5 * np.abs(wc)) / LEN_G + .5 * r; k = np.floor(u)
    zone[gz] = 5; fr[gz] = (1 - (u - k))[gz]; sidx[gz] = (k + 40 * LEG)[gz]; row[gz] = np.clip(r, 0, 120)[gz]
    swell[gz] = (.1 + 1.15 * (u - k))[gz]
    return zone, fr, sidx, row, swell, torso | gz


def surface_point(v, p, d, DB, tmax=30.0):
    """march from p along d until the body skin (DB > 0); returns the last inside point."""
    last = np.array(p, float)
    for t in np.arange(0, tmax, .5):
        q = p + d * t; ix = np.floor(q - v.lo).astype(int)
        if (ix < 0).any() or (ix >= np.array(v.n)).any(): break
        if DB[tuple(ix)] > 0: break
        last = q
    return last


def spines(v):
    """ice-crystal dorsal crest. Returns (mask, list of crystal records for the rig / README)."""
    sp = np.load(os.path.join(cfg.CKPT, '02_blockout', 'spine.npz'))
    Cc, s, T, N, Bn = (sp[k] for k in ('C', 's', 'T', 'N', 'B')); L = float(sp['L']); DB = v.extra['DB']
    s_hip, s_ch = float(sp['s_hip']), float(sp['s_chest'])
    M = np.zeros(v.n, bool); n = 0; cls = {1: 0, 2: 0, 3: 0}
    k = 0; sk = 14.0
    while sk < L - 9:
        i = int(np.argmin(np.abs(s - sk))); f = sk / L
        # height profile along the spine: tail small -> back -> biggest over the shoulders -> tall narrow on the neck -> small at the skull
        H = float(np.interp(sk, [0, 30, s_hip - 30, s_hip, s_ch - 22, s_ch + 4, s_ch + 30, L - 40, L - 14, L],
                            [4.0, 6.0, 11.0, 13.5, 15.5, 21.0, 15.0, 14.5, 11.0, 7.0]))
        H *= 1.0 if k % 2 == 0 else .72                                        # alternate tall / short = sawtooth crest
        R = max(1.5, H * .2)
        lean = np.radians(30 if sk > s_ch else 36)
        base = surface_point(v, Cc[i], N[i], DB) - N[i] * 1.8
        d = N[i] * np.cos(lean) - T[i] * np.sin(lean)
        M |= geo.crystal(v, base, d, H + 1.8, R, roll=.5 * k, tip=.5); n += 1
        cl = 1 if H > 13 else (2 if H > 7.5 else 3); cls[cl] += 1
        if H > 7.5:                                                            # companions: a splayed pair beside every tall crystal
            for sg in (1, -1):
                b2 = surface_point(v, Cc[i], N[i] * np.cos(.42) + Bn[i] * np.sin(.42) * sg, DB) - N[i] * 1.5
                d2 = N[i] * .78 + Bn[i] * .5 * sg - T[i] * .42
                M |= geo.crystal(v, b2, d2, H * .46 + 1.5, max(1.3, R * .62), roll=.9 * k + sg, tip=.55); n += 1; cls[3] += 1
        sk += float(np.interp(H, [4, 21], [5.2, 9.5])); k += 1
    return M, n, cls


def run(v):
    zone, fr, sidx, row, swell, region = layout(v)
    DB = v.extra['DB']
    skin = region & (DB < swell) & (zone > 0)
    add = skin & (v.kind == 0)
    base = {1: C['NAVY'], 2: C['STEEL'], 3: C['CYANP'], 5: C['STEEL']}
    for zc, rgb in base.items(): v.put(skin & (zone == zc), rgb)
    v.pid[skin] = (zone[skin].astype(np.int32) * 1000000 + (row[skin].astype(np.int32) + 130) * 3000 + sidx[skin] % 3000)
    v.lvl[skin] = zone[skin] + geo.hash3(sidx[skin], row[skin].astype(np.int64), 5) * .9
    zone = np.where(skin, zone, 0).astype(np.int8)
    sm, n, cls = spines(v)
    zone[sm] = 4
    v.extra.update(ZONE=zone, FR=fr, SIDX=sidx, ROW=row, SPINE=sm)
    print('ARMOUR added %d bricks; dorsal crystals %d (large %d / medium %d / small %d), %d bricks' % (int(add.sum()), n, cls[1], cls[2], cls[3], int(sm.sum())))
    return v


if __name__ == '__main__':
    v = geo.Vox.load('03_anatomy'); run(v)
    nc = geo.components(v.kind > 0)[1]
    print('ARMOUR bricks %d components %d %s' % (int((v.kind > 0).sum()), len(nc), nc[:5]))
    v.save('04_armor')
