"""Stage 3b - the frost crown: two great ridged horns sweeping back, out and up (hide cuff -> silver bone -> ice-crystal tips),
swept temple horns, a crest of ice crystals down the midline, cheek-frill shards, brow knobs, a nasal crystal and chin shards.
Horn curves live in head coordinates (x lateral mirrored, y up, z forward); crystals are placed through the head frame."""
import numpy as np
from scipy.spatial import cKDTree
import config as cfg, geometry as geo, head
C = cfg.C


def horn(v, H, pts, r0, r1, grad, band0=4.2, rings=True, lvl=6, keel=False):
    """curved, tapering, ringed horn along a spline (mirrored in x). grad = [(fraction, colour name)], root -> tip."""
    Cv = geo.resample(geo.catmull(pts, .2), .25); s = geo.arclen(Cv); f_ = s / s[-1]
    Rs = (r0 + (r1 - r0) * f_ ** .9) * np.clip((1 - f_) / .05, .45, 1)
    u = np.concatenate([[0], np.cumsum(np.diff(s) / (band0 * (1 - .5 * f_[1:])))])               # rings tighten toward the tip
    P = np.stack([H.ax.ravel(), H.y.ravel(), H.z.ravel()], 1)
    near = np.flatnonzero(((P > Cv.min(0) - r0 * 1.5 - 2) & (P < Cv.max(0) + r0 * 1.5 + 2)).all(1))
    d, i = cKDTree(Cv).query(P[near], workers=-1)
    f = f_[i]; fr = u[i] - np.floor(u[i]); R = Rs[i]
    Rr = R * (1 + (.17 * (fr < .45) * (f < .88) if rings else 0))                                # each growth ring: raised lip on its root side
    ins = d < Rr
    m = np.zeros(P.shape[0], bool); m[near[ins]] = True; m = m.reshape(H.kind.shape)
    f, fr = f[ins], fr[ins]
    name = np.array([n_ for _, n_ in grad])[np.searchsorted([e for e, _ in grad], f).clip(0, len(grad) - 1)]
    rgb = np.array([C[n_] for n_ in name])
    if rings:
        gr = (fr > .76) & (f < .88)                                                             # thin dark line in each ring groove
        for a_, b_ in (('SILV', 'SILVD'), ('SILVD', 'STEEL'), ('STEEL', 'NAVY'), ('ICE', 'ICED'), ('ICED', 'ICEDD')): rgb[gr & (name == a_)] = C[b_]
        hi = (fr < .16) & (name == 'SILVD'); rgb[hi] = C['SILV']                                 # lit lip
    H.col[m] = rgb; H.kind[m] = 1; H.pid[m] = v.newid(); H.lvl[m] = lvl
    return m


def shard(v, H, p, d, ln, r, G=None, **kw):
    """mirrored pair of ice crystals given in head coords (p = base, d = direction)."""
    M = np.zeros(v.n, bool)
    for sx in ((1, -1) if abs(p[0]) > .01 or abs(d[0]) > .01 else (1,)):
        w = H.world((p[0] * sx, p[1], p[2])); dw = H.r * d[0] * sx + H.u * d[1] + H.f * d[2]
        M |= geo.crystal(v, w, dw, ln * H.s, r * H.s, roll=.3 * sx, **kw)
    return M


def run(v):
    H, J = head.frames(v)
    HORN = [(.07, 'STEEL'), (.13, 'STEELL'), (.52, 'SILVD'), (.74, 'SILV'), (.86, 'ICED'), (.95, 'ICE'), (1.0, 'FROST')]
    M = np.zeros(H.kind.shape, bool)
    # great horns: massive root behind the brow, back along the skull, then out, the last third hooking up
    M |= horn(v, H, [(6.8, 6.5, 10.5), (9.0, 11.5, 4.0), (12.0, 16.0, -6.0), (16.5, 20.0, -18.5), (21.5, 24.5, -30.0), (25.0, 31.5, -38.0),
                     (26.0, 40.0, -42.0), (24.5, 48.0, -42.5)], 5.3, 1.3, HORN)
    # temple horns: lower, swept back and slightly down, then up at the tip
    M |= horn(v, H, [(10.6, 4.0, 4.0), (14.0, 5.0, -5.0), (18.0, 7.0, -16.0), (21.0, 11.0, -25.0), (22.0, 16.5, -31.0)], 3.2, .9, HORN, band0=3.4)
    # brow horns: short forward echo above the eyes
    M |= horn(v, H, [(8.8, 8.6, 21.0), (10.4, 12.5, 18.5), (12.6, 15.5, 14.0), (14.4, 19.5, 11.5)], 2.3, .7,
              [(.3, 'STEELL'), (.75, 'SILVD'), (1.0, 'SILV')], band0=3.0)
    Mg = H.full(M)
    # crest of ice crystals down the midline of the crown (tallest in the middle), leaning back
    for z, h, r in ((15.0, 6.5, 1.9), (9.0, 10.5, 2.4), (2.5, 13.0, 2.7), (-4.0, 10.0, 2.3)):
        Mg |= shard(v, H, (0, head.up_t(z) - 1.5, z), (0, 1, -.55), h, r)
    # cheek frill: three flat ice shards per side fanning back from the cheek shield
    for k, (y, ln, r) in enumerate(((4.0, 17, 2.6), (-.5, 14, 2.3), (-4.5, 10.5, 1.9))):
        Mg |= shard(v, H, (11.2, y, 4.0), (.48 + .05 * k, .30 - .26 * k, -1.0), ln, r, flat=.62, tip=.6)
    # nasal crystal + small brow knobs + chin shards
    Mg |= shard(v, H, (0, head.up_t(46) - .8, 46.0), (0, 1, -.35), 6.5, 1.7)
    kid = v.newid()
    for z in (17.5, 21.5, 25.5, 29.5):
        k_ = geo.sd_ell(H.ax, H.y, H.z, (head.up_w(z) - 1.3 + (z < 27) * .9, 9.4 + (22 - z) * .16, z), (1.5, 1.5, 1.5)) < 0
        H.put(k_, C['CYANP'], p=kid, l=7); Mg |= H.full(k_)
    chin = np.zeros(v.n, bool)
    for z, ln, r in ((40.0, 6.0, 1.5), (30.0, 8.5, 1.9), (19.0, 11.0, 2.2)):
        jb = float(np.interp(z, head.JW[:, 0], head.JW[:, 1]))
        for sx in (1, -1):
            w = J.world((sx * (2.2 + (40 - z) * .12), jb + 1.2, z)); dw = J.r * sx * .25 - J.u * 1.0 - J.f * .55
            chin |= geo.crystal(v, w, dw, ln * J.s, r * J.s, roll=.4 * sx)
    v.extra['HORN'] = Mg | chin
    v.extra['HEAD'] = v.extra['HEAD'] | v.extra['HORN']
    v.extra['JAW'] = v.extra['JAW'] | chin
    return v
