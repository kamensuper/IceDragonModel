"""Stage 3a - skull: wedge snout with a nasal hump, wide cheeks, heavy brows, recessed glowing eyes, layered plates, nostrils,
hinged lower jaw (open), recessed indigo mouth, ice fangs + secondary teeth, tongue. Built in the head-local frame
(x lateral, y up, z forward, origin = skull base); the frame is yawed + pitched, so nothing here assumes world axes."""
import numpy as np
import config as cfg, geometry as geo
C = cfg.C
HS = .96                                    # head scale about the skull base
HINGE = (0.0, -5.0, 2.0)                    # jaw hinge (x, y, z) in head coords

# (z, bottom, top, half-width) of the upper skull; bottom = upper lip line
UP = np.array([(-8, -6, 8, 9.6), (0, -6, 10.6, 12), (8, -5.6, 11.8, 13.2), (16, -5.2, 11.4, 13.2), (21, -5, 10.4, 12.6),
               (26.5, -4.6, 8.2, 8.8), (36, -4.1, 6.6, 8.0), (45, -3.6, 5.0, 6.8), (51, -3.2, 3.2, 5.2), (54.5, -3, .4, 3.4)])
# (z, bottom, half-width) of the lower jaw in jaw coords (top = 0 = lower lip line)
JW = np.array([(-5, -8, 10.2), (4, -9, 11.8), (12, -8.6, 12.2), (22, -7.4, 10.6), (30, -6.4, 7.8), (40, -5.4, 6.0), (47, -4.8, 4.6), (51.5, -3.4, 3.0)])


def head_basis():
    yw, pt = np.radians(cfg.HEAD_YAW), np.radians(cfg.HEAD_PITCH)
    f = np.array((-np.sin(yw) * np.cos(pt), np.sin(pt), -np.cos(yw) * np.cos(pt)))
    return f, np.array((0, 1.0, 0))


def frames(v):
    h = v.meta['head']; o, f, u = (np.array(h[k]) for k in 'ofu')
    p0, p1 = o - 78, o + 78
    H = geo.Frame(v, o, f, u, p0, p1, scale=HS)
    a = np.radians(cfg.JAW_OPEN)
    jo = o + (H.u * HINGE[1] + H.f * HINGE[2]) * HS
    J = geo.Frame(v, jo, H.f * np.cos(a) - H.u * np.sin(a), H.u * np.cos(a) + H.f * np.sin(a), p0, p1, scale=HS)
    return H, J


def up_w(z): return np.interp(z, UP[:, 0], UP[:, 3])
def up_b(z): return np.interp(z, UP[:, 0], UP[:, 1])
def up_t(z): return np.interp(z, UP[:, 0], UP[:, 2])
def jw_w(z): return np.interp(z, JW[:, 0], JW[:, 2])


def skull_mask(H):
    b = up_b(H.z); t = up_t(H.z); w = up_w(H.z)
    tt = np.clip((H.y - b) / (t - b), 0, 2)
    px = np.interp(H.z, [24, 33], [2.6, 1.8])                           # boxy cranium -> wedge-sided snout
    m = ((H.ax / w) ** px + tt ** 3 < 1) & (H.y > b) & (H.z > UP[0, 0]) & (H.z < UP[-1, 0])
    for c, r in (((12.2, -.5, 9.5), (5.0, 6.2, 9.0)), ((9.5, 6.5, 6), (4.5, 4.5, 8))):       # cheek bulge, temple bulge
        m |= (geo.sd_ell(H.ax, H.y, H.z, c, r) < 0) & (H.y > b)
    m |= (geo.sd_ell(H.ax, H.y, H.z, (0, 4.2, 41), (3.4, 3.6, 7.5)) < 0) & (H.y > b)         # nasal hump
    return m


def jaw_mask(J):
    b = np.interp(J.z, JW[:, 0], JW[:, 1]); w = jw_w(J.z)
    tt = np.clip(-J.y / -b, 0, 2)
    m = ((J.ax / w) ** 2.6 + tt ** 3 < 1) & (J.y < 0) & (J.z > JW[0, 0]) & (J.z < JW[-1, 0])
    m |= geo.sd_ell(J.ax, J.y, J.z, (10.2, -3.2, 7), (3.4, 4.6, 7.5)) < 0                    # jaw muscle
    return m


def build_skull(v, H, J):
    sk = skull_mask(H); jw = jaw_mask(J)
    hid, jid = v.newid(), v.newid()
    th = geo.sd_ell(H.ax, H.y, H.z, (0, -5.5, -1), (10.0, 7.5, 9.5)) < 0                     # throat bag joins jaw, skull and neck
    H.put(th & (H.kind == 0), C['NAVY'], p=1, l=0)
    H.put(jw, C['STEEL'], p=jid, l=1)
    H.put(jw & (J.y < np.interp(J.z, JW[:, 0], JW[:, 1]) * .55), C['NAVY'], p=v.newid(), l=0)      # darker underside
    H.put(sk, C['NAVY'], p=hid, l=2)
    return sk, jw


def carve_mouth(v, H, J, sk, jw):
    """recessed oral cavity: palate and jaw trough carved inside the lip rims, indigo lining, dark back wall."""
    RIM = 2.6
    pal = sk & (H.z > 9) & (H.z < 50) & (H.ax < up_w(H.z) - RIM) & (H.y < up_b(H.z) + 2.4)
    tro = jw & (J.z > 9) & (J.z < 47.5) & (J.ax < jw_w(J.z) - RIM) & (J.y > -2.0)
    H.kind[pal | tro] = 0
    H.put(sk & ~pal & (H.z > 6) & (H.z < 50) & (H.ax < up_w(H.z) - RIM) & (H.y < up_b(H.z) + 4.4), C['VIOD'], p=v.newid(), l=1)   # palate
    J.put(jw & ~tro & (J.z > 6) & (J.ax < jw_w(J.z) - RIM) & (J.y > -4.4), C['VIOD'], p=v.newid(), l=1)                      # mouth floor
    wedge = (H.y < up_b(H.z) + .5) & (J.y > -.5) & (H.z > 1) & (H.z < 11) & (H.ax < 10.3)                                     # gullet wall joins the jaws
    H.put(wedge, C['VIOD'], p=v.newid(), l=0)
    H.put(wedge & (H.z > 8.5) & (H.ax < 6.5), C['SEAM'])
    mem = (H.y < up_b(H.z) + .5) & (J.y > -.5) & (H.z > 6) & (H.z < 17) & (H.ax < np.minimum(up_w(H.z), jw_w(J.z)) - 1.6) & (H.ax > 6)
    H.put(mem & (H.kind == 0), C['VIOD'], p=v.newid(), l=0)                                                                    # mouth-corner membrane
    v.extra['MOUTH'] = H.full(pal | tro | wedge)
    v.extra['GULLET'] = H.full(wedge & (H.z > 8.5) & (H.ax < 4.2))


def teeth(v, H, J):
    tid = v.newid(); T = np.zeros(H.kind.shape, bool)
    FANG = ((46.0, 12.5, 2.2), (30.0, 6.0, 1.5))                         # (z, length, base radius) upper fangs
    for z, ln, r in FANG:                                                # large curved fangs: down, then hooking back
        x = up_w(z) - 1.5; b = up_b(z)
        T |= H.chain([(x, b + 1.5, z), (x, b - ln * .55, z + .8), (x - .5, b - ln, z - 1.6)], [r, r * .72, .5])
    for z in np.arange(13.5, 52.5, 4.1):                                 # secondary upper teeth
        if min(abs(z - f[0]) for f in FANG) < 2.6: continue
        x = up_w(z) - 1.35; b = up_b(z); ln = 3.4 - (48 - z) * .035
        T |= H.chain([(x, b + 1, z), (x, b - ln, z - .4)], [.9, .4])
    T |= H.chain([(1.6, up_b(53.5) + 1, 53.2), (1.6, up_b(53.5) - 3.2, 53.0)], [.9, .45])        # front incisors
    TU = T.copy()
    for z, ln, r in ((46.5, 7.0, 1.55), (27.0, 4.8, 1.25)):              # lower fangs
        x = jw_w(z) - 1.4
        T |= J.chain([(x, -1.5, z), (x, ln * .6, z - .2), (x - .3, ln, z - 1.4)], [r, r * .7, .5])
    for z in np.arange(20.4, 44.5, 4.1):
        if abs(z - 27) < 2.6: continue
        x = jw_w(z) - 1.3; ln = 3.0 - (44 - z) * .03
        T |= J.chain([(x, -1, z), (x, ln, z - .3)], [.85, .4])
    H.put(T, C['FROST'], p=tid, l=3)
    v.extra['TEETH'] = H.full(T); v.extra['TEETH_LO'] = H.full(T & ~TU)


def tongue(v, J):
    tid = v.newid()
    m = J.chain([(0, -2.2, 10), (0, -1.9, 26), (0, -1.2, 38), (0, .2, 46), (0, 2.2, 52)], [2.7, 2.5, 2.2, 1.9, 1.5], sym=False)
    m |= J.chain([(0, 2.2, 52), (2.2, 3.4, 57), (3.3, 3.2, 61.5)], [1.45, 1.05, .8])              # forked tips (mirrored)
    m &= ~((J.y < -2.4) & (J.z < 46))                                                           # flat underside in the trough
    J.put(m, C['VIO'], p=tid, l=2)
    J.put(m & (J.ax < .6) & (J.z < 51) & (J.y > -1), C['VIOD'])                                   # centre groove
    v.extra['TONGUE'] = J.full(m)


def plates(v, H, sk):
    """layered skull armour: every plate is real geometry (1-3 bricks proud of the skull) with its own part id for seams."""
    out = geo.edt_out(sk); top = up_t(H.z)
    def plate(region, thick, rgb, lvl):
        m = region & (out > 0) & (out <= thick) & (H.kind == 0)
        H.put(m, rgb, p=v.newid(), l=lvl); return m
    up = H.y > top - 4.5
    plate(up & (H.z > 29.5) & (H.z < 53) & (H.ax < 3.3 - (H.z - 29) * .03), 1.8, C['STEEL'], 4)                        # nasal ridge
    for a, b in ((30.5, 36.5), (37.5, 43.5), (44.5, 50)):                                                              # snout side plates
        plate((H.y > up_b(H.z) + 2.6) & (H.z > a) & (H.z < b) & (H.ax > 4.2), 1.0, C['STEEL'], 3)
    plate(up & (H.z > 13) & (H.z < 29) & (H.ax < 7.4 - np.abs(H.z - 21) * .12), 1.0, C['STEEL'], 4)                    # frontals
    plate((H.y > 5.5) & (H.z > -7.5) & (H.z < 12 - (H.ax * .35)) & (H.ax < 11), 1.5, C['STEEL'], 5)                    # crown plate
    plate((H.y > -4.4) & (H.y < 3.0 - (H.z - 4) * .12) & (H.z > 1.5) & (H.z < 19.5) & (H.ax > 8), 1.5, C['STEEL'], 4)  # cheek shields
    br = plate((H.z > 14.5) & (H.z < 33.5) & (H.ax > 4.6) & (H.y > 5.4 + (22 - H.z) * .24) & (H.y < 11.0 + (22 - H.z) * .1), 3.0, C['STEELL'], 6)  # brow armour
    H.put(br & (H.y < 6.6 + (22 - H.z) * .24), C['SEAM'])                                                              # brow underside in shadow
    lip = sk & (H.kind == 1) & (H.y < up_b(H.z) + 2.6) & (H.ax > up_w(H.z) - 2.4) & (H.z > 12)                         # labial scales
    H.pid[lip] = 400 + np.floor(H.z[lip] / 3.3).astype(int); H.lvl[lip] = 3 + np.floor(H.z[lip] / 3.3) % 2
    H.col[lip] = C['STEEL']


def jaw_detail(v, J, jw):
    lip = jw & (J.kind == 1) & (J.y > -2.6) & (J.ax > jw_w(J.z) - 2.4) & (J.z > 10)
    J.pid[lip] = 500 + np.floor(J.z[lip] / 3.3).astype(int); J.lvl[lip] = 1 + np.floor(J.z[lip] / 3.3) % 2
    out = geo.edt_out(jw)
    m = (out > 0) & (out <= 1.2) & (J.kind == 0) & (J.y < -2.8) & (J.y > -9) & (J.z > 2) & (J.z < 21) & (J.ax > 7.5)   # jaw shield
    J.put(m, C['STEELL'], p=v.newid(), l=3)


def eyes(v, H, sk):
    """recessed glowing eyes under the brow: one almond per side, sunk one brick into the skull where the cranium steps in to
    the snout (so it reads from the front, the side and 3/4), dark socket ring, vertical slit pupil (solid)."""
    dep = geo.edt_in(sk); out = geo.edt_out(sk)
    zc = 25.2
    yc = 2.6 - .17 * (H.z - zc)                                           # outer (rear) corner high = fierce
    hh = 2.75 * np.sqrt(np.clip(1 - ((H.z - zc) / 5.2) ** 2, 0, 1))
    side = (H.ax > 4.5) & (H.z > 19.4) & (H.z < 31)
    alm = side & (np.abs(H.y - yc) < hh)
    H.kind[alm & (dep <= 1.0) & (out <= 3.2)] = 0                         # recess (also clears any plate lip in front of the eye)
    ring = side & (np.abs(H.y - yc) < hh + 1.4) & (np.abs(H.z - zc) < 6.8) & ~alm & sk & (dep <= 1.7)
    H.put(ring, C['SEAM'], p=v.newid(), l=1)
    eye = alm & sk & (dep > 1.0) & (dep <= 2.6)
    H.put(eye, cfg.GLOW[2], k=2)
    H.put(alm & sk & (dep > 2.6) & (dep <= 4.0), C['SEAM'], p=v.newid(), l=0)                 # backing
    H.put(eye & (np.abs(H.z - zc - .3) < .62), C['SEAM'], k=1)                                # slit pupil
    v.extra['EYE'] = H.full(eye)


def nostrils(v, H):
    dep = geo.edt_in(H.kind > 0)
    n_ = (H.ax > 1.9) & (H.ax < 3.8) & (H.z > 49.2) & (H.y > -.4 + (H.ax - 1.9) * .9) & (H.y < 2.4 + (H.ax - 1.9) * .9) & (H.kind > 0)
    H.kind[n_ & (dep <= 1.25)] = 0
    H.put(n_ & (dep > 1.25) & (dep <= 2.5), C['SEAM'], p=v.newid(), l=0)
    H.put((H.ax > 1.4) & (H.ax < 4.6) & (H.z > 46.6) & (H.z < 48.8) & (H.y > 3.0) & (H.y < 6.4) & (H.kind > 0), C['STEELL'], p=v.newid(), l=5)   # raised rim


def run(v):
    sp = np.load(__import__('os').path.join(cfg.CKPT, '02_blockout', 'spine.npz'))
    f, u = head_basis()
    v.meta['head'] = dict(o=sp['C'][-1].tolist(), f=f.tolist(), u=u.tolist())
    H, J = frames(v)
    v.meta['head'].update(r=H.r.tolist(), u=H.u.tolist(), jo=J.o.tolist(), jf=J.f.tolist(), ju=J.u.tolist(), hs=HS)
    sk, jw = build_skull(v, H, J)
    plates(v, H, sk); jaw_detail(v, J, jw)
    carve_mouth(v, H, J, sk, jw)
    teeth(v, H, J); tongue(v, J)
    eyes(v, H, sk); nostrils(v, H)
    box = (H.ax < 46) & (H.y > -40) & (H.y < 50) & (H.z < 72)
    v.extra['HEAD'] = H.full(box & (H.z > -8) & (H.kind > 0) & ((H.pid != 1) | (H.z > 0)))
    jm = box & (J.y < .3) & (J.z > 1) & (H.y < up_b(H.z) + .5) & (J.kind > 0)                  # lower jaw body ...
    v.extra['JAW'] = (H.full(jm) & v.extra['HEAD']) | v.extra['TONGUE'] | v.extra['TEETH_LO']   # ... + tongue + lower teeth
    return v
