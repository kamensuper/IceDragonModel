"""Stage 3d - tail weapon: an ice-crystal blade cluster grown from the raised tail tip - one great flattened blade crystal,
two flanking blades, a rosette of shards round a silver collar. Frame: d = along the tail (out of the tip), e = blade width
(toward the dragon's back), n = blade thickness."""
import os
import numpy as np
import config as cfg, geometry as geo
C = cfg.C


def run(v):
    sp = np.load(os.path.join(cfg.CKPT, '02_blockout', 'spine.npz'))
    o = sp['C'][0]; d = -sp['T'][0]; n = np.array((-.62, 0, -.78)); n -= d * n.dot(d); n /= np.linalg.norm(n); e = np.cross(n, d)   # blade faces the 3/4 camera
    def dirv(a, b=0.0, c=0.0): w = d * a + e * b + n * c; return w / np.linalg.norm(w)
    M = np.zeros(v.n, bool)
    # silver collar: two stepped rings where the hide ends
    sl = v.sub(o - 16, o + 16); X, Y, Z = v.X[sl], v.Y[sl], v.Z[sl]
    q = (X - o[0], Y - o[1], Z - o[2]); a = q[0] * d[0] + q[1] * d[1] + q[2] * d[2]
    rad = np.sqrt(np.maximum(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 - a * a, 0))
    for a0, a1, r, cn, lv in ((-7.5, -4.5, 4.4, 'SILVD', 5), (-4.5, -1.5, 5.2, 'SILV', 6), (-1.5, 1.5, 6.2, 'SILVD', 7)):
        m = (a > a0) & (a < a1) & (rad < r)
        v.col[sl][m] = C[cn]; v.kind[sl][m] = 1; v.pid[sl][m] = v.newid(); v.lvl[sl][m] = lv; M[sl] |= m
    # rosette: six shards splayed round the collar
    for k in range(6):
        an = np.pi / 3 * k + .3; ln = 13.0 if k % 2 else 10.0
        M |= geo.crystal(v, o + (e * np.cos(an) + n * np.sin(an)) * 3.0, dirv(.75, np.cos(an) * .8, np.sin(an) * .8), ln, 2.4, roll=an)
    # the great blade (flattened hexagonal crystal) + two flanking blades + inner spikes
    M |= geo.crystal(v, o - d * 2.0, dirv(1, .06), 46.0, 8.6, tip=.5, flat=.42, base=1.0, wide=e)
    M |= geo.crystal(v, o + e * 2.5, dirv(1, .42), 29.0, 5.2, tip=.5, flat=.5, wide=e)
    M |= geo.crystal(v, o - e * 2.5, dirv(1, -.46), 24.0, 4.6, tip=.5, flat=.5, wide=e)
    M |= geo.crystal(v, o + n * 2.0, dirv(1, .1, .5), 17.0, 3.0, roll=.5)
    M |= geo.crystal(v, o - n * 2.0, dirv(1, -.1, -.5), 15.0, 2.8, roll=.9)
    v.extra['BLADE'] = M
    v.meta['blade'] = dict(o=o.tolist(), d=d.tolist(), e=e.tolist(), n=n.tolist())
    print('TAIL crystal cluster %d bricks' % int(M.sum()))
    return v
