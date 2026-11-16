"""Stage 5b - materials: colour jobs per armour zone (countershaded navy -> steel flanks, deep-ice tiger bands, pale banded belly,
rime on the upward scale lips), then the skill's finishing pass: surface seams, 26-connectivity flood fill, snap to the palette."""
import numpy as np
import config as cfg, geometry as geo, armor
C = cfg.C


def is_col(v, name): return np.abs(v.col - C[name]).sum(-1) < .02


def scale_pattern(v):
    Z, FR, S, RB, TH, LS, LTH = (v.extra[k] for k in ('ZONE', 'FR', 'S', 'RB', 'TH', 'LS', 'LTH'))
    arm = (v.kind == 1) & (Z > 0) & (v.pid >= 1000000)
    lat = arm & (Z == 2); dor = arm & (Z == 1); ven = arm & (Z == 3); leg = arm & (Z == 5)
    ath = np.abs(TH); wl = (ath - armor.TH_D) * RB
    t = (ath - armor.TH_D) / (np.pi - armor.TH_D)                         # 0 at the ridge -> 1 at the belly line
    lip = FR < np.where(Z == 2, .30, .24)
    up = geo.up_open(v.kind)
    # --- flanks: countershading in three tones, every scale's free edge one tone lighter
    v.col[lat] = C['STEEL']; v.col[lat & (t < .30)] = C['NAVY']; v.col[lat & (t > .74)] = C['STEELL']
    v.col[lat & lip] = C['STEELL']; v.col[lat & lip & (t < .30)] = C['STEEL']; v.col[lat & lip & (t > .74)] = C['GLAC']
    # deep-ice tiger bands sweeping back from the ridge at a fixed rhythm (ordered, never random speckle)
    band = (np.mod((S + 1.15 * wl) / 44.0, 1.0) < .20) & (t < .72) & (S > 26)
    v.col[lat & band] = C['ICEDD']; v.col[lat & band & lip] = C['ICED']
    # --- dorsal plates: darkest tone, shadow line under the next plate, lit lip, pale keel carrying the crystals
    v.col[dor] = C['NAVY']; v.col[dor & lip] = C['STEEL']
    v.col[dor & (FR > .84)] = C['SEAM']
    v.col[dor & (ath > armor.TH_D - .07)] = C['SEAM']                      # dark channel between ridge armour and flank scales
    v.col[dor & (ath * RB < .15 * RB)] = C['STEELL']
    # --- belly: pale banded plates, frost-white leading lip, glacier shadow at the overlap, silver end caps
    wv = (np.pi - ath) * RB
    v.col[ven] = C['CYANP']; v.col[ven & (FR < .26)] = C['FROST']; v.col[ven & (FR > .80)] = C['GLAC']
    chest = np.clip(1 - np.abs(S - (v.meta['s_chest'] + 6)) / 34, 0, 1)
    v.col[ven & (wv > armor.W_V + 3.5 * chest - 2.2)] = C['SILVD']
    v.col[ven & (wv > armor.W_V + 3.5 * chest - 2.2) & (FR < .26)] = C['SILV']
    # --- legs: outer side steel, inner side navy; tiger bands on thighs / upper arms echo the flanks
    v.col[leg] = C['STEEL']; v.col[leg & lip] = C['STEELL']
    inner = leg & (np.abs(LTH) > 2.0)
    v.col[inner] = C['NAVY']; v.col[inner & lip] = C['STEEL']
    lb = leg & (np.mod((LS + 5.5 * np.abs(LTH)) / 17.0, 1.0) < .26) & (LS < 30) & (np.abs(LTH) < 2.0)
    v.col[lb] = C['ICEDD']; v.col[lb & lip] = C['ICED']
    # --- rime: frost settles on the upward lips of the ridge plates and the top flank rows (deliberate, follows the scales)
    v.col[dor & lip & up] = C['CYANP']
    v.col[lat & lip & up & (t < .22)] = C['CYANP']
    v.col[dor & up & (FR < .10)] = C['FROST']
    return arm


def head_tones(v):
    """skull: plate tops catch the light; rime on the brow and crown edges."""
    up = geo.up_open(v.kind)
    hd = v.extra['HEAD'] & ~v.extra['HORN'] & (v.kind == 1) & up
    m = hd & is_col(v, 'STEEL') & (v.lvl >= 4); v.col[m] = C['STEELL']
    m = hd & is_col(v, 'NAVY'); v.col[m] = C['STEEL']
    m = hd & is_col(v, 'STEELL') & (v.lvl >= 6); v.col[m] = C['GLAC']        # brow tops


def paint(v):
    scale_pattern(v); head_tones(v)
    import wings; wings.paint_membrane(v)
    return v


def bbox(v, m, pad=0.0):
    ix = np.argwhere(m)
    return [(ix.min(0) + v.lo - pad).tolist(), (ix.max(0) + 1 + v.lo + pad).tolist()]


def finalize(v, name):
    """skill finishing pass on a COPY of the working grid: seams -> connectivity -> palette snap -> '<name>' checkpoint for scene.py."""
    kind = v.kind.copy(); col = v.col.astype(np.float64).copy(); pid, lvl = v.pid, v.lvl
    keep = np.zeros(v.n, bool)
    for k in ('TEETH', 'SOLIDGLOW', 'CLAW'):
        if k in v.extra: keep |= v.extra[k]
    dark = np.zeros(v.n, bool)
    for ax in range(3):
        for s in (1, -1):
            np_ = np.roll(pid, s, axis=ax); nl = np.roll(lvl, s, axis=ax)
            dark |= (pid > 1) & (np_ > 1) & (np_ != pid) & ((nl > lvl) | ((nl == lvl) & (np_ > pid)))
    col[dark & (kind == 1) & ~keep] *= .6                                   # seams: darken the lower part where two parts meet
    w = geo.Vox(v.lo, v.hi); w.kind = kind; w.col = col.astype(np.float32); w.meta = dict(v.meta)
    ncomp, dropped = geo.drop_floating(w)
    sol = w.kind == 1; cs = w.col[sol].astype(np.float64)
    j = np.zeros(len(cs), np.int64)
    for a in range(0, len(cs), 200000):
        j[a:a + 200000] = np.argmin(((cs[a:a + 200000, None, :] - cfg.PAL_RGB[None]) ** 2).sum(-1), 1)
    w.col[sol] = cfg.PAL_RGB[j]
    used = np.unique(j); nglow = len([k for k in cfg.GLOW if (w.kind == k).any()])
    for k, c in cfg.GLOW.items(): w.col[w.kind == k] = c
    on = w.kind > 0; X = v.X
    B = {}
    def add(nm, m, pad=1.0):
        if (m & on).any(): B[nm] = bbox(v, m & on, pad)
    add('head', v.extra['HEAD'])
    if 'S' in v.extra: add('bust', v.extra['HEAD'] | ((v.extra['S'] > v.meta['s_chest'] - 6) & (v.extra.get('LEG', 0) == 0) & ~v.extra.get('WING', np.zeros(v.n, bool)) & ~v.extra.get('BASE', np.zeros(v.n, bool))), 2.0)
    if 'WING' in v.extra: add('wing', v.extra['WING'] & (X < 0), 2.0)
    if 'BLADE' in v.extra: add('tail', v.extra['BLADE'] | ((v.extra['S'] >= 0) & (v.extra['S'] < 40) & (v.extra['D'] < 6)), 3.0)
    if 'SPINE' in v.extra: add('spine', v.extra['SPINE'] & (v.extra['S'] > v.meta['s_hip'] - 20) & (v.extra['S'] < v.meta['s_chest'] + 30), 6.0)
    for nm, leg, r in (('claw', 'FL', (17, 15, 21)), ('hindclaw', 'HL', (19, 17, 23))):
        o = np.array(v.meta['feet'][leg]['o']); f = np.array(v.meta['feet'][leg]['f']); c = o + f * 7 + np.array((0, 7, 0))
        B[nm] = [(c - r).tolist(), (c + r).tolist()]
    B['chest'] = [[-26, 34, -48], [26, 96, -4]]
    w.meta['boxes'] = B
    w.meta['stats'] = dict(bricks=int(on.sum()), solid_colours=int(len(used)), glow_colours=nglow,
                           components_before_drop=int(ncomp), dropped_bricks=int(dropped),
                           glow_bricks={int(k): int((w.kind == k).sum()) for k in cfg.GLOW})
    print('PALETTE %d solid + %d glow colours in use' % (len(used), nglow))
    print('CONN_OK components before drop %d, dropped floating %d, bricks %d' % (ncomp, dropped, w.meta['stats']['bricks']))
    w.extra = {}; w.save(name)
    return w


if __name__ == '__main__':
    v = geo.Vox.load('04_armor'); paint(v); v.save('05_materials'); finalize(v, '05_materials_fin')
