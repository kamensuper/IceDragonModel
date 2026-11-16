"""Fast numpy preview (no Blender): orthographic z-sorted voxel splats with baked lambert shading. For blockout iteration only -
every image in renders/ is a real Blender render (scene.py).
  python3 preview.py <stage> <out.png> [px=3] [views=34,front,side,back,top,back34]"""
import os, sys
import numpy as np
from PIL import Image
import config as cfg

VIEWS = {'34': (-36, 14), 'front': (0, 4), 'side': (-90, 4), 'side_r': (90, 4), 'back': (180, 6), 'top': (0, 89), 'back34': (148, 16), 'alt34': (38, 14), 'low': (-28, -8)}


def render(kind, col, az, el, px=3, bg=(52, 58, 70), crop=None, lo=None):
    from scipy import ndimage
    solid = kind > 0
    inner = solid.copy()
    for ax in range(3):
        for s in (1, -1):
            nb = np.roll(solid, s, axis=ax); sl = [slice(None)] * 3; sl[ax] = 0 if s > 0 else -1; nb[tuple(sl)] = False; inner &= nb
    idx = np.argwhere(solid & ~inner)
    if crop is not None and lo is not None:
        p = idx + lo + .5; ok = ((p >= np.array(crop[0])) & (p <= np.array(crop[1]))).all(1); idx = idx[ok]
    occ = ndimage.uniform_filter(solid.astype(np.float32), 3)
    g = np.stack(np.gradient(occ), -1)[tuple(idx.T)]; n = -g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)
    a, e = np.radians(az), np.radians(el)
    c = np.array((np.cos(e) * np.sin(a), np.sin(e), -np.cos(e) * np.cos(a))); view = -c
    up = np.array((0, 1.0, 0)) if abs(c[1]) < .99 else np.array((0, 0, 1.0))
    right = np.cross(up, view); right /= np.linalg.norm(right); upv = np.cross(view, right)
    P = idx.astype(np.float64)
    sx = P @ right; sy = P @ upv; dp = P @ view
    L = np.array((-.45, .78, -.44)); L /= np.linalg.norm(L)
    sh = .42 + .5 * np.clip(n @ L, 0, 1) + .16 * np.clip(n @ -view, 0, 1)
    rgb = col[tuple(idx.T)] * sh[:, None]
    gl = kind[tuple(idx.T)] > 1; rgb[gl] = col[tuple(idx.T)][gl]
    w = int((sx.max() - sx.min() + 3) * px); h = int((sy.max() - sy.min() + 3) * px)
    img = np.zeros((h, w, 3), np.uint8); img[:] = bg
    o = np.argsort(-dp)
    ix = ((sx[o] - sx.min() + 1) * px).astype(int); iy = h - 1 - ((sy[o] - sy.min() + 1) * px).astype(int)
    cc = (np.clip(rgb[o], 0, 1) * 255).astype(np.uint8)
    k = int(np.ceil(px * 1.45))
    for dy in range(k):
        for dx in range(k):
            img[np.clip(iy - dy, 0, h - 1), np.clip(ix + dx, 0, w - 1)] = cc
    return Image.fromarray(img)


def sheet(kind, col, out, views=('34', 'front', 'side', 'back34', 'top', 'alt34'), px=3, cols=3, tile=620, crop=None, lo=None):
    ims = [render(kind, col, *VIEWS[v] if isinstance(v, str) else v, px=px, crop=crop, lo=lo) for v in views]
    rows = (len(ims) + cols - 1) // cols
    S = Image.new('RGB', (cols * tile, rows * tile), (52, 58, 70))
    for i, im in enumerate(ims):
        im.thumbnail((tile, tile), Image.LANCZOS)
        S.paste(im, ((i % cols) * tile + (tile - im.width) // 2, (i // cols) * tile + (tile - im.height) // 2))
    S.save(out); print('PREVIEW_OK', out, S.size)


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]; opt = dict(a.split('=', 1) for a in sys.argv[3:])
    z = np.load(os.path.join(cfg.CKPT, stage, 'grid.npz'))
    vs = opt.get('views', '34,front,side,back34,top,alt34').split(',')
    crop = None
    if 'crop' in opt:
        q = [float(x) for x in opt['crop'].split(',')]; crop = (q[:3], q[3:])
    sheet(z['kind'], z['col'], out, vs, int(opt.get('px', 3)), int(opt.get('cols', 3)), int(opt.get('tile', 620)), crop, z['lo'])
