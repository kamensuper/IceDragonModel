"""Blender side (LihFolk brick pipeline): checkpoint grid -> exposed-face mesh with vertex colours + square-stud OVERLAY texture,
glow objects per glow colour, studio lighting, EEVEE beauty + Workbench mask + glow pass per view (inputs of post_glow.py).
  blender -b --factory-startup --python scene.py -- <stage> <outdir> [res=450] [views=a,b,..] [emit=5] [blend=path] [samples=16] [turn=N]"""
import bpy, math, os, sys, json
import numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

argv = sys.argv[sys.argv.index('--') + 1:]
STAGE, REN = argv[0], argv[1]
OPT = dict(a.split('=', 1) for a in argv[2:])
RES = int(OPT.get('res', 450)); EMIT = float(OPT.get('emit', 5)); ONLY = [x for x in OPT.get('views', '').split(',') if x]
os.makedirs(REN, exist_ok=True)
z = np.load(os.path.join(cfg.CKPT, STAGE, 'grid.npz')); meta = json.load(open(os.path.join(cfg.CKPT, STAGE, 'meta.json')))
lo = z['lo']; kind = z['kind']; col = z['col'].astype(np.float64); n = kind.shape
solid = kind > 0

def lin(c): return np.where(c > .04045, ((c + .055) / 1.055) ** 2.4, c / 12.92)
def exposed(mask, ax, s):
    nb = np.roll(solid, -s, axis=ax)
    sl = [slice(None)] * 3; sl[ax] = -1 if s > 0 else 0; nb[tuple(sl)] = False
    return mask & ~nb

def build(name, mask):
    verts = []; faces = []; cols = []; uvs = []; nv = 0
    for ax in range(3):
        u, w = [k for k in range(3) if k != ax]
        for s in (1, -1):
            e = np.argwhere(exposed(mask, ax, s))
            if not len(e): continue
            base = lo + e; off = np.zeros(3); off[ax] = 1 if s > 0 else 0
            q = []; t = []
            for du, dw in ((0, 0), (1, 0), (1, 1), (0, 1)):
                dd = off.copy(); dd[u] = du; dd[w] = dw; q.append(base + dd)
                t.append(np.stack([(e[:, u] + du) / 16.0, (e[:, w] + dw) / 16.0], 1))     # one stud per 2x2 bricks
            q = np.stack(q, 1); t = np.stack(t, 1); k = len(e)
            if (s > 0) == (ax == 1): q = q[:, ::-1]; t = t[:, ::-1]                       # outward winding (Blender swaps y/z)
            verts.append(q.reshape(-1, 3)); faces.append(nv + np.arange(4 * k).reshape(k, 4)); nv += 4 * k; uvs.append(t.reshape(-1, 2))
            cols.append(np.repeat(col[e[:, 0], e[:, 1], e[:, 2]], 4, axis=0))
    Vt = np.concatenate(verts); Cv = np.concatenate(cols); UV = np.concatenate(uvs); F = np.concatenate(faces)
    Vb = np.stack([Vt[:, 0], Vt[:, 2], Vt[:, 1]], 1)
    me = bpy.data.meshes.new(name); me.vertices.add(len(Vb)); me.vertices.foreach_set('co', Vb.ravel())
    me.loops.add(F.size); me.loops.foreach_set('vertex_index', F.ravel().astype(np.int32))
    me.polygons.add(len(F)); me.polygons.foreach_set('loop_start', np.arange(0, F.size, 4, dtype=np.int32)); me.polygons.foreach_set('loop_total', np.full(len(F), 4, np.int32))
    me.update(calc_edges=True); me.validate()
    ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', np.concatenate([lin(np.clip(Cv, 0, 1)), np.ones((len(Cv), 1))], 1).ravel())
    ul = me.uv_layers.new(name='UVMap'); ul.data.foreach_set('uv', UV[F.ravel()].ravel())
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob); return ob

for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
body = build(cfg.ID, kind == 1)
m = bpy.data.materials.new('Brick'); m.use_nodes = True
nt = m.node_tree; bs = nt.nodes['Principled BSDF']; at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = 'Col'
bs.inputs['Roughness'].default_value = .42
tx = nt.nodes.new('ShaderNodeTexImage'); tx.image = bpy.data.images.load(cfg.STUD_PNG); tx.interpolation = 'Linear'
tx.image.colorspace_settings.name = 'Non-Color'
mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'OVERLAY'; mx.inputs['Factor'].default_value = 1
nt.links.new(at.outputs['Color'], mx.inputs[6]); nt.links.new(tx.outputs['Color'], mx.inputs[7]); nt.links.new(mx.outputs[2], bs.inputs['Base Color'])
body.data.materials.append(m); body['glow'] = False
GL = []
for k, c in cfg.GLOW.items():
    if not (kind == k).any(): continue
    o = build('Glow%d' % k, kind == k)
    gm = bpy.data.materials.new('Glow%d' % k); gm.use_nodes = True; b2 = gm.node_tree.nodes['Principled BSDF']
    cl = (*lin(c), 1); b2.inputs['Base Color'].default_value = cl; b2.inputs['Emission Color'].default_value = cl
    b2.inputs['Emission Strength'].default_value = EMIT; gm.diffuse_color = cl
    o.data.materials.append(gm); o.color = cl; GL.append(o)
NFACE = len(body.data.polygons) + sum(len(o.data.polygons) for o in GL)
print('VOX_OK', int(solid.sum()), 'bricks', NFACE, 'faces', 1 + len(GL), 'mesh objects')

def B(x, y, z): return Vector((x, z, y))
sc = bpy.context.scene
ENG = 'BLENDER_EEVEE' if 'BLENDER_EEVEE' in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys() else 'BLENDER_EEVEE_NEXT'
# --- cold three-point studio: neutral-white key from the front-left, blue fill, strong cyan rim from behind, low kicker
for nm, e, rot, an, c in (('Key', 3.6, (52, 0, -30), 12, (1.0, .97, .93)), ('Fill', 1.0, (64, 0, 78), 45, (.70, .82, 1.0)),
                          ('Rim', 2.8, (60, 0, 168), 18, (.62, .88, 1.0)), ('Kick', .6, (72, 0, -105), 45, (.85, .92, 1.0))):
    L = bpy.data.objects.new(nm, bpy.data.lights.new(nm, 'SUN')); sc.collection.objects.link(L)
    L.data.energy = e; L.data.angle = math.radians(an); L.data.color = c; L.rotation_euler = tuple(math.radians(v_) for v_ in rot)
HF = meta.get('head')
if HF and EMIT > 0 and (kind > 1).any():                                       # subtle emission spill: eyes + throat only
    H0 = np.array(HF['o']); Hf = np.array(HF['f']); Hu = np.array(HF['u']); Hr = np.array(HF['r']); hs = HF.get('hs', 1)
    def hp(fw, up, side=0): return tuple(H0 + (Hf * fw + Hu * up + Hr * side) * hs)
    for nm, pos, e, c in (('EyeL', hp(26, 3, 13.5), 420, cfg.GLOW[2]), ('EyeR', hp(26, 3, -13.5), 420, cfg.GLOW[2]), ('Throat', hp(16, -9), 1500, cfg.GLOW[3])):
        L = bpy.data.objects.new(nm, bpy.data.lights.new(nm, 'POINT')); sc.collection.objects.link(L)
        L.data.energy = e; L.data.color = tuple(lin(c)); L.data.shadow_soft_size = 1.0; L.location = B(*pos)
sh = sc.display.shading; sc.view_settings.view_transform = 'Standard'
sc.render.resolution_x = sc.render.resolution_y = RES
sc.world = sc.world or bpy.data.worlds.new('W'); sc.world.color = (.016, .022, .034)
for attr in ('use_gtao', 'use_raytracing'):
    try: setattr(sc.eevee, attr, True)
    except Exception: pass
try: sc.eevee.taa_render_samples = int(OPT.get('samples', 16))
except Exception: pass
me = bpy.data.meshes.new('G'); s_ = 4000
me.from_pydata([(-s_, -s_, -.5), (s_, -s_, -.5), (s_, s_, -.5), (-s_, s_, -.5)], [], [(0, 1, 2, 3)])
g = bpy.data.objects.new('Ground', me); sc.collection.objects.link(g)
gm = bpy.data.materials.new('Gnd'); gm.use_nodes = True; gb = gm.node_tree.nodes['Principled BSDF']
gb.inputs['Base Color'].default_value = (.010, .014, .022, 1); gb.inputs['Roughness'].default_value = .8
me.materials.append(gm)
cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.clip_end = 30000

AV = np.concatenate([np.array([v_.co[:] for v_ in o.data.vertices]) for o in [body] + GL])            # blender coords (x, z, y)
BOXES = meta.get('boxes', {})
def pts_in(box):
    if box is None: return AV
    b0, b1 = np.array(BOXES[box][0]), np.array(BOXES[box][1])
    P = AV[((AV[:, [0, 2, 1]] >= b0) & (AV[:, [0, 2, 1]] <= b1)).all(1)]
    return P if len(P) else AV

def fit(dv, lens, ortho, margin, PTS):                  # tight framing: centre the projected subject, size to fit
    dv = np.array(dv[:]); up = np.array((0, 0, 1.0)) if abs(dv[2]) < .99 else np.array((0, 1.0, 0))
    r = np.cross(up, dv); r /= np.linalg.norm(r); u = np.cross(dv, r)
    pr = PTS @ r; pu = PTS @ u; pf = PTS @ dv
    c = r * (pr.max() + pr.min()) / 2 + u * (pu.max() + pu.min()) / 2 + dv * (pf.max() + pf.min()) / 2
    q = PTS - c; qr = np.abs(q @ r); qu = np.abs(q @ u); qf = q @ dv
    if ortho: return Vector(c), 4000, 2 * margin * max(qr.max(), qu.max())
    th = 18.0 / lens
    return Vector(c), margin * max(((qr / th) + qf).max(), ((qu / th) + qf).max()), None

FRAMING = {}
def place(name, box, az, el, lens=60, ortho=False, margin=1.07, fixed=None):
    dv = Vector((math.cos(math.radians(el)) * math.sin(math.radians(az)), -math.cos(math.radians(el)) * math.cos(math.radians(az)), math.sin(math.radians(el))))
    P = pts_in(box)
    if fixed: t, dist, osc = fixed
    else: t, dist, osc = fit(dv, lens, ortho, margin, P[::3] if len(P) > 30000 else P)
    cam.location = t + dv * dist; cam.rotation_euler = (t - cam.location).to_track_quat('-Z', 'Y').to_euler(); cam.data.lens = lens
    cam.data.type = 'ORTHO' if ortho else 'PERSP'
    if ortho: cam.data.ortho_scale = osc
    bpy.context.view_layer.update()
    M = np.array(cam.matrix_world.inverted()); pc = P @ M[:3, :3].T + M[:3, 3]      # framing test: camera-space bounds of every subject vertex
    if ortho: nx = pc[:, 0] / osc + .5; ny = pc[:, 1] / osc + .5
    else:
        k = lens / 36.0; nx = .5 + k * pc[:, 0] / -pc[:, 2]; ny = .5 + k * pc[:, 1] / -pc[:, 2]
    FRAMING[name] = dict(subject=box or 'whole model', x=[float(nx.min()), float(nx.max())], y=[float(ny.min()), float(ny.max())],
                         inside=bool(nx.min() >= 0 and nx.max() <= 1 and ny.min() >= 0 and ny.max() <= 1))

def shoot(name):
    p = os.path.join(REN, '%s_%s.png' % (cfg.ID, name))
    sc.render.engine = ENG; sc.render.film_transparent = False; sc.render.filepath = p; bpy.ops.render.render(write_still=True)
    sc.render.engine = 'BLENDER_WORKBENCH'; sh.light = 'FLAT'; sh.color_type = 'OBJECT'; sh.show_shadows = False; sh.show_cavity = False
    g.hide_render = True; sc.render.film_transparent = True; sc.render.filepath = p[:-4] + '_mask.png'; bpy.ops.render.render(write_still=True)
    sc.render.film_transparent = False; keep_ = tuple(sc.world.color); sc.world.color = (0, 0, 0); sh.background_type = 'WORLD'
    body.color = (0, 0, 0, 1)
    sc.render.filepath = p[:-4] + '_glow.png'; bpy.ops.render.render(write_still=True)
    sc.world.color = keep_; g.hide_render = False; body.color = (1, 1, 1, 1)
    print('VIEW_OK', name, flush=True)

# name: (subject box, azimuth, elevation, lens, ortho, margin)
VIEWS = [('ortho_front', (None, 0, 0, 50, True, 1.05)), ('ortho_side', (None, -90, 0, 50, True, 1.05)), ('ortho_back', (None, 180, 0, 50, True, 1.05)),
         ('ortho_top', (None, 0, 89.9, 50, True, 1.05)),
         ('rest_34', (None, -36, 13, 60, False, 1.06)), ('rest_front', (None, 0, 7, 60, False, 1.06)), ('rest_side', (None, -90, 6, 60, False, 1.06)),
         ('side_right', (None, 90, 6, 60, False, 1.06)), ('rest_back', (None, 180, 10, 60, False, 1.06)), ('back34', (None, 146, 16, 60, False, 1.06)),
         ('alt_34', (None, 40, 13, 60, False, 1.06)), ('high_34', (None, -30, 38, 60, False, 1.06)), ('top', (None, 0, 72, 60, False, 1.06)),
         ('hero_low', (None, -24, -6, 35, False, 1.04)),
         ('head34', ('head', -40, 6, 70, False, 1.06)), ('face', ('head', -24, -2, 70, False, 1.06)), ('headside', ('head', -114, 2, 70, False, 1.06)),
         ('bust', ('bust', -34, 4, 70, False, 1.05)),
         ('wing', ('wing', -52, 10, 70, False, 1.04)), ('wing_back', ('wing', -150, 14, 70, False, 1.04)),
         ('claw', ('claw', -48, 16, 70, False, 1.1)), ('hindclaw', ('hindclaw', -70, 14, 70, False, 1.1)),
         ('tailtip', ('tail', -100, 8, 70, False, 1.08)), ('spine', ('spine', -62, 34, 70, False, 1.05)), ('chest', ('chest', -18, -4, 70, False, 1.05))]
VD = dict(VIEWS)
if OPT.get('render', '1') == '1':
    for nm, a in VIEWS:
        if ONLY and nm not in ONLY: continue
        if a[0] is not None and a[0] not in BOXES: print('VIEW_SKIP', nm, '(no box %s)' % a[0]); continue
        place(nm, *a); shoot(nm)
    NT = int(OPT.get('turn', 0))
    if NT:                                               # turntable: fixed target + distance (bounding sphere), camera orbits
        c = (AV.max(0) + AV.min(0)) / 2; rad = np.linalg.norm(AV - c, axis=1).max(); lens = 60
        dist = rad / math.sin(math.atan(18.0 / lens)) * 1.0
        for i in range(NT):
            nm = 'turn_%03d' % i
            place(nm, None, -36 + 360.0 * i / NT, 12, lens, False, 1.0, fixed=(Vector(c), dist, None)); shoot(nm)
    json.dump(FRAMING, open(os.path.join(REN, 'framing.json'), 'w'), indent=1)
if 'blend' in OPT:                                       # editable static scene: beauty camera, EEVEE, stud texture packed
    place('rest_34', *VD['rest_34']); sc.render.engine = ENG; sc.render.film_transparent = False
    body.color = (1, 1, 1, 1); g.hide_render = False
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=OPT['blend']); print('BLEND_SAVED', OPT['blend'])
print('BL_OK')
