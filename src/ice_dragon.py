"""ICE DRAGON - generator entry point (pure numpy; no Blender needed for the geometry).
  python3 ice_dragon.py            resume: run only the stages whose checkpoint is missing
  python3 ice_dragon.py --from 4   re-run from stage 4 (earlier checkpoints are reused, never rebuilt)
Stages: 2 blockout (body.py) -> 3 anatomy (head.py, horns.py, wings.py, tail.py) -> 4 armour (armor.py: scales, plates, crystal spines)
        -> 5 base + materials (base.py, materials.py) -> 6 signature features (features.py)
        -> finished grid 'checkpoints/06_features_fin' (seams, flood fill, palette snap).
Blender side: render.sh (scene.py -> post_glow.py), export.py (rig + GLB via the skill's brick_rig.py), verify.py."""
import os, sys, time
import config as cfg, geometry as geo


def s2():
    import body; body.run()
def s3():
    import head, horns, wings, tail
    v = geo.Vox.load('02_blockout'); head.run(v); horns.run(v); wings.run(v); tail.run(v)
    nc = geo.components(v.kind > 0)[1]; print('ANATOMY bricks %d components %d %s' % (int((v.kind > 0).sum()), len(nc), nc[:5]))
    v.save('03_anatomy')
def s4():
    import armor
    v = geo.Vox.load('03_anatomy'); armor.run(v); v.save('04_armor')
def s5():
    import base, materials
    v = geo.Vox.load('04_armor'); base.run(v); materials.paint(v); v.save('05_materials'); materials.finalize(v, '05_materials_fin')
def s6():
    import features, materials
    v = geo.Vox.load('05_materials'); features.run(v); v.save('06_features'); materials.finalize(v, '06_features_fin')

STAGES = [(2, '02_blockout', s2), (3, '03_anatomy', s3), (4, '04_armor', s4), (5, '05_materials', s5), (6, '06_features_fin', s6)]

if __name__ == '__main__':
    frm = int(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else None
    to = int(sys.argv[sys.argv.index('--to') + 1]) if '--to' in sys.argv else 99
    for n, ck, fn in STAGES:
        if n > to: break
        have = os.path.exists(os.path.join(cfg.CKPT, ck, 'grid.npz'))
        if (frm is None and have) or (frm is not None and n < frm and have):
            print('STAGE %d skipped (checkpoint %s exists)' % (n, ck)); continue
        t = time.time(); fn(); print('STAGE %d done in %.1fs' % (n, time.time() - t), flush=True)
    print('GENERATOR_OK')
