"""ICE DRAGON - parameters, palette, pose. Design coords (LihFolk brick convention): x right, y up, z back (front = -z).
1 voxel = 1 brick = 1 unit (= 2.5 studs in game)."""
import os
import numpy as np

ID = 'IceDragon'
SEED = 1207
SKILL = os.environ.get('SKILL_DIR', '/workspace/lihfolk-stud-voxel')
STUD_PNG = os.path.join(SKILL, 'assets', 'roblox_square_studs.png')
ROOT = os.environ.get('ID_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(ROOT, 'checkpoints')
OUT = os.path.join(ROOT, 'output')

LO = np.array((-120.0, -0.5, -104.0))        # grid bounds (voxel centres at lo + .5 + i)
HI = np.array((120.0, 193.5, 118.0))


def hx(h): return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ---- palette: 16 solid colours + 2 emissive glow colours = 18 total
PAL = dict(
    SEAM='0b1226',    # deepest shadow / seams / pupils
    NAVY='18294f',    # deep navy (hide ground, under-scales)
    STEEL='2a4a80',   # steel blue (main scales)
    STEELL='4474b2',  # light steel (scale lips, plate tops)
    GLAC='6aa6d8',    # glacier blue (saddle bands, membrane mid)
    CYANP='a9dcf2',   # pale cyan (belly plates, membrane light)
    FROST='eef8fd',   # frost white (teeth, crystal tips, rime, snow)
    SILV='b9c4d0',    # silver (horn mid, claws, wing fingers)
    SILVD='7c8a9e',   # dark silver (horn shade, finger shade)
    ICEDD='1466a8',   # deep ice (crystal cores / bases)
    ICED='2fa3dc',    # ice mid (crystal faces)
    ICE='7fe3f6',     # ice highlight (lit crystal faces)
    VIOD='2a2266',    # deep indigo (mouth lining, membrane shadow veins)
    VIO='5a4fb0',     # violet (tongue, flank sheen accent)
    ROCKD='27303c',   # base rock dark
    ROCK='4a5564',    # base rock
)
C = {k: hx(v) for k, v in PAL.items()}
GLOW = {2: hx('58ecff'),   # eyes (one flat glow colour)
        3: hx('b4f6ff')}   # throat core / frost breath / crystal hearts
PAL_RGB = np.array([C[k] for k in PAL])

# ---------------------------------------------------------------- pose
GROUND = 11.0                                # top of the ice-rock base under the feet

# spine: tail tip -> tail -> hips -> barrel -> chest -> S-neck -> skull base.   (x, y, z, radius)
TAIL = [(-86, 61, 84, 2.8), (-84.5, 45, 90, 3.4), (-77, 31, 97, 4.4), (-61, 21.5, 102, 5.6), (-41, 18.5, 102, 6.9),
        (-21, 22, 95, 8.3), (-6, 33, 80, 9.9), (0, 48, 60, 11.8)]
TORSO = [(0, 59, 40, 14.5), (0, 63, 24, 16.2), (0, 66.5, 6, 17.0), (0, 71, -10, 17.2)]
NECK = [(0, 80, -23, 16.2), (0, 92, -32.5, 13.8), (-2, 104, -34.5, 12.0), (-5, 116, -30.5, 11.0), (-9, 127, -28, 10.3), (-12, 135.5, -33, 9.9),
        (-14, 140, -43, 9.6), (-15, 140.5, -53, 9.5)]

# head frame: yaw toward -x (the 3/4 camera side), pitched down
HEAD_YAW, HEAD_PITCH = 9.0, -13.0
JAW_OPEN = 27.0

# legs: (hip/shoulder, knee/elbow, hock/wrist, foot centre) + radii; feet point along `toe` (yaw in degrees, 0 = -z)
LEGS = dict(
    FL=dict(P=[(-18, 66, -13), (-24.5, 49, -5), (-25.5, 37, -19), (-26.5, 29.5, -24)], R=[10.5, 7.6, 5.4, 5.2], yaw=10, ground=25.0),   # raised on an ice rock
    FR=dict(P=[(18, 65, -13), (24, 44, -3), (25, 24, -15), (26, 16.5, -19)], R=[10.5, 7.6, 5.4, 5.2], yaw=-8, ground=GROUND),
    HL=dict(P=[(-16, 58, 33), (-25.5, 41, 17), (-26, 26, 35), (-27, 16.5, 30)], R=[12.5, 8.6, 5.6, 5.4], yaw=14, ground=GROUND),
    HR=dict(P=[(16, 58, 33), (25.5, 41, 19), (26, 26, 37), (27, 16.5, 32)], R=[12.5, 8.6, 5.6, 5.4], yaw=-12, ground=GROUND),
)

# wings (right side, x > 0; the left wing is the mirror image)
WING = dict(
    S=(10, 83, -9),            # shoulder root
    E=(35, 112, 7),            # elbow
    W=(60, 141, -13),          # wrist
    FING=[[(60, 141, -13), (82, 158, -10), (104, 170, -1)],            # leading spar
          [(60, 141, -13), (88, 134, 1), (113, 118, 22)],
          [(60, 141, -13), (79, 110, 14), (95, 73, 44)],
          [(60, 141, -13), (64, 97, 22), (62, 52, 55)]],
    H=(13, 63, 43),            # trailing membrane meets the flank at the hip
)
