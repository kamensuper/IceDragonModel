#!/usr/bin/env bash
# render.sh <stage> <outdir> [res=..] [views=..] [emit=..] [blend=..] [samples=..] [turn=N]
#   -> scene.py (headless Blender, EEVEE + Workbench through xvfb / Mesa software GL) -> post_glow.py (outline + glow) -> contact sheet
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; export ID_ROOT="${ID_ROOT:-$(dirname "$HERE")}"
SKILL="${SKILL_DIR:-/workspace/lihfolk-stud-voxel}"; BLENDER="${BLENDER_EXE:-blender}"; PY="${PYTHON:-python3}"
STAGE="$1"; OUT="$2"; shift 2
xvfb-run -a -s '-screen 0 1280x1024x24' "$BLENDER" -b --factory-startup --python "$HERE/scene.py" -- "$STAGE" "$OUT" "$@" 2>&1 \
  | grep -E 'VOX_OK|PALETTE|BL_OK|BLEND_|RIG_|CONN_|VIEW_|Traceback|Error|line [0-9]+'
unset GLOW_STRONG            # subtle glow model: restrained emission, no heavy bloom
"$PY" "$SKILL/scripts/post_glow.py" "$OUT"
