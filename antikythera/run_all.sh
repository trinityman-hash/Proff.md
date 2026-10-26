#!/bin/bash
# usage:  bash run_all.sh            -> full film  (1440 frames, 1280x720)
#         bash run_all.sh test       -> 3 test frames only, prints seconds per frame
set -e
cd /content/antikythera
W=1280; H=720; SPP=32
apt-get install -y -qq libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libsm6 libegl1 libxkbcommon0 >/dev/null 2>&1 || true
pip install -q uv
if [ ! -x /content/venv/bin/python ]; then
  uv venv --python 3.13 /content/venv
  uv pip install --python /content/venv/bin/python -q bpy==5.2.3 numpy scipy opencv-python-headless usd-core
fi
PY=/content/venv/bin/python
nvidia-smi -L || echo "NO GPU FOUND - Runtime > Change runtime type > T4 GPU"
mkdir -p usd audio
[ -f usd/antikythera.usda ] || $PY build_usd.py
[ -f audio/antikythera.wav ] || $PY audio.py
if [ "$1" = "test" ]; then
  $PY render_blender.py --frames 300,700,1262 --w $W --h $H --spp $SPP --out /content/test_raw 2>&1 | grep -E "RENDER DEVICE|^frame|Error|Traceback" || true
  exit 0
fi
OUT=/content/antikythera_render
mkdir -p $OUT/raw
$PY render_blender.py --range 1 1440 1 --w $W --h $H --spp $SPP --out $OUT/raw 2>&1 | grep -E "RENDER DEVICE|^frame +[0-9]*[05]0 |Error|Traceback" || true
$PY post_all.py $OUT/raw $OUT/frames
ffmpeg -y -loglevel error -framerate 24 -i $OUT/frames/%04d.png -i audio/antikythera.wav \
  -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p -c:a aac -b:a 256k -shortest -movflags +faststart $OUT/ANTIKYTHERA.mp4
ls -la $OUT/ANTIKYTHERA.mp4
