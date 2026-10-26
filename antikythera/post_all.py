import os, sys, time
raw, out = sys.argv[1], sys.argv[2]
sys.argv = ["post.py"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post
os.makedirs(out, exist_ok=True)
files = sorted(f for f in os.listdir(raw) if f.endswith(".hdr"))
n = 0
for f in files:
    k = int(f[:4])
    if os.path.exists(f"{out}/{k:04d}.png"):
        continue
    post.process(k, raw, out); n += 1
    if n % 100 == 0: print("post-processed", n, flush=True)
print("post done:", n, "new frames,", len(files), "total")
