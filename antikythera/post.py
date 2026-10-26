"""
Post: linear EXR -> bloom + anamorphic streaks + grade + whiteout + vignette +
grain -> sRGB PNG.   Reads cues from usd/shots.json so the look is frame-locked.

usage: python3 post.py raw_dir out_dir [frame ...]
"""
import os, sys, json
import numpy as np
import cv2

SIDE = json.load(open("/content/antikythera/usd/shots.json"))["frames"]


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-9), 1 / 2.4) - 0.055)


def blur(img, s):
    k = int(s * 6) | 1
    return cv2.GaussianBlur(img, (k, k), s)


def hblur(img, sx, sy=1.2):
    kx = int(sx * 6) | 1
    ky = int(sy * 6) | 1
    return cv2.GaussianBlur(img, (kx, ky), sx, sigmaY=sy)


def process(frame, raw_dir, out_dir):
    d = SIDE[frame - 1]
    L, t, B = d["L"], d["t"], d["scene"] == "B"
    img = cv2.imread(os.path.join(raw_dir, "%04d.hdr" % frame), cv2.IMREAD_UNCHANGED)
    img = img[:, :, :3][:, :, ::-1].astype(np.float32)      # BGR -> RGB
    img = np.nan_to_num(img, nan=0.0, posinf=60.0)
    H, W = img.shape[:2]
    s = W / 1280.0

    # --- exposure per set
    img *= 1.15 if not B else 1.0

    # --- bloom (multi-scale) + anamorphic streak from the hot pixels
    hot = np.maximum(img - 0.9, 0)
    bloom = 0.0
    for sig, wgt in ((3, .30), (9, .26), (28, .22), (80, .20), (200, .16)):
        bloom = bloom + wgt * blur(hot, sig * s)
    glare = max(L["sigma_flare"], L["diamond"], L["tot"] * 0.8)
    strength = 0.45 + 0.9 * glare
    img = img + strength * bloom
    if glare > 0.02:
        streak = hblur(np.maximum(img - 6.0, 0), 90 * s, 1.5 * s)
        img = img + (0.05 + 0.5 * glare) * streak

    # --- whiteout (HDR flash)
    img = img + (L["flash"] ** 1.5) * 9.0 + L["crack_flash"] * 1.2

    # --- tone map
    img = aces(img * 0.9)

    # --- grade: teal shadows / amber highlights (A); cool-blue shadows (B)
    lum = img.mean(axis=2, keepdims=True)
    if not B:
        shadow = np.array([0.02, 0.055, 0.06], np.float32)
        high = np.array([1.06, 1.0, 0.9], np.float32)
    else:
        shadow = np.array([0.02, 0.03, 0.07], np.float32)
        high = np.array([1.03, 1.0, 0.95], np.float32)
    img = img + (1 - lum) ** 2 * shadow * 0.9
    img = img * (1 + (high - 1) * lum)
    img = np.clip(img, 0, 1)
    # gentle S-curve for contrast
    img = img + 0.14 * (img - 0.5) * (1 - np.abs(img - 0.5) * 2) * 2.0 * 0.5
    img = np.clip(img, 0, 1)

    # --- chromatic aberration (tiny, radial)
    def zoom(ch, z):
        M = np.array([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]], np.float32)
        return cv2.warpAffine(ch, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img[:, :, 0] = zoom(img[:, :, 0], 1.0016)
    img[:, :, 2] = zoom(img[:, :, 2], 0.9984)

    # --- vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = 1.0 - 0.38 * np.clip(rr - 0.35, 0, 1.4) ** 1.7
    img = img * vig[:, :, None]

    # --- sRGB + grain
    img = srgb(img)
    rng = np.random.default_rng(frame * 7919)
    grain = rng.normal(0, 1, (H, W)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), 0.8)
    img = img + grain[:, :, None] * (0.018 * (1.0 - 0.6 * img.mean(axis=2, keepdims=True)))

    # --- fade to black
    img = img * (1.0 - L["fade"])
    img = np.clip(img, 0, 1)
    out = (img * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "%04d.png" % frame), out)


if __name__ == "__main__":
    raw, outd = sys.argv[1], sys.argv[2]
    fr = [int(x) for x in sys.argv[3:]] or sorted(int(f[:4]) for f in os.listdir(raw) if f.endswith(".hdr"))
    for f in fr:
        process(f, raw, outd)
    print("post done:", len(fr), "frames")
