"""
ANTIKYTHERA - score + SFX, synthesized from scratch with numpy.
Everything is placed from timeline.py / shots.py / usd/info.json, so hits land
on the exact frames of the picture.   120 BPM, D minor -> D major.
"""
import sys, json
sys.path.insert(0, "/content/antikythera")
import numpy as np
from scipy.signal import butter, lfilter, fftconvolve
import wave
import timeline as T

SR = 48000
DUR = 60.0
N = int(SR * DUR)
rng = np.random.default_rng(1901)
EV = T.EV
BEAT, BAR = T.BEAT, T.BAR

dryL = np.zeros(N); dryR = np.zeros(N)
sndL = np.zeros(N); sndR = np.zeros(N)         # reverb send
subL = np.zeros(N); subR = np.zeros(N)         # kept dry (no reverb)


def tarr(d):
    return np.arange(int(d * SR)) / SR


def place(sig, t0, gain=1.0, pan=0.0, send=0.25, dry=True):
    i0 = int(round(t0 * SR))
    if i0 >= N:
        return
    sig = np.asarray(sig, float)
    n = min(len(sig), N - i0) if i0 >= 0 else 0
    if i0 < 0:
        sig = sig[-i0:]; i0 = 0; n = min(len(sig), N)
    if n <= 0:
        return
    s = sig[:n] * gain
    a = (pan + 1) * np.pi / 4
    gl, gr = np.cos(a), np.sin(a)
    if dry:
        dryL[i0:i0 + n] += s * gl; dryR[i0:i0 + n] += s * gr
    sndL[i0:i0 + n] += s * gl * send; sndR[i0:i0 + n] += s * gr * send


def place_sub(sig, t0, gain=1.0):
    i0 = int(round(t0 * SR)); n = min(len(sig), N - i0)
    if n > 0:
        subL[i0:i0 + n] += sig[:n] * gain; subR[i0:i0 + n] += sig[:n] * gain


def env_ad(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(t / max(a, 1e-4), 1.0) * np.exp(-t / d)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def sweep_noise(dur, f0, f1, kind="band", q=1.25, curve=1.0):
    n = int(dur * SR)
    x = rng.normal(size=n)
    out = np.zeros(n); blk = 512
    zi = None
    for i in range(0, n, blk):
        u = (i / n) ** curve
        fc = f0 * (f1 / f0) ** u
        fc = min(fc, SR * 0.45)
        if kind == "band":
            b, a = butter(2, [fc / q / (SR / 2), min(fc * q / (SR / 2), 0.98)], "band")
        elif kind == "low":
            b, a = butter(2, fc / (SR / 2), "low")
        else:
            b, a = butter(2, fc / (SR / 2), "high")
        if zi is None:
            zi = np.zeros(max(len(a), len(b)) - 1)
        y, zi = lfilter(b, a, x[i:i + blk], zi=zi)
        out[i:i + blk] = y
    return out / (np.abs(out).max() + 1e-9)


def lp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "low"); return lfilter(b, a, x)


def hp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "high"); return lfilter(b, a, x)


def bell(freq, dur=3.0, bright=1.0, decay=1.2):
    t = tarr(dur)
    ratio = 3.5
    idx = bright * 4.0 * np.exp(-t / 0.35)
    mod = np.sin(2 * np.pi * freq * ratio * t)
    car = np.sin(2 * np.pi * freq * t + idx * mod)
    car += 0.35 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-t / (decay * 0.5))
    car += 0.25 * np.sin(2 * np.pi * freq * 5.4 * t) * np.exp(-t / (decay * 0.25))
    return car * np.exp(-t / decay) * np.minimum(t / 0.002, 1)


def pluck(freq, dur=1.2, bright=3500):
    t = tarr(dur)
    ph = 2 * np.pi * freq * t
    s = sum(np.sin(k * ph) / k for k in range(1, 14) if k * freq < SR / 2 - 1000)
    s = lp(s, bright, 2) * np.exp(-t / (0.22 + 60 / freq * 0.2))
    return s * np.minimum(t / 0.003, 1)


def pad_tone(freq, dur, vib=0.0045, nh=10, det=(0.996, 1.0, 1.004), dark=0.5):
    t = tarr(dur)
    s = np.zeros_like(t)
    for dtn in det:
        v = 1 + vib * np.sin(2 * np.pi * (4.3 + rng.uniform(-.4, .4)) * t + rng.uniform(0, 6.28))
        ph = 2 * np.pi * np.cumsum(freq * dtn * v) / SR
        for k in range(1, nh + 1):
            if k * freq * dtn < 9000:
                s += np.sin(k * ph) / (k ** (1.0 + dark))
    return s / len(det)


def sub_boom(dur=2.0, f0=95, f1=27, decay=0.9):
    t = tarr(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.18)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / decay) * np.minimum(t / 0.004, 1)


def noise_burst(dur, decay, fc=None, hpf=None):
    n = rng.normal(size=int(dur * SR)) * np.exp(-tarr(dur) / decay)
    if fc: n = lp(n, fc)
    if hpf: n = hp(n, hpf)
    return n


def snare(dur=0.28):
    t = tarr(dur)
    n = hp(rng.normal(size=len(t)), 1500) * np.exp(-t / 0.07)
    tone = np.sin(2 * np.pi * 190 * t * (1 + 0.3 * np.exp(-t / 0.02))) * np.exp(-t / 0.05)
    return 0.8 * n + 0.6 * tone


def kick(dur=0.45):
    return sub_boom(dur, 150, 48, 0.16)


def formant_voice(f0, dur, formants, vib=0.008):
    t = tarr(dur)
    v = 1 + vib * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6))
    ph = 2 * np.pi * np.cumsum(f0 * v) / SR
    s = np.zeros_like(t)
    for k in range(1, 40):
        fk = k * f0
        if fk > 6000: break
        a = sum(g * np.exp(-0.5 * ((fk - fc) / bw) ** 2) for fc, bw, g in formants)
        s += a * np.sin(k * ph + 0.3 * k)
    return s / 6.0


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


# ====================================================================== 0-10  THE DEEP
# hydrophone drone
t = tarr(DUR)
drone = np.sin(2 * np.pi * 36.7 * t) * 0.5 + np.sin(2 * np.pi * 55.0 * t + 0.4) * 0.3 + np.sin(2 * np.pi * 73.4 * t) * 0.15
drone = drone * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t - 1))
dr_env = np.where(t < 10, np.clip(t / 2.5, 0, 1), np.clip(1 - (t - 10) / 3.0, 0, 1))
place_sub(drone * dr_env, 0, 0.30)
water = lp(rng.normal(size=N), 300) * 0.5 + lp(rng.normal(size=N), 1500) * 0.08
water *= np.where(t < 36, np.clip(t / 3, 0, 1) * np.clip(1 - (t - 34.5) / 1.5, 0, 1), 0)
place(water, 0, 0.30, 0.0, send=0.05)

# sonar pings (echo'd): 1.0 and 5.0
for tp, f in ((EV["sonar1"], 1318.5), (EV["sonar2"], 987.8)):
    tt = tarr(2.4)
    p = np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.35) * np.minimum(tt / 0.006, 1)
    p += 0.3 * np.sin(2 * np.pi * f * 2 * tt) * np.exp(-tt / 0.12)
    place(p, tp, 0.30, -0.15, send=0.9)
    place(p, tp + 0.5, 0.11, 0.35, send=0.9)
    place(p, tp + 1.0, 0.05, -0.4, send=0.9)

# bubbles
for _ in range(34):
    tb = rng.uniform(0.3, 7.4); f = rng.uniform(500, 1800)
    tt = tarr(0.09)
    b = np.sin(2 * np.pi * f * (1 + 4 * tt) * tt) * np.exp(-tt / 0.018)
    place(b, tb, rng.uniform(0.02, 0.07), rng.uniform(-.8, .8), send=0.4)

# rising dread 6.0 -> 8.0, cut to silence into the hit
rs = sweep_noise(1.98, 250, 7000, "band", 1.6, 1.6) * np.linspace(0, 1, int(1.98 * SR)) ** 2
place(rs, EV["riser_start"], 0.55, 0.0, send=0.3)
tt = tarr(2.0)
sw = np.sin(2 * np.pi * np.cumsum(55 * 2 ** (tt * 1.6)) / SR) * np.linspace(0, 1, len(tt)) ** 2
place_sub(sw, 6.0, 0.30)
# heartbeat-like thumps before the crack
for k, tb in enumerate((EV["crack_pre"], EV["crack_pre"] + 0.25)):
    place_sub(sub_boom(0.4, 70, 38, 0.1), tb, 0.55 if k == 0 else 0.35)
# micro fractures: pops with growing density 7.0 -> 8.0
n_pop = 38
for tp in np.sort(7.0 + 1.0 * rng.random(n_pop) ** 0.65):
    place(noise_burst(0.02, 0.004, hpf=2500), tp, rng.uniform(.12, .38), rng.uniform(-.8, .8), send=0.2)

# ====================================================================== 8.0  THE CRACK
place_sub(sub_boom(2.6, 100, 25, 1.0), EV["crack"], 1.0)
place(noise_burst(0.8, 0.22, fc=3500), EV["crack"], 0.7, 0, send=0.6)
place(noise_burst(0.06, 0.012, hpf=1500), EV["crack"], 0.9, 0, send=0.3)
place(bell(587.3, 5.0, 1.4, 2.0), EV["crack"], 0.28, 0.0, send=0.9)         # warm gold light released
place(bell(880.0, 5.0, 1.0, 1.6), EV["crack"] + 0.02, 0.17, 0.3, send=0.9)
# camera whip 8.0 -> 9.8
wh = sweep_noise(1.8, 200, 7500, "band", 1.5, 0.55) * np.sin(np.linspace(0, np.pi, int(1.8 * SR))) ** 1.5
place(wh, EV["camera_swing"], 0.50, -0.4, send=0.3)
# shards: one thud per piece at its exact launch time
info = json.load(open("/content/antikythera/usd/info.json"))
for sh in info["shards"]:
    t0 = sh["t0"]
    f = rng.uniform(70, 150)
    th = noise_burst(0.4, 0.07, fc=900) * 0.8 + sub_boom(0.4, f * 1.5, f * 0.6, 0.12) * 0.6
    place(th, t0, 0.5, rng.uniform(-.9, .9), send=0.35)
    # tumbling clatter later as pieces fly away
    for _ in range(4):
        place(noise_burst(0.03, 0.007, hpf=900, fc=4500), t0 + rng.uniform(0.2, 1.3), rng.uniform(.08, .22),
              rng.uniform(-.9, .9), send=0.3)
# golden glints as the mechanism is revealed
for k in range(11):
    place(bell(midi(86 + int(rng.integers(0, 5)) * 2), 2.2, 1.2, 0.8), EV["crack"] + 0.35 + k * 0.11, 0.07,
          rng.uniform(-.8, .8), send=0.9)

# ====================================================================== 10.0  THE MECHANISM WAKES
# the exact tooth passes of the real gear train
CLICKS = []
for k in range(10):
    tt = tarr(0.03)
    f = rng.uniform(2300, 4300)
    c = hp(rng.normal(size=len(tt)), 1800) * np.exp(-tt / 0.0012)
    c += np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.0045) * 0.9
    c += np.sin(2 * np.pi * f * 0.5 * tt) * np.exp(-tt / 0.009) * 0.3
    CLICKS.append(c)


def tick_layer(gid, gain, pan, pitch_shift=1.0, t_lo=EV["first_tick"], t_hi=36.0):
    times = T.tooth_ticks(gid, t_hi)
    times = times[times >= t_lo]
    for i, tk in enumerate(times):
        # level eases with speed so the dense later ticks blend to a whirr
        w = float(T.omega1(tk))
        lvl = gain * (0.55 + 0.45 / (1 + 0.55 * w)) * (0.85 + 0.3 * rng.random())
        edge = smooth((tk - t_lo) / 0.35)
        c = CLICKS[(i * 7 + hash(gid)) % 10]
        if pitch_shift != 1.0:
            idx = (np.arange(int(len(c) / pitch_shift)) * pitch_shift).astype(int)
            c = c[idx]
        place(c, tk, lvl * edge * (1 - smooth((tk - 35.6) / 0.4)), pan, send=0.18)
    return len(times)


ticks_G1 = tick_layer("G1", 0.14, -0.30)
tick_layer("G3", 0.10, 0.40, 1.5)
tick_layer("G5", 0.09, 0.65, 0.8)
tick_layer("G8", 0.09, -0.65, 1.2)
print("tooth ticks placed (G1 mesh):", ticks_G1)

# first engagement: a big clunk + resonant bell at the exact tick
place(noise_burst(0.12, 0.03, fc=2500), EV["first_tick"], 0.6, 0, send=0.5)
place(bell(293.7, 4.0, 1.0, 1.8), EV["first_tick"], 0.30, -0.2, send=0.9)
place_sub(sub_boom(1.2, 80, 36, 0.5), EV["first_tick"], 0.55)

# machine hum that follows the drive speed
hum_t = tarr(26.0)
w1 = T.omega1(10.0 + hum_t)
f_h = 60 + 28 * w1
hum = np.sin(2 * np.pi * np.cumsum(f_h) / SR) + 0.4 * np.sin(2 * np.pi * np.cumsum(2 * f_h) / SR)
hum *= np.clip((hum_t) / 4, 0, 1) * np.clip((26.0 - hum_t) / 0.8, 0, 1) * 0.07
place(hum, 10.0, 1.0, 0, send=0.2)

# ---------------------------------------------------------------- MUSIC part 1 (D minor)
CH = [  # root, pad notes (midi), arp notes (midi)
    (38, [50, 53, 57, 62], [62, 65, 69, 74, 69, 65]),   # Dm
    (34, [46, 50, 53, 58], [58, 62, 65, 70, 65, 62]),   # Bb
    (31, [43, 46, 50, 55], [55, 58, 62, 67, 62, 58]),   # Gm
    (33, [45, 49, 52, 57], [57, 61, 64, 69, 64, 61]),   # A  (C#)
]


def chord_at(t0):
    return CH[int((t0 - EV["first_tick"]) // BAR) % 4]


# pads, one chord per bar, crossfaded
for b in range(int(EV["first_tick"] / BAR), int(36.0 / BAR)):
    tb = b * BAR
    if tb < EV["first_tick"]: continue
    root, pad, arp = chord_at(tb)
    lvl = 0.12 if tb < EV["pulse_in"] else (0.17 if tb < EV["build"] else 0.22)
    for nn in pad:
        s = pad_tone(midi(nn), BAR + 1.2, dark=0.7)
        e = np.minimum(tarr(BAR + 1.2) / 0.9, 1) * np.exp(-np.maximum(tarr(BAR + 1.2) - BAR, 0) / 0.4)
        place(s * e, tb, lvl * 0.5, rng.uniform(-.6, .6), send=0.55)

# bell on every downbeat (D minor motif)
MOTIF = [74, 72, 69, 65]
for b in range(int(EV["first_tick"] / BAR), int(36.0 / BAR)):
    tb = b * BAR
    if tb < EV["first_tick"] or tb >= 35.0: continue
    root, pad, arp = chord_at(tb)
    place(bell(midi(arp[3] + 12), 3.5, 0.9, 1.4), tb, 0.12 if tb < EV["pulse_in"] else 0.16, 0.25, send=0.9)

# bass pulse + kick from 16.0, on the beat
for tb in np.arange(EV["pulse_in"], 35.0, BEAT):
    root, pad, arp = chord_at(tb)
    bar_pos = ((tb - EV["first_tick"]) % BAR) / BEAT
    g = 0.9 if tb < EV["build"] else 1.0
    place_sub(kick(0.4), tb, 0.75 * g)
    f = midi(root)
    tt = tarr(0.45)
    bsig = np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt)
    place_sub(bsig * np.exp(-tt / 0.22) * np.minimum(tt / 0.004, 1), tb, 0.42)
# gear-driven off-beat "tock" so the machine sounds part of the song
for tb in np.arange(EV["pulse_in"] + BEAT / 2, 30.0, BEAT):
    c = CLICKS[int(tb * 3) % 10]
    place(c, tb, 0.22, 0.5, send=0.25)

# arpeggio: 8th notes from 16.0, bar-length phrase of 6 notes (3 over 2 feel)
step = BEAT / 2
for tb in np.arange(EV["pulse_in"], 35.0, step):
    root, pad, arp = chord_at(tb)
    k = int(round((tb - EV["pulse_in"]) / step)) % 6
    g = 0.11 if tb < EV["build"] else 0.15
    place(pluck(midi(arp[k]), 1.1), tb, g, -0.5 + 1.0 * (k / 5), send=0.45)

# snare roll 30.0 -> 35.0, accelerating
tb = EV["build"]
gap = BEAT
while tb < 35.0:
    u = (tb - EV["build"]) / 5.0
    place(snare(), tb, 0.25 + 0.5 * u, rng.uniform(-.3, .3), send=0.35)
    gap = max(BEAT * (1 - 0.88 * u ** 0.8), 0.045)
    tb += gap
# rising shepard-ish riser under it
rs2 = sweep_noise(5.0, 400, 9000, "band", 1.4, 1.5) * np.linspace(0, 1, int(5.0 * SR)) ** 1.8
place(rs2, EV["build"], 0.42, 0, send=0.3)
tt = tarr(5.0)
for base in (293.7, 440.0, 587.3):
    st = np.sin(2 * np.pi * np.cumsum(base * 2 ** (tt / 5.0 * 1.0)) / SR) * (tt / 5.0) ** 2
    place(st, EV["build"], 0.10, rng.uniform(-.5, .5), send=0.5)

# ================================================================ 35.0  SIGMA LOCK (slider hits the eclipse cell)
place(noise_burst(0.07, 0.01, hpf=1200), EV["sigma_lock"], 0.9, 0, send=0.4)               # latch clack
for f in (587.3, 1174.7, 1760.0, 2349.3):
    place(bell(f, 6.0, 1.3, 3.2), EV["sigma_lock"], 0.22 / (1 + (f > 1000)), 0.0, send=1.0)
place_sub(sub_boom(1.5, 90, 36, 0.6), EV["sigma_lock"], 0.9)
# the long swell into the flash (35.0 -> 35.55 -> 36.0)
sw2 = sweep_noise(1.0, 800, 12000, "band", 1.3, 1.0) * np.linspace(0.2, 1, int(SR)) ** 2.2
place(sw2, 35.0, 0.45, 0, send=0.4)
tt = tarr(1.0)
for base in (587.3, 880.0, 1174.7):
    s_ = np.sin(2 * np.pi * np.cumsum(base * 2 ** (tt * 1.0)) / SR) * np.linspace(0.3, 1, len(tt)) ** 2
    place(s_, 35.0, 0.12, rng.uniform(-.4, .4), send=0.8)

# ================================================================ 36.0  WHITEOUT -> THE SKY
# everything else cuts; one enormous, long-tailed hit
place_sub(sub_boom(4.0, 70, 22, 2.0), EV["flash"], 1.0)
place(noise_burst(1.6, 0.5, fc=5000), EV["flash"], 0.6, 0, send=0.8)
for f in (146.8, 293.7, 440.0, 587.3, 880.0):
    place(bell(f, 8.0, 0.6, 4.0), EV["flash"], 0.18, rng.uniform(-.4, .4), send=1.0)

# sea, wind, gulls-free calm
t2 = tarr(DUR - 36.0)
waves = lp(rng.normal(size=len(t2)), 800)
waves *= (0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t2 - 1.3)) ** 1.5)
wind = sweep_noise(DUR - 36.0, 500, 900, "band", 2.0, 0.3) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t2 + 2)) ** 1.2
amb_env = np.clip((t2 - 0.5) / 3.0, 0, 1) * np.where(t2 < (EV["totality"] - 36.0 - 0.3), 1.0, 0.0)
amb_env = lp(amb_env, 6)  # soften the gate
amb_late = np.where(t2 > (EV["diamond_out"] - 36.0), np.clip((t2 - (EV["diamond_out"] - 36)) / 1.5, 0, 1) * 0.8, 0)
amb_env = np.clip(amb_env + amb_late, 0, 1)
fade_end = np.clip((DUR - 36.0 - t2 - 0.0) / 3.0, 0, 1)
place(waves * amb_env * fade_end, 36.0, 0.14, -0.1, send=0.2)
place(wind * amb_env * fade_end, 36.0, 0.10, 0.2, send=0.3)

# celestial pad in D major, one chord every 2 bars (4 s)
DM = [(38, [50, 54, 57, 62, 66]), (45, [45, 52, 57, 61, 64]), (47, [47, 54, 59, 62, 66]), (43, [43, 50, 55, 59, 62])]
for k, tb in enumerate(np.arange(EV["sun_pad"], 51.0, 4.0)):
    root, notes = DM[k % 4]
    nlen = 5.5
    swell = np.minimum(tarr(nlen) / 1.6, 1) * np.exp(-np.maximum(tarr(nlen) - 4.0, 0) / 0.6)
    for nn in notes:
        place(pad_tone(midi(nn), nlen, dark=0.55) * swell, tb, 0.085, rng.uniform(-.7, .7), send=0.7)
    place_sub(np.sin(2 * np.pi * midi(root) * tarr(nlen)) * swell, tb, 0.18)
# harp-like quarter-note arpeggio under it (pentatonic), 37.0 -> 50.0
PENT = [74, 76, 78, 81, 83, 86, 81, 78]
for i, tb in enumerate(np.arange(EV["sun_pad"] + BEAT, 50.0, BEAT)):
    n = PENT[(i * 3) % len(PENT)]
    place(pluck(midi(n), 1.8, 5000), tb, 0.07, -0.6 + 1.2 * ((i % 8) / 7), send=0.9)

# --- 44.0 totality riser: low strings tension + rising choir air
tt = tarr(8.0)
rise = smooth(tt / 8.0)
for base in (36.7, 55.0, 73.4, 110.0):
    g = np.sin(2 * np.pi * np.cumsum(base * (1 + 0.08 * rise)) / SR) * (0.3 + 0.7 * rise)
    place_sub(g, EV["totality_riser"], 0.14)
rs3 = sweep_noise(8.0, 300, 5000, "band", 1.7, 1.3) * rise ** 2.0
place(rs3, EV["totality_riser"], 0.30 * (1 / 1.0), 0, send=0.5)
# pulse that speeds up as the moon closes in (anchored on the beat grid)
tb = EV["totality_riser"]
while tb < EV["totality"] - 0.05:
    u = (tb - EV["totality_riser"]) / 8.0
    place_sub(sub_boom(0.4, 85, 38, 0.14), tb, 0.25 + 0.55 * u)
    tb += BEAT * (1.0 - 0.55 * u)

# --- crickets awaken in the sudden dusk (51.2 ->)
def crickets(t0, t1, level):
    for i in range(7):
        base = rng.uniform(3900, 5200)
        rate = rng.uniform(17, 24)
        gate_rate = rng.uniform(1.2, 2.2)
        tt = tarr(t1 - t0)
        am = (np.sin(2 * np.pi * rate * tt) > 0.2).astype(float)
        gate = (np.sin(2 * np.pi * gate_rate * tt + rng.uniform(0, 6)) > 0.0).astype(float)
        env = lp(am * gate, 900) * smooth((tt) / 1.5) * smooth((t1 - t0 - tt) / 0.8)
        s = np.sin(2 * np.pi * base * tt)
        place(s * env, t0, level, rng.uniform(-.9, .9), send=0.3)
crickets(EV["diamond_in"] + 0.2, EV["diamond_out"] + 0.1, 0.020)

# --- diamond ring approach: shimmer climbing, wind dies (by amb_env), pad lowers
tt = tarr(1.8)
for base in (1568.0, 2093.0, 2637.0):
    s_ = np.sin(2 * np.pi * np.cumsum(base * (1 + 0.5 * smooth(tt / 1.8))) / SR) * smooth(tt / 1.6) ** 2
    place(s_, 50.2, 0.045, rng.uniform(-.5, .5), send=0.9)

# ================================================================ 52.0  TOTALITY
# the world drops out: boom, then choir + deep drone only
place_sub(sub_boom(5.0, 80, 24, 2.4), EV["totality"], 1.0)
place(noise_burst(0.9, 0.2, fc=1800), EV["totality"], 0.45, 0, send=1.0)
place(bell(73.4 * 4, 9.0, 0.5, 4.5), EV["totality"], 0.22, 0, send=1.0)
dur_tot = EV["diamond_out"] - EV["totality"] + 0.6
AH = [(650, 80, 1.0), (1080, 90, 0.6), (2650, 150, 0.25)]
OO = [(330, 60, 1.0), (750, 80, 0.4), (2500, 150, 0.1)]
for f0, g, pan in ((midi(38), 0.30, 0.0), (midi(45), 0.24, -0.3), (midi(50), 0.22, 0.3), (midi(54), 0.18, -0.15), (midi(57), 0.16, 0.2),
                   (midi(62), 0.12, 0.0), (midi(66), 0.09, 0.15)):
    v = formant_voice(f0, dur_tot, AH if f0 < 150 else OO)
    e = smooth(tarr(dur_tot) / 1.5) * (1 - smooth((tarr(dur_tot) - (dur_tot - 0.7)) / 0.7))
    place(v * e, EV["totality"], g * 0.32, pan, send=0.85)
# slow heartbeat of the moon: 1 thump per beat-pair inside totality
for tb in np.arange(EV["totality"] + 1.0, EV["diamond_out"] - 0.2, BAR / 2):
    place_sub(sub_boom(0.5, 60, 32, 0.15), tb, 0.30)

# ================================================================ 55.5  DIAMOND RING (out)
place(noise_burst(0.5, 0.06, hpf=3000), EV["diamond_out"], 0.5, 0, send=0.6)
place_sub(sub_boom(2.0, 100, 30, 0.9), EV["diamond_out"], 0.85)
for k, nn in enumerate((86, 90, 93, 98, 102)):
    place(bell(midi(nn), 6.0, 1.6, 2.5), EV["diamond_out"] + 0.04 * k, 0.17, -0.6 + 0.3 * k, send=1.0)
for f0, g in ((midi(50), 0.34), (midi(57), 0.30), (midi(62), 0.28), (midi(66), 0.22), (midi(69), 0.18), (midi(74), 0.14)):
    s = pad_tone(f0, 5.2, dark=0.5)
    e = smooth(tarr(5.2) / 0.5) * (1 - smooth((tarr(5.2) - 3.6) / 1.6))
    place(s * e, EV["diamond_out"], g * 0.35, rng.uniform(-.6, .6), send=0.8)
place_sub(np.sin(2 * np.pi * midi(38) * tarr(4.6)) * smooth(tarr(4.6) / 0.4) * (1 - smooth((tarr(4.6) - 3.0) / 1.6)),
          EV["diamond_out"], 0.45)
# farewell figure: three rising pluck notes on the beat after the ring
for i, nn in enumerate((81, 86, 90, 93)):
    place(pluck(midi(nn), 2.4, 6000), EV["diamond_out"] + 1.5 + BEAT * i, 0.09, -0.4 + 0.27 * i, send=1.0)

# ====================================================================== reverb + master
def make_ir(dur, decay, seed):
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    ir = r.normal(size=n) * np.exp(-np.arange(n) / SR / decay)
    ir = lp(ir, 5200, 1)
    ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
    return ir / np.sqrt((ir ** 2).sum())

IRl, IRr = make_ir(3.6, 1.15, 5), make_ir(3.6, 1.15, 6)
wetL = fftconvolve(sndL, IRl)[:N]
wetR = fftconvolve(sndR, IRr)[:N]
L = dryL + subL + wetL * 0.62
R = dryR + subR + wetR * 0.62

# gentle bus compression via soft clipping, final fade with picture fade (57.6 -> 60)
tt = np.arange(N) / SR
fade = 1.0 - smooth((tt - 57.4) / 2.5)
fade = np.where(tt < 0.05, tt / 0.05, fade)
peak = max(np.abs(L).max(), np.abs(R).max())
g = 0.95 / peak
L = np.tanh(L * g * 1.25) / np.tanh(1.25)
R = np.tanh(R * g * 1.25) / np.tanh(1.25)
L *= fade * 0.93; R *= fade * 0.93
out = np.stack([L, R], 1)
pcm = (np.clip(out, -1, 1) * 32767).astype("<i2")
with wave.open("/content/antikythera/audio/antikythera.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote wav: %.2fs  peak=%.3f  pre-limit-peak=%.2f" % (N / SR, np.abs(out).max(), peak))
