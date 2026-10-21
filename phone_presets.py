#!/usr/bin/env python3
"""Derive Lightroom presets per phone and per brand from sample folders.

  phone_presets.py SAMPLES_DIR OUT_DIR

Per phone, 2 presets: photos are split into bright and dim scenes (by median
luminance, split at the median of all photos). Each preset = that phone's median
render in that light vs the average of all phones in the same light.
Per brand, 1 preset: the brand's overall median render vs the average of all phones.
Photos far from their phone's typical render are ignored.

Note: on held-out same-scene pairs the X300-vs-X500 and 17-vs-18 differences
predicted the other phone no better than chance (vivo 10/19, xiaomi 3/6), so
same-brand phone presets come out similar; the brand split is the robust part.
"""
import glob, os, sys
import warnings
import numpy as np
warnings.filterwarnings('ignore', 'All-NaN')  # colour absent in a small set; hsl_from skips it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_preset import BANDS, load, band_stats, channel_curves, apply_curves, xmp, rgb_to_hsv, Q

PHONES = {'vivo_300_pro': 'Vivo X300 Pro', 'vivo_500_pro': 'Vivo X500 Pro',
          'xiaomi_17': 'Xiaomi 17', 'xiaomi_18': 'Xiaomi 18'}
LIGHTS = ['Bright Light', 'Low Light']
BRANDS = {'Vivo': ['vivo_300_pro', 'vivo_500_pro'], 'Xiaomi': ['xiaomi_17', 'xiaomi_18']}
EXCLUDE = {'262830-329.jpg',       # EXIF says X300 Pro, sits in the X500 folder
           'HUOVrgla4AAowpt.jpg'}  # watermark: vivo X300 Ultra
NB = len(BANDS)


def features(px):
    """11x3 channel quantiles + 8 bands x (sat, hue, lum), NaN where a colour is absent."""
    q = np.percentile(px, Q, axis=0).ravel()
    bs = band_stats(px)
    b = np.array([bs.get(n, (np.nan,) * 3) for n in BANDS]).ravel()
    return np.concatenate([q, b])


def robust_keep(F, names, label, z=3.5):
    """Drop photos whose tone quantiles sit far from the phone's median render."""
    T = F[:, :33]
    med = np.median(T, 0)
    mad = np.median(np.abs(T - med), 0) * 1.4826 + 1
    score = np.mean(np.abs(T - med) / mad, 1)
    keep = score < z
    for n, s in zip(np.array(names)[~keep], score[~keep]):
        print(f'  [{label}] ignoring outlier {n} (score {s:.1f})')
    return F[keep]


def hsl_from(target_b, base_b, sim_b):
    """Slider values from band stats (sat, hue, lum) after removing what the curves already do."""
    out = {}
    for i, n in enumerate(BANDS):
        t, b, s = target_b[i], base_b[i], sim_b[i]
        if np.isnan(t).any() or np.isnan(b).any():
            continue
        # ponytail: linear slider scales are estimates (sat %, ~3 steps/deg hue, lum x200), not LR-calibrated
        out[f'SaturationAdjustment{n}'] = (t[0] - b[0] - s[0]) / b[0] * 100
        out[f'HueAdjustment{n}'] = (t[1] - b[1] - s[1]) * 3
        out[f'LuminanceAdjustment{n}'] = (t[2] - b[2] - s[2]) * 200
    return {k: int(np.clip(round(v), -30, 30)) for k, v in out.items()}


def render(px, curves, hsl):
    """Approximate preset render (curves + HSL) for previews; not a Lightroom render."""
    px = apply_curves(px, curves)
    h, s, v = rgb_to_hsv(px)
    for n, c in BANDS.items():
        w = np.clip(1 - np.abs((h - c + 180) % 360 - 180) / 30, 0, 1) * np.clip(s * 4, 0, 1)
        s = s * (1 + w * hsl.get(f'SaturationAdjustment{n}', 0) / 100)
        h = h + w * hsl.get(f'HueAdjustment{n}', 0) / 3
        v = v * (1 + w * hsl.get(f'LuminanceAdjustment{n}', 0) / 200)
    s, v, h = np.clip(s, 0, 1), np.clip(v, 0, 1), (h % 360) / 60
    i, f = np.floor(h).astype(int) % 6, h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    sel = [i == k for k in range(6)]
    rgb = [np.select(sel, c) for c in ([v, q, p, p, t, v], [t, v, v, q, p, p], [p, p, t, v, v, q])]
    return np.stack(rgb, 1) * 255


def describe(name, d):
    q = d[:33].reshape(11, 3)
    lum = q.mean(1)
    sat = d[33:].reshape(NB, 3)[:, 0]
    parts = [f'shadows {lum[1]:+.0f}', f'mids {lum[5]:+.0f}', f'highlights {lum[9]:+.0f}',
             f'warmth (R-B mids) {q[5, 0] - q[5, 2]:+.0f}', f'tint (G mids) {q[5, 1] - (q[5, 0] + q[5, 2]) / 2:+.0f}']
    top = [f'{n} {s * 100:+.0f}%' for n, s in zip(BANDS, sat) if not np.isnan(s) and abs(s) >= 0.02]
    print(f'  {name:28s} ' + ', '.join(parts) + ' | saturation: ' + (', '.join(top) or 'same'))


def main(src, out):
    os.makedirs(out, exist_ok=True)
    feats, pix = {}, {}
    for d in PHONES:
        files = [f for f in sorted(glob.glob(os.path.join(src, d, '*.jpg'))) if os.path.basename(f) not in EXCLUDE]
        pix[d] = {os.path.basename(f): load(f) for f in files}
        feats[d] = {k: features(v) for k, v in pix[d].items()}

    def mid(f): return f[:33].reshape(11, 3)[5].mean()
    split = np.median([mid(f) for d in PHONES for f in feats[d].values()])
    light = {d: {k: LIGHTS[int(mid(f) < split)] for k, f in feats[d].items()} for d in PHONES}
    print(f'Bright/low light split at median luminance {split:.0f}/255')

    def profile(d, keep, label):
        names = [k for k in feats[d] if keep(k)]
        if len(names) < 3:
            sys.exit(f'{label}: only {len(names)} photos, need 3')
        print(f'  {label}: {len(names)} photos')
        return np.nanmedian(robust_keep(np.array([feats[d][k] for k in names]), names, label), 0)

    jobs = []  # (preset name, target profile, baseline profile, sample photos for HSL correction)
    for lt in LIGHTS:
        prof = {d: profile(d, lambda k, d=d: light[d][k] == lt, f'{PHONES[d]} / {lt}') for d in PHONES}
        base = np.nanmean(list(prof.values()), 0)
        sample = [pix[d][k] for d in PHONES for k in pix[d] if light[d][k] == lt][::2]
        jobs += [(f'{PHONES[d]} - {lt}', prof[d], base, sample) for d in PHONES]
    prof = {d: profile(d, lambda k: True, f'{PHONES[d]} / all') for d in PHONES}
    base = np.nanmean(list(prof.values()), 0)
    sample = [px for d in PHONES for px in list(pix[d].values())[::4]]
    jobs += [(f'{b} Look', np.mean([prof[d] for d in ds], 0), base, sample) for b, ds in BRANDS.items()]

    print('\nSummary vs the average of all four phones in the same light (tone in 0-255 levels):')
    for name, t, b, _ in jobs:
        describe(name, t - b)
    print()

    previews = {}
    for name, t, b, sample in jobs:
        curves = channel_curves(t[:33].reshape(11, 3), b[:33].reshape(11, 3), 1.0)
        before = np.nanmedian([features(px)[33:] for px in sample], 0)
        after = np.nanmedian([features(apply_curves(px, curves))[33:] for px in sample], 0)
        hsl = hsl_from(t[33:].reshape(NB, 3), b[33:].reshape(NB, 3), (after - before).reshape(NB, 3))
        path = os.path.join(out, name.replace(' ', '_') + '.xmp')
        with open(path, 'w') as f:
            f.write(xmp(name, curves, hsl, 'Median phone render vs the average of the sampled phones; outlier photos ignored.')
                    .replace('Generated Looks', 'Phone Looks'))
        previews[name] = (curves, hsl)
        print('wrote', path)
    return previews, pix


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
