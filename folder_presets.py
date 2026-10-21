#!/usr/bin/env python3
"""Derive Lightroom presets for every sub-folder of a samples directory.

  folder_presets.py SAMPLES_DIR OUT_DIR [--exclude NAME ...] [--no-groups] [--min-photos N]

Each sub-folder with images is one "device" (a phone, camera, film stock, editor...);
its preset name comes from the folder name (vivo_300_pro -> "Vivo 300 Pro").

Per folder, 2 presets: photos are split into bright and dim scenes (by median
luminance, split at the median of all photos). Each preset = that folder's median
render in that light vs the average of all folders in the same light.

Per group, 1 preset: folders sharing the name prefix before the first "_" or "-"
(vivo_300_pro + vivo_500_pro -> "Vivo") get a combined look from all their photos.

Photos far from their folder's typical render are ignored.
"""
import argparse, os, re, sys
import warnings
import numpy as np
warnings.filterwarnings('ignore', 'All-NaN')  # colour absent in a small set; hsl_from skips it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_preset import BANDS, load, list_images, band_stats, channel_curves, apply_curves, xmp, rgb_to_hsv, Q

LIGHTS = ['Bright Light', 'Low Light']
NB = len(BANDS)


def pretty(name):
    return ' '.join(w if w.isupper() else w.capitalize() for w in re.split(r'[_\-\s]+', name) if w)


def group_of(name):
    return re.split(r'[_\-]', name, maxsplit=1)[0]


def features(px):
    """11x3 channel quantiles + 8 bands x (sat, hue, lum), NaN where a colour is absent."""
    q = np.percentile(px, Q, axis=0).ravel()
    bs = band_stats(px)
    b = np.array([bs.get(n, (np.nan,) * 3) for n in BANDS]).ravel()
    return np.concatenate([q, b])


def robust_keep(F, names, label, z=3.5):
    """Drop photos whose tone quantiles sit far from the folder's median render."""
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
    print(f'  {name:32s} ' + ', '.join(parts) + ' | saturation: ' + (', '.join(top) or 'same'))


def scan(src, exclude, min_photos):
    """{folder name: {file name: pixels}} for every sub-folder with enough readable images."""
    pix = {}
    for d in sorted(os.listdir(src)):
        path = os.path.join(src, d)
        if d in exclude or not os.path.isdir(path):
            continue
        imgs = {}
        for f in list_images(path):
            if os.path.basename(f) in exclude:
                continue
            try:
                imgs[os.path.basename(f)] = load(f)
            except Exception as e:  # skip unreadable files, keep going
                print(f'skip {f}: {e}', file=sys.stderr)
        if len(imgs) >= min_photos:
            pix[d] = imgs
        elif imgs:
            print(f'skip folder {d}: {len(imgs)} photos, need {min_photos}')
    if len(pix) < 2:
        sys.exit(f'{src}: need at least 2 sub-folders with {min_photos}+ photos to compare, found {len(pix)}')
    return pix


def main(src, out, exclude=(), groups=True, min_photos=3):
    os.makedirs(out, exist_ok=True)
    pix = scan(src, set(exclude), min_photos)
    feats = {d: {k: features(v) for k, v in imgs.items()} for d, imgs in pix.items()}
    print('Folders:', ', '.join(f'{d} ({len(v)})' for d, v in pix.items()))

    def mid(f): return f[:33].reshape(11, 3)[5].mean()
    split = np.median([mid(f) for fs in feats.values() for f in fs.values()])
    light = {d: {k: LIGHTS[int(mid(f) < split)] for k, f in fs.items()} for d, fs in feats.items()}
    print(f'Bright/low light split at median luminance {split:.0f}/255')

    def profile(d, names, label):
        print(f'  {label}: {len(names)} photos')
        return np.nanmedian(robust_keep(np.array([feats[d][k] for k in names]), names, label), 0)

    jobs = []  # (preset name, target profile, baseline profile, sample photos for HSL correction)
    for lt in LIGHTS:
        names = {d: [k for k in feats[d] if light[d][k] == lt] for d in feats}
        prof = {d: profile(d, n, f'{d} / {lt}') for d, n in names.items() if len(n) >= min_photos}
        for d in sorted(set(feats) - set(prof)):
            print(f'  {d} / {lt}: only {len(names[d])} photos, no preset')
        if len(prof) < 2:
            continue  # nothing to compare against in this light
        base = np.nanmean(list(prof.values()), 0)
        sample = [pix[d][k] for d in prof for k in names[d]][::2]
        jobs += [(f'{pretty(d)} - {lt}', p, base, sample) for d, p in prof.items()]

    if groups:
        grouped = {}
        for d in feats:
            grouped.setdefault(group_of(d), []).append(d)
        grouped = {g: ds for g, ds in grouped.items() if len(ds) > 1}
        if grouped:
            prof = {d: profile(d, list(feats[d]), f'{d} / all') for d in feats}
            base = np.nanmean(list(prof.values()), 0)
            sample = [px for d in pix for px in list(pix[d].values())[::4]]
            jobs += [(f'{pretty(g)} Look', np.mean([prof[d] for d in ds], 0), base, sample) for g, ds in grouped.items()]

    print('\nSummary vs the average of all folders in the same light (tone in 0-255 levels):')
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
            f.write(xmp(name, curves, hsl, 'Median render vs the average of the sampled folders; outlier photos ignored.')
                    .replace('Generated Looks', 'Folder Looks'))
        previews[name] = (curves, hsl)
        print('wrote', path)
    return previews, pix


def selftest():
    assert pretty('vivo_300_pro') == 'Vivo 300 Pro' and pretty('my-FILM stock') == 'My FILM Stock'
    assert group_of('vivo_300_pro') == 'vivo' and group_of('xiaomi-17') == 'xiaomi' and group_of('pixel') == 'pixel'
    print('selftest ok')


if __name__ == '__main__':
    if '--selftest' in sys.argv[1:]:
        selftest()
        sys.exit()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('samples_dir', help='directory whose sub-folders each hold one look')
    ap.add_argument('out_dir', help='where to write the .xmp presets')
    ap.add_argument('--exclude', nargs='+', default=[], metavar='NAME',
                    help='file or folder names to ignore (e.g. wrong-camera shots)')
    ap.add_argument('--no-groups', action='store_true', help='skip the combined preset per name prefix')
    ap.add_argument('--min-photos', type=int, default=3, help='photos needed per folder and light (default 3)')
    ap.add_argument('--selftest', action='store_true', help='run the built-in check and exit')
    a = ap.parse_args()
    main(a.samples_dir, a.out_dir, a.exclude, not a.no_groups, a.min_photos)
