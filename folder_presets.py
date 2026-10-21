#!/usr/bin/env python3
"""Derive Lightroom presets for every sub-folder of a samples directory.

Improved version:
- Basic tonal sliders + soft HSL (±12)
- Clean XMP format compatible with Lightroom
- --absolute mode (recommended): each folder = absolute look vs neutral
- Relative mode: vs average of all folders (Bright / Low Light + groups)

Usage:
  python3 folder_presets.py SAMPLES_DIR OUT_DIR --absolute -s 0.85
  python3 folder_presets.py SAMPLES_DIR OUT_DIR              # relative
"""
import argparse, os, re, sys
import warnings
import numpy as np
warnings.filterwarnings('ignore', 'All-NaN')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_preset import (
    BANDS, load, list_images, band_stats, channel_curves, apply_curves,
    xmp, Q, estimate_basic_sliders, HSL_LIMIT
)

LIGHTS = ['Bright Light', 'Low Light']
NB = len(BANDS)


def pretty(name):
    return ' '.join(w if w.isupper() else w.capitalize()
                    for w in re.split(r'[_\-\s]+', name) if w)


def group_of(name):
    return re.split(r'[_\-]', name, maxsplit=1)[0]


def features(px):
    q = np.percentile(px, Q, axis=0).ravel()
    bs = band_stats(px)
    b = np.array([bs.get(n, (np.nan,) * 3) for n in BANDS]).ravel()
    return np.concatenate([q, b])


def robust_keep(F, names, label, z=3.5):
    T = F[:, :33]
    med = np.median(T, 0)
    mad = np.median(np.abs(T - med), 0) * 1.4826 + 1
    score = np.mean(np.abs(T - med) / mad, 1)
    keep = score < z
    for n, s in zip(np.array(names)[~keep], score[~keep]):
        print(f'  [{label}] ignoring outlier {n} (score {s:.1f})')
    return F[keep]


def identity_quantiles():
    q = np.array(Q, dtype=float)
    vals = np.clip(q / 100.0 * 255.0, 0, 255)
    return np.stack([vals, vals, vals], axis=1)


def compute_hsl(sample_px, curves, strength):
    st = [band_stats(px) for px in sample_px]
    rf = [band_stats(apply_curves(px, curves)) for px in sample_px]
    out = {}
    for b in BANDS:
        s = [x[b] for x in st if b in x]
        r = [x[b] for x in rf if b in x]
        if len(s) < 2 or len(r) < 2:
            continue
        (s_sat, s_hue, s_lum), (r_sat, r_hue, r_lum) = np.mean(s, 0), np.mean(r, 0)
        sat = (s_sat / max(r_sat, 1e-6) - 1) * 40
        hue = (s_hue - r_hue) * 1.2
        lum = (s_lum - r_lum) * 70
        out[f'SaturationAdjustment{b}'] = sat
        out[f'HueAdjustment{b}'] = hue
        out[f'LuminanceAdjustment{b}'] = lum
    return {k: int(np.clip(round(v * strength), -HSL_LIMIT, HSL_LIMIT)) for k, v in out.items()}


def write_preset(name, style_q, ref_q, sample_px, strength, out_dir, group, desc):
    curves = channel_curves(style_q, ref_q, strength)
    basic = estimate_basic_sliders(style_q, ref_q, strength)
    hsl = compute_hsl(sample_px, curves, strength)

    path = os.path.join(out_dir, name.replace(' ', '_') + '.xmp')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xmp(name, curves, hsl, basic, desc, group=group))
    print('wrote', path)
    print('  Basic:', {k: v for k, v in basic.items() if v != 0 and v != 0.0})
    print('  HSL:', {k: v for k, v in hsl.items() if v})
    return path


def scan(src, exclude, min_photos):
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
            except Exception as e:
                print(f'skip {f}: {e}', file=sys.stderr)
        if len(imgs) >= min_photos:
            pix[d] = imgs
        elif imgs:
            print(f'skip folder {d}: {len(imgs)} photos, need {min_photos}')
    if len(pix) < 1:
        sys.exit(f'{src}: need at least 1 sub-folder with {min_photos}+ photos')
    return pix


def main(src, out, exclude=(), groups=True, min_photos=3, strength=1.0, absolute=False):
    os.makedirs(out, exist_ok=True)
    pix = scan(src, set(exclude), min_photos)
    feats = {d: {k: features(v) for k, v in imgs.items()} for d, imgs in pix.items()}
    print('Folders:', ', '.join(f'{d} ({len(v)})' for d, v in pix.items()))
    print('Mode:', 'ABSOLUTE look' if absolute else 'RELATIVE (vs average of folders)')

    if absolute:
        neutral_q = identity_quantiles()
        for d, imgs in pix.items():
            names = list(imgs.keys())
            F = np.array([feats[d][k] for k in names])
            F = robust_keep(F, names, d)
            if len(F) < min_photos:
                print(f'  {d}: too few photos after outlier removal, skip')
                continue
            style_q = np.nanmedian(F[:, :33], 0).reshape(11, 3)
            sample_px = list(imgs.values())[::max(1, len(imgs)//8)]
            name = pretty(d)
            desc = f'Absolute look from {len(F)} photos'
            write_preset(name, style_q, neutral_q, sample_px, strength, out,
                         group='Absolute Looks', desc=desc)
        return

    # relative mode
    if len(pix) < 2:
        sys.exit('Relative mode needs at least 2 folders. Use --absolute for single-folder sets.')

    def mid(f): return f[:33].reshape(11, 3)[5].mean()
    split = np.median([mid(f) for fs in feats.values() for f in fs.values()])
    light = {d: {k: LIGHTS[int(mid(f) < split)] for k, f in fs.items()} for d, fs in feats.items()}
    print(f'Bright/low light split at median luminance {split:.0f}/255')

    def profile(d, names, label):
        print(f'  {label}: {len(names)} photos')
        return np.nanmedian(robust_keep(np.array([feats[d][k] for k in names]), names, label), 0)

    jobs = []
    for lt in LIGHTS:
        names = {d: [k for k in feats[d] if light[d][k] == lt] for d in feats}
        prof = {d: profile(d, n, f'{d} / {lt}') for d, n in names.items() if len(n) >= min_photos}
        for d in sorted(set(feats) - set(prof)):
            print(f'  {d} / {lt}: only {len(names[d])} photos, no preset')
        if len(prof) < 2:
            continue
        base = np.nanmean(list(prof.values()), 0)
        for d, p in prof.items():
            sample = [pix[d][k] for k in names[d]][::2]
            jobs.append((f'{pretty(d)} - {lt}', p, base, sample))

    if groups:
        grouped = {}
        for d in feats:
            grouped.setdefault(group_of(d), []).append(d)
        grouped = {g: ds for g, ds in grouped.items() if len(ds) > 1}
        if grouped:
            prof = {d: profile(d, list(feats[d]), f'{d} / all') for d in feats}
            base = np.nanmean(list(prof.values()), 0)
            for g, ds in grouped.items():
                sample = [px for d in ds for px in list(pix[d].values())[::4]]
                target = np.mean([prof[d] for d in ds], 0)
                jobs.append((f'{pretty(g)} Look', target, base, sample))

    print()
    for name, t, b, sample in jobs:
        style_q = t[:33].reshape(11, 3)
        ref_q = b[:33].reshape(11, 3)
        desc = 'Relative look vs average of folders'
        write_preset(name, style_q, ref_q, sample, strength, out,
                     group='Folder Looks', desc=desc)


def selftest():
    assert pretty('vivo_300_pro') == 'Vivo 300 Pro'
    assert group_of('vivo_300_pro') == 'vivo'
    print('selftest ok')


if __name__ == '__main__':
    if '--selftest' in sys.argv[1:]:
        selftest()
        sys.exit()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('samples_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--exclude', nargs='+', default=[], metavar='NAME')
    ap.add_argument('--no-groups', action='store_true')
    ap.add_argument('--min-photos', type=int, default=3)
    ap.add_argument('-s', '--strength', type=float, default=1.0)
    ap.add_argument('--absolute', action='store_true',
                    help='Absolute looks (no comparison to other folders). Recommended.')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    main(a.samples_dir, a.out_dir, a.exclude, not a.no_groups,
         a.min_photos, a.strength, a.absolute)
