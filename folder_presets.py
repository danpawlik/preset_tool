#!/usr/bin/env python3
"""Derive Lightroom presets for every sub-folder of a samples directory.

Biased toward hand-crafted phone looks (vivo_grok style):
- Strong shadow lift, controlled highlights
- Mild clarity/texture, pleasant vibrance
- Soft HSL, mild split toning + vignette

Usage:
  python3 folder_presets.py SAMPLES_DIR OUT_DIR --absolute -s 0.9
  python3 folder_presets.py SAMPLES_DIR OUT_DIR -s 0.9
"""
import argparse, os, re, sys
import warnings
import numpy as np
warnings.filterwarnings('ignore', 'All-NaN')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_preset import (
    BANDS, load, list_images, band_stats, channel_curves, apply_curves,
    xmp, Q, estimate_basic_sliders, HSL_LIMIT, BASE
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


def analyse_character(imgs):
    qs = [np.percentile(im, Q, axis=0) for im in imgs]
    q = np.mean(qs, axis=0)
    lum = q.mean(axis=1)
    deep, shad, mid, high, whit = lum[1], lum[2], lum[5], lum[8], lum[9]
    shadow_lift = max(0, 40 - shad)
    highlight_rolloff = max(0, high - 200)
    mid_bright = mid - 128
    warmth = float(q[5, 0] - q[5, 2])
    sats = []
    for im in imgs:
        mx, mn = im.max(1), im.min(1)
        s = np.where(mx > 30, (mx - mn) / np.maximum(mx, 1), 0)
        mid_mask = (im.mean(1) > 60) & (im.mean(1) < 200)
        if mid_mask.any():
            sats.append(s[mid_mask].mean())
    mean_sat = float(np.mean(sats)) if sats else 0.3
    return dict(lum=lum, q=q, shadow_lift=shadow_lift, highlight_rolloff=highlight_rolloff,
                mid_bright=mid_bright, warmth=warmth, mean_sat=mean_sat,
                deep=deep, shad=shad, mid=mid, high=high, whit=whit)


def absolute_basic_sliders(char, strength=1.0):
    """Start from vivo_grok BASE, nudge by measured character."""
    s = strength
    def n(base, delta, lo, hi):
        return float(np.clip((base + delta) * s, lo, hi))

    shadows = n(BASE['Shadows2012'], char['shadow_lift'] * 0.2, 15, 48)
    highlights = n(BASE['Highlights2012'], -char['highlight_rolloff'] * 0.15, -48, -10)
    contrast = n(BASE['Contrast2012'], (char['high'] - char['shad'] - 150) * 0.05, 6, 24)
    exposure = n(BASE['Exposure2012'], char['mid_bright'] / 100.0, -0.3, 0.3)
    whites = n(BASE['Whites2012'], (char['whit'] - 230) * 0.1, -15, 20)
    blacks = n(BASE['Blacks2012'], (30 - char['deep']) * 0.08, -20, 10)
    clarity = n(BASE['Clarity2012'], 0, 3, 16)
    texture = n(BASE['Texture'], 0, 2, 14)
    vibrance = n(BASE['Vibrance'], (0.35 - char['mean_sat']) * 8, 4, 18)
    dehaze = n(BASE['Dehaze'], 0, 0, 8)

    return {
        'Exposure2012': round(exposure, 2),
        'Contrast2012': int(round(contrast)),
        'Highlights2012': int(round(highlights)),
        'Shadows2012': int(round(shadows)),
        'Whites2012': int(round(whites)),
        'Blacks2012': int(round(blacks)),
        'Texture': int(round(texture)),
        'Clarity2012': int(round(clarity)),
        'Vibrance': int(round(vibrance)),
        'Saturation': int(round(BASE['Saturation'] * s)),
        'Dehaze': int(round(dehaze)),
    }


def absolute_hsl(char, strength=1.0):
    s = strength
    hsl = {
        'SaturationAdjustmentGreen': int(np.clip(6 * s, -10, 12)),
        'LuminanceAdjustmentGreen': int(np.clip(3 * s, -8, 8)),
        'SaturationAdjustmentOrange': int(np.clip(5 * s, -10, 12)),
        'SaturationAdjustmentYellow': int(np.clip(3 * s, -8, 10)),
        'LuminanceAdjustmentOrange': int(np.clip(3 * s, -8, 8)),
        'HueAdjustmentOrange': int(np.clip(-2 * s, -6, 6)),
        'HueAdjustmentYellow': int(np.clip(-2 * s, -6, 6)),
        'HueAdjustmentAqua': int(np.clip(2 * s, -6, 6)),
    }
    if char['warmth'] > 8:
        hsl['HueAdjustmentBlue'] = int(np.clip(3 * s, -8, 8))
    return {k: int(np.clip(v, -HSL_LIMIT, HSL_LIMIT)) for k, v in hsl.items()}


def absolute_curves(char, strength=1.0):
    s = strength
    shadow_y = int(np.clip(10 + char['shadow_lift'] * 0.12, 6, 20) * s)
    mid_y = 128 + int(np.clip(char['mid_bright'] * 0.12, -6, 6) * s)
    pts = [(0, 0), (32, 18 + shadow_y), (64, 52 + shadow_y // 2),
           (128, mid_y), (192, 210), (255, 255)]
    ys = [p[1] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i-1] + 1)
    pts = [(p[0], min(255, ys[i])) for i, p in enumerate(pts)]
    return [pts, pts, pts]


def write_absolute_preset(name, imgs, strength, out_dir):
    char = analyse_character(imgs)
    basic = absolute_basic_sliders(char, strength)
    hsl = absolute_hsl(char, strength)
    curves = absolute_curves(char, strength)
    desc = f'Absolute vivo_grok-biased look from {len(imgs)} photos'
    path = os.path.join(out_dir, name.replace(' ', '_') + '.xmp')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xmp(name, curves, hsl, basic, desc, group='Absolute Looks'))
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


def compute_hsl_relative(sample_px, curves, strength):
    st = [band_stats(px) for px in sample_px]
    rf = [band_stats(apply_curves(px, curves)) for px in sample_px]
    out = {}
    for b in BANDS:
        s = [x[b] for x in st if b in x]
        r = [x[b] for x in rf if b in x]
        if len(s) < 2 or len(r) < 2:
            continue
        (s_sat, s_hue, s_lum), (r_sat, r_hue, r_lum) = np.mean(s, 0), np.mean(r, 0)
        sat = (s_sat / max(r_sat, 1e-6) - 1) * 35
        hue = (s_hue - r_hue) * 1.1
        lum = (s_lum - r_lum) * 60
        out[f'SaturationAdjustment{b}'] = sat
        out[f'HueAdjustment{b}'] = hue
        out[f'LuminanceAdjustment{b}'] = lum
    # mild green/orange bias
    out['SaturationAdjustmentGreen'] = out.get('SaturationAdjustmentGreen', 0) + 4 * strength
    out['SaturationAdjustmentOrange'] = out.get('SaturationAdjustmentOrange', 0) + 3 * strength
    return {k: int(np.clip(round(v * strength), -HSL_LIMIT, HSL_LIMIT)) for k, v in out.items()}


def write_relative_preset(name, style_q, ref_q, sample_px, strength, out_dir, group, desc):
    curves = channel_curves(style_q, ref_q, strength)
    basic = estimate_basic_sliders(style_q, ref_q, strength)  # already vivo_grok-biased
    hsl = compute_hsl_relative(sample_px, curves, strength)
    path = os.path.join(out_dir, name.replace(' ', '_') + '.xmp')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xmp(name, curves, hsl, basic, desc, group=group))
    print('wrote', path)
    print('  Basic:', {k: v for k, v in basic.items() if v != 0 and v != 0.0})
    print('  HSL:', {k: v for k, v in hsl.items() if v})
    return path


def main(src, out, exclude=(), groups=True, min_photos=3, strength=1.0, absolute=False):
    os.makedirs(out, exist_ok=True)
    pix = scan(src, set(exclude), min_photos)
    print('Folders:', ', '.join(f'{d} ({len(v)})' for d, v in pix.items()))
    print('Mode:', 'ABSOLUTE (vivo_grok-biased)' if absolute else 'RELATIVE (vivo_grok-biased)')

    if absolute:
        for d, imgs in pix.items():
            kept = list(imgs.values())
            if len(kept) < min_photos:
                print(f'  {d}: too few photos, skip')
                continue
            write_absolute_preset(pretty(d), kept, strength, out)
        return

    if len(pix) < 2:
        sys.exit('Relative mode needs at least 2 folders. Use --absolute for single-folder sets.')

    feats = {d: {k: features(v) for k, v in imgs.items()} for d, imgs in pix.items()}

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
        write_relative_preset(name, style_q, ref_q, sample, strength, out,
                              group='Folder Looks', desc='Relative vivo_grok-biased')


def selftest():
    assert pretty('vivo_300_pro') == 'Vivo 300 Pro'
    rng = np.random.default_rng(0)
    fake = [rng.uniform(20, 240, (3000, 3)) for _ in range(4)]
    char = analyse_character(fake)
    basic = absolute_basic_sliders(char, 1.0)
    assert basic['Shadows2012'] >= 15
    assert basic['Highlights2012'] <= -10
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
                    help='Character-based absolute looks (recommended)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    main(a.samples_dir, a.out_dir, a.exclude, not a.no_groups,
         a.min_photos, a.strength, a.absolute)
