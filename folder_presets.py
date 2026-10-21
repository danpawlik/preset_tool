#!/usr/bin/env python3
"""Derive Lightroom presets for every sub-folder of a samples directory.

Absolute mode is strongly biased toward hand-crafted phone looks (vivo_grok):
- Base: Shadows +35, Highlights -30, Contrast +12, Vibrance +10, etc.
- Single luminance curve with shadow lift (not crushing RGB curves)
- Soft HSL + sharpening + NR + split toning + vignette
- Measured character only gently nudges the base (small samples won't drag Exposure negative)

Usage:
  python3 folder_presets.py SAMPLES_DIR OUT_DIR --absolute -s 0.95
  python3 folder_presets.py SAMPLES_DIR OUT_DIR -s 0.9
  python3 folder_presets.py SAMPLES_DIR OUT_DIR -s 0.9 --by-light
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
    # Use median of per-image mids so one dark night shot doesn't dominate
    per_mid = [np.percentile(im, 50, axis=0).mean() for im in imgs]
    mid_bright = float(np.median(per_mid) - 128)
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
                deep=deep, shad=shad, mid=mid, high=high, whit=whit, n=len(imgs))


def absolute_basic_sliders(char, strength=1.0):
    """Almost pure BASE; character only tiny nudges. Prevents dark samples → negative Exposure."""
    s = strength
    # With few photos, trust BASE even more
    w = 0.12 if char['n'] < 6 else 0.22

    def n(base, delta, lo, hi):
        return float(np.clip((base + delta * w) * s, lo, hi))

    return {
        'Exposure2012': round(n(BASE['Exposure2012'], char['mid_bright'] / 120.0, -0.15, 0.25), 2),
        'Contrast2012': int(round(n(BASE['Contrast2012'], 0, 8, 22))),
        'Highlights2012': int(round(n(BASE['Highlights2012'], -char['highlight_rolloff'] * 0.1, -45, -15))),
        'Shadows2012': int(round(n(BASE['Shadows2012'], char['shadow_lift'] * 0.1, 20, 45))),
        'Whites2012': int(round(n(BASE['Whites2012'], 0, 0, 16))),
        'Blacks2012': int(round(n(BASE['Blacks2012'], 0, -16, 0))),
        'Texture': int(round(n(BASE['Texture'], 0, 3, 12))),
        'Clarity2012': int(round(n(BASE['Clarity2012'], 0, 3, 14))),
        'Vibrance': int(round(n(BASE['Vibrance'], 0, 6, 16))),
        'Saturation': int(round(BASE['Saturation'] * s)),
        'Dehaze': int(round(n(BASE['Dehaze'], 0, 2, 8))),
    }


def absolute_hsl(char, strength=1.0):
    """Close to vivo_grok HSL."""
    s = strength
    hsl = {
        'HueAdjustmentOrange': int(np.clip(-3 * s, -8, 8)),
        'HueAdjustmentYellow': int(np.clip(-4 * s, -8, 8)),
        'HueAdjustmentGreen': int(np.clip(2 * s, -6, 6)),
        'HueAdjustmentAqua': int(np.clip(3 * s, -6, 6)),
        'SaturationAdjustmentOrange': int(np.clip(8 * s, -12, 12)),
        'SaturationAdjustmentYellow': int(np.clip(5 * s, -10, 12)),
        'SaturationAdjustmentGreen': int(np.clip(8 * s, -12, 12)),
        'SaturationAdjustmentAqua': int(np.clip(4 * s, -10, 10)),
        'SaturationAdjustmentBlue': int(np.clip(3 * s, -8, 8)),
        'SaturationAdjustmentRed': int(np.clip(4 * s, -8, 10)),
        'LuminanceAdjustmentOrange': int(np.clip(5 * s, -10, 10)),
        'LuminanceAdjustmentYellow': int(np.clip(3 * s, -8, 8)),
        'LuminanceAdjustmentGreen': int(np.clip(5 * s, -10, 10)),
    }
    return {k: int(np.clip(v, -HSL_LIMIT, HSL_LIMIT)) for k, v in hsl.items()}


def absolute_curves(char, strength=1.0):
    """Single luminance curve with shadow lift — matches vivo_grok style.
    Returns list of (x,y) points (not per-channel RGB).
    """
    s = strength
    # vivo_grok: (0,12), (40,38), (95,95), (170,172), (230,232), (255,255)
    y0 = int(np.clip(12 * s, 6, 18))
    pts = [
        (0, y0),
        (40, int(38 * s + (1 - s) * 40)),
        (95, 95),
        (170, 172),
        (230, 232),
        (255, 255),
    ]
    # monotonic
    ys = [p[1] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 1)
    return [(p[0], min(255, ys[i])) for i, p in enumerate(pts)]


def write_absolute_preset(name, imgs, strength, out_dir):
    char = analyse_character(imgs)
    basic = absolute_basic_sliders(char, strength)
    hsl = absolute_hsl(char, strength)
    curves = absolute_curves(char, strength)  # single luminance curve
    desc = f'Absolute vivo_grok-biased look from {len(imgs)} photos'
    path = os.path.join(out_dir, name.replace(' ', '_') + '.xmp')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xmp(name, curves, hsl, basic, desc, group='Absolute Looks'))
    print('wrote', path)
    print('  Basic:', {k: v for k, v in basic.items() if v != 0 and v != 0.0})
    print('  HSL:', {k: v for k, v in hsl.items() if v})
    print(f'  Character: n={char["n"]}, mid_bright={char["mid_bright"]:+.1f}, warmth={char["warmth"]:+.1f}')
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
        out[f'SaturationAdjustment{b}'] = (s_sat / max(r_sat, 1e-6) - 1) * 30
        out[f'HueAdjustment{b}'] = (s_hue - r_hue) * 1.0
        out[f'LuminanceAdjustment{b}'] = (s_lum - r_lum) * 50
    for k, v in [('SaturationAdjustmentGreen', 5), ('SaturationAdjustmentOrange', 4)]:
        out[k] = out.get(k, 0) + v * strength
    return {k: int(np.clip(round(v * strength), -HSL_LIMIT, HSL_LIMIT)) for k, v in out.items()}


def write_relative_preset(name, style_q, ref_q, sample_px, strength, out_dir, group, desc):
    curves = channel_curves(style_q, ref_q, strength)
    basic = estimate_basic_sliders(style_q, ref_q, strength)
    hsl = compute_hsl_relative(sample_px, curves, strength)
    path = os.path.join(out_dir, name.replace(' ', '_') + '.xmp')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xmp(name, curves, hsl, basic, desc, group=group))
    print('wrote', path)
    print('  Basic:', {k: v for k, v in basic.items() if v != 0 and v != 0.0})
    print('  HSL:', {k: v for k, v in hsl.items() if v})
    return path


def run_relative(pix, feats, strength, out, groups, min_photos, by_light):
    def profile(d, names, label):
        print(f'  {label}: {len(names)} photos')
        return np.nanmedian(robust_keep(np.array([feats[d][k] for k in names]), names, label), 0)

    jobs = []
    prof_all = {d: profile(d, list(feats[d]), f'{d} / all') for d in feats}
    base_all = np.nanmean(list(prof_all.values()), 0)
    for d, p in prof_all.items():
        sample = list(pix[d].values())[::max(1, len(pix[d]) // 8)]
        jobs.append((pretty(d), p, base_all, sample, 'Folder Looks',
                     'Relative vs average of folders'))

    if by_light:
        def mid(f):
            return f[:33].reshape(11, 3)[5].mean()
        split = np.median([mid(f) for fs in feats.values() for f in fs.values()])
        light = {d: {k: LIGHTS[int(mid(f) < split)] for k, f in fs.items()} for d, fs in feats.items()}
        print(f'  (optional) Bright/Low split at {split:.0f}/255')
        for lt in LIGHTS:
            names = {d: [k for k in feats[d] if light[d][k] == lt] for d in feats}
            prof = {d: profile(d, n, f'{d} / {lt}') for d, n in names.items() if len(n) >= min_photos}
            if len(prof) < 2:
                continue
            base = np.nanmean(list(prof.values()), 0)
            for d, p in prof.items():
                sample = [pix[d][k] for k in names[d]][::2]
                jobs.append((f'{pretty(d)} - {lt}', p, base, sample, 'Folder Looks',
                             f'Relative {lt}'))

    if groups:
        grouped = {}
        for d in feats:
            grouped.setdefault(group_of(d), []).append(d)
        grouped = {g: ds for g, ds in grouped.items() if len(ds) > 1}
        for g, ds in grouped.items():
            sample = [px for d in ds for px in list(pix[d].values())[::4]]
            target = np.mean([prof_all[d] for d in ds], 0)
            jobs.append((f'{pretty(g)} Look', target, base_all, sample, 'Folder Looks',
                         'Group relative'))

    print()
    for name, t, b, sample, group, desc in jobs:
        write_relative_preset(name, t[:33].reshape(11, 3), b[:33].reshape(11, 3),
                              sample, strength, out, group, desc)


def main(src, out, exclude=(), groups=True, min_photos=3, strength=1.0,
         absolute=False, by_light=False):
    os.makedirs(out, exist_ok=True)
    pix = scan(src, set(exclude), min_photos)
    print('Folders:', ', '.join(f'{d} ({len(v)})' for d, v in pix.items()))

    if absolute:
        print('Mode: ABSOLUTE (vivo_grok-biased, strong base)')
        for d, imgs in pix.items():
            kept = list(imgs.values())
            if len(kept) < min_photos:
                print(f'  {d}: too few photos, skip')
                continue
            write_absolute_preset(pretty(d), kept, strength, out)
        return

    if len(pix) < 2:
        sys.exit('Relative mode needs at least 2 folders. Use --absolute for single-folder sets.')

    print('Mode: RELATIVE' + (' + by-light' if by_light else ''))
    feats = {d: {k: features(v) for k, v in imgs.items()} for d, imgs in pix.items()}
    run_relative(pix, feats, strength, out, groups, min_photos, by_light)


def selftest():
    assert pretty('vivo_300_pro') == 'Vivo 300 Pro'
    rng = np.random.default_rng(0)
    # mix dark + bright
    fake = [rng.uniform(10, 80, (3000, 3)) for _ in range(2)]
    fake += [rng.uniform(40, 220, (3000, 3)) for _ in range(2)]
    char = analyse_character(fake)
    basic = absolute_basic_sliders(char, 1.0)
    assert basic['Shadows2012'] >= 20
    assert basic['Highlights2012'] <= -15
    assert basic['Exposure2012'] > -0.2  # must not go strongly negative
    curves = absolute_curves(char, 1.0)
    assert curves[0] == (0, 12) or curves[0][1] >= 6
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
                    help='One character-based preset per folder (recommended)')
    ap.add_argument('--by-light', action='store_true',
                    help='Also generate Bright/Low Light relative variants')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    main(a.samples_dir, a.out_dir, a.exclude, not a.no_groups,
         a.min_photos, a.strength, a.absolute, a.by_light)
