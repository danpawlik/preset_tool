#!/usr/bin/env python3
"""Build a Lightroom .xmp preset that makes REFERENCE photos look like STYLE photos.

Improved version with:
- Basic tonal sliders
- Soft HSL (limit ±12)
- Clean XMP format compatible with Lightroom Classic / ACR

Usage:
  python3 make_preset.py STYLE_DIR REFERENCE_DIR -n "My Look" [-o out.xmp] [-s STRENGTH] [--preview]
"""
import argparse, glob, os, sys, uuid
import numpy as np
from PIL import Image, ImageOps

Q = [0.5, 2, 8, 20, 35, 50, 65, 80, 92, 98, 99.5]
BANDS = dict(Red=0, Orange=30, Yellow=60, Green=120, Aqua=180, Blue=225, Purple=270, Magenta=315)
HSL_LIMIT = 12


def crop_frame(a):
    def solid(line):
        lum = line.mean(-1)
        return (lum > 225).mean() > 0.9 or (lum < 12).mean() > 0.97
    t, b, l, r = 0, a.shape[0], 0, a.shape[1]
    while b - t > a.shape[0] // 2 and solid(a[b - 1]): b -= 1
    while b - t > a.shape[0] // 2 and solid(a[t]): t += 1
    while r - l > a.shape[1] // 2 and solid(a[t:b, r - 1]): r -= 1
    while r - l > a.shape[1] // 2 and solid(a[t:b, l]): l += 1
    a = a[t:b, l:r]
    mh, mw = a.shape[0] // 50, a.shape[1] // 50
    return a[mh:a.shape[0] - mh, mw:a.shape[1] - mw]


def load(path, size=600):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    im.thumbnail((size, size))
    return crop_frame(np.asarray(im, float)).reshape(-1, 3)


def list_images(d):
    return sorted(f for f in glob.glob(os.path.join(d, '*'))
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')))


def load_dir(d):
    files = list_images(d)
    imgs = []
    for f in files:
        try:
            imgs.append(load(f))
        except Exception as e:
            print(f'skip {f}: {e}', file=sys.stderr)
    if len(imgs) < 3:
        sys.exit(f'{d}: need at least 3 readable images, got {len(imgs)}')
    return imgs


def mean_quantiles(imgs):
    return np.mean([np.percentile(im, Q, axis=0) for im in imgs], axis=0)


def channel_curves(style_q, ref_q, strength):
    curves = []
    for c in range(3):
        x, y = ref_q[:, c], style_q[:, c]
        y = x + strength * (y - x)
        x = np.concatenate([[0], x, [255]])
        y = np.concatenate([[max(0, y[0] - x[1])], y, [min(255, y[-1] + 255 - x[-2])]])
        y = np.maximum.accumulate(np.clip(y, 0, 255))
        keep = np.concatenate([[True], np.diff(np.round(x)) >= 4])
        curves.append([(int(round(a)), int(round(b))) for a, b in zip(x[keep], y[keep])])
    return curves


def apply_curves(px, curves):
    return np.stack([np.interp(px[:, c], [p[0] for p in curves[c]], [p[1] for p in curves[c]])
                     for c in range(3)], axis=1)


def rgb_to_hsv(px):
    r, g, b = px.T / 255
    mx, mn = px.max(1) / 255, px.min(1) / 255
    d = np.where(mx > mn, mx - mn, 1)
    h = np.where(mx == r, (g - b) / d % 6,
                 np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0), mx


def band_stats(px):
    h, s, v = rgb_to_hsv(px)
    ok = (s > 0.15) & (v > 0.15) & (v < 0.97)
    out = {}
    for name, centre in BANDS.items():
        d = (h - centre + 180) % 360 - 180
        m = ok & (np.abs(d) < 22)
        if m.sum() > 200:
            out[name] = (s[m].mean(), d[m].mean(), v[m].mean())
    return out


def estimate_basic_sliders(style_q, ref_q, strength):
    s_lum = style_q.mean(axis=1)
    r_lum = ref_q.mean(axis=1)

    def diff(i):
        return float(s_lum[i] - r_lum[i])

    shadows    = np.clip(diff(2) * 1.8 * strength, -55, 55)
    highlights = np.clip(-diff(8) * 1.5 * strength, -55, 55)
    whites     = np.clip(diff(9) * 1.1 * strength, -35, 35)
    blacks     = np.clip(diff(1) * 1.3 * strength, -35, 35)
    mid_stretch = (diff(7) - diff(3))
    contrast   = np.clip(mid_stretch * 1.0 * strength, -35, 35)
    exposure   = np.clip(diff(5) / 65.0 * strength, -0.7, 0.7)
    clarity    = np.clip(mid_stretch * 0.30 * strength, -18, 22)
    texture    = np.clip(mid_stretch * 0.22 * strength, -12, 18)
    vibrance   = np.clip(6 * strength, 0, 15)

    return {
        'Exposure2012': round(float(exposure), 2),
        'Contrast2012': int(round(contrast)),
        'Highlights2012': int(round(highlights)),
        'Shadows2012': int(round(shadows)),
        'Whites2012': int(round(whites)),
        'Blacks2012': int(round(blacks)),
        'Texture': int(round(texture)),
        'Clarity2012': int(round(clarity)),
        'Vibrance': int(round(vibrance)),
        'Saturation': 0,
        'Dehaze': 0,
    }


def hsl_sliders(style_imgs, ref_imgs, curves, strength):
    st = [band_stats(im) for im in style_imgs]
    rf = [band_stats(apply_curves(im, curves)) for im in ref_imgs]
    out = {}
    for b in BANDS:
        s = [x[b] for x in st if b in x]
        r = [x[b] for x in rf if b in x]
        if len(s) < 3 or len(r) < 3:
            continue
        (s_sat, s_hue, s_lum), (r_sat, r_hue, r_lum) = np.mean(s, 0), np.mean(r, 0)
        sat = (s_sat / max(r_sat, 1e-6) - 1) * 40
        hue = (s_hue - r_hue) * 1.2
        lum = (s_lum - r_lum) * 70
        out[f'SaturationAdjustment{b}'] = sat
        out[f'HueAdjustment{b}'] = hue
        out[f'LuminanceAdjustment{b}'] = lum
    return {k: int(np.clip(round(v * strength), -HSL_LIMIT, HSL_LIMIT)) for k, v in out.items()}


def fmt_val(v):
    if isinstance(v, float):
        return f"{v:+.2f}"
    return f"{int(v):+d}"


def curve_xml(pts):
    lines = ['    <rdf:Seq>']
    for x, y in pts:
        lines.append(f'     <rdf:li>{x}, {y}</rdf:li>')
    lines.append('    </rdf:Seq>')
    return '\n'.join(lines)


def xmp(name, curves, hsl, basic, desc, group='Generated Looks'):
    uid = uuid.uuid5(uuid.NAMESPACE_URL, name).hex.upper()

    # Build attribute block in the style Lightroom accepts reliably
    attrs = [
        ('PresetType', 'Normal'),
        ('UUID', uid),
        ('SupportsAmount2', 'True'),
        ('SupportsAmount', 'True'),
        ('SupportsColor', 'True'),
        ('SupportsMonochrome', 'True'),
        ('SupportsHighDynamicRange', 'True'),
        ('SupportsNormalDynamicRange', 'True'),
        ('SupportsSceneReferred', 'True'),
        ('SupportsOutputReferred', 'True'),
        ('Version', '15.0'),
        ('ProcessVersion', '15.0'),
        ('WhiteBalance', 'As Shot'),
        ('ToneCurveName2012', 'Custom'),
        ('HasSettings', 'True'),
        ('Copyright', desc[:80]),
    ]

    for k, v in basic.items():
        attrs.append((k, fmt_val(v)))

    for k, v in sorted(hsl.items()):
        if v != 0:
            attrs.append((k, fmt_val(v)))

    attr_lines = '\n'.join(f'    crs:{k}="{v}"' for k, v in attrs)

    r, g, b = curves
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
{attr_lines}>
   <crs:ToneCurvePV2012>
{curve_xml([(0, 0), (255, 255)])}
   </crs:ToneCurvePV2012>
   <crs:ToneCurvePV2012Red>
{curve_xml(r)}
   </crs:ToneCurvePV2012Red>
   <crs:ToneCurvePV2012Green>
{curve_xml(g)}
   </crs:ToneCurvePV2012Green>
   <crs:ToneCurvePV2012Blue>
{curve_xml(b)}
   </crs:ToneCurvePV2012Blue>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
'''


def preview(ref_dir, curves, out):
    f = sorted(glob.glob(os.path.join(ref_dir, '*.jp*g')))[0]
    im = ImageOps.exif_transpose(Image.open(f)).convert('RGB')
    im.thumbnail((900, 900))
    a = np.asarray(im, float)
    b = apply_curves(a.reshape(-1, 3), curves).reshape(a.shape)
    Image.fromarray(np.hstack([a, b]).astype(np.uint8)).save(out)
    print('preview:', out)


def selftest():
    rng = np.random.default_rng(0)
    ref = [rng.uniform(0, 255, (5000, 3)) for _ in range(4)]
    style = [np.clip(im * [1.1, 1.0, 0.9], 0, 255) for im in ref]
    r, g, b = channel_curves(mean_quantiles(style), mean_quantiles(ref), 1.0)
    mid = lambda c: np.interp(128, [p[0] for p in c], [p[1] for p in c])
    assert abs(mid(r) - 141) < 4 and abs(mid(g) - 128) < 4 and abs(mid(b) - 115) < 4
    assert all(np.all(np.diff([y for _, y in c]) >= 0) for c in (r, g, b))
    # also test xmp generation doesn't crash
    basic = estimate_basic_sliders(mean_quantiles(style), mean_quantiles(ref), 1.0)
    xml = xmp('Selftest', (r, g, b), {}, basic, 'selftest')
    assert 'crs:Exposure2012' in xml and 'ToneCurvePV2012Red' in xml
    print('selftest ok')


def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('style_dir')
    ap.add_argument('reference_dir')
    ap.add_argument('-n', '--name', default='Generated Look')
    ap.add_argument('-o', '--out')
    ap.add_argument('-s', '--strength', type=float, default=1.0)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    style, ref = load_dir(a.style_dir), load_dir(a.reference_dir)
    style_q, ref_q = mean_quantiles(style), mean_quantiles(ref)

    curves = channel_curves(style_q, ref_q, a.strength)
    basic = estimate_basic_sliders(style_q, ref_q, a.strength)
    hsl = hsl_sliders(style, ref, curves, a.strength)

    out = a.out or a.name.replace(' ', '_') + '.xmp'
    desc = f'From {len(style)} style / {len(ref)} ref images'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(xmp(a.name, curves, hsl, basic, desc))
    print('wrote', out)
    print('  Basic:', {k: v for k, v in basic.items() if v != 0 and v != 0.0})
    print('  HSL:', {k: v for k, v in hsl.items() if v})
    if a.preview:
        preview(a.reference_dir, curves, os.path.splitext(out)[0] + '_preview.jpg')


if __name__ == '__main__':
    main()
