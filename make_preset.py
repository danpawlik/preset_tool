#!/usr/bin/env python3
"""Build a Lightroom .xmp preset that makes REFERENCE photos look like STYLE photos.

Tuned toward hand-crafted phone looks (vivo_grok style).

Usage:
  python3 make_preset.py STYLE_DIR REFERENCE_DIR -n "My Look" [-o out.xmp] [-s STRENGTH] [--preview]
"""
import argparse, glob, os, sys, uuid
import numpy as np
from PIL import Image, ImageOps

Q = [0.5, 2, 8, 20, 35, 50, 65, 80, 92, 98, 99.5]
BANDS = dict(Red=0, Orange=30, Yellow=60, Green=120, Aqua=180, Blue=225, Purple=270, Magenta=315)
HSL_LIMIT = 12

# Photographic base — closely matches hand-crafted Vivo X500 ZEISS look
BASE = {
    'Exposure2012': 0.10,
    'Contrast2012': 12,
    'Highlights2012': -30,
    'Shadows2012': 35,
    'Whites2012': 8,
    'Blacks2012': -8,
    'Texture': 6,
    'Clarity2012': 6,
    'Vibrance': 10,
    'Saturation': 1,
    'Dehaze': 4,
}


def crop_frame(a):
    """Strip solid white/black border bars (marketing frames, letterbox)."""
    def solid(line):
        lum = line.mean(-1)
        return (lum > 220).mean() > 0.88 or (lum < 12).mean() > 0.97
    t, b, l, r = 0, a.shape[0], 0, a.shape[1]
    while b - t > a.shape[0] // 2 and solid(a[b - 1]):
        b -= 1
    while b - t > a.shape[0] // 2 and solid(a[t]):
        t += 1
    while r - l > a.shape[1] // 2 and solid(a[t:b, r - 1]):
        r -= 1
    while r - l > a.shape[1] // 2 and solid(a[t:b, l]):
        l += 1
    a = a[t:b, l:r]
    # extra safety crop ~2%
    mh, mw = max(1, a.shape[0] // 50), max(1, a.shape[1] // 50)
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

    d_shadows = diff(2) * 1.4
    d_highlights = -diff(8) * 1.2
    d_whites = diff(9) * 0.8
    d_blacks = diff(1) * 1.0
    mid_stretch = (diff(7) - diff(3))
    d_contrast = mid_stretch * 0.7
    d_exposure = diff(5) / 80.0
    d_clarity = mid_stretch * 0.18
    d_texture = mid_stretch * 0.14

    # Heavy base bias (70% base / 30% measured) — keeps results near hand-tuned look
    def blend(base, delta, lo, hi):
        return float(np.clip(base * 0.70 + delta * 0.30, lo, hi) * strength)

    return {
        'Exposure2012': round(blend(BASE['Exposure2012'], d_exposure, -0.25, 0.35), 2),
        'Contrast2012': int(round(blend(BASE['Contrast2012'], d_contrast, 4, 28))),
        'Highlights2012': int(round(blend(BASE['Highlights2012'], d_highlights, -50, -8))),
        'Shadows2012': int(round(blend(BASE['Shadows2012'], d_shadows, 12, 50))),
        'Whites2012': int(round(blend(BASE['Whites2012'], d_whites, -20, 22))),
        'Blacks2012': int(round(blend(BASE['Blacks2012'], d_blacks, -22, 12))),
        'Texture': int(round(blend(BASE['Texture'], d_texture, 0, 16))),
        'Clarity2012': int(round(blend(BASE['Clarity2012'], d_clarity, 0, 18))),
        'Vibrance': int(round(BASE['Vibrance'] * strength)),
        'Saturation': int(round(BASE['Saturation'] * strength)),
        'Dehaze': int(round(BASE['Dehaze'] * strength)),
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
        out[f'SaturationAdjustment{b}'] = (s_sat / max(r_sat, 1e-6) - 1) * 30
        out[f'HueAdjustment{b}'] = (s_hue - r_hue) * 1.0
        out[f'LuminanceAdjustment{b}'] = (s_lum - r_lum) * 50

    # vivo_grok-like mild biases
    biases = {
        'SaturationAdjustmentOrange': 6,
        'SaturationAdjustmentYellow': 4,
        'SaturationAdjustmentGreen': 6,
        'SaturationAdjustmentAqua': 3,
        'SaturationAdjustmentBlue': 2,
        'SaturationAdjustmentRed': 3,
        'HueAdjustmentOrange': -2,
        'HueAdjustmentYellow': -3,
        'HueAdjustmentGreen': 1,
        'HueAdjustmentAqua': 2,
        'LuminanceAdjustmentOrange': 4,
        'LuminanceAdjustmentYellow': 2,
        'LuminanceAdjustmentGreen': 4,
    }
    for k, v in biases.items():
        out[k] = out.get(k, 0) + v * strength

    return {k: int(np.clip(round(v * strength if k not in biases else v), -HSL_LIMIT, HSL_LIMIT))
            for k, v in out.items()}


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


def xmp(name, curves, hsl, basic, desc, group='Generated Looks', extra_attrs=None):
    """curves: either list of 3 RGB curves, or a single luminance curve (list of points)."""
    uid = uuid.uuid5(uuid.NAMESPACE_URL, name).hex.upper()
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
    # Detail + effects (close to vivo_grok); basic/hsl/extra_attrs below override these
    attrs += [
        ('Sharpness', '32'),
        ('SharpenRadius', '0.9'),
        ('SharpenDetail', '18'),
        ('SharpenEdgeMasking', '15'),
        ('LuminanceSmoothing', '8'),
        ('ColorNoiseReduction', '22'),
        ('SplitToningShadowHue', '210'),
        ('SplitToningShadowSaturation', '3'),
        ('SplitToningHighlightHue', '35'),
        ('SplitToningHighlightSaturation', '4'),
        ('SplitToningBalance', '-5'),
        ('ColorGradeMidtoneHue', '20'),
        ('ColorGradeMidtoneSat', '3'),
        ('VignetteAmount', '-6'),
        ('PostCropVignetteAmount', '-8'),
        ('PostCropVignetteMidpoint', '50'),
        ('PostCropVignetteFeather', '55'),
        ('GrainAmount', '3'),
        ('GrainSize', '25'),
    ]
    attrs += [(k, fmt_val(v)) for k, v in basic.items()]
    attrs += [(k, fmt_val(v)) for k, v in sorted(hsl.items()) if v != 0]
    if extra_attrs:
        attrs += list(extra_attrs.items() if isinstance(extra_attrs, dict) else extra_attrs)
    attrs = dict(attrs).items()  # one value per attribute, later entries win (duplicates = invalid XMP)

    attr_lines = '\n'.join(f'    crs:{k}="{v}"' for k, v in attrs)

    # Support single luminance curve or per-channel RGB
    if curves and isinstance(curves[0], tuple):
        # single curve: list of (x,y)
        lum = curve_xml(curves)
        rgb_block = f'''   <crs:ToneCurvePV2012>
{lum}
   </crs:ToneCurvePV2012>'''
    else:
        r, g, b = curves
        rgb_block = f'''   <crs:ToneCurvePV2012>
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
   </crs:ToneCurvePV2012Blue>'''

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
{attr_lines}>
{rgb_block}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
'''


def preview(ref_dir, curves, out):
    f = sorted(glob.glob(os.path.join(ref_dir, '*.jp*g')))[0]
    im = ImageOps.exif_transpose(Image.open(f)).convert('RGB')
    im.thumbnail((900, 900))
    a = np.asarray(im, float)
    if curves and isinstance(curves[0], tuple):
        # luminance curve approx: apply to mean
        pass
    else:
        b = apply_curves(a.reshape(-1, 3), curves).reshape(a.shape)
        Image.fromarray(np.hstack([a, b]).astype(np.uint8)).save(out)
        print('preview:', out)


def selftest():
    rng = np.random.default_rng(0)
    ref = [rng.uniform(0, 255, (5000, 3)) for _ in range(4)]
    style = [np.clip(im * [1.1, 1.0, 0.9], 0, 255) for im in ref]
    r, g, b = channel_curves(mean_quantiles(style), mean_quantiles(ref), 1.0)
    basic = estimate_basic_sliders(mean_quantiles(style), mean_quantiles(ref), 1.0)
    assert basic['Shadows2012'] > 15
    assert basic['Highlights2012'] < 0
    xml = xmp('Selftest', (r, g, b), {}, basic, 'selftest')
    assert 'crs:Shadows2012' in xml and 'Sharpness' in xml
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
