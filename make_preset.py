#!/usr/bin/env python3
"""Build a Lightroom .xmp preset that makes REFERENCE photos look like STYLE photos.

  make_preset.py STYLE_DIR REFERENCE_DIR -n "My Look" [-o out.xmp] [--preview]

STYLE_DIR:     photos with the look you want (e.g. vivo X300 shots).
REFERENCE_DIR: photos from the camera you'll apply the preset to. Same scenes
               as STYLE_DIR give the best result; similar scenes work too.

How: per-channel quantile matching -> R/G/B tone curves, then the remaining
saturation/hue gap per Lightroom HSL band is measured after those curves.

Examples:
  # Make your Pixel photos look like vivo X300 Pro shots
  make_preset.py ~/photos/vivo_x300 ~/photos/my_pixel -n "Vivo X300 Look"

  # Same, softer (70%), custom output file, plus a before/after JPG
  make_preset.py ~/photos/vivo_x300 ~/photos/my_pixel -n "Vivo X300 Soft" \\
      -s 0.7 -o ~/presets/vivo_x300_soft.xmp --preview

  # Turn X500 Pro shots into the X300 Pro look (best: same scenes in both dirs)
  make_preset.py vivo_300_pro vivo_500_pro -n "X500 to X300"

  # Check the maths still works
  make_preset.py --selftest

Import the .xmp in Lightroom: Presets panel -> "+" -> Import Presets.
"""
import argparse, glob, os, sys, uuid
import numpy as np
from PIL import Image, ImageOps

Q = [0.5, 2, 8, 20, 35, 50, 65, 80, 92, 98, 99.5]
# Lightroom HSL bands, centre hue in degrees (HSV hue)
BANDS = dict(Red=0, Orange=30, Yellow=60, Green=120, Aqua=180, Blue=225, Purple=270, Magenta=315)


def crop_frame(a):
    # strip white/black border bars (e.g. Leica-style watermark frames), then a 2% safety margin
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


def load_dir(d):
    files = sorted(f for f in glob.glob(os.path.join(d, '*'))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.heic')))
    imgs = []
    for f in files:
        try:
            imgs.append(load(f))
        except Exception as e:  # skip unreadable files, keep going
            print(f'skip {f}: {e}', file=sys.stderr)
    if len(imgs) < 3:
        sys.exit(f'{d}: need at least 3 readable images, got {len(imgs)}')
    return imgs


def mean_quantiles(imgs):
    # average per-image quantiles so one huge/odd scene doesn't dominate
    return np.mean([np.percentile(im, Q, axis=0) for im in imgs], axis=0)  # (len(Q), 3)


def channel_curves(style_q, ref_q, strength):
    curves = []
    for c in range(3):
        x, y = ref_q[:, c], style_q[:, c]
        y = x + strength * (y - x)
        x = np.concatenate([[0], x, [255]])
        y = np.concatenate([[max(0, y[0] - x[1])], y, [min(255, y[-1] + 255 - x[-2])]])
        y = np.maximum.accumulate(np.clip(y, 0, 255))  # monotonic, in range
        keep = np.concatenate([[True], np.diff(np.round(x)) >= 4])  # LR wants distinct x
        curves.append([(int(round(a)), int(round(b))) for a, b in zip(x[keep], y[keep])])
    return curves


def apply_curves(px, curves):
    return np.stack([np.interp(px[:, c], [p[0] for p in curves[c]], [p[1] for p in curves[c]]) for c in range(3)], axis=1)


def rgb_to_hsv(px):
    r, g, b = px.T / 255
    mx, mn = px.max(1) / 255, px.min(1) / 255
    d = np.where(mx > mn, mx - mn, 1)
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
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


def hsl_sliders(style_imgs, ref_imgs, curves, strength):
    st = [band_stats(im) for im in style_imgs]
    rf = [band_stats(apply_curves(im, curves)) for im in ref_imgs]
    out = {}
    for b in BANDS:
        s = [x[b] for x in st if b in x]
        r = [x[b] for x in rf if b in x]
        if len(s) < 3 or len(r) < 3:
            continue  # colour too rare in the samples to judge
        (s_sat, s_hue, s_lum), (r_sat, r_hue, r_lum) = np.mean(s, 0), np.mean(r, 0)
        # ponytail: linear slider guesses (sat ratio*100, ~0.3deg/hue step, lum*200); calibrate against LR if off
        out[f'SaturationAdjustment{b}'] = (s_sat / r_sat - 1) * 100
        out[f'HueAdjustment{b}'] = (s_hue - r_hue) * 3
        out[f'LuminanceAdjustment{b}'] = (s_lum - r_lum) * 200
    return {k: int(np.clip(round(v * strength), -40, 40)) for k, v in out.items()}


def xmp(name, curves, hsl, desc):
    def alt(v): return f'<rdf:Alt><rdf:li xml:lang="x-default">{v}</rdf:li></rdf:Alt>'
    def seq(pts): return '<rdf:Seq>' + ''.join(f'<rdf:li>{x}, {y}</rdf:li>' for x, y in pts) + '</rdf:Seq>'
    attrs = dict(PresetType='Normal', UUID=uuid.uuid5(uuid.NAMESPACE_URL, name).hex.upper(),
                 SupportsAmount='True', SupportsColor='True', SupportsMonochrome='False',
                 SupportsHighDynamicRange='True', SupportsNormalDynamicRange='True',
                 SupportsSceneReferred='True', SupportsOutputReferred='True',
                 Version='15.0', ProcessVersion='11.0', ToneCurveName2012='Custom', HasSettings='True',
                 **{k: f'{v:+d}' for k, v in hsl.items()})
    a = '\n   '.join(f'crs:{k}="{v}"' for k, v in attrs.items())
    r, g, b = curves
    return f'''<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   {a}>
   <crs:Name>{alt(name)}</crs:Name>
   <crs:Group>{alt('Generated Looks')}</crs:Group>
   <crs:Description>{alt(desc)}</crs:Description>
   <crs:ToneCurvePV2012>{seq([(0, 0), (255, 255)])}</crs:ToneCurvePV2012>
   <crs:ToneCurvePV2012Red>{seq(r)}</crs:ToneCurvePV2012Red>
   <crs:ToneCurvePV2012Green>{seq(g)}</crs:ToneCurvePV2012Green>
   <crs:ToneCurvePV2012Blue>{seq(b)}</crs:ToneCurvePV2012Blue>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
'''


def preview(ref_dir, curves, out):
    # curves only (HSL not simulated) - quick sanity look, not an LR render
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
    style = [np.clip(im * [1.1, 1.0, 0.9], 0, 255) for im in ref]  # warmer
    r, g, b = channel_curves(mean_quantiles(style), mean_quantiles(ref), 1.0)
    mid = lambda c: np.interp(128, [p[0] for p in c], [p[1] for p in c])
    assert abs(mid(r) - 141) < 4 and abs(mid(g) - 128) < 4 and abs(mid(b) - 115) < 4, (mid(r), mid(g), mid(b))
    assert all(np.all(np.diff([y for _, y in c]) >= 0) for c in (r, g, b))
    print('selftest ok')


def main():
    if '--selftest' in sys.argv[1:]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('style_dir', help='photos with the look you want')
    ap.add_argument('reference_dir', help='photos from the camera the preset will be applied to')
    ap.add_argument('-n', '--name', default='Generated Look')
    ap.add_argument('-o', '--out')
    ap.add_argument('-s', '--strength', type=float, default=1.0, help='0..1.5, scales every adjustment')
    ap.add_argument('--preview', action='store_true', help='save before/after JPG (curves only)')
    ap.add_argument('--selftest', action='store_true', help='run the built-in check and exit')
    a = ap.parse_args()

    style, ref = load_dir(a.style_dir), load_dir(a.reference_dir)
    curves = channel_curves(mean_quantiles(style), mean_quantiles(ref), a.strength)
    hsl = hsl_sliders(style, ref, curves, a.strength)
    out = a.out or a.name.replace(' ', '_') + '.xmp'
    desc = f'Generated from {len(style)} style / {len(ref)} reference images.'
    with open(out, 'w') as f:
        f.write(xmp(a.name, curves, hsl, desc))
    print('wrote', out)
    for n, c in zip('RGB', curves):
        print(f'  {n} curve: {c}')
    print('  HSL:', {k: v for k, v in hsl.items() if v})
    if a.preview:
        preview(a.reference_dir, curves, os.path.splitext(out)[0] + '_preview.jpg')


if __name__ == '__main__':
    main()
