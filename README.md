# preset_tool

Generate Lightroom presets (`.xmp`) from pictures you provide.

Two scripts:

- `make_preset.py` - make **your** photos look like a set of style photos.
- `phone_presets.py` - derive one "look" preset per phone brand from sample folders.

Presets work in Lightroom Classic, Lightroom (cloud) and Adobe Camera Raw, for raw files and JPEGs.

## Requirements

Python 3, `numpy`, `Pillow`:

```bash
pip install --user numpy pillow
```

## make_preset.py

```bash
python3 make_preset.py STYLE_DIR REFERENCE_DIR -n "My Look" [-o out.xmp] [-s STRENGTH] [--preview]
```

| Argument | Meaning |
|---|---|
| `STYLE_DIR` | photos with the look you want (min. 3) |
| `REFERENCE_DIR` | photos from the camera you'll apply the preset to (min. 3) |
| `-n, --name` | preset name shown in Lightroom |
| `-o, --out` | output file (default: `<name>.xmp`) |
| `-s, --strength` | scale all adjustments, e.g. `0.7` softer, `1.3` stronger (default `1.0`) |
| `--preview` | save a before/after JPG next to the preset (tone curves only) |
| `--selftest` | run the built-in check and exit |

Examples:

```bash
# Make your Pixel photos look like vivo X300 Pro shots
python3 make_preset.py ~/photos/vivo_x300 ~/photos/my_pixel -n "Vivo X300 Look"

# Softer (70%), custom output file, plus a before/after JPG
python3 make_preset.py ~/photos/vivo_x300 ~/photos/my_pixel -n "Vivo X300 Soft" \
    -s 0.7 -o ~/presets/vivo_x300_soft.xmp --preview

# Check the maths still works
python3 make_preset.py --selftest
```

**Tip:** the closer the scenes in both folders, the better. Same scenes are ideal,
similar light and subjects are fine. Very different scenes (e.g. flowers vs. night
city) get mistaken for "style".

How it works: per-channel quantile matching gives R/G/B tone curves; the remaining
saturation / hue / luminance gap per Lightroom HSL band is measured after those curves
and written as HSL sliders. White/black watermark frames (e.g. Leica borders) are
cropped before measuring.

## phone_presets.py

```bash
python3 phone_presets.py SAMPLES_DIR OUT_DIR
```

Expects `SAMPLES_DIR` to contain `vivo_300_pro/`, `vivo_500_pro/`, `xiaomi_17/`,
`xiaomi_18/`. Writes 10 presets:

- 2 per phone: `<Phone>_-_Bright_Light.xmp` and `<Phone>_-_Low_Light.xmp`
- 1 per brand: `Vivo_Look.xmp`, `Xiaomi_Look.xmp`

Photos are split into bright and dim scenes by median luminance (split at the
median of all photos, ~80/255). Each phone preset = that phone's median render in
that light vs. the average of all four phones in the same light, so bright scenes
are compared with bright scenes. Brand presets use all photos. Photos far from their
phone's typical render are ignored, and known wrong-phone files are excluded
(`EXCLUDE` in the script).

Current result from the sample set (vs. the average phone, 0-255 levels):

| Preset | Photos | Mids | Highlights | Warmth | Notes |
|---|---|---|---|---|---|
| Vivo X300 Pro - Bright Light | 20 | -8 | +9 | +6 | yellow/green +5-7% |
| Vivo X300 Pro - Low Light | 6 | +5 | +4 | +7 | yellow +10%; few photos, low confidence |
| Vivo X500 Pro - Bright Light | 19 | +1 | +10 | +8 | aqua +4% |
| Vivo X500 Pro - Low Light | 11 | +10 | +10 | +1 | brighter, neutral |
| Xiaomi 17 - Bright Light | 21 | +7 | -5 | -7 | reds/blues +5%, greens -7% |
| Xiaomi 17 - Low Light | 37 | -11 | -4 | -18 | cool and dark, yellow -11% |
| Xiaomi 18 - Bright Light | 15 | +1 | -14 | -7 | yellow -9% |
| Xiaomi 18 - Low Light | 19 | -4 | -9 | +9 | reds +5% |
| Vivo Look | 56 | +10 | +13 | +5 | |
| Xiaomi Look | 92 | -10 | -13 | -5 | |

Confidence: the brand split (Vivo brighter than Xiaomi) holds under resampling.
The difference between phones of the same brand does not: on same-scene pairs,
X300-vs-X500 and 17-vs-18 predicted the held-out phone no better than chance
(vivo 10/19 scenes, xiaomi 3/6). Per-phone presets are each phone's own average
render, so they're usable, but treat small same-brand differences as noise.

## Installing presets in Lightroom

Lightroom Classic: **Develop -> Presets panel -> `+` -> Import Presets**, pick the `.xmp`.
Generated presets support the **Amount** slider, so you can dial them down.

## Limitations

- HSL slider scales are estimates, not calibrated against Lightroom; if colours
  look too strong, lower Amount or rerun with `-s 0.7`.
- `--preview` and the internal previews approximate the result; they are not a
  Lightroom render.
- Only global tone/colour is reproduced, not phone-specific local processing
  (local HDR, skin smoothing, sharpening).
- `phone_presets.py` has no neutral reference, so presets are relative to the
  average of the sampled phones, and scene mix between folders can bias them.
