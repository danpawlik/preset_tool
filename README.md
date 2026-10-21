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
`xiaomi_18/`. Writes `Vivo_Look.xmp` and `Xiaomi_Look.xmp`.

Each brand look = median render of the brand's photos vs. the average of all four
phones. Photos far from their phone's typical render are ignored, and known
wrong-phone files are excluded (`EXCLUDE` in the script).

Current result from the sample set:

| | Shadows | Mids | Highlights | Colour |
|---|---|---|---|---|
| Vivo | same | +10 | +13 | slightly warm, yellows +6%, reds -4% |
| Xiaomi | same | -10 | -13 | slightly cool, reds +4%, yellows -6% |

Why only one preset per brand: on same-scene pairs, the X300-vs-X500 and
17-vs-18 differences predicted the held-out phone no better than chance
(vivo 10/19 scenes, xiaomi 3/6), so per-model presets would be noise.

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
