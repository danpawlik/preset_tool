# preset_tool

Generate Lightroom presets (`.xmp`) from pictures you provide.

Two scripts:

- `make_preset.py` - make **your** photos look like a set of style photos.
- `folder_presets.py` - derive presets for every sub-folder of a samples directory
  (one folder = one phone, camera, film stock, ...), compared against each other.

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

# Make absolute preset
python3 ./folder_presets.py --absolute ~/Downloads/photo_samples examples --exclude presets 262830-329.jpg HUOVrgla4AAowpt.jpg

# Another example
python3 ./folder_presets.py  ~/Downloads/photo_samples examples --exclude presets 262830-329.jpg HUOVrgla4AAowpt.jpg
```

**Tip:** the closer the scenes in both folders, the better. Same scenes are ideal,
similar light and subjects are fine. Very different scenes (e.g. flowers vs. night
city) get mistaken for "style".

How it works: per-channel quantile matching gives R/G/B tone curves; the remaining
saturation / hue / luminance gap per Lightroom HSL band is measured after those curves
and written as HSL sliders. White/black watermark frames (e.g. Leica borders) are
cropped before measuring.

## folder_presets.py

```bash
python3 folder_presets.py SAMPLES_DIR OUT_DIR [--exclude NAME ...] [--no-groups] [--min-photos N]
```

| Argument | Meaning |
|---|---|
| `SAMPLES_DIR` | directory whose sub-folders each hold one look (min. 2 folders) |
| `OUT_DIR` | where to write the `.xmp` presets |
| `--exclude` | file or folder names to ignore (e.g. wrong-camera shots, an output folder) |
| `--no-groups` | skip the combined preset per name prefix |
| `--min-photos` | photos needed per folder and light (default 3) |
| `--selftest` | run the built-in check and exit |

Nothing is hardcoded - any folder names work:

```
samples/
  vivo_300_pro/     -> "Vivo 300 Pro - Bright Light", "Vivo 300 Pro - Low Light"
  vivo_500_pro/     -> "Vivo 500 Pro - Bright Light", "Vivo 500 Pro - Low Light"
  xiaomi_17/        -> "Xiaomi 17 - ..."
  xiaomi_18/        -> "Xiaomi 18 - ..."
                    -> "Vivo Look", "Xiaomi Look"   (groups by prefix)
```

Examples:

```bash
# Presets for every folder in ~/samples
python3 folder_presets.py ~/samples ~/presets

# Ignore some files/folders (wrong camera, output dir)
python3 folder_presets.py ~/Downloads/photo_samples ./examples \
    --exclude presets oppo_find_x9_pro 262830-329.jpg HUOVrgla4AAowpt.jpg

# Film stocks, no group presets, require more photos per preset
python3 folder_presets.py ~/film_scans ~/presets --no-groups --min-photos 5
```

How it works:

- **2 presets per folder.** Photos are split into bright and dim scenes by median
  luminance (split at the median of all photos). Each preset = that folder's median
  render in that light vs. the average of all folders in the same light, so bright
  scenes are compared with bright scenes. A light with fewer than `--min-photos`
  photos in a folder gets no preset (the script says so).
- **1 preset per group.** Folders sharing the name prefix before the first `_` or
  `-` (`vivo_300_pro` + `vivo_500_pro` -> `Vivo`) get a combined look from all
  their photos. Single-folder prefixes get no group preset.
- **Outliers ignored.** Photos far from their folder's typical render are skipped
  and listed in the output.
- The script prints a summary per preset (shadows / mids / highlights / warmth /
  saturation vs. the average folder).

Notes from the vivo / Xiaomi sample set: the brand split (Vivo ~20 levels brighter
in mids/highlights than Xiaomi) holds under resampling. The difference between
phones of the same brand does not: on same-scene pairs, X300-vs-X500 and 17-vs-18
predicted the held-out phone no better than chance (vivo 10/19 scenes, xiaomi 3/6).
Per-folder presets are each folder's own average render, so they're usable, but
treat small differences between similar folders as noise.

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
- `folder_presets.py` has no neutral reference, so presets are relative to the
  average of the sampled folders (adding/removing a folder changes all presets),
  and different scene mixes between folders can bias them.
