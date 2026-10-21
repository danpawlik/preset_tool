# preset_tool

Generate Lightroom presets (`.xmp`) from sample photos.

Two scripts:

- **`make_preset.py`** — make *your* photos look like a set of style photos (style → reference).
- **`folder_presets.py`** — one preset per sub-folder of a samples directory (one folder = one phone / camera / film look).

Presets work in **Lightroom Classic**, **Lightroom (cloud)** and **Adobe Camera Raw**, on RAW and JPEG.

---

## Requirements

Python 3 + `numpy` + `Pillow`:

```bash
pip install --user numpy pillow
```

---

## Quick start (recommended)

Put sample JPEGs into folders (one look per folder), then:

```bash
# One solid preset per folder (character-based, vivo_grok-style bias)
python3 folder_presets.py ./samples ./presets --absolute -s 0.9
```

Import the `.xmp` files in Lightroom: **Develop → Presets → + → Import Presets**.

Use the **Amount** slider to soften the effect if needed.

---

## make_preset.py

Build a single preset that pushes reference photos toward a style set.

```bash
python3 make_preset.py STYLE_DIR REFERENCE_DIR -n "My Look" [-o out.xmp] [-s STRENGTH] [--preview]
```

| Argument | Meaning |
|----------|---------|
| `STYLE_DIR` | Photos with the look you want (min. 3) |
| `REFERENCE_DIR` | Photos from the camera you’ll apply the preset to (min. 3) |
| `-n, --name` | Preset name in Lightroom |
| `-o, --out` | Output file (default: `<name>.xmp`) |
| `-s, --strength` | Scale all adjustments (`0.7` softer, `1.2` stronger; default `1.0`) |
| `--preview` | Save a before/after JPG (tone curves only) |
| `--selftest` | Run built-in checks and exit |

Examples:

```bash
# Make your Pixel photos look like Vivo shots
python3 make_preset.py ~/photos/vivo_samples ~/photos/my_pixel -n "Vivo Look" -s 0.9

# Softer + preview
python3 make_preset.py ~/photos/vivo_samples ~/photos/my_pixel -n "Vivo Soft" \
    -s 0.7 -o ~/presets/vivo_soft.xmp --preview
```

**Tip:** Similar scenes in both folders work best. Very different content (flowers vs night city) is often mistaken for “style”.

What it writes:

- Basic tone (Exposure, Contrast, Highlights, Shadows, Whites, Blacks, Texture, Clarity, Vibrance, Dehaze) with a photographic base inspired by hand-tuned phone looks
- Soft HSL (±12)
- RGB tone curves
- Mild split toning, vignette and grain

---

## folder_presets.py

Generate presets for every sub-folder under a samples directory.

```bash
python3 folder_presets.py SAMPLES_DIR OUT_DIR [options]
```

### Modes

| Mode | Flag | What you get |
|------|------|----------------|
| **Absolute** (recommended) | `--absolute` | One character-based preset per folder: a shared photographic base plus how this folder differs from the other folders. Best for “give me the Vivo look”. |
| **Relative** (default without `--absolute`) | — | One preset per folder vs the **average of all folders**. Useful to compare phones. |
| **By light** (optional) | `--by-light` | Also generates Bright Light / Low Light relative variants. Off by default. |

### Options

| Argument | Meaning |
|----------|---------|
| `--absolute` | Character-based absolute look per folder (**recommended**) |
| `--by-light` | Also create Bright / Low Light relative presets |
| `--no-groups` | Skip combined presets for name prefixes (`vivo_300` + `vivo_500` → `Vivo`) |
| `--min-photos N` | Minimum photos per folder (default 3) |
| `-s, --strength` | Scale adjustments (default `1.0`) |
| `--character G` | `--absolute`: how strongly each folder differs from the others (`0` = identical BASE preset for every folder, default `0.5`, `1.0` = strong, often hits slider limits) |
| `--exclude NAME ...` | Ignore folders or filenames |
| `--selftest` | Run checks and exit |

Examples:

```bash
# Recommended: one good preset per phone
python3 folder_presets.py ./samples ./presets --absolute -s 0.9

# Relative comparison between phones (no light split)
python3 folder_presets.py ./samples ./presets -s 0.9

# Relative + optional Bright/Low Light variants
python3 folder_presets.py ./samples ./presets -s 0.9 --by-light

# Exclude junk
python3 folder_presets.py ./samples ./presets --absolute -s 0.9 \
    --exclude presets trash.jpg
```

### How absolute mode works

1. Measures character of the folder (shadow lift, highlight rolloff, warmth, saturation, mid brightness).
2. Starts from a photographic base (stronger shadows, controlled highlights, mild clarity/vibrance — similar to hand-tuned phone presets).
3. Adds this folder's difference from the **average of all folders** (`--character` scales it):
   brightness → Exposure, shadow / highlight levels → Shadows / Highlights, tonal spread → Contrast,
   saturation → Vibrance and per-band HSL saturation, warmth → Color Grading midtone (warm 40° / cool 220°).
   With a single folder there is nothing to compare, so you get the plain base.
4. Adds soft HSL, a gentle S-curve, light split toning, vignette and grain.

It does **not** compare to a linear “neutral” (that used to reverse JPEG processing and looked bad).
Because the differences are measured against the other folders, adding or removing a folder changes the presets.

### How relative mode works

1. Builds a median profile per folder (outliers dropped).
2. Compares each folder to the average of all folders.
3. Writes basic tone (blended with the same photographic base), soft HSL and RGB curves.
4. Optionally (`--by-light`) splits scenes by median luminance into Bright / Low Light.

---

## Installing presets in Lightroom

**Lightroom Classic:** Develop → Presets panel → `+` → Import Presets → select the `.xmp` files.

Generated presets support the **Amount** slider — lower it if the look is too strong.

---

## Tips for better results

1. **Prefer `--absolute`** when you want “the look of this phone”.
2. **8–15 similar, good photos** per folder beat 40 mixed scenes.
3. Avoid mixing night city, green forests and studio portraits in one folder if you can.
4. Start with `-s 0.85`–`0.95`; raise only if the look feels too weak.
5. Bright / Low Light split is optional — for most phone JPEG looks one preset is enough.

---

## Limitations

- Automatic presets will rarely match a carefully hand-tuned preset on a curated set of images.
- HSL scales are approximate; use Amount or `-s` if colours feel strong.
- Only global tone/colour is modelled — not local HDR, skin smoothing or phone-specific sharpening.
- Relative presets change if you add/remove folders (they are relative to the average).
- `--preview` approximates curves only; it is not a full Lightroom render.

---

## Self-tests

```bash
python3 make_preset.py --selftest
python3 folder_presets.py --selftest
```
