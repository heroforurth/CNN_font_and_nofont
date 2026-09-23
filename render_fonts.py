"""Synthesize extra training images by rendering every class with Thai fonts installed on Windows.

Why: several classes have almost no real samples (ฃ=1, ฑ=1, ฬ=3, ๗=4, ฮ=10 ...), so the model
can't learn what they look like in other styles. Rendering them in many fonts fills that gap.

IMPORTANT: the fonts used by the test set (Angsana, Cordia, Leelawadee, Tahoma) are EXCLUDED so the
test score stays honest. Microsoft Sans Serif is also excluded because its Thai glyphs are
near-identical to Tahoma's.

Output: dataset/synth_fonts/<split>/<class>/<font>_<k>.png   (split = train | val)
Two font families are held out as a font-validation split for model selection.

Usage: python render_fonts.py [--variants 8]
"""
import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = r"C:\Windows\Fonts"
REAL_DIR = "dataset/round2"
OUT_DIR = "dataset/synth_fonts"

# UPC families: d=Dillenia e=Eucrosia f=Freesia i=Iris j=Jasmine k=Kodchiang l=Lily
# suffix: l=regular b=bold i=italic bi=bold-italic
FAMILIES = {f"upc{c}": [f"upc{c}{s}.ttf" for s in ("l", "b", "i", "bi")] for c in "defijkl"}
FAMILIES["browallia"] = ["browalia.ttc"]
VAL_FAMILIES = {"upck", "upcl"}   # Kodchiang + Lily held out for validation


def class_char(cls):
    """Folder names are TIS-620 byte codes, e.g. '161' -> 'ก'."""
    return bytes([int(cls)]).decode("cp874")


def render(ch, font_path, size, rng):
    font = ImageFont.truetype(font_path, size)
    # big canvas + offset so zero-width combining marks (ั ิ ่ ...) aren't clipped at x<0
    canvas = Image.new("L", (size * 4, size * 4), 255)
    ImageDraw.Draw(canvas).text((size * 1.5, size * 1.2), ch, font=font, fill=0)
    a = np.array(canvas)
    ys, xs = np.where(a < 128)
    if len(xs) == 0:
        return None
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # random white margin: 0 px matches the tight real crops, 1-4 px matches the font test style
    m = rng.choice([0, 0, 1, 2, 3, 4])
    crop = np.pad(crop, m, constant_values=255)
    # ~40% hard-binarized, like the pixelated real scans
    if rng.random() < 0.4:
        crop = np.where(crop < rng.randint(100, 170), 0, 255).astype(np.uint8)
    return Image.fromarray(crop)


def glyph_signature(font_path, ch, size=40):
    img = render(ch, font_path, size, random.Random(0))
    return None if img is None else np.array(img.resize((16, 16)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=8, help="renders per font per class")
    args = ap.parse_args()

    classes = sorted(c for c in os.listdir(REAL_DIR) if c.isdigit())
    rng = random.Random(42)
    counts = {"train": 0, "val": 0}

    for family, files in FAMILIES.items():
        split = "val" if family in VAL_FAMILIES else "train"
        for fname in files:
            path = os.path.join(FONT_DIR, fname)
            if not os.path.exists(path):
                print(f"skip missing font {fname}")
                continue
            # a glyph identical to the font's "missing glyph" box means the font lacks that char
            notdef = glyph_signature(path, "\uffff")
            for cls in classes:
                ch = class_char(cls)
                sig = glyph_signature(path, ch)
                if sig is None or (notdef is not None and np.array_equal(sig, notdef)):
                    continue
                out = os.path.join(OUT_DIR, split, cls)
                os.makedirs(out, exist_ok=True)
                for k in range(args.variants):
                    img = render(ch, path, rng.randint(32, 72), rng)
                    if img is not None:
                        img.save(os.path.join(out, f"{os.path.splitext(fname)[0]}_{k}.png"))
                        counts[split] += 1
    print(f"rendered: {counts}")


if __name__ == "__main__":
    main()
