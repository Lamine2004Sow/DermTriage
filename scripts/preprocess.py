"""Pré-redimensionne les images HAM10000 (petit côté = 256 px).

Écrit une copie dans data/interim/256/<image_id>.jpg ; les images déjà
présentes sont ignorées, le script est donc relançable.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUTPUT = ROOT / "data/interim/256"
SHORT_SIDE = 256


def resize_short_side(image, short_side=SHORT_SIDE):
    width, height = image.size
    scale = short_side / min(width, height)
    return image.resize(
        (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        path
        for part in ("HAM10000_images_part_1", "HAM10000_images_part_2")
        for path in (RAW / part).glob("*.jpg")
    )
    done = 0
    for source in sources:
        target = OUTPUT / source.name
        if target.exists():
            continue
        with Image.open(source) as image:
            resize_short_side(image.convert("RGB")).save(target, quality=95)
        done += 1
    print(f"{done} image(s) écrite(s), {len(sources) - done} déjà présente(s) -> {OUTPUT}")


if __name__ == "__main__":
    main()
