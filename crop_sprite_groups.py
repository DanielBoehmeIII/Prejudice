from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image

# Bounding boxes are tuned for the provided 1536x1024 sprite sheet.
# Format: (left, top, right, bottom)
GROUP_BOXES = {
    "player": (0, 0, 458, 499),
    "enemies": (458, 0, 1161, 499),
    "tiles": (0, 499, 392, 731),
}


def scale_box(box: tuple[int, int, int, int], sx: float, sy: float) -> tuple[int, int, int, int]:
    l, t, r, b = box
    return (round(l * sx), round(t * sy), round(r * sx), round(b * sy))


def crop_groups(image_path: Path, output_dir: Path) -> list[Path]:
    image = Image.open(image_path)
    width, height = image.size

    base_width, base_height = 1536, 1024
    sx, sy = width / base_width, height / base_height

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for label, box in GROUP_BOXES.items():
        actual = scale_box(box, sx, sy)
        crop = image.crop(actual)
        out_path = output_dir / f"{label}_group.png"
        crop.save(out_path)
        saved.append(out_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop player/enemies/tiles groups from the provided sprite sheet.")
    parser.add_argument("image", type=Path, help="Path to the source sprite sheet image.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory for cropped files.")
    args = parser.parse_args()

    outputs = crop_groups(args.image, args.output)
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
