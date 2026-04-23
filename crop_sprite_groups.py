from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Background removal tuning.
BACKGROUND_THRESHOLD_MIN = 50
BACKGROUND_THRESHOLD_MAX = 70
BACKGROUND_THRESHOLD_PERCENTILE = 78
ALPHA_RAMP = 18

# Row scanning tuning.
ROW_ACTIVE_PIXEL_RATIO = 0.01
ROW_ACTIVE_PIXEL_MIN = 4
ROW_MERGE_GAP = 4
ROW_MIN_HEIGHT = 12
ROW_MAX_HEIGHT_RATIO = 0.30
ROW_MAX_MEDIAN_HEIGHT_RATIO = 2.4

# Column scanning and frame extraction tuning.
COLUMN_ACTIVE_PIXEL_RATIO = 0.04
COLUMN_ACTIVE_PIXEL_MIN = 3
COLUMN_MERGE_GAP = 4
COLUMN_MIN_WIDTH = 6
FRAME_PADDING = 3
MIN_ROW_FRAMES = 3
MAX_SPACING_DEVIATION_RATIO = 0.65
MAX_WIDTH_DEVIATION_RATIO = 0.80


@dataclass
class Sprite:
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    center_x: float
    center_y: float


@dataclass
class RowGroup:
    index: int
    bbox: tuple[int, int, int, int]
    sprites: list[Sprite]
    source_columns: list[tuple[int, int]]


def remove_background(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dark background into alpha and return an RGBA image plus a hard mask."""
    if image is None:
        raise ValueError("Input image could not be loaded.")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = image[:, :, :3]
    else:
        bgr = image

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    threshold_value = int(
        np.clip(
            np.percentile(gray, BACKGROUND_THRESHOLD_PERCENTILE),
            BACKGROUND_THRESHOLD_MIN,
            BACKGROUND_THRESHOLD_MAX,
        )
    )

    _, hard_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    lower = max(0, threshold_value - ALPHA_RAMP)
    upper = min(255, threshold_value + ALPHA_RAMP)
    alpha = np.interp(gray.astype(np.float32), [lower, upper], [0.0, 255.0]).astype(np.uint8)
    alpha = np.where(hard_mask > 0, np.maximum(alpha, hard_mask), alpha).astype(np.uint8)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba, hard_mask


def find_active_runs(
    counts: np.ndarray,
    threshold: int,
    merge_gap: int,
    min_length: int,
) -> list[tuple[int, int]]:
    """Convert a 1D foreground projection into merged active ranges."""
    active_indices = np.flatnonzero(counts >= threshold)
    if active_indices.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(active_indices[0])
    end = start

    for value in active_indices[1:]:
        position = int(value)
        if position - end <= merge_gap + 1:
            end = position
            continue

        if end - start + 1 >= min_length:
            runs.append((start, end + 1))
        start = position
        end = position

    if end - start + 1 >= min_length:
        runs.append((start, end + 1))
    return runs


def detect_row_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Detect logical sprite rows by scanning the image horizontally."""
    image_h, image_w = mask.shape
    row_counts = np.count_nonzero(mask > 0, axis=1)
    nonzero_counts = row_counts[row_counts > 0]
    if nonzero_counts.size == 0:
        return []

    threshold = max(
        ROW_ACTIVE_PIXEL_MIN,
        int(image_w * ROW_ACTIVE_PIXEL_RATIO),
        int(np.percentile(nonzero_counts, 25) * 0.40),
    )
    rows = find_active_runs(row_counts, threshold, ROW_MERGE_GAP, ROW_MIN_HEIGHT)

    if not rows:
        return []

    heights = [y1 - y0 for y0, y1 in rows]
    median_height = float(np.median(heights))
    max_height = min(int(image_h * ROW_MAX_HEIGHT_RATIO), int(max(median_height * ROW_MAX_MEDIAN_HEIGHT_RATIO, 1)))

    filtered: list[tuple[int, int]] = []
    for y0, y1 in rows:
        height = y1 - y0
        if height > max_height:
            continue
        filtered.append((y0, y1))

    return filtered


def trim_columns(mask_slice: np.ndarray) -> tuple[int, int] | None:
    """Trim a row strip to the active X-range."""
    column_counts = np.count_nonzero(mask_slice > 0, axis=0)
    active_columns = np.flatnonzero(column_counts > 0)
    if active_columns.size == 0:
        return None
    return int(active_columns[0]), int(active_columns[-1]) + 1


def detect_frame_columns(mask_slice: np.ndarray) -> list[tuple[int, int]]:
    """Split a row strip into sprite frames using X-axis activity."""
    row_height = mask_slice.shape[0]
    kernel_width = max(5, min(15, (row_height // 6) | 1))
    processed = cv2.morphologyEx(mask_slice, cv2.MORPH_CLOSE, np.ones((3, kernel_width), dtype=np.uint8))
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    column_counts = np.count_nonzero(processed > 0, axis=0)
    nonzero_counts = column_counts[column_counts > 0]
    if nonzero_counts.size == 0:
        return []

    threshold = max(
        COLUMN_ACTIVE_PIXEL_MIN,
        int(row_height * COLUMN_ACTIVE_PIXEL_RATIO),
        int(np.percentile(nonzero_counts, 30) * 0.35),
    )
    return find_active_runs(column_counts, threshold, COLUMN_MERGE_GAP, COLUMN_MIN_WIDTH)


def ratio_within_tolerance(values: list[float], max_deviation_ratio: float) -> bool:
    if len(values) <= 1:
        return True

    median = float(np.median(values))
    if median <= 0:
        return False

    deviations = [abs(value - median) / median for value in values]
    return max(deviations) <= max_deviation_ratio


def select_consistent_columns(columns: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep the longest contiguous frame run with reasonably even spacing."""
    if len(columns) < MIN_ROW_FRAMES:
        return []

    best_run: list[tuple[int, int]] = []
    for start in range(len(columns)):
        for end in range(start + MIN_ROW_FRAMES, len(columns) + 1):
            candidate = columns[start:end]
            widths = [x1 - x0 for x0, x1 in candidate]
            centers = [(x0 + x1) / 2.0 for x0, x1 in candidate]
            center_steps = [right - left for left, right in zip(centers, centers[1:])]

            if not ratio_within_tolerance(widths, MAX_WIDTH_DEVIATION_RATIO):
                continue
            if center_steps and not ratio_within_tolerance(center_steps, MAX_SPACING_DEVIATION_RATIO):
                continue

            if len(candidate) > len(best_run):
                best_run = candidate

    return best_run


def pad_sprite(sprite: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Center-pad a frame to a consistent cell size without scaling."""
    height, width = sprite.shape[:2]
    canvas = np.zeros((target_height, target_width, 4), dtype=np.uint8)
    x_offset = (target_width - width) // 2
    y_offset = (target_height - height) // 2
    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = sprite
    return canvas


def make_sprite(
    rgba_image: np.ndarray,
    mask: np.ndarray,
    row_y0: int,
    row_y1: int,
    col_x0: int,
    col_x1: int,
) -> Sprite | None:
    """Crop one frame and trim it to the true foreground bounds inside the row cell."""
    x0 = max(0, col_x0 - FRAME_PADDING)
    x1 = min(mask.shape[1], col_x1 + FRAME_PADDING)
    y0 = max(0, row_y0 - FRAME_PADDING)
    y1 = min(mask.shape[0], row_y1 + FRAME_PADDING)

    frame_mask = mask[y0:y1, x0:x1]
    active_y = np.flatnonzero(np.count_nonzero(frame_mask > 0, axis=1) > 0)
    active_x = np.flatnonzero(np.count_nonzero(frame_mask > 0, axis=0) > 0)
    if active_y.size == 0 or active_x.size == 0:
        return None

    trim_y0 = y0 + int(active_y[0])
    trim_y1 = y0 + int(active_y[-1]) + 1
    trim_x0 = x0 + int(active_x[0])
    trim_x1 = x0 + int(active_x[-1]) + 1

    sprite_image = rgba_image[trim_y0:trim_y1, trim_x0:trim_x1].copy()
    if sprite_image.size == 0 or np.count_nonzero(sprite_image[:, :, 3]) == 0:
        return None

    return Sprite(
        image=sprite_image,
        bbox=(trim_x0, trim_y0, trim_x1, trim_y1),
        center_x=(trim_x0 + trim_x1) / 2.0,
        center_y=(trim_y0 + trim_y1) / 2.0,
    )


def extract_row_groups(rgba_image: np.ndarray, mask: np.ndarray) -> list[RowGroup]:
    """Build logical rows directly from horizontal and vertical mask projections."""
    row_ranges = detect_row_ranges(mask)
    groups: list[RowGroup] = []

    for row_index, (row_y0, row_y1) in enumerate(row_ranges):
        row_mask = mask[row_y0:row_y1, :]
        trimmed_columns = trim_columns(row_mask)
        if trimmed_columns is None:
            continue

        row_x0, row_x1 = trimmed_columns
        row_mask = row_mask[:, row_x0:row_x1]
        column_ranges = detect_frame_columns(row_mask)
        if len(column_ranges) < MIN_ROW_FRAMES:
            continue

        selected_columns = select_consistent_columns(column_ranges)
        if len(selected_columns) < MIN_ROW_FRAMES:
            continue

        global_columns = [(row_x0 + x0, row_x0 + x1) for x0, x1 in selected_columns]

        sprites: list[Sprite] = []
        kept_columns: list[tuple[int, int]] = []
        for col_x0, col_x1 in global_columns:
            sprite = make_sprite(rgba_image, mask, row_y0, row_y1, col_x0, col_x1)
            if sprite is None:
                continue
            sprites.append(sprite)
            kept_columns.append((col_x0, col_x1))

        if len(sprites) < MIN_ROW_FRAMES:
            continue

        group_x0 = min(sprite.bbox[0] for sprite in sprites)
        group_y0 = min(sprite.bbox[1] for sprite in sprites)
        group_x1 = max(sprite.bbox[2] for sprite in sprites)
        group_y1 = max(sprite.bbox[3] for sprite in sprites)

        groups.append(
            RowGroup(
                index=len(groups),
                bbox=(group_x0, group_y0, group_x1, group_y1),
                sprites=sprites,
                source_columns=kept_columns,
            )
        )

    return groups


def save_individual_sprites(groups: list[RowGroup], sprites_dir: Path) -> list[Sprite]:
    """Write each extracted frame as an RGBA PNG."""
    ordered_sprites = [sprite for group in groups for sprite in group.sprites]
    ordered_sprites.sort(key=lambda sprite: (sprite.center_y, sprite.center_x))

    for index, sprite in enumerate(ordered_sprites):
        cv2.imwrite(str(sprites_dir / f"sprite_{index}.png"), sprite.image)

    return ordered_sprites


def build_sprite_sheets(groups: list[RowGroup], sheets_dir: Path) -> None:
    """Write one normalized sprite sheet per detected row."""
    for row_index, group in enumerate(groups):
        max_width = max(sprite.image.shape[1] for sprite in group.sprites)
        max_height = max(sprite.image.shape[0] for sprite in group.sprites)
        padded_frames = [pad_sprite(sprite.image, max_width, max_height) for sprite in group.sprites]
        sheet = np.concatenate(padded_frames, axis=1)
        cv2.imwrite(str(sheets_dir / f"row_{row_index}.png"), sheet)


def build_manifest(
    image_path: Path,
    image_shape: tuple[int, int],
    ordered_sprites: list[Sprite],
    groups: list[RowGroup],
) -> dict[str, object]:
    """Describe extracted sprites and row-based sprite sheets in source-image coordinates."""
    image_h, image_w = image_shape
    sprite_ids = {id(sprite): index for index, sprite in enumerate(ordered_sprites)}

    sprite_entries: list[dict[str, object]] = []
    for index, sprite in enumerate(ordered_sprites):
        x0, y0, x1, y1 = sprite.bbox
        sprite_entries.append(
            {
                "id": index,
                "file": f"sprites/sprite_{index}.png",
                "bbox": [x0, y0, x1, y1],
                "center": [sprite.center_x, sprite.center_y],
                "size": [x1 - x0, y1 - y0],
            }
        )

    group_entries: list[dict[str, object]] = []
    for row_index, group in enumerate(groups):
        x0, y0, x1, y1 = group.bbox
        row_sprite_ids = [sprite_ids[id(sprite)] for sprite in group.sprites]
        group_entries.append(
            {
                "id": row_index,
                "file": f"sheets/row_{row_index}.png",
                "sprite_ids": row_sprite_ids,
                "rows": [row_sprite_ids],
                "source_bbox": [x0, y0, x1, y1],
                "grid": {
                    "rows": 1,
                    "columns": len(group.sprites),
                },
            }
        )

    return {
        "source_image": str(image_path),
        "source_size": {"width": image_w, "height": image_h},
        "sprite_count": len(sprite_entries),
        "group_count": len(group_entries),
        "sprites": sprite_entries,
        "groups": group_entries,
    }


def process_sprite_sheet(image_path: Path, output_dir: Path) -> None:
    """Run the row-based extraction pipeline and emit one sheet per detected row."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    sprites_dir = output_dir / "sprites"
    sheets_dir = output_dir / "sheets"
    sprites_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    rgba_image, hard_mask = remove_background(image)
    groups = extract_row_groups(rgba_image, hard_mask)
    if not groups:
        raise RuntimeError("No valid sprite rows were detected. Adjust the row and column scan thresholds at the top.")

    ordered_sprites = save_individual_sprites(groups, sprites_dir)
    build_sprite_sheets(groups, sheets_dir)
    manifest = build_manifest(image_path, image.shape[:2], ordered_sprites, groups)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Extracted {len(ordered_sprites)} sprites into {sprites_dir}")
    print(f"Built {len(groups)} row sheets into {sheets_dir}")
    print(f"Wrote manifest to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove background, scan sprite rows, split frames by columns, and rebuild row sprite sheets."
    )
    parser.add_argument("image", help="Path to the source sprite sheet.")
    parser.add_argument("--output", default="outputs", help="Output directory for sprites/ and sheets/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_sprite_sheet(Path(args.image), Path(args.output))


if __name__ == "__main__":
    main()
