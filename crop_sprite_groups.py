from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BACKGROUND_THRESHOLD_MIN = 48
BACKGROUND_THRESHOLD_MAX = 78
BACKGROUND_THRESHOLD_PERCENTILE = 78
ALPHA_RAMP = 18
PANEL_BORDER_PADDING = 8
DEFAULT_FRAME_PADDING = 3
TILE_SIZE = 16


@dataclass(frozen=True)
class PanelConfig:
    name: str
    bbox: tuple[int, int, int, int]
    mode: str
    header_height: int
    border_padding: int
    animation_labels: list[str]
    component_min_area: int
    row_min_height: int
    row_merge_gap: int
    row_active_ratio: float
    column_min_width: int
    column_merge_gap: int
    column_active_ratio: float
    close_kernel: tuple[int, int]
    open_kernel: tuple[int, int]
    frame_padding: int
    row_bands: list[tuple[int, int]]
    object_boxes: list[tuple[int, int, int, int]]


@dataclass
class ExtractionItem:
    panel: str
    kind: str
    file: str
    bbox: tuple[int, int, int, int]
    panel_bbox: tuple[int, int, int, int]
    frame_count: int
    row: int | None = None
    row_label: str | None = None
    source_columns: list[tuple[int, int]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layout-driven sprite extraction for labeled multi-panel concept boards."
    )
    parser.add_argument("image", help="Path to the source layout board.")
    parser.add_argument("--layout", default="layout.json", help="Path to the layout config JSON.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    return parser.parse_args()


def remove_background(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def load_layout(layout_path: Path, image_shape: tuple[int, int]) -> list[PanelConfig]:
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    image_h, image_w = image_shape
    panels = layout.get("panels", {})
    if not panels:
        raise ValueError(f"No panels were defined in {layout_path}")

    configs: list[PanelConfig] = []
    for name, raw in panels.items():
        bbox = clamp_bbox(tuple(int(value) for value in raw["bbox"]), image_w, image_h)
        close_kernel = tuple(raw.get("close_kernel", [3, 3]))
        open_kernel = tuple(raw.get("open_kernel", [3, 3]))
        configs.append(
            PanelConfig(
                name=name,
                bbox=bbox,
                mode=str(raw["mode"]),
                header_height=int(raw.get("header_height", 36)),
                border_padding=int(raw.get("border_padding", PANEL_BORDER_PADDING)),
                animation_labels=[str(value) for value in raw.get("animation_labels", [])],
                component_min_area=int(raw.get("component_min_area", 12)),
                row_min_height=int(raw.get("row_min_height", 12)),
                row_merge_gap=int(raw.get("row_merge_gap", 4)),
                row_active_ratio=float(raw.get("row_active_ratio", 0.015)),
                column_min_width=int(raw.get("column_min_width", 8)),
                column_merge_gap=int(raw.get("column_merge_gap", 4)),
                column_active_ratio=float(raw.get("column_active_ratio", 0.02)),
                close_kernel=(max(1, int(close_kernel[0])), max(1, int(close_kernel[1]))),
                open_kernel=(max(1, int(open_kernel[0])), max(1, int(open_kernel[1]))),
                frame_padding=int(raw.get("frame_padding", DEFAULT_FRAME_PADDING)),
                row_bands=[tuple(int(value) for value in band) for band in raw.get("row_bands", [])],
                object_boxes=[tuple(int(value) for value in box) for box in raw.get("object_boxes", [])],
            )
        )

    return configs


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    directories = {
        "panels": output_dir / "panels",
        "sprites": output_dir / "sprites",
        "sheets": output_dir / "sheets",
        "debug": output_dir / "debug",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def find_active_runs(
    counts: np.ndarray,
    threshold: int,
    merge_gap: int,
    min_length: int,
) -> list[tuple[int, int]]:
    active = np.flatnonzero(counts >= threshold)
    if active.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(active[0])
    end = start
    for value in active[1:]:
        current = int(value)
        if current - end <= merge_gap + 1:
            end = current
            continue
        if end - start + 1 >= min_length:
            runs.append((start, end + 1))
        start = current
        end = current

    if end - start + 1 >= min_length:
        runs.append((start, end + 1))
    return runs


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.copy()

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == component_id] = 255
    return filtered


def make_seed_mask(mask: np.ndarray, config: PanelConfig) -> np.ndarray:
    seed = remove_small_components(mask, config.component_min_area)

    if config.close_kernel[0] > 1 or config.close_kernel[1] > 1:
        kernel = np.ones(config.close_kernel, dtype=np.uint8)
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
    if config.open_kernel[0] > 1 or config.open_kernel[1] > 1:
        kernel = np.ones(config.open_kernel, dtype=np.uint8)
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel)
    return seed


def extract_component_bboxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def detect_row_ranges(mask: np.ndarray, config: PanelConfig) -> list[tuple[int, int]]:
    image_h, image_w = mask.shape
    row_counts = np.count_nonzero(mask > 0, axis=1)
    nonzero = row_counts[row_counts > 0]
    if nonzero.size == 0:
        return []

    threshold = max(
        1,
        int(image_w * config.row_active_ratio),
        int(np.percentile(nonzero, 25) * 0.35),
    )
    return find_active_runs(row_counts, threshold, config.row_merge_gap, config.row_min_height)


def detect_column_ranges(mask: np.ndarray, config: PanelConfig) -> list[tuple[int, int]]:
    image_h = mask.shape[0]
    column_counts = np.count_nonzero(mask > 0, axis=0)
    nonzero = column_counts[column_counts > 0]
    if nonzero.size == 0:
        return []

    threshold = max(
        1,
        int(image_h * config.column_active_ratio),
        int(np.percentile(nonzero, 25) * 0.35),
    )
    return find_active_runs(column_counts, threshold, config.column_merge_gap, config.column_min_width)


def trim_to_mask(bbox: tuple[int, int, int, int], mask: np.ndarray) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = bbox
    cropped = mask[y0:y1, x0:x1]
    if cropped.size == 0:
        return None

    ys = np.flatnonzero(np.count_nonzero(cropped > 0, axis=1) > 0)
    xs = np.flatnonzero(np.count_nonzero(cropped > 0, axis=0) > 0)
    if ys.size == 0 or xs.size == 0:
        return None

    return (
        x0 + int(xs[0]),
        y0 + int(ys[0]),
        x0 + int(xs[-1]) + 1,
        y0 + int(ys[-1]) + 1,
    )


def crop_with_padding(
    rgba: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(mask.shape[1], x1 + padding)
    y1 = min(mask.shape[0], y1 + padding)

    trimmed = trim_to_mask((x0, y0, x1, y1), mask)
    if trimmed is None:
        return None

    tx0, ty0, tx1, ty1 = trimmed
    sprite = rgba[ty0:ty1, tx0:tx1].copy()
    if sprite.size == 0 or np.count_nonzero(sprite[:, :, 3]) == 0:
        return None
    return sprite, trimmed


def pad_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    x_offset = (width - frame_w) // 2
    y_offset = (height - frame_h) // 2
    canvas[y_offset : y_offset + frame_h, x_offset : x_offset + frame_w] = frame
    return canvas


def build_sheet(frames: list[np.ndarray]) -> np.ndarray:
    max_width = max(frame.shape[1] for frame in frames)
    max_height = max(frame.shape[0] for frame in frames)
    padded = [pad_frame(frame, max_width, max_height) for frame in frames]
    return np.concatenate(padded, axis=1)


def draw_boxes(image: np.ndarray, title: str, boxes: list[dict[str, Any]]) -> np.ndarray:
    overlay = image.copy()
    for entry in boxes:
        x0, y0, x1, y1 = (int(value) for value in entry["bbox"])
        color = tuple(int(value) for value in entry.get("color", (0, 255, 255)))
        label = str(entry.get("label", ""))
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 1, lineType=cv2.LINE_AA)
        if label:
            text_y = max(14, y0 - 4)
            cv2.putText(
                overlay,
                label,
                (x0 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        overlay,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def save_panel_crop(image: np.ndarray, config: PanelConfig, panels_dir: Path) -> tuple[np.ndarray, str]:
    x0, y0, x1, y1 = config.bbox
    panel_image = image[y0:y1, x0:x1].copy()
    file_name = f"{config.name}.png"
    cv2.imwrite(str(panels_dir / file_name), panel_image)
    return panel_image, f"panels/{file_name}"


def get_content_slices(panel_image: np.ndarray, config: PanelConfig) -> tuple[slice, slice]:
    height, width = panel_image.shape[:2]
    pad = config.border_padding
    y0 = min(height, max(0, config.header_height))
    x0 = min(width, max(0, pad))
    x1 = max(x0, width - pad)
    y1 = max(y0, height - pad)
    return slice(y0, y1), slice(x0, x1)


def is_text_like_ui_bbox(bbox: tuple[int, int, int, int], alpha_pixels: int) -> bool:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return True
    fill_ratio = alpha_pixels / float(width * height)
    return width >= 110 and height >= 28 and fill_ratio < 0.22


def save_item(
    panel_name: str,
    kind: str,
    prefix: str,
    index: int,
    sprite: np.ndarray,
    bbox: tuple[int, int, int, int],
    sprites_dir: Path,
    items: list[ExtractionItem],
    row: int | None = None,
    row_label: str | None = None,
    frame_count: int = 1,
    source_columns: list[tuple[int, int]] | None = None,
) -> None:
    file_name = f"{prefix}_{index:02d}.png"
    cv2.imwrite(str(sprites_dir / file_name), sprite)
    items.append(
        ExtractionItem(
            panel=panel_name,
            kind=kind,
            file=f"sprites/{file_name}",
            bbox=bbox,
            panel_bbox=bbox,
            frame_count=frame_count,
            row=row,
            row_label=row_label,
            source_columns=source_columns,
        )
    )


def extract_animation_panel(
    panel_rgba: np.ndarray,
    panel_mask: np.ndarray,
    config: PanelConfig,
    global_offset: tuple[int, int],
    sprites_dir: Path,
    sheets_dir: Path,
    debug_dir: Path,
) -> tuple[list[ExtractionItem], list[ExtractionItem]]:
    items: list[ExtractionItem] = []
    sheets: list[ExtractionItem] = []
    crop_y, crop_x = get_content_slices(panel_rgba, config)
    content_rgba = panel_rgba[crop_y, crop_x]
    content_mask = panel_mask[crop_y, crop_x]
    seed_mask = make_seed_mask(content_mask, config)
    row_ranges = detect_row_ranges(seed_mask, config)

    debug_boxes: list[dict[str, Any]] = []
    base_x = crop_x.start or 0
    base_y = crop_y.start or 0
    global_x = global_offset[0] + base_x
    global_y = global_offset[1] + base_y

    for row_index, (row_y0, row_y1) in enumerate(row_ranges):
        row_seed = seed_mask[row_y0:row_y1, :]
        columns = detect_column_ranges(row_seed, config)
        if not columns:
            continue

        row_frames: list[np.ndarray] = []
        row_items: list[ExtractionItem] = []
        row_boxes: list[tuple[int, int, int, int]] = []
        for frame_index, (col_x0, col_x1) in enumerate(columns):
            cropped = crop_with_padding(
                content_rgba,
                content_mask,
                (col_x0, row_y0, col_x1, row_y1),
                config.frame_padding,
            )
            if cropped is None:
                continue

            sprite, local_bbox = cropped
            lx0, ly0, lx1, ly1 = local_bbox
            gx0 = global_x + lx0
            gy0 = global_y + ly0
            gx1 = global_x + lx1
            gy1 = global_y + ly1
            row_label = (
                config.animation_labels[row_index]
                if row_index < len(config.animation_labels)
                else f"row_{row_index:02d}"
            )
            file_name = f"{config.name}_{row_label}_frame_{frame_index:02d}.png"
            cv2.imwrite(str(sprites_dir / file_name), sprite)
            item = ExtractionItem(
                panel=config.name,
                kind="frame",
                file=f"sprites/{file_name}",
                bbox=(gx0, gy0, gx1, gy1),
                panel_bbox=(lx0, ly0, lx1, ly1),
                frame_count=1,
                row=row_index,
                row_label=row_label,
            )
            items.append(item)
            row_items.append(item)
            row_frames.append(sprite)
            row_boxes.append((lx0, ly0, lx1, ly1))
            debug_boxes.append(
                {
                    "bbox": (base_x + lx0, base_y + ly0, base_x + lx1, base_y + ly1),
                    "label": f"{row_label}:{frame_index}",
                    "color": (0, 220, 255),
                }
            )

        if not row_frames:
            continue

        row_label = row_items[0].row_label or f"row_{row_index:02d}"
        sheet = build_sheet(row_frames)
        sheet_name = f"{config.name}_{row_label}.png"
        cv2.imwrite(str(sheets_dir / sheet_name), sheet)
        union_x0 = min(box[0] for box in row_boxes)
        union_y0 = min(box[1] for box in row_boxes)
        union_x1 = max(box[2] for box in row_boxes)
        union_y1 = max(box[3] for box in row_boxes)
        sheets.append(
            ExtractionItem(
                panel=config.name,
                kind="sheet",
                file=f"sheets/{sheet_name}",
                bbox=(global_x + union_x0, global_y + union_y0, global_x + union_x1, global_y + union_y1),
                panel_bbox=(union_x0, union_y0, union_x1, union_y1),
                frame_count=len(row_frames),
                row=row_index,
                row_label=row_label,
                source_columns=columns,
            )
        )

    debug = draw_boxes(
        cv2.cvtColor(panel_rgba, cv2.COLOR_BGRA2BGR),
        f"{config.name} ({config.mode})",
        debug_boxes,
    )
    cv2.imwrite(str(debug_dir / f"{config.name}_debug.png"), debug)
    return items, sheets


def extract_objects_panel(
    panel_rgba: np.ndarray,
    panel_mask: np.ndarray,
    config: PanelConfig,
    global_offset: tuple[int, int],
    sprites_dir: Path,
    debug_dir: Path,
) -> list[ExtractionItem]:
    items: list[ExtractionItem] = []
    crop_y, crop_x = get_content_slices(panel_rgba, config)
    content_rgba = panel_rgba[crop_y, crop_x]
    content_mask = panel_mask[crop_y, crop_x]
    seed_mask = make_seed_mask(content_mask, config)
    debug_boxes: list[dict[str, Any]] = []
    base_x = crop_x.start or 0
    base_y = crop_y.start or 0
    global_x = global_offset[0] + base_x
    global_y = global_offset[1] + base_y

    candidate_boxes: list[tuple[int, int, int, int]] = []
    if config.object_boxes:
        candidate_boxes = list(config.object_boxes)
    else:
        row_ranges = list(config.row_bands) if config.row_bands else detect_row_ranges(seed_mask, config)
        if not row_ranges and np.count_nonzero(seed_mask) > 0:
            row_ranges = [(0, seed_mask.shape[0])]
        for row_y0, row_y1 in row_ranges:
            row_seed = seed_mask[row_y0:row_y1, :]
            columns = detect_column_ranges(row_seed, config)
            if not columns and np.count_nonzero(row_seed) > 0:
                columns = [(0, row_seed.shape[1])]
            candidate_boxes.extend((col_x0, row_y0, col_x1, row_y1) for col_x0, col_x1 in columns)

    item_index = 0
    for candidate_bbox in candidate_boxes:
        cropped = crop_with_padding(content_rgba, content_mask, candidate_bbox, config.frame_padding)
        if cropped is None:
            continue

        sprite, local_bbox = cropped
        lx0, ly0, lx1, ly1 = local_bbox
        alpha_pixels = int(np.count_nonzero(sprite[:, :, 3] > 0))
        if config.name == "ui" and is_text_like_ui_bbox(local_bbox, alpha_pixels):
            continue

        gx0 = global_x + lx0
        gy0 = global_y + ly0
        gx1 = global_x + lx1
        gy1 = global_y + ly1
        file_name = f"{config.name}_{item_index:02d}.png"
        cv2.imwrite(str(sprites_dir / file_name), sprite)
        items.append(
            ExtractionItem(
                panel=config.name,
                kind="sprite",
                file=f"sprites/{file_name}",
                bbox=(gx0, gy0, gx1, gy1),
                panel_bbox=(lx0, ly0, lx1, ly1),
                frame_count=1,
            )
        )
        debug_boxes.append(
            {
                "bbox": (base_x + lx0, base_y + ly0, base_x + lx1, base_y + ly1),
                "label": f"{item_index}",
                "color": (120, 255, 120),
            }
        )
        item_index += 1

    debug = draw_boxes(
        cv2.cvtColor(panel_rgba, cv2.COLOR_BGRA2BGR),
        f"{config.name} ({config.mode})",
        debug_boxes,
    )
    cv2.imwrite(str(debug_dir / f"{config.name}_debug.png"), debug)
    return items


def extract_tiles_panel(
    panel_rgba: np.ndarray,
    panel_mask: np.ndarray,
    config: PanelConfig,
    global_offset: tuple[int, int],
    sprites_dir: Path,
    debug_dir: Path,
) -> list[ExtractionItem]:
    crop_y, crop_x = get_content_slices(panel_rgba, config)
    content_rgba = panel_rgba[crop_y, crop_x]
    content_mask = panel_mask[crop_y, crop_x]
    seed_mask = make_seed_mask(content_mask, config)
    active_bbox = trim_to_mask((0, 0, seed_mask.shape[1], seed_mask.shape[0]), seed_mask)
    if active_bbox is None:
        debug = draw_boxes(cv2.cvtColor(panel_rgba, cv2.COLOR_BGRA2BGR), f"{config.name} ({config.mode})", [])
        cv2.imwrite(str(debug_dir / f"{config.name}_debug.png"), debug)
        return []

    ax0, ay0, ax1, ay1 = active_bbox
    grid_width = int(np.ceil((ax1 - ax0) / TILE_SIZE) * TILE_SIZE)
    grid_height = int(np.ceil((ay1 - ay0) / TILE_SIZE) * TILE_SIZE)
    grid_x1 = min(content_mask.shape[1], ax0 + grid_width)
    grid_y1 = min(content_mask.shape[0], ay0 + grid_height)

    items: list[ExtractionItem] = []
    debug_boxes: list[dict[str, Any]] = []
    base_x = crop_x.start or 0
    base_y = crop_y.start or 0
    item_index = 0

    for tile_y in range(ay0, grid_y1, TILE_SIZE):
        for tile_x in range(ax0, grid_x1, TILE_SIZE):
            tile_mask = content_mask[tile_y : tile_y + TILE_SIZE, tile_x : tile_x + TILE_SIZE]
            if tile_mask.size == 0 or np.count_nonzero(tile_mask > 0) == 0:
                continue

            sprite = content_rgba[tile_y : tile_y + TILE_SIZE, tile_x : tile_x + TILE_SIZE].copy()
            if sprite.shape[0] != TILE_SIZE or sprite.shape[1] != TILE_SIZE:
                continue

            gx0 = global_offset[0] + base_x + tile_x
            gy0 = global_offset[1] + base_y + tile_y
            gx1 = gx0 + TILE_SIZE
            gy1 = gy0 + TILE_SIZE
            file_name = f"{config.name}_{item_index:03d}.png"
            cv2.imwrite(str(sprites_dir / file_name), sprite)
            items.append(
                ExtractionItem(
                    panel=config.name,
                    kind="tile",
                    file=f"sprites/{file_name}",
                    bbox=(gx0, gy0, gx1, gy1),
                    panel_bbox=(tile_x, tile_y, tile_x + TILE_SIZE, tile_y + TILE_SIZE),
                    frame_count=1,
                )
            )
            debug_boxes.append(
                {
                    "bbox": (base_x + tile_x, base_y + tile_y, base_x + tile_x + TILE_SIZE, base_y + tile_y + TILE_SIZE),
                    "label": f"{item_index}",
                    "color": (255, 170, 0),
                }
            )
            item_index += 1

    debug = draw_boxes(
        cv2.cvtColor(panel_rgba, cv2.COLOR_BGRA2BGR),
        f"{config.name} ({config.mode})",
        debug_boxes,
    )
    cv2.imwrite(str(debug_dir / f"{config.name}_debug.png"), debug)
    return items


def extract_panel(
    panel_image: np.ndarray,
    config: PanelConfig,
    output_dirs: dict[str, Path],
) -> tuple[list[ExtractionItem], list[ExtractionItem]]:
    panel_rgba, panel_mask = remove_background(panel_image)
    x0, y0, _, _ = config.bbox

    if config.mode == "animation_rows":
        items, sheets = extract_animation_panel(
            panel_rgba,
            panel_mask,
            config,
            (x0, y0),
            output_dirs["sprites"],
            output_dirs["sheets"],
            output_dirs["debug"],
        )
        return items, sheets

    if config.mode == "tiles":
        items = extract_tiles_panel(
            panel_rgba,
            panel_mask,
            config,
            (x0, y0),
            output_dirs["sprites"],
            output_dirs["debug"],
        )
        return items, []

    items = extract_objects_panel(
        panel_rgba,
        panel_mask,
        config,
        (x0, y0),
        output_dirs["sprites"],
        output_dirs["debug"],
    )
    return items, []


def build_manifest(
    image_path: Path,
    layout_path: Path,
    image_shape: tuple[int, int],
    panel_results: list[dict[str, Any]],
) -> dict[str, Any]:
    image_h, image_w = image_shape
    return {
        "source_image": str(image_path),
        "layout_file": str(layout_path),
        "source_size": {"width": image_w, "height": image_h},
        "panel_count": len(panel_results),
        "panels": panel_results,
    }


def process_layout_board(image_path: Path, layout_path: Path, output_dir: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    output_dirs = ensure_output_dirs(output_dir)
    configs = load_layout(layout_path, image.shape[:2])

    layout_boxes: list[dict[str, Any]] = []
    panel_results: list[dict[str, Any]] = []

    for config in configs:
        panel_image, panel_file = save_panel_crop(image, config, output_dirs["panels"])
        items, sheets = extract_panel(panel_image, config, output_dirs)
        x0, y0, x1, y1 = config.bbox
        layout_boxes.append(
            {
                "bbox": config.bbox,
                "label": config.name,
                "color": (0, 255, 255),
            }
        )

        panel_results.append(
            {
                "name": config.name,
                "mode": config.mode,
                "bbox": [x0, y0, x1, y1],
                "panel_file": panel_file,
                "frame_count": int(sum(item.frame_count for item in items)),
                "output_files": [item.file for item in items] + [sheet.file for sheet in sheets],
                "bounding_boxes": [list(item.bbox) for item in items] + [list(sheet.bbox) for sheet in sheets],
                "items": [
                    {
                        "kind": item.kind,
                        "file": item.file,
                        "bbox": list(item.bbox),
                        "panel_bbox": list(item.panel_bbox),
                        "frame_count": item.frame_count,
                        "row": item.row,
                        "row_label": item.row_label,
                    }
                    for item in items
                ],
                "sheets": [
                    {
                        "kind": sheet.kind,
                        "file": sheet.file,
                        "bbox": list(sheet.bbox),
                        "panel_bbox": list(sheet.panel_bbox),
                        "frame_count": sheet.frame_count,
                        "row": sheet.row,
                        "row_label": sheet.row_label,
                        "source_columns": [list(pair) for pair in (sheet.source_columns or [])],
                    }
                    for sheet in sheets
                ],
            }
        )

    layout_debug = draw_boxes(image[:, :, :3].copy(), "layout panels", layout_boxes)
    cv2.imwrite(str(output_dirs["debug"] / "layout_panels.png"), layout_debug)

    manifest = build_manifest(image_path, layout_path, image.shape[:2], panel_results)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Cropped {len(configs)} panels into {output_dirs['panels']}")
    print(f"Wrote sprites to {output_dirs['sprites']}")
    print(f"Wrote sheets to {output_dirs['sheets']}")
    print(f"Wrote debug overlays to {output_dirs['debug']}")
    print(f"Wrote manifest to {manifest_path}")


def main() -> None:
    args = parse_args()
    process_layout_board(Path(args.image), Path(args.layout), Path(args.output))


if __name__ == "__main__":
    main()
