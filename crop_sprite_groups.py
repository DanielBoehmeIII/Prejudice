from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BG_DISTANCE_THRESHOLD = 42.0
ALPHA_MIN = 120
ALPHA_MAX = 255
EDGE_BOOST = 28
BACKGROUND_CONSISTENCY_TOLERANCE = 14.0
BACKGROUND_SAMPLE_RATIO = 0.08
BACKGROUND_SAMPLE_MIN = 6
BACKGROUND_SEED_RATIO = 0.55
ALPHA_RAMP_RATIO = 1.65
MASK_ALPHA_THRESHOLD = 32
PANEL_BORDER_PADDING = 8
DEFAULT_FRAME_PADDING = 3
TILE_SIZE = 16
ATLAS_MAX_SIZE = 1024
ATLAS_PADDING = 1
BACKGROUND_LAYER_MIN_AREA = 9_000
BACKGROUND_LAYER_MIN_FILL_RATIO = 0.45
BACKGROUND_LAYER_MAX_ALPHA_STDDEV = 42.0
ANIMATION_SIZE_TOLERANCE = 0.10
ANIMATION_ASPECT_TOLERANCE = 0.12
ANIMATION_ROW_OVERLAP_RATIO = 0.65
ANIMATION_SPACING_TOLERANCE = 0.35
ANIMATION_MAX_GAP_FACTOR = 1.75
ANIMATION_MIN_FRAMES = 3
TEXT_LIKE_MAX_STROKE_RATIO = 0.24
TEXT_LIKE_MIN_COMPONENT_DENSITY = 0.0025

SINGULAR_PANEL_NAMES = {
    "backgrounds": "background",
    "cloaks": "cloak",
    "enemies": "enemy",
    "platforms": "platform",
    "props": "prop",
    "tiles": "tile",
    "weapons": "weapon",
}

UNIFORM_FRAME_DURATIONS_MS = {
    "idle": 120,
    "walk": 80,
    "run": 60,
    "jump": 90,
    "effects": 80,
}


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
    frame_index: int | None = None
    row: int | None = None
    row_label: str | None = None
    source_columns: list[tuple[int, int]] | None = None
    asset_name: str | None = None
    animation_name: str | None = None
    animation_role: str | None = None
    animation_type: str | None = None
    normalized_size: tuple[int, int] | None = None
    anchor: tuple[int, int] | None = None
    anchor_type: str | None = None
    classification: str | None = None


@dataclass(frozen=True)
class SpriteComponent:
    bbox: tuple[int, int, int, int]
    area: int
    alpha_pixels: int
    disconnected_parts: int
    fill_ratio: float
    stroke_ratio: float

    @property
    def x0(self) -> int:
        return self.bbox[0]

    @property
    def y0(self) -> int:
        return self.bbox[1]

    @property
    def x1(self) -> int:
        return self.bbox[2]

    @property
    def y1(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1.0, float(self.height))

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) * 0.5


@dataclass
class AtlasPlacement:
    sprite_name: str
    atlas_index: int
    atlas_file: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class FrameAlignmentDebug:
    raw_anchor: tuple[int, int]
    smoothed_anchor: tuple[int, int]
    reference_anchor: tuple[int, int]
    offset: tuple[int, int]
    baseline_anchor: tuple[int, int]
    aligned_anchor: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layout-driven sprite extraction for labeled multi-panel concept boards."
    )
    parser.add_argument("image", help="Path to the source layout board.")
    parser.add_argument("--layout", default="layout.json", help="Path to the layout config JSON.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    return parser.parse_args()


def sample_background_color(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = bgr.shape[:2]
    sample_size = max(BACKGROUND_SAMPLE_MIN, int(min(height, width) * BACKGROUND_SAMPLE_RATIO))
    sample_size = min(sample_size, height, width)

    patches = [
        bgr[:sample_size, :sample_size],
        bgr[:sample_size, width - sample_size : width],
        bgr[height - sample_size : height, :sample_size],
        bgr[height - sample_size : height, width - sample_size : width],
    ]
    flat_patches = [patch.reshape(-1, 3).astype(np.float32) for patch in patches if patch.size > 0]
    if not flat_patches:
        raise ValueError("Could not sample background color from an empty image.")

    all_samples = np.concatenate(flat_patches, axis=0)
    background_color = np.median(all_samples, axis=0)

    patch_medians = np.stack([np.median(values, axis=0) for values in flat_patches], axis=0)
    patch_distances = np.linalg.norm(patch_medians - background_color, axis=1)
    is_consistent = bool(np.max(patch_distances, initial=0.0) <= BACKGROUND_CONSISTENCY_TOLERANCE)
    return background_color.astype(np.float32), is_consistent


def smoothstep(values: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    if edge1 <= edge0:
        return (values >= edge1).astype(np.float32)
    scaled = np.clip((values - edge0) / float(edge1 - edge0), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def edge_connected_background(seed_mask: np.ndarray) -> np.ndarray:
    if seed_mask.size == 0 or not np.any(seed_mask):
        return np.zeros_like(seed_mask, dtype=bool)

    seed_u8 = seed_mask.astype(np.uint8)
    component_count, labels = cv2.connectedComponents(seed_u8, connectivity=8)
    if component_count <= 1:
        return np.zeros_like(seed_mask, dtype=bool)

    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros_like(seed_mask, dtype=bool)
    return np.isin(labels, border_labels)


def build_alpha_mask(
    bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    background_color, is_consistent = sample_background_color(bgr)
    pixel_values = bgr.astype(np.float32)
    color_distance = np.linalg.norm(pixel_values - background_color[None, None, :], axis=2)

    seed_threshold = BG_DISTANCE_THRESHOLD * BACKGROUND_SEED_RATIO
    alpha_threshold = BG_DISTANCE_THRESHOLD
    alpha_outer = max(alpha_threshold + 1.0, alpha_threshold * ALPHA_RAMP_RATIO)

    background_seed = color_distance <= seed_threshold
    background_region = edge_connected_background(background_seed) if is_consistent else background_seed

    alpha_curve = smoothstep(color_distance, seed_threshold, alpha_outer)
    alpha = np.round(alpha_curve * ALPHA_MAX).astype(np.uint8)

    definite_background = background_region & (color_distance <= seed_threshold)
    alpha[definite_background] = 0

    surviving_pixels = (~background_region) & (alpha > 0)
    alpha[surviving_pixels] = np.maximum(alpha[surviving_pixels], ALPHA_MIN).astype(np.uint8)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = cv2.magnitude(grad_x, grad_y)
    edge_scale = float(np.percentile(edge_strength, 95)) if np.any(edge_strength > 0) else 0.0
    if edge_scale > 0:
        edge_weight = np.clip(edge_strength / edge_scale, 0.0, 1.0)
        alpha_boost = np.round(edge_weight * EDGE_BOOST).astype(np.uint8)
        boosted = surviving_pixels & (alpha < ALPHA_MAX)
        alpha[boosted] = np.minimum(ALPHA_MAX, alpha[boosted].astype(np.int16) + alpha_boost[boosted].astype(np.int16))
        alpha = alpha.astype(np.uint8)

    mask = np.where(alpha >= MASK_ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    return alpha, mask, color_distance


def remove_background(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image is None:
        raise ValueError("Input image could not be loaded.")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = image[:, :, :3]
    else:
        bgr = image

    alpha, hard_mask, _ = build_alpha_mask(bgr)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba, hard_mask


def save_background_debug(panel_name: str, panel_rgba: np.ndarray, panel_mask: np.ndarray, debug_dir: Path) -> None:
    alpha = panel_rgba[:, :, 3]
    alpha_vis = np.repeat(alpha[:, :, None], 3, axis=2)
    mask_vis = np.repeat(panel_mask[:, :, None], 3, axis=2)
    cv2.imwrite(str(debug_dir / f"{panel_name}_alpha_preview.png"), alpha_vis)
    cv2.imwrite(str(debug_dir / f"{panel_name}_mask_preview.png"), mask_vis)


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
        "atlases": output_dir / "atlases",
        "metadata": output_dir / "metadata",
        "panels": output_dir / "panels",
        "sprites": output_dir / "sprites",
        "sheets": output_dir / "sheets",
        "debug": output_dir / "debug",
        "debug_animations": output_dir / "debug" / "animations",
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


def count_nonempty_components(mask: np.ndarray) -> int:
    if mask.size == 0 or np.count_nonzero(mask) == 0:
        return 0
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return sum(1 for component_id in range(1, component_count) if int(stats[component_id, cv2.CC_STAT_AREA]) > 0)


def component_stroke_ratio(mask: np.ndarray) -> float:
    if mask.size == 0 or np.count_nonzero(mask) == 0:
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    edge_pixels = int(np.count_nonzero(mask > eroded))
    alpha_pixels = int(np.count_nonzero(mask))
    return edge_pixels / float(max(1, alpha_pixels))


def build_sprite_components(mask: np.ndarray, min_area: int) -> list[SpriteComponent]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[SpriteComponent] = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component_mask = np.where(labels[y : y + h, x : x + w] == component_id, 255, 0).astype(np.uint8)
        alpha_pixels = int(np.count_nonzero(component_mask))
        fill_ratio = alpha_pixels / float(max(1, w * h))
        disconnected_parts = count_nonempty_components(component_mask)
        components.append(
            SpriteComponent(
                bbox=(x, y, x + w, y + h),
                area=area,
                alpha_pixels=alpha_pixels,
                disconnected_parts=disconnected_parts,
                fill_ratio=fill_ratio,
                stroke_ratio=component_stroke_ratio(component_mask),
            )
        )
    components.sort(key=lambda component: (component.y0, component.x0))
    return components


def component_from_sprite_crop(
    bbox: tuple[int, int, int, int],
    sprite: np.ndarray,
    content_bbox: tuple[int, int, int, int],
) -> SpriteComponent:
    alpha_mask = np.where(sprite[:, :, 3] > 0, 255, 0).astype(np.uint8)
    alpha_pixels = int(np.count_nonzero(alpha_mask))
    disconnected_parts = count_nonempty_components(alpha_mask)
    cx0, cy0, cx1, cy1 = content_bbox
    content_area = max(1, (cx1 - cx0) * (cy1 - cy0))
    return SpriteComponent(
        bbox=bbox,
        area=content_area,
        alpha_pixels=alpha_pixels,
        disconnected_parts=disconnected_parts,
        fill_ratio=alpha_pixels / float(content_area),
        stroke_ratio=component_stroke_ratio(alpha_mask),
    )


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


def expand_bbox(
    bbox: tuple[int, int, int, int],
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding)
    y1 = min(height, y1 + padding)
    return x0, y0, x1, y1


def to_relative_bbox(
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ox0, oy0, _, _ = outer_bbox
    ix0, iy0, ix1, iy1 = inner_bbox
    return ix0 - ox0, iy0 - oy0, ix1 - ox0, iy1 - oy0


def extract_region(
    rgba: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    x0, y0, x1, y1 = bbox
    sprite = rgba[y0:y1, x0:x1].copy()
    if sprite.size == 0 or np.count_nonzero(sprite[:, :, 3]) == 0:
        return None
    return sprite


def extract_cropped_region(
    rgba: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    expanded = expand_bbox(bbox, padding, mask.shape[1], mask.shape[0])
    content_bbox = trim_to_mask(expanded, mask)
    if content_bbox is None:
        return None

    sprite = extract_region(rgba, expanded)
    if sprite is None:
        return None
    return sprite, expanded, to_relative_bbox(expanded, content_bbox)


LOCOMOTION_ROLES = {"idle", "walk", "run", "jump", "fall"}

ANCHOR_COMPONENT_MIN_PIXELS = 4
ANCHOR_COMPONENT_MIN_RATIO = 0.12
ANCHOR_BOTTOM_PERCENTILE = 92.0
ANCHOR_DOWNWARD_BIAS = 0.28
ANCHOR_SMOOTHING_WINDOW = 3
ANCHOR_DENSE_BAND_RATIO = 0.45
EFFECT_HEIGHT_SCALE_MIN = 0.7
EFFECT_HEIGHT_SCALE_MAX = 1.3


def anchor_type_for_panel(panel_name: str) -> str:
    if panel_name in {"player", "enemy", "enemies"}:
        return "ground"
    if panel_name in {"light_beam", "aura"}:
        return "center"
    if panel_name == "ui":
        return "top_left"
    if panel_name == "backgrounds":
        return "none"
    return "ground"


def visible_alpha_bounds(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    if frame.size == 0:
        return None
    alpha = frame[:, :, 3]
    ys, xs = np.nonzero(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def lowest_visible_pixel_y(frame: np.ndarray) -> int:
    bounds = visible_alpha_bounds(frame)
    if bounds is None:
        return max(0, frame.shape[0] - 1)
    return bounds[3] - 1


def filtered_anchor_alpha(frame: np.ndarray) -> np.ndarray:
    if frame.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    alpha = frame[:, :, 3].astype(np.float32)
    mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        return alpha

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return alpha

    areas = [int(stats[component_id, cv2.CC_STAT_AREA]) for component_id in range(1, component_count)]
    largest_area = max(areas, default=0)
    min_area = max(ANCHOR_COMPONENT_MIN_PIXELS, int(round(largest_area * ANCHOR_COMPONENT_MIN_RATIO)))

    filtered = np.zeros_like(alpha)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component_mask = labels == component_id
        filtered[component_mask] = alpha[component_mask]

    if np.count_nonzero(filtered) == 0:
        return alpha
    return filtered


def weighted_row_percentile(row_weights: np.ndarray, percentile: float) -> float:
    total = float(np.sum(row_weights))
    if total <= 0:
        return 0.0
    clamped = min(100.0, max(0.0, percentile))
    threshold = total * (clamped / 100.0)
    cumulative = np.cumsum(row_weights, dtype=np.float64)
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(min(len(row_weights) - 1, max(0, index)))


def lowest_dense_band_y(alpha: np.ndarray) -> float:
    if alpha.size == 0:
        return 0.0

    row_counts = np.count_nonzero(alpha > 0, axis=1).astype(np.float32)
    if not np.any(row_counts > 0):
        return 0.0

    max_count = float(np.max(row_counts))
    threshold = max(1.0, max_count * ANCHOR_DENSE_BAND_RATIO)
    dense_rows = np.flatnonzero(row_counts >= threshold)
    if dense_rows.size == 0:
        return weighted_row_percentile(np.sum(alpha, axis=1), ANCHOR_BOTTOM_PERCENTILE)
    return float(dense_rows[-1])


def animation_anchor_type(panel_name: str, row_role: str, anchor_type: str) -> str:
    if row_role == "effects":
        return "center"
    if panel_name in {"aura", "light_beam"}:
        return "center"
    return anchor_type


def anchor_bias_for_role(row_role: str, anchor_type: str) -> float:
    if anchor_type != "ground":
        return 0.0
    if row_role in LOCOMOTION_ROLES:
        return ANCHOR_DOWNWARD_BIAS
    if row_role == "attack":
        return 0.18
    if row_role == "effects":
        return 0.0
    return ANCHOR_DOWNWARD_BIAS


def compute_frame_anchor(frame: np.ndarray, anchor_type: str, downward_bias: float = ANCHOR_DOWNWARD_BIAS) -> tuple[int, int]:
    if frame.size == 0:
        return (0, 0)

    if anchor_type in {"top_left", "none"}:
        return (0, 0)

    alpha = filtered_anchor_alpha(frame)
    weight_sum = float(np.sum(alpha))
    if weight_sum <= 0:
        bounds = visible_alpha_bounds(frame)
        if bounds is None:
            return (0, 0)
        x0, y0, x1, y1 = bounds
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    ys, xs = np.indices(alpha.shape, dtype=np.float32)
    center_x = float(np.sum(xs * alpha) / weight_sum)
    center_y = float(np.sum(ys * alpha) / weight_sum)

    if anchor_type == "center":
        return (int(round(center_x)), int(round(center_y)))

    foot_y = lowest_dense_band_y(alpha)
    blended_y = center_y * (1.0 - downward_bias) + foot_y * downward_bias
    return (int(round(center_x)), int(round(blended_y)))


def smooth_anchors(anchors: list[tuple[int, int]], window: int = ANCHOR_SMOOTHING_WINDOW) -> list[tuple[int, int]]:
    if len(anchors) <= 1 or window <= 1:
        return list(anchors)

    half_window = window // 2
    smoothed: list[tuple[int, int]] = []
    for index in range(len(anchors)):
        start = max(0, index - half_window)
        end = min(len(anchors), index + half_window + 1)
        sample = anchors[start:end]
        mean_x = int(round(float(np.mean([anchor[0] for anchor in sample]))))
        mean_y = int(round(float(np.mean([anchor[1] for anchor in sample]))))
        smoothed.append((mean_x, mean_y))
    return smoothed


def default_reference_frame_index(frame_count: int) -> int:
    return 0 if frame_count <= 2 else frame_count // 2


def baseline_frame_offset(frame: np.ndarray, canvas_size: tuple[int, int], anchor_type: str) -> tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    frame_height, frame_width = frame.shape[:2]

    if anchor_type == "center":
        return ((canvas_width - frame_width) // 2, (canvas_height - frame_height) // 2)
    if anchor_type in {"top_left", "none"}:
        return (0, 0)
    return ((canvas_width - frame_width) // 2, canvas_height - frame_height)


def crop_frame_to_visible_content(
    frame: np.ndarray,
    raw_anchor: tuple[int, int],
    smoothed_anchor: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    bounds = visible_alpha_bounds(frame)
    if bounds is None:
        height, width = frame.shape[:2]
        bounds = (0, 0, width, height)

    x0, y0, x1, y1 = bounds
    cropped = frame[y0:y1, x0:x1].copy()
    return (
        cropped,
        (raw_anchor[0] - x0, raw_anchor[1] - y0),
        (smoothed_anchor[0] - x0, smoothed_anchor[1] - y0),
        bounds,
    )


def alignment_template(
    frames: list[np.ndarray],
    anchors: list[tuple[int, int]],
    anchor_type: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if anchor_type in {"top_left", "none"}:
        max_width = max(frame.shape[1] for frame in frames)
        max_height = max(frame.shape[0] for frame in frames)
        return (max_width, max_height), (0, 0)

    left_extent = max(anchor[0] for anchor in anchors)
    top_extent = max(anchor[1] for anchor in anchors)
    right_extent = max(frame.shape[1] - anchor[0] for frame, anchor in zip(frames, anchors, strict=False))
    bottom_extent = max(frame.shape[0] - anchor[1] for frame, anchor in zip(frames, anchors, strict=False))
    return (left_extent + right_extent, top_extent + bottom_extent), (left_extent, top_extent)


def shared_template_from_reference(
    frames: list[np.ndarray],
    anchors: list[tuple[int, int]],
    reference_index: int,
    anchor_type: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if not frames:
        return (0, 0), (0, 0)

    if anchor_type in {"top_left", "none"}:
        return alignment_template(frames, anchors, anchor_type)

    preferred_anchor = anchors[reference_index]
    left_extent = max(preferred_anchor[0], max(anchor[0] for anchor in anchors))
    top_extent = max(preferred_anchor[1], max(anchor[1] for anchor in anchors))
    right_extent = max(frame.shape[1] - anchor[0] for frame, anchor in zip(frames, anchors, strict=False))
    bottom_extent = max(frame.shape[0] - anchor[1] for frame, anchor in zip(frames, anchors, strict=False))
    return (left_extent + right_extent, top_extent + bottom_extent), (left_extent, top_extent)


def visible_content_height(frame: np.ndarray) -> int:
    bounds = visible_alpha_bounds(frame)
    if bounds is None:
        return 0
    return bounds[3] - bounds[1]


def scale_frames_to_target_height(frames: list[np.ndarray], target_height: int) -> list[np.ndarray]:
    if not frames or target_height <= 0:
        return frames

    visible_heights = [visible_content_height(frame) for frame in frames if visible_content_height(frame) > 0]
    if not visible_heights:
        return frames

    current_height = int(round(float(np.median(visible_heights))))
    if current_height <= 0:
        return frames

    scale = float(target_height) / float(current_height)
    scale = min(EFFECT_HEIGHT_SCALE_MAX, max(EFFECT_HEIGHT_SCALE_MIN, scale))
    if abs(scale - 1.0) < 0.02:
        return frames

    scaled_frames: list[np.ndarray] = []
    for frame in frames:
        new_width = max(1, int(round(frame.shape[1] * scale)))
        new_height = max(1, int(round(frame.shape[0] * scale)))
        interpolation = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
        if scaled.ndim == 2:
            scaled = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGRA)
        elif scaled.shape[2] == 3:
            scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2BGRA)
        scaled_frames.append(scaled)
    return scaled_frames


def normalize_animation_frames(
    frames: list[np.ndarray],
    anchor_type: str,
    animation_role: str,
    panel_name: str,
    shared_anchor: tuple[int, int] | None = None,
    shared_size: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], tuple[int, int], tuple[int, int], list[FrameAlignmentDebug]]:
    if not frames:
        raise ValueError("normalize_animation_frames requires at least one frame.")

    effective_anchor_type = animation_anchor_type(panel_name, animation_role, anchor_type)
    downward_bias = anchor_bias_for_role(animation_role, effective_anchor_type)
    raw_anchors = [compute_frame_anchor(frame, effective_anchor_type, downward_bias=downward_bias) for frame in frames]
    smoothed_anchors = smooth_anchors(raw_anchors)
    reference_index = default_reference_frame_index(len(frames))
    reference_anchor = raw_anchors[reference_index]
    smoothed_anchors[reference_index] = reference_anchor

    prepared = [
        crop_frame_to_visible_content(frame, raw_anchor, smoothed_anchor)
        for frame, raw_anchor, smoothed_anchor in zip(frames, raw_anchors, smoothed_anchors, strict=False)
    ]
    cropped_frames = [entry[0] for entry in prepared]
    local_raw_anchors = [entry[1] for entry in prepared]
    local_smoothed_anchors = [entry[2] for entry in prepared]

    if shared_size is None or shared_anchor is None:
        computed_size, computed_anchor = alignment_template(cropped_frames, local_smoothed_anchors, effective_anchor_type)
    else:
        computed_size, computed_anchor = shared_size, shared_anchor

    canvas_width, canvas_height = computed_size

    if shared_anchor is None:
        shared_anchor = computed_anchor

    normalized: list[np.ndarray] = []
    debug_entries: list[FrameAlignmentDebug] = []
    for cropped_frame, raw_anchor, smoothed_anchor in zip(cropped_frames, local_raw_anchors, local_smoothed_anchors, strict=False):
        canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
        frame_h, frame_w = cropped_frame.shape[:2]

        baseline_offset = baseline_frame_offset(cropped_frame, (canvas_width, canvas_height), effective_anchor_type)
        baseline_anchor = (
            baseline_offset[0] + smoothed_anchor[0],
            baseline_offset[1] + smoothed_anchor[1],
        )
        x_offset = shared_anchor[0] - smoothed_anchor[0]
        y_offset = shared_anchor[1] - smoothed_anchor[1]
        x_offset = max(0, min(canvas_width - frame_w, x_offset))
        y_offset = max(0, min(canvas_height - frame_h, y_offset))
        canvas[y_offset : y_offset + frame_h, x_offset : x_offset + frame_w] = cropped_frame
        normalized.append(canvas)
        debug_entries.append(
            FrameAlignmentDebug(
                raw_anchor=raw_anchor,
                smoothed_anchor=smoothed_anchor,
                reference_anchor=local_raw_anchors[reference_index],
                offset=(x_offset - baseline_offset[0], y_offset - baseline_offset[1]),
                baseline_anchor=baseline_anchor,
                aligned_anchor=(x_offset + smoothed_anchor[0], y_offset + smoothed_anchor[1]),
            )
        )

    return normalized, (canvas_width, canvas_height), shared_anchor, debug_entries


def build_sheet(frames: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(frames, axis=1)


def normalize_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "sprite"


def panel_asset_prefix(panel_name: str) -> str:
    return normalize_key(SINGULAR_PANEL_NAMES.get(panel_name, panel_name))


def infer_animation_role(panel_name: str, row_index: int, frame_count: int, row_count: int) -> str:
    if panel_name != "player":
        return "effects"

    if row_index == 0:
        return "idle"
    if row_index == 1:
        return "walk"
    if row_index == 2 and frame_count >= 5:
        return "run"
    if row_index == 3:
        return "jump"
    if row_index == row_count - 1 and frame_count <= 4:
        return "attack"
    if row_index >= max(0, row_count - 3) and frame_count <= 3:
        return "attack"
    if row_index == 4 and frame_count >= 4:
        return "effects"
    return "effects"


def frame_humanoid_score(frame: np.ndarray) -> float:
    bounds = visible_alpha_bounds(frame)
    if bounds is None:
        return 0.0
    x0, y0, x1, y1 = bounds
    alpha = frame[y0:y1, x0:x1, 3]
    mask = alpha > 0
    height, width = mask.shape
    if width <= 0 or height <= 0:
        return 0.0

    fill_ratio = float(np.count_nonzero(mask)) / float(max(1, width * height))
    aspect = height / float(max(1, width))
    thirds = np.array_split(mask, 3, axis=0)
    densities = [float(np.count_nonzero(band)) / float(max(1, band.size)) for band in thirds]
    column_centers: list[float] = []
    for band in thirds:
        ys, xs = np.nonzero(band)
        if xs.size == 0:
            column_centers.append(width * 0.5)
        else:
            column_centers.append(float(xs.mean()))
    center_spread = max(column_centers) - min(column_centers)
    center_alignment = 1.0 - min(1.0, center_spread / float(max(1, width)))

    score = 0.0
    if 0.18 <= fill_ratio <= 0.72:
        score += 0.25
    if aspect >= 1.0:
        score += 0.3
    if densities[1] >= 0.18 and densities[2] >= 0.12:
        score += 0.25
    if center_alignment >= 0.55:
        score += 0.2
    return score


def row_humanoid_score(frames: list[np.ndarray]) -> float:
    if not frames:
        return 0.0
    return float(np.mean([frame_humanoid_score(frame) for frame in frames]))


def player_body_label_sequence(body_row_count: int) -> list[str]:
    base = ["idle", "walk", "run"]
    if body_row_count <= len(base):
        return base[:body_row_count]
    if body_row_count == 4:
        return base + ["jump"]
    if body_row_count == 5:
        return base + ["jump", "attack"]
    if body_row_count == 6:
        return base + ["jump", "fall", "attack"]
    if body_row_count == 7:
        return base + ["jump", "fall", "attack", "shield"]
    return base + ["jump", "fall", "attack", "shield", "ability"][: max(0, body_row_count - 3)]


def semantic_player_labels(row_entries: list[dict[str, Any]]) -> None:
    sorted_rows = sorted(row_entries, key=lambda entry: entry["bbox"][1])
    body_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for entry in sorted_rows:
        humanoid_score = row_humanoid_score(entry["row_frames"])
        entry["humanoid_score"] = humanoid_score
        if humanoid_score >= 0.55:
            body_rows.append(entry)
        else:
            effect_rows.append(entry)

    body_labels = player_body_label_sequence(len(body_rows))
    for entry, label in zip(body_rows, body_labels, strict=False):
        entry["row_role"] = label
        entry["animation_type"] = label
        entry["classification"] = "player"

    for entry in effect_rows:
        entry["row_role"] = "effects"
        entry["animation_type"] = "effects"
        entry["classification"] = "effect"


def apply_semantic_animation_labels(panel_name: str, row_entries: list[dict[str, Any]]) -> None:
    if panel_name == "player":
        semantic_player_labels(row_entries)
        return

    default_role = "effects" if panel_name in {"aura", "light_beam"} else "misc"
    default_classification = "effect" if panel_name in {"aura", "light_beam"} else group_classification_label(panel_name, default_role)
    for entry in sorted(row_entries, key=lambda row: row["bbox"][1]):
        entry["row_role"] = default_role
        entry["animation_type"] = default_role
        entry["classification"] = default_classification


def group_classification_label(panel_name: str, role: str) -> str:
    if panel_name == "player":
        return "effect" if role == "effects" else "player"
    if panel_name in {"aura", "light_beam"}:
        return "effect"
    if panel_name in {"enemy", "enemies"}:
        return "enemy"
    if panel_name == "ui":
        return "ui"
    return "misc"


def validate_animation_row(components: list[SpriteComponent]) -> tuple[bool, str | None]:
    if len(components) < ANIMATION_MIN_FRAMES:
        return False, "frame_count_lt_3"

    widths = [component.width for component in components]
    heights = [component.height for component in components]
    aspects = [component.aspect_ratio for component in components]
    spacing = [components[index + 1].x0 - components[index].x0 for index in range(len(components) - 1)]
    gaps = [components[index + 1].x0 - components[index].x1 for index in range(len(components) - 1)]

    if not consistent_series_ratio(widths, ANIMATION_SIZE_TOLERANCE):
        return False, "inconsistent_width"
    if not consistent_series_ratio(heights, ANIMATION_SIZE_TOLERANCE):
        return False, "inconsistent_height"
    if not consistent_series_ratio(aspects, ANIMATION_ASPECT_TOLERANCE):
        return False, "inconsistent_aspect"
    if row_vertical_overlap_ratio(components) < ANIMATION_ROW_OVERLAP_RATIO:
        return False, "row_misaligned"
    if not consistent_series_ratio(spacing, ANIMATION_SPACING_TOLERANCE):
        return False, "spacing_inconsistent"

    median_gap = float(np.median(gaps)) if gaps else 0.0
    median_width = float(np.median(widths))
    max_gap = max(gaps, default=0)
    if max_gap > median_width * ANIMATION_MAX_GAP_FACTOR:
        return False, "gap_too_large"
    return True, None


def animation_timing_for_role(role: str, frame_count: int) -> dict[str, Any]:
    if role == "attack":
        if frame_count <= 1:
            return {"frame_duration": 80}
        durations = np.linspace(50, 140, num=frame_count)
        return {"frame_durations": [int(round(value)) for value in durations.tolist()]}
    return {"frame_duration": UNIFORM_FRAME_DURATIONS_MS.get(role, 80)}


def composite_on_checker(image: np.ndarray, tile: int = 8) -> np.ndarray:
    height, width = image.shape[:2]
    checker = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            shade = 72 if ((x // tile) + (y // tile)) % 2 == 0 else 104
            checker[y : y + tile, x : x + tile] = shade

    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    rgb = image[:, :, :3].astype(np.float32)
    base = checker.astype(np.float32)
    return np.round(rgb * alpha + base * (1.0 - alpha)).astype(np.uint8)


def is_background_layer_candidate(
    region: np.ndarray,
    content_bbox: tuple[int, int, int, int] | None,
) -> bool:
    if region.size == 0 or content_bbox is None:
        return False

    height, width = region.shape[:2]
    area = width * height
    if area < BACKGROUND_LAYER_MIN_AREA:
        return False

    alpha = region[:, :, 3]
    fill_ratio = float(np.count_nonzero(alpha > 0)) / float(area)
    alpha_stddev = float(np.std(alpha.astype(np.float32)))
    x0, y0, x1, y1 = content_bbox
    content_w = x1 - x0
    content_h = y1 - y0
    bbox_fill_ratio = float(content_w * content_h) / float(area)

    return (
        fill_ratio >= BACKGROUND_LAYER_MIN_FILL_RATIO
        and bbox_fill_ratio >= 0.72
        and alpha_stddev <= BACKGROUND_LAYER_MAX_ALPHA_STDDEV
    )


def save_animation_preview(
    name: str,
    frames: list[np.ndarray],
    anchor: tuple[int, int],
    alignment_debug: list[FrameAlignmentDebug],
    debug_dir: Path,
) -> None:
    preview = composite_on_checker(build_sheet(frames))
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]
    anchor_x, anchor_y = anchor
    for index, frame_debug in enumerate(alignment_debug):
        frame_x0 = index * frame_width
        frame_x1 = frame_x0 + frame_width
        cv2.rectangle(preview, (frame_x0, 0), (frame_x1 - 1, frame_height - 1), (96, 96, 96), 1, lineType=cv2.LINE_AA)
        bounds = visible_alpha_bounds(frames[index])
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            cv2.rectangle(
                preview,
                (frame_x0 + x0, y0),
                (frame_x0 + x1 - 1, y1 - 1),
                (0, 255, 0),
                1,
                lineType=cv2.LINE_AA,
            )
        baseline_x = frame_x0 + frame_debug.baseline_anchor[0]
        baseline_y = frame_debug.baseline_anchor[1]
        cv2.arrowedLine(
            preview,
            (baseline_x, baseline_y),
            (frame_x0 + anchor_x, anchor_y),
            (0, 255, 255),
            1,
            line_type=cv2.LINE_AA,
            tipLength=0.2,
        )
        cv2.circle(preview, (baseline_x, baseline_y), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(preview, (frame_x0 + anchor_x, anchor_y), 2, (255, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{name}_preview.png"), preview)


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


def classify_component_for_animation(component: SpriteComponent, panel_name: str, panel_shape: tuple[int, int]) -> tuple[str, str | None]:
    panel_height, panel_width = panel_shape
    width = component.width
    height = component.height
    area = width * height
    if width <= 0 or height <= 0:
        return "misc", "empty_component"
    if max(width, height) <= 8:
        return "misc", "tiny_fragment"
    if panel_name not in {"aura", "light_beam"} and (
        area >= int(panel_width * panel_height * 0.22) or width >= int(panel_width * 0.7) or height >= int(panel_height * 0.7)
    ):
        return "misc", "background_like"
    if (
        panel_name in {"ui", "player", "light_beam", "aura"}
        and component.stroke_ratio >= 0.82
        and component.fill_ratio <= TEXT_LIKE_MAX_STROKE_RATIO
        and component.disconnected_parts <= 2
    ):
        return "ui", "thin_strokes"
    if (
        component.disconnected_parts >= 4
        and component.fill_ratio <= 0.34
        and (component.alpha_pixels / float(max(1, area))) <= 0.22
    ):
        return "ui", "glyph_cluster"
    return "", None


def bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def row_vertical_overlap_ratio(components: list[SpriteComponent]) -> float:
    if not components:
        return 0.0
    overlap_y0 = max(component.y0 for component in components)
    overlap_y1 = min(component.y1 for component in components)
    if overlap_y1 <= overlap_y0:
        return 0.0
    min_height = min(component.height for component in components)
    return (overlap_y1 - overlap_y0) / float(max(1, min_height))


def group_components_into_rows(components: list[SpriteComponent]) -> list[list[SpriteComponent]]:
    rows: list[list[SpriteComponent]] = []
    for component in sorted(components, key=lambda entry: (entry.center_y, entry.x0)):
        placed = False
        for row in rows:
            row_y0 = min(entry.y0 for entry in row)
            row_y1 = max(entry.y1 for entry in row)
            row_height = max(1.0, float(np.median([entry.height for entry in row])))
            center_gap = abs(component.center_y - float(np.median([entry.center_y for entry in row])))
            overlap_y0 = max(row_y0, component.y0)
            overlap_y1 = min(row_y1, component.y1)
            overlap_ratio = (overlap_y1 - overlap_y0) / row_height if overlap_y1 > overlap_y0 else 0.0
            if center_gap <= row_height * 0.55 or overlap_ratio >= ANIMATION_ROW_OVERLAP_RATIO:
                row.append(component)
                placed = True
                break
        if not placed:
            rows.append([component])

    for row in rows:
        row.sort(key=lambda component: component.x0)
    rows.sort(key=lambda row: min(component.y0 for component in row))
    return rows


def consistent_series_ratio(values: list[float], tolerance: float) -> bool:
    if len(values) <= 1:
        return True
    median = float(np.median(values))
    lower = median * (1.0 - tolerance)
    upper = median * (1.0 + tolerance)
    return all(lower <= value <= upper for value in values)


def best_animation_subgroup(components: list[SpriteComponent]) -> tuple[list[SpriteComponent], str | None]:
    if len(components) < ANIMATION_MIN_FRAMES:
        return [], "frame_count_lt_3"

    best_group: list[SpriteComponent] = []
    best_reason = "frame_count_lt_3"
    best_score: tuple[int, int] | None = None
    for start in range(len(components)):
        for end in range(start + ANIMATION_MIN_FRAMES, len(components) + 1):
            subgroup = components[start:end]
            is_valid, reason = validate_animation_row(subgroup)
            if not is_valid:
                if len(subgroup) > len(best_group):
                    best_reason = reason
                continue
            score = (len(subgroup), -start)
            if best_score is None or score > best_score:
                best_group = subgroup
                best_score = score
                best_reason = None
    return best_group, best_reason


def columns_for_bounds(
    seed_mask: np.ndarray,
    row_y0: int,
    row_y1: int,
    config: PanelConfig,
    x_bounds: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    row_seed = seed_mask[row_y0:row_y1, :]
    columns = detect_column_ranges(row_seed, config)
    if x_bounds is None:
        return columns
    bound_x0, bound_x1 = x_bounds
    filtered: list[tuple[int, int]] = []
    for col_x0, col_x1 in columns:
        overlap = max(0, min(col_x1, bound_x1) - max(col_x0, bound_x0))
        if overlap <= 0:
            continue
        if overlap / float(max(1, col_x1 - col_x0)) < 0.45 and overlap / float(max(1, bound_x1 - bound_x0)) < 0.12:
            continue
        filtered.append((max(col_x0, bound_x0), min(col_x1, bound_x1)))
    return filtered


def extract_row_frame_candidates(
    content_rgba: np.ndarray,
    content_mask: np.ndarray,
    seed_mask: np.ndarray,
    config: PanelConfig,
    row_y0: int,
    row_y1: int,
    x_bounds: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[tuple[int, int]], list[SpriteComponent]]:
    columns = columns_for_bounds(seed_mask, row_y0, row_y1, config, x_bounds=x_bounds)
    frames: list[np.ndarray] = []
    row_boxes: list[tuple[int, int, int, int]] = []
    row_content_boxes: list[tuple[int, int, int, int]] = []
    frame_components: list[SpriteComponent] = []
    filtered_columns: list[tuple[int, int]] = []

    for col_x0, col_x1 in columns:
        cropped = extract_cropped_region(content_rgba, content_mask, (col_x0, row_y0, col_x1, row_y1), config.frame_padding)
        if cropped is None:
            continue
        sprite, local_bbox, content_bbox = cropped
        frames.append(sprite)
        row_boxes.append(local_bbox)
        row_content_boxes.append(content_bbox)
        filtered_columns.append((col_x0, col_x1))
        frame_components.append(component_from_sprite_crop(local_bbox, sprite, content_bbox))

    return frames, row_boxes, row_content_boxes, filtered_columns, frame_components


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
    frame_index: int | None = None,
    source_columns: list[tuple[int, int]] | None = None,
    asset_name: str | None = None,
    animation_name: str | None = None,
    animation_role: str | None = None,
    normalized_size: tuple[int, int] | None = None,
    anchor: tuple[int, int] | None = None,
    anchor_type: str | None = None,
    classification: str | None = None,
) -> None:
    file_name = f"{prefix}_{index}.png"
    cv2.imwrite(str(sprites_dir / file_name), sprite)
    items.append(
        ExtractionItem(
            panel=panel_name,
            kind=kind,
            file=f"sprites/{file_name}",
            bbox=bbox,
            panel_bbox=bbox,
            frame_count=frame_count,
            frame_index=frame_index,
            row=row,
            row_label=row_label,
            source_columns=source_columns,
            asset_name=asset_name or f"{prefix}_{index}",
            animation_name=animation_name,
            animation_role=animation_role,
            normalized_size=normalized_size,
            anchor=anchor,
            anchor_type=anchor_type,
            classification=classification,
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
    animation_debug_dir: Path,
) -> tuple[list[ExtractionItem], list[ExtractionItem], list[dict[str, Any]]]:
    items: list[ExtractionItem] = []
    sheets: list[ExtractionItem] = []
    crop_y, crop_x = get_content_slices(panel_rgba, config)
    content_rgba = panel_rgba[crop_y, crop_x]
    content_mask = panel_mask[crop_y, crop_x]
    seed_mask = make_seed_mask(content_mask, config)

    debug_boxes: list[dict[str, Any]] = []
    base_x = crop_x.start or 0
    base_y = crop_y.start or 0
    global_x = global_offset[0] + base_x
    global_y = global_offset[1] + base_y
    role_counts: dict[str, int] = {}
    anchor_type = anchor_type_for_panel(config.name)
    row_entries: list[dict[str, Any]] = []
    group_debug: list[dict[str, Any]] = []
    panel_shape = content_mask.shape
    components = build_sprite_components(seed_mask, config.component_min_area)

    accepted_components: list[SpriteComponent] = []
    rejected_components: list[tuple[SpriteComponent, str, str]] = []
    for component in components:
        classification, reason = classify_component_for_animation(component, config.name, panel_shape)
        if reason:
            rejected_components.append((component, classification, reason))
            debug_boxes.append(
                {
                    "bbox": (base_x + component.x0, base_y + component.y0, base_x + component.x1, base_y + component.y1),
                    "label": classification,
                    "color": (0, 0, 255),
                }
            )
            group_debug.append(
                {
                    "bbox": [global_x + component.x0, global_y + component.y0, global_x + component.x1, global_y + component.y1],
                    "frame_count": 1,
                    "classification": classification,
                    "accepted": False,
                    "reason": reason,
                }
            )
            continue
        accepted_components.append(component)

    grouped_rows = group_components_into_rows(accepted_components)
    for row_index, coarse_row in enumerate(grouped_rows):
        coarse_row = sorted(coarse_row, key=lambda component: component.x0)
        valid_group, reason = best_animation_subgroup(coarse_row)
        coarse_bbox = bbox_union([component.bbox for component in coarse_row])
        if not valid_group:
            for component in coarse_row:
                debug_boxes.append(
                    {
                        "bbox": (base_x + component.x0, base_y + component.y0, base_x + component.x1, base_y + component.y1),
                        "label": "misc",
                        "color": (0, 0, 255),
                    }
                )
            group_debug.append(
                {
                    "bbox": [global_x + coarse_bbox[0], global_y + coarse_bbox[1], global_x + coarse_bbox[2], global_y + coarse_bbox[3]],
                    "frame_count": len(coarse_row),
                    "classification": "misc",
                    "accepted": False,
                    "reason": reason,
                }
            )
            continue

        rejected_remainder = [component for component in coarse_row if component not in valid_group]
        if rejected_remainder:
            remainder_bbox = bbox_union([component.bbox for component in rejected_remainder])
            for component in rejected_remainder:
                debug_boxes.append(
                    {
                        "bbox": (base_x + component.x0, base_y + component.y0, base_x + component.x1, base_y + component.y1),
                        "label": "misc",
                        "color": (0, 0, 255),
                    }
                )
            group_debug.append(
                {
                    "bbox": [global_x + remainder_bbox[0], global_y + remainder_bbox[1], global_x + remainder_bbox[2], global_y + remainder_bbox[3]],
                    "frame_count": len(rejected_remainder),
                    "classification": "misc",
                    "accepted": False,
                    "reason": "outlier_in_row",
                }
            )

        row_y0 = min(component.y0 for component in valid_group)
        row_y1 = max(component.y1 for component in valid_group)
        x_bounds = (min(component.x0 for component in valid_group), max(component.x1 for component in valid_group))
        row_frames, row_boxes, row_content_boxes, columns, frame_components = extract_row_frame_candidates(
            content_rgba,
            content_mask,
            seed_mask,
            config,
            row_y0,
            row_y1,
            x_bounds=x_bounds,
        )
        selected_frames, _ = best_animation_subgroup(frame_components)
        if selected_frames:
            keep_indexes = [index for index, component in enumerate(frame_components) if component in selected_frames]
            row_frames = [row_frames[index] for index in keep_indexes]
            row_boxes = [row_boxes[index] for index in keep_indexes]
            row_content_boxes = [row_content_boxes[index] for index in keep_indexes]
            columns = [columns[index] for index in keep_indexes]

        if len(row_frames) < ANIMATION_MIN_FRAMES:
            group_debug.append(
                {
                    "bbox": [global_x + coarse_bbox[0], global_y + coarse_bbox[1], global_x + coarse_bbox[2], global_y + coarse_bbox[3]],
                    "frame_count": len(row_frames),
                    "classification": "misc",
                    "accepted": False,
                    "reason": "frame_extraction_failed",
                }
            )
            continue

        row_entries.append(
            {
                "row_index": row_index,
                "row_frames": row_frames,
                "row_boxes": row_boxes,
                "row_content_boxes": row_content_boxes,
                "columns": columns,
                "bbox": bbox_union(row_boxes),
            }
        )

    if not row_entries and config.name in {"aura", "light_beam"}:
        fallback_rows = detect_row_ranges(seed_mask, config)
        for row_index, (row_y0, row_y1) in enumerate(fallback_rows):
            row_frames, row_boxes, row_content_boxes, columns, frame_components = extract_row_frame_candidates(
                content_rgba,
                content_mask,
                seed_mask,
                config,
                row_y0,
                row_y1,
            )
            valid_frames, reason = best_animation_subgroup(frame_components)
            if not valid_frames:
                if row_boxes:
                    row_bbox = bbox_union(row_boxes)
                    group_debug.append(
                        {
                            "bbox": [global_x + row_bbox[0], global_y + row_bbox[1], global_x + row_bbox[2], global_y + row_bbox[3]],
                            "frame_count": len(row_boxes),
                            "classification": "misc",
                            "accepted": False,
                            "reason": reason,
                        }
                    )
                continue
            keep_indexes = [index for index, component in enumerate(frame_components) if component in valid_frames]
            row_entries.append(
                {
                    "row_index": row_index,
                    "row_frames": [row_frames[index] for index in keep_indexes],
                    "row_boxes": [row_boxes[index] for index in keep_indexes],
                    "row_content_boxes": [row_content_boxes[index] for index in keep_indexes],
                    "columns": [columns[index] for index in keep_indexes],
                    "bbox": bbox_union([row_boxes[index] for index in keep_indexes]),
                }
            )

    apply_semantic_animation_labels(config.name, row_entries)
    for row_index, entry in enumerate(sorted(row_entries, key=lambda row: row["bbox"][1])):
        entry["row_index"] = row_index

    locomotion_rows = [
        entry for entry in row_entries if config.name == "player" and entry["row_role"] in LOCOMOTION_ROLES
    ]
    player_shared_anchor: tuple[int, int] | None = None
    player_shared_size: tuple[int, int] | None = None
    player_reference_height: int | None = None
    player_body_rows = [
        entry for entry in row_entries if config.name == "player" and entry["row_role"] != "effects"
    ]
    if player_body_rows and anchor_type == "ground":
        idle_reference = next((entry for entry in player_body_rows if entry["row_role"] == "idle"), player_body_rows[0])
        player_body_frames = [frame for entry in player_body_rows for frame in entry["row_frames"]]
        player_body_roles = [entry["row_role"] for entry in player_body_rows for _ in entry["row_frames"]]
        body_anchors = [
            compute_frame_anchor(frame, anchor_type, downward_bias=anchor_bias_for_role(role, anchor_type))
            for frame, role in zip(player_body_frames, player_body_roles, strict=False)
        ]
        smoothed_body_anchors = smooth_anchors(body_anchors)
        body_prepared = [
            crop_frame_to_visible_content(frame, anchor, anchor)
            for frame, anchor in zip(player_body_frames, smoothed_body_anchors, strict=False)
        ]
        body_crops = [entry[0] for entry in body_prepared]
        body_local_anchors = [entry[2] for entry in body_prepared]
        reference_frame = idle_reference["row_frames"][default_reference_frame_index(len(idle_reference["row_frames"]))]
        reference_anchor = compute_frame_anchor(
            reference_frame,
            anchor_type,
            downward_bias=anchor_bias_for_role(idle_reference["row_role"], anchor_type),
        )
        _, _, idle_local_anchor, _ = crop_frame_to_visible_content(reference_frame, reference_anchor, reference_anchor)
        try:
            idle_reference_index = next(
                index for index, anchor in enumerate(body_local_anchors) if anchor == idle_local_anchor
            )
        except StopIteration:
            idle_reference_index = 0
        player_shared_size, player_shared_anchor = shared_template_from_reference(
            body_crops,
            body_local_anchors,
            idle_reference_index,
            anchor_type,
        )
        player_reference_height = int(round(float(np.median([visible_content_height(frame) for frame in idle_reference["row_frames"] if visible_content_height(frame) > 0]))))

    for entry in row_entries:
        row_index = entry["row_index"]
        row_role = entry["row_role"]
        animation_type = entry["animation_type"]
        row_frames = entry["row_frames"]
        row_boxes = entry["row_boxes"]
        row_content_boxes = entry["row_content_boxes"]
        columns = entry["columns"]
        group_classification = entry["classification"]
        group_bbox = entry["bbox"]
        role_count = role_counts.get(row_role, 0)
        role_counts[row_role] = role_count + 1
        base_animation_prefix = f"{panel_asset_prefix(config.name)}_{row_role}"
        animation_prefix = base_animation_prefix if role_count == 0 else f"{base_animation_prefix}_{role_count}"
        if player_reference_height is not None and (row_role == "effects" or config.name in {"aura", "light_beam"}):
            row_frames = scale_frames_to_target_height(row_frames, player_reference_height)
        shared_size = player_shared_size if config.name == "player" and row_role != "effects" else None
        shared_anchor = player_shared_anchor if config.name == "player" and row_role != "effects" else None
        if config.name in {"aura", "light_beam"} and player_shared_anchor is not None:
            shared_anchor = player_shared_anchor
        normalized_frames, normalized_size, shared_anchor, alignment_debug = normalize_animation_frames(
            row_frames,
            anchor_type,
            row_role,
            config.name,
            shared_anchor=shared_anchor,
            shared_size=shared_size,
        )

        for frame_index, (frame, local_bbox) in enumerate(zip(normalized_frames, row_boxes, strict=False)):
            lx0, ly0, lx1, ly1 = local_bbox
            gx0 = global_x + lx0
            gy0 = global_y + ly0
            gx1 = global_x + lx1
            gy1 = global_y + ly1
            asset_name = f"{animation_prefix}_{frame_index}"
            file_name = f"{asset_name}.png"
            cv2.imwrite(str(sprites_dir / file_name), frame)
            items.append(
                ExtractionItem(
                    panel=config.name,
                    kind="frame",
                    file=f"sprites/{file_name}",
                    bbox=(gx0, gy0, gx1, gy1),
                    panel_bbox=(lx0, ly0, lx1, ly1),
                    frame_count=1,
                    frame_index=frame_index,
                    row=row_index,
                    row_label=row_role,
                    asset_name=asset_name,
                    animation_name=animation_prefix,
                    animation_role=row_role,
                    animation_type=animation_type,
                    normalized_size=normalized_size,
                    anchor=shared_anchor,
                    anchor_type=anchor_type,
                    classification=group_classification,
                )
            )
            debug_boxes.append(
                {
                    "bbox": (base_x + lx0, base_y + ly0, base_x + lx1, base_y + ly1),
                    "label": f"{group_classification}:{frame_index}",
                    "color": (0, 220, 255),
                }
            )
            debug_boxes.append(
                {
                    "bbox": (
                        base_x + lx0 + row_content_boxes[frame_index][0],
                        base_y + ly0 + row_content_boxes[frame_index][1],
                        base_x + lx0 + row_content_boxes[frame_index][2],
                        base_y + ly0 + row_content_boxes[frame_index][3],
                    ),
                    "label": "",
                    "color": (0, 128, 255),
            }
        )
        group_debug.append(
            {
                "bbox": [global_x + group_bbox[0], global_y + group_bbox[1], global_x + group_bbox[2], global_y + group_bbox[3]],
                "frame_count": len(normalized_frames),
                "classification": group_classification,
                "accepted": True,
                "animation_role": row_role,
                "animation_type": animation_type,
            }
        )

        sheet = build_sheet(normalized_frames)
        sheet_name = f"{animation_prefix}.png"
        cv2.imwrite(str(sheets_dir / sheet_name), sheet)
        save_animation_preview(animation_prefix, normalized_frames, shared_anchor, alignment_debug, animation_debug_dir)
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
                frame_count=len(normalized_frames),
                row=row_index,
                row_label=row_role,
                source_columns=columns,
                asset_name=animation_prefix,
                animation_name=animation_prefix,
                animation_role=row_role,
                animation_type=animation_type,
                normalized_size=normalized_size,
                anchor=shared_anchor,
                anchor_type=anchor_type,
                classification=group_classification,
            )
        )

    debug = draw_boxes(
        cv2.cvtColor(panel_rgba, cv2.COLOR_BGRA2BGR),
        f"{config.name} ({config.mode})",
        debug_boxes,
    )
    cv2.imwrite(str(debug_dir / f"{config.name}_debug.png"), debug)
    return items, sheets, group_debug


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
        expanded_bbox = expand_bbox(candidate_bbox, config.frame_padding, content_mask.shape[1], content_mask.shape[0])
        content_bbox = trim_to_mask(expanded_bbox, content_mask)
        if content_bbox is None:
            continue

        classification = "sprite"
        local_bbox = content_bbox
        sprite = extract_region(content_rgba, content_bbox)
        if sprite is None:
            continue

        expanded_region = extract_region(content_rgba, expanded_bbox)
        is_background_panel = config.name == "backgrounds"
        if expanded_region is not None and (
            is_background_panel
            or is_background_layer_candidate(expanded_region, to_relative_bbox(expanded_bbox, content_bbox))
        ):
            classification = "background_layer"
            local_bbox = expanded_bbox
            sprite = expanded_region

        lx0, ly0, lx1, ly1 = local_bbox
        alpha_pixels = int(np.count_nonzero(sprite[:, :, 3] > 0))
        if classification == "sprite" and config.name == "ui" and is_text_like_ui_bbox(local_bbox, alpha_pixels):
            continue

        gx0 = global_x + lx0
        gy0 = global_y + ly0
        gx1 = global_x + lx1
        gy1 = global_y + ly1
        asset_name = f"{panel_asset_prefix(config.name)}_{item_index}"
        file_name = f"{asset_name}.png"
        cv2.imwrite(str(sprites_dir / file_name), sprite)
        items.append(
            ExtractionItem(
                panel=config.name,
                kind="sprite",
                file=f"sprites/{file_name}",
                bbox=(gx0, gy0, gx1, gy1),
                panel_bbox=(lx0, ly0, lx1, ly1),
                frame_count=1,
                asset_name=asset_name,
                anchor=(0, 0),
                anchor_type=anchor_type_for_panel(config.name),
                classification=classification,
            )
        )
        debug_boxes.append(
            {
                "bbox": (base_x + lx0, base_y + ly0, base_x + lx1, base_y + ly1),
                "label": f"{item_index}",
                "color": (180, 120, 255) if classification == "background_layer" else (120, 255, 120),
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
            asset_name = f"{panel_asset_prefix(config.name)}_{item_index}"
            file_name = f"{asset_name}.png"
            cv2.imwrite(str(sprites_dir / file_name), sprite)
            items.append(
                ExtractionItem(
                    panel=config.name,
                    kind="tile",
                    file=f"sprites/{file_name}",
                    bbox=(gx0, gy0, gx1, gy1),
                panel_bbox=(tile_x, tile_y, tile_x + TILE_SIZE, tile_y + TILE_SIZE),
                frame_count=1,
                asset_name=asset_name,
                anchor=(0, 0),
                anchor_type=anchor_type_for_panel(config.name),
                classification="tile",
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
) -> tuple[list[ExtractionItem], list[ExtractionItem], list[dict[str, Any]]]:
    panel_rgba, panel_mask = remove_background(panel_image)
    save_background_debug(config.name, panel_rgba, panel_mask, output_dirs["debug"])
    x0, y0, _, _ = config.bbox

    if config.mode == "animation_rows":
        items, sheets, groups = extract_animation_panel(
            panel_rgba,
            panel_mask,
            config,
            (x0, y0),
            output_dirs["sprites"],
            output_dirs["sheets"],
            output_dirs["debug"],
            output_dirs["debug_animations"],
        )
        return items, sheets, groups

    if config.mode == "tiles":
        items = extract_tiles_panel(
            panel_rgba,
            panel_mask,
            config,
            (x0, y0),
            output_dirs["sprites"],
            output_dirs["debug"],
        )
        return items, [], []

    items = extract_objects_panel(
        panel_rgba,
        panel_mask,
        config,
        (x0, y0),
        output_dirs["sprites"],
        output_dirs["debug"],
    )
    return items, [], []


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
    all_items: list[ExtractionItem] = []

    for config in configs:
        panel_image, panel_file = save_panel_crop(image, config, output_dirs["panels"])
        items, sheets, groups = extract_panel(panel_image, config, output_dirs)
        all_items.extend(items)
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
                        "frame_index": item.frame_index,
                        "row": item.row,
                        "row_label": item.row_label,
                        "asset_name": item.asset_name,
                        "animation_name": item.animation_name,
                        "animation_role": item.animation_role,
                        "animation_type": item.animation_type,
                        "normalized_size": list(item.normalized_size) if item.normalized_size else None,
                        "anchor": list(item.anchor) if item.anchor else None,
                        "anchor_type": item.anchor_type,
                        "classification": item.classification,
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
                        "frame_index": sheet.frame_index,
                        "row": sheet.row,
                        "row_label": sheet.row_label,
                        "asset_name": sheet.asset_name,
                        "animation_name": sheet.animation_name,
                        "animation_role": sheet.animation_role,
                        "animation_type": sheet.animation_type,
                        "normalized_size": list(sheet.normalized_size) if sheet.normalized_size else None,
                        "anchor": list(sheet.anchor) if sheet.anchor else None,
                        "anchor_type": sheet.anchor_type,
                        "classification": sheet.classification,
                        "source_columns": [list(pair) for pair in (sheet.source_columns or [])],
                    }
                    for sheet in sheets
                ],
                "groups": groups,
            }
        )

    layout_debug = draw_boxes(image[:, :, :3].copy(), "layout panels", layout_boxes)
    cv2.imwrite(str(output_dirs["debug"] / "layout_panels.png"), layout_debug)

    manifest = build_manifest(image_path, layout_path, image.shape[:2], panel_results)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    placements, atlas_infos = build_atlases(
        all_items,
        output_dirs["sprites"],
        output_dirs["atlases"],
        output_dirs["debug"],
    )
    write_engine_metadata(all_items, placements, atlas_infos, output_dirs["metadata"])

    print(f"Cropped {len(configs)} panels into {output_dirs['panels']}")
    print(f"Wrote sprites to {output_dirs['sprites']}")
    print(f"Wrote sheets to {output_dirs['sheets']}")
    print(f"Wrote atlases to {output_dirs['atlases']}")
    print(f"Wrote engine metadata to {output_dirs['metadata']}")
    print(f"Wrote debug overlays to {output_dirs['debug']}")
    print(f"Wrote manifest to {manifest_path}")


def pack_sprites_into_rows(
    sprites: list[tuple[ExtractionItem, np.ndarray]],
    max_size: int = ATLAS_MAX_SIZE,
    padding: int = ATLAS_PADDING,
) -> list[tuple[np.ndarray, list[AtlasPlacement]]]:
    atlases: list[tuple[np.ndarray, list[AtlasPlacement]]] = []
    remaining = sorted(sprites, key=lambda entry: (entry[1].shape[0], entry[1].shape[1]), reverse=True)

    current_canvas = np.zeros((max_size, max_size, 4), dtype=np.uint8)
    current_placements: list[AtlasPlacement] = []
    atlas_index = 0
    cursor_x = 0
    cursor_y = 0
    row_height = 0
    used_width = 0
    used_height = 0

    def finalize_current() -> None:
        nonlocal current_canvas, current_placements, atlas_index, cursor_x, cursor_y, row_height, used_width, used_height
        if not current_placements:
            return
        atlas_width = max(1, used_width)
        atlas_height = max(1, used_height)
        trimmed = current_canvas[:atlas_height, :atlas_width].copy()
        atlases.append((trimmed, list(current_placements)))
        atlas_index += 1
        current_canvas = np.zeros((max_size, max_size, 4), dtype=np.uint8)
        current_placements = []
        cursor_x = 0
        cursor_y = 0
        row_height = 0
        used_width = 0
        used_height = 0

    for item, image in remaining:
        height, width = image.shape[:2]
        if width > max_size or height > max_size:
            raise ValueError(f"Sprite {item.asset_name or item.file} exceeds atlas size limit of {max_size}px.")

        if cursor_x > 0 and cursor_x + width > max_size:
            cursor_x = 0
            cursor_y += row_height + padding
            row_height = 0

        if cursor_y > 0 and cursor_y + height > max_size:
            finalize_current()

        if cursor_x > 0 and cursor_x + width > max_size:
            cursor_x = 0
            cursor_y += row_height + padding
            row_height = 0

        if cursor_y + height > max_size:
            raise ValueError(f"Sprite {item.asset_name or item.file} could not be packed into an empty atlas.")

        current_canvas[cursor_y : cursor_y + height, cursor_x : cursor_x + width] = image
        current_placements.append(
            AtlasPlacement(
                sprite_name=item.asset_name or Path(item.file).stem,
                atlas_index=atlas_index,
                atlas_file=f"atlases/atlas_{atlas_index}.png",
                x=cursor_x,
                y=cursor_y,
                w=width,
                h=height,
            )
        )
        used_width = max(used_width, cursor_x + width)
        used_height = max(used_height, cursor_y + height)
        cursor_x += width + padding
        row_height = max(row_height, height)

    finalize_current()
    return atlases


def build_atlas_preview(atlas: np.ndarray, placements: list[AtlasPlacement]) -> np.ndarray:
    preview = composite_on_checker(atlas)
    for placement in placements:
        x0, y0 = placement.x, placement.y
        x1, y1 = x0 + placement.w, y0 + placement.h
        cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            preview,
            placement.sprite_name,
            (x0 + 2, min(preview.shape[0] - 4, y0 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def build_atlases(
    items: list[ExtractionItem],
    sprites_dir: Path,
    atlases_dir: Path,
    debug_dir: Path,
) -> tuple[list[AtlasPlacement], list[dict[str, Any]]]:
    sprites: list[tuple[ExtractionItem, np.ndarray]] = []
    for item in items:
        sprite_path = sprites_dir / Path(item.file).name
        image = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Missing extracted sprite for atlas packing: {sprite_path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        sprites.append((item, image))

    atlas_results = pack_sprites_into_rows(sprites)
    placements: list[AtlasPlacement] = []
    atlas_infos: list[dict[str, Any]] = []
    for atlas_index, (atlas_image, atlas_placements) in enumerate(atlas_results):
        atlas_path = atlases_dir / f"atlas_{atlas_index}.png"
        cv2.imwrite(str(atlas_path), atlas_image)
        atlas_infos.append(
            {
                "file": f"atlases/atlas_{atlas_index}.png",
                "width": int(atlas_image.shape[1]),
                "height": int(atlas_image.shape[0]),
            }
        )
        preview = build_atlas_preview(atlas_image, atlas_placements)
        cv2.imwrite(str(debug_dir / f"atlas_{atlas_index}_preview.png"), preview)
        placements.extend(atlas_placements)

    return placements, atlas_infos


def write_engine_metadata(
    items: list[ExtractionItem],
    placements: list[AtlasPlacement],
    atlas_infos: list[dict[str, Any]],
    metadata_dir: Path,
) -> None:
    placement_map = {placement.sprite_name: placement for placement in placements}
    item_map = {item.asset_name: item for item in items if item.asset_name}
    atlas_entries: dict[str, Any] = {}

    for placement in placements:
        item = item_map.get(placement.sprite_name)
        anchor = item.anchor if item and item.anchor else (0, 0)
        atlas_entries[placement.sprite_name] = {
            "atlas": placement.atlas_file,
            "x": placement.x,
            "y": placement.y,
            "w": placement.w,
            "h": placement.h,
            "anchor": {
                "x": int(anchor[0]),
                "y": int(anchor[1]),
            },
            "anchor_type": item.anchor_type if item else None,
            "classification": item.classification if item else None,
            "animation_type": item.animation_type if item else None,
        }

    animations: dict[str, dict[str, dict[str, Any]]] = {}
    for item in items:
        if not item.asset_name:
            continue
        if item.asset_name not in placement_map:
            continue
        if item.animation_role:
            entity = normalize_key(item.panel)
            animation = animations.setdefault(entity, {}).setdefault(
                item.animation_role,
                {"frames": [], "animation_type": item.animation_type or item.animation_role},
            )
            animation["frames"].append((item.frame_index if item.frame_index is not None else 0, item.asset_name))

    for entity_animations in animations.values():
        for role, animation in entity_animations.items():
            animation["frames"] = [name for _, name in sorted(animation["frames"], key=lambda entry: entry[0])]
            timing = animation_timing_for_role(role, len(animation["frames"]))
            animation.update(timing)

    atlas_payload = {
        "atlases": atlas_infos,
        "sprites": atlas_entries,
    }
    animations_payload = animations

    (metadata_dir / "atlas.json").write_text(json.dumps(atlas_payload, indent=2), encoding="utf-8")
    (metadata_dir / "animations.json").write_text(json.dumps(animations_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    process_layout_board(Path(args.image), Path(args.layout), Path(args.output))


if __name__ == "__main__":
    main()
