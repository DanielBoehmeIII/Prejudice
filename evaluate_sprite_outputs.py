from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from crop_sprite_groups import remove_background, split_regions


MATCH_IOU_THRESHOLD = 0.5


@dataclass
class SpriteEntry:
    sprite_id: int
    file: Path
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


@dataclass
class SourceComponent:
    component_id: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


def load_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_sprite_entries(manifest: dict[str, object], manifest_dir: Path) -> list[SpriteEntry]:
    entries: list[SpriteEntry] = []
    for sprite_entry in manifest["sprites"]:
        assert isinstance(sprite_entry, dict)
        sprite_id = int(sprite_entry["id"])
        file_path = manifest_dir / str(sprite_entry["file"])
        bbox_values = sprite_entry["bbox"]
        assert isinstance(bbox_values, list)
        bbox = tuple(int(value) for value in bbox_values)

        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read sprite image: {file_path}")
        if image.ndim != 3 or image.shape[2] < 4:
            raise ValueError(f"Expected RGBA sprite image: {file_path}")

        alpha_mask = image[:, :, 3] > 0
        entries.append(SpriteEntry(sprite_id=sprite_id, file=file_path, bbox=bbox, mask=alpha_mask))

    return entries


def load_source_components(source_mask: np.ndarray) -> list[SourceComponent]:
    contours = split_regions(source_mask)
    components: list[SourceComponent] = []

    for component_id, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        local_mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
        cv2.drawContours(local_mask, [shifted], -1, 255, thickness=cv2.FILLED)
        components.append(
            SourceComponent(
                component_id=component_id,
                bbox=(x, y, x + w, y + h),
                mask=local_mask > 0,
            )
        )

    return components


def bbox_intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def overlap_pixel_count(component: SourceComponent, sprite: SpriteEntry) -> int:
    intersection = bbox_intersection(component.bbox, sprite.bbox)
    if intersection is None:
        return 0

    x0, y0, x1, y1 = intersection
    component_x0, component_y0, _, _ = component.bbox
    sprite_x0, sprite_y0, _, _ = sprite.bbox

    component_slice = component.mask[y0 - component_y0 : y1 - component_y0, x0 - component_x0 : x1 - component_x0]
    sprite_slice = sprite.mask[y0 - sprite_y0 : y1 - sprite_y0, x0 - sprite_x0 : x1 - sprite_x0]
    return int(np.count_nonzero(component_slice & sprite_slice))


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    intersection = bbox_intersection(a, b)
    if intersection is None:
        return 0.0

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    area_i = max(0, intersection[2] - intersection[0]) * max(0, intersection[3] - intersection[1])
    area_u = area_a + area_b - area_i
    if area_u <= 0:
        return 0.0
    return area_i / float(area_u)


def project_sprites_to_canvas(
    sprite_entries: list[SpriteEntry], canvas_shape: tuple[int, int]
) -> np.ndarray:
    canvas = np.zeros(canvas_shape, dtype=bool)
    canvas_h, canvas_w = canvas_shape

    for sprite in sprite_entries:
        x0, y0, x1, y1 = sprite.bbox
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(canvas_w, x1)
        y1 = min(canvas_h, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        target = canvas[y0:y1, x0:x1]
        sprite_mask = sprite.mask[: y1 - y0, : x1 - x0]
        target |= sprite_mask

    return canvas


def compute_binary_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))

    precision = true_positive / float(true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / float(true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = true_positive / float(true_positive + false_positive + false_negative) if (
        true_positive + false_positive + false_negative
    ) else 0.0

    return {
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
    }


def write_diff_visualization(predicted: np.ndarray, expected: np.ndarray, output_path: Path) -> None:
    image = np.zeros((expected.shape[0], expected.shape[1], 3), dtype=np.uint8)
    image[expected & predicted] = (255, 255, 255)
    image[expected & ~predicted] = (0, 0, 255)
    image[predicted & ~expected] = (255, 0, 0)
    cv2.imwrite(str(output_path), image)


def evaluate_individual_sprites(
    sprite_entries: list[SpriteEntry], source_components: list[SourceComponent], source_mask: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    reconstructed = project_sprites_to_canvas(sprite_entries, source_mask.shape)
    pixel_metrics = compute_binary_metrics(reconstructed, source_mask)

    component_overlap_counts: list[int] = []
    matched_component_count = 0
    fragmented_component_count = 0

    for component in source_components:
        overlaps = [sprite for sprite in sprite_entries if overlap_pixel_count(component, sprite) > 0]
        overlap_count = len(overlaps)
        component_overlap_counts.append(overlap_count)
        if overlap_count > 0:
            matched_component_count += 1
        if overlap_count > 1:
            fragmented_component_count += 1

    merged_sprite_count = 0
    false_positive_sprite_count = 0
    best_iou_scores: list[float] = []
    matched_by_iou = 0

    for sprite in sprite_entries:
        overlapping_components = [component for component in source_components if overlap_pixel_count(component, sprite) > 0]
        if not overlapping_components:
            false_positive_sprite_count += 1
            best_iou_scores.append(0.0)
            continue

        if len(overlapping_components) > 1:
            merged_sprite_count += 1

        best_iou = max(bbox_iou(component.bbox, sprite.bbox) for component in overlapping_components)
        best_iou_scores.append(best_iou)
        if best_iou >= MATCH_IOU_THRESHOLD:
            matched_by_iou += 1

    metrics: dict[str, object] = {
        "generated_sprite_count": len(sprite_entries),
        "source_component_count": len(source_components),
        "pixel_metrics": pixel_metrics,
        "matched_component_count": matched_component_count,
        "component_match_rate": round(
            matched_component_count / float(len(source_components)) if source_components else 0.0,
            4,
        ),
        "missed_component_count": sum(1 for count in component_overlap_counts if count == 0),
        "fragmented_component_count": fragmented_component_count,
        "merged_sprite_count": merged_sprite_count,
        "false_positive_sprite_count": false_positive_sprite_count,
        "bbox_iou_match_count": matched_by_iou,
        "bbox_iou_match_rate": round(matched_by_iou / float(len(sprite_entries)) if sprite_entries else 0.0, 4),
        "average_best_bbox_iou": round(float(np.mean(best_iou_scores)) if best_iou_scores else 0.0, 4),
    }

    return metrics, reconstructed


def evaluate_groups(
    manifest: dict[str, object],
    sprite_lookup: dict[int, SpriteEntry],
    source_mask: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    groups = manifest["groups"]
    assert isinstance(groups, list)

    all_grouped_sprite_ids: list[int] = []
    grouped_sprite_ids: set[int] = set()
    per_group: list[dict[str, object]] = []
    grouped_entries: list[SpriteEntry] = []

    for group_entry in groups:
        assert isinstance(group_entry, dict)
        sprite_ids = [int(sprite_id) for sprite_id in group_entry["sprite_ids"]]
        all_grouped_sprite_ids.extend(sprite_ids)
        grouped_sprite_ids.update(sprite_ids)
        entries = [sprite_lookup[sprite_id] for sprite_id in sprite_ids]
        grouped_entries.extend(entries)

        source_bbox_values = group_entry["source_bbox"]
        assert isinstance(source_bbox_values, list)
        x0, y0, x1, y1 = (int(value) for value in source_bbox_values)

        predicted = project_sprites_to_canvas(entries, source_mask.shape)[y0:y1, x0:x1]
        expected = source_mask[y0:y1, x0:x1]
        pixel_metrics = compute_binary_metrics(predicted, expected)

        rows = group_entry["rows"]
        assert isinstance(rows, list)
        row_count = len(rows)
        column_count = max((len(row) for row in rows), default=0)
        occupancy = len(sprite_ids) / float(max(1, row_count * column_count))

        per_group.append(
            {
                "id": int(group_entry["id"]),
                "file": group_entry["file"],
                "sprite_count": len(sprite_ids),
                "source_bbox": [x0, y0, x1, y1],
                "row_count": row_count,
                "column_count": column_count,
                "grid_occupancy": round(occupancy, 4),
                "pixel_metrics": pixel_metrics,
            }
        )

    grouped_union = project_sprites_to_canvas(grouped_entries, source_mask.shape)
    all_sprite_ids = {int(sprite_entry["id"]) for sprite_entry in manifest["sprites"]}
    ungrouped_sprite_ids = sorted(all_sprite_ids - grouped_sprite_ids)
    grouped_sprite_id_counts: dict[int, int] = {}
    for sprite_id in all_grouped_sprite_ids:
        grouped_sprite_id_counts[sprite_id] = grouped_sprite_id_counts.get(sprite_id, 0) + 1
    duplicate_grouped_sprite_ids = sorted(
        sprite_id for sprite_id, count in grouped_sprite_id_counts.items() if count > 1
    )
    grouping_integrity_pass = len(ungrouped_sprite_ids) == 0 and len(duplicate_grouped_sprite_ids) == 0
    metrics: dict[str, object] = {
        "group_count": len(groups),
        "grouped_sprite_count": len(grouped_sprite_ids),
        "grouped_sprite_ratio": round(
            len(grouped_sprite_ids) / float(len(manifest["sprites"])) if manifest["sprites"] else 0.0,
            4,
        ),
        "ungrouped_sprite_count": len(ungrouped_sprite_ids),
        "ungrouped_sprite_ids": ungrouped_sprite_ids,
        "duplicate_grouped_sprite_count": len(duplicate_grouped_sprite_ids),
        "duplicate_grouped_sprite_ids": duplicate_grouped_sprite_ids,
        "all_sprites_grouped": len(ungrouped_sprite_ids) == 0,
        "grouping_integrity_pass": grouping_integrity_pass,
        "pixel_metrics": compute_binary_metrics(grouped_union, source_mask),
        "per_group": per_group,
    }
    return metrics, grouped_union


def write_markdown_summary(report: dict[str, object], output_path: Path) -> None:
    individual = report["individual"]
    assert isinstance(individual, dict)
    groups = report["groups"]
    assert isinstance(groups, dict)
    individual_pixels = individual["pixel_metrics"]
    assert isinstance(individual_pixels, dict)
    grouped_pixels = groups["pixel_metrics"]
    assert isinstance(grouped_pixels, dict)

    lines = [
        "# Sprite Generation Evaluation",
        "",
        "## Individual sprites",
        "",
        f"- Generated sprites: {individual['generated_sprite_count']}",
        f"- Source components: {individual['source_component_count']}",
        f"- Pixel precision / recall / F1: {individual_pixels['precision']} / {individual_pixels['recall']} / {individual_pixels['f1']}",
        f"- Component match rate: {individual['component_match_rate']}",
        f"- Missed components: {individual['missed_component_count']}",
        f"- Fragmented components: {individual['fragmented_component_count']}",
        f"- Merged sprites: {individual['merged_sprite_count']}",
        f"- False-positive sprites: {individual['false_positive_sprite_count']}",
        f"- Average best bbox IoU: {individual['average_best_bbox_iou']}",
        "",
        "## Grouped sheets",
        "",
        f"- Generated groups: {groups['group_count']}",
        f"- Grouping integrity pass: {groups['grouping_integrity_pass']}",
        f"- Grouped sprite ratio: {groups['grouped_sprite_ratio']}",
        f"- All sprites grouped: {groups['all_sprites_grouped']}",
        f"- Ungrouped sprites: {groups['ungrouped_sprite_count']}",
        f"- Duplicate grouped sprites: {groups['duplicate_grouped_sprite_count']}",
        f"- Pixel precision / recall / F1: {grouped_pixels['precision']} / {grouped_pixels['recall']} / {grouped_pixels['f1']}",
        "",
        "## Per-group detail",
        "",
    ]

    per_group = groups["per_group"]
    assert isinstance(per_group, list)
    if not per_group:
        lines.append("- No grouped sheets were produced.")
    else:
        for group in per_group:
            assert isinstance(group, dict)
            pixels = group["pixel_metrics"]
            assert isinstance(pixels, dict)
            lines.append(
                f"- Group {group['id']}: sprites={group['sprite_count']}, grid={group['row_count']}x{group['column_count']}, "
                f"occupancy={group['grid_occupancy']}, precision={pixels['precision']}, recall={pixels['recall']}, f1={pixels['f1']}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate extracted sprites and grouped sprite sheets against the original source image."
    )
    parser.add_argument("image", help="Path to the original source image.")
    parser.add_argument(
        "--manifest",
        default="outputs/manifest.json",
        help="Path to the generation manifest produced by crop_sprite_groups.py.",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation",
        help="Directory for evaluation reports and diff visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    sprite_entries = load_sprite_entries(manifest, manifest_path.parent)
    sprite_lookup = {entry.sprite_id: entry for entry in sprite_entries}

    source_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if source_image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")
    _, source_mask = remove_background(source_image)
    source_mask = source_mask > 0
    source_components = load_source_components((source_mask.astype(np.uint8) * 255))

    individual_metrics, individual_union = evaluate_individual_sprites(sprite_entries, source_components, source_mask)
    group_metrics, grouped_union = evaluate_groups(manifest, sprite_lookup, source_mask)

    report = {
        "source_image": str(image_path),
        "manifest": str(manifest_path),
        "individual": individual_metrics,
        "groups": group_metrics,
    }

    report_path = output_dir / "report.json"
    summary_path = output_dir / "summary.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_summary(report, summary_path)
    write_diff_visualization(individual_union, source_mask, output_dir / "individual_diff.png")
    write_diff_visualization(grouped_union, source_mask, output_dir / "grouped_diff.png")

    print(f"Wrote JSON report to {report_path}")
    print(f"Wrote Markdown summary to {summary_path}")
    print(f"Wrote diff visualizations to {output_dir}")


if __name__ == "__main__":
    main()
