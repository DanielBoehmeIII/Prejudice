# Prejudice

Automatic sprite extraction lives in [crop_sprite_groups.py](/home/dboehmeiii/Desktop/repos/Prejudice/crop_sprite_groups.py), and output evaluation lives in [evaluate_sprite_outputs.py](/home/dboehmeiii/Desktop/repos/Prejudice/evaluate_sprite_outputs.py).

## Generate Outputs

```bash
python3 crop_sprite_groups.py sprite-sheet01.png --output outputs
```

This writes:

- `outputs/sprites/*.png`: engine-facing extracted sprites with automatic naming
- `outputs/sheets/*.png`: normalized animation strips grouped by detected row animation
- `outputs/atlases/atlas_*.png`: packed texture atlases capped at `1024x1024`
- `outputs/metadata/atlas.json`: atlas coordinates, per-sprite anchor metadata, and classification tags
- `outputs/metadata/animations.json`: detected animation frame lists plus frame timing metadata
- `outputs/debug/atlas_*_preview.png`: atlas previews with bounding boxes
- `outputs/debug/animations/*_preview.png`: animation preview strips with shared center and anchor overlays
- `outputs/manifest.json`: source-coordinate metadata for every generated sprite and group

## Evaluate Outputs

```bash
python3 evaluate_sprite_outputs.py sprite-sheet01.png --manifest outputs/manifest.json --output outputs/evaluation
```

The evaluator measures success in two ways:

- Individual sprite success: reconstructs all extracted sprites back into source coordinates and scores pixel precision, recall, F1, missed components, fragmentation, merges, and false positives against the original image foreground.
- Group sprite sheet success: verifies that every extracted sprite is assigned to a grouped sheet exactly once, then scores grouped coverage globally and per group using source-crop pixel metrics plus grid occupancy.

Generated evaluation artifacts:

- `outputs/evaluation/report.json`
- `outputs/evaluation/summary.md`
- `outputs/evaluation/individual_diff.png`
- `outputs/evaluation/grouped_diff.png`

## Demo Loop

The repo now includes a small browser demo in [demo/index.html](/home/dboehmeiii/Desktop/repos/Prejudice/demo/index.html) that drives a simple side-view game loop from the generated metadata in `outputs/metadata/atlas.json` and `outputs/metadata/animations.json`.

Run a static server from the repo root, then open `/demo/`:

```bash
python3 -m http.server 8000
```

Controls:

- `A` / `D`: move
- `Shift`: run
- `Space`: jump
- `Z`: attack animation
- `X`: aura animation
- `C`: light beam animation
