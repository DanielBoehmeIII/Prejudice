# Prejudice

Automatic sprite extraction lives in [crop_sprite_groups.py](/home/dboehmeiii/Desktop/repos/Prejudice/crop_sprite_groups.py), and output evaluation lives in [evaluate_sprite_outputs.py](/home/dboehmeiii/Desktop/repos/Prejudice/evaluate_sprite_outputs.py).

## Generate Outputs

```bash
python3 crop_sprite_groups.py sprite-sheet01.png --output outputs
```

This writes:

- `outputs/sprites/*.png`: individual extracted sprites
- `outputs/sheets/*.png`: grouped sprite sheets covering every extracted sprite
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
