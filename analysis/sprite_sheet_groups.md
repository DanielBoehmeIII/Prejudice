# Sprite-sheet group segmentation

Identified groups from the supplied sheet:

- `player`: the **PLAYER SPRITES (SPIRITUAL KNIGHT)** panel in the top-left region.
- `enemies`: the **ENEMIES (CORRUPTED FORCES)** panel in the top-middle region.
- `tiles`: the **TILES (16x16)** panel in the mid-left region.

The cropping script `crop_sprite_groups.py` extracts these three labeled outputs:

- `player_group.png`
- `enemies_group.png`
- `tiles_group.png`

Reference boxes are calibrated to a 1536x1024 source image and automatically scaled if the input size differs.

## Usage

```bash
python crop_sprite_groups.py /path/to/sprite_sheet.png --output outputs
```

## Bounding boxes (1536x1024 basis)

- `player`: `(0, 0, 458, 499)`
- `enemies`: `(458, 0, 1161, 499)`
- `tiles`: `(0, 499, 392, 731)`
