# Sediment Grid Grouping Implementation

## Problem
Connected components grouping treats large continuous sediment patches as ONE giant group, preventing proper cross-validation. With only 1 sediment group, CV can't split it across folds.

## Solution
**Grid-based grouping for sediment only**, while keeping connected components for bombs/dark.

## Implementation Details

### New Parameters in `train_svm_with_cv()`

```python
cube.train_svm_with_cv(
    training_rois=training_rois,
    ...
    # 🔥 NEW: Sediment grid parameters
    use_grid_for_sediment=True,        # Enable grid for sediment (default: True)
    sediment_grid_tile_slit=100,       # Grid tile: 100 pixels in slit direction
    sediment_grid_tile_track=200,      # Grid tile: 200 pixels in track direction
    sediment_min_group_size=10,        # Min pixels per tile (merge if less)
    
    # Existing: For bombs/dark only
    closing_radius=3,                  # Morphological closing (bombs/dark)
    min_group_size=20,                 # Min pixels per group (bombs/dark)
    ...
)
```

### How It Works

#### For `training_sediment` (Grid Method):
1. **Divide into tiles**: 100×200 pixel grid across the datacube
2. **Assign pixels to tiles**: Each sediment pixel → grid tile based on (track//200, slit//100)
3. **Filter small tiles**: Tiles with <10 pixels are merged to neighboring tiles
4. **Name sequentially**: `sed#01`, `sed#02`, `sed#03`, ...

#### For `training_bombs` and `training_dark` (Connected Components):
- Uses morphological closing + 8-connectivity (unchanged)
- Creates `bomb#1`, `bomb#2`, `dark#A`, `dark#B`, etc.

### Grid Tile Merging Logic

Small tiles (<10 pixels) are merged:
1. **First**: Try merging to an 8-connected neighbor tile
2. **If no neighbors**: Merge to nearest large tile (by centroid distance)
3. **Result**: No isolated tiny tiles

### Expected Output

**Before (Connected Components for All)**:
```
training_sediment: 5000 pixels → 1 group via connected_components
   sed#01: 5000 pixels  ❌ Can't do CV with 1 group!
```

**After (Grid for Sediment)**:
```
training_sediment: 5000 pixels → 35 groups via grid
   sed#01: 156 pixels
   sed#02: 143 pixels
   sed#03: 178 pixels
   ...
   sed#35: 134 pixels  ✅ Many groups for proper CV!
```

## Advantages

1. **Multiple sediment groups**: Instead of 1 giant group, get 20-50 smaller groups
2. **Better CV splits**: Can now do 5-fold, 10-fold CV with sediment
3. **Spatial diversity**: Each fold tests different spatial regions
4. **Consistent tiles**: Grid ensures reproducible grouping

## Grid Tile Size Selection

### Current Settings:
- **Slit direction**: 100 pixels
- **Track direction**: 200 pixels
- **Rationale**: Datacube is typically ~970 slits × ~2400 tracks
  - 100px slit → ~10 tiles across width
  - 200px track → ~12 tiles along track
  - Total: ~120 potential tiles

### Tuning Guidelines:
- **Larger tiles** → Fewer groups → Faster CV but less spatial diversity
- **Smaller tiles** → More groups → Slower CV but better generalization test
- **Min group size** (10px) → Filters noise, merges sparse regions

## Test Results

```bash
python test_sediment_grid_grouping.py
```

**Test validates**:
- ✅ Sediment uses grid (42 groups created)
- ✅ Bombs use connected components (2 groups)
- ✅ Dark uses connected components (1 group)
- ✅ Small tiles merged correctly

## Usage Example

```python
# Train with sediment grid grouping (default)
cv_results = cube.train_svm_with_cv(
    training_rois=["training_dark", "training_sediment", "training_bombs"],
    segment_start=config.UHI_TRACK_RANGE_5[0],
    segment_end=config.UHI_TRACK_RANGE_5[1],
    wavelength_range=(490, 680),
    cv_folds=10,  # ✅ Now possible with many sediment groups!
    use_corrected=True,
    optimize_params=True,
    
    # Grid settings (all default, can be omitted)
    use_grid_for_sediment=True,
    sediment_grid_tile_slit=100,
    sediment_grid_tile_track=200,
    sediment_min_group_size=10,
)
```

## Expected CV Behavior

### With Grid Grouping:
```
📂 Fold 1/10:
   Train groups: ['bomb#1', 'dark#A', 'sed#02', 'sed#03', ..., 'sed#35']
   Val groups:   ['bomb#2', 'sed#01']
   ✅ No overlap (clean split)
```

- Sediment tiles distributed across folds
- True spatial cross-validation
- Each fold validates different sediment regions

### Without Grid (Old Behavior):
```
⚠️  WARNING: Requested 10 folds but only 3 ROI groups available
   Reducing to 3-fold CV (Leave-One-ROI-Out)

📂 Fold 1/3:
   Train groups: ['training_dark', 'training_bombs']
   Val groups:   ['training_sediment']  ❌ All sediment or none!
```

## Troubleshooting

**"Too many sediment groups (>50)"**
- Increase `sediment_grid_tile_slit` or `sediment_grid_tile_track`
- Example: Use 150×300 for coarser grid

**"Too few sediment groups (<5)"**
- Decrease tile sizes for finer grid
- Example: Use 50×100 for finer grid
- Check if sediment ROI is very sparse

**"CV still slow"**
- Use `subsample_per_group=500` to cap pixels per tile
- Or reduce number of CV folds

## Technical Notes

- Grid is calculated as: `(track_idx // tile_size_track, slit_idx // tile_size_slit)`
- Only triggers for exact match: `class_name == "training_sediment"`
- Other sediment-named ROIs (e.g., "sediment_test") will use connected components
- Grid grouping is **deterministic** (same pixels → same tile every time)
