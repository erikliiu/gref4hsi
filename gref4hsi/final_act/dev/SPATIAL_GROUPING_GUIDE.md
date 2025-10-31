# Spatial Grouping for Grouped Cross-Validation

## Problem Solved
Previously, grouped CV used **class names as groups** (e.g., all "training_bombs" pixels in one group). This meant:
- ❌ If you have 2 bombs at different locations, both get treated as the same group
- ❌ CV can't leave out "bomb #1" while training on "bomb #2"
- ❌ Spatial autocorrelation within each bomb still causes leakage

## Solution: Spatial Clustering
Now, each ROI class is **spatially clustered into separate groups**:
- ✅ `training_bombs` → `bomb#1`, `bomb#2`, `bomb#3`, ... (one per spatially distinct region)
- ✅ `training_dark` → `dark#A`, `dark#B`, `dark#C`, ...
- ✅ `training_sediment` → `sed#01`, `sed#02`, `sed#03`, ...

CV now splits by these **spatial groups**, so:
- Fold 1: Train on [bomb#1, dark#A, sed#01], Validate on [bomb#2, dark#B, sed#02]
- No pixels from bomb#2 appear in training when bomb#2 is being validated

## Implementation Details

### 1. Spatial Clustering (`_create_spatial_groups`)
**Method 1 (Preferred): Connected Components**
```python
- Create binary mask from ROI pixels
- Apply morphological closing (radius=3 px) to merge nearby pixels
- Use 8-connectivity to find connected components
- Each component = one spatial group
```

**Method 2 (Fallback): DBSCAN**
```python
- If connected components fails
- DBSCAN with eps=10 pixels, min_samples=20
- Clusters nearby pixels into groups
```

**Post-processing:**
- Drop groups with <20 pixels
- Merge small groups to nearest large group of same class
- Name groups: `bomb#1`, `dark#A`, `sed#01`, etc.

### 2. New Parameters in `train_svm_with_cv`

```python
cube.train_svm_with_cv(
    training_rois=training_rois,
    segment_start=...,
    segment_end=...,
    
    # 🔥 NEW PARAMETERS
    use_spatial_groups=True,        # Enable spatial clustering (default: True)
    closing_radius=3,                # Morphological closing radius (default: 3)
    min_group_size=20,               # Minimum pixels per group (default: 20)
    subsample_per_group=None,        # Max pixels per group (None = no limit)
    use_sample_weights=False,        # Weight by 1/group_size (default: False)
    
    # Existing parameters...
    wavelength_range=(490, 680),
    cv_folds=5,
    ...
)
```

### 3. Subsampling Per Group (Optional)
**Problem**: Giant sediment region (5000 pixels) dominates training over small bombs (50 pixels)

**Solution**:
```python
subsample_per_group=500  # Cap each group to max 500 pixels
```
- Randomly samples 500 pixels from each group that has >500
- Prevents large groups from dominating
- All groups get equal representation

### 4. Sample Weighting (Optional)
**Alternative approach**:
```python
use_sample_weights=True  # Weight each pixel by 1/group_size
```
- Bomb pixels get weight = 1/50 = 0.02
- Sediment pixels get weight = 1/5000 = 0.0002
- SVM sees bombs as more important during training
- Use this OR subsampling, not both

### 5. Enhanced Logging

Each fold now prints:
```
📂 Fold 1/3:
   Train groups: ['bomb#2', 'dark#A', 'dark#B', 'sed#01', 'sed#02']
   Val groups:   ['bomb#1', 'dark#C', 'sed#03']
   ✅ No overlap (clean split)
   Train samples: {'training_bombs': 150, 'training_dark': 400, 'training_sediment': 1200}
   Val samples:   {'training_bombs': 75, 'training_dark': 180, 'training_sediment': 600}
```

**Critical checks:**
- ✅ **No overlap**: train_groups ∩ val_groups = ∅
- ✅ **All classes present**: All 3 classes appear in validation
- ⚠️  **Missing classes**: Warns if a class is missing in validation fold

## Expected Behavior

### Before (Simple Grouping by Class)
```
📂 Fold 1/3:
   Train groups: ['training_dark', 'training_sediment']
   Val groups:   ['training_bombs']
```
**Problem**: All bombs lumped together, can't do LOBO (Leave-One-Bomb-Out)

### After (Spatial Grouping)
```
📂 Fold 1/3:
   Train groups: ['bomb#2', 'dark#A', 'sed#01', 'sed#02']
   Val groups:   ['bomb#1', 'dark#B', 'sed#03']
```
**Benefit**: True Leave-One-Bomb-Out CV!

## Usage Examples

### Basic Usage (Default Settings)
```python
cv_results = cube.train_svm_with_cv(
    training_rois=training_rois,
    segment_start=config.UHI_TRACK_RANGE_5[0],
    segment_end=config.UHI_TRACK_RANGE_5[1],
    wavelength_range=(490, 680),
    cv_folds=5,
    use_corrected=True,
    optimize_params=True,
    quiet=False,
)
```
- ✅ Spatial grouping enabled by default
- ✅ Creates bomb#1, bomb#2, dark#A, sed#01, etc.
- ✅ Proper grouped CV with overlap verification

### With Subsampling (Prevent Dominance)
```python
cv_results = cube.train_svm_with_cv(
    training_rois=training_rois,
    segment_start=config.UHI_TRACK_RANGE_5[0],
    segment_end=config.UHI_TRACK_RANGE_5[1],
    wavelength_range=(490, 680),
    subsample_per_group=500,  # 🔥 Cap each group to 500 pixels
    use_corrected=True,
    optimize_params=True,
)
```
- Large sediment patches won't dominate
- All groups contribute equally

### With Sample Weighting
```python
cv_results = cube.train_svm_with_cv(
    training_rois=training_rois,
    segment_start=config.UHI_TRACK_RANGE_5[0],
    segment_end=config.UHI_TRACK_RANGE_5[1],
    wavelength_range=(490, 680),
    use_sample_weights=True,  # 🔥 Weight by 1/group_size
    use_corrected=True,
    optimize_params=True,
)
```
- Small groups get higher weight per pixel
- Alternative to subsampling

## What to Expect

### Output During Training
```
🔬 Creating spatial groups for CV...
   Settings: closing_radius=3, min_group_size=20
   training_bombs: 201 pixels → 2 groups via connected_components
      bomb#1: 98 pixels
      bomb#2: 103 pixels
   training_dark: 364 pixels → 3 groups via connected_components
      dark#A: 145 pixels
      dark#B: 112 pixels
      dark#C: 107 pixels
   training_sediment: 397 pixels → 2 groups via connected_components
      sed#01: 234 pixels
      sed#02: 163 pixels

📊 Final training data:
   Classes: ['training_bombs', 'training_dark', 'training_sediment']
   Total pixels: 962
   Unique spatial groups: 7
   Group names: ['bomb#1', 'bomb#2', 'dark#A', 'dark#B', 'dark#C', 'sed#01', 'sed#02']

⚠️  WARNING: Requested 5 folds but only 7 ROI groups available
   Reducing to 7-fold CV (Leave-One-Group-Out)
```

### CV Results
- **Folds**: Automatically adjusted to number of groups (7-fold for 7 groups)
- **Scores**: Likely lower than before (more honest evaluation)
- **Variability**: Higher ±std (fewer samples per fold)

## Scientific Benefits

✅ **No spatial leakage**: Nearby pixels never in both train and val
✅ **True LOBO**: Can evaluate on completely unseen bombs/pits
✅ **Publishable**: Standard method for spatial data
✅ **Realistic**: CV scores reflect real-world generalization
✅ **Flexible**: Control group dominance via subsampling or weighting

## Troubleshooting

**"Only 1 class in fold"**
- You need more spatial groups
- Tighten your ROIs (more distinct regions)
- Or reduce cv_folds

**"Groups too small (< 20 pixels)"**
- Increase `min_group_size` to merge more
- Or label larger ROI regions

**"Giant sediment dominates"**
- Use `subsample_per_group=500`
- Or `use_sample_weights=True`

**"Scores dropped a lot"**
- **This is expected and GOOD**
- Previous scores were inflated by leakage
- New scores are honest estimates
