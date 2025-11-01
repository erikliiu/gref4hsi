# Post-Classification Connected-Component Filtering

## Problem: Isolated Pixels (Salt-and-Pepper Noise)

After per-pixel SVM classification, you often get:
- **Single isolated pixels** misclassified in the wrong class
- **Small noisy clusters** (1-5 pixels) scattered throughout
- **"Salt-and-pepper" noise** that makes visualization messy

### Example
```
Before filtering:
  B B B B B . . . .     B = bomb
  B B B B B . d . .     d = dark (isolated!)
  B B B B B . . . .     s = sediment
  . . . . . s s s s
  . . . b . s s s s     b = bomb (isolated!)

After filtering (min_area=20):
  B B B B B . . . .     Removed: isolated 'd' (1 pixel)
  B B B B B . . . .     Removed: isolated 'b' (1 pixel)
  B B B B B . . . .     Kept: large 'B' cluster (15 pixels)
  . . . . . s s s s     Kept: large 's' cluster (8 pixels)
  . . . . . s s s s
```

---

## Solution: Connected-Component Filtering

Post-classification filtering applies **morphological operations** and **connected-component analysis** to:
1. Fill small holes within classified regions (closing)
2. Remove small protrusions from regions (opening)
3. **Remove isolated components** smaller than minimum area
4. (Optional) Merge nearby components

### Not K-means Clustering!
This is **NOT** k-means. It's **connected-component labeling** on binary masks:
- Each class is processed separately
- Uses spatial connectivity (8-connectivity or 4-connectivity)
- Deterministic (same input → same output)

---

## Parameters

### Core Parameters

```python
cube.classify_segment_with_validation(
    ...
    # Enable/disable filtering
    apply_post_filtering=True,         # Master switch (default: True)
    
    # Per-class filtering (enable for classes with noise)
    post_cc_bomb=True,                 # Filter bombs (default: True)
    post_cc_dark=True,                 # Filter dark spots (default: True)
    post_cc_sediment=False,            # Don't filter sediment (default: False)
    
    # Connected-component parameters
    cc_connectivity=8,                 # 8 or 4 connectivity (default: 8)
    cc_min_area=20,                    # Minimum pixels per component (default: 20)
    
    # Morphological operations
    morph_close_radius=1,              # Closing radius (0=disable, default: 1)
    morph_open_radius=1,               # Opening radius (0=disable, default: 1)
    merge_proximity_px=0,              # Merge distance (0=disable, default: 0)
)
```

### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_post_filtering` | bool | `True` | Master switch - disable to skip all filtering |
| `post_cc_bomb` | bool | `True` | Apply filtering to bomb class |
| `post_cc_dark` | bool | `True` | Apply filtering to dark class |
| `post_cc_sediment` | bool | `False` | Apply filtering to sediment class |
| `cc_connectivity` | int | `8` | Connectivity (4 or 8) - 8 includes diagonals |
| `cc_min_area` | int | `20` | Minimum component size (pixels) |
| `morph_close_radius` | int | `1` | Closing radius (0 to disable) |
| `morph_open_radius` | int | `1` | Opening radius (0 to disable) |
| `merge_proximity_px` | int | `0` | Merge components within distance (0 to disable) |

---

## Algorithm Steps

For each enabled class (e.g., `training_bombs`):

### 1. Create Binary Mask
```python
binary_mask = (classification_map == "training_bombs")
# Result: 1 where bomb, 0 elsewhere
```

### 2. Morphological Closing (Fill Holes)
If `morph_close_radius > 0`:
```python
# Fills small holes INSIDE bomb regions
# Useful for connecting nearby pixels
closed_mask = binary_closing(binary_mask, disk(morph_close_radius))
```

**Before closing:**
```
B B . B B    (2-pixel hole)
B B . B B
```

**After closing (radius=1):**
```
B B B B B    (hole filled)
B B B B B
```

### 3. Morphological Opening (Remove Protrusions)
If `morph_open_radius > 0`:
```python
# Removes small protrusions FROM bomb regions
# Smooths boundaries
opened_mask = binary_opening(closed_mask, disk(morph_open_radius))
```

**Before opening:**
```
B B B B B
B B B B B
B . . . .    (1-pixel protrusion)
```

**After opening (radius=1):**
```
B B B B B
B B B B B
. . . . .    (protrusion removed)
```

### 4. Label Connected Components
```python
# Label each connected region with unique ID
labeled_mask, n_components = label(binary_mask, structure=8-connectivity)
# labeled_mask: [0, 1, 1, 1, 0, 2, 2, 0, ...] (0=background, 1=component1, 2=component2)
```

**Connectivity:**
- **8-connectivity**: Diagonal neighbors count (more connected)
- **4-connectivity**: Only horizontal/vertical neighbors (more isolated components)

```
8-connectivity:           4-connectivity:
  B . B  → 1 component      B . B  → 2 components
  . B .                     . B .
```

### 5. Filter by Area
```python
for component_id in range(1, n_components + 1):
    component_mask = (labeled_mask == component_id)
    area = np.sum(component_mask)
    
    if area >= cc_min_area:
        keep_component()  # Large enough
    else:
        remove_component()  # Too small (isolated noise)
```

**Example (min_area=20):**
```
Component #1: 150 pixels  ✅ Keep
Component #2: 5 pixels    ❌ Remove (isolated noise)
Component #3: 80 pixels   ✅ Keep
Component #4: 1 pixel     ❌ Remove (single stray)
```

### 6. Merge Nearby Components (Optional)
If `merge_proximity_px > 0`:
```python
# Dilate each component by merge_proximity_px
# Merge overlapping dilated components
dilated = binary_dilation(cleaned_mask, disk(merge_proximity_px))
merged_labeled = label(dilated)
```

**Before merging:**
```
B B B . . d d d    (2 separate components, 2px apart)
B B B . . d d d
```

**After merging (proximity=2):**
```
B B B B B d d d    (merged into 1 component)
B B B B B d d d
```

### 7. Write Back to Classification Map
```python
# Clear old pixels
classification_map[classification_map == "training_bombs"] = ""

# Add cleaned pixels
classification_map[cleaned_mask] = "training_bombs"
```

---

## Usage Examples

### Example 1: Standard Filtering (Remove Isolated Pixels)

**Goal:** Remove single stray pixels and small clusters

```python
classification_results = cube.classify_segment_with_validation(
    ...
    apply_post_filtering=True,
    post_cc_bomb=True,
    post_cc_dark=True,
    post_cc_sediment=False,  # Keep all sediment (large regions)
    cc_connectivity=8,
    cc_min_area=20,           # Remove components < 20 pixels
    morph_close_radius=1,     # Fill 1-pixel holes
    morph_open_radius=1,      # Remove 1-pixel protrusions
    merge_proximity_px=0,     # Don't merge
)
```

**Expected output:**
```
🧹 POST-CLASSIFICATION FILTERING
   Classes to filter: ['training_bombs', 'training_dark']
   
   🔍 Filtering training_bombs (bomb)...
      Found 5 connected components
      Kept: 2 components (450 pixels)
      Removed: 3 components (8 pixels)
      Final: 450 pixels (8 removed, 1.7%)
```

### Example 2: Aggressive Filtering (Large Regions Only)

**Goal:** Only keep large, confident regions

```python
classification_results = cube.classify_segment_with_validation(
    ...
    cc_min_area=50,           # Remove components < 50 pixels
    morph_close_radius=2,     # Fill larger holes
    morph_open_radius=2,      # More aggressive smoothing
)
```

### Example 3: Conservative Filtering (Keep Small Features)

**Goal:** Remove only single stray pixels

```python
classification_results = cube.classify_segment_with_validation(
    ...
    cc_min_area=5,            # Keep components ≥ 5 pixels
    morph_close_radius=0,     # No closing
    morph_open_radius=0,      # No opening
)
```

### Example 4: Merge Nearby Regions

**Goal:** Connect nearby bomb clusters (e.g., bomb fragments)

```python
classification_results = cube.classify_segment_with_validation(
    ...
    cc_min_area=10,
    merge_proximity_px=3,     # Merge components within 3 pixels
)
```

**Before merging:**
```
B B B . . . d d    (3 separate bomb components)
. . . . . . d d
B B . . . . . .
B B . . d d d .
```

**After merging (proximity=3):**
```
B B B B B B d d    (merged into 1 large bomb component)
B B B . . . d d
B B B . . . . .
B B B . d d d .
```

### Example 5: Disable Filtering

**Goal:** See raw SVM predictions (debugging)

```python
classification_results = cube.classify_segment_with_validation(
    ...
    apply_post_filtering=False,  # Disable all filtering
)
```

---

## When to Use Each Parameter

### Enable Filtering For
- ✅ **Bombs**: Often scattered, benefit from removing isolated pixels
- ✅ **Dark spots**: Typically discrete, remove salt-and-pepper noise
- ❌ **Sediment**: Large continuous regions, filtering may remove edges

### Connectivity
- **8-connectivity (default)**: More connected → fewer components → more filtering
- **4-connectivity**: More isolated → more components → less filtering
- **Use 8** for typical cases (diagonal neighbors should count)

### Minimum Area
- **Small (5-10 px)**: Conservative - keeps small features but some noise
- **Medium (20-30 px)**: Standard - good balance for most cases
- **Large (50+ px)**: Aggressive - only keeps large confident regions

**Tuning guidelines:**
- If seeing **too many isolated pixels** → Increase `cc_min_area`
- If **losing real small features** → Decrease `cc_min_area`

### Morphological Operations
- **Closing (fill holes)**: Use `radius=1` to fill 1-2 pixel gaps
- **Opening (remove protrusions)**: Use `radius=1` to smooth boundaries
- **Disable both (radius=0)**: Only use area filtering, no morphology

### Merge Proximity
- **0 (default)**: Don't merge - each component independent
- **2-5 pixels**: Merge nearby clusters (e.g., bomb fragments)
- **>5 pixels**: Aggressive merging - may merge unrelated regions

---

## Output Interpretation

### Filtering Output

```
🧹 POST-CLASSIFICATION FILTERING
   Classes to filter: ['training_bombs', 'training_dark']
   Connectivity: 8
   Min area: 20 pixels
   Morph close radius: 1 (enabled)
   Morph open radius: 1 (enabled)
   Merge proximity: 0 px (disabled)

   🔍 Filtering training_bombs (bomb)...
      Found 12 connected components
      Kept: 3 components (480 pixels)
      Removed: 9 components (35 pixels)
      Final: 480 pixels (35 removed, 6.8%)

   🔍 Filtering training_dark (dark)...
      Found 45 connected components
      Kept: 8 components (1240 pixels)
      Removed: 37 components (142 pixels)
      Final: 1240 pixels (142 removed, 10.3%)
```

**Interpretation:**
- **Bombs**: 12 components found → 3 large regions kept, 9 small clusters removed (35 px noise)
- **Dark**: 45 components found → 8 regions kept, 37 small clusters removed (142 px noise)
- **Result**: ~7-10% of pixels were isolated noise and correctly removed

### Before/After Comparison

```
📊 Classification distribution (before filtering):
   training_bombs: 515 pixels (2.1%)
   training_dark: 1382 pixels (5.6%)
   training_sediment: 22803 pixels (92.3%)

📊 Classification distribution (after filtering):
   training_bombs: 480 pixels (1.9%)      ← 35 px removed
   training_dark: 1240 pixels (5.0%)      ← 142 px removed
   training_sediment: 22803 pixels (92.3%)  ← unchanged (not filtered)
```

---

## Validation Impact

### Before Filtering (Noisy)
```
📈 VALIDATION RESULTS
Overall Accuracy: 0.812
   training_bombs: P=0.650, R=0.780, F1=0.709  ← Low precision (FPs)
   training_dark:  P=0.720, R=0.850, F1=0.780  ← Low precision (FPs)
```
Many false positives from isolated noise pixels.

### After Filtering (Clean)
```
📈 VALIDATION RESULTS
Overall Accuracy: 0.847
   training_bombs: P=0.820, R=0.760, F1=0.789  ← Higher precision ✅
   training_dark:  P=0.860, R=0.830, F1=0.845  ← Higher precision ✅
```
Fewer false positives, higher precision!

**Trade-off:**
- ✅ **Precision increases** (fewer false positives)
- ⚠️ **Recall may decrease slightly** (some real small features removed)
- ✅ **Overall accuracy improves** (cleaner classification)

---

## Troubleshooting

### "Too many pixels removed"
**Problem:** Filtering removes real features
**Solution:**
- Decrease `cc_min_area` (e.g., 20 → 10)
- Use 4-connectivity instead of 8
- Disable morphological operations (`radius=0`)

### "Still seeing isolated pixels"
**Problem:** Filtering not aggressive enough
**Solution:**
- Increase `cc_min_area` (e.g., 20 → 50)
- Increase `morph_open_radius` (e.g., 1 → 2)
- Enable merging (`merge_proximity_px=3`)

### "Validation accuracy decreased"
**Problem:** Filtering removes pixels that match validation ROIs
**Solution:**
- Check if validation ROIs contain small features
- Adjust `cc_min_area` to keep those features
- Compare before/after filtering distributions

### "Sediment boundaries look jagged"
**Problem:** Filtering sediment removes edges
**Solution:**
- Keep `post_cc_sediment=False` (don't filter sediment)
- Sediment forms large continuous regions, doesn't need filtering

---

## Summary

| Aspect | Before Filtering | After Filtering |
|--------|------------------|-----------------|
| **Isolated pixels** | Many (salt-and-pepper noise) | Removed (min_area threshold) |
| **Precision** | Lower (false positives) | Higher (cleaner regions) |
| **Recall** | Higher (all pixels kept) | Slightly lower (small features removed) |
| **Visualization** | Noisy, hard to interpret | Clean, clear regions |
| **Component count** | Many small clusters | Fewer large regions |

**Recommendation:** Use default settings for most cases:
- `apply_post_filtering=True`
- `cc_min_area=20`
- Filter bombs and dark, not sediment
- This removes ~5-10% of pixels (mostly noise)
