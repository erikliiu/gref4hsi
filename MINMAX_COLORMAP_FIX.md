# Fix for MinMax Normalization "All Red" Heatmap

## Problem
When using `normalization_method="minmax"` in `plot_uhi_mbes_comparison`, the difference heatmap (Δ panel) was completely red/pink, showing no variation.

## Root Cause
The colormap limits were computed incorrectly for minmax normalization:

### BEFORE (Buggy):
```python
# Compute difference: MBES - UHI (both normalized to [0, 1])
diff_z = z_mbes - z_uhi  # Range: [-1, +1] theoretically

# Use symmetric limits based on absolute values
vmax_diff = np.nanpercentile(np.abs(diff_z), 98.0)  
vmin_diff = -vmax_diff  # e.g., [-0.75, +0.75]
```

**Why this failed:**
- With minmax normalization, `z_mbes` and `z_uhi` are each scaled to [0, 1] independently
- Their difference `diff_z = z_mbes - z_uhi` can be biased
- Example: If `z_mbes` is mostly in [0.5-0.7] and `z_uhi` is mostly in [0.2-0.4], then `diff_z` is mostly in [0.2-0.5] (all positive!)
- The symmetric limits `[-0.75, +0.75]` don't match the actual data range [0.2-0.5]
- All values fall in the positive (red) half of the colormap
- Result: Entire heatmap appears red with no contrast

### AFTER (Fixed):
```python
if normalization_method.lower() == "minmax":
    # Use actual data distribution (2nd to 98th percentiles)
    vmin_diff = np.nanpercentile(diff_data, 2.0)   # e.g., 0.22
    vmax_diff = np.nanpercentile(diff_data, 98.0)  # e.g., 0.48
    
    # THEN symmetrize around zero for consistent visualization
    vlim = max(abs(vmin_diff), abs(vmax_diff))  # e.g., 0.48
    vmin_diff, vmax_diff = -vlim, vlim          # [-0.48, +0.48]
```

**Why this works:**
1. First finds the actual data range (removing outliers with 2nd/98th percentiles)
2. Then symmetrizes around zero using the maximum absolute value
3. This ensures:
   - The colormap spans the actual data range
   - Zero (white) is still at the center
   - Both positive (red) and negative (blue) regions are visible
   - Contrast is preserved

## Comparison: ZScore vs MinMax

### ZScore Normalization (Original Approach - Still Correct)
```python
# Zscore: (x - median) / MAD
# Both datasets centered around 0 with similar scales
# Difference is also centered around 0
# Symmetric limits work perfectly!
vmax_diff = np.nanpercentile(np.abs(diff_z), 98.0)
vmin_diff = -vmax_diff  # ✅ Correct for zscore
```

### MinMax Normalization (New Fixed Approach)
```python
# Minmax: (x - min) / (max - min)  → [0, 1]
# Datasets NOT centered around 0
# Difference can be biased positive or negative
# Need data-driven limits first, then symmetrize
vmin_diff = np.nanpercentile(diff_data, 2.0)
vmax_diff = np.nanpercentile(diff_data, 98.0)
vlim = max(abs(vmin_diff), abs(vmax_diff))
vmin_diff, vmax_diff = -vlim, vlim  # ✅ Correct for minmax
```

## Expected Results

### Before Fix (MinMax):
- Third panel (Δ heatmap) completely red/pink
- No visible variation or features
- Colorbar shows narrow range like [0.2, 0.3] but colormap is [-0.75, +0.75]

### After Fix (MinMax):
- Third panel shows proper red/blue contrast
- Features and variations visible
- Colorbar matches actual data range (e.g., [-0.5, +0.5])
- White regions near zero show agreement
- Red/blue regions show differences

## Testing

Re-run the minmax comparison:
```python
fig2 = mb_ned.plot_uhi_mbes_comparison(
    use_adjusted=True, 
    normalization_method="minmax", 
    show=True
)
```

The third panel should now show:
- ✅ Proper red/blue/white color distribution
- ✅ Visible features and contrast
- ✅ Colorbar limits that match the visual range

## Files Modified
- `gref4hsi/final_act/utils/other/detrend_mbes.py` - Fixed `plot_uhi_mbes_comparison` method (lines ~1867-1890)
- `gref4hsi/final_act/notebooks/detrend_mbes.ipynb` - Added explanation note

## Related Issues
This issue only affected `plot_uhi_mbes_comparison` with `normalization_method="minmax"`. The zscore method was working correctly all along.
