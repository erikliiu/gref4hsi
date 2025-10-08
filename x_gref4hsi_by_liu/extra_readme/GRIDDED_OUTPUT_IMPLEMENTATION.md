# Gridded Output Format Implementation

## Changes Made

Your `main.py` now saves **GRIDDED format** like `test_eely.py`!

### Summary of Changes:

1. **Updated `utils.save_intersection_to_h5()` function:**
   - Added parameters: `n_frames`, `n_slits`, `gridded=True/False`
   - Gridded mode creates `(T, S, 3)` arrays preserving hypercube structure
   - Failed rays filled with `NaN` values
   - Compatible with test_eely and orthorectification pipeline

2. **Updated `main.py` to pass required parameters:**
   - Passes `n_frames`, `n_slits`, and `gridded=True` to save function
   - Fixed imports to use `from utils import utils`

3. **Output format now matches test_eely:**
   ```python
   processed/georef/points_ecef_crs    (T, S, 3)  # GRIDDED: (frames, slits, XYZ)
   processed/georef/pixel_nr_grid      (T, S)      # Pixel indices per frame
   processed/georef/frame_nr_grid      (T, S)      # Frame indices per pixel
   ```

---

## Before vs After

### BEFORE (Flattened):
```python
points_ecef_crs:  (1,883,728, 3)   # Only successful rays
ray_indices:      (1,883,728,)
frame_indices:    (1,883,728,)
```

### AFTER (Gridded):
```python
points_ecef_crs:  (1946, 968, 3)   # ALL rays (T, S, 3)
pixel_nr_grid:    (1946, 968)      # Pixel indices
frame_nr_grid:    (1946, 968)      # Frame indices
```

---

## Verification

Tested with `rad_uhi_20241029_115057_2.h5`:

```
processed/georef/ datasets:
  frame_nr_grid     shape=(1946, 968)     dtype=int32
  pixel_nr_grid     shape=(1946, 968)     dtype=int32
  points_ecef_crs   shape=(1946, 968, 3)  dtype=float64   ✓ GRIDDED

GRIDDED FORMAT CONFIRMED:
  Shape: (1946, 968, 3) - GRIDDED ✓
  Total rays: 1,883,728
  Valid (non-NaN): 1,883,728 (100.00%)
  This matches test_eely output format! ✓
```

---

## Benefits of Gridded Format

### ✅ Advantages:
1. **Preserves hypercube structure** - maintains (T, S) grid from original datacube
2. **Easy indexing** - `data[frame, pixel]` directly accesses any point
3. **Orthorectification compatible** - can run `orthorectification.main()` if you add other ancillary data
4. **test_eely compatible** - same data structure as reference implementation
5. **Easy to access frame/pixel** - no need to reconstruct indices

### 📊 Comparison with Radiance Datacube:
```python
Radiance:   (T, S, bands) = (1946, 968, 210)
Geometry:   (T, S, 3)     = (1946, 968, 3)     ← Your new format
```
Perfect correspondence!

---

## What About Your Existing Plotting Functions?

Your existing functions **will still work** with minor updates:

### Update needed in `dev2_html.py`:

The `_read_hsi_points_ecef_from_file()` function already handles both formats:

```python
if d.ndim == 3 and d.shape[-1] == 3:
    # Gridded format (T, S, 3) ✓
    X = d[:, :, 0]
    Y = d[:, :, 1]
    Z = d[:, :, 2]
    return X, Y, Z, dset_path
```

So **no changes needed** - your HTML map generation should work as-is!

---

## File Size Comparison

Since you have 100% hit rate, file size is essentially the same:

- **Flattened:** 1,883,728 × 3 × 8 bytes = ~45 MB
- **Gridded:** 1946 × 968 × 3 × 8 bytes = ~45 MB (same!)

If you had failed rays (e.g., 90% success):
- **Flattened:** Would be ~10% smaller (only successful rays)
- **Gridded:** Same size but with NaN for failed rays

---

## How to Switch Back to Flattened (if needed)

Simply change one line in `main.py`:

```python
utils.save_intersection_to_h5(
    str(output_h5_path), 
    intersections_ecef, 
    pixel_indices, 
    frame_indices,
    n_frames=n_frames,
    n_slits=n_slits,
    gridded=False  # ← Change to False for flattened format
)
```

---

## Next Steps (Optional)

If you want **full test_eely compatibility** for orthorectification, you'd need to add:

1. **Surface normals** (`normals_ned_crs`, `normals_hsi_crs`)
   - Extract from MBES mesh at intersection points
   
2. **View angles** (`theta_v`, `phi_v`)
   - Compute from ray directions
   
3. **Sun angles** (`theta_s`, `phi_s`)
   - Requires sun position calculation (astronomy libs)
   
4. **Timestamps per pixel** (`unix_time_grid`)
   - Replicate HSI timestamps to (T, S) grid

But for now, **your gridded ECEF points are fully compatible with test_eely's data structure!** ✓

---

## Summary

✅ **Done:** Converted to gridded `(T, S, 3)` format  
✅ **Compatible:** Matches test_eely output structure  
✅ **Working:** Tested with actual H5 file  
✅ **Plotting:** Your existing functions should work  
✅ **Future-proof:** Ready for orthorectification if you add ancillary data  

Your georeferencing pipeline now produces **professional, standardized output** matching the gref4hsi reference implementation! 🎉
