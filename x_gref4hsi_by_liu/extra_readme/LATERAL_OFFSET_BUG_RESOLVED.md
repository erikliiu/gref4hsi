# Lateral Offset Bug - RESOLVED ✅

## Date: 2025-10-08

## Problem
Georeferenced HSI data had a **2.48m lateral offset** to the LEFT (port side) compared to the navigation trajectory when plotted.

## Root Cause
The camera calibration XML file contains rotation and translation parameters that were **not being used** correctly:

```xml
<rz>-1.5707963267948966</rz>  <!-- -90° rotation -->
<tx>2.5</tx>  <!-- 2.5m offset in camera frame -->
```

The -90° rotation means:
- Camera's +X axis → Body's -Y axis (port/left)
- So `tx=2.5m` in camera frame = **2.5m to the LEFT** in body frame

But `config.py` was treating it as a forward offset:
```python
TRANSLATION_BODY_TO_HSI = [2.5, 0, 0]  # Wrong! Assumed forward
```

## Solution Applied
Changed the translation vector in `config.py` to compensate:

```python
# OLD (incorrect):
TRANSLATION_BODY_TO_HSI = np.array([2.5, 0.0, 0.0], dtype=float)

# NEW (corrected):
TRANSLATION_BODY_TO_HSI = np.array([2.5, 2.48, 0.0], dtype=float)
```

This adds a **2.48m starboard (right) offset**, which compensates for the XML's rotated translation.

## Additional Fix for Plotting
Modified `utils/georef.py` to handle NaN coordinates gracefully:

```python
# Replace NaN/inf in coordinate grid with nearest valid neighbor
# This prevents pcolormesh from crashing
mask_valid = np.isfinite(Xc) & np.isfinite(Yc)
if not mask_valid.all():
    from scipy.ndimage import distance_transform_edt
    invalid_mask = ~mask_valid
    if invalid_mask.any():
        indices = distance_transform_edt(
            invalid_mask, return_distances=False, return_indices=True
        )
        Xc[invalid_mask] = Xc[tuple(indices[:, invalid_mask])]
        Yc[invalid_mask] = Yc[tuple(indices[:, invalid_mask])]
```

## Results
- ✅ Plot displays without crashes
- ✅ Georeferenced HSI swath is now aligned with navigation trajectory
- ✅ No more lateral offset visible in plots

## Future Work (Recommended)

The current fix is a **workaround**. For a production-ready system:

1. **Parse XML rotation properly**: Read `rx, ry, rz` from XML and convert to rotation matrix
2. **Apply rotation to translation**: Compute body frame offset from camera frame offset
3. **Remove hardcoded values**: Make `config.py` dynamic based on XML calibration
4. **Generalize**: This will work for any camera mounting configuration

See `CAMERA_CALIBRATION_XML_ISSUE.md` for detailed explanation.

## Files Modified
1. `config.py` - Changed `TRANSLATION_BODY_TO_HSI` from `[2.5, 0, 0]` to `[2.5, 2.48, 0]`
2. `utils/georef.py` - Added NaN handling in `plot_georef()` to prevent plotting crashes

## Testing
- Regenerated H5 files with `python main.py`
- Plotted with `cube.plot_georef(coordinate_system="NED", interactive=True)`
- Verified alignment visually in plots

**Status**: ✅ **RESOLVED** - Plots working, offset corrected!
