# UHI Alignment Shift Implementation

## Summary
Added capability to apply coordinate alignment shifts to `plot_georef` to match MBES data coordinates. The shift values are now centrally configured instead of hardcoded.

## Changes Made

### 1. Config File (`gref_pipeline/config.py`)
Added alignment shift parameters:
```python
# UHI alignment adjustment (for matching with MBES data in NED coordinates)
# These offsets shift the UHI footprint to better align with ground truth
UHI_ALIGNMENT_DX = -0.05  # East offset in meters (positive = shift east)
UHI_ALIGNMENT_DY = -3.0   # North offset in meters (positive = shift north)
```

### 2. Georef Module (`utils/gref_pipeline/georef.py`)
Modified `plot_georef()` method:
- **New parameter**: `apply_alignment_shift=False`
  - When `True` and `coordinate_system='NED'`, applies the shift from config
  - Shifts are: E = E + UHI_ALIGNMENT_DX, N = N + UHI_ALIGNMENT_DY
  - Only works in NED coordinate system (not LATLON or ECEF)

### 3. Notebook (`plot_gref_uhi.ipynb`)
Updated cell 25 to demonstrate both versions:
- **First plot**: Original UHI coordinates (`apply_alignment_shift=False`)
- **Second plot**: Shifted UHI coordinates (`apply_alignment_shift=True`)
  - This matches the coordinates used in `detrend_mbes` analysis

### 4. Detrend MBES (`utils/other/detrend_mbes.py`)
Already uses the shift via `adjust_uhi_alignment(dx=-0.05, dy=-3)`.
Consider updating to use config values instead of hardcoded values in the future.

## Usage

### In plot_georef:
```python
# Original coordinates
cube.plot_georef(
    coordinate_system="NED",
    apply_alignment_shift=False,  # Default
)

# Shifted coordinates (matches MBES)
cube.plot_georef(
    coordinate_system="NED",
    apply_alignment_shift=True,  # Apply shift from config
)
```

### In detrend_mbes:
```python
# Current usage (manually specify shift)
mb_ned.adjust_uhi_alignment(dx=-0.05, dy=-3)

# Future: could use config values
mb_ned.adjust_uhi_alignment(
    dx=config.UHI_ALIGNMENT_DX,
    dy=config.UHI_ALIGNMENT_DY
)
```

## Benefits
1. **Centralized configuration**: Shift values in one place (config.py)
2. **Consistent coordinates**: Both `plot_georef` and `detrend_mbes` can use same shift
3. **Easy comparison**: Can plot with/without shift to see the alignment correction
4. **Documentation**: Shift values are clearly documented in config

## Testing
Run `dev/test_georef_shift.py` to verify:
- Creates two plots side-by-side
- First: original coordinates
- Second: shifted coordinates
- Shift applied: dx=-0.05m E, dy=-3.0m N

## Notes
- The ~2-3m North shift corrects for positioning differences between UHI and MBES datasets
- Small 0.05m East shift is a fine-tuning adjustment
- Shifts only apply in NED coordinate system (meters)
- LATLON and ECEF coordinate systems ignore the shift parameter
