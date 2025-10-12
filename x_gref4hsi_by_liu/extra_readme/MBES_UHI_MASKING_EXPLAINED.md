# MBES-UHI Footprint Masking

## Overview

When comparing MBES bathymetry with UHI hyperspectral data, we have two approaches for extracting the matching region:

### 1. **Bounding Box Approach** (Rectangular)
- Extracts min/max X and Y coordinates from UHI data
- Creates a rectangular region containing all UHI measurements
- **Result**: Perfect rectangle that includes areas NOT measured by UHI
- **Use case**: When you want to see all bathymetry in the general area

### 2. **Exact Footprint Masking** (Masked to UHI coverage)
- Extracts ALL individual UHI pixel coordinates
- Masks MBES pixels to only show where UHI actually measured
- **Result**: "Raggedy" shape matching exact UHI coverage contours
- **Use case**: When you want 1:1 comparison of co-located measurements

## Why Different Shapes?

The UHI georeferenced image has an irregular shape because:
- The vehicle follows a path (not perfectly straight)
- Altitude changes affect footprint size
- Field of view creates angled edges at the sides
- Roll/pitch variations change the pixel positions

The MBES GeoTIFF is a perfect grid because it was collected separately and processed into a regular raster format.

## Implementation

### Functions Added to `dev3_mbes2.py`:

#### 1. `extract_uhi_coordinates_from_cube()`
```python
uhi_x_coords, uhi_y_coords = extract_uhi_coordinates_from_cube(
    cube, 
    track_start=3039, 
    track_end=4029,
    coordinate_system="UTM",
    epsg_utm=32632
)
```
- Returns: Full 2D arrays of UHI pixel coordinates (T × S shape)
- Converts from ECEF → UTM (or NED/ECEF)
- Used for creating coverage masks

#### 2. `MBESDetrender.plot_subregion_residuals_masked()`
```python
fig, mask = mb.plot_subregion_residuals_masked(
    uhi_x_coords=uhi_x_coords,
    uhi_y_coords=uhi_y_coords,
    x_bounds=x_bounds,  # Optional
    y_bounds=y_bounds,  # Optional
    figsize=(8, 10),
    title_suffix="Matching UHI tracks 3039-4029",
    buffer_m=0.0  # Expand footprint by this many meters
)
```
- Creates boolean mask showing which MBES pixels have UHI coverage
- Only plots MBES data where mask is True
- Returns both the figure and the mask array

### Algorithm

For each UHI pixel at position (x_uhi, y_uhi):
1. Find nearest MBES pixel
2. Check if distance < half-pixel size
3. If yes, mark that MBES pixel as "covered"

This creates a binary mask that matches the exact UHI footprint.

## Usage Example

```python
# 1. Load and detrend MBES
mb = MBESDetrender(config.MBES_GEOTIFF).load()
mb.detrend(order_x=4, smooth_baseline_m=0.1, smooth_tilt_m=0.1)

# 2. Extract UHI coordinates (not just bounds)
uhi_x, uhi_y = extract_uhi_coordinates_from_cube(
    cube, 3039, 4029, coordinate_system="UTM", epsg_utm=32632
)

# 3. Plot masked MBES matching exact UHI footprint
fig, mask = mb.plot_subregion_residuals_masked(
    uhi_x_coords=uhi_x,
    uhi_y_coords=uhi_y,
    figsize=(8, 10)
)
```

## Comparison

| Aspect | Bounding Box | Exact Masking |
|--------|-------------|---------------|
| Shape | Rectangle | Irregular (matches UHI) |
| MBES pixels shown | All in region | Only where UHI measured |
| Computation | Fast (just bounds) | Slower (checks all pixels) |
| Coverage | ~100% of box | ~60-80% (depends on swath) |
| Use case | Overview | Direct comparison |

## Performance

For a typical dataset:
- **UHI pixels**: ~1 million (990 tracks × 968 slits)
- **MBES pixels in region**: ~500,000 (depends on resolution)
- **Masking time**: ~2-5 seconds
- **Memory**: Minimal (uses boolean mask)

## Visual Comparison

### Bounding Box Approach:
```
┌─────────────────┐
│ ░░░░░░░░░░░░░░░ │  ← Full rectangle
│ ░░░░░MBES░░░░░░ │     Including areas
│ ░░░░░DATA░░░░░░ │     NOT measured by UHI
│ ░░░░░░░░░░░░░░░ │
└─────────────────┘
```

### Exact Masking:
```
    ╱─────────╲      ← Irregular shape
   ╱  MBES    ╲        Matching UHI
  │   DATA     │       coverage exactly
   ╲  ONLY    ╱
    ╲────────╱
```

## Files Modified

1. **`dev3_mbes2.py`**:
   - Added `extract_uhi_coordinates_from_cube()` function
   - Added `MBESDetrender.plot_subregion_residuals_masked()` method

2. **`dev4_plot_georef_uhi_with_mbes.ipynb`**:
   - Added cell demonstrating masked plotting
   - Added side-by-side comparison visualization

## When to Use Each Approach

### Use Bounding Box when:
- Quick overview of bathymetry in survey area
- Want to see full context around UHI measurements
- Making figures showing survey location
- Don't need exact pixel-to-pixel correspondence

### Use Exact Masking when:
- Comparing UHI spectral features to bathymetric features
- Need co-located measurements only
- Creating correlation plots or statistical analysis
- Want to show actual survey coverage
- Making publication figures showing data alignment

## Notes

- The masking algorithm uses half-pixel buffer to ensure coverage
- Coordinate systems must match (both use UTM Zone 32N)
- NaN values in UHI coordinates are automatically filtered
- The mask is returned and can be reused for other analyses
- Buffer parameter can be adjusted to expand/contract footprint

## Future Enhancements

Possible improvements:
- [ ] Use KDTree for faster nearest-neighbor search
- [ ] Add interpolation option to fill small gaps
- [ ] Support for saving masked MBES as GeoTIFF
- [ ] Overlay UHI and MBES in same plot
- [ ] Statistical comparison tools (correlation, RMSE, etc.)
