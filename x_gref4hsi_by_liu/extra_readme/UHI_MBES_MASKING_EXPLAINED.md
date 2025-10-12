# UHI-MBES Footprint Masking - Implementation Guide

## Problem Statement

Previously, the code used a **bounding box approach** to mask MBES data to UHI coverage. This resulted in:
- ❌ Rectangular masks instead of the actual ragged UHI footprint shape
- ❌ Slow loop-based nearest-neighbor searches
- ❌ CRS mismatches causing incorrect masks
- ❌ Half-pixel heuristics that were fragile and inaccurate

## Solution: Vectorized Pixel-Perfect Masking

### New Method: `uhi_footprint_mask`

Added to `MBESDetrender` class in `dev3_mbes2.py`:

```python
def uhi_footprint_mask(self, uhi_x_coords, uhi_y_coords, buffer_m=0.0, dilate_px=0):
```

**Key Features:**
1. **Vectorized coordinate transformation** - No loops!
2. **Direct pixel mapping** - Converts world coords → pixel indices using raster geometry
3. **Exact footprint preservation** - Creates mask with same shape as UHI coverage
4. **Optional buffer/dilation** - Can expand coverage if needed
5. **Respects MBES valid_mask** - Only includes pixels with actual data

### How It Works

#### Step 1: Stack Valid UHI Points
```python
x = np.asarray(uhi_x_coords, dtype=float).ravel()
y = np.asarray(uhi_y_coords, dtype=float).ravel()
m = np.isfinite(x) & np.isfinite(y)
x = x[m]; y = y[m]
```

#### Step 2: Convert World Coords → Pixel Indices
Using constant-scale, north-up geometry (no rotation):

```python
# Column index: X coordinate → column number
col0_left = self.x_cols[0] - self.px_x * 0.5
cols = np.floor((x - col0_left) / self.px_x).astype(int)

# Row index: Y coordinate → row number (Y decreases with row)
row0_top = self.y_rows[0] + self.px_y * 0.5
rows = np.floor((row0_top - y) / self.px_y).astype(int)
```

#### Step 3: Clip to Raster Bounds
```python
valid_indices = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
rows = rows[valid_indices]
cols = cols[valid_indices]
```

#### Step 4: Paint Mask
```python
mask = np.zeros((H, W), dtype=bool)
mask[rows, cols] = True
```

#### Step 5: Optional Dilation (Buffer)
```python
if buffer_m > 0:
    dx = int(np.ceil(buffer_m / self.px_x))
    dy = int(np.ceil(buffer_m / self.px_y))
    total_dilate = max(dx, dy)
    mask = binary_dilation(mask, iterations=total_dilate)
```

#### Step 6: Intersect with Valid MBES Data
```python
mask &= self.valid_mask
```

## Usage Example

### In dev5_redo_uhi_vs_mbes.ipynb:

```python
# 1. Extract UHI coordinates in UTM (same CRS as MBES!)
uhi_x_utm, uhi_y_utm = extract_uhi_coordinates_from_cube(
    cube,
    track_start=3039,
    track_end=4029,
    coordinate_system="UTM",
    epsg_utm=config.EPSG_MBES  # Critical: match MBES CRS!
)

# 2. Create mask on full MBES grid
mask_full = mb.uhi_footprint_mask(uhi_x_utm, uhi_y_utm, buffer_m=0.0)

# 3. Extract subregion and apply mask
subregion = mb.residuals[row_start:row_end, col_start:col_end]
mask_sub = mask_full[row_start:row_end, col_start:col_end]
masked_data = np.ma.masked_where(~mask_sub, subregion)
```

### Or Use the Convenience Method:

```python
# All-in-one comparison plot
fig, mask = mb.compare_with_uhi_footprint(
    uhi_x_utm,
    uhi_y_utm,
    buffer_m=5.0,
    title_suffix="Tracks 3039-4029"
)
```

## Critical Requirements

### 1. CRS Matching ⚠️
**MOST IMPORTANT:** UHI coordinates MUST be in the same CRS as MBES!

```python
# ✅ CORRECT:
uhi_x, uhi_y = extract_uhi_coordinates_from_cube(
    cube, track_start, track_end,
    coordinate_system="UTM",
    epsg_utm=config.EPSG_MBES  # e.g., 32632 for UTM Zone 32N
)

# ❌ WRONG:
uhi_x, uhi_y = extract_uhi_coordinates_from_cube(
    cube, track_start, track_end,
    coordinate_system="NED"  # Different CRS → mask will be garbage!
)
```

### 2. North-Up Assumption
Method assumes `transform.b == transform.d == 0` (no rotation). This is true for most GeoTIFFs.

For rotated rasters, use rasterization approach (see comments in code).

### 3. Pixel Geometry
Uses pixel **center** coordinates, not corners. Assumes:
- Column 0 spans `[x_cols[0] - px_x/2, x_cols[0] + px_x/2)`
- Row 0 spans `[y_rows[0] + px_y/2, y_rows[0] - px_y/2]` (Y decreases)

## Advantages Over Old Method

| Aspect | Old Method | New Method |
|--------|-----------|------------|
| Speed | O(N) loops | O(N) vectorized |
| Accuracy | Nearest neighbor + half-pixel check | Direct pixel geometry |
| Shape | Boxy bounding box | Exact ragged footprint |
| CRS handling | Manual checks per point | Single transform operation |
| Dependencies | None | scipy.ndimage (for dilation) |

## Validation

### Visual Check:
The masked MBES plot should show:
- ✅ Same tilted, ragged edges as UHI footprint
- ✅ No rectangular "bounding box" artifacts
- ✅ Smooth transitions at boundaries

### Quantitative Check:
```python
# Count coverage
covered = np.sum(mask)
total = mask.size
print(f"Coverage: {100*covered/total:.2f}%")

# Extract values
mbes_at_uhi = mb.residuals[mask]
print(f"Valid MBES at UHI: {np.sum(np.isfinite(mbes_at_uhi)):,} pixels")
```

## Files Modified

1. **dev3_mbes2.py**:
   - Added `uhi_footprint_mask()` method
   - Added `compare_with_uhi_footprint()` convenience method

2. **dev5_redo_uhi_vs_mbes.ipynb**:
   - Complete workflow for UHI-MBES comparison
   - Includes visualization and statistical comparison

## Future Improvements

1. **For rotated rasters**: Use `rasterio.features.rasterize()` instead of manual pixel math
2. **For large datasets**: Add chunking to process mask in tiles
3. **For different resolutions**: Add resampling option if UHI and MBES have very different pixel sizes

## References

- Rasterio documentation: https://rasterio.readthedocs.io/
- Affine transforms: https://en.wikipedia.org/wiki/Affine_transformation
- Binary morphology: https://docs.scipy.org/doc/scipy/reference/ndimage.html
