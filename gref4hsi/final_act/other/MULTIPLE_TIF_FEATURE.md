# Multiple TIF Overlay Feature - Documentation

## Summary
Added optional feature to `create_html.py` that allows loading multiple GeoTIFF files from a folder and adding them as separate toggleable layers in the interactive map.

## Changes Made

### 1. Function Signature Update
Added two new optional parameters to `build_map_with_mbes()`:
- `tif_folder=None`: Path to folder containing additional TIF files
- `tif_opacity=0.7`: Opacity for the additional TIF overlays

### 2. Implementation Details

**Location in code**: After the main MBES overlay (step 5b)

**Key features**:
- Discovers all `.tif` and `.tiff` files in the specified folder
- Processes each TIF file sequentially
- Reprojects to EPSG:4326 (WGS84)
- Uses **same colormap and vmin/vmax** as main MBES overlay for consistency
- Each TIF becomes a separate layer named by its filename (e.g., "2024-10-29_12-59-05.tif")
- All additional TIF layers are **hidden by default** (toggle via layer control)
- Error handling: If a TIF fails to load, it prints error and continues with next file

### 3. Backward Compatibility
✅ **100% backward compatible**
- If `tif_folder=None` (default), the function works exactly as before
- Original MBES overlay behavior is completely unchanged
- Only adds new functionality when `tif_folder` parameter is provided

## Usage

### Basic Usage (Original behavior - unchanged)
```python
build_map_with_mbes(
    nav_csv,
    geotiff_path,
    output_html="map.html",
    # ... other parameters
)
```

### With Multiple TIF Overlays (New feature)
```python
build_map_with_mbes(
    nav_csv,
    geotiff_path,
    output_html="map.html",
    # NEW: Add multiple TIF overlays
    tif_folder=r"E:\mjosa_new_oct_2025\all_tifs_from_eiva\relevant_tifs_only",
    tif_opacity=0.70,
    # ... other parameters
)
```

## Test Function
Added `test_multiple_tifs()` function to verify the feature:
- Loads main MBES overlay
- Adds 3 TIF files from the specified folder
- Creates test HTML map: `test_multiple_tifs_map.html`

Run test with:
```bash
python test_run_multiple_tifs.py
```

## Test Results
✅ **Successfully tested** with 3 TIF files:
- 2024-10-29_10-49-21.tif
- 2024-10-29_12-59-05.tif
- 2024-10-29_13-10-07.tif

Output:
- HTML map: 14.3 MB
- All TIF overlays use same colormap (viridis)
- Shared vmin/vmax: -135.300 to -111.838
- Each TIF is toggleable in layer control (top right)
- All TIF layers hidden by default ✓

## Layer Control Appearance
After opening the map, the layer control (top right) shows:
```
☐ MBES overlay (default: visible)
☐ 2024-10-29_10-49-21.tif (default: hidden)
☐ 2024-10-29_12-59-05.tif (default: hidden)
☐ 2024-10-29_13-10-07.tif (default: hidden)
☐ Navigation Track
☐ HSI footprint (outline)
... other layers
```

## Benefits
1. **Time-series visualization**: TIF filenames contain timestamps, making it easy to see temporal changes
2. **Consistent coloring**: All overlays use same colormap range for easy comparison
3. **Non-intrusive**: Hidden by default to avoid clutter
4. **Flexible**: User can toggle any combination of layers
5. **Optional**: Doesn't affect existing functionality

## Implementation Notes
- TIF files are sorted alphabetically (which works for timestamp-based names)
- Uses same reprojection and colorization pipeline as main MBES overlay
- Error handling per file - one failure doesn't stop the rest
- Progress indication shows which files are being processed
