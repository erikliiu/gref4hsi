# MBES Visualization Added to dev2_sim_v3_mbes.py

## Summary

Added MBES seafloor point cloud visualization to the simulation script. The MBES bathymetry data is now displayed alongside the HSI camera trajectory, body frames, and ray tracing.

---

## New Features

### 1. MBES Point Cloud Display
- Loads MBES data from GeoTIFF (configured in `config.MBES_GEOTIFF`)
- Displays as brown-colored scatter points representing the seafloor
- Points are subsampled for performance (configurable via `MBES_SUBSAMPLE`)

### 2. Toggle Button
- New "MBES On/Off" button to show/hide the seafloor points
- Located below the "Play/Pause" button
- Independent from ray visibility control

### 3. Configuration Parameters

Added at the top of the script:

```python
# MBES mesh visuals
SHOW_MBES_BY_DEFAULT = True       # Show MBES on startup
MBES_POINT_SIZE = 0.5             # Point size for scatter plot
MBES_ALPHA = 0.3                  # Transparency (0=invisible, 1=opaque)
MBES_COLOR = (0.6, 0.4, 0.2)      # Brown-ish color for seafloor
MBES_SUBSAMPLE = 50               # Use every 50th point to reduce clutter
```

---

## How It Works

### Loading Process

1. **Check for GeoTIFF**: Looks for `config.MBES_GEOTIFF` path
2. **Try Pre-converted Mesh**: First checks for `*_mesh.ply` file (faster)
3. **Convert GeoTIFF**: If no PLY exists, loads and converts GeoTIFF using rasterio
4. **Subsample**: Reduces point count by `MBES_SUBSAMPLE` factor for performance
5. **Display**: Adds as 3D scatter plot in UTM coordinates

### Data Flow

```
MBES GeoTIFF
    ↓
Load with rasterio (or PyVista if PLY exists)
    ↓
Extract X, Y, Z in UTM coordinates
    ↓
Subsample every Nth point
    ↓
Display as 3D scatter (brown, semi-transparent)
```

---

## Visual Integration

The plot now shows:

1. ✅ **Full mission track** (gray line)
2. ✅ **H5 segments** (colored lines)
3. ✅ **MBES seafloor** (brown scatter points) ← NEW
4. ✅ **IMU position** (moving colored dot)
5. ✅ **Camera position** (moving magenta dot)
6. ✅ **Body axes** (RGB arrows)
7. ✅ **Camera rays** (magenta/red lines)

---

## Controls

| Button | Function |
|--------|----------|
| Slider | Scrub through frames |
| Play/Pause | Start/stop animation |
| Rays On/Off | Toggle camera ray visibility |
| MBES On/Off | Toggle seafloor visibility ← NEW |

---

## Performance Notes

### Subsampling
- Default `MBES_SUBSAMPLE = 50` uses every 50th point
- Adjust based on your system:
  - Lower (e.g., 20) = more detail, slower
  - Higher (e.g., 100) = less detail, faster

### Point Count Example
- Original GeoTIFF: 10,000,000 points
- After subsampling by 50: 200,000 points
- Displayed in plot: 200,000 scatter points

### Transparency
- `MBES_ALPHA = 0.3` allows seeing through the seafloor
- Increase to 0.6-0.8 for more solid appearance
- Decrease to 0.1-0.2 for more transparent (better for dense data)

---

## Dependencies

### Required
- `numpy` (already required)
- `matplotlib` (already required)

### Optional (for MBES loading)
- `pyvista` - For loading mesh files (PLY format)
- `rasterio` - For loading GeoTIFF directly

If missing, the script will print a warning and continue without MBES visualization.

---

## Troubleshooting

### "PyVista not available"
```bash
conda install -c conda-forge pyvista
# or
pip install pyvista
```

### "Rasterio not available"
```bash
conda install -c conda-forge rasterio
# or
pip install rasterio
```

### "MBES GeoTIFF not found"
- Check `config.MBES_GEOTIFF` path is correct
- Verify the file exists: `E:\mjosa_new\DTM\geotiff_2.tif`

### "No valid MBES data points found"
- GeoTIFF may contain all NaN/invalid values
- Check the file in QGIS or other GIS software

### Plot too cluttered
- Increase `MBES_SUBSAMPLE` (e.g., from 50 to 100)
- Decrease `MBES_ALPHA` (e.g., from 0.3 to 0.1)
- Decrease `MBES_POINT_SIZE` (e.g., from 0.5 to 0.2)

---

## Coordinate Systems

All data displayed in **UTM Zone 32N (EPSG:32632)**:
- MBES points: Loaded directly from GeoTIFF (already in UTM)
- Navigation: Converted from WGS84 lat/lon to UTM
- Camera rays: Transformed to world frame (UTM)

Everything aligned in the same coordinate system! ✅

---

## Example Usage

```bash
# Run with default settings
python dev2_sim_v3_mbes.py

# Run with fewer frames per H5 file
python dev2_sim_v3_mbes.py --frames 100
```

Once running:
1. Use slider to navigate frames
2. Click "Play/Pause" to animate
3. Click "Rays On/Off" to show/hide camera rays
4. Click "MBES On/Off" to show/hide seafloor
5. Rotate/zoom the 3D plot with mouse

---

## What to Expect

The visualization should now show:
- Your underwater vehicle path above the seafloor
- Camera rays shooting down toward the MBES surface
- Ray intersection points should align with MBES depth
- Seafloor topography visible as brown point cloud

This helps verify:
✅ Navigation alignment with bathymetry
✅ Ray directions are correct (pointing down)
✅ Camera FOV covers the seafloor
✅ Time synchronization is reasonable

---

**Note**: First run may be slow while loading/converting GeoTIFF. Subsequent runs will be faster if PLY is cached.
