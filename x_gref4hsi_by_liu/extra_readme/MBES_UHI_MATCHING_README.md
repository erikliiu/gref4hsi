# MBES-UHI Region Matching

This guide explains how to plot MBES detrended bathymetry for the same geographic region as your UHI data.

## Overview

Two new functions have been added to `dev3_mbes2.py`:

1. **`plot_subregion_residuals()`** - Plots detrended MBES bathymetry for a specific X, Y region
2. **`extract_uhi_bounds_from_cube()`** - Extracts geographic bounds from a UHI data cube

## Quick Start (Notebook)

In your `dev4_plot_georef_uhi_with_mbes.ipynb` notebook, a new cell has been added at the bottom:

```python
# Load MBES data and extract the same region as the UHI plot
import sys
sys.path.append("../")
from dev.dev3_mbes2 import MBESDetrender, extract_uhi_bounds_from_cube

# Load and detrend MBES
mb = MBESDetrender(
    config.MBES_GEOTIFF,
    crop_start_m=1.0,
    crop_end_m=10.0,
).load()

mb.detrend(
    order_x=4,
    smooth_baseline_m=0.1,
    smooth_tilt_m=0.1,
    smooth_center_m=1,
    central_frac=0.8,
    robust=True,
)

# Extract bounds from UHI cube
track_start = 3039
track_end = 4029

x_bounds, y_bounds = extract_uhi_bounds_from_cube(
    cube,
    track_start,
    track_end,
    coordinate_system="NED",
    origin=(config.LAT0, config.LON0, config.H0),
)

# Plot matching MBES region
fig_mbes = mb.plot_subregion_residuals(
    x_bounds=x_bounds,
    y_bounds=y_bounds,
    figsize=(8, 10),
    title_suffix=f"Matching UHI tracks {track_start}-{track_end}",
)
```

## Usage from `dev3_mbes2.py`

The example code is also available in `dev3_mbes2.py` (commented out at the bottom). 

To use it:
1. Uncomment the section starting with `# ========== OPTIONAL: Plot MBES subregion...`
2. Adjust the file names and track range as needed
3. Run the script

## How It Works

### Step 1: Extract UHI Bounds

The `extract_uhi_bounds_from_cube()` function:
- Takes your UHI cube and track range (e.g., tracks 3039-4029)
- Converts ECEF coordinates to NED (or keeps as ECEF)
- Returns X (Easting) and Y (Northing) bounds

```python
x_bounds, y_bounds = extract_uhi_bounds_from_cube(
    cube,           # Your UHI CombinedTransectCube
    3039,           # track_start
    4029,           # track_end
    "NED",          # coordinate system
    (lat, lon, h)   # origin for NED conversion
)
# Returns: (x_min, x_max), (y_min, y_max)
```

### Step 2: Plot MBES Subregion

The `plot_subregion_residuals()` method:
- Finds the corresponding region in the MBES raster
- Extracts and displays the detrended bathymetry
- Shows the same visualization as `plot_residual_heatmap()` but for your specific region

```python
fig = mb.plot_subregion_residuals(
    x_bounds=(x_min, x_max),  # Easting bounds
    y_bounds=(y_min, y_max),  # Northing bounds
    figsize=(8, 10),          # Figure size
    cmap="RdBu_r",            # Colormap
    vmax=None,                # Auto color limits
    title_suffix="Custom"     # Optional title text
)
```

## Coordinate Systems

Both functions work in the same coordinate system:
- **NED**: North-East-Down (meters from origin)
- **ECEF**: Earth-Centered Earth-Fixed (X, Y, Z in meters)

The MBES GeoTIFF is typically in UTM/projected coordinates. Make sure your UHI coordinates are converted to the same system (NED works if the origin is in the same area).

## Troubleshooting

### No MBES data found

If you get: `⚠️ No MBES data found in region...`

Possible causes:
1. **Coordinate mismatch**: UHI and MBES are in different coordinate systems
2. **Region outside MBES coverage**: The UHI track range is outside the MBES area
3. **Wrong origin**: Check that `config.LAT0`, `config.LON0`, `config.H0` are correct

**Solution**: Print the bounds to check:
```python
print(f"UHI X bounds: {x_bounds}")
print(f"UHI Y bounds: {y_bounds}")
print(f"MBES extent: {mb.extent}")
```

### Wrong detrending

If the MBES plot looks strange:
1. Check detrending parameters (they should match `dev3_mbes2.py` settings)
2. Try adjusting `smooth_baseline_m`, `smooth_tilt_m` values
3. Make sure `mb.detrend()` is called before `plot_subregion_residuals()`

## Example Output

The plot shows:
- **Color**: Red = elevated features, Blue = depressions (relative to polynomial trend)
- **Title**: Detrending settings and track range
- **Axes**: Easting (X) and Northing (Y) in meters
- **Colorbar**: Residual depth in meters

This matches the UHI georeferenced plot, allowing direct comparison of HSI and bathymetry features.

## Parameters Reference

### `plot_subregion_residuals()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_bounds` | tuple | required | (x_min, x_max) in meters |
| `y_bounds` | tuple | required | (y_min, y_max) in meters |
| `figsize` | tuple | (8, 10) | Figure size |
| `cmap` | str | "RdBu_r" | Matplotlib colormap |
| `vmax` | float | None | Symmetric color limit (auto if None) |
| `title_suffix` | str | "" | Extra text for title |

### `extract_uhi_bounds_from_cube()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cube` | CombinedTransectCube | required | UHI data cube |
| `track_start` | int | required | Starting track index |
| `track_end` | int | required | Ending track index |
| `coordinate_system` | str | "NED" | "NED", "ECEF", or "LATLON" |
| `origin` | tuple | None | (lat, lon, h) for NED conversion |

## Tips

1. **Same area, different resolutions**: MBES typically has finer resolution (~0.05m) than UHI pixel footprints
2. **Use same colormap**: For visual comparison, you might want to use the same colormap for both UHI and MBES
3. **Save figures**: Use `fig.savefig("mbes_region.png", dpi=300)` to save high-res plots
4. **Adjust track range**: Experiment with different `track_start`/`track_end` values to focus on features of interest

