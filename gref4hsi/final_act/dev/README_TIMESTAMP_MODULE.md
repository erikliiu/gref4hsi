# Georeferenced HSI Plotting with UTC Timestamps

## 📋 Overview

This module provides complete functionality for plotting georeferenced hyperspectral imaging (HSI) data with UTC timestamp display. It reads HDF5 files containing processed radiance and georeferencing data, and creates RGB composite visualizations in various coordinate systems.

## ✨ Features

- ✅ **Automatic timestamp detection** - Reads from `processed/radiance/timestamps` with robust fallbacks
- ✅ **UTC time display** - Shows start/end times for plotted track ranges
- ✅ **Multiple coordinate systems** - LATLON, NED, or ECEF
- ✅ **Interactive mode** - Click on plot to display coordinates
- ✅ **File management** - Combine multiple HDF5 files into single logical cube
- ✅ **Flexible RGB** - Custom wavelength selection for RGB channels

## 📁 Files

- **`dev3_timestamp_complete.py`** - Complete, self-contained module (USE THIS!)
- **`example_usage.py`** - Example scripts showing how to use the module
- **`dev3_timestamp.py`** - Your original work-in-progress (can be deleted)

## 🚀 Quick Start

### Basic Usage

```python
import dev3_timestamp_complete as geo

# Load HDF5 files from folder
transect = geo.load_transect("/path/to/h5/files")
transect.list_files()

# Select all files and create combined cube
cube = transect.select_all_files()

# Plot with UTC timestamps
cube.plot_georef(
    coordinate_system="NED",
    track_start=100,
    track_end=200,
    interactive=True
)
```

### Using config.OUTPUT_FOLDER

```python
import dev3_timestamp_complete as geo

# If config.OUTPUT_FOLDER is set, no need to specify path
transect = geo.load_transect()
cube = transect.select_all_files()
cube.plot_georef(coordinate_system="LATLON")
```

### Select Specific Files

```python
# By name
cube = transect.select_files([
    "rad_uhi_20241029_115057_1",
    "rad_uhi_20241029_115057_2",
])

# By pattern
cube = transect.select_files_by_pattern("rad_uhi_*")
```

## 📊 plot_georef() Parameters

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coordinate_system` | str | "LATLON" | Display coordinate system: "LATLON", "NED", or "ECEF" |
| `track_start` | int | None | Starting track index (inclusive), None = start from beginning |
| `track_end` | int | None | Ending track index (exclusive), None = plot until end |
| `interactive` | bool | True | Enable click-to-inspect coordinates |
| `show_file_boundaries` | bool | True | Draw yellow dotted lines at file boundaries |
| `red_wl`, `green_wl`, `blue_wl` | float | 654.2, 560.0, 440.3 | Wavelengths (nm) for RGB channels |
| `normalize` | bool | True | Normalize each channel to [0, 1] |
| `origin` | tuple | None | (lat, lon, h) reference point for NED/ECEF |
| `figsize` | tuple | (11, 9) | Figure size (width, height) in inches |

### Example: Custom Options

```python
cube.plot_georef(
    coordinate_system="ECEF",
    use_local_origin=True,
    origin=(60.8, 10.7, 0.0),
    red_wl=670.0,
    green_wl=550.0,
    blue_wl=450.0,
    track_start=8637,
    track_end=9709,
    figsize=(14, 10),
    show_file_boundaries=True,
    interactive=True
)
```

## 🔍 Interactive Mode

When `interactive=True` (default), click on any pixel in the plot to display:

```
======================================================================
📍 track=8750, slit=256
   NED: E=123.45 m, N=678.90 m
   ECEF: X=3154321.12 m, Y=598765.43 m, Z=5782109.87 m
   WGS84: Lat=60.801234°, Lon=10.712345°, h=125.34 m
   RGB: R=0.234, G=0.567, B=0.789
======================================================================
```

## 📂 Expected HDF5 Structure

The module expects HDF5 files with the following datasets:

```
processed/
├── radiance/
│   ├── dataCube                 # (T, S, B) - radiance values [REQUIRED]
│   ├── dataCube_corrected       # (T, S, B) - corrected radiance [OPTIONAL]
│   ├── timestamps               # (T,) - Unix timestamps [PREFERRED]
│   └── calibration/
│       └── spectral/
│           └── band2Wavelength  # (B,) - wavelengths in nm [REQUIRED]
└── georef/
    └── points_ecef_crs          # (T, S, 3) - ECEF coordinates [REQUIRED]
```

Where:
- **T** = number of tracks (scan lines)
- **S** = spatial pixels per track
- **B** = number of spectral bands

### Timestamp Fallbacks

If `processed/radiance/timestamps` is not found, the module tries:
1. `processed/timestamp`
2. `timestamp`
3. `timestamps`

If no timestamps are found, time display shows "N/A".

## 🔧 Classes

### GeoFile
Wraps a single HDF5 file, reads metadata and provides methods to extract georeferenced RGB data.

```python
gf = geo.GeoFile("/path/to/file.h5")
print(gf.name)          # Filename without extension
print(gf.shape)         # (T, S, B)
print(gf.has_georef)    # True/False
print(gf.timestamps)    # Unix timestamps array or None
```

### TransectDataSet
Discovers and manages multiple HDF5 files in a folder.

```python
transect = geo.TransectDataSet("/path/to/folder")
transect.list_files()                    # Print available files
cube = transect.select_all_files()       # Combine all files
cube = transect.select_files([...])      # Select specific files
cube = transect.select_files_by_pattern("rad_*")  # Pattern matching
```

### CombinedTransectCube
Combines multiple files into a single logical cube along the track axis.

```python
cube = transect.select_files([...])
print(cube.X_ecef.shape)    # (T_combined, S)
print(cube.timestamps)       # Combined timestamps
cube.plot_georef(...)        # Plot with options
```

## 🆚 Differences from Notebook

### dev3_timestamp_complete.py vs plot_gref_uhi.ipynb

| Feature | Notebook | dev3_timestamp_complete.py |
|---------|----------|---------------------------|
| Structure | Multiple cells | Single module |
| Execution | Cell-by-cell | Import and call |
| Cell tracking | Confusing cell numbers | Clear function/class names |
| Monkey-patching | Required | Built-in to class |
| Documentation | Scattered | Comprehensive docstrings |
| Reusability | Copy-paste cells | Simple import |

### What's the Same?

- ✅ Same timestamp logic (`_normalize_epoch_scalar`, `_read_ts_from_file`, `_utc_range_for_slice`)
- ✅ Same coordinate conversions (ECEF → NED, LATLON)
- ✅ Same RGB extraction and georeferencing
- ✅ Same interactive click functionality
- ✅ Same file boundary display

### What's Better?

- ✅ **No cell confusion** - Everything has clear names
- ✅ **No monkey-patching needed** - `plot_georef` is part of `CombinedTransectCube` class
- ✅ **Better error handling** - Descriptive error messages
- ✅ **Complete docstrings** - Every function documented
- ✅ **Example script** - Clear usage examples in `example_usage.py`

## 🐛 Troubleshooting

### "Import config could not be resolved"
This warning is harmless. The module tries to import `config.py` but falls back gracefully if not found. You can either:
- Add `# type: ignore` to suppress the warning (already done)
- Provide `folder_path` explicitly: `load_transect(folder_path="/path/to/files")`

### "No timestamps found"
If timestamps are missing from your HDF5 files, the plot will show `Time: N/A → N/A`. All other functionality works normally.

### "No georef dataset found"
Your HDF5 file must contain either:
- `processed/georef/points_ecef_crs` (gridded format, preferred), or
- `processed/georef/points_ecef` + `processed/georef/frame_indices` + `processed/georef/ray_indices` (legacy flat format)

### Module import fails
Make sure you're in the correct directory:
```python
import sys
sys.path.insert(0, r"C:\path\to\gref4hsi\final_act\dev")
import dev3_timestamp_complete as geo
```

## 📝 Example Output

When you run the module, you'll see:

```
🔄 Rebuilding grids from georef hits for selected files...
   • rad_uhi_20241029_115057_1
   • rad_uhi_20241029_115057_2
   • rad_uhi_20241029_115057_3
✅ Combined shapes: X/Y/Z (12450, 512), RGB (12450, 512)

💡 Interactive mode: Click on the plot to display coordinates
```

The plot title will show:
```
RGB georeferenced composite (Combined_OUTPUT_FOLDER) [NED]
origin @ 60.801158°, 10.712235°
Time: 2024-10-29 09:50:57 UTC → 2024-10-29 09:52:34 UTC
(Click to show coordinates)
```

## 🎓 Next Steps

1. **Test with your data:**
   ```python
   import dev3_timestamp_complete as geo
   transect = geo.load_transect()
   transect.list_files()
   cube = transect.select_all_files()
   cube.plot_georef(coordinate_system="NED")
   ```

2. **Try different track ranges:**
   ```python
   cube.plot_georef(track_start=8637, track_end=9709)
   ```

3. **Experiment with coordinate systems:**
   ```python
   cube.plot_georef(coordinate_system="LATLON")
   cube.plot_georef(coordinate_system="ECEF", use_local_origin=True)
   ```

4. **Customize RGB wavelengths:**
   ```python
   cube.plot_georef(red_wl=670, green_wl=550, blue_wl=450)
   ```

## 📞 Support

If you encounter issues:
1. Check that your HDF5 files have the expected structure
2. Verify timestamps exist: `print(gf.timestamps)` for a GeoFile instance
3. Try plotting without `track_start`/`track_end` first
4. Use `interactive=True` to inspect pixel coordinates

---

**Created:** 2024
**Purpose:** Consolidate notebook functionality into reusable Python module
**Status:** ✅ Complete and tested
