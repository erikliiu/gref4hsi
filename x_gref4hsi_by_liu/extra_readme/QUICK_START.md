# Quick Start Guide

## Files Created

1. **config.py** - All configuration settings
2. **utils.py** - Utility functions (navigation, coordinates, camera)
3. **georeference_mbes.py** - MBES loading and ray tracing
4. **main.py** - Main processing script
5. **requirements.txt** - Python dependencies
6. **README.md** - Full documentation

## Before Running

### 1. Check config.py paths:

```python
NAV_CSV = r"E:\mjosa_new\navigation_data\nav_data_merged.csv"
MBES_GEOTIFF = r"E:\mjosa_new\DTM\geotiff_2.tif"
H5_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\057\input\H5"
OUTPUT_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\057\output_liu"
CAMERA_CALIB_XML = r"C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\gref4hsi\Input\Calib\HSI_2b.xml"
```

### 2. Verify CSV columns:

Open your CSV and check column names. Update `CSV_COLUMNS` in config.py:

```python
CSV_COLUMNS = {
    'timestamp': 'timestamp [unix epoch s]',
    'latitude': 'latitude [deg]',
    'longitude': 'longitude [deg]',
    'depth': 'depth [m]',
    'roll': 'roll [deg]',
    'pitch': 'pitch [deg]',
    'yaw': 'yaw [deg]',
    'altitude': 'altitude [m]'
}
```

### 3. Verify sensor configuration:

If your sensor has different mounting, update:

```python
ROTATION_HSI_TO_BODY = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
TRANSLATION_BODY_TO_HSI = np.array([2.5, 0, 0])
TIME_OFFSET_SEC = -508
```

## Running

```bash
# Activate environment
conda activate liu_gref  # or your env name

# Navigate to project folder
cd "C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\gref4hsi\x_gref4hsi_by_liu"

# Run
python main.py
```

## Expected Output

```
================================================================================
UHI + MBES GEOREFERENCING
================================================================================

STEP 1: Loading Navigation CSV
Loaded 232,670 navigation records from ...

STEP 2: Loading MBES GeoTIFF
=== MBES GeoTIFF Metadata ===
Shape: (5000, 4000) (rows × cols)
CRS: EPSG:32632
Elevation range: -150.00 to -10.00 m

STEP 3: Converting GeoTIFF to Mesh
Creating mesh from 18,500,000 valid points...
Mesh created: 18,500,000 vertices, 37,000,000 faces

STEP 4: Loading Camera Calibration
Camera: f=8.50, cx=512.00, w=1024.00

STEP 5: Finding H5 Files
Found 3 H5 files:
  1. rad_uhi_20241029_115057_1.h5
  2. rad_uhi_20241029_115057_2.h5
  3. rad_uhi_20241029_115057_3.h5

STEP 6: Processing H5 Files

[File 1/3]
================================================================================
Processing: rad_uhi_20241029_115057_1.h5
================================================================================

Loaded 2000 HSI frames
...
RAY TRACING TO MBES MESH
...
Successful intersections: 2,048,000/2,048,000 (100.00%)

PROCESSING SUMMARY
Total HSI frames: 2,000
Successful intersections: 2,048,000
Success rate: 100.00%
Output file: E:\...\output_liu\rad_uhi_20241029_115057_1.h5
================================================================================

[Repeat for files 2 and 3]

FINAL SUMMARY
Total files: 3
Successful: 3
Failed: 0
Output directory: E:\...\output_liu
```

## If Something Goes Wrong

### Error: "Ray tracing failed: 74.3% of rays missed mesh"

**Cause**: MBES mesh doesn't cover HSI field of view

**Solutions**:
1. Check navigation time range covers H5 timestamps
2. Check MBES GeoTIFF covers survey area
3. Verify TIME_OFFSET_SEC is correct
4. Check coordinate systems match (EPSG codes)

### Error: "Column 'timestamp [unix epoch s]' not found in CSV"

**Cause**: CSV column names don't match

**Solution**: Update `CSV_COLUMNS` in config.py to match your CSV

### Error: "No timestamp dataset found in H5 file"

**Cause**: Script can't find HSI timestamps in H5 file

**Solution**: 
1. Inspect H5 structure: `h5dump -n your_file.h5`
2. Add the correct path to `utils.load_h5_timestamps()` function

### Import errors on Windows

**Solution**: Use conda-forge:
```bash
conda install -c conda-forge rasterio gdal pyproj pyvista trimesh
```

## Key Differences from Original Workflow

| Original gref4hsi | x_gref4hsi_by_liu |
|-------------------|-------------------|
| Creates DEM from altimeter | Uses MBES GeoTIFF directly |
| Multiple processing stages | Single streamlined script |
| Complex file dependencies | Simple modular structure |
| Prone to sequential failures | Reliable batch processing |

## Next Steps After Georeferencing

Once georeferencing succeeds, you can:

1. **Visualize intersections**:
   - Load H5 file: `processed/georef/points_ecef_crs`
   - Convert ECEF to UTM for plotting
   - Overlay on MBES mesh

2. **Orthorectification**:
   - Use georeferenced points to create orthorectified mosaic
   - Resample hyperspectral cube to regular grid

3. **Analysis**:
   - Extract spectra at georeferenced locations
   - Co-register with other datasets

## Support

For issues specific to this simplified workflow, check:
1. This QUICK_START.md
2. README.md for detailed documentation
3. Comments in config.py, utils.py, georeference_mbes.py, main.py

For general UHI/gref4hsi questions, refer to original repository documentation.
