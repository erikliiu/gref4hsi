# Simplified UHI Georeferencing with MBES

A streamlined workflow for georeferencing UHI hyperspectral data using MBES GeoTIFF directly.

## Key Differences from Original gref4hsi:

1. **Uses MBES GeoTIFF directly** - No DEM creation from altimeter data
2. **Simplified workflow** - Only essential steps for georeferencing
3. **CSV navigation input** - Direct loading of processed navigation data
4. **Clear separation** - Utils, georeferencing, and main script

## Project Structure:

```
x_gref4hsi_by_liu/
├── README.md                 # This file
├── config.py                 # Configuration and paths
├── utils.py                  # Navigation loading, coordinate transforms
├── georeference_mbes.py      # Ray tracing against MBES mesh
├── main.py                   # Main processing script
└── requirements.txt          # Python dependencies
```

## Installation:

1. **Create conda environment** (recommended):
   ```bash
   conda create -n liu_gref python=3.10
   conda activate liu_gref
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Windows note**: You may need prebuilt wheels for GDAL/rasterio:
   - Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/
   - Or use: `conda install -c conda-forge rasterio gdal`

## Configuration:

Edit `config.py` to set your paths:

```python
# Input paths
NAV_CSV = r"E:\mjosa_new\navigation_data\nav_data_merged.csv"
MBES_GEOTIFF = r"E:\mjosa_new\DTM\geotiff_2.tif"
H5_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\057\input\H5"

# Output path
OUTPUT_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\057\output_liu"

# Sensor configuration
ROTATION_HSI_TO_BODY = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
TRANSLATION_BODY_TO_HSI = np.array([2.5, 0, 0])  # 2.5m forward
TIME_OFFSET_SEC = -508

# Camera calibration
CAMERA_CALIB_XML = r"C:\path\to\HSI_2b.xml"
```

**Important**: Verify CSV column names in `CSV_COLUMNS` match your file.

## Workflow:

1. Load navigation CSV (timestamps, position, attitude, altitude)
2. Load MBES GeoTIFF and convert to 3D mesh
3. Load UHI H5 files (hyperspectral cubes + timestamps)
4. Interpolate navigation to HSI frame times
5. Ray trace from camera to MBES mesh
6. Save georeferenced intersection points to H5 files

## Usage:

```bash
python main.py
```

The script will:
1. Load navigation from CSV
2. Load MBES GeoTIFF and create mesh (saved as `mbes_mesh.ply`)
3. Process all H5 files in `H5_FOLDER`
4. Save georeferenced results to `OUTPUT_FOLDER`

## Output:

For each input H5 file, creates georeferenced version with:
- `processed/georef/points_ecef_crs` - Intersection points (ECEF)
- `processed/georef/ray_indices` - Pixel indices
- `processed/georef/frame_indices` - Frame indices

## Troubleshooting:

### Ray tracing failures (>30% rays miss)

**Causes**:
- MBES mesh doesn't cover HSI field of view
- Coordinate system mismatch
- Wrong time offset or sensor offsets

**Solutions**:
1. Check MBES GeoTIFF bounds vs navigation extents
2. Verify EPSG codes match (script warns if mismatch)
3. Check `TIME_OFFSET_SEC` is correct
4. Visualize MBES mesh: `mbes_mesh.ply` in CloudCompare

### Module import errors

Windows users: Use conda-forge for spatial packages:
```bash
conda install -c conda-forge rasterio gdal pyproj pyvista
```

## Requirements:

- numpy>=1.20.0, pandas>=1.3.0, h5py>=3.0.0
- rasterio>=1.2.0 (GeoTIFF)
- pyvista>=0.37.0, trimesh>=3.10.0 (ray tracing)
- pyproj>=3.2.0 (coordinate transforms)
- scipy>=1.7.0, tqdm>=4.60.0, xmltodict>=0.12.0
