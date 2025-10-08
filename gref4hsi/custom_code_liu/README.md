# Custom Code by Liu

This folder contains custom analysis scripts for the gref4hsi project.

## camera_imu_altitude_comparison.py

Compares altitude estimates between the IMU (body frame) and the camera frame, using the same geometric configuration as `gref4hsi/tests/test_eely.py`.

### Quick Start

**Option 1: Run from VS Code (RECOMMENDED)**
- Open the script in VS Code
- Press F5 or click "Run Python File" button in the top-right
- An interactive plot window will open automatically

**Option 2: Run from terminal**
```powershell
cd gref4hsi/custom_code_liu
python camera_imu_altitude_comparison.py
```

The plot will show:
- Top panel: IMU altitude, camera altitude, and range sensor readings vs. time
- Bottom panel: Difference between camera and IMU altitude

### Configuration

Edit these constants at the top of the script:

| Constant | Default | Description |
|----------|---------|-------------|
| `RUN_DIRECTLY` | `True` | Run without command-line arguments |
| `DEFAULT_STRIDE` | `5` | Subsample factor (1 = all data, 5 = every 5th point) |
| `SHOW_PLOT` | `True` | Display interactive plot window |
| `SAVE_PLOT` | `True` | Save PNG file to outputs/ |
| `H5_FOLDER` | `E:\mjosa_new\2024-10-29` | Path to H5 files |
| `H5_PATTERN` | `"rad_uhi_*.h5"` | Glob pattern for H5 files |
| `FILTER_BY_H5_TIME` | `True` | Only show nav data during H5 acquisitions |

### How H5 Time Filtering Works

When `FILTER_BY_H5_TIME = True`:
1. Script searches for all H5 files matching the pattern in `H5_FOLDER`
2. Extracts timestamp ranges from each H5 file (e.g., `1730202657.31` to `1730202724.05`)
3. Filters navigation data to only include times within those ranges (with 10s buffer)
4. Plots only show the transect periods when HSI data was being collected

Example output:
```
Searching for H5 files in E:\mjosa_new_oct_2025\use_gref4hsi\057_own_code_1to2\input...
  rad_uhi_20241029_115057_1.h5: 1730202657.31 to 1730202724.05
  rad_uhi_20241029_115057_2.h5: 1730202724.10 to 1730202821.56

Filtered navigation: 786 / 46534 records within H5 time ranges
```

**To see the full mission** instead of just the transects, set:
```python
FILTER_BY_H5_TIME = False  # Shows all navigation data
```

**To change which transects you're looking at**, update:
```python
H5_FOLDER = Path(r"E:\your\path\to\h5\files")  # Point to different folder
H5_PATTERN = "rad_uhi_*.h5"  # Or use more specific pattern like "rad_uhi_20241029_115057_*.h5"
```

### Outputs

- **Interactive plot**: Shows immediately when `SHOW_PLOT = True`
- **`outputs/camera_vs_imu_altitude.png`**: Saved figure (when `SAVE_PLOT = True`)
- **`outputs/camera_vs_imu_altitude_stats.csv`**: Summary statistics

### Command-Line Mode

Set `RUN_DIRECTLY = False` to use command-line arguments instead:

```powershell
python camera_imu_altitude_comparison.py --stride 10 --nav-csv path/to/nav.csv
```

### Geometry Configuration

The script uses these transforms from `test_eely.py`:
- **Camera lever arm**: `[2.5, 0.0, 0.0]` m (camera is 2.5m forward of IMU)
- **Camera rotation**: 90° clockwise around Z-axis (nadir-looking)
- **Origin**: `(10.7122345, 60.8011575, 0.0)` lon/lat/alt for NED frame

These values match the production georeferencing pipeline to ensure consistency.
