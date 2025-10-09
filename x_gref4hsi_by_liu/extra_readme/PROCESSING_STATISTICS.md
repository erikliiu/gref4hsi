# Processing Statistics Documentation

## Overview

The `main.py` script now automatically saves comprehensive processing statistics to:
```
{OUTPUT_FOLDER}/processing_statistics.json
```

This JSON file contains all metrics useful for reports, analysis, and quality control.

## Statistics Structure

### 1. Processing Session
```json
"processing_session": {
  "start_time": "2025-10-09 14:40:03",
  "end_time": "2025-10-09 14:41:44",
  "duration_sec": 101.5,
  "duration_human": "1m 41s"
}
```
- Processing start and end times
- Total duration in seconds and human-readable format

### 2. Configuration
```json
"configuration": {
  "nav_csv": "path/to/nav.csv",
  "mbes_geotiff": "path/to/mbes.tif",
  "camera_calib_xml": "path/to/calib.xml",
  "output_folder": "path/to/output",
  "max_ray_length_m": 100.0,
  "early_failure_threshold_pct": 50.0,
  "epsg_mbes": 32632,
  "sensor_translation": [2.5, 2.48, 0.0],
  "sensor_rotation": [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
}
```
- All input file paths
- Processing parameters used
- Sensor mounting configuration

### 3. Navigation Data
```json
"navigation_data": {
  "total_records": 232670,
  "time_range": {
    "start": 1730196822.84,
    "end": 1730207699.44,
    "duration_sec": 10876.6
  }
}
```
- Total navigation records
- Time coverage
- Duration

### 4. MBES Mesh
```json
"mbes_mesh": {
  "n_vertices": 7390592,
  "n_faces": 7383936,
  "bounds": {
    "x_min": 592829.45,
    "x_max": 592857.59,
    "y_min": 6741790.71,
    "y_max": 6741895.67,
    "z_min": -136.01,
    "z_max": -110.83
  }
}
```
- Mesh size and complexity
- Spatial bounds (UTM coordinates)
- Elevation range

### 5. Camera Calibration
```json
"camera_calibration": {
  "focal_length": 930.63,
  "principal_point": 484.75,
  "width_pixels": 968,
  "distortion": {
    "k1": -228.02,
    "k2": 437.40,
    "k3": -0.00053
  }
}
```
- Lens parameters
- Image dimensions
- Distortion coefficients

### 6. Files Processed Summary
```json
"files_processed": {
  "total": 2,
  "successful": 2,
  "failed": 0,
  "success_rate_pct": 100.0
}
```
- Number of H5 files processed
- Success/failure counts
- Overall file processing success rate

### 7. Aggregated Statistics
```json
"aggregated_statistics": {
  "total_frames": 3279,
  "total_rays": 3174072,
  "total_hits": 3020391,
  "overall_success_rate_pct": 95.16
}
```
- Combined statistics across all files
- Total HSI frames georeferenced
- Total ray tracing attempts
- Overall georeferencing success rate

### 8. Individual File Statistics
```json
"individual_files": [
  {
    "filename": "rad_uhi_20241029_115057_1.h5",
    "processing_time": "2025-10-09 14:41:20",
    "hsi_data": {
      "n_frames": 1333,
      "n_slits": 968,
      "n_bands": 210,
      "time_range": {
        "start": 1730202657.31,
        "end": 1730202724.05,
        "duration_sec": 66.74
      }
    },
    "navigation": {
      "interpolated_positions": 1333,
      "lat_range": {"min": 60.801148, "max": 60.801212},
      "lon_range": {"min": 10.712228, "max": 10.712244},
      "depth_range": {"min": 118.46, "max": 119.17, "mean": 118.82},
      "attitude_ranges": {
        "roll": {"min": -2.45, "max": -1.23, "std": 0.31},
        "pitch": {"min": 0.12, "max": 0.89, "std": 0.18},
        "yaw": {"min": 179.76, "max": 179.89, "std": 0.03}
      }
    },
    "ray_tracing": {
      "total_rays": 1290344,
      "successful_hits": 1136664,
      "failed_rays": 153680,
      "success_rate_pct": 88.09,
      "trimesh_hits": 1136664,
      "pyvista_retries": 0,
      "pyvista_recoveries": 0,
      "max_ray_length_m": 100.0
    },
    "georef_coverage": {
      "frames_with_data": 1333,
      "coverage_pct": 88.09
    },
    "output": {
      "h5_file": "path/to/output/rad_uhi_20241029_115057_1.h5",
      "format": "gridded",
      "shape": [1333, 968, 3]
    }
  },
  // ... more files
]
```

For each file:
- **HSI Data**: Frame count, spatial resolution, spectral bands, time coverage
- **Navigation**: Position ranges, depth statistics, attitude stability (roll/pitch/yaw variability)
- **Ray Tracing**: Success rates, recovery statistics, quality metrics
- **Coverage**: Spatial coverage quality
- **Output**: File paths and format details

## Key Metrics for Reports

### Quality Metrics
- `overall_success_rate_pct`: Overall georeferencing quality (aim for >90%)
- `failed_rays`: Number of points without terrain intersections
- `pyvista_recoveries`: Rays recovered by secondary ray tracer

### Coverage Metrics
- `frames_with_data`: Number of HSI frames with valid georeferencing
- `coverage_pct`: Percentage of pixels successfully georeferenced

### Stability Metrics
- `attitude_ranges.roll/pitch/yaw.std`: Platform stability during acquisition (lower is better)
- `depth_range.std`: Depth variation (platform steadiness)

### Performance Metrics
- `duration_sec`: Processing time
- `total_rays`: Computational load
- `n_vertices`, `n_faces`: Mesh complexity

## Usage Examples

### Load and Analyze Statistics
```python
import json
import numpy as np

# Load statistics
with open('output/processing_statistics.json', 'r') as f:
    stats = json.load(f)

# Get overall success rate
success_rate = stats['aggregated_statistics']['overall_success_rate_pct']
print(f"Overall georeferencing success: {success_rate:.2f}%")

# Analyze per-file quality
for file in stats['individual_files']:
    if file.get('status') != 'failed':
        print(f"{file['filename']}: {file['ray_tracing']['success_rate_pct']:.1f}%")

# Check platform stability
for file in stats['individual_files']:
    if file.get('status') != 'failed':
        roll_std = file['navigation']['attitude_ranges']['roll']['std']
        pitch_std = file['navigation']['attitude_ranges']['pitch']['std']
        print(f"{file['filename']}: Roll±{roll_std:.2f}°, Pitch±{pitch_std:.2f}°")
```

### Generate Report Tables
```python
import pandas as pd

# Create summary table
data = []
for file in stats['individual_files']:
    if file.get('status') != 'failed':
        data.append({
            'File': file['filename'],
            'Frames': file['hsi_data']['n_frames'],
            'Total Rays': file['ray_tracing']['total_rays'],
            'Success Rate (%)': file['ray_tracing']['success_rate_pct'],
            'Mean Depth (m)': file['navigation']['depth_range']['mean']
        })

df = pd.DataFrame(data)
print(df.to_markdown(index=False))
```

## Notes

- Statistics are saved **after each processing run** (overwrites previous)
- Times are in local system time
- Coordinates are in the MBES CRS (typically UTM)
- All distances in meters, all angles in degrees
- Failed files include error information in the `error` field
