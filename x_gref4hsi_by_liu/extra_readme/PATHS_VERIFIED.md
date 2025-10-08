# Data Paths - Verified vs Guessed

This document tracks which H5 dataset paths were **verified** from the original gref4hsi codebase vs **guessed**.

## ✅ VERIFIED PATHS (from gref4hsi code/config)

### Input Paths (Reading from H5):

| Path | Purpose | Source |
|------|---------|--------|
| `processed/radiance/timestamp` | HSI frame timestamps | configuration_uhi.ini line 53 |
| `processed/radiance/dataCube` | Calibrated radiance cube | configuration_uhi.ini line 51 |
| `processed/nav/timestamp_hsi` | Alternative HSI timestamps | configuration_uhi.ini line 60 |
| `raw/nav/timestamp` | Raw navigation timestamp | configuration_uhi.ini line 44 |

### Output Paths (Writing to H5):

| Path | Purpose | Source |
|------|---------|--------|
| `processed/georef/points_ecef_crs` | Intersection points (ECEF) | configuration_uhi.ini line 74, georeference.py line 135 |
| `processed/georef/normals_hsi_frame` | Surface normals (HSI frame) | configuration_uhi.ini line 76 |
| `processed/georef/normals_ned_crs` | Surface normals (NED frame) | configuration_uhi.ini line 81 |

**Note**: The original gref4hsi uses a config file to define these paths, and the `write_intersection_geometry_2_h5_file()` function reads the config to know where to save data.

## ⚠️ CUSTOM PATHS (Added by x_gref4hsi_by_liu)

These are **not** in the original gref4hsi but were added for debugging/reference:

| Path | Purpose | Notes |
|------|---------|-------|
| `processed/georef/ray_indices` | Pixel (slit) index for each intersection | Custom - helps debug which pixels succeeded |
| `processed/georef/frame_indices` | Frame index for each intersection | Custom - helps debug temporal patterns |

**These are safe to add** - they don't interfere with existing gref4hsi workflows.

## ❌ INITIALLY WRONG GUESSES (Now Fixed)

### 1. Radiance Datacube Path
- **Initially guessed**: `processed/radiance` (❌ This is a GROUP, not a dataset!)
- **Actually is**: `processed/radiance/dataCube` (✅ This is the dataset)
- **Error**: `AttributeError: 'Group' object has no attribute 'shape'`
- **Fixed in**: main.py line 95-115

### 2. Timestamp Path Priority
- **Initially guessed**: `processed/times/hsi_time` as first option
- **Actually is**: `processed/radiance/timestamp` (standard UHI format)
- **Impact**: Would fail to find timestamps
- **Fixed in**: utils.py line 340-350

## 📋 Complete Path Hierarchy (from your H5 files)

Based on the error messages, your H5 files have this structure:

```
rad_uhi_20241029_115057_1.h5
├── processed/
│   ├── radiance/
│   │   ├── dataCube          [dataset: (frames, slits, bands)]
│   │   ├── timestamp         [dataset: (frames,)]
│   │   ├── exposureTime      [dataset: (frames,)]
│   │   └── calibration/
│   │       ├── spectral/
│   │       │   ├── band2Wavelength
│   │       │   └── fwhm
│   │       ├── radiometric/
│   │       │   ├── darkFrame
│   │       │   └── radiometricFrame
│   │       └── geometric/
│   │           ├── view_angles
│   │           └── fieldOfView
│   ├── nav/
│   │   └── timestamp_hsi
│   └── georef/              [created by georeferencing]
│       ├── points_ecef_crs  [we write this]
│       ├── ray_indices      [custom]
│       └── frame_indices    [custom]
└── rawdata/
    ├── hsi/
    │   └── datacube
    └── rgb/
        └── timestamp
```

## 🔍 How to Verify Paths in Your H5 Files

If you want to check the structure yourself:

### Option 1: Python
```python
import h5py

with h5py.File('your_file.h5', 'r') as f:
    # List all datasets recursively
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:50s} shape={obj.shape}")
    
    f.visititems(print_structure)
```

### Option 2: Command line (if h5dump installed)
```bash
h5dump -n your_file.h5  # List all paths
h5dump -d "processed/radiance/dataCube" your_file.h5  # Check specific dataset
```

### Option 3: HDFView GUI
Download from: https://www.hdfgroup.org/downloads/hdfview/

## 📝 Lessons Learned

1. **Always check if H5 path is a Group vs Dataset** before calling `.shape`
2. **Use `isinstance(obj, h5py.Dataset)` check** before accessing dataset attributes
3. **Follow existing conventions** - gref4hsi has well-defined paths in config files
4. **Document custom additions** - so users know what's standard vs custom

## ✅ Current Status

All paths are now **verified** and match gref4hsi conventions:
- ✅ Reading timestamps from correct path
- ✅ Reading radiance datacube from correct path  
- ✅ Saving intersections to standard gref4hsi path
- ✅ Type checking (Group vs Dataset) added
- ✅ Better error messages with path suggestions

**Ready to run!** 🚀
