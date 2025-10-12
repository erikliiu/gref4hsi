# Illumination Correction with Automatic Persistence

## Overview

The `apply_illumination_correction()` method now automatically saves and loads corrected data to/from HDF5 files, making subsequent runs much faster!

## Features

### ✅ Automatic Check
- Checks if correction with the same parameters already exists
- Compares `window_size` and `strength` parameters

### 💾 Automatic Save
- Saves corrected data to each HDF5 file after computing
- Stored as `processed/radiance/dataCube_illum_corrected`
- Includes metadata: `window_size`, `strength`, `description`

### 📂 Automatic Load
- Loads from disk if parameters match
- Much faster than recomputing!

## Usage

### Basic Usage (Recommended)
```python
# First run: computes and saves (slow - several minutes)
cube.apply_illumination_correction(window_size=1000, strength=1.0)

# Subsequent runs: loads from disk (fast - seconds!)
cube.apply_illumination_correction(window_size=1000, strength=1.0)
```

### Check if Already Computed
```python
# Check if correction exists
has_corr = cube.has_illumination_correction(window_size=1000, strength=1.0)
print(f"Already computed? {has_corr}")
```

### Manual Load (Optional)
```python
# Manually load if you know it exists
if cube.has_illumination_correction(window_size=1000, strength=1.0):
    cube.load_illumination_correction()
```

### Force Recompute
```python
# Force recompute even if saved version exists
cube.apply_illumination_correction(window_size=1000, strength=1.0, force_recompute=True)
```

### Manual Save (Optional)
```python
# Manually save after computing
cube.apply_illumination_correction(window_size=1000, strength=1.0)
cube.save_illumination_correction(window_size=1000, strength=1.0)
```

## Data Storage

### File Location
The corrected data is stored in each HDF5 file:
```
<file>.h5/
  └── processed/
      └── radiance/
          ├── dataCube                      # Original
          ├── dataCube_corrected            # From radiometric correction
          └── dataCube_illum_corrected      # NEW! From illumination correction
```

### Metadata
Each saved dataset includes attributes:
- `window_size`: Window size parameter used
- `strength`: Strength parameter used
- `description`: "Illumination-corrected radiance data"

## Parameters

### window_size
- `None` or `<= 1` or `>= T_total`: Global correction (single median per slit-band)
- `int`: Rolling median across tracks of specified length (e.g., 1000)

### strength
- Float in range [0, 1]
- `0.0`: No correction (returns original)
- `1.0`: Full correction
- `0.5`: Blend of 50% original + 50% corrected

### force_recompute
- `False` (default): Load from disk if available
- `True`: Recompute even if saved version exists

## Performance

### First Run (Computing)
- Time: ~2-5 minutes (depends on data size)
- Progress bar shows slit-by-slit processing
- Automatically saves to disk

### Subsequent Runs (Loading)
- Time: ~5-10 seconds
- No computation needed
- Loads from disk

## Example Workflow

```python
# Load transect
transect = georef.load_transect(config.OUTPUT_FOLDER)
cube = transect.select_files(['file_1', 'file_2'])

# First time: computes and saves
print("First run:")
cube.apply_illumination_correction(window_size=1000, strength=1.0)
# 🔄 Computing illumination correction: rolling (window=1000), strength=1.0
#    Processing slits: 100%|████████████| 968/968
# ✅ data_corrected ready (rolling (window=1000))
# 💾 Saving illumination correction to 2 files...
#    ✓ file_1: saved (2463, 968, 210)
#    ✓ file_2: saved (2462, 968, 210)
# ✅ Illumination correction saved to disk

# Next time: loads instantly
print("\nSecond run:")
cube.apply_illumination_correction(window_size=1000, strength=1.0)
# ✅ Illumination correction already applied with window=1000, strength=1.0
#    Loading from disk...
# 📂 Loading saved illumination correction from 2 files...
#    file_1: window=1000, strength=1.0
#    file_2: window=1000, strength=1.0
# ✅ Loaded illumination correction from disk
```

## Benefits

1. **Speed**: Subsequent runs are 30-60x faster
2. **Reproducibility**: Same parameters always produce same results
3. **Disk-based caching**: Results persist across Python sessions
4. **No manual tracking**: Automatic parameter matching
5. **Safe**: Original data never modified

## Notes

- The correction is computed per-slit, per-band using median filtering
- Different parameter combinations are stored separately
- Safe to call multiple times with same parameters
- Original `dataCube` is never modified
- Uses GZIP compression (level 4) for efficient storage

## See Also

- `apply_illumination_correction_method2()`: Alternative method
- `apply_illumination_correction_method3()`: Alternative method  
- Original implementation: `utils/georef.py` lines 1385-1470
