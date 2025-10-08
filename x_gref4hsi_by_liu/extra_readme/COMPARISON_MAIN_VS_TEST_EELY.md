# Comparison: main.py vs test_eely.py Output

## Summary

**NO, your `main.py` does NOT output the same data as `test_eely.py`.**

Your `main.py` outputs a **minimal subset** (just ECEF intersection points + indices), while `test_eely.py` outputs **full ancillary georeferencing data** (normals, angles, timestamps, etc.).

---

## Detailed Comparison

### Your `main.py` Output (Minimal)

**Location:** `processed/georef/` group in H5 file

**Datasets saved:**
```
✓ points_ecef_crs      (N, 3)    ECEF intersection points [X, Y, Z]
✓ ray_indices          (N,)      Pixel/slit indices (which ray hit)
✓ frame_indices        (N,)      Frame indices (which frame)
```

**Total:** 3 datasets

**Code location:** `x_gref4hsi_by_liu/utils/utils.py::save_intersection_to_h5()`

---

### test_eely.py Output (Full)

**Location:** `processed/georef/` group in H5 file

**Datasets saved (from config):**
```
✓ points_ecef_crs      (T, S, 3) or (N, 3)    ECEF intersection points
✓ points_hsi_crs       (T, S, 3)              Points in HSI camera frame
✓ normals_hsi_crs      (T, S, 3)              Surface normals in HSI frame
✓ normals_ned_crs      (T, S, 3)              Surface normals in NED frame
✓ theta_v              (T, S)                 View zenith angles
✓ theta_s              (T, S)                 Sun zenith angles
✓ phi_v                (T, S)                 View azimuth angles
✓ phi_s                (T, S)                 Sun azimuth angles
✓ unix_time_grid       (T, S)                 Timestamps for each pixel
✓ pixel_nr_grid        (T, S)                 Pixel indices
✓ frame_nr_grid        (T, S)                 Frame indices
✓ hsi_tide_gridded     (T, S)                 Tide corrections (if available)
✓ hsi_alts_msl         (T, S)                 Altitudes above MSL
```

**Total:** 13 datasets (typical)

**Code location:** `gref4hsi/scripts/georeference.py::write_intersection_geometry_2_h5_file()`

---

## Key Differences

| Aspect | main.py | test_eely.py |
|--------|---------|--------------|
| **Primary goal** | Get intersection points for visualization | Full georeferencing for orthorectification |
| **Data structure** | Flattened (N, 3) - only successful rays | Gridded (T, S, 3) - preserves hypercube structure |
| **Ancillary data** | None (just points) | Full geometry (normals, angles, times) |
| **Orthorectification ready** | ❌ No | ✅ Yes |
| **Sun angles** | ❌ No | ✅ Yes |
| **Surface normals** | ❌ No | ✅ Yes |
| **Tide corrections** | ❌ No | ✅ Yes (if configured) |
| **Preserves grid structure** | ❌ No (flattened) | ✅ Yes (T×S grid) |

---

## Why the Difference?

### main.py Philosophy:
- **Purpose:** Quick georeferencing for MBES + HSI visualization
- **Focus:** Get 3D intersection points in ECEF
- **Use case:** Interactive maps, point cloud visualization, coverage analysis
- **Trade-off:** Minimal data, fast processing, no orthorectification support

### test_eely.py Philosophy:
- **Purpose:** Full hyperspectral processing pipeline
- **Focus:** Preserve all geometry for advanced radiometric corrections
- **Use case:** Orthorectification → ENVI datacubes → Remote sensing analysis
- **Trade-off:** More data, slower processing, ready for scientific analysis

---

## What's Missing in main.py?

If you want to do orthorectification (like test_eely does), you need to add:

1. **Surface normals** (`normals_ned_crs`, `normals_hsi_crs`)
   - Needed for: Bidirectional reflectance correction, shading effects
   
2. **View geometry** (`theta_v`, `phi_v`)
   - Needed for: BRDF corrections, angular analysis
   
3. **Sun geometry** (`theta_s`, `phi_s`)
   - Needed for: Illumination corrections, shadow detection
   
4. **Gridded structure** (T, S, ...) instead of flattened (N, ...)
   - Needed for: Maintaining hypercube structure for orthorectification
   
5. **Timestamps per pixel** (`unix_time_grid`)
   - Needed for: Temporal analysis, motion blur corrections
   
6. **Camera frame points** (`points_hsi_crs`)
   - Needed for: Debugging, sensor-relative analysis

---

## Can Your Data Be Used for Orthorectification?

**Technically YES, but with limitations:**

Your `points_ecef_crs` contains the essential intersection points in ECEF, which is the minimum requirement. However:

### ❌ Missing for full orthorectification:
- No surface normals → can't do BRDF corrections
- No sun angles → can't do illumination corrections
- No gridded structure → harder to resample to regular grid
- No timestamps per pixel → can't do temporal corrections

### ✅ You CAN still do:
- **Basic orthorectification** (geometric correction only)
- **Flattened point cloud → raster** conversion
- **Visualization in GIS software**
- **Interactive maps** (like your `dev2_html.py`)

---

## Example: Actual Data Shapes

### Your main.py Output:
```python
processed/georef/points_ecef_crs    (1290344, 3)    # Flattened
processed/georef/ray_indices        (1290344,)
processed/georef/frame_indices      (1290344,)
```

### Expected test_eely Output (for same file):
```python
processed/georef/points_ecef_crs    (1333, 968, 3)  # Gridded (T=1333 frames, S=968 slits)
processed/georef/normals_ned_crs    (1333, 968, 3)
processed/georef/theta_v            (1333, 968)
processed/georef/phi_v              (1333, 968)
processed/georef/unix_time_grid     (1333, 968)
...
```

Notice:
- **main.py:** 1,290,344 successful rays out of 1333×968 = 1,290,344 total rays (100% hit rate!)
- **test_eely:** Would store ALL rays (including misses), preserving grid structure

---

## Recommendations

### If you want to keep main.py minimal (current approach):
✅ **Good for:**
- Quick visualization
- Interactive maps
- Point cloud analysis
- Coverage checking

❌ **Cannot do:**
- Full orthorectification pipeline
- BRDF corrections
- Advanced radiometric processing

### If you want to match test_eely (full pipeline):
You need to modify `main.py` to:

1. **Store ALL rays** (not just successful ones) → preserves (T, S) grid
2. **Compute surface normals** from MBES mesh
3. **Compute view angles** (theta_v, phi_v) from ray directions
4. **Add sun angles** (theta_s, phi_s) - requires sun position calculation
5. **Grid the data** (T, S, ...) instead of flattening (N, ...)
6. **Add timestamps per pixel** (replicate frame timestamps)

This would make your code more complex but compatible with orthorectification.

---

## Conclusion

Your `main.py` is a **simplified, focused tool** for georeferencing HSI to MBES bathymetry. It produces the **minimum necessary output** for visualization but **not the full ancillary data** needed for advanced orthorectification workflows.

If your goal is visualization and interactive maps → **keep it as is** ✅  
If your goal is full processing pipeline → **need to add ancillary data** 📊
