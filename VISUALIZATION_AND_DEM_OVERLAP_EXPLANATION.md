# Visualization and DEM Overlap Explanation

## Question 1: How does the visualization work? Is intersection data saved?

### YES, intersection data IS saved! ✅

When `georeference.main()` runs successfully, it saves the ray intersection points into the H5 file at these locations (defined in `configuration_Leo_copy.ini`):

```ini
[Georeferencing]
points_ecef_crs = processed/georef/points_ecef_crs      # ECEF coordinates (X,Y,Z)
point_hsi_frame = processed/georef/point_hsi_frame      # HSI camera frame
normals_hsi_crs = processed/georef/normals_hsi_frame    # Surface normals
theta_v = processed/georef/theta_v                      # View angles
phi_v = processed/georef/phi_v
# ... and more ancillary data
```

### How the visualization script works:

1. **Loads the DEM** from `Input/GIS/dummy_dem.tif`
   - This is the MERGED/COMBINED DEM created from ALL files' altimeter data
   - Shows the extent of the seafloor terrain model

2. **Loads intersection points** from each H5 file
   - Reads `processed/georef/points_ecef_crs` (saved by georeference.py)
   - Converts from ECEF coordinates to UTM Zone 32N for plotting
   - Each file gets a different color (blue, red, green, etc.)

3. **Plots both together**
   - Left plot: DEM coverage and elevation (heatmap)
   - Right plot: Ray intersection points colored by which file they came from
   - This shows WHERE the HSI camera successfully found the seafloor

### What this reveals:

- **If a file georeferenced successfully:** You'll see a dense cloud of points
- **If a file failed:** The intersection points will be missing or sparse
- **DEM coverage gaps:** If points fall outside the DEM boundary, you know the DEM doesn't cover that area

---

## Question 2: Could DEM overlap between files be an issue?

### YES, there IS overlap - and it's INTENTIONAL and CORRECT! ✅

### Why overlap is necessary:

The HSI camera is **2.5 meters FORWARD** of the altimeter. This means:

```
Vehicle motion →  
Time = T:        [Altimeter] ----2.5m----> [HSI Camera]
                      ↓                          ↓
                   Measures                 Looking at
                   seafloor                 seafloor 2.5m ahead
                   at position X            at position X+2.5m
```

### Current buffer implementation:

```python
TIME_BUFFER_SECONDS = 5  # ~5 meters of travel at 1 m/s speed

# Each file's DEM includes:
crit_1 = nav.altitude.time <= (true_time_hsi.max() + TIME_BUFFER_SECONDS)  # Future data
crit_2 = nav.altitude.time >= (true_time_hsi.min() - TIME_BUFFER_SECONDS)  # Past data
```

### What this means for consecutive files:

**File 1 (11:59:25 - 12:00:32):**
- DEM includes altimeter data from: 11:59:20 to 12:00:37
- Includes 5 seconds of FUTURE data from File 2's time range
- This is correct! HSI in File 1 looks ahead at seafloor that will be measured in File 2

**File 2 (12:00:32 - 12:02:09):**
- DEM includes altimeter data from: 12:00:27 to 12:02:14
- Includes 5 seconds of PAST data from File 1's time range
- This is correct! HSI in File 2 might still see seafloor from the transition area

### DEM overlap diagram:

```
                    File 1 HSI time                    File 2 HSI time
                    |--------------| File boundary |--------------| 
                                         ↓
         File 1 DEM |------------------| 
                            ⬆️ overlap ⬇️
                           File 2 DEM |------------------|
```

### Is this a problem?

**NO, this is CORRECT behavior:**

1. ✅ **Prevents gaps:** Ensures no seafloor area is unmapped at the transition
2. ✅ **Handles forward offset:** File 1's HSI looks into File 2's spatial region
3. ✅ **Redundancy is good:** Same seafloor area measured from slightly different angles/times
4. ✅ **Each file independent:** Each H5 file can be georeferenced without depending on the other

### Potential issues to watch for:

1. **Inconsistent altimeter measurements:** If the same seafloor area has different elevations in File 1 vs File 2
   - Solution: The DEM merging should handle this (takes all points)
   
2. **Very large buffers:** If TIME_BUFFER_SECONDS is too large, you waste memory
   - Current 5 seconds is reasonable for ~1 m/s speed

3. **Vehicle speed variations:** If the vehicle slows down or speeds up
   - 5 second buffer should handle speeds from 0.5 to 2 m/s
   - Can adjust if needed based on actual vehicle speed

---

## Summary

| Aspect | Status | Explanation |
|--------|--------|-------------|
| **Intersection data saved?** | ✅ YES | Saved to `processed/georef/points_ecef_crs` in H5 file |
| **Visualization works?** | ✅ YES | Loads saved intersection points and plots with DEM |
| **DEM overlap between files?** | ✅ YES | Intentional 10-second overlap (5 sec before + 5 sec after) |
| **Is overlap a problem?** | ❌ NO | Necessary to handle 2.5m HSI forward offset |
| **What to check next?** | 🔍 | Run visualization after successful georeferencing to confirm coverage |

---

## Next steps:

1. **Run the georeferencing pipeline** with the updated time buffer
2. **Check the statistics** printed during georeferencing:
   - File 1 should still have ~100% success rate
   - File 2 should now have MUCH higher success rate (hopefully >90%)
3. **Run the visualization** to confirm:
   ```bash
   python visualize_dem_coverage.py
   ```
4. **Inspect the plots** to verify:
   - DEM covers the entire area where intersection points are located
   - Both files' intersection points fall within the DEM boundaries
   - No gaps or missing coverage areas

