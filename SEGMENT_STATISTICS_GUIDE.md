# Segment-Specific Statistics Guide

## Overview
The `print_processing_statistics()` function now provides **comprehensive segment-specific statistics** in addition to full transect statistics. This feature extracts detailed navigation, coverage, and mission data **directly from H5 files** for the exact track range you're analyzing with `cube.plot_georef()`.

## What's New

### Enhanced Function Signature
```python
print_processing_statistics(stats_file_path=None, track_start=None, track_end=None)
```

### New Parameters
- `track_start` (int, optional): Starting track index for segment statistics (inclusive)
- `track_end` (int, optional): Ending track index for segment statistics (exclusive)

### Key Features
✅ **Direct H5 file reading**: Extracts actual navigation data from H5 files (not estimates)  
✅ **Comprehensive metrics**: Same level of detail as full transect statistics  
✅ **File transparency**: Shows exactly which H5 files were analyzed  
✅ **Accurate calculations**: Computes real statistics from segment data, not proportional estimates  

## Usage Examples

### Example 1: Full Transect Statistics (Original Behavior)
```python
# Get statistics for the entire transect
print_processing_statistics()
```

### Example 2: Segment Statistics (New Feature!)
```python
# Get statistics for tracks 8637 to 9709 (same range as your plot)
print_processing_statistics(track_start=8637, track_end=9709)
```

### Example 3: Combined with Plotting
```python
# Plot a specific segment
cube.plot_georef(
    coordinate_system="NED",
    show_file_boundaries=True,
    interactive=True,
    track_start=8637,
    track_end=9709,
)

# Get statistics for the SAME segment
print_processing_statistics(track_start=8637, track_end=9709)
```

## What Information Does Segment Statistics Provide?

When you provide `track_start` and `track_end`, you'll get a **comprehensive additional section** with the same level of detail as full transect statistics:

```
================================================================================
🎯 SEGMENT STATISTICS (tracks 8637 to 9709)
================================================================================

  📏 Segment Info:
    Track range: [8637, 9709) (length: 1072 tracks)
    Percentage of transect: 21.8%

  💡 Note: Extracting detailed navigation/coverage data from H5 files...

  🎯 Ray Tracing (segment):
    Total rays: 1,037,696
    Successful hits: 891,234
    Success rate: 85.89%

  ⏱️  Time Range (segment):
    Start: 2024-10-29 11:52:34 UTC
    End: 2024-10-29 11:54:12 UTC
    Duration: 98.0 s (1.6 min)

  🧭 Navigation (segment, from H5 files):
    Depth: 12.34 to 18.67 m (mean: 15.21 m, σ=1.45 m)
    Roll: -2.34° to 3.12° (mean: 0.45°, σ=0.89°)
    Pitch: -1.87° to 2.23° (mean: 0.12°, σ=0.67°)
    Heading: 245.3° to 268.9° (mean: 257.4°, σ=5.2°)

  � Mission Metrics (segment, from H5 files):
    Distance traveled: 145.3 m (0.145 km)
    Average speed: 1.48 m/s (5.33 km/h)

  📐 Coverage Metrics (segment, from H5 files):
    Swath width: 8.45 ± 1.23 m
    Swath range: [5.67, 11.23] m
    Total covered area: 1,227.5 m² (0.1228 hectares)

  📁 H5 Files Analyzed (detailed data extracted from):
    • rad_uhi_20241029_115057_4.h5
    • rad_uhi_20241029_115057_5.h5

  �📄 Files Involved in Segment:
    • rad_uhi_20241029_115057_4
    • rad_uhi_20241029_115057_5

  📸 HSI Data (segment):
    Frames: 1072
    Slits per frame: 968
    Bands: 462
    Total pixels: 1,037,696
```

### Key Metrics Explained

1. **Segment Info**
   - Track range: Shows the exact range you specified
   - Length: Number of tracks in the segment
   - Percentage: What portion of the full transect this segment represents

2. **Ray Tracing Statistics**
   - Total rays: Number of rays cast in this segment (from statistics file)
   - Successful hits: Number of rays that successfully hit the MBES surface
   - Success rate: Percentage of successful intersections (quality metric)

3. **Time Range**
   - Start/End: UTC timestamps for the segment
   - Duration: How long this segment took to acquire

4. **Navigation Data (from H5 files)**
   - **Depth**: Min/max/mean/std of depth measurements
   - **Roll**: Min/max/mean/std of roll angles (vehicle rotation)
   - **Pitch**: Min/max/mean/std of pitch angles (vehicle tilt)
   - **Heading**: Min/max/mean/std of heading angles (compass direction)
   - *Note: Extracted directly from `processed/navigation_data` in H5 files*

5. **Mission Metrics (from H5 files)**
   - **Distance traveled**: Calculated from ECEF positions in segment
   - **Average speed**: Computed from distance and duration
   - *Note: Calculated from `processed/georef/navigation_ecef` positions*

6. **Coverage Metrics (from H5 files)**
   - **Swath width**: Mean ± std of cross-track coverage width
   - **Swath range**: Min and max swath widths observed
   - **Total covered area**: Sum of coverage area for all frames
   - *Note: Read from `processed/coverage_metrics` in H5 files*

7. **H5 Files Analyzed**
   - **Explicit listing**: Shows which H5 files were opened and analyzed
   - **Transparency**: You know exactly where the data came from

8. **HSI Data Summary**
   - Frames, slits per frame, bands, total pixels in segment

## Important Notes

- ✅ **Non-destructive**: Original statistics unchanged; segment stats are **additional** information
- ✅ **Direct data extraction**: Reads actual navigation/coverage data from H5 files (not estimates!)
- ✅ **File transparency**: Shows which H5 files were analyzed
- ✅ **Accurate calculations**: Real statistics computed from actual segment data
- ✅ **File overlap handling**: Correctly handles segments spanning multiple H5 files
- 📊 **Same detail level**: Provides same metrics as individual file statistics
- 💾 **H5 file requirement**: Requires H5 files to exist in the output folder

## Practical Applications

1. **Quality Assessment**: Check if a specific area has good ray tracing success rates
2. **Time Correlation**: Match plotted segments with mission timeline
3. **Focused Analysis**: Analyze problem areas or regions of interest separately
4. **Report Generation**: Include segment-specific metrics in analysis reports
5. **Mission Planning**: Use navigation statistics to understand vehicle behavior in specific areas
6. **Coverage Validation**: Verify swath width and coverage area for specific regions
7. **Comparative Analysis**: Compare different segments within same transect

## Data Sources

The function extracts data from multiple sources:

### From Statistics JSON File (processing_statistics.json):
- Ray tracing totals (hits, rays, success rate)
- File processing information
- Time range estimates

### From H5 Files (*.h5 in output folder):
The function reads these HDF5 groups/datasets:
- `processed/navigation_data/depth` - Depth measurements
- `processed/navigation_data/roll` - Roll angles
- `processed/navigation_data/pitch` - Pitch angles
- `processed/navigation_data/heading` - Heading angles
- `processed/georef/navigation_ecef` - ECEF positions for distance calculation
- `processed/coverage_metrics/swath_width` - Cross-track swath width
- `processed/coverage_metrics/coverage_area_per_frame` - Coverage area per frame

**The function tells you which H5 files it reads from**, ensuring full transparency!

## Example Workflow

```python
# Step 1: Load transect
transect = load_transect(config.OUTPUT_FOLDER)
cube = transect.select_files([...])

# Step 2: Get full transect overview
print_processing_statistics()

# Step 3: Identify interesting region from plots
cube.plot_georef(coordinate_system="NED")

# Step 4: Focus on specific area (e.g., tracks 8637-9709)
cube.plot_georef(
    coordinate_system="NED",
    track_start=8637,
    track_end=9709,
)

# Step 5: Get detailed stats for that exact area
print_processing_statistics(track_start=8637, track_end=9709)
```

## Location in Code

**Modified file**: `gref4hsi/final_act/utils/gref_pipeline/georef.py`
- Function: `print_processing_statistics()` (line ~1967)
- New segment statistics section appears after "Combined Transect Statistics"

**Updated notebook**: `gref4hsi/final_act/notebooks/plot_gref_uhi.ipynb`
- Added markdown cell: "get stats for specific track segment (same as plotting range)"
- Added example code cell with `track_start` and `track_end` parameters
