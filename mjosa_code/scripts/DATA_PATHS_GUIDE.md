# Data Paths Guide for Visualization Scripts

## 📂 Data Location Overview

The visualization scripts in `mjosa_code/scripts/` need several data files to run. Here's where everything is located:

---

## 🗺️ Current Data Paths (from config.py)

### **Navigation Data**
```
E:\mjosa_complete\data\processed\navigation\nav_data_merged.csv
```
- **What it is:** Corrected navigation with dead-reckoned transects
- **Used by:** All 3 scripts (create_html.py, create_sim.py, create_stability_plot.py)
- **How to update:** Already updated in `mjosa_code/utils/common/config.py` → `MERGED_CSV_FILE`

---

### **MBES Bathymetry GeoTIFF** ⚠️ YOU CHOOSE THIS

**Option 1: Full Coverage (RECOMMENDED)**
```
E:\mjosa_new_oct_2025\anxillary_data\EIVA\all.tif
```
- **Coverage:** Complete MBES survey of Lake Mjøsa area
- **File size:** Larger file
- **Use for:** Complete visualizations with full bathymetry context

**Option 2: Partial Coverage**
```
E:\mjosa_new\DTM\geotiff_2.tif
```
- **Coverage:** Smaller area (subset of full survey)
- **File size:** Smaller file  
- **Use for:** Testing or focused area visualization

**Option 3: Test File**
```
E:\mjosa_new\DTM\104921.tif
```
- **Note:** Original comment says "the fucked one to check orientation"
- **Use for:** Debugging orientation issues (not recommended for final viz)

**How to choose:**
1. Open `mjosa_code/utils/common/config.py`
2. Find the `MBES_GEOTIFF` variable (around line 40)
3. Uncomment the option you want to use
4. Comment out the others

**Current setting:** Option 1 (all.tif) - Full coverage ✅

---

### **Georeferenced Hyperspectral Data (H5 Files)**
```
E:\mjosa_new_oct_2025\use_gref4hsi\057_final\input\
```
- **What it is:** Folder containing georeferenced .h5 hyperspectral files
- **Used by:** create_sim.py, create_stability_plot.py
- **Files inside:** `rad_uhi_20241029_115057_*.h5` (6 files for 057_final)

---

### **Output Folder (Where Plots Are Saved)**
```
E:\mjosa_new_oct_2025\use_gref4hsi\057_final\output\
```
- **What it is:** Destination for all generated plots and HTML files
- **Used by:** All 3 scripts

---

### **Camera Calibration XML**
```
E:\mjosa_new_oct_2025\anxillary_data\HSI_2_body.xml
```
- **What it is:** HSI camera calibration parameters and body-to-camera transform
- **Used by:** create_sim.py

---

### **ROS Database (Optional)**
```
E:\mjosa\29\log_files\LOG_2024-10-29_10-13-32.db3
```
- **What it is:** ROS2 bag database with DVL and full 6DOF data
- **Used by:** create_stability_plot.py (optional - for DVL analysis)

---

## 🔧 How to Change MBES GeoTIFF

### Method 1: Edit config.py directly

Open: `mjosa_code/utils/common/config.py`

Find this section (around line 40):
```python
# MBES Bathymetry GeoTIFF Options:
# Option 1: Full MBES coverage from EIVA (recommended for visualization)
MBES_GEOTIFF = Path(r"E:\mjosa_new_oct_2025\anxillary_data\EIVA\all.tif")
# Option 2: Partial coverage (if you want to test with smaller area)
# MBES_GEOTIFF = Path(r"E:\mjosa_new\DTM\geotiff_2.tif")
```

**To switch to Option 2:**
```python
# Option 1: Full MBES coverage from EIVA (recommended for visualization)
# MBES_GEOTIFF = Path(r"E:\mjosa_new_oct_2025\anxillary_data\EIVA\all.tif")
# Option 2: Partial coverage (if you want to test with smaller area)
MBES_GEOTIFF = Path(r"E:\mjosa_new\DTM\geotiff_2.tif")
```

### Method 2: Use your own GeoTIFF

If you have a different MBES GeoTIFF file:
```python
MBES_GEOTIFF = Path(r"E:\your\path\to\your_bathymetry.tif")
```

**Requirements:**
- Must be a GeoTIFF file with proper georeferencing
- Should cover the area where your UHI transects are located
- Coordinate system should match `EPSG_MBES` (default: UTM 32N, EPSG:32632)

---

## 📊 What Each Script Needs

### create_html.py (Interactive Folium Map)
**Required:**
- ✅ NAV_CSV: `E:\mjosa_complete\data\processed\navigation\nav_data_merged.csv`
- ✅ MBES_GEOTIFF: Choose from options above
- ✅ OUTPUT_FOLDER: `E:\mjosa_new_oct_2025\use_gref4hsi\057_final\output\`

**Output:** `output/uhi_footprint_overlay.html`

---

### create_sim.py (3D UHI Simulation)
**Required:**
- ✅ NAV_CSV: `E:\mjosa_complete\data\processed\navigation\nav_data_merged.csv`
- ✅ MBES_GEOTIFF: Choose from options above
- ✅ H5_FOLDER: `E:\mjosa_new_oct_2025\use_gref4hsi\057_final\input\`
- ✅ CAMERA_CALIB_XML: `E:\mjosa_new_oct_2025\anxillary_data\HSI_2_body.xml`
- ✅ OUTPUT_FOLDER: `E:\mjosa_new_oct_2025\use_gref4hsi\057_final\output\`

**Output:** `output/uhi_sim_3d.html`

---

### create_stability_plot.py (6DOF Navigation Plots)
**Required:**
- ✅ NAV_CSV: `E:\mjosa_complete\data\processed\navigation\nav_data_merged.csv`
- ✅ H5_FOLDER: `E:\mjosa_new_oct_2025\use_gref4hsi\057_final\input\`
- ✅ OUTPUT_FOLDER: `E:\mjosa_new_oct_2025\use_gref4hsi\057_final\output\`

**Optional:**
- DB_PATH: `E:\mjosa\29\log_files\LOG_2024-10-29_10-13-32.db3` (for DVL analysis)

**Outputs:**
- `output/6dof_attitude_dvl_stacked.png`
- `output/6dof_transect_navigation.png`
- `output/6dof_combined_orientation.png`

---

## 🚀 Quick Start Checklist

Before running the visualization scripts, verify these paths exist:

- [ ] Navigation CSV exists: `E:\mjosa_complete\data\processed\navigation\nav_data_merged.csv`
- [ ] MBES GeoTIFF exists: Check which option you selected in config.py
- [ ] H5 folder exists: `E:\mjosa_new_oct_2025\use_gref4hsi\057_final\input\`
- [ ] Output folder exists: `E:\mjosa_new_oct_2025\use_gref4hsi\057_final\output\`
- [ ] Camera XML exists: `E:\mjosa_new_oct_2025\anxillary_data\HSI_2_body.xml`

---

## 🔍 Checking MBES GeoTIFF Coverage

To see which MBES file covers your area best:

1. **Check file sizes:**
   - `all.tif` - Largest file, full coverage
   - `geotiff_2.tif` - Medium file, partial coverage
   
2. **Open in QGIS or ArcGIS:**
   - Load the GeoTIFF
   - Load your navigation CSV as points
   - Check which MBES file covers your transect locations

3. **Quick Python check:**
   ```python
   import rasterio
   
   with rasterio.open(r"E:\mjosa_new_oct_2025\anxillary_data\EIVA\all.tif") as src:
       print(f"Bounds: {src.bounds}")
       print(f"CRS: {src.crs}")
       print(f"Shape: {src.shape}")
   ```

---

## 💡 Pro Tips

1. **Start with full coverage** (`all.tif`) - It's more complete
2. **Navigation CSV is now in mjosa_complete** - Already updated! ✅
3. **Scripts use mjosa_code config** - Once you update imports (coming next)
4. **Output folder location** - Plots will be saved to the 057_final/output folder
5. **Check file paths** - Use Windows Explorer to verify paths exist before running

---

## ⚠️ Common Issues

**"File not found" error:**
- Check that all paths in `config.py` are correct
- Make sure drive letter (E:) matches your system
- Verify files exist using Windows Explorer

**"CRS/EPSG mismatch" warning:**
- Check your MBES GeoTIFF metadata
- Update `EPSG_MBES` in config.py if needed
- Common for Mjøsa: EPSG:32632 (UTM Zone 32N)

**"Navigation CSV wrong format":**
- Make sure you're using `nav_data_merged.csv` (the corrected version)
- Not `nav_data_from_logfile.csv` (the raw GNSS)

---

## 📝 Summary

**Main navigation file location (UPDATED):** ✅
```
E:\mjosa_complete\data\processed\navigation\nav_data_merged.csv
```

**MBES GeoTIFF (YOU CHOOSE):**
- Currently set to: `all.tif` (full coverage)
- Alternative: `geotiff_2.tif` (partial)
- Edit `config.py` to switch

**Next step:** Update the scripts to import from mjosa_code config instead of gref_pipeline config!
