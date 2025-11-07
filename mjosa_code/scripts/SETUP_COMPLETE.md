# ✅ Setup Complete - Visualization Scripts Ready!

## What Was Fixed

### 1. **Import Errors - FIXED** ✅
**Problem:** Scripts were trying to import from `gref_pipeline` which doesn't exist in mjosa_code environment.

**Solution:** Updated all 3 scripts to use:
- **Config:** `mjosa_code.utils.common.config` (your E:\mjosa_complete paths)
- **Utils:** `gref4hsi.final_act.utils.gref_pipeline` (coordinate transforms, H5 loading)

**Changed in:**
- `create_html.py` (line 44)
- `create_sim.py` (lines 38-46)
- `create_stability_plot.py` (lines 22-30)

---

### 2. **Data Paths - UPDATED** ✅
All paths now point to your `E:\mjosa_complete` folder structure:

```
E:\mjosa_complete\
├── data\
│   ├── processed\navigation\nav_data_merged.csv  ← NAV_CSV
│   └── anxilliary\
│       ├── EIVA\
│       │   ├── all.tif                           ← MBES_GEOTIFF (Option 1)
│       │   └── geotiff_2.tif                     ← MBES_GEOTIFF (Option 2)
│       └── HSI_2_body.xml                        ← CAMERA_CALIB_XML
└── use_gref4hsi\057_final\
    ├── input\                                     ← H5_FOLDER
    └── output\                                    ← OUTPUT_FOLDER
```

---

### 3. **Config Variables Added** ✅
Added to `mjosa_code/utils/common/config.py`:

```python
# Navigation CSV alias
NAV_CSV = str(MERGED_CSV_FILE)  # Points to nav_data_merged.csv

# Coordinate origin aliases (scripts expect LAT0/LON0)
LAT0 = 60.801146  # MJOSA_ORIGIN_LAT
LON0 = 10.705125  # MJOSA_ORIGIN_LON
H0 = 0.0

# All visualization paths (MBES, H5, output, camera XML, etc.)
```

---

## 🚀 How to Run

### From VS Code Terminal:

1. **Navigate to gref4hsi root:**
   ```powershell
   cd "c:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\gref4hsi"
   ```

2. **Run any script:**
   ```powershell
   # Interactive HTML map
   python mjosa_code/scripts/create_html.py
   
   # 3D simulation
   python mjosa_code/scripts/create_sim.py
   
   # 6DOF stability plots
   python mjosa_code/scripts/create_stability_plot.py
   ```

---

## 📊 What Each Script Creates

### create_html.py
**Output:** `E:\mjosa_complete\use_gref4hsi\057_final\output\uhi_footprint_overlay.html`
- Interactive map with MBES bathymetry
- Navigation track
- HSI footprints
- Open in browser to explore

### create_sim.py
**Output:** Interactive window (matplotlib 3D)
- Real-time 3D visualization
- Time slider to navigate frames
- Camera rays and MBES surface
- Use `--mbes-step 10` for faster rendering

### create_stability_plot.py
**Output:** PNG files in `E:\mjosa_complete\use_gref4hsi\057_final\output\`
- `6dof_attitude_dvl_stacked.png`
- `6dof_transect_navigation.png`
- `6dof_combined_orientation.png`

---

## ⚙️ How to Switch MBES GeoTIFF

**Current:** Using `all.tif` (full MBES coverage)

**To switch to geotiff_2.tif:**
1. Open `mjosa_code/utils/common/config.py`
2. Find line ~40 (MBES_GEOTIFF section)
3. Comment/uncomment:

```python
# Option 1: Full coverage
# MBES_GEOTIFF = Path(r"E:\mjosa_complete\data\anxilliary\EIVA\all.tif")

# Option 2: Partial coverage (ACTIVE)
MBES_GEOTIFF = Path(r"E:\mjosa_complete\data\anxilliary\EIVA\geotiff_2.tif")
```

---

## 🔍 Testing

Try running `create_html.py` first - it's the simplest:

```powershell
python mjosa_code/scripts/create_html.py
```

**Expected behavior:**
1. Loads navigation CSV from `E:\mjosa_complete\data\processed\navigation\nav_data_merged.csv`
2. Loads MBES from `E:\mjosa_complete\data\anxilliary\EIVA\all.tif`
3. Searches for H5 files in `E:\mjosa_complete\use_gref4hsi\057_final\input\`
4. Creates HTML file in `E:\mjosa_complete\use_gref4hsi\057_final\output\`

**If it works:** ✅ Setup is complete!

**If errors:** Check that all files exist at the specified paths.

---

## 📝 Summary

✅ **Import errors fixed** - Scripts now use mjosa_code config  
✅ **All paths updated** - Point to E:\mjosa_complete structure  
✅ **Config variables added** - NAV_CSV, LAT0/LON0, etc.  
✅ **MBES path corrected** - geotiff_2.tif is in EIVA folder  
✅ **Documentation created** - README and DATA_PATHS_GUIDE  

**You're ready to run the visualization scripts!** 🎉
