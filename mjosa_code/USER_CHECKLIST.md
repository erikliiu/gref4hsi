# 🎯 User Checklist - Before Running the Pipeline

**Date:** November 7, 2025  
**Package:** mjosa_code v1.0.0

---

## ⚙️ Configuration (Required)

Update these in `mjosa_code/utils/common/config.py`:

### 1. Data Paths
- [ ] **Update `RAW_DATA_DIR`** to your raw data location
  ```python
  RAW_DATA_DIR = Path(r"YOUR_PATH_HERE")  # e.g., E:\mjosa_complete\data\raw
  ```

- [ ] **Update `PROCESSED_DATA_DIR`** to your processed data location
  ```python
  PROCESSED_DATA_DIR = Path(r"YOUR_PATH_HERE")  # e.g., E:\mjosa_complete\data\processed
  ```

- [ ] **Verify `LOG_DB3_FILE`** points to your actual log file
  ```python
  LOG_DB3_FILE = RAW_LOG_FILES_DIR / "LOG_2024-10-29_10-13-32.db3"
  ```

### 2. Mission Parameters (Verify)
- [ ] **Check `MISSION_DATE`** matches your mission
  ```python
  MISSION_DATE = datetime.date(2024, 10, 29)
  ```

- [ ] **Check `MISSION_START_TIME` and `MISSION_END_TIME`**
  ```python
  MISSION_START_TIME = datetime.datetime(2024, 10, 29, 10, 10, 0, ...)
  MISSION_END_TIME = datetime.datetime(2024, 10, 29, 13, 15, 0, ...)
  ```

### 3. Transect Intervals (Verify)
- [ ] **Verify `TRANSECT_TIME_INTERVALS`** match your HSI data collection times
  ```python
  TRANSECT_TIME_INTERVALS = [
      ("10:50:51", "10:59:06"),  # Should match your actual transects
      # ... 8 more intervals
  ]
  ```

### 4. Dead Reckoning (Optional - Adjust if Needed)
- [ ] **Check `CONSTANT_VELOCITY_M_S`** (default 0.2 m/s is proven)
- [ ] **Check `ADJUST_SPEED_TO_HIT_END`** (default True)

---

## 🔍 Validation (Required Before Running)

Run these validation checks:

### 1. Config Validation
```python
from mjosa_code.utils.common import config
config.validate_config()  # Should print "✅ Configuration validation passed!"
```

### 2. File Existence
- [ ] **Verify .db3 file exists:**
  ```python
  from mjosa_code.utils.common import config
  print(config.LOG_DB3_FILE.exists())  # Should be True
  ```

### 3. Directory Structure
- [ ] **Verify output directories can be created:**
  ```python
  config.create_output_directories()  # Should create all needed dirs
  ```

### 4. Print Summary
```python
config.print_config_summary()  # Review all settings
```

---

## 🧪 Test Run (Recommended)

Before full pipeline, test with one transect:

### Option A: Test Functions
```python
from mjosa_code.utils.nav import nav_processing

# Test Step 1: Load .db3
db, analyzer = nav_processing.test_load_db3()

# Test Step 2: Extract transects
transect_csvs = nav_processing.test_extract_transects(analyzer)

# Test Step 3: Dead reckoning (first 2 transects only)
dr_csvs = nav_processing.test_constant_velocity_dr(transect_csvs, analyzer)
```

### Option B: Manual Single Transect
```python
# Run just for first transect
from mjosa_code.utils.common import config
transect_to_test = [config.TRANSECT_TIME_INTERVALS[0]]  # First one only

# Then run pipeline with this single transect
# (modify config temporarily or pass to functions)
```

---

## 🚀 Ready to Run Full Pipeline

Once all checks pass:

### Option 1: Complete Pipeline (Easiest)
```python
from mjosa_code.utils.nav import nav_processing

results = nav_processing.run_complete_pipeline(
    use_dvl=False,  # Use constant velocity (proven)
    verbose=True    # See progress
)
```

### Option 2: Use Notebook (Recommended for First Time)
1. Open: `mjosa_code/notebooks/nav/1_create_corrected_nav_csv.ipynb`
2. Run all cells
3. Review output at each step

---

## 📊 Expected Results

After successful run:

### Files Created:
- [ ] `processed/navigation/nav_data_from_logfile.csv` (main CSV from .db3)
- [ ] `processed/navigation/nav_data_merged.csv` (corrected with DR)
- [ ] `processed/navigation/merged_data_for_EIVA.txt` (EIVA format)
- [ ] 9 transect directories (e.g., `105051/`, `112350/`, ...)
- [ ] 9 original transect CSVs (`nav_data_HHMMSS.csv`)
- [ ] 9 DR transect CSVs (`nav_data_HHMMSS_dr.csv`)

### Console Output Should Show:
```
==================================================================
STEP 1: Loading .db3 file to CSV
==================================================================
Input:  E:\mjosa_complete\data\raw\log_files\LOG_2024-10-29_10-13-32.db3
Output: E:\mjosa_complete\data\processed\navigation\nav_data_from_logfile.csv
...
✅ CSV created successfully!
...
==================================================================
STEP 2: Extracting Transect Segments
==================================================================
...
✅ Extracted 9/9 transects
...
==================================================================
STEP 3A: Constant Velocity Dead Reckoning
==================================================================
...
✅ Completed: 9/9 files
...
==================================================================
STEP 4: Merging Corrected Transects
==================================================================
...
✅ Merged CSV created successfully!
...
==================================================================
STEP 5: Converting to EIVA Format
==================================================================
...
✅ EIVA file created successfully!
...
🎉 PIPELINE COMPLETED SUCCESSFULLY!
```

---

## 🐛 If Something Goes Wrong

### Error: "ModuleNotFoundError: No module named 'Eelume'"
- **Check:** Package structure intact
- **Check:** Running from correct location
- **Solution:** Path added automatically in nav_processing.py

### Error: "FileNotFoundError: .db3 file"
- **Check:** `config.LOG_DB3_FILE` path correct
- **Check:** File exists at that location
- **Solution:** Update path in config.py

### Error: "Config validation failed"
- **Check:** Run `config.validate_config()` to see specific issue
- **Common issues:** Time intervals wrong format, start > end time
- **Solution:** Fix reported issues in config.py

### Error: "KeyError: 'timestamp [unix epoch s]'"
- **Check:** CSV file has correct column names
- **Check:** Using correct CSV file (from analyzer.create_aligned_csv_file)
- **Solution:** Re-run Step 1 to create properly formatted CSV

### Pipeline Runs but Positions Look Wrong
- **Check:** `MJOSA_ORIGIN` coordinates correct (60.8011575, 10.7122345)
- **Check:** Transect time intervals match actual data collection
- **Check:** `CONSTANT_VELOCITY_M_S` reasonable for your vehicle (0.2 m/s default)
- **Solution:** Adjust config values, re-run pipeline

---

## 📋 Final Pre-Flight Checklist

Before running on production data:

- [ ] ✅ All config paths updated
- [ ] ✅ `.db3` file exists and accessible
- [ ] ✅ `config.validate_config()` passes
- [ ] ✅ Transect intervals verified against mission log
- [ ] ✅ Output directories are writable
- [ ] ✅ Test run completed successfully (at least one transect)
- [ ] ✅ Reviewed `config.print_config_summary()` output
- [ ] ✅ Understand which DR method to use (constant velocity default)
- [ ] ✅ Have backup of any existing processed files
- [ ] ✅ Ready to run full pipeline!

---

## 🎯 After Pipeline Completes

### Verify Results:
1. **Check file sizes:** All CSVs should have data (not empty)
2. **Check point counts:** Merged CSV should have ~same points as original
3. **Check coordinates:** Should be within Lake Mjøsa bounds
4. **Check EIVA format:** First line should be header, data rows below

### Quick Verification:
```python
import pandas as pd
from mjosa_code.utils.common import config

# Load merged CSV
df = pd.read_csv(config.MERGED_CSV_FILE)

print(f"Total points: {len(df)}")
print(f"Lat range: {df['latitude [deg]'].min():.6f} to {df['latitude [deg]'].max():.6f}")
print(f"Lon range: {df['longitude [deg]'].min():.6f} to {df['longitude [deg]'].max():.6f}")
print(f"Depth range: {df['depth [m]'].min():.2f} to {df['depth [m]'].max():.2f} m")

# Should look reasonable for Lake Mjøsa
```

### Next Steps:
- [ ] Use merged CSV for georeferencing hyperspectral data
- [ ] Import EIVA TXT into EIVA for visualization
- [ ] Compare with known reference points if available
- [ ] Document any issues or adjustments made

---

## 📞 Need Help?

Check these resources:
1. **README.md** - Complete package documentation
2. **SETUP_SUMMARY.md** - What was created and why
3. **config.py** - All settings with inline comments
4. **nav_processing.py** - Function docstrings

---

**Last Updated:** November 7, 2025  
**Package Version:** 1.0.0  
**Status:** Ready for production use ✅
