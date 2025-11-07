# Mjøsa Code - Self-Contained Navigation Processing Package

**Purpose:** Process Eelume robot navigation data for Lake Mjøsa underwater hyperspectral imaging mission

**Author:** Erik Liu (NTNU)  
**Date:** November 2025  
**Mission Date:** October 29, 2024

---

## 📦 Package Structure

```
mjosa_code/
├── __init__.py                              # Package initialization
├── README.md                                # This file
├── external_libs/                           # External dependencies
│   └── eelume_pypost/                      # Eelume PyPost library (ROS2 .db3 handler)
│       └── Eelume/
│           └── PyPost/                     # DatabaseHandler, MotionAnalyzer, etc.
├── utils/                                   # Utility modules
│   ├── common/
│   │   ├── __init__.py
│   │   └── config.py                       # Central configuration file
│   └── nav/
│       ├── __init__.py
│       ├── analyze_log_file.py             # LogData class (existing)
│       └── nav_processing.py               # Navigation processing pipeline (NEW)
└── notebooks/                               # Jupyter notebooks
    └── nav/
        └── 1_create_corrected_nav_csv.ipynb  # Clean navigation processing notebook
```

---

## 🚀 Quick Start

### 1. Configure Your Paths

Edit `utils/common/config.py` to match your data locations:

```python
# Set your data directory
RAW_DATA_DIR = Path(r"E:\mjosa_complete\data\raw")
PROCESSED_DATA_DIR = Path(r"E:\mjosa_complete\data\processed")

# Input file
LOG_DB3_FILE = RAW_LOG_FILES_DIR / "LOG_2024-10-29_10-13-32.db3"
```

### 2. Run the Pipeline

**Option A: Complete Pipeline (Recommended)**

```python
from mjosa_code.utils.nav import nav_processing

# Run everything in one call
results = nav_processing.run_complete_pipeline(
    use_dvl=False,  # Use constant velocity (proven reliable)
    verbose=True    # Show progress
)
```

**Option B: Use the Notebook**

Open `notebooks/nav/1_create_corrected_nav_csv.ipynb` and run all cells.

---

## 📊 Processing Pipeline Overview

### Input
- ROS2 .db3 log file from Eelume robot
- Contains 6DOF pose data (lat, lon, depth, roll, pitch, yaw, altitude)

### Steps
1. **Load .db3 → CSV**: Extract aligned 6DOF navigation data
2. **Extract Transects**: Identify 9 time segments for correction
3. **Dead Reckoning**: Apply constant velocity or DVL-based corrections
4. **Merge**: Replace transect segments in main CSV
5. **EIVA Export**: Convert to EIVA-compatible text format

### Output
- `nav_data_from_logfile.csv` - Raw 6DOF data
- `nav_data_merged.csv` - Corrected navigation with DR transects
- `merged_data_for_EIVA.txt` - EIVA format for visualization

---

## 🔧 Configuration

All settings are in `utils/common/config.py`:

- Mission parameters (date, time range)
- Lake Mjøsa origin coordinates
- 9 transect time intervals
- Dead reckoning parameters (constant velocity 0.2 m/s)
- Data paths (input/output directories)

Run validation:
```python
from mjosa_code.utils.common import config
config.validate_config()
config.print_config_summary()
```

---

## 🎯 Dead Reckoning Methods

### Method 1: Constant Velocity (Default)
- **Speed:** Fixed 0.2 m/s (proven reliable)
- **Bearing:** Constant from start → end position
- **Use Case:** Default, proven method for this mission

### Method 2: DVL Velocity (Alternative)
- **Speed:** Actual DVL measurements
- **Bearing:** Constant from start → end position
- **Use Case:** When vehicle speed varies significantly

**Recommendation:** Use constant velocity (Method 1) - it's proven reliable for Eelume at 0.2 m/s.

---

## 📚 Dependencies

### External Libraries (Included)
- **Eelume PyPost** - ROS2 .db3 file handler (already included in `external_libs/`)

### Python Packages (Required)
```bash
pip install pandas numpy geopy
```

---

## 🐛 Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'Eelume'"**  
Solution: The path is added automatically. Ensure package structure is intact.

**Issue: ".db3 file not found"**  
Solution: Update `config.LOG_DB3_FILE` to your actual log file location.

**Issue: "Config validation failed"**  
Solution: Run `config.validate_config()` to see specific issues.

---

## ✅ Validation Checklist

Before using:

- [ ] Update `config.py` with your data paths
- [ ] Verify `.db3` file exists
- [ ] Run `config.validate_config()` 
- [ ] Check transect time intervals match your mission
- [ ] Test with one transect before running full pipeline

---

**Last Updated:** November 7, 2025  
**Package Version:** 1.0.0
