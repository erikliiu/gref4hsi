# Config Architecture Refactoring

## What Changed

### Before (❌ BAD)
The `config.py` file had **both data AND functions** mixed together:
- Configuration constants (good)
- Helper functions like `get_transect_output_dir()`, `validate_config()`, etc. (bad - doesn't belong in config)

### After (✅ GOOD)
**Clean separation of concerns:**

1. **`config.py`** - Pure data only, user-adjustable settings:
   - File paths
   - Mission dates/times
   - Transect intervals
   - Processing parameters
   - **NO FUNCTIONS AT ALL**

2. **`config_utils.py`** - All helper functions:
   - `parse_transect_intervals()` - Convert time strings to datetime objects
   - `get_transect_output_dir()` - Generate output directory paths
   - `get_transect_csv_path()` - Generate CSV file paths
   - `create_output_directories()` - Create directory structure
   - `validate_config()` - Validate configuration settings
   - `print_config_summary()` - Display config summary

## Why This Matters

### Principle: Configuration = Data Only
A config file should contain **ONLY user-adjustable constants**. No logic, no functions.

**Benefits:**
- ✅ Users can easily see what they can change
- ✅ Clear separation: config = data, utils = logic
- ✅ No confusion about what's a setting vs. a helper
- ✅ Easier to maintain and understand
- ✅ Standard Python best practice

## Files Modified

### Created
- `mjosa_code/utils/common/config_utils.py` - New utility module for config helpers

### Updated
- `mjosa_code/utils/common/config.py` - Removed all 6 functions, pure data only
- `mjosa_code/notebooks/nav/1_create_corrected_nav_csv.ipynb` - Updated cell 5 to import from `config_utils`
- `mjosa_code/utils/nav/nav_processing.py` - Updated imports and function calls

## How to Use

### In Your Code

**Old way (don't do this):**
```python
from mjosa_code.utils.common import config

config.validate_config()  # ❌ Function was in config
config.print_config_summary()  # ❌ Function was in config
```

**New way (correct):**
```python
from mjosa_code.utils.common import config, config_utils

config_utils.validate_config()  # ✅ Function in utils
config_utils.print_config_summary()  # ✅ Function in utils

# Config has ONLY data
path = config.MAIN_CSV_FILE  # ✅ Data in config
intervals = config.TRANSECT_TIME_INTERVALS  # ✅ Data in config
```

### In Notebooks

**Cell 5 now imports both:**
```python
from mjosa_code.utils.common import config, config_utils

config_utils.print_config_summary()
config_utils.validate_config()
```

## What's in config.py Now

**Only user-adjustable constants:**
- `RAW_DATA_DIR`, `PROCESSED_DATA_DIR` - File paths
- `MISSION_DATE`, `MISSION_START_TIME`, `MISSION_END_TIME` - Time ranges
- `TRANSECT_TIME_INTERVALS` - List of transect time strings
- `MJOSA_ORIGIN_LAT`, `MJOSA_ORIGIN_LON` - Coordinate reference
- `CONSTANT_VELOCITY_M_S` - Dead reckoning speed
- `USE_DVL_VELOCITY` - Processing options
- `VERBOSE_LOGGING`, `DEFAULT_DR_METHOD` - Behavior flags

**No functions, no logic, just pure data.**

## Migration Checklist

If you have other code using the old config functions:

- [ ] Import `config_utils` module
- [ ] Replace `config.validate_config()` → `config_utils.validate_config()`
- [ ] Replace `config.print_config_summary()` → `config_utils.print_config_summary()`
- [ ] Replace `config.create_output_directories()` → `config_utils.create_output_directories()`
- [ ] Replace `config.get_transect_output_dir()` → `config_utils.get_transect_output_dir()`
- [ ] Replace `config.get_transect_csv_path()` → `config_utils.get_transect_csv_path()`
- [ ] Replace `config.TRANSECT_DATETIME_INTERVALS` → `config_utils.parse_transect_intervals()`

## Architecture Principles

### Good Config Design
```python
# config.py
DATA_DIR = Path("data/raw")
VELOCITY = 0.2
USE_DVL = True
```

### Bad Config Design
```python
# config.py - DON'T DO THIS
DATA_DIR = Path("data/raw")

def get_data_path():  # ❌ Function in config!
    return DATA_DIR / "file.csv"
```

### Correct Separation
```python
# config.py
DATA_DIR = Path("data/raw")

# config_utils.py
def get_data_path():  # ✅ Function in utils!
    return config.DATA_DIR / "file.csv"
```

---

**Bottom line:** Config files are for configuration data. Helper functions belong in utility modules. This refactoring enforces that clean architecture.
