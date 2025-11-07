# Navigation Output Files

## 🎯 MAIN FILE FOR GEOREFERENCING

### **`nav_data_merged.csv`**

**This is THE file you need for georeferencing hyperspectral data!**

**Method:** Constant Velocity Dead Reckoning (0.2 m/s)

**Contents:**
- Corrected navigation with dead-reckoned positions during transect segments
- Original GNSS positions outside of transects
- Full 6DOF data: latitude, longitude, depth, roll, pitch, yaw, altitude

---

## All Files Explained

### Main Output Files

| File | Description | Use For |
|------|-------------|---------|
| **`nav_data_merged.csv`** | ⭐ Final corrected navigation (constant velocity DR) | **Georeferencing HSI data** |
| `nav_data_merged_dvl.csv` | Alternative corrected navigation (DVL velocity DR) | Comparison/validation |
| `merged_data_for_EIVA.txt` | Constant velocity merged in EIVA format | EIVA software import |
| `merged_data_for_EIVA_dvl.txt` | DVL velocity merged in EIVA format | EIVA software (alternative) |
| `nav_data_from_logfile.csv` | Original raw GNSS data from .db3 file | Reference/comparison |

### Transect Folders

Each folder (e.g., `105051/`, `112350/`, etc.) contains individual transect segments:

- **`nav_data_HHMMSS.csv`** - Original GNSS data for that transect
- **`nav_data_HHMMSS_dr.csv`** - Dead-reckoned positions (constant velocity) ⭐ used in main merged file
- **`nav_data_HHMMSS_dr_dvl_v2.csv`** - Dead-reckoned positions (DVL-based) used in DVL merged file

Folder names are transect start times with colons removed:
- `105051` = Transect starting at 10:50:51
- `112350` = Transect starting at 11:23:50
- etc.

---

## Which Method Was Used?

### Constant Velocity (Main Method) ⭐
**Output:** `nav_data_merged.csv`
- Simple and proven reliable for this mission
- Uses constant 0.2 m/s velocity
- Consistent results across all transects
- **This is what you should use for georeferencing**

### DVL Velocity (Alternative Method) 🔬
**Output:** `nav_data_merged_dvl.csv`
- Uses actual DVL sensor velocity measurements
- More complex, experimental
- Good for comparison and validation
- Results may vary depending on DVL data quality

Both methods are available so you can compare results!

---

## Processing Details

- **Mission Date:** 2024-10-29
- **Mission Location:** Lake Mjøsa, Norway
- **Platform:** Eelume underwater robot
- **Total Transects:** 9 segments
- **Coordinate Reference:** Lat 60.8011575°N, Lon 10.7122345°E

---

## Need Help?

See the notebook: `mjosa_code/notebooks/nav/1_create_corrected_nav_csv.ipynb`

Or check the documentation: `mjosa_code/README.md`
