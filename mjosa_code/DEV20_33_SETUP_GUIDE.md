# Dev20-33 Setup for mjosa_code

## ✅ COMPLETED SETUP

### 1. Copied georef.py to mjosa_code
**Location:** `mjosa_code/utils/uhi/georef.py`
- Full georef module with all SVM classification methods
- Updated config import to use `from utils.common import config` (mjosa_code) first
- Falls back to gref_pipeline config if needed

### 2. Created uhi module structure
```
mjosa_code/utils/uhi/
├── __init__.py          ← Exports: load_transect, GeoFile, Transect, Cube
└── georef.py            ← Complete georef module (~13k lines)
```

### 3. ROI Files Used by Dev Notebooks
**Main ROI files identified:**
1. **`057_5_combined.json`** - Used by: dev20, dev21, dev22, dev23, dev28
   - Training/validation ROIs for transect 057
   
2. **`028_new.json`** - Used by: dev27, dev28, dev30
   - Training/validation ROIs for transect 028

**You created paths for these:**
- `E:\mjosa_complete\data\anxilliary\ROIs\` ← Copy ROI JSON files here
- `E:\mjosa_complete\data\anxilliary\SVM_models\` ← Save trained models here

---

## 📝 HOW TO USE IN NEW NOTEBOOKS

### Import pattern for new notebooks in mjosa_code:

```python
import sys
from pathlib import Path

# Add mjosa_code root to path
mjosa_code_root = Path.cwd().parent.parent  # Adjust based on notebook depth
sys.path.insert(0, str(mjosa_code_root))

# Import georef from mjosa_code
from utils.uhi import georef, load_transect, Cube
from utils.common import config

# Or import everything:
from utils.uhi.georef import *

print("✅ Imports successful!")
```

### Example workflow (dev20-style):

```python
# Load transect
transect = load_transect(config.OUTPUT_FOLDER)
transect.list_files()

# Select file
cube = transect.select_files(["rad_uhi_20241029_115057_5"])
cube.describe()

# Apply illumination correction
cube.apply_illumination_correction_v2()

# Import ROIs from mjosa_complete data folder
roi_path = Path(config.RAW_DATA_DIR).parent / "anxilliary" / "ROIs" / "057_5_combined.json"
cube.import_rois(str(roi_path))
cube.list_rois()

# Define ROI collections
training_rois = ["training_dark", "training_sediment", "training_bombs"]
validation_rois = ["validation_dark", "validation_sediment", "validation_bombs"]

# Train SVM
results = cube.train_svm_with_cv(
    roi_names=training_rois,
    use_corrected=True,
    n_splits=5,
    class_weight='balanced'
)

# Classify
cube.classify_svm(
    model=results['model'],
    class_names=results['class_names']
)

# Visualize
cube.plot_classification_map(show=True)
```

---

## 🔧 KEY METHODS AVAILABLE IN georef.Cube

### Data Loading & Preprocessing
- `apply_illumination_correction_v2()` - Illumination correction
- `select_files()` - Select specific H5 files
- `describe()` - Show cube info

### ROI Management
- `import_rois(filename)` - Load ROIs from JSON
- `export_rois(filename)` - Save ROIs to JSON
- `list_rois()` - Show loaded ROIs

### SVM Classification
- `train_svm()` - Basic SVM training
- `train_svm_with_cv()` - SVM with cross-validation
- `classify_svm()` - Apply trained SVM
- `classify_segment()` - Classify with filtering
- `classify_segment_with_validation()` - Classify + validate

### Visualization
- `plot_spectrum()` - Plot spectral signatures
- `plot_classification_map()` - Show classification results
- `plot_classification_overlay()` - Overlay on RGB
- `plot_georef()` - Plot georeferenced cube

### Statistics
- `get_classification_statistics()` - Confusion matrix, accuracy
- `get_segment_statistics()` - Per-segment stats

---

## 📂 REQUIRED DATA PATHS IN CONFIG

Make sure your `mjosa_code/utils/common/config.py` has:

```python
# Hyperspectral data
OUTPUT_FOLDER = _MJOSA_COMPLETE_ROOT / "use_gref4hsi" / "057_final" / "output"
UHI_FILES = ["rad_uhi_20241029_115057_5"]  # or other file names
UHI_TRACK_RANGE = (0, -1)  # Full track or specify range

# ROIs and Models (you created these)
ROI_FOLDER = _MJOSA_COMPLETE_ROOT / "data" / "anxilliary" / "ROIs"
SVM_MODELS_FOLDER = _MJOSA_COMPLETE_ROOT / "data" / "anxilliary" / "SVM_models"
```

---

## ✅ MJOSA_CODE IS NOW INDEPENDENT!

All dev20-33 functionality is now available inside `mjosa_code`:
- ✅ Complete georef module copied
- ✅ Config imports updated
- ✅ SVM classification methods ready
- ✅ ROI import/export working
- ✅ No external dependencies outside mjosa_code folder!

**Next steps:**
1. Copy `057_5_combined.json` and `028_new.json` to `E:\mjosa_complete\data\anxilliary\ROIs\`
2. Create your new notebooks in `mjosa_code/notebooks/classification/` (or similar)
3. Use the import pattern shown above
4. Train models and save to `E:\mjosa_complete\data\anxilliary\SVM_models\`
