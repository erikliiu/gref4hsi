# final_act - Reorganized Georeferencing Pipeline

## Folder Structure

```
final_act/
├── gref_pipeline/          # Main georeferencing pipeline
│   ├── config.py          # Configuration (paths, sensor params, origins)
│   ├── main.py            # Main processing script
│   └── __init__.py
├── utils/                  # Utility modules
│   ├── gref_pipeline/     # Georeferencing utilities
│   │   ├── georef.py      # Core georeferencing classes (GeoFile, TransectDataSet, CombinedTransectCube)
│   │   ├── georeference_mbes.py  # MBES loading and ray tracing
│   │   ├── utils.py       # Helper functions (navigation, transforms, calibration)
│   │   └── __init__.py
│   ├── other/             # Other utilities
│   │   ├── detrend_mbes.py  # MBES detrending (MBESDetrender, AutoTuneMBES)
│   │   └── __init__.py
│   └── __init__.py
├── pose_and_html/         # Visualization tools
│   ├── create_sim.py      # 3D simulation
│   ├── create_html.py     # Interactive HTML map
│   └── __init__.py
├── notebooks/             # Jupyter notebooks
│   ├── detrend_mbes.ipynb # MBES detrending examples
│   └── plot_gref_uhi.ipynb # UHI plotting examples
├── test/                  # Test scripts
│   └── test.ipynb
└── readme.md
```

## Import Patterns

### From notebooks/ or test/ directories:
```python
import sys, os
sys.path.append(os.path.abspath("../"))  # Go up to final_act/

# Import config
from gref_pipeline import config

# Import georeferencing utils
from utils.gref_pipeline import georef
from utils.gref_pipeline import utils
from utils.gref_pipeline import georeference_mbes

# Import MBES detrending
from utils.other.detrend_mbes import MBESDetrender, AutoTuneMBES, ScoreWeights
```

### From gref_pipeline/main.py:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Go up to final_act/

from gref_pipeline import config
from utils.gref_pipeline import utils
from utils.gref_pipeline import georeference_mbes
```

### From pose_and_html/ scripts:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Go up to final_act/

from gref_pipeline import config
from utils.gref_pipeline import utils
```

### From utils/gref_pipeline/georef.py:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Go up to final_act/

from gref_pipeline import config  # optional
from utils.gref_pipeline import utils  # when needed in plot_georef_with_trajectory
```

## Key Files

### Configuration
- **gref_pipeline/config.py**: All paths and parameters
  - NAV_CSV: Navigation data
  - MBES_GEOTIFF: Bathymetry
  - H5_FOLDER: Input HSI files
  - OUTPUT_FOLDER: Georeferenced output
  - Sensor calibration, origin (LAT0, LON0, H0)

### Main Processing
- **gref_pipeline/main.py**: Run full georeferencing pipeline
  - Processes H5 files in H5_FOLDER
  - Ray traces to MBES mesh
  - Saves georeferenced data to OUTPUT_FOLDER

### Utilities
- **utils/gref_pipeline/georef.py**: Core classes for loading and plotting georeferenced data
- **utils/gref_pipeline/utils.py**: Navigation interpolation, coordinate transforms, calibration loading
- **utils/gref_pipeline/georeference_mbes.py**: MBES loading and ray tracing with PyVista/Trimesh
- **utils/other/detrend_mbes.py**: MBES detrending to remove large-scale trends

### Visualization
- **pose_and_html/create_sim.py**: 3D matplotlib simulation with IMU/camera frames and rays
- **pose_and_html/create_html.py**: Interactive Folium map with MBES and HSI footprints

## Dependencies

All import paths have been fixed to work from the final_act/ root directory with proper sys.path.append() statements.
