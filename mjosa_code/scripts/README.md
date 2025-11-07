# Mjøsa Scripts

Standalone scripts for visualization and analysis of Mjøsa UHI mission data.

## 📁 Scripts

### 1. `create_html.py`
**Interactive HTML Map Visualization**

Creates an interactive Folium map showing:
- MBES bathymetry (colorized GeoTIFF overlay)
- Navigation track from CSV
- HSI footprints from georeferenced H5 files
- HSI RGB overlay (thinned for performance)

**Usage:**
```bash
python create_html.py
```

**Output:** 
- HTML file with interactive map
- Can be opened in any web browser

---

### 2. `create_sim.py`
**3D UHI Simulation**

Interactive 3D visualization with:
- Multi-H5 segment visualization
- IMU and Camera frames
- Camera rays
- MBES surface with dual seabed layers
- Altitude plot with highlight intervals
- Side-view MBES with gray base and purple overlay

**Features:**
- Interactive slider for frame navigation
- Animation controls
- Customizable highlight intervals

**Usage:**
```bash
python create_sim.py
```

**Output:**
- Interactive 3D matplotlib window
- Animation controls for temporal navigation

---

### 3. `create_stability_plot.py`
**6DOF Navigation Stability Plot**

Plots navigation data for specific transect sections:
- **Position:** East/North scatter plot with alignment shift
- **Orientation:** Roll, Pitch, Yaw time series
- **Depth and Altitude:** Time series plots

**Uses config values:**
- `NAV_CSV`: Navigation data source
- `UHI_FILES`, `UHI_TRACK_RANGE`: Transect section definition
- `UHI_ALIGNMENT_DX`, `UHI_ALIGNMENT_DY`: Coordinate alignment
- `LAT0`, `LON0`: NED frame origin

**Usage:**
```bash
python create_stability_plot.py
```

**Output:**
- Multi-panel matplotlib figure showing 6DOF stability
- Useful for assessing navigation quality during transects

---

## 🔧 Dependencies

All scripts depend on:
- Configuration from `gref_pipeline.config` (from final_act/)
- Navigation utilities
- H5 file readers
- Common Python packages: numpy, pandas, matplotlib, folium

## 📝 Notes

- These scripts are copied from `gref4hsi/final_act/other/`
- They are standalone visualization tools
- Each script can be run independently
- Output files are typically saved to the working directory or configured output paths
