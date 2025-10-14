"""
Example: How to use the multiple TIF overlay feature in your own scripts

This shows how to integrate the new feature into your workflow.
"""

import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gref_pipeline import config
from create_html import build_map_with_mbes

# Standard inputs from config
NAV_CSV = config.NAV_CSV
GEOTIFF = config.MBES_GEOTIFF  # Main MBES overlay
HSI_DIR = config.OUTPUT_FOLDER

# OPTIONAL: Add multiple TIF overlays
USE_MULTIPLE_TIFS = True  # Set to False to disable
TIF_FOLDER = r"E:\mjosa_new_oct_2025\all_tifs_from_eiva\relevant_tifs_only"

# Output
OUTPUT_HTML = "my_custom_map.html"

print("=" * 70)
print("Creating custom map with optional multiple TIF overlays")
print("=" * 70)

# Build the map
build_map_with_mbes(
    NAV_CSV,
    GEOTIFF,
    output_html=OUTPUT_HTML,
    cmap_name="viridis",
    opacity=0.70,
    pct_clip=(2, 98),
    legend_caption="MBES (depth, m)",
    # === NEW FEATURE: Multiple TIF overlays (OPTIONAL) ===
    tif_folder=TIF_FOLDER if USE_MULTIPLE_TIFS else None,
    tif_opacity=0.70,
    # ====================================================
    # HSI settings
    hsi_h5_dir=HSI_DIR,
    hsi_recursive=False,
    hsi_layer_name="HSI footprint (outline)",
    hsi_stride_tracks=10,
    hsi_stride_slits=10,
    hsi_add_lines=True,
    hsi_add_points=False,
    hsi_color="#ffff00",
    # HSI RGB overlay
    hsi_add_rgb=True,
    hsi_rgb_layer_name="HSI RGB (spectral, thinned)",
    hsi_rgb_stride_tracks=10,
    hsi_rgb_stride_slits=10,
    hsi_rgb_keep_fraction=0.05,
    hsi_rgb_add_tooltip=False,
    hsi_rgb_red_wl=654.2,
    hsi_rgb_green_wl=560,
    hsi_rgb_blue_wl=440.3,
    hsi_rgb_normalize=True,
    hsi_rgb_use_corrected=False,
    hsi_rgb_point_radius=0.3,
    hsi_rgb_point_opacity=0.8,
    # CRS settings
    hsi_ecef_epsg=config.EPSG_ECEF,
    hsi_geodetic_epsg=4979,
)

print("\n" + "=" * 70)
print("✅ COMPLETE!")
print("=" * 70)
print(f"📂 Open: {OUTPUT_HTML}")
print("=" * 70)
