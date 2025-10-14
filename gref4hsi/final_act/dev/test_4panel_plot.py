"""
Test script for 4-panel UHI-MBES comparison plot.
This script tests that the plot_uhi_mbes_comparison function correctly generates
a 4-panel figure with: Raw RGB | Corrected RGB | MBES Residuals | Delta z
"""

import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))

from utils.other.detrend_mbes import MBESDetrender
from gref_pipeline import config
import matplotlib.pyplot as plt

print("=" * 80)
print("Testing 4-panel UHI-MBES Comparison Plot")
print("=" * 80)

# Create MBESDetrender object
print("\n1. Creating MBESDetrender object...")
tif_path = config.MBES_GEOTIFF
mb_ned = MBESDetrender(
    tif_path,
    coord_system="ned",
    ned_origin=(config.LON0, config.LAT0, config.H0),
    epsg_utm=config.EPSG_UTM,
    epsg_geo=config.EPSG_GEOGRAPHIC,
)
print("   ✅ MBESDetrender created")

# Load data
print("\n2. Loading data...")
mb_ned.load()
print("   ✅ Data loaded")

# Detrend
print("\n3. Detrending MBES data...")
mb_ned.detrend(
    order_x=3,
    smooth_baseline_m=0.80,
    smooth_tilt_m=0.70,
    smooth_center_m=1.0,
    robust=True,
    central_frac=0.8,
)
print("   ✅ Detrending complete")

# Adjust UHI alignment
print("\n4. Adjusting UHI alignment...")
mb_ned.adjust_uhi_alignment(dx=-0.05, dy=-3)
print("   ✅ Alignment adjusted")

# Create the 4-panel plot
print("\n5. Creating 4-panel comparison plot...")
print("   Expected panels:")
print("      Panel 1: Raw UHI RGB (no illumination correction)")
print("      Panel 2: Corrected UHI RGB (illumination corrected)")
print("      Panel 3: MBES Residuals")
print("      Panel 4: Δz (normalized difference)")

fig1 = mb_ned.plot_uhi_mbes_comparison(
    use_adjusted=True, normalization_method="zscore", show=True
)

# Verify the figure has 4 axes (main plot axes, not colorbar axes)
axes = fig1.get_axes()
# Filter to only main axes (those with titles)
main_axes = [ax for ax in axes if ax.get_title()]
num_panels = len(main_axes)
print(f"\n   Figure created with {num_panels} panels")

if num_panels == 4:
    print("   ✅ SUCCESS: 4 panels created!")
    print("\n   Panel titles:")
    for i, ax in enumerate(main_axes, 1):
        title = ax.get_title()
        print(f"      Panel {i}: {title}")

    # Verify the titles are correct
    expected_keywords = ["raw", "illum-corrected", "Residuals", "Δ"]
    for i, (ax, keyword) in enumerate(zip(main_axes, expected_keywords), 1):
        title = ax.get_title()
        if keyword.lower() in title.lower() or keyword in title:
            print(f"   ✅ Panel {i} title correct (contains '{keyword}')")
        else:
            print(
                f"   ⚠️  Panel {i} title might be wrong: '{title}' (expected '{keyword}')"
            )
else:
    print(f"   ❌ FAILED: Expected 4 panels, got {num_panels}")
    print("   Panel titles:")
    for i, ax in enumerate(main_axes, 1):
        print(f"      Panel {i}: {ax.get_title()}")

print("\n" + "=" * 80)
print("Test complete! The plot should be displayed.")
print("=" * 80)
