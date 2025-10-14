"""
Test script for plot_georef_interactive_lines function
"""

import sys
import os

# Add paths
sys.path.append(os.path.abspath("gref4hsi/final_act"))
from utils.gref_pipeline.georef import *

sys.path.append(os.path.abspath("gref4hsi/final_act"))
from gref_pipeline import config

print("✓ Imports successful")

# Load data
transect = load_transect(config.OUTPUT_FOLDER)
cube = transect.select_files(["rad_uhi_20241029_115057_5"])

print(
    f"✓ Loaded cube: {cube.data.shape if hasattr(cube, 'data') and cube.data is not None else 'data not loaded'}"
)
print(f"✓ X_ecef shape: {cube.X_ecef.shape}")
print(f"✓ track_range: {config.UHI_TRACK_RANGE_5}")

# Check if function exists
if hasattr(cube, "plot_georef_interactive_lines"):
    print("✓ Function plot_georef_interactive_lines exists")
else:
    print("✗ Function plot_georef_interactive_lines NOT FOUND")
    sys.exit(1)

# Try to call it (will open a window)
print("\n" + "=" * 60)
print("Testing plot_georef_interactive_lines...")
print("=" * 60)

try:
    lines = cube.plot_georef_interactive_lines(
        use_corrected=False,
        coordinate_system="NED",
        track_start=config.UHI_TRACK_RANGE_5[0],
        track_end=config.UHI_TRACK_RANGE_5[1],
        figsize=(15, 10),
    )
    print(f"\n✓ Function executed successfully!")
    print(f"✓ Returned {len(lines)} lines: {lines}")
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST PASSED!")
print("=" * 60)
