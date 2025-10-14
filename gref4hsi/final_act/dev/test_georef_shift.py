"""
Test script to verify plot_georef with alignment shift.
Tests that apply_alignment_shift=True correctly shifts NED coordinates.
"""

import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))

from gref_pipeline import config
from utils.gref_pipeline import georef
import matplotlib.pyplot as plt

print("=" * 80)
print("Testing plot_georef with alignment shift")
print("=" * 80)

print(f"\nAlignment shift from config:")
print(f"  UHI_ALIGNMENT_DX = {config.UHI_ALIGNMENT_DX} m")
print(f"  UHI_ALIGNMENT_DY = {config.UHI_ALIGNMENT_DY} m")

# Load transect
print("\n1. Loading transect data...")
transect = georef.load_transect(config.OUTPUT_FOLDER)
print("   ✅ Transect loaded")

# Select files
print("\n2. Selecting UHI files...")
cube = transect.select_files(config.UHI_FILES)
print("   ✅ Files selected")

# Apply illumination correction
print("\n3. Applying illumination correction...")
cube.apply_illumination_correction(window_size=1000, strength=1.0)
print("   ✅ Correction applied")

# Get track range
track_start, track_end = config.UHI_TRACK_RANGE
print(f"\n4. Using track range: {track_start} to {track_end}")

# Create two plots: without and with alignment shift
print("\n5. Creating plots...")

# Plot 1: WITHOUT alignment shift (original coordinates)
print("\n   a) Plotting WITHOUT alignment shift...")
fig1, ax1 = plt.subplots(figsize=(6, 8))
plt.close(fig1)  # Close to recreate properly
cube.plot_georef(
    coordinate_system="NED",
    track_start=track_start,
    track_end=track_end,
    use_corrected=True,
    apply_alignment_shift=False,  # Original coordinates
    show_file_boundaries=True,
    interactive=False,
    quiet=False,
)
print("      ✅ Plot 1 created (original coordinates)")

# Plot 2: WITH alignment shift (shifted coordinates)
print("\n   b) Plotting WITH alignment shift...")
cube.plot_georef(
    coordinate_system="NED",
    track_start=track_start,
    track_end=track_end,
    use_corrected=True,
    apply_alignment_shift=True,  # Shifted coordinates (matches MBES)
    show_file_boundaries=True,
    interactive=False,
    quiet=False,
)
print("      ✅ Plot 2 created (shifted coordinates)")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
print("\nYou should see TWO plots:")
print("  1. Original UHI coordinates (no shift)")
print("  2. Shifted UHI coordinates (dx=-0.05m E, dy=-3m N)")
print("\nThe second plot should match the coordinates in detrend_mbes plots.")
print("=" * 80)

plt.show()
