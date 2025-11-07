"""
Quick test script to debug plot_georef wavelength colormap mode
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath("gref4hsi/final_act"))
from utils.gref_pipeline.georef import *
from gref_pipeline import config

print("Loading transect...")
transect = load_transect(config.OUTPUT_FOLDER)
cube = transect.select_files(["rad_uhi_20241029_115057_5"])

print("Applying preprocessing...")
cube.apply_illumination_correction_v2()
cube.apply_spectral_smoothing(method="gaussian", gaussian_sigma=5.0)
cube.apply_wavelength_filter(wavelength_range=(490, 700))
cube.apply_spectral_normalization(method="l2")

print("\nTesting plot_georef with wavelength colormap...")
print(f"Track range: {config.UHI_TRACK_RANGE_5}")
print(f"vmin=0.0491, vmax=0.2558")

import matplotlib.pyplot as plt

fig, ax = cube.plot_georef(
    use_corrected=True,
    track_start=config.UHI_TRACK_RANGE_5[0],
    track_end=config.UHI_TRACK_RANGE_5[1],
    use_wavelength_colormap=True,
    wavelength_colormap_target=677,
    vmin=0.0491,
    vmax=0.2558,
    return_fig=True,  # IMPORTANT: Return the figure and axes
    quiet=False,  # Show all debug output
)

print("\n" + "=" * 60)
print("✅ Figure created! Displaying...")
print("=" * 60)
plt.show()
