"""
Enhanced Spectral Analysis for UHI Data
========================================

This script demonstrates the full functionality of the Combined TransectCube class
with enhanced plotting and ROI management capabilities.

Author: Erik Liu
Date: 2025-01-14
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to access utils
sys.path.append(os.path.abspath("../"))

# Import core transect handling classes
from utils.gref_pipeline.transect import (
    load_transect,
    TransectDataSet,
    CombinedTransectCube,
)

# Import configuration
sys.path.append(os.path.abspath("../../"))
from gref_pipeline import config


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("ENHANCED SPECTRAL ANALYSIS FOR UHI DATA")
    print("=" * 80)

    # ========================================================================
    # STEP 1: LOAD TRANSECT DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING TRANSECT DATA")
    print("=" * 80)

    transect = load_transect(config.OUTPUT_FOLDER)
    transect.list_files()

    # ========================================================================
    # STEP 2: SELECT AND COMBINE FILES
    # ========================================================================
    print("\n" + "=" * 80)
    print("SELECTING AND COMBINING FILES")
    print("=" * 80)

    cube = transect.select_files(
        [
            "rad_uhi_20241029_115057_4",
            "rad_uhi_20241029_115057_5",
        ]
    )
    cube.describe()

    # Apply illumination correction
    print("\nApplying illumination correction...")
    cube.apply_illumination_correction()

    # ========================================================================
    # STEP 3: RGB VISUALIZATION WITH PERIMETER LINES
    # ========================================================================
    print("\n" + "=" * 80)
    print("RGB VISUALIZATION WITH PERIMETER LINES")
    print("=" * 80)

    # Define analysis lines
    line1 = [(510, 1648), (510, 1848)]
    line2 = [(390, 1820), (390, 2020)]
    line3 = [(640, 1450), (640, 1650)]
    line4 = [(500, 2040), (500, 2240)]

    # Color palette for maximum distinction
    optimal_colors = ["red", "blue", "orange", "magenta"]
    line_names = ["Line 1", "Line 2", "Line 3", "Line 4"]

    # Plot RGB with lines
    cube.plot_rgb(
        perimeter_line=[line1, line2, line3, line4],
        line_colors=optimal_colors,
        line_width=3,
        use_corrected=True,
        figsize=(30, 10),
        line_labels=line_names,
    )

    # ========================================================================
    # STEP 4: INTENSITY PROFILE ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("INTENSITY PROFILE ANALYSIS")
    print("=" * 80)

    lines = [line1, line2, line3, line4]

    # Raw intensity profiles
    print("\n1. Raw intensity profiles (average wavelength):")
    cube.plot_line_intensity_profile(
        perimeter_line=lines,
        use_average=True,
        moving_average_window=10,
        show_markers=False,
        line_labels=line_names,
        line_colors=optimal_colors,
        use_corrected=True,
    )

    # Normalized intensity profiles (mean normalization)
    print("\n2. Normalized intensity profiles (mean normalization):")
    cube.plot_line_intensity_profile(
        perimeter_line=lines,
        use_average=True,
        moving_average_window=10,
        show_markers=False,
        line_labels=line_names,
        line_colors=optimal_colors,
        normalize_intensities=True,
        normalization_method="mean",
        use_corrected=True,
    )

    # ========================================================================
    # STEP 5: INTERACTIVE ROI SELECTION (OPTIONAL)
    # ========================================================================
    print("\n" + "=" * 80)
    print("INTERACTIVE ROI SELECTION")
    print("=" * 80)
    print("\nInstructions:")
    print("- Left click: Add pixel to ROI")
    print("- Right click: Remove pixel from ROI")
    print("- Close window: Save ROI\n")
    print("⚠️  Interactive ROI selection is commented out by default.")
    print("Uncomment the cube.plot_interactive_rgb() calls below to use.\n")

    # Example: Create ROI for dark areas - UNCOMMENT TO USE
    # cube.plot_interactive_rgb(roi_name="dark1", use_corrected=True)
    # cube.plot_interactive_rgb(roi_name="dark_bomb", use_corrected=True)
    # cube.plot_interactive_rgb(roi_name="vertical_bomb", use_corrected=True)

    # ========================================================================
    # STEP 6: ROI VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("ROI VISUALIZATION")
    print("=" * 80)

    # Check if ROIs exist
    if hasattr(cube, "roi_collection") and cube.roi_collection:
        print(f"\nFound {len(cube.roi_collection)} saved ROIs")
        cube.list_rois()

        # Plot RGB with all ROIs
        print("\nPlotting RGB with all saved ROIs...")
        cube.plot_rgb(
            use_corrected=True,
            roi_collection="all",
            figsize=(25, 12),
        )
    else:
        print("\n⚠️  No ROIs found. Use plot_interactive_rgb() to create ROIs first.")

    # ========================================================================
    # STEP 7: SPECTRAL ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SPECTRAL ANALYSIS")
    print("=" * 80)

    # Check if ROIs exist
    if hasattr(cube, "roi_collection") and cube.roi_collection:
        # Plot spectra for all ROIs
        print("\n1. All ROI spectra (with std bands):")
        cube.plot_spectrum(
            roi_names="all",
            use_corrected=True,
            wavelength_range=(475, 650),
            wavelength_smoothing=5,
        )

        # Normalized spectra without std bands for cleaner comparison
        print("\n2. Normalized spectra (clean lines, no std):")
        cube.plot_spectrum(
            roi_names="all",
            use_corrected=True,
            wavelength_range=(475, 650),
            wavelength_smoothing=5,
            normalize=True,
            show_std=False,
        )

        # Example: Plot specific ROIs - UNCOMMENT AND MODIFY TO USE
        # example_rois = ["dark1", "dark_bomb", "vertical_bomb"]
        # if all(roi in cube.roi_collection for roi in example_rois):
        #     print(f"\n3. Specific ROIs: {example_rois}")
        #     cube.plot_spectrum(
        #         roi_names=example_rois,
        #         use_corrected=True,
        #         wavelength_range=(475, 650),
        #         wavelength_smoothing=5,
        #         normalize=True,
        #         show_std=False,
        #     )
    else:
        print("\n⚠️  No ROIs found. Create ROIs first using plot_interactive_rgb()")

    # ========================================================================
    # STEP 8: ROI MANAGEMENT
    # ========================================================================
    print("\n" + "=" * 80)
    print("ROI MANAGEMENT")
    print("=" * 80)

    if hasattr(cube, "roi_collection") and cube.roi_collection:
        # List all ROIs
        print("\nCurrent ROIs:")
        cube.list_rois()

        # Export/Import/Delete examples - UNCOMMENT TO USE
        # cube.export_rois("my_rois.json")
        # cube.import_rois("my_rois.json")
        # cube.delete_roi("roi_name_here")

        print("\n⚠️  Export/Import/Delete functions are commented out by default.")
    else:
        print("\n⚠️  No ROIs found. Create ROIs first.")

    # ========================================================================
    # ANALYSIS COMPLETE
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nAll methods are now available on the CombinedTransectCube class:")
    print("- cube.plot_rgb() - RGB visualization with ROI support")
    print("- cube.plot_line_intensity_profile() - Intensity profiles along lines")
    print("- cube.plot_interactive_rgb() - Interactive ROI selection")
    print("- cube.plot_spectrum() - Spectral analysis with ROI support")
    print("- cube.list_rois() - List all saved ROIs")
    print("- cube.export_rois() - Export ROIs to JSON")
    print("- cube.import_rois() - Import ROIs from JSON")
    print("- cube.delete_roi() - Delete specific ROI")

    return cube


if __name__ == "__main__":
    # Run the main analysis
    cube = main()

    # The cube object is now available for further interactive analysis
    print("\n💡 The 'cube' object is ready for interactive use!")
    print("   Try: cube.plot_rgb(use_corrected=True)")
    print("   Or: cube.plot_spectrum(roi_names='all', use_corrected=True)")
