"""
Enhanced Spectral Analysis for UHI Data
========================================

This script demonstrates the full functionality of the CombinedTransectCube class
with enhanced plotting and ROI management capabilities.

Features:
- RGB composite visualization with multi-ROI support
- Interactive ROI selection
- Intensity profile analysis along lines
- Spectral analysis with normalization options
- ROI management (save, load, export, import)

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
from utils.gref_pipeline.transect import load_transect, TransectDataSet, CombinedTransectCube

# Import configuration
sys.path.append(os.path.abspath("../../"))
from gref_pipeline import config


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("ENHANCED SPECTRAL ANALYSIS FOR UHI DATA")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD TRANSECT DATA
    # ========================================================================
    print("\n" + "="*80)
    print("LOADING TRANSECT DATA")
    print("="*80)
    
    transect = load_transect(config.OUTPUT_FOLDER)
    transect.list_files()
    
    # ========================================================================
    # STEP 2: SELECT AND COMBINE FILES
    # ========================================================================
    print("\n" + "="*80)
    print("SELECTING AND COMBINING FILES")
    print("="*80)
    
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
    print("\n" + "="*80)
    print("RGB VISUALIZATION WITH PERIMETER LINES")
    print("="*80)
    
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
    print("\n" + "="*80)
    print("INTENSITY PROFILE ANALYSIS")
    print("="*80)
    
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
    print("\n" + "="*80)
    print("INTERACTIVE ROI SELECTION")
    print("="*80)
    print("\n⚠️  Interactive ROI selection is commented out by default.")
    print("Uncomment the cube.plot_interactive_rgb() calls below to use:")
    print("\nInstructions:")
    print("- Left click: Add pixel to ROI")
    print("- Right click: Remove pixel from ROI")
    print("- Close window: Save ROI\n")
    
    # Example: Create ROI for dark areas
    # Uncomment the following lines to run interactively:
    # cube.plot_interactive_rgb(roi_name="dark1", use_corrected=True)
    # cube.plot_interactive_rgb(roi_name="dark_bomb", use_corrected=True)
    # cube.plot_interactive_rgb(roi_name="vertical_bomb", use_corrected=True)
    
    # ========================================================================
    # STEP 6: ROI VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("ROI VISUALIZATION")
    print("="*80)
    
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
    print("\n" + "="*80)
    print("SPECTRAL ANALYSIS")
    print("="*80)
    
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
        
        # Example: Plot specific ROIs only
        # Uncomment and modify the following to plot specific ROIs:
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
    print("\n" + "="*80)
    print("ROI MANAGEMENT")
    print("="*80)
    
    if hasattr(cube, "roi_collection") and cube.roi_collection:
        # List all ROIs
        print("\nCurrent ROIs:")
        cube.list_rois()
        
        # Export ROIs to file (uncomment to use)
        # print("\nExporting ROIs...")
        # cube.export_rois("my_rois.json")
        
        # Import ROIs from file (uncomment to use)
        # print("\nImporting ROIs...")
        # cube.import_rois("my_rois.json")
        
        # Delete specific ROI (uncomment and modify to use)
        # print("\nDeleting ROI...")
        # cube.delete_roi("roi_name_here")
        
        print("\n⚠️  Export/Import/Delete functions are commented out by default.")
    else:
        print("\n⚠️  No ROIs found. Create ROIs first.")
    
    # ========================================================================
    # ANALYSIS COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the main analysis
    cube = main()
    
    # The cube object is now available for further interactive analysis
    print("\n💡 The 'cube' object is ready for interactive use!")
    print("   Try: cube.plot_rgb(use_corrected=True)")
    print("   Or: cube.plot_spectrum(roi_names='all', use_corrected=True, wavelength_range=(475, 650), wavelength_smoothing=5)")
    green_wl=560,
    blue_wl=440.3,
    spacing=4,
    track_index=None,
    slit_index=None,
    normalize=True,
    show_file_boundaries=True,
    figsize=None,
    use_corrected=False,
    perimeter_line=None,
    roi_pixels=None,  # Single ROI (backward compatibility)
    roi_collection=None,  # NEW: Multiple ROIs with names
    roi_colors=["yellow", "cyan", "magenta", "orange", "lime", "red", "blue"],
    roi_marker_size=100,
    roi_show_numbers=False,
    line_colors=["red", "blue", "orange", "magenta"],
    line_width=2,
    line_style="-",
    line_labels=None,
):
    """
    Enhanced RGB plot supporting multiple named ROIs with different colors.

    NEW MULTI-ROI FEATURES:
    - roi_collection: Dict of named ROIs or 'all' to use cube.roi_collection
    - Automatic color assignment per ROI
    - Legend shows ROI names and pixel counts
    """

    # Original RGB setup (unchanged)
    cube_data = (
        self.data_corrected
        if (use_corrected and hasattr(self, "data_corrected"))
        else self.data
    )

    red_idx = np.argmin(np.abs(self.wavelengths - red_wl))
    green_idx = np.argmin(np.abs(self.wavelengths - green_wl))
    blue_idx = np.argmin(np.abs(self.wavelengths - blue_wl))

    R = cube_data[:, :, red_idx].T.copy()
    G = cube_data[:, :, green_idx].T.copy()
    B = cube_data[:, :, blue_idx].T.copy()

    if normalize:
        for C in (R, G, B):
            if C.max() != C.min():
                C[:] = (C - C.min()) / (C.max() - C.min())

    rgb_image = np.stack([R, G, B], axis=-1)
    n_tracks, n_slits = cube_data.shape[0], cube_data.shape[1]

    if figsize is None:
        figsize = (12 * spacing, 6)
    plt.figure(figsize=figsize)

    plt.imshow(
        rgb_image, aspect="auto", origin="lower", extent=[0, n_tracks, 0, n_slits]
    )

    # File boundaries (unchanged)
    if show_file_boundaries:
        for b in self.file_boundaries[1:]:
            plt.axvline(
                b["start_track"],
                color="yellow",
                linestyle=":",
                linewidth=2,
                alpha=0.8,
                label="File boundary" if b == self.file_boundaries[1] else "",
            )

    # Cross-hairs (unchanged)
    if track_index is not None:
        plt.axvline(track_index, color="cyan", linestyle="--", linewidth=2)
    if slit_index is not None:
        plt.axhline(slit_index, color="lime", linestyle="--", linewidth=2)

    # Perimeter lines (unchanged)
    if perimeter_line is not None:
        if isinstance(perimeter_line[0], (int, float)):
            lines_to_plot = [perimeter_line]
        else:
            lines_to_plot = perimeter_line

        for line_idx, line in enumerate(lines_to_plot):
            (slit1, track1), (slit2, track2) = line
            color = line_colors[line_idx % len(line_colors)]

            if line_labels and line_idx < len(line_labels):
                label = line_labels[line_idx]
            elif len(lines_to_plot) > 1:
                label = f"Line {line_idx + 1}"
            else:
                label = "Perimeter line"

            plt.plot(
                [track1, track2],
                [slit1, slit2],
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                label=label,
            )

    # NEW: Handle multiple ROIs
    if roi_collection is not None:
        # Determine which ROIs to plot
        if roi_collection == "all":
            if hasattr(self, "roi_collection") and self.roi_collection:
                rois_to_plot = self.roi_collection
            else:
                print("⚠️  No ROI collection found. Use plot_interactive_rgb() first.")
                rois_to_plot = {}
        elif isinstance(roi_collection, dict):
            rois_to_plot = roi_collection
        else:
            print("❌ roi_collection must be 'all' or a dictionary")
            rois_to_plot = {}

        # Plot each ROI with different color
        for roi_idx, (roi_name, roi_pixels_list) in enumerate(rois_to_plot.items()):
            if roi_pixels_list:
                # Validate coordinates
                valid_rois = [
                    (slit, track)
                    for slit, track in roi_pixels_list
                    if 0 <= slit < n_slits and 0 <= track < n_tracks
                ]

                if valid_rois:
                    roi_tracks = [track for slit, track in valid_rois]
                    roi_slits = [slit for slit, track in valid_rois]

                    # Assign color
                    color = roi_colors[roi_idx % len(roi_colors)]

                    # Plot ROI with unique color
                    plt.scatter(
                        roi_tracks,
                        roi_slits,
                        c=color,
                        s=roi_marker_size,
                        marker="o",
                        edgecolors="black",
                        linewidths=2,
                        alpha=0.8,
                        label=f"{roi_name} ({len(valid_rois)})",
                    )

                    # Optional: Add numbers for each ROI
                    if roi_show_numbers:
                        for i, (slit, track) in enumerate(valid_rois, 1):
                            plt.text(
                                track,
                                slit + 3,
                                str(i),
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                fontweight="bold",
                                color="black",
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor="white",
                                    alpha=0.8,
                                ),
                            )

    # Backward compatibility: single ROI support
    elif roi_pixels is not None and len(roi_pixels) > 0:
        valid_rois = [
            (slit, track)
            for slit, track in roi_pixels
            if 0 <= slit < n_slits and 0 <= track < n_tracks
        ]

        if valid_rois:
            roi_tracks = [track for slit, track in valid_rois]
            roi_slits = [slit for slit, track in valid_rois]

            plt.scatter(
                roi_tracks,
                roi_slits,
                c=roi_colors[0],
                s=roi_marker_size,
                marker="o",
                edgecolors="black",
                linewidths=2,
                alpha=0.8,
                label=f"ROI Pixels ({len(valid_rois)})",
            )

    # Grid (unchanged)
    for x in np.arange(0, n_tracks + 1, 50):
        plt.axvline(x, color="black", linewidth=0.5, alpha=0.3)
    for y in np.arange(0, n_slits + 1, 50):
        plt.axhline(y, color="black", linewidth=0.5, alpha=0.3)

    plt.xlabel("Track Index")
    plt.ylabel("Slit Pixel Index")
    title = (
        f"RGB Composite - {self.name}\n(R={red_wl}nm, G={green_wl}nm, B={blue_wl}nm)"
    )
    if use_corrected:
        title = "Corrected " + title
    plt.title(title)

    # Smart legend display
    has_overlays = (
        (show_file_boundaries and len(self.file_boundaries) > 1)
        or perimeter_line is not None
        or roi_collection is not None
        or (roi_pixels is not None and len(roi_pixels) > 0)
    )
    if has_overlays:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


# Apply the enhanced function
CombinedTransectCube.plot_rgb = plot_rgb
print("✅ RGB plot enhanced with multi-ROI support!")


def plot_line_intensity_profile(
    self,
    perimeter_line,
    wavelength=None,
    use_average=False,
    use_corrected=False,
    figsize=(12, 6),
    # line_colors=["blue", "red", "green", "orange", "purple"],
    line_colors=["red", "blue", "purple", "magenta", "pink"],
    line_width=2,
    moving_average_window=1,
    show_markers=True,
    marker_size=4,
    line_labels=None,
    normalize_intensities=False,  # NEW: Normalize intensity values
    normalization_method="minmax",  # NEW: 'minmax', 'zscore', or 'mean'
):
    """
    Plot intensity profiles with optional intensity normalization for better comparison.

    NEW NORMALIZATION OPTIONS:
    - normalize_intensities=True: Enable intensity normalization
    - normalization_method='minmax': Scale to [0,1] range
    - normalization_method='zscore': Z-score normalization (mean=0, std=1)
    - normalization_method='mean': Divide by mean (relative to average)
    """

    # Handle single vs multiple lines
    if isinstance(perimeter_line[0], (int, float)):
        lines_to_plot = [perimeter_line]
    else:
        lines_to_plot = perimeter_line

    # Validate lines
    for i, line in enumerate(lines_to_plot, 1):
        if not (isinstance(line, list) and len(line) == 2):
            raise ValueError(f"Line {i} must be [(slit1,track1), (slit2,track2)]")

    # Calculate consistent sampling length
    line_lengths = []
    for line in lines_to_plot:
        (slit1, track1), (slit2, track2) = line
        length = max(abs(track2 - track1), abs(slit2 - slit1)) + 1
        line_lengths.append(length)

    max_length = max(line_lengths)

    # Get data and wavelength setup
    cube_data = (
        self.data_corrected
        if (use_corrected and hasattr(self, "data_corrected"))
        else self.data
    )
    n_tracks, n_slits, n_wavelengths = cube_data.shape

    # Wavelength selection
    if use_average:
        mean_wl = self.wavelengths.mean()
        wl_idx = np.argmin(np.abs(self.wavelengths - mean_wl))
        method = "average"
        base_ylabel = "Average Intensity"
    elif wavelength is not None:
        wl_idx = np.argmin(np.abs(self.wavelengths - wavelength))
        method = "specific"
        base_ylabel = f"Intensity at {self.wavelengths[wl_idx]:.1f} nm"
    else:
        median_wl = np.median(self.wavelengths)
        wl_idx = np.argmin(np.abs(self.wavelengths - median_wl))
        method = "median"
        base_ylabel = f"Intensity at {self.wavelengths[wl_idx]:.1f} nm"

    # NEW: Adjust ylabel based on normalization
    if normalize_intensities:
        if normalization_method == "minmax":
            ylabel = f"Normalized {base_ylabel} [0-1]"
        elif normalization_method == "zscore":
            ylabel = f"Z-Score {base_ylabel}"
        elif normalization_method == "mean":
            ylabel = f"Relative {base_ylabel}"
        else:
            ylabel = f"Normalized {base_ylabel}"
    else:
        ylabel = base_ylabel

    actual_wl = self.wavelengths[wl_idx]

    plt.figure(figsize=figsize)
    all_results = []

    for line_idx, line in enumerate(lines_to_plot):
        (slit1, track1), (slit2, track2) = line

        # Validate coordinates
        for i, (s, t) in enumerate([(slit1, track1), (slit2, track2)], 1):
            if not (0 <= s < n_slits):
                raise ValueError(
                    f"Line {line_idx+1}, Point {i}: slit {s} out of range [0, {n_slits-1}]"
                )
            if not (0 <= t < n_tracks):
                raise ValueError(
                    f"Line {line_idx+1}, Point {i}: track {t} out of range [0, {n_tracks-1}]"
                )

        # Generate consistent sampling points
        track_indices = np.linspace(track1, track2, max_length, dtype=int)
        slit_indices = np.linspace(slit1, slit2, max_length, dtype=int)

        # Extract raw intensities
        intensities = []
        for track_idx, slit_idx in zip(track_indices, slit_indices):
            if 0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits:
                intensity = cube_data[track_idx, slit_idx, wl_idx]
                intensities.append(intensity)

        intensities = np.array(intensities)

        # NEW: Apply intensity normalization
        if normalize_intensities and len(intensities) > 0:
            if normalization_method == "minmax":
                # Scale to [0, 1]
                if intensities.max() != intensities.min():
                    intensities = (intensities - intensities.min()) / (
                        intensities.max() - intensities.min()
                    )
            elif normalization_method == "zscore":
                # Z-score normalization (mean=0, std=1)
                if intensities.std() != 0:
                    intensities = (intensities - intensities.mean()) / intensities.std()
            elif normalization_method == "mean":
                # Divide by mean (relative values)
                if intensities.mean() != 0:
                    intensities = intensities / intensities.mean()

        sample_indices = np.arange(len(intensities))

        # Apply smoothing
        if moving_average_window > 1:
            if moving_average_window > len(intensities):
                moving_average_window = len(intensities)

            window = np.ones(moving_average_window) / moving_average_window
            smoothed = np.convolve(intensities, window, mode="valid")

            std_devs = []
            for i in range(len(smoothed)):
                window_data = intensities[i : i + moving_average_window]
                std_devs.append(np.std(window_data))
            std_devs = np.array(std_devs)

            half_window = moving_average_window // 2
            plot_intensities = smoothed
            plot_sample_indices = sample_indices[
                half_window : half_window + len(smoothed)
            ]
            has_std = True
        else:
            plot_intensities = intensities
            plot_sample_indices = sample_indices
            std_devs = None
            has_std = False

        # Styling and labels
        color = line_colors[line_idx % len(line_colors)]

        if line_labels and line_idx < len(line_labels):
            label = line_labels[line_idx]
        elif len(lines_to_plot) > 1:
            label = f"Line {line_idx + 1}"
        else:
            label = "Smoothed" if has_std else "Raw intensity"

        # Plot the line
        marker_style = "o" if show_markers else None
        marker_size_actual = marker_size if show_markers else 0

        plt.plot(
            plot_sample_indices,
            plot_intensities,
            color=color,
            linewidth=line_width,
            marker=marker_style,
            markersize=marker_size_actual,
            label=label,
        )

        # Add std dev bands
        if has_std:
            plt.fill_between(
                plot_sample_indices,
                plot_intensities - std_devs,
                plot_intensities + std_devs,
                color=color,
                alpha=0.1,
            )

        all_results.append(
            {
                "line_index": line_idx + 1,
                "start_point": (slit1, track1),
                "end_point": (slit2, track2),
                "sample_indices": plot_sample_indices,
                "intensities": plot_intensities,
                "std_deviation": std_devs,
                "raw_intensities": intensities,
                "normalized": normalize_intensities,
                "normalization_method": (
                    normalization_method if normalize_intensities else None
                ),
                "color": color,
            }
        )

    # Plot formatting
    plt.xlabel("Sample Index")
    plt.ylabel(ylabel)

    # Smart title with normalization info
    if len(lines_to_plot) == 1:
        title = f"Intensity Profile ({method} wavelength)"
    else:
        title = (
            f"Multi-Line Comparison ({len(lines_to_plot)} lines, {method} wavelength)"
        )

    if normalize_intensities:
        title += f" - {normalization_method.upper()} normalized"

    if moving_average_window > 1:
        title += f"\nSmoothed with {moving_average_window}-point moving average"

    plt.title(title)
    plt.grid(True, alpha=0.3)

    if len(lines_to_plot) > 1 or has_std:
        plt.legend()

    # Updated stats
    norm_info = f" ({normalization_method} normalized)" if normalize_intensities else ""
    stats_text = f"Wavelength: {actual_wl:.1f} nm{norm_info} | Sample points: {max_length} | Lines: {len(lines_to_plot)}"
    plt.figtext(0.02, 0.01, stats_text, fontsize=9, ha="left")

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    return {
        "wavelength": actual_wl,
        "wavelength_method": method,
        "smoothing_window": moving_average_window,
        "number_of_lines": len(lines_to_plot),
        "sample_points": max_length,
        "normalized": normalize_intensities,
        "normalization_method": normalization_method if normalize_intensities else None,
        "lines": all_results,
    }


# Apply the enhanced function
CombinedTransectCube.plot_line_intensity_profile = plot_line_intensity_profile
print("✅ Added intensity normalization options!")


# Define your lines and colors
line1 = [(510, 1648), (510, 1848)]
line2 = [(390, 1820), (390, 2020)]
line3 = [(640, 1450), (640, 1650)]
line4 = [(500, 2040), (500, 2240)]

# Optimal color palette for maximum distinction
optimal_colors = ["red", "blue", "orange", "magenta"]
line_names = ["Line 1", "Line 2", "Line 3", "Line 4"]

# Plot RGB with consistent colors
cube.plot_rgb(
    perimeter_line=[line1, line2, line3, line4],
    line_colors=optimal_colors,  # ADDED: Color parameter
    line_width=3,
    use_corrected=True,
    figsize=(30, 10),
)

# Fixed intensity profiles with matching colors
cube.plot_line_intensity_profile(
    perimeter_line=[line1, line2, line3, line4],
    use_average=True,
    moving_average_window=10,
    show_markers=False,
    line_labels=line_names,
    line_colors=optimal_colors,  # ADDED: Same colors
    use_corrected=True,
)

# cube.plot_line_intensity_profile(
#     perimeter_line=[line1, line2, line3, line4],
#     use_average=True,
#     moving_average_window=10,
#     show_markers=False,
#     line_labels=line_names,
#     line_colors=optimal_colors,  # ADDED: Consistent
#     normalize_intensities=True,
#     normalization_method="minmax",
# )


cube.plot_line_intensity_profile(
    perimeter_line=[line1, line2, line3, line4],
    use_average=True,
    moving_average_window=10,
    show_markers=False,
    line_labels=line_names,
    line_colors=optimal_colors,  # ADDED: Synchronized
    normalize_intensities=True,
    normalization_method="mean",
    use_corrected=True,
)


# ROI Management Functions
def list_rois(self):
    """Display all saved ROIs"""
    if not hasattr(self, "roi_collection") or not self.roi_collection:
        print("❌ No ROIs saved yet")
        return

    print(f"\n📂 ROI Collection ({len(self.roi_collection)} ROIs):")
    print("=" * 50)
    for i, (name, pixels) in enumerate(self.roi_collection.items(), 1):
        slits = [s for s, t in pixels]
        tracks = [t for s, t in pixels]
        print(f"{i:2d}. '{name}': {len(pixels)} pixels")
        print(f"     Slit range:  {min(slits):3d} - {max(slits):3d}")
        print(f"     Track range: {min(tracks):3d} - {max(tracks):3d}")


def delete_roi(self, roi_name):
    """Delete a specific ROI"""
    if not hasattr(self, "roi_collection") or roi_name not in self.roi_collection:
        print(f"❌ ROI '{roi_name}' not found")
        return

    pixel_count = len(self.roi_collection[roi_name])
    del self.roi_collection[roi_name]
    print(f"🗑️  Deleted ROI '{roi_name}' ({pixel_count} pixels)")


def export_rois(self, filename=None):
    """Export ROIs to JSON file"""
    if not hasattr(self, "roi_collection") or not self.roi_collection:
        print("❌ No ROIs to export")
        return

    if filename is None:
        filename = f"roi_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    export_data = {
        "transect_name": self.name,
        "export_date": datetime.now().isoformat(),
        "roi_collection": self.roi_collection,
    }

    with open(filename, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"💾 Exported {len(self.roi_collection)} ROIs to {filename}")


def import_rois(self, filename):
    """Import ROIs from JSON file"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        if not hasattr(self, "roi_collection"):
            self.roi_collection = {}

        imported_rois = data["roi_collection"]
        self.roi_collection.update(imported_rois)

        print(f"📂 Imported {len(imported_rois)} ROIs from {filename}")

    except Exception as e:
        print(f"❌ Import failed: {e}")


def plot_interactive_rgb(
    self,
    red_wl=654.2,
    green_wl=560,
    blue_wl=440.3,
    normalize=True,
    use_corrected=False,
    figsize=(50, 10),
    roi_name=None,
    load_existing=True,
    highlight_color=[1, 0, 0, 0.8],  # Bright red with transparency
):
    """
    Interactive RGB with pixel highlighting and right-click deletion.

    CONTROLS:
    - Left click: Add pixel (prevents duplicates)
    - Right click: Remove pixel
    - Close window: Save ROI
    """

    # ROI management setup
    if not hasattr(self, "roi_collection"):
        self.roi_collection = {}

    if roi_name is None:
        roi_name = input("Enter ROI name: ").strip()
        if not roi_name:
            roi_name = f"ROI_{len(self.roi_collection) + 1}"

    # Load existing or start fresh
    if load_existing and roi_name in self.roi_collection:
        current_roi = set(self.roi_collection[roi_name])
        print(f"📂 Loaded '{roi_name}' with {len(current_roi)} pixels")
    else:
        current_roi = set()
        print(f"✨ Creating new ROI '{roi_name}'")

    # RGB data preparation
    cube_data = (
        self.data_corrected
        if (use_corrected and hasattr(self, "data_corrected"))
        else self.data
    )

    red_idx = np.argmin(np.abs(self.wavelengths - red_wl))
    green_idx = np.argmin(np.abs(self.wavelengths - green_wl))
    blue_idx = np.argmin(np.abs(self.wavelengths - blue_wl))

    R = cube_data[:, :, red_idx].T.copy()
    G = cube_data[:, :, green_idx].T.copy()
    B = cube_data[:, :, blue_idx].T.copy()

    if normalize:
        for C in (R, G, B):
            if C.max() != C.min():
                C[:] = (C - C.min()) / (C.max() - C.min())

    rgb_image = np.stack([R, G, B], axis=-1)
    n_tracks, n_slits = cube_data.shape[0], cube_data.shape[1]

    # Create plot with overlay
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        rgb_image, aspect="auto", origin="lower", extent=[0, n_tracks, 0, n_slits]
    )

    # Highlight overlay layer
    highlight_overlay = np.zeros((n_slits, n_tracks, 4), dtype=np.float32)
    overlay_im = ax.imshow(
        highlight_overlay,
        aspect="auto",
        origin="lower",
        extent=[0, n_tracks, 0, n_slits],
        interpolation="nearest",
    )

    def update_highlights():
        """Refresh pixel highlights"""
        highlight_overlay[:, :, :] = 0

        for slit_idx, track_idx in current_roi:
            if 0 <= slit_idx < n_slits and 0 <= track_idx < n_tracks:
                highlight_overlay[slit_idx, track_idx, :] = highlight_color

        overlay_im.set_array(highlight_overlay)
        ax.set_title(f"{roi_name}: {len(current_roi)} pixels selected")
        fig.canvas.draw_idle()

    # Initialize highlights
    update_highlights()

    def on_click(event):
        """Handle both adding and removing pixels"""
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        # Get pixel coordinates
        track_idx = int(np.round(event.xdata))
        slit_idx = int(np.round(event.ydata))

        # Validate bounds
        if not (0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits):
            print("⚠️ Click outside valid area")
            return

        pixel = (slit_idx, track_idx)

        if event.button == 1:  # Left click - ADD pixel
            if pixel in current_roi:
                print(
                    f"📌 Pixel already selected: (slit={slit_idx}, track={track_idx})"
                )
            else:
                current_roi.add(pixel)
                print(
                    f"✅ Added pixel: (slit={slit_idx}, track={track_idx}) - Total: {len(current_roi)}"
                )

        elif event.button == 3:  # Right click - REMOVE pixel
            if pixel in current_roi:
                current_roi.remove(pixel)
                print(
                    f"❌ Removed pixel: (slit={slit_idx}, track={track_idx}) - Total: {len(current_roi)}"
                )
            else:
                print(f"⚠️ Pixel not selected: (slit={slit_idx}, track={track_idx})")

        # Update display
        update_highlights()

    def on_close(event):
        """Save ROI collection"""
        if current_roi:
            self.roi_collection[roi_name] = list(current_roi)
            print(f"💾 Saved ROI '{roi_name}' with {len(current_roi)} pixels")
        else:
            print(f"🗑️ No pixels in '{roi_name}'")

    # Event handlers
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", on_close)

    # Grid for better pixel visibility
    for x in range(0, n_tracks + 1, 50):
        ax.axvline(x, color="black", linewidth=0.5, alpha=0.2)
    for y in range(0, n_slits + 1, 50):
        ax.axhline(y, color="black", linewidth=0.5, alpha=0.2)

    ax.set_xlabel("Track Index")
    ax.set_ylabel("Slit Pixel Index")

    # Updated instructions
    instructions = f"""
    🖱️ PIXEL EDITOR: '{roi_name}'
    • Left click: Add pixel (red highlight)
    • Right click: Remove pixel
    • Close window: Save ROI
    """
    fig.text(
        0.02,
        0.98,
        instructions,
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    return list(current_roi)


# Apply the enhanced function
CombinedTransectCube.plot_interactive_rgb = plot_interactive_rgb
print("✅ Added right-click deletion and red highlighting!")


# Create different ROIs for analysis
cube.plot_interactive_rgb(
    roi_name="dark1",
    use_corrected=True,
)


# After using plot_interactive_rgb(), your ROIs are saved
# Display them automatically:
cube.plot_rgb(
    use_corrected=True,
    roi_collection="all",  # Use saved ROIs
    figsize=(25, 12),
)


def plot_spectrum(
    self,
    track_index=None,
    slit_index=None,
    roi_pixels=None,
    roi_name=None,
    roi_names=None,
    ylabel="Intensity",  # Will update dynamically
    use_corrected=False,
    wavelength_range=None,
    wavelength_smoothing=1,
    figsize=(12, 6),
    colors=[
        "#FF0000",
        "#0000FF",
        "#00FFFF",
        "#FF00FF",
        "#800080",
        "#A52A2A",
        "#000000",
        "#FFFF00",
        "#00FF00",
        "#B8860B",
    ],
    use_inline_labels=True,
    normalize=False,
    show_std=True,  # NEW: Control standard deviation bands
):
    """Enhanced spectrum plotting with optional normalization, inline labels, and std control."""

    cube = (
        self.data_corrected
        if (use_corrected and hasattr(self, "data_corrected"))
        else self.data
    )
    data_label = "Corrected" if use_corrected else "Raw"

    if cube is None:
        print("❌ No data loaded.")
        return

    # ROI handling logic (unchanged)
    is_roi_analysis = False
    rois_to_plot = {}

    if roi_names is not None:
        if not hasattr(self, "roi_collection"):
            print("❌ No ROI collection found")
            return

        if roi_names == "all":
            rois_to_plot = self.roi_collection.copy()
        elif isinstance(roi_names, list):
            for name in roi_names:
                if name in self.roi_collection:
                    rois_to_plot[name] = self.roi_collection[name]
        is_roi_analysis = True

    elif roi_name is not None:
        if hasattr(self, "roi_collection") and roi_name in self.roi_collection:
            rois_to_plot[roi_name] = self.roi_collection[roi_name]
            is_roi_analysis = True
        else:
            print(f"❌ ROI '{roi_name}' not found")
            return

    elif roi_pixels is not None:
        rois_to_plot["ROI"] = roi_pixels
        is_roi_analysis = True

    elif track_index is not None and slit_index is not None:
        pass
    else:
        print("❌ Must specify pixel coordinates or ROI")
        return

    # Wavelength filtering
    wavelengths = self.wavelengths
    if wavelength_range is not None:
        wl_start, wl_end = wavelength_range
        wl_mask = (wavelengths >= wl_start) & (wavelengths <= wl_end)
        wavelengths = wavelengths[wl_mask]
        range_info = f" ({wl_start}-{wl_end}nm)"
    else:
        wl_mask = slice(None)
        range_info = ""

    def smooth_spectrum(spectrum, window_size):
        if window_size <= 1:
            return spectrum
        if window_size > len(spectrum):
            window_size = len(spectrum)
        window = np.ones(window_size) / window_size
        return np.convolve(spectrum, window, mode="valid")

    # Normalization function (only applies if normalize=True)
    def normalize_spectrum(spectrum):
        """Normalize spectrum by dividing by its mean (only if normalize=True)"""
        if normalize and len(spectrum) > 0:
            spectrum_mean = np.mean(spectrum)
            if spectrum_mean != 0:
                return spectrum / spectrum_mean
        return spectrum

    # Dynamic y-label based on normalization
    if ylabel == "Intensity":  # Only change default label
        ylabel = "Normalized Intensity" if normalize else "Intensity"

    plt.figure(figsize=figsize)

    if is_roi_analysis:
        for i, (roi_name_key, roi_pixels_list) in enumerate(rois_to_plot.items()):
            valid_spectra = []

            for slit_idx, track_idx in roi_pixels_list:
                rel_track = track_idx - self.track_offset
                if 0 <= rel_track < cube.shape[0] and 0 <= slit_idx < cube.shape[1]:
                    spectrum = cube[rel_track, slit_idx, :][wl_mask]
                    valid_spectra.append(spectrum)

            if valid_spectra:
                spectra_array = np.array(valid_spectra)
                avg_spectrum = np.mean(spectra_array, axis=0)
                std_spectrum = np.std(spectra_array, axis=0)

                # Apply smoothing first
                if wavelength_smoothing > 1:
                    smoothed_avg = smooth_spectrum(avg_spectrum, wavelength_smoothing)
                    smoothed_std = smooth_spectrum(std_spectrum, wavelength_smoothing)
                    half_window = wavelength_smoothing // 2
                    plot_wavelengths = wavelengths[
                        half_window : half_window + len(smoothed_avg)
                    ]
                    plot_avg = smoothed_avg
                    plot_std = smoothed_std
                else:
                    plot_wavelengths = wavelengths
                    plot_avg = avg_spectrum
                    plot_std = std_spectrum

                # Apply normalization (only if normalize=True)
                plot_avg = normalize_spectrum(plot_avg)
                if show_std:  # Only normalize std if we're going to show it
                    plot_std = normalize_spectrum(plot_std)

                color = colors[i % len(colors)]

                # Always plot the main line
                plt.plot(plot_wavelengths, plot_avg, color=color, linewidth=2)

                # NEW: Only show std bands if show_std=True
                if show_std:
                    plt.fill_between(
                        plot_wavelengths,
                        plot_avg - plot_std,
                        plot_avg + plot_std,
                        color=color,
                        alpha=0.15,
                    )

                if use_inline_labels:
                    label_x = plot_wavelengths[-1] * 0.95
                    label_y = plot_avg[-10:].mean()

                    plt.text(
                        label_x,
                        label_y,
                        roi_name_key,
                        color=color,
                        fontweight="bold",
                        fontsize=10,
                        ha="right",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            alpha=0.8,
                            edgecolor=color,
                        ),
                    )

        title = f"{data_label} ROI Spectra{range_info}"
        if normalize:
            title += " (Normalized)"
        if wavelength_smoothing > 1:
            title += f" (λ-smooth: {wavelength_smoothing})"
        title += f"\n({self.name})"

    else:
        # Single pixel mode
        rel_track = track_index - self.track_offset
        spectrum = cube[rel_track, slit_index, :][wl_mask]

        if wavelength_smoothing > 1:
            smoothed_spectrum = smooth_spectrum(spectrum, wavelength_smoothing)
            half_window = wavelength_smoothing // 2
            plot_wavelengths = wavelengths[
                half_window : half_window + len(smoothed_spectrum)
            ]
            plot_spectrum = smoothed_spectrum
        else:
            plot_wavelengths = wavelengths
            plot_spectrum = spectrum

        # Apply normalization (only if normalize=True)
        plot_spectrum = normalize_spectrum(plot_spectrum)

        plt.plot(plot_wavelengths, plot_spectrum, color=colors[0], linewidth=2)

        title = f"{data_label} Spectrum"
        if normalize:
            title += " (Normalized)"
        if wavelength_smoothing > 1:
            title += f" (λ-smooth: {wavelength_smoothing})"

    plt.xlabel("Wavelength (nm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Force exact wavelength range with no padding
    if wavelength_range is not None:
        ax = plt.gca()
        ax.set_xlim(wavelength_range[0], wavelength_range[1])
        ax.margins(x=0)
        ax.autoscale(enable=False, axis="x")

    if not use_inline_labels or not is_roi_analysis:
        plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the main analysis
    cube = main()
    
    # The cube object is now available for further interactive analysis
    print("\n💡 The 'cube' object is ready for interactive use!")
    print("   Try: cube.plot_rgb(use_corrected=True)")
    print("   Or: cube.plot_spectrum(roi_names='all', use_corrected=True)")
    
    # Example: If you want to plot specific ROI spectra after the main analysis
    # Uncomment these lines if you have created ROIs:
    # 
    # cube.plot_spectrum(
    #     roi_names=["dark1", "dark_bomb", "vertical_bomb", "small bomb", "bomb_right", "sediment"],
    #     use_corrected=True,
    #     wavelength_range=(475, 650),
    #     wavelength_smoothing=5,
    #     normalize=True,
    #     show_std=False,
    # )