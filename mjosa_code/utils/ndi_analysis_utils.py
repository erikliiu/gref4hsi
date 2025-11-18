"""
NDI Analysis Utilities for UXO Detection
=========================================

This module contains functions for computing and visualizing Normalized Difference Indices (NDI)
and spectral derivatives for underwater hyperspectral imaging analysis, specifically for
detecting corrosion, biofilm, and other surface features on underwater objects.

Author: Extracted from dev25_same_as_dev24_but_without_bug.ipynb
Date: November 3, 2025
Ported to mjosa_code: November 14, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ==============================================================================
# WAVELENGTH TO RGB CONVERSION
# ==============================================================================


def wavelength_to_rgb(wavelength):
    """
    Convert wavelength (nm) to RGB color tuple.

    Uses approximation based on visible spectrum (380-780 nm).

    Args:
        wavelength: Wavelength in nanometers (float)

    Returns:
        RGB tuple (r, g, b) with values in [0, 1]
    """
    wavelength = float(wavelength)

    # Visible spectrum boundaries
    if wavelength < 380:
        wavelength = 380
    elif wavelength > 780:
        wavelength = 780

    # RGB calculation based on wavelength ranges
    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    # Intensity correction for edges of visible spectrum
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength < 700:
        factor = 1.0
    elif 700 <= wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 1.0

    R *= factor
    G *= factor
    B *= factor

    return (R, G, B)


def create_wavelength_colormap(wavelength, darkness_factor=0.3):
    """
    Create a colormap: white -> light color -> bright color -> dark color

    For example, 660nm (red) goes: white -> pink -> bright red -> dark red

    Args:
        wavelength: Target wavelength in nm
        darkness_factor: How dark the top should be (0=black, 1=full color)
                        Default 0.3 gives dark but still recognizable color

    Returns:
        LinearSegmentedColormap: white -> light color -> bright color -> dark color
    """
    # Get the RGB color for this wavelength
    wl_rgb = wavelength_to_rgb(wavelength)

    # Create 4 color stops:
    # 0.0 = white
    # 0.33 = light/pastel version (color + white mixed)
    # 0.67 = full bright color
    # 1.0 = dark version (color * darkness_factor)

    white = (1.0, 1.0, 1.0)
    light_color = tuple((c + 2 * 1.0) / 3 for c in wl_rgb)  # Mix 1/3 color + 2/3 white
    bright_color = wl_rgb
    dark_color = tuple(c * darkness_factor for c in wl_rgb)

    # Create colormap with 4 stops
    colors = [white, light_color, bright_color, dark_color]
    positions = [0.0, 0.33, 0.67, 1.0]

    cmap = LinearSegmentedColormap.from_list(
        f"wl_{wavelength}nm", list(zip(positions, colors)), N=256
    )

    return cmap


# ==============================================================================
# MULTI-CROP WAVELENGTH ANALYSIS
# ==============================================================================


def plot_wavelength_crops(
    cube,
    crop_configs,
    wavelength_target=660,
    derivative_order=0,
    derivative_window=2,
    flip_axes=True,
    flip_horizontal=True,
    crop_aspect_ratio=3.5,
    show_file_boundaries=False,
    **kwargs,
):
    """
    Analyze and plot multiple crop regions with automatic global normalization.

    This function eliminates the need to manually copy-paste plot_rgb() calls for
    multiple regions. It analyzes all crops, finds the global normalization range,
    and plots everything with the same color scale for direct comparison.

    Args:
        cube: CombinedTransectCube object with data_corrected
        crop_configs: List of dicts with keys:
            - 'track': Center track coordinate
            - 'slit': Center slit coordinate
            - 'label': Optional label for the crop (default: "Crop N")
        wavelength_target: Target wavelength in nm (default: 660)
        derivative_order: 0 (raw), 1 (slope), or 2 (curvature) (default: 0)
        derivative_window: Window size for derivative calculation (default: 2)
        flip_axes: Transpose the image (default: True)
        flip_horizontal: Flip horizontally (default: True)
        crop_aspect_ratio: Aspect ratio for cropped plots (default: 3.5)
        show_file_boundaries: Show file boundaries in plots (default: False)
        **kwargs: Additional arguments passed to cube.plot_rgb()

    Returns:
        dict with keys:
            - 'vmin': Global minimum normalization value
            - 'vmax': Global maximum normalization value
            - 'crop_stats': List of dicts with statistics for each crop
            - 'global_min': Absolute minimum across all crops
            - 'global_max': Absolute maximum across all crops

    Example:
        ```python
        crop_regions = [
            {'track': 1258, 'slit': 212, 'label': 'Region 1'},
            {'track': 5592, 'slit': 765, 'label': 'Region 2'},
            {'track': 5160, 'slit': 613, 'label': 'Region 3'},
        ]

        # Plot all crops with automatic global normalization
        results = plot_wavelength_crops(
            cube,
            crop_regions,
            wavelength_target=660,
            derivative_order=1,
            derivative_window=2
        )

        print(f"Global range: {results['vmin']:.6f} to {results['vmax']:.6f}")
        ```
    """
    print("\n" + "=" * 80)
    print(f"🔍 MULTI-CROP WAVELENGTH ANALYSIS")
    print("=" * 80)
    print(f"📊 Analyzing {len(crop_configs)} crop regions at {wavelength_target} nm")
    print(f"📈 Derivative order: {derivative_order}")
    print("=" * 80 + "\n")

    # Find wavelength index
    wl_idx = np.argmin(np.abs(cube.wavelengths - wavelength_target))
    actual_wavelength = cube.wavelengths[wl_idx]

    crop_stats = []

    # Step 1: Analyze each crop region
    print("📋 Step 1: Analyzing each crop region...")
    print("-" * 80)

    for i, config in enumerate(crop_configs, 1):
        track = config["track"]
        slit = config["slit"]
        label = config.get("label", f"Crop {i}")

        # Extract crop region (assume default crop_width=500)
        crop_width = kwargs.get("crop_width", 500)
        half_width = crop_width // 2

        # Calculate crop bounds
        track_start = max(0, track - half_width)
        track_end = min(cube.data_corrected.shape[0], track + half_width)
        slit_start = max(0, slit - half_width)
        slit_end = min(cube.data_corrected.shape[1], slit + half_width)

        # Extract intensity for this crop
        if derivative_order == 0:
            # Raw intensity
            intensity = cube.data_corrected[
                track_start:track_end, slit_start:slit_end, wl_idx
            ]
        elif derivative_order == 1:
            # 1st derivative
            wl_forward = cube.wavelengths[wl_idx + derivative_window]
            wl_backward = cube.wavelengths[wl_idx - derivative_window]
            intensity_forward = cube.data_corrected[
                track_start:track_end, slit_start:slit_end, wl_idx + derivative_window
            ]
            intensity_backward = cube.data_corrected[
                track_start:track_end, slit_start:slit_end, wl_idx - derivative_window
            ]
            intensity = (intensity_forward - intensity_backward) / (
                wl_forward - wl_backward
            )
        else:  # derivative_order == 2
            # 2nd derivative
            h = cube.wavelengths[wl_idx + derivative_window] - cube.wavelengths[wl_idx]
            intensity_center = cube.data_corrected[
                track_start:track_end, slit_start:slit_end, wl_idx
            ]
            intensity_forward = cube.data_corrected[
                track_start:track_end, slit_start:slit_end, wl_idx + derivative_window
            ]
            intensity_backward = cube.data_corrected[
                track_start:track_end, slit_start:slit_end, wl_idx - derivative_window
            ]
            intensity = (
                intensity_forward - 2 * intensity_center + intensity_backward
            ) / (h**2)

        # Compute statistics
        stats = {
            "label": label,
            "track": track,
            "slit": slit,
            "min": np.nanmin(intensity),
            "max": np.nanmax(intensity),
            "mean": np.nanmean(intensity),
            "std": np.nanstd(intensity),
            "p2": np.nanpercentile(intensity, 2),
            "p98": np.nanpercentile(intensity, 98),
        }
        crop_stats.append(stats)

        print(f"   {label} (track={track}, slit={slit}):")
        print(f"      Raw range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"      Mean ± std: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"      Percentiles (2-98): [{stats['p2']:.6f}, {stats['p98']:.6f}]")
        print()

    # Step 2: Find global normalization range
    print("-" * 80)
    print("📊 Step 2: Computing global normalization range...")
    print("-" * 80)

    global_min = min(s["min"] for s in crop_stats)
    global_max = max(s["max"] for s in crop_stats)
    global_p2 = min(s["p2"] for s in crop_stats)
    global_p98 = max(s["p98"] for s in crop_stats)

    print(f"   Global min/max: [{global_min:.6f}, {global_max:.6f}]")
    print(f"   Global percentiles (2-98): [{global_p2:.6f}, {global_p98:.6f}]")
    print(f"   ✅ Using percentile range for normalization")
    print()

    # Step 3: Plot all crops with global normalization
    print("-" * 80)
    print("🎨 Step 3: Plotting all crops with global normalization...")
    print("-" * 80)
    print()

    # Get crop_width from kwargs (use same default as in analysis)
    crop_width = kwargs.pop("crop_width", 500)

    for i, config in enumerate(crop_configs, 1):
        track = config["track"]
        slit = config["slit"]
        label = config.get("label", f"Crop {i}")

        print(f"   [{i}/{len(crop_configs)}] Plotting {label}...")

        cube.plot_rgb(
            use_corrected=True,
            flip_axes=flip_axes,
            flip_horizontal=flip_horizontal,
            crop_center_track=track,
            crop_center_slit=slit,
            crop_width=crop_width,
            show_file_boundaries=show_file_boundaries,
            crop_aspect_ratio=crop_aspect_ratio,
            use_wavelength_colormap=True,
            wavelength_colormap_target=wavelength_target,
            derivative_order=derivative_order,
            derivative_window=derivative_window,
            vmin=global_p2,
            vmax=global_p98,
            **kwargs,
        )

    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        "vmin": global_p2,
        "vmax": global_p98,
        "crop_stats": crop_stats,
        "global_min": global_min,
        "global_max": global_max,
    }
