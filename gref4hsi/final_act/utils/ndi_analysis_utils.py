"""
NDI Analysis Utilities for UXO Detection
=========================================

This module contains functions for computing and visualizing Normalized Difference Indices (NDI)
and spectral derivatives for underwater hyperspectral imaging analysis, specifically for
detecting corrosion, biofilm, and other surface features on underwater objects.

Author: Extracted from dev25_same_as_dev24_but_without_bug.ipynb
Date: November 3, 2025
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


def create_wavelength_colormap(wavelength):
    """
    Create a colormap from white (0) to wavelength color (1).

    Args:
        wavelength: Target wavelength in nm

    Returns:
        LinearSegmentedColormap from white to wavelength color
    """
    # Get the RGB color for this wavelength
    wl_rgb = wavelength_to_rgb(wavelength)

    # Create colormap: white (0) -> wavelength color (1)
    colors = [(1.0, 1.0, 1.0), wl_rgb]
    cmap = LinearSegmentedColormap.from_list(f"wl_{wavelength}nm", colors, N=256)

    return cmap


# ==============================================================================
# WAVELENGTH UTILITIES
# ==============================================================================


def find_closest_wavelength_index(wavelengths, target_wl):
    """
    Find the index of the wavelength closest to the target.

    Args:
        wavelengths: Array of wavelength values (nm)
        target_wl: Target wavelength (nm)

    Returns:
        Index of closest wavelength
    """
    idx = np.argmin(np.abs(wavelengths - target_wl))
    actual_wl = wavelengths[idx]
    print(f"   Target {target_wl} nm → using {actual_wl:.1f} nm (index {idx})")
    return idx


# ==============================================================================
# NDI COMPUTATION FUNCTIONS (0-1 Normalized)
# ==============================================================================


def compute_ndi(R1, R2, name="NDI"):
    """
    Compute Normalized Difference Index: (R1 - R2) / (R1 + R2)

    Args:
        R1: Reflectance at wavelength 1
        R2: Reflectance at wavelength 2
        name: Name of the index for logging

    Returns:
        NDI map (normalized to 0-1)
    """
    # Avoid division by zero
    denominator = R1 + R2
    denominator[denominator == 0] = 1e-10

    ndi = (R1 - R2) / denominator

    # Normalize to 0-1 range for visualization
    ndi_min = np.nanmin(ndi)
    ndi_max = np.nanmax(ndi)
    ndi_normalized = (ndi - ndi_min) / (ndi_max - ndi_min)

    print(f"\n{name}:")
    print(f"   Raw range: [{ndi_min:.4f}, {ndi_max:.4f}]")
    print(f"   Normalized to: [0, 1]")

    return ndi_normalized


def compute_first_derivative(
    data, wavelengths, wl_start, wl_end, name="1st Derivative"
):
    """
    Compute first derivative (slope) of spectrum between two wavelengths.

    Args:
        data: Hyperspectral datacube (tracks, slits, wavelengths)
        wavelengths: Array of wavelength values
        wl_start: Start wavelength for derivative
        wl_end: End wavelength for derivative
        name: Name of the derivative for logging

    Returns:
        Derivative map (normalized to 0-1)
    """
    idx_start = find_closest_wavelength_index(wavelengths, wl_start)
    idx_end = find_closest_wavelength_index(wavelengths, wl_end)

    R_start = data[:, :, idx_start]
    R_end = data[:, :, idx_end]

    delta_wl = wavelengths[idx_end] - wavelengths[idx_start]
    derivative = (R_end - R_start) / delta_wl

    # Normalize to 0-1 range
    deriv_min = np.nanmin(derivative)
    deriv_max = np.nanmax(derivative)
    deriv_normalized = (derivative - deriv_min) / (deriv_max - deriv_min)

    print(f"\n{name}:")
    print(
        f"   Wavelength range: {wavelengths[idx_start]:.1f} - {wavelengths[idx_end]:.1f} nm"
    )
    print(f"   Raw range: [{deriv_min:.6f}, {deriv_max:.6f}]")
    print(f"   Normalized to: [0, 1]")

    return deriv_normalized


def compute_second_derivative_at_wavelength(
    data, wavelengths, target_wl, window=3, name="2nd Derivative"
):
    """
    Compute second derivative at a specific wavelength using finite differences.

    Args:
        data: Hyperspectral datacube (tracks, slits, wavelengths)
        wavelengths: Array of wavelength values
        target_wl: Target wavelength for 2nd derivative
        window: Number of bands on each side for finite difference (default: 3)
        name: Name of the derivative for logging

    Returns:
        Second derivative map (normalized to 0-1)
    """
    idx = find_closest_wavelength_index(wavelengths, target_wl)

    # Need at least 'window' bands on each side
    if idx < window or idx >= len(wavelengths) - window:
        raise ValueError(
            f"Cannot compute 2nd derivative at {target_wl} nm - too close to edge"
        )

    # Use centered finite difference: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    R_center = data[:, :, idx]
    R_left = data[:, :, idx - window]
    R_right = data[:, :, idx + window]

    # For non-uniform spacing, use average of left and right spacing
    h_left = wavelengths[idx] - wavelengths[idx - window]
    h_right = wavelengths[idx + window] - wavelengths[idx]
    h_avg = (h_left + h_right) / 2

    second_deriv = (R_right - 2 * R_center + R_left) / (h_avg**2)

    # Normalize to 0-1 range
    deriv_min = np.nanmin(second_deriv)
    deriv_max = np.nanmax(second_deriv)
    deriv_normalized = (second_deriv - deriv_min) / (deriv_max - deriv_min)

    print(f"\n{name}:")
    print(f"   Center wavelength: {wavelengths[idx]:.1f} nm")
    print(f"   Window: ±{window} bands")
    print(
        f"   Left spacing: {h_left:.1f} nm, Right spacing: {h_right:.1f} nm, Avg: {h_avg:.1f} nm"
    )
    print(f"   Raw range: [{deriv_min:.8f}, {deriv_max:.8f}]")
    print(f"   Normalized to: [0, 1]")

    return deriv_normalized


def compute_brightness(data, name="Brightness"):
    """
    Compute brightness as mean reflectance across all wavelengths.

    Args:
        data: Hyperspectral datacube (tracks, slits, wavelengths)
        name: Name for logging

    Returns:
        Brightness map (normalized to 0-1)
    """
    brightness = np.mean(data, axis=2)

    # Normalize to 0-1 range
    bright_min = np.nanmin(brightness)
    bright_max = np.nanmax(brightness)
    brightness_normalized = (brightness - bright_min) / (bright_max - bright_min)

    print(f"\n{name}:")
    print(f"   Computed as mean across {data.shape[2]} wavelengths")
    print(f"   Raw range: [{bright_min:.4f}, {bright_max:.4f}]")
    print(f"   Normalized to: [0, 1]")

    return brightness_normalized


# ==============================================================================
# RAW BAND RATIO FUNCTIONS (NOT normalized to 0-1)
# ==============================================================================


def compute_band_ratio(R1, R2, name="Ratio"):
    """
    Compute simple band ratio: R1 / R2 (not normalized)

    Args:
        R1: Reflectance at wavelength 1
        R2: Reflectance at wavelength 2
        name: Name of the ratio for logging

    Returns:
        Ratio map (raw values, not normalized to 0-1)
    """
    # Avoid division by zero
    R2_safe = R2.copy()
    R2_safe[R2_safe == 0] = 1e-10

    ratio = R1 / R2_safe

    # Get statistics but DON'T normalize
    ratio_min = np.nanmin(ratio)
    ratio_max = np.nanmax(ratio)
    ratio_mean = np.nanmean(ratio)
    ratio_std = np.nanstd(ratio)

    print(f"\n{name}:")
    print(f"   Range: [{ratio_min:.4f}, {ratio_max:.4f}]")
    print(f"   Mean: {ratio_mean:.4f} ± {ratio_std:.4f}")
    print(f"   ⚠️  NOT normalized - raw ratio values")

    return ratio


# ==============================================================================
# BASELINE-CENTERED NDI FUNCTIONS (Normalized to [-1, +1] with mean=0)
# ==============================================================================


def compute_ndi_baseline_centered(R1, R2, name="NDI"):
    """
    Compute Normalized Difference Index with baseline centering: (R1 - R2) / (R1 + R2)
    Normalized to [-1, +1] range with mean at 0.

    Args:
        R1: Reflectance at wavelength 1
        R2: Reflectance at wavelength 2
        name: Name of the index for logging

    Returns:
        NDI map (normalized to [-1, +1] with mean=0)
    """
    # Avoid division by zero
    denominator = R1 + R2
    denominator[denominator == 0] = 1e-10

    ndi = (R1 - R2) / denominator

    # Subtract mean to center at 0
    ndi_mean = np.nanmean(ndi)
    ndi_centered = ndi - ndi_mean

    # Scale to [-1, +1] range
    ndi_abs_max = np.nanmax(np.abs(ndi_centered))
    if ndi_abs_max > 0:
        ndi_normalized = ndi_centered / ndi_abs_max
    else:
        ndi_normalized = ndi_centered

    print(f"\n{name} (Baseline-Centered):")
    print(f"   Raw range: [{np.nanmin(ndi):.4f}, {np.nanmax(ndi):.4f}]")
    print(f"   Raw mean (baseline): {ndi_mean:.4f}")
    print(
        f"   After centering: [{np.nanmin(ndi_centered):.4f}, {np.nanmax(ndi_centered):.4f}]"
    )
    print(
        f"   Final range: [{np.nanmin(ndi_normalized):.4f}, {np.nanmax(ndi_normalized):.4f}]"
    )
    print(f"   ✅ Normalized to [-1, +1] with mean=0")

    return ndi_normalized


def compute_first_derivative_baseline_centered(
    data, wavelengths, wl_start, wl_end, name="1st Derivative"
):
    """
    Compute first derivative with baseline centering.

    Args:
        data: Hyperspectral datacube (tracks, slits, wavelengths)
        wavelengths: Array of wavelength values
        wl_start: Start wavelength for derivative
        wl_end: End wavelength for derivative
        name: Name of the derivative for logging

    Returns:
        Derivative map (normalized to [-1, +1] with mean=0)
    """
    idx_start = find_closest_wavelength_index(wavelengths, wl_start)
    idx_end = find_closest_wavelength_index(wavelengths, wl_end)

    R_start = data[:, :, idx_start]
    R_end = data[:, :, idx_end]

    delta_wl = wavelengths[idx_end] - wavelengths[idx_start]
    derivative = (R_end - R_start) / delta_wl

    # Subtract mean to center at 0
    deriv_mean = np.nanmean(derivative)
    deriv_centered = derivative - deriv_mean

    # Scale to [-1, +1] range
    deriv_abs_max = np.nanmax(np.abs(deriv_centered))
    if deriv_abs_max > 0:
        deriv_normalized = deriv_centered / deriv_abs_max
    else:
        deriv_normalized = deriv_centered

    print(f"\n{name} (Baseline-Centered):")
    print(
        f"   Wavelength range: {wavelengths[idx_start]:.1f} - {wavelengths[idx_end]:.1f} nm"
    )
    print(f"   Raw mean (baseline): {deriv_mean:.6f}")
    print(
        f"   Final range: [{np.nanmin(deriv_normalized):.4f}, {np.nanmax(deriv_normalized):.4f}]"
    )
    print(f"   ✅ Normalized to [-1, +1] with mean=0")

    return deriv_normalized


def compute_second_derivative_baseline_centered(
    data, wavelengths, target_wl, window=3, name="2nd Derivative"
):
    """
    Compute second derivative with baseline centering.

    Args:
        data: Hyperspectral datacube (tracks, slits, wavelengths)
        wavelengths: Array of wavelength values
        target_wl: Target wavelength for 2nd derivative
        window: Number of bands on each side for finite difference (default: 3)
        name: Name of the derivative for logging

    Returns:
        Second derivative map (normalized to [-1, +1] with mean=0)
    """
    idx = find_closest_wavelength_index(wavelengths, target_wl)

    if idx < window or idx >= len(wavelengths) - window:
        raise ValueError(
            f"Cannot compute 2nd derivative at {target_wl} nm - too close to edge"
        )

    R_center = data[:, :, idx]
    R_left = data[:, :, idx - window]
    R_right = data[:, :, idx + window]

    h_left = wavelengths[idx] - wavelengths[idx - window]
    h_right = wavelengths[idx + window] - wavelengths[idx]
    h_avg = (h_left + h_right) / 2

    second_deriv = (R_right - 2 * R_center + R_left) / (h_avg**2)

    # Subtract mean to center at 0
    deriv_mean = np.nanmean(second_deriv)
    deriv_centered = second_deriv - deriv_mean

    # Scale to [-1, +1] range
    deriv_abs_max = np.nanmax(np.abs(deriv_centered))
    if deriv_abs_max > 0:
        deriv_normalized = deriv_centered / deriv_abs_max
    else:
        deriv_normalized = deriv_centered

    print(f"\n{name} (Baseline-Centered):")
    print(f"   Center wavelength: {wavelengths[idx]:.1f} nm")
    print(f"   Raw mean (baseline): {deriv_mean:.8f}")
    print(
        f"   Final range: [{np.nanmin(deriv_normalized):.4f}, {np.nanmax(deriv_normalized):.4f}]"
    )
    print(f"   ✅ Normalized to [-1, +1] with mean=0")

    return deriv_normalized


def compute_brightness_baseline_centered(data, name="Brightness"):
    """
    Compute brightness with baseline centering.

    Args:
        data: Hyperspectral datacube (tracks, slits, wavelengths)
        name: Name for logging

    Returns:
        Brightness map (normalized to [-1, +1] with mean=0)
    """
    brightness = np.mean(data, axis=2)

    # Subtract mean to center at 0
    bright_mean = np.nanmean(brightness)
    bright_centered = brightness - bright_mean

    # Scale to [-1, +1] range
    bright_abs_max = np.nanmax(np.abs(bright_centered))
    if bright_abs_max > 0:
        brightness_normalized = bright_centered / bright_abs_max
    else:
        brightness_normalized = bright_centered

    print(f"\n{name} (Baseline-Centered):")
    print(f"   Computed as mean across {data.shape[2]} wavelengths")
    print(f"   Raw mean (baseline): {bright_mean:.4f}")
    print(
        f"   Final range: [{np.nanmin(brightness_normalized):.4f}, {np.nanmax(brightness_normalized):.4f}]"
    )
    print(f"   ✅ Normalized to [-1, +1] with mean=0")

    return brightness_normalized


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================


def plot_ndi_heatmap(ndi_map, title, cmap="hot", figsize=(20, 10), track_range=None):
    """
    Plot NDI map as heatmap using plot_rgb style.

    Args:
        ndi_map: 2D array (tracks, slits) with NDI values (0-1)
        title: Plot title
        cmap: Matplotlib colormap (default: 'hot')
        figsize: Figure size (default: (20, 10))
        track_range: Optional tuple (start, end) to plot subset
    """
    if track_range is not None:
        start, end = track_range
        ndi_map = ndi_map[start : end + 1, :]

    n_tracks, n_slits = ndi_map.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Use plot_rgb style: transpose and set extent like plot_rgb does
    im = ax.imshow(
        ndi_map.T,  # Transpose to match plot_rgb orientation
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
        origin="lower",  # Match plot_rgb
        extent=[0, n_tracks, 0, n_slits],  # Match plot_rgb extent
    )

    ax.set_xlabel("Track", fontsize=14)
    ax.set_ylabel("Slit", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Index (0-1)", fontsize=12)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\n{title} Statistics:")
    print(f"   Mean: {np.nanmean(ndi_map):.4f}")
    print(f"   Std:  {np.nanstd(ndi_map):.4f}")
    print(f"   Min:  {np.nanmin(ndi_map):.4f}")
    print(f"   Max:  {np.nanmax(ndi_map):.4f}")


def plot_all_ndi_grid(ndi_maps, titles, figsize=(24, 20), track_range=None):
    """
    Plot all NDI maps in a grid layout.

    Args:
        ndi_maps: List of 2D arrays (tracks, slits)
        titles: List of titles for each map
        figsize: Figure size (default: (24, 20))
        track_range: Optional tuple (start, end) to plot subset
    """
    n_maps = len(ndi_maps)
    n_cols = 2
    n_rows = (n_maps + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Colormaps for different indices
    cmaps = ["Reds", "YlOrRd", "Greens", "YlGn", "PuOr", "gray"]

    for idx, (ndi_map, title, cmap) in enumerate(zip(ndi_maps, titles, cmaps)):
        if track_range is not None:
            start, end = track_range
            ndi_map_plot = ndi_map[start : end + 1, :]
        else:
            ndi_map_plot = ndi_map

        n_tracks, n_slits = ndi_map_plot.shape

        im = axes[idx].imshow(
            ndi_map_plot.T,  # Transpose to match plot_rgb orientation
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            origin="lower",  # Match plot_rgb
            extent=[0, n_tracks, 0, n_slits],  # Match plot_rgb extent
        )

        axes[idx].set_xlabel("Track", fontsize=11)
        axes[idx].set_ylabel("Slit", fontsize=11)
        axes[idx].set_title(title, fontsize=13, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label("Index (0-1)", fontsize=10)

    # Hide unused subplots
    for idx in range(n_maps, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "NDI Heatmaps for UXO Detection", fontsize=18, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.show()


def plot_raw_ratio_heatmap(
    ratio_map, title, cmap="hot", figsize=(24, 10), track_range=None
):
    """
    Plot raw ratio map as heatmap (auto-scaled, not fixed to 0-1).

    Args:
        ratio_map: 2D array (tracks, slits) with raw ratio values
        title: Plot title
        cmap: Matplotlib colormap (default: 'hot')
        figsize: Figure size (default: (24, 10))
        track_range: Optional tuple (start, end) to plot subset
    """
    if track_range is not None:
        start, end = track_range
        ratio_map = ratio_map[start : end + 1, :]

    fig, ax = plt.subplots(figsize=figsize)

    # Auto-scale colorbar to actual data range
    vmin = np.nanpercentile(ratio_map, 1)  # 1st percentile to avoid outliers
    vmax = np.nanpercentile(ratio_map, 99)  # 99th percentile

    n_tracks, n_slits = ratio_map.shape

    im = ax.imshow(
        ratio_map.T,  # Transpose to match plot_rgb orientation
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        origin="lower",  # Match plot_rgb
        extent=[0, n_tracks, 0, n_slits],  # Match plot_rgb extent
    )

    ax.set_xlabel("Track", fontsize=14)
    ax.set_ylabel("Slit", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Raw Ratio Value", fontsize=12)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\n{title} Statistics:")
    print(f"   Mean: {np.nanmean(ratio_map):.4f}")
    print(f"   Std:  {np.nanstd(ratio_map):.4f}")
    print(f"   Min:  {np.nanmin(ratio_map):.4f}")
    print(f"   Max:  {np.nanmax(ratio_map):.4f}")
    print(f"   1st percentile: {vmin:.4f}, 99th percentile: {vmax:.4f}")


def plot_ndi_baseline_centered(
    ndi_map, title, cmap="RdYlGn", figsize=(24, 10), track_range=None
):
    """
    Plot baseline-centered NDI map as heatmap with white at 0.

    Args:
        ndi_map: 2D array (tracks, slits) with NDI values (-1 to +1, mean=0)
        title: Plot title
        cmap: Diverging colormap with white center (default: 'RdYlGn')
              - 'RdYlGn': Red → Yellow → White (0) → Light Green → Dark Green
              - 'PiYG': Pink → White (0) → Green
              - 'RdBu_r': Blue → White (0) → Red (reversed)
              - 'PuOr_r': Orange → White (0) → Purple (reversed)
        figsize: Figure size (default: (24, 10))
        track_range: Optional tuple (start, end) to plot subset
    """
    if track_range is not None:
        start, end = track_range
        ndi_map = ndi_map[start : end + 1, :]

    fig, ax = plt.subplots(figsize=figsize)

    n_tracks, n_slits = ndi_map.shape

    im = ax.imshow(
        ndi_map.T,  # Transpose to match plot_rgb orientation
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=-1,
        vmax=1,
        origin="lower",  # Match plot_rgb
        extent=[0, n_tracks, 0, n_slits],  # Match plot_rgb extent
    )

    ax.set_xlabel("Track", fontsize=14)
    ax.set_ylabel("Slit", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Add colorbar with white at 0
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Baseline-Centered Index (-1 to +1, 0=normal)", fontsize=12)

    # Add horizontal line at 0 on colorbar
    cbar.ax.axhline(y=0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\n{title} Statistics:")
    print(f"   Mean: {np.nanmean(ndi_map):.4f} (should be ~0)")
    print(f"   Std:  {np.nanstd(ndi_map):.4f}")
    print(f"   Min:  {np.nanmin(ndi_map):.4f}")
    print(f"   Max:  {np.nanmax(ndi_map):.4f}")
    print(
        f"   Pixels > 0.5 (high): {np.sum(ndi_map > 0.5)} ({100*np.sum(ndi_map > 0.5)/ndi_map.size:.2f}%)"
    )
    print(
        f"   Pixels < -0.5 (low): {np.sum(ndi_map < -0.5)} ({100*np.sum(ndi_map < -0.5)/ndi_map.size:.2f}%)"
    )


# ==============================================================================
# GEOREFERENCED VISUALIZATION FUNCTIONS (using plot_georef)
# ==============================================================================


def plot_ndi_georef(
    cube,
    ndi_map,
    title="NDI Map",
    cmap="hot",
    vmin=0,
    vmax=1,
    use_georef=True,
    figsize=(24, 10),
    coordinate_system="NED",
    apply_alignment_shift=True,
    track_start=None,
    track_end=None,
    **kwargs,
):
    """
    Plot NDI map using plot_georef for proper georeferencing.

    Args:
        cube: Hyperspectral cube object with georeferencing info
        ndi_map: 2D array (tracks, slits) with NDI values
        title: Plot title
        cmap: Matplotlib colormap (default: 'hot')
        vmin: Minimum value for colormap (default: 0)
        vmax: Maximum value for colormap (default: 1)
        use_georef: If True, use plot_georef; if False, use simple imshow (default: True)
        figsize: Figure size (default: (24, 10))
        coordinate_system: "NED", "ECEF", or "LATLON" (default: "NED")
        apply_alignment_shift: Apply UHI alignment shift from config (default: True)
        track_start: Start track index (optional)
        track_end: End track index (optional)
        **kwargs: Additional arguments passed to plot_georef or imshow

    Returns:
        Matplotlib figure object
    """
    if not use_georef:
        # Simple imshow plotting (non-georeferenced)
        n_tracks, n_slits = ndi_map.shape

        if track_start is not None and track_end is not None:
            ndi_map = ndi_map[track_start:track_end, :]
            n_tracks = track_end - track_start

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            ndi_map.T,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=[0, n_tracks, 0, n_slits],
        )
        ax.set_xlabel("Track", fontsize=14)
        ax.set_ylabel("Slit", fontsize=14)
        ax.set_title(title, fontsize=16, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("NDI Value", fontsize=12)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\n{title} Statistics:")
        print(f"   Mean: {np.nanmean(ndi_map):.4f}")
        print(f"   Std:  {np.nanstd(ndi_map):.4f}")
        print(f"   Min:  {np.nanmin(ndi_map):.4f}")
        print(f"   Max:  {np.nanmax(ndi_map):.4f}")

        return fig

    # Georeferenced plotting using modified cube
    # Create a temporary "pseudo-RGB" where all channels are the NDI map
    original_data = (
        cube.data_corrected if hasattr(cube, "data_corrected") else cube.data
    )
    original_wavelengths = cube.wavelengths

    # Store original data shape
    n_tracks, n_slits, n_bands = original_data.shape

    # Create pseudo-RGB datacube: replicate NDI map across 3 wavelength channels
    # This allows us to use plot_georef which expects (tracks, slits, wavelengths)
    pseudo_rgb_data = np.zeros((n_tracks, n_slits, 3))

    # Normalize NDI map to 0-1 range for RGB
    ndi_normalized = (ndi_map - vmin) / (vmax - vmin)
    ndi_normalized = np.clip(ndi_normalized, 0, 1)

    # Replicate across R, G, B channels
    for i in range(3):
        pseudo_rgb_data[:, :, i] = ndi_normalized

    # Temporarily replace cube data
    cube.data_corrected = pseudo_rgb_data
    cube.wavelengths = np.array([650.0, 550.0, 450.0])  # Dummy wavelengths for R, G, B

    try:
        # Use plot_georef with the pseudo-RGB data
        fig = cube.plot_georef(
            red_wl=650.0,
            green_wl=550.0,
            blue_wl=450.0,
            normalize=False,  # Already normalized
            figsize=figsize,
            coordinate_system=coordinate_system,
            use_corrected=True,
            apply_alignment_shift=apply_alignment_shift,
            track_start=track_start,
            track_end=track_end,
            return_fig=True,
            quiet=True,
            **kwargs,
        )

        # Modify the plot to show colormap instead of grayscale
        ax = fig.gca()

        # Clear and replot with proper colormap
        for child in ax.get_children():
            if hasattr(child, "get_array"):
                child.remove()

        # Get coordinate arrays from georeferencing
        if hasattr(ax, "_gref_coord_arrays"):
            Xp, Yp, start_idx = ax._gref_coord_arrays

            # Adjust NDI map for track range
            if track_start is not None or track_end is not None:
                start = track_start if track_start is not None else 0
                end = track_end if track_end is not None else n_tracks
                ndi_plot = ndi_map[start:end, :]
            else:
                ndi_plot = ndi_map
                start = 0

            # Create coordinate grid
            Xc = np.pad(Xp, ((0, 1), (0, 1)), mode="edge")
            Yc = np.pad(Yp, ((0, 1), (0, 1)), mode="edge")

            # Plot with colormap
            im = ax.pcolormesh(
                Xc, Yc, ndi_plot, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
            )

            # Update title
            ax.set_title(title, fontsize=16, fontweight="bold")

            # Add colorbar
            if hasattr(fig, "colorbar"):
                fig.colorbar.remove()
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("NDI Value", fontsize=12)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\n{title} Statistics:")
        print(f"   Mean: {np.nanmean(ndi_map):.4f}")
        print(f"   Std:  {np.nanstd(ndi_map):.4f}")
        print(f"   Min:  {np.nanmin(ndi_map):.4f}")
        print(f"   Max:  {np.nanmax(ndi_map):.4f}")

    finally:
        # Restore original cube data
        cube.data_corrected = original_data
        cube.wavelengths = original_wavelengths

    return fig


def plot_ndi_baseline_centered_georef(
    cube,
    ndi_map,
    title="Baseline-Centered NDI Map",
    cmap="RdYlGn",
    use_georef=True,
    figsize=(24, 10),
    coordinate_system="NED",
    apply_alignment_shift=True,
    track_start=None,
    track_end=None,
    **kwargs,
):
    """
    Plot baseline-centered NDI map using plot_georef with diverging colormap.

    Args:
        cube: Hyperspectral cube object with georeferencing info
        ndi_map: 2D array (tracks, slits) with baseline-centered NDI values (-1 to +1, mean=0)
        title: Plot title
        cmap: Diverging colormap with white center (default: 'RdYlGn')
        use_georef: If True, use plot_georef; if False, use simple imshow (default: True)
        figsize: Figure size (default: (24, 10))
        coordinate_system: "NED", "ECEF", or "LATLON" (default: "NED")
        apply_alignment_shift: Apply UHI alignment shift from config (default: True)
        track_start: Start track index (optional)
        track_end: End track index (optional)
        **kwargs: Additional arguments passed to plot_georef or imshow

    Returns:
        Matplotlib figure object
    """
    # Call plot_ndi_georef with baseline-centered parameters
    fig = plot_ndi_georef(
        cube=cube,
        ndi_map=ndi_map,
        title=title,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        use_georef=use_georef,
        figsize=figsize,
        coordinate_system=coordinate_system,
        apply_alignment_shift=apply_alignment_shift,
        track_start=track_start,
        track_end=track_end,
        **kwargs,
    )

    # Additional statistics for baseline-centered
    print(
        f"   Pixels > 0.5 (high): {np.sum(ndi_map > 0.5)} ({100*np.sum(ndi_map > 0.5)/ndi_map.size:.2f}%)"
    )
    print(
        f"   Pixels < -0.5 (low): {np.sum(ndi_map < -0.5)} ({100*np.sum(ndi_map < -0.5)/ndi_map.size:.2f}%)"
    )

    return fig


# ==============================================================================
# SINGLE WAVELENGTH VISUALIZATION WITH WAVELENGTH-SPECIFIC COLORS
# ==============================================================================


def plot_single_wavelength_with_color(
    cube_data,
    wavelengths,
    target_wavelength,
    derivative_order=0,
    window=2,
    title=None,
    figsize=(20, 10),
):
    """
    Plot single wavelength intensity with wavelength-specific color.

    The visualization uses a colormap from white (minimum value) to the
    wavelength's natural color (maximum value).

    Args:
        cube_data: Hyperspectral data cube (tracks, slits, wavelengths)
        wavelengths: Array of wavelengths (nm)
        target_wavelength: Target wavelength to visualize (nm)
        derivative_order: 0 (raw), 1 (first derivative), 2 (second derivative)
        window: Window size for derivative computation (default=2)
        title: Plot title (optional)
        figsize: Figure size (default=(20, 10))

    Returns:
        intensity_map: 2D intensity map (tracks, slits)
    """
    # Find closest wavelength index
    wl_idx = np.argmin(np.abs(wavelengths - target_wavelength))
    actual_wl = wavelengths[wl_idx]

    # Compute intensity based on derivative order
    if derivative_order == 0:
        # Raw intensity
        intensity_map = cube_data[:, :, wl_idx]
        derivative_label = "Raw Intensity"

    elif derivative_order == 1:
        # First derivative: dI/dλ
        if wl_idx < window or wl_idx >= len(wavelengths) - window:
            print(
                f"⚠️  Warning: Wavelength {target_wavelength} nm too close to edge for derivative"
            )
            return None

        intensity_forward = cube_data[:, :, wl_idx + window]
        intensity_backward = cube_data[:, :, wl_idx - window]
        wl_forward = wavelengths[wl_idx + window]
        wl_backward = wavelengths[wl_idx - window]

        intensity_map = (intensity_forward - intensity_backward) / (
            wl_forward - wl_backward
        )
        derivative_label = "1st Derivative (dI/dλ)"

    elif derivative_order == 2:
        # Second derivative: d²I/dλ²
        if wl_idx < window or wl_idx >= len(wavelengths) - window:
            print(
                f"⚠️  Warning: Wavelength {target_wavelength} nm too close to edge for 2nd derivative"
            )
            return None

        intensity_center = cube_data[:, :, wl_idx]
        intensity_forward = cube_data[:, :, wl_idx + window]
        intensity_backward = cube_data[:, :, wl_idx - window]

        h = wavelengths[wl_idx + window] - wavelengths[wl_idx]
        intensity_map = (
            intensity_forward - 2 * intensity_center + intensity_backward
        ) / (h**2)
        derivative_label = "2nd Derivative (d²I/dλ²)"

    else:
        raise ValueError(
            f"Invalid derivative_order: {derivative_order}. Must be 0, 1, or 2."
        )

    # Normalize intensity to [0, 1] range (min -> 0 = white, max -> 1 = full color)
    intensity_min = np.nanmin(intensity_map)
    intensity_max = np.nanmax(intensity_map)

    if intensity_max > intensity_min:
        intensity_normalized = (intensity_map - intensity_min) / (
            intensity_max - intensity_min
        )
    else:
        intensity_normalized = np.zeros_like(intensity_map)

    # Create wavelength-specific colormap
    cmap = create_wavelength_colormap(actual_wl)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        intensity_normalized.T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=0,
        vmax=1,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{derivative_label}\n(white = min, color = max)", fontsize=12)

    # Labels
    ax.set_xlabel("Track Index", fontsize=12)
    ax.set_ylabel("Slit Pixel Index", fontsize=12)

    # Title
    if title is None:
        title = f"{derivative_label} at {actual_wl:.1f} nm"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"📊 {derivative_label} at {actual_wl:.1f} nm")
    print(f"   Original value range: [{intensity_min:.6f}, {intensity_max:.6f}]")
    print(f"   Mean: {intensity_map.mean():.6f}, Std: {intensity_map.std():.6f}")
    print(f"   Colormap: White (min) → {wavelength_to_rgb(actual_wl)} (max)")

    return intensity_map


def plot_wavelength_grid_with_colors(
    cube_data,
    wavelengths,
    target_wavelengths,
    wavelength_labels=None,
    derivative_order=0,
    window=2,
    figsize=(24, 14),
    title=None,
):
    """
    Create a grid comparison of multiple wavelengths with wavelength-specific colors.

    Args:
        cube_data: Hyperspectral data cube (tracks, slits, wavelengths)
        wavelengths: Array of wavelengths (nm)
        target_wavelengths: List of target wavelengths to plot
        wavelength_labels: List of labels for each wavelength (optional)
        derivative_order: 0 (raw), 1 (first derivative), 2 (second derivative)
        window: Window size for derivative computation (default=2)
        figsize: Figure size (default=(24, 14))
        title: Overall plot title (optional)

    Returns:
        List of intensity maps
    """
    n_wavelengths = len(target_wavelengths)
    n_cols = 3
    n_rows = (n_wavelengths + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    intensity_maps = []
    derivative_labels = [
        "Raw Intensity",
        "1st Derivative (dI/dλ)",
        "2nd Derivative (d²I/dλ²)",
    ]
    derivative_label = derivative_labels[derivative_order]

    for idx, target_wl in enumerate(target_wavelengths):
        # Find closest wavelength
        wl_idx = np.argmin(np.abs(wavelengths - target_wl))
        actual_wl = wavelengths[wl_idx]

        # Compute intensity based on derivative order
        if derivative_order == 0:
            intensity_map = cube_data[:, :, wl_idx]
        elif derivative_order == 1:
            if wl_idx < window or wl_idx >= len(wavelengths) - window:
                print(f"⚠️  Skipping {target_wl} nm: too close to edge")
                continue
            intensity_forward = cube_data[:, :, wl_idx + window]
            intensity_backward = cube_data[:, :, wl_idx - window]
            wl_forward = wavelengths[wl_idx + window]
            wl_backward = wavelengths[wl_idx - window]
            intensity_map = (intensity_forward - intensity_backward) / (
                wl_forward - wl_backward
            )
        elif derivative_order == 2:
            if wl_idx < window or wl_idx >= len(wavelengths) - window:
                print(f"⚠️  Skipping {target_wl} nm: too close to edge")
                continue
            intensity_center = cube_data[:, :, wl_idx]
            intensity_forward = cube_data[:, :, wl_idx + window]
            intensity_backward = cube_data[:, :, wl_idx - window]
            h = wavelengths[wl_idx + window] - wavelengths[wl_idx]
            intensity_map = (
                intensity_forward - 2 * intensity_center + intensity_backward
            ) / (h**2)

        intensity_maps.append(intensity_map)

        # Normalize to [0, 1]
        intensity_min = np.nanmin(intensity_map)
        intensity_max = np.nanmax(intensity_map)
        if intensity_max > intensity_min:
            intensity_normalized = (intensity_map - intensity_min) / (
                intensity_max - intensity_min
            )
        else:
            intensity_normalized = np.zeros_like(intensity_map)

        # Create wavelength-specific colormap
        cmap = create_wavelength_colormap(actual_wl)

        # Plot
        im = axes[idx].imshow(
            intensity_normalized.T,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            vmin=0,
            vmax=1,
        )

        # Label
        if wavelength_labels is not None and idx < len(wavelength_labels):
            label = wavelength_labels[idx]
        else:
            label = f"{actual_wl:.1f} nm"

        axes[idx].set_title(label, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Track", fontsize=10)
        axes[idx].set_ylabel("Slit", fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity", fontsize=9)

    # Hide unused subplots
    for idx in range(n_wavelengths, len(axes)):
        axes[idx].axis("off")

    # Overall title
    if title is None:
        title = f"Wavelength Comparison - {derivative_label}"
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    plt.tight_layout()
    plt.show()

    return intensity_maps


# ==============================================================================
# MAIN EXECUTION (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("NDI Analysis Utils - Function Test")
    print("=" * 80)
    print("\n✅ All functions loaded successfully!")
    print("\nAvailable functions:")
    print("\n🔹 Wavelength to RGB conversion:")
    print("   - wavelength_to_rgb()")
    print("   - create_wavelength_colormap()")
    print("\n🔹 Wavelength utilities:")
    print("   - find_closest_wavelength_index()")
    print("\n🔹 NDI computation (0-1 normalized):")
    print("   - compute_ndi()")
    print("   - compute_first_derivative()")
    print("   - compute_second_derivative_at_wavelength()")
    print("   - compute_brightness()")
    print("\n🔹 Raw band ratios:")
    print("   - compute_band_ratio()")
    print("\n🔹 Baseline-centered NDI ([-1, +1] with mean=0):")
    print("   - compute_ndi_baseline_centered()")
    print("   - compute_first_derivative_baseline_centered()")
    print("   - compute_second_derivative_baseline_centered()")
    print("   - compute_brightness_baseline_centered()")
    print("\n🔹 Visualization:")
    print("   - plot_ndi_heatmap()")
    print("   - plot_all_ndi_grid()")
    print("   - plot_raw_ratio_heatmap()")
    print("   - plot_ndi_baseline_centered()")
    print("\n🔹 Georeferenced visualization (using plot_georef):")
    print("   - plot_ndi_georef()")
    print("   - plot_ndi_baseline_centered_georef()")
    print("\n🔹 Wavelength-specific color visualization:")
    print("   - plot_single_wavelength_with_color()")
    print("   - plot_wavelength_grid_with_colors()")
    print("\n" + "=" * 80)
