# ============================================================================
# RAW BAND RATIO CODE FOR DEV24
# Copy-paste these cells into dev24_plot_NDI.ipynb at the bottom
# ============================================================================

# CELL 1: MARKDOWN HEADER
"""
---

## Raw Band Ratios (Non-Normalized)

Instead of NDI normalization `(R1-R2)/(R1+R2)`, compute simple ratios `R1/R2` to see raw spectral differences.
"""


# CELL 2: COMPUTE FUNCTION
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


print("✅ Raw ratio function defined!")


# CELL 3: COMPUTE RAW RATIOS
print("=" * 60)
print("🔬 COMPUTING RAW BAND RATIOS")
print("=" * 60)

# 1. RUST RATIO: R600 / R500
print("\n" + "=" * 60)
print("1️⃣ RUST RATIO - R600/R500")
print("=" * 60)
rust_ratio = compute_band_ratio(R600, R500, name="Rust Ratio (R600/R500)")

# 2. CHLOROPHYLL-A RATIO: R550 / R675
print("\n" + "=" * 60)
print("2️⃣ CHLOROPHYLL-A RATIO - R550/R675")
print("=" * 60)
chla_ratio = compute_band_ratio(R550, R675, name="Chl-a Ratio (R550/R675)")

# 3. CYANOBACTERIA RATIO: R550 / R620
print("\n" + "=" * 60)
print("3️⃣ CYANOBACTERIA RATIO - R550/R620")
print("=" * 60)
cyano_ratio = compute_band_ratio(R550, R620, name="Cyano Ratio (R550/R620)")

print("\n" + "=" * 60)
print("✅ ALL RAW RATIOS COMPUTED!")
print("=" * 60)


# CELL 4: PLOTTING FUNCTION
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

    im = ax.imshow(
        ratio_map.T,  # Transpose for correct orientation
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
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


print("✅ Raw ratio plotting function defined!")


# CELL 5: MARKDOWN HEADER
"""
### Plot Raw Ratio Maps
"""


# CELL 6: PLOT RUST RATIO
# 1. Rust Ratio
plot_raw_ratio_heatmap(
    rust_ratio,
    title="Rust Ratio - R600/R500 (Raw)\nDetects: Red/green ratio (rust signature)",
    cmap="Reds",
    figsize=(24, 10),
)


# CELL 7: PLOT CHLOROPHYLL RATIO
# 2. Chlorophyll-a Ratio
plot_raw_ratio_heatmap(
    chla_ratio,
    title="Chlorophyll-a Ratio - R550/R675 (Raw)\nDetects: Green/red ratio (chlorophyll absorption)",
    cmap="Greens",
    figsize=(24, 10),
)


# CELL 8: PLOT CYANOBACTERIA RATIO
# 3. Cyanobacteria Ratio
plot_raw_ratio_heatmap(
    cyano_ratio,
    title="Cyanobacteria Ratio - R550/R620 (Raw)\nDetects: Green/orange ratio (phycobilin absorption)",
    cmap="PuOr",
    figsize=(24, 10),
)


# CELL 9: MARKDOWN - COMPARISON
"""
### Comparison: NDI vs Raw Ratios

**NDI (Normalized Difference Index):**
- Formula: `(R1-R2)/(R1+R2)`
- Range: Always between -1 and +1
- Then normalized to 0-1 for visualization
- Better for **comparing different materials** (normalized scale)

**Raw Ratios:**
- Formula: `R1/R2`
- Range: Varies depending on absolute reflectance values
- Not normalized
- Better for **absolute spectral signatures** (preserves magnitude)

**When to use which:**
- Use **NDI** when you want to compare spatial patterns independent of brightness
- Use **Raw Ratios** when absolute reflectance differences matter (e.g., depth effects, illumination)
"""
