"""
Fixed version of plot_density_scatter function.

This version includes:
1. Better coordinate validation
2. Explicit debugging output
3. Consistent normalization
4. Proper spatial resampling
"""


def plot_density_scatter_fixed(
    self,
    gridsize: int = 80,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 9),
    use_adjusted: bool = False,
    flip_uhi_sign: bool = False,
    normalization_method: str = "zscore",
    show: bool = True,
) -> plt.Figure:
    """Fixed version - plot hexbin density scatter comparing normalized UHI vs MBES.

    Key fixes:
    - Validates coordinate ranges and overlap
    - Uses consistent coordinate system
    - Proper nearest-neighbor resampling
    - Robust normalization with diagnostics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from scipy.stats import median_abs_deviation

    # Validation
    if self.coord_system.lower() != "ned":
        raise ValueError(
            "plot_density_scatter requires coord_system='ned'. "
            "Create MBESDetrender with coord_system='ned' and ned_origin=(lon0, lat0, h0)."
        )

    if self.uhi_footprint is None:
        raise ValueError(
            "UHI footprint not loaded. Ensure uhi_transect_folder, uhi_files, "
            "and uhi_track_range are set."
        )

    if self.residuals is None:
        raise ValueError("Residuals not computed. Call .detrend() before plotting.")

    if self.ned_origin is None:
        raise ValueError("ned_origin required for NED coordinate system.")

    # Select footprint (original or adjusted)
    if use_adjusted and self.uhi_footprint_adjusted is not None:
        E_uhi, N_uhi, mask_valid = self.uhi_footprint_adjusted
        using_adjusted = True
    else:
        E_uhi, N_uhi, mask_valid = self.uhi_footprint
        using_adjusted = False

    # Load UHI cube (cached after first call)
    data_corr, uhi_mean, rgb, cube = self._ensure_uhi_cube_loaded(
        window_size=1000, strength=1.0
    )

    # Build pixel-perfect footprint path
    from gref4hsi.final_act.utils.other.detrend_mbes import (
        pcolormesh_pad,
        _boundary_segments_from_mask,
        _trace_loops_from_segments,
        _compound_path_from_loops,
        _robust_z,
        _minmax_normalize,
    )

    Xc_uhi, Yc_uhi = pcolormesh_pad(E_uhi, N_uhi)
    segs_xy, segs_idx = _boundary_segments_from_mask(mask_valid, Xc_uhi, Yc_uhi)
    loops = _trace_loops_from_segments(segs_idx)
    footprint_path = _compound_path_from_loops(loops, Xc_uhi, Yc_uhi)

    # === KEY FIX: Proper MBES grid creation ===
    # MBES coordinates are already in NED after load() with coord_system='ned'
    # x_cols = East coordinates (W elements)
    # y_rows = North coordinates (H elements)
    # residuals shape = (H, W)

    H, W = self.residuals.shape
    assert len(self.y_rows) == H, f"y_rows length {len(self.y_rows)} != H {H}"
    assert len(self.x_cols) == W, f"x_cols length {len(self.x_cols)} != W {W}"

    # Create coordinate arrays for each MBES pixel
    # Standard meshgrid: meshgrid(x, y) returns arrays of shape (len(y), len(x))
    # where first output varies along x (columns) and second varies along y (rows)
    MBES_E, MBES_N = np.meshgrid(self.x_cols, self.y_rows)

    # Validate shapes
    assert (
        MBES_E.shape == self.residuals.shape
    ), f"MBES_E shape {MBES_E.shape} != residuals shape {self.residuals.shape}"
    assert (
        MBES_N.shape == self.residuals.shape
    ), f"MBES_N shape {MBES_N.shape} != residuals shape {self.residuals.shape}"

    # Validate coordinate mapping
    # MBES_E[i, j] should equal x_cols[j]
    # MBES_N[i, j] should equal y_rows[i]
    assert np.allclose(MBES_E[0, :], self.x_cols), "E coordinate mapping incorrect!"
    assert np.allclose(MBES_N[:, 0], self.y_rows), "N coordinate mapping incorrect!"

    print(f"\n🔍 Coordinate Validation:")
    print(f"   MBES shape: {self.residuals.shape}")
    print(f"   MBES_E range: [{MBES_E.min():.2f}, {MBES_E.max():.2f}] m")
    print(f"   MBES_N range: [{MBES_N.min():.2f}, {MBES_N.max():.2f}] m")
    print(f"   UHI E range:  [{np.nanmin(E_uhi):.2f}, {np.nanmax(E_uhi):.2f}] m")
    print(f"   UHI N range:  [{np.nanmin(N_uhi):.2f}, {np.nanmax(N_uhi):.2f}] m")

    # Determine which MBES pixels are inside UHI footprint
    pts_mbes = np.column_stack([MBES_E.ravel(), MBES_N.ravel()])
    inside = footprint_path.contains_points(pts_mbes).reshape(MBES_E.shape)

    n_inside = np.sum(inside)
    print(
        f"   MBES pixels inside UHI footprint: {n_inside} / {np.prod(inside.shape)} "
        f"({100*n_inside/np.prod(inside.shape):.1f}%)"
    )

    if n_inside == 0:
        raise RuntimeError(
            "No MBES pixels inside UHI footprint! Check coordinate system consistency."
        )

    # Resample UHI mean to MBES grid using nearest neighbor
    if self._uhi_resampled_cache is not None and not using_adjusted:
        uhi_on_mbes = self._uhi_resampled_cache
        print(f"   Using cached UHI resampling")
    else:
        valid_uhi = (
            mask_valid & np.isfinite(uhi_mean) & np.isfinite(E_uhi) & np.isfinite(N_uhi)
        )
        uhi_on_mbes = np.full(self.residuals.shape, np.nan, dtype=float)

        if np.any(valid_uhi):
            # Build KDTree from valid UHI points
            uhi_pts = np.column_stack([E_uhi[valid_uhi], N_uhi[valid_uhi]])
            uhi_vals = uhi_mean[valid_uhi]
            tree = cKDTree(uhi_pts)

            # Query only for MBES points inside footprint
            idx_inside = np.where(inside.ravel())[0]
            d, nn = tree.query(pts_mbes[idx_inside], k=1)

            # Store resampled values
            flat = uhi_on_mbes.ravel()
            flat[idx_inside] = uhi_vals[nn]
            uhi_on_mbes = flat.reshape(MBES_E.shape)

            # Cache the resampled data
            if not using_adjusted:
                self._uhi_resampled_cache = uhi_on_mbes

            print(
                f"   Resampled UHI to MBES grid: {np.sum(np.isfinite(uhi_on_mbes))} valid pixels"
            )

    # === KEY FIX: Apply inside mask BEFORE normalization ===
    # Only use pixels inside the overlap region
    mbes_in = np.where(inside, self.residuals, np.nan)
    uhi_in = np.where(inside, uhi_on_mbes, np.nan)

    # Optionally flip UHI sign
    if flip_uhi_sign:
        uhi_in = -uhi_in
        print(f"   Applied UHI sign flip")

    # Check data ranges before normalization
    print(f"\n📊 Raw Data (inside overlap only):")
    print(f"   MBES range: [{np.nanmin(mbes_in):.4f}, {np.nanmax(mbes_in):.4f}]")
    print(f"   UHI range:  [{np.nanmin(uhi_in):.4f}, {np.nanmax(uhi_in):.4f}]")

    # Apply normalization method
    if normalization_method.lower() == "minmax":
        z_mbes = _minmax_normalize(mbes_in)
        z_uhi = _minmax_normalize(uhi_in)
        norm_label = "normalized [0-1]"
    elif normalization_method.lower() == "zscore":
        z_mbes = _robust_z(mbes_in)
        z_uhi = _robust_z(uhi_in)
        norm_label = "z-units"
    else:
        raise ValueError(
            f"Invalid normalization_method: {normalization_method}. "
            f"Choose 'zscore' or 'minmax'."
        )

    print(f"\n📊 After {normalization_method} normalization:")
    print(f"   z_MBES range: [{np.nanmin(z_mbes):.4f}, {np.nanmax(z_mbes):.4f}]")
    print(f"   z_UHI range:  [{np.nanmin(z_uhi):.4f}, {np.nanmax(z_uhi):.4f}]")

    # Extract valid pairs for scatter plot
    valid_pairs = np.isfinite(z_mbes) & np.isfinite(z_uhi)
    zM = z_mbes[valid_pairs].ravel()
    zU = z_uhi[valid_pairs].ravel()

    if zM.size == 0:
        raise RuntimeError(
            "No overlapping valid samples to plot. Check footprints/masks."
        )

    print(f"   Valid pairs for correlation: {zM.size}")

    # Compute statistics
    pearson_r = float(np.corrcoef(zM, zU)[0, 1]) if zM.size > 1 else np.nan
    from scipy.stats import spearmanr

    spearman_rho = (
        float(spearmanr(zM, zU, nan_policy="omit").correlation)
        if zM.size > 1
        else np.nan
    )
    # Linear fit: zU = a + b*zM
    b, a = np.polyfit(zM, zU, 1) if zM.size > 1 else (np.nan, np.nan)
    r2 = float(pearson_r**2) if np.isfinite(pearson_r) else np.nan

    print(f"\n📈 Correlation Results:")
    print(f"   Pearson r:  {pearson_r:.4f}")
    print(f"   Spearman ρ: {spearman_rho:.4f}")
    print(f"   Linear fit: y = {a:.3f} + {b:.3f}x  (R² = {r2:.3f})")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, num="Density Scatter: UHI vs MBES (FIXED)")

    # Hexbin density plot
    hb = ax.hexbin(zM, zU, gridsize=gridsize, mincnt=1, cmap=cmap)

    # Determine axis limits (symmetric)
    lim = np.nanmax(np.abs([np.nanmin([zM, zU]), np.nanmax([zM, zU])]))
    lim = float(np.clip(lim, 2.0, 6.0))

    # Plot 1:1 line
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="1:1 line")

    # Plot regression line
    ax.plot(
        [-lim, lim],
        [a + b * (-lim), a + b * (lim)],
        "r-",
        lw=1.3,
        label=f"fit: y = {a:.2f} + {b:.2f}x",
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"MBES residuals ({norm_label})", fontsize=12)
    ax.set_ylabel(f"UHI mean intensity ({norm_label})", fontsize=12)
    ax.set_title(
        f"Density Scatter: UHI vs MBES ({norm_label}) [FIXED VERSION]", fontsize=14
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Colorbar
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Count", fontsize=11)

    # Legend
    ax.legend(loc="upper left", fontsize=10)

    # Add statistics text box
    stats_text = (
        f"n = {zM.size:,}\n"
        f"r = {pearson_r:.3f}\n"
        f"ρ = {spearman_rho:.3f}\n"
        f"R² = {r2:.3f}"
    )
    ax.text(
        0.97,
        0.03,
        stats_text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=10,
    )

    if show:
        plt.show()

    return fig


# Monkey-patch the method (use with caution!)
# MBESDetrender.plot_density_scatter_fixed = plot_density_scatter_fixed
