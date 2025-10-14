"""
Test script to debug plot_density_scatter coordinate issues.

This script helps identify if the problem is:
1. Coordinate system mismatch (swapped E/N)
2. Sign inversion
3. Data resampling issues
4. Normalization problems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def test_coordinate_alignment(mb_ned):
    """Test if MBES and UHI coordinates are properly aligned.

    Parameters
    ----------
    mb_ned : MBESDetrender
        Detrender object with loaded MBES and UHI data
    """
    print("\n" + "=" * 60)
    print("COORDINATE ALIGNMENT DIAGNOSTIC")
    print("=" * 60)

    # Check if UHI footprint is loaded
    if mb_ned.uhi_footprint is None:
        print("❌ ERROR: UHI footprint not loaded!")
        return

    E_uhi, N_uhi, mask_valid = mb_ned.uhi_footprint

    # Basic shape checks
    print(f"\n📊 Data Shapes:")
    print(f"   MBES residuals:  {mb_ned.residuals.shape}")
    print(f"   UHI footprint:   {E_uhi.shape}")
    print(f"   MBES x_cols:     {mb_ned.x_cols.shape}")
    print(f"   MBES y_rows:     {mb_ned.y_rows.shape}")

    # Coordinate range checks
    print(f"\n📍 Coordinate Ranges:")
    print(
        f"   MBES East (x_cols):  [{mb_ned.x_cols.min():.2f}, {mb_ned.x_cols.max():.2f}] m"
    )
    print(
        f"   MBES North (y_rows): [{mb_ned.y_rows.min():.2f}, {mb_ned.y_rows.max():.2f}] m"
    )
    print(f"   UHI East:            [{np.nanmin(E_uhi):.2f}, {np.nanmax(E_uhi):.2f}] m")
    print(f"   UHI North:           [{np.nanmin(N_uhi):.2f}, {np.nanmax(N_uhi):.2f}] m")

    # Check overlap
    mbes_e_min, mbes_e_max = mb_ned.x_cols.min(), mb_ned.x_cols.max()
    mbes_n_min, mbes_n_max = mb_ned.y_rows.min(), mb_ned.y_rows.max()
    uhi_e_min, uhi_e_max = np.nanmin(E_uhi), np.nanmax(E_uhi)
    uhi_n_min, uhi_n_max = np.nanmin(N_uhi), np.nanmax(N_uhi)

    overlap_e = min(mbes_e_max, uhi_e_max) - max(mbes_e_min, uhi_e_min)
    overlap_n = min(mbes_n_max, uhi_n_max) - max(mbes_n_min, uhi_n_min)

    print(f"\n🔍 Overlap Check:")
    print(f"   East overlap:  {overlap_e:.2f} m")
    print(f"   North overlap: {overlap_n:.2f} m")

    if overlap_e <= 0 or overlap_n <= 0:
        print("   ❌ NO OVERLAP! Datasets don't intersect in NED space.")
        print("   💡 This suggests a coordinate system issue.")
    else:
        print("   ✅ Datasets overlap spatially")

    # Test meshgrid interpretation
    print(f"\n🧪 Meshgrid Test:")
    MBES_E, MBES_N = np.meshgrid(mb_ned.x_cols, mb_ned.y_rows)
    print(
        f"   meshgrid output shape: {MBES_E.shape} (should match residuals: {mb_ned.residuals.shape})"
    )
    print(f"   MBES_E[0, 0] = {MBES_E[0, 0]:.2f}, x_cols[0] = {mb_ned.x_cols[0]:.2f}")
    print(f"   MBES_N[0, 0] = {MBES_N[0, 0]:.2f}, y_rows[0] = {mb_ned.y_rows[0]:.2f}")

    if MBES_E.shape != mb_ned.residuals.shape:
        print("   ❌ SHAPE MISMATCH!")

    # Sample a few points to check coordinate interpretation
    print(f"\n🎯 Coordinate Spot Check:")
    for i, j in [(0, 0), (10, 10), (50, 50)]:
        if i < MBES_E.shape[0] and j < MBES_E.shape[1]:
            print(f"   [{i}, {j}]: E={MBES_E[i,j]:.2f}, N={MBES_N[i,j]:.2f}")
            print(
                f"           expect: E={mb_ned.x_cols[j]:.2f}, N={mb_ned.y_rows[i]:.2f}"
            )

            if abs(MBES_E[i, j] - mb_ned.x_cols[j]) > 1e-6:
                print(f"   ❌ East coordinate mismatch at [{i},{j}]!")
            if abs(MBES_N[i, j] - mb_ned.y_rows[i]) > 1e-6:
                print(f"   ❌ North coordinate mismatch at [{i},{j}]!")

    print("\n" + "=" * 60)


def test_data_correlation(mb_ned, use_adjusted=False):
    """Test correlation between MBES and UHI data.

    This helps identify if the correlation is spurious (coordinates wrong)
    or real (data actually matches).
    """
    print("\n" + "=" * 60)
    print("DATA CORRELATION DIAGNOSTIC")
    print("=" * 60)

    if mb_ned.uhi_footprint is None:
        print("❌ ERROR: UHI footprint not loaded!")
        return

    if use_adjusted and mb_ned.uhi_footprint_adjusted is not None:
        E_uhi, N_uhi, mask_valid = mb_ned.uhi_footprint_adjusted
    else:
        E_uhi, N_uhi, mask_valid = mb_ned.uhi_footprint

    # Load UHI data
    data_corr, uhi_mean, rgb, cube = mb_ned._ensure_uhi_cube_loaded(
        window_size=1000, strength=1.0
    )

    # Create MBES grid
    MBES_E, MBES_N = np.meshgrid(mb_ned.x_cols, mb_ned.y_rows)

    # Build footprint path
    from gref4hsi.final_act.utils.other.detrend_mbes import (
        pcolormesh_pad,
        _boundary_segments_from_mask,
        _trace_loops_from_segments,
        _compound_path_from_loops,
    )

    Xc_uhi, Yc_uhi = pcolormesh_pad(E_uhi, N_uhi)
    segs_xy, segs_idx = _boundary_segments_from_mask(mask_valid, Xc_uhi, Yc_uhi)
    loops = _trace_loops_from_segments(segs_idx)
    footprint_path = _compound_path_from_loops(loops, Xc_uhi, Yc_uhi)

    # Check which MBES points are inside
    pts_mbes = np.column_stack([MBES_E.ravel(), MBES_N.ravel()])
    inside = footprint_path.contains_points(pts_mbes).reshape(MBES_E.shape)

    print(f"\n📍 Overlap Statistics:")
    print(f"   Total MBES pixels: {np.prod(mb_ned.residuals.shape)}")
    print(
        f"   Inside UHI footprint: {np.sum(inside)} ({100*np.sum(inside)/np.prod(inside.shape):.1f}%)"
    )

    # Resample UHI to MBES grid
    valid_uhi = (
        mask_valid & np.isfinite(uhi_mean) & np.isfinite(E_uhi) & np.isfinite(N_uhi)
    )
    uhi_on_mbes = np.full(mb_ned.residuals.shape, np.nan, dtype=float)

    if np.any(valid_uhi):
        uhi_pts = np.column_stack([E_uhi[valid_uhi], N_uhi[valid_uhi]])
        uhi_vals = uhi_mean[valid_uhi]
        tree = cKDTree(uhi_pts)
        idx_inside = np.where(inside.ravel())[0]
        d, nn = tree.query(pts_mbes[idx_inside], k=1)
        flat = uhi_on_mbes.ravel()
        flat[idx_inside] = uhi_vals[nn]
        uhi_on_mbes = flat.reshape(MBES_E.shape)

    # Compute correlations for different scenarios
    mbes_in = np.where(inside, mb_ned.residuals, np.nan)
    uhi_in = np.where(inside, uhi_on_mbes, np.nan)

    valid_pairs = np.isfinite(mbes_in) & np.isfinite(uhi_in)
    n_valid = np.sum(valid_pairs)

    print(f"   Valid pairs for correlation: {n_valid}")

    if n_valid < 10:
        print("   ❌ Too few valid pairs!")
        return

    mbes_flat = mbes_in[valid_pairs]
    uhi_flat = uhi_in[valid_pairs]

    # Raw correlation
    r_raw = np.corrcoef(mbes_flat, uhi_flat)[0, 1]
    print(f"\n📊 Correlation Tests:")
    print(f"   Raw data correlation: {r_raw:.4f}")

    # Correlation with sign flip
    r_flip = np.corrcoef(mbes_flat, -uhi_flat)[0, 1]
    print(f"   With UHI sign flip:   {r_flip:.4f}")

    # Normalized correlation
    def robust_z(a):
        from scipy.stats import median_abs_deviation

        finite = np.isfinite(a)
        if not np.any(finite):
            return np.full_like(a, np.nan, dtype=float)
        m = np.nanmedian(a)
        mad = median_abs_deviation(a[finite], scale="normal")
        if not np.isfinite(mad) or mad == 0:
            mad = 1.0
        return (a - m) / mad

    z_mbes = robust_z(mbes_flat)
    z_uhi = robust_z(uhi_flat)
    r_norm = np.corrcoef(z_mbes, z_uhi)[0, 1]
    print(f"   Normalized (z-score):  {r_norm:.4f}")

    # Test with random shuffle (should be ~0 if spatial alignment matters)
    np.random.seed(42)
    uhi_shuffled = np.random.permutation(uhi_flat)
    r_shuffle = np.corrcoef(mbes_flat, uhi_shuffled)[0, 1]
    print(f"   With shuffled UHI:     {r_shuffle:.4f} (should be ~0)")

    if abs(r_shuffle) > 0.1:
        print("   ⚠️  WARNING: Shuffled correlation is high!")
        print("      This suggests the correlation might be spurious.")

    # Quick visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. MBES residuals
    im1 = axes[0, 0].imshow(mb_ned.residuals, cmap="RdBu_r", aspect="auto")
    axes[0, 0].set_title("MBES Residuals")
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. UHI resampled to MBES grid
    im2 = axes[0, 1].imshow(uhi_on_mbes, cmap="viridis", aspect="auto")
    axes[0, 1].set_title("UHI (resampled to MBES grid)")
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Inside mask
    axes[1, 0].imshow(inside, cmap="gray", aspect="auto")
    axes[1, 0].set_title(f"Inside Mask ({np.sum(inside)} pixels)")

    # 4. Scatter plot
    axes[1, 1].hexbin(mbes_flat, uhi_flat, gridsize=50, cmap="viridis")
    axes[1, 1].set_xlabel("MBES Residuals")
    axes[1, 1].set_ylabel("UHI Intensity")
    axes[1, 1].set_title(f"Scatter (r={r_raw:.3f})")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Import this module and run:")
    print("  test_coordinate_alignment(mb_ned)")
    print("  test_data_correlation(mb_ned)")
