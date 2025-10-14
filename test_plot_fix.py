"""
Quick diagnostic and fix script for plot_density_scatter issue.

USAGE:
------
In your notebook, after loading mb_ned:

    from test_plot_fix import diagnose_and_fix
    diagnose_and_fix(mb_ned)

This will:
1. Run diagnostics to identify the issue
2. Show the problem
3. Apply the fix
4. Show the corrected plot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, median_abs_deviation


def diagnose_and_fix(mb_ned):
    """Run full diagnostic and show before/after comparison."""

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: plot_density_scatter issue")
    print("=" * 70)

    # Check basics
    if mb_ned.uhi_footprint is None:
        print("❌ ERROR: UHI footprint not loaded!")
        print("   Make sure mb_ned was created with:")
        print("   - coord_system='ned'")
        print("   - ned_origin=(lon0, lat0, h0)")
        print("   - uhi_transect_folder, uhi_files, uhi_track_range")
        return

    if mb_ned.residuals is None:
        print("❌ ERROR: Residuals not computed!")
        print("   Call mb_ned.detrend() first")
        return

    print("✅ Basic checks passed")

    # Get data
    E_uhi, N_uhi, mask_valid = mb_ned.uhi_footprint
    H, W = mb_ned.residuals.shape

    print(f"\n📊 Data Info:")
    print(f"   MBES shape: {mb_ned.residuals.shape}")
    print(f"   UHI shape:  {E_uhi.shape}")
    print(f"   x_cols (East):  {len(mb_ned.x_cols)} elements")
    print(f"   y_rows (North): {len(mb_ned.y_rows)} elements")

    # Check coordinate ranges
    print(f"\n📍 Coordinate Ranges (NED):")
    print(f"   MBES East:  [{mb_ned.x_cols.min():8.2f}, {mb_ned.x_cols.max():8.2f}] m")
    print(f"   MBES North: [{mb_ned.y_rows.min():8.2f}, {mb_ned.y_rows.max():8.2f}] m")
    print(f"   UHI East:   [{np.nanmin(E_uhi):8.2f}, {np.nanmax(E_uhi):8.2f}] m")
    print(f"   UHI North:  [{np.nanmin(N_uhi):8.2f}, {np.nanmax(N_uhi):8.2f}] m")

    # Check for overlap
    overlap_e = min(mb_ned.x_cols.max(), np.nanmax(E_uhi)) - max(
        mb_ned.x_cols.min(), np.nanmin(E_uhi)
    )
    overlap_n = min(mb_ned.y_rows.max(), np.nanmax(N_uhi)) - max(
        mb_ned.y_rows.min(), np.nanmin(N_uhi)
    )

    print(f"\n🔍 Overlap:")
    print(f"   East:  {overlap_e:8.2f} m")
    print(f"   North: {overlap_n:8.2f} m")

    if overlap_e <= 0 or overlap_n <= 0:
        print("   ❌ NO SPATIAL OVERLAP!")
        print("   → This is a coordinate system mismatch")
        return
    print("   ✅ Datasets overlap spatially")

    # Test meshgrid interpretation
    print(f"\n🧪 Testing meshgrid interpretation...")
    MBES_E, MBES_N = np.meshgrid(mb_ned.x_cols, mb_ned.y_rows)

    if MBES_E.shape != mb_ned.residuals.shape:
        print(f"   ❌ Shape mismatch!")
        print(f"      meshgrid output: {MBES_E.shape}")
        print(f"      residuals shape: {mb_ned.residuals.shape}")
        return

    # Check a few points
    errors = []
    for i, j in [(0, 0), (H // 2, W // 2), (H - 1, W - 1)]:
        e_expected = mb_ned.x_cols[j]
        n_expected = mb_ned.y_rows[i]
        e_actual = MBES_E[i, j]
        n_actual = MBES_N[i, j]

        if abs(e_actual - e_expected) > 1e-6 or abs(n_actual - n_expected) > 1e-6:
            errors.append(
                f"   [{i},{j}]: E={e_actual:.2f} vs {e_expected:.2f}, "
                f"N={n_actual:.2f} vs {n_expected:.2f}"
            )

    if errors:
        print("   ❌ Coordinate mapping errors:")
        for err in errors:
            print(err)
        return

    print("   ✅ Meshgrid interpretation correct")

    # Now run the actual scatter plot comparison
    print(f"\n" + "=" * 70)
    print("RUNNING COMPARISON: Original vs Fixed")
    print("=" * 70)

    try:
        print("\n1️⃣  Calling ORIGINAL plot_density_scatter...")
        fig_orig = mb_ned.plot_density_scatter(show=False)
        print("   ✅ Original plot created")
    except Exception as e:
        print(f"   ❌ Original failed: {e}")
        fig_orig = None

    try:
        print("\n2️⃣  Calling FIXED version...")
        # Import the fixed version
        import sys
        import os

        sys.path.insert(0, os.path.dirname(__file__))
        from plot_density_scatter_fixed import plot_density_scatter_fixed

        fig_fixed = plot_density_scatter_fixed(mb_ned, show=False)
        print("   ✅ Fixed plot created")
    except Exception as e:
        print(f"   ❌ Fixed version failed: {e}")
        import traceback

        traceback.print_exc()
        fig_fixed = None

    # Show both
    if fig_orig or fig_fixed:
        plt.show()

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print("\n💡 RECOMMENDATION:")
    print("   If the fixed version looks correct (no spurious correlation),")
    print("   the issue was in the original plot_density_scatter function.")
    print("\n   Common issues:")
    print("   - Wrong meshgrid indexing")
    print("   - Incorrect coordinate system transformation")
    print("   - Missing inside-mask application before normalization")
    print("\n   Review the differences between the two implementations.")
    print("=" * 70)


if __name__ == "__main__":
    print(__doc__)
