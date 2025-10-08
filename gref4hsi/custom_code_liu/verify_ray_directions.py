"""
Verify that x_gref4hsi_by_liu ray directions match test_eely convention.

This script compares ray direction calculations between the two implementations
to ensure they produce identical results.
"""

import numpy as np
import sys
from pathlib import Path

# Add paths for both implementations
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "x_gref4hsi_by_liu"))
sys.path.insert(0, str(repo_root))

# Import Liu's utils
import utils as liu_utils


def test_eely_ray_formula(calib_dict, num_pixels):
    """
    Reproduce test_eely ray direction formula exactly.
    From gref4hsi/scripts/georeference.py:cal_file_to_rays()
    """
    f = calib_dict["f"]
    u_c = calib_dict["cx"]
    k1 = calib_dict["k1"]
    k2 = calib_dict["k2"]
    k3 = calib_dict["k3"]

    # Pixel indices (1-based)
    u = np.arange(1, num_pixels + 1)

    # Linear component
    x_norm_lin = (u - u_c) / f

    # Nonlinear distortion
    x_norm_nonlin = (
        -(
            k1 * ((u - u_c) / 1000) ** 5
            + k2 * ((u - u_c) / 1000) ** 3
            + k3 * ((u - u_c) / 1000) ** 2
        )
        / f
    )

    x_norm = x_norm_lin + x_norm_nonlin

    # Build rays
    p_dir = np.zeros((len(x_norm), 3))
    p_dir[:, 0] = x_norm
    p_dir[:, 2] = 1

    return p_dir


def compare_ray_directions(xml_path=None):
    """
    Compare ray directions from both implementations.
    """
    print("=" * 80)
    print("RAY DIRECTION VERIFICATION TEST")
    print("=" * 80)

    # Load calibration
    if xml_path is None:
        xml_path = r"E:\mjosa_new_oct_2025\use_gref4hsi\057_own_code_1to2\input\Calib\HSI_2b.xml"

    print(f"\nLoading calibration from: {xml_path}")
    calib_dict = liu_utils.load_camera_calibration(xml_path)

    print(f"\nCamera parameters:")
    print(f"  f  = {calib_dict['f']:.4f}")
    print(f"  cx = {calib_dict['cx']:.4f}")
    print(f"  k1 = {calib_dict['k1']:.6e}")
    print(f"  k2 = {calib_dict['k2']:.6e}")
    print(f"  k3 = {calib_dict['k3']:.6e}")

    # Test with typical UHI spatial binning
    num_pixels = 968  # Example: 2x binned UHI (1936 / 2)

    print(f"\nComputing ray directions for {num_pixels} pixels...")

    # Method 1: test_eely reference implementation
    rays_test_eely = test_eely_ray_formula(calib_dict, num_pixels)

    # Method 2: Your implementation (after fix)
    rays_liu = liu_utils.build_ray_directions(calib_dict, num_pixels)

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Check if arrays are identical
    max_diff = np.max(np.abs(rays_test_eely - rays_liu))
    mean_diff = np.mean(np.abs(rays_test_eely - rays_liu))

    print(f"\nArray shape test_eely: {rays_test_eely.shape}")
    print(f"Array shape liu:       {rays_liu.shape}")
    print(f"\nMaximum difference: {max_diff:.10e}")
    print(f"Mean difference:    {mean_diff:.10e}")

    if max_diff < 1e-10:
        print("\n✅ PASS: Ray directions are IDENTICAL (within numerical precision)")
    elif max_diff < 1e-6:
        print("\n⚠️  WARN: Ray directions are very close but not identical")
        print("   This is likely due to floating-point rounding")
    else:
        print("\n❌ FAIL: Ray directions differ significantly!")

    # Test specific pixels
    print("\n" + "=" * 80)
    print("SAMPLE RAY DIRECTIONS")
    print("=" * 80)

    test_pixels = [0, num_pixels // 2, num_pixels - 1]  # First, center, last
    pixel_numbers = [1, num_pixels // 2 + 1, num_pixels]  # 1-based indices

    for i, pix_num in zip(test_pixels, pixel_numbers):
        print(f"\nPixel {pix_num} (index {i}):")
        print(
            f"  test_eely: [{rays_test_eely[i,0]:+.8f}, {rays_test_eely[i,1]:+.8f}, {rays_test_eely[i,2]:+.8f}]"
        )
        print(
            f"  liu:       [{rays_liu[i,0]:+.8f}, {rays_liu[i,1]:+.8f}, {rays_liu[i,2]:+.8f}]"
        )
        print(
            f"  diff:      [{rays_test_eely[i,0]-rays_liu[i,0]:+.8e}, "
            f"{rays_test_eely[i,1]-rays_liu[i,1]:+.8e}, "
            f"{rays_test_eely[i,2]-rays_liu[i,2]:+.8e}]"
        )

    # Check center pixel (should be nearly [0, 0, 1])
    center_idx = num_pixels // 2
    center_ray = rays_liu[center_idx]
    print(
        f"\nCenter pixel ray direction: [{center_ray[0]:+.8f}, {center_ray[1]:+.8f}, {center_ray[2]:+.8f}]"
    )
    if (
        abs(center_ray[0]) < 0.01
        and abs(center_ray[1]) < 1e-10
        and abs(center_ray[2] - 1.0) < 1e-10
    ):
        print("  ✅ Center pixel looks correct (nearly [0, 0, 1])")
    else:
        print("  ⚠️  Center pixel doesn't look nadir-pointing")

    # Check edge pixels (should fan out symmetrically)
    edge_left = rays_liu[0, 0]
    edge_right = rays_liu[-1, 0]
    print(f"\nEdge pixel X-components:")
    print(f"  Left edge:  {edge_left:+.8f}")
    print(f"  Right edge: {edge_right:+.8f}")
    if abs(edge_left + edge_right) < 0.01:  # Should be nearly symmetric
        print("  ✅ Edge pixels are symmetric")
    else:
        print("  ⚠️  Edge pixels asymmetric (check calibration cx)")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80 + "\n")

    return max_diff < 1e-6


if __name__ == "__main__":
    try:
        success = compare_ray_directions()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
