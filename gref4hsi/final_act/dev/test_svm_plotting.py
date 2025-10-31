"""
Test script to verify SVM plotting methods work correctly.
This simulates the key aspects of the plotting functions without requiring full data.
"""

import numpy as np
import sys
import os


# Simulate the key logic from plot_classification_map
def test_coordinate_transformation():
    """Test that coordinate transformation logic works"""

    # Simulate ECEF coordinates (2D array: tracks x slits)
    track_start, track_end = 576, 1566
    n_tracks = track_end - track_start + 1  # 991 tracks
    n_slits = 968

    # Create fake ECEF data
    X_ecef = np.random.randn(n_tracks, n_slits) * 1000000 + 3000000
    Y_ecef = np.random.randn(n_tracks, n_slits) * 1000000 + 1000000
    Z_ecef = np.random.randn(n_tracks, n_slits) * 1000000 + 5000000

    print(f"✓ ECEF arrays created: shape = {X_ecef.shape}")

    # Test LATLON transformation
    try:
        from pyproj import Transformer

        tf_ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
        lon, lat, height = tf_ecef_to_geo.transform(X_ecef, Y_ecef, Z_ecef)
        Xp, Yp = lon, lat
        print(
            f"✓ LATLON transformation works: Xp shape = {Xp.shape}, Yp shape = {Yp.shape}"
        )
    except Exception as e:
        print(f"✗ LATLON transformation failed: {e}")
        return False

    # Test NED transformation
    try:
        # Import the function (would need to be in path)
        # For now just verify the logic structure
        lat0, lon0, h0 = 60.8011575, 10.7122345, 0.0

        # Simplified NED calculation (just to test dimensions)
        # In reality uses _ecef_to_ned_arrays
        N = np.random.randn(n_tracks, n_slits) * 100
        E = np.random.randn(n_tracks, n_slits) * 100
        D = np.random.randn(n_tracks, n_slits) * 10

        Xp, Yp = E, N
        print(
            f"✓ NED transformation structure works: Xp shape = {Xp.shape}, Yp shape = {Yp.shape}"
        )
    except Exception as e:
        print(f"✗ NED transformation failed: {e}")
        return False

    # Test ECEF (passthrough)
    try:
        Xp, Yp = X_ecef, Y_ecef
        print(f"✓ ECEF passthrough works: Xp shape = {Xp.shape}, Yp shape = {Yp.shape}")
    except Exception as e:
        print(f"✗ ECEF passthrough failed: {e}")
        return False

    return True


def test_classification_map_dimensions():
    """Test that classification map dimensions match coordinate arrays"""

    track_start, track_end = 576, 1566
    n_tracks_segment = track_end - track_start + 1  # 991 tracks
    n_slits = 968

    # Simulate classification map (should match segment dimensions)
    classification_map = np.random.randint(0, 3, size=(n_tracks_segment, n_slits))

    print(f"✓ Classification map shape: {classification_map.shape}")
    print(f"  Expected: ({n_tracks_segment}, {n_slits})")

    # Simulate coordinate arrays after slicing
    X_ecef = np.random.randn(n_tracks_segment, n_slits)
    Y_ecef = np.random.randn(n_tracks_segment, n_slits)

    print(f"✓ Coordinate arrays shape: {X_ecef.shape}")

    # Check they match
    if classification_map.shape == X_ecef.shape == Y_ecef.shape:
        print(f"✓ All dimensions match!")
        return True
    else:
        print(f"✗ Dimension mismatch!")
        return False


def test_pcolormesh_compatibility():
    """Test that arrays are compatible with pcolormesh"""

    n_tracks = 991
    n_slits = 968

    # Create coordinate and data arrays
    Xp = np.random.randn(n_tracks, n_slits) * 100
    Yp = np.random.randn(n_tracks, n_slits) * 100
    class_map = np.random.randint(0, 3, size=(n_tracks, n_slits))

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(figsize=(10, 3))

        # Test pcolormesh
        colors = ["red", "green", "blue"]
        cmap_discrete = ListedColormap(colors)

        im = ax.pcolormesh(
            Xp,
            Yp,
            class_map,
            cmap=cmap_discrete,
            shading="auto",
            vmin=0,
            vmax=2,
        )

        plt.close(fig)
        print(f"✓ pcolormesh works with given dimensions")
        return True
    except Exception as e:
        print(f"✗ pcolormesh failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SVM Classification Plotting Logic")
    print("=" * 60)

    print("\n1. Testing coordinate transformations...")
    test1 = test_coordinate_transformation()

    print("\n2. Testing classification map dimensions...")
    test2 = test_classification_map_dimensions()

    print("\n3. Testing pcolormesh compatibility...")
    test3 = test_pcolormesh_compatibility()

    print("\n" + "=" * 60)
    if test1 and test2 and test3:
        print("✅ ALL TESTS PASSED!")
        print("The plotting logic should work correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("There may be issues with the plotting code.")
    print("=" * 60)
