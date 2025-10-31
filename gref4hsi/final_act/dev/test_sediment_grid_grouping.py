"""
Test sediment grid grouping functionality
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../"))
from utils.gref_pipeline import georef


def test_sediment_grid_grouping():
    """Test that sediment uses grid grouping while bombs/dark use connected components"""

    print("=" * 60)
    print("TEST: Sediment Grid Grouping")
    print("=" * 60)

    # Create mock datacube
    n_tracks, n_slits, n_wavelengths = 2000, 968, 108
    datacube_shape = (n_tracks, n_slits, n_wavelengths)

    # Create mock ROI pixels
    roi_pixels_per_class = {}

    # Sediment: Large spread-out region (should be split into grid tiles)
    sediment_pixels = []
    for track in range(500, 1500, 50):  # Spread across 1000 tracks
        for slit in range(100, 800, 50):  # Spread across 700 slits
            sediment_pixels.append((slit, track))
    roi_pixels_per_class["training_sediment"] = sediment_pixels

    # Bombs: Two small clusters (should stay as 2 groups)
    bomb_pixels = []
    # Bomb #1
    for track in range(100, 120):
        for slit in range(50, 70):
            bomb_pixels.append((slit, track))
    # Bomb #2
    for track in range(1800, 1820):
        for slit in range(900, 920):
            bomb_pixels.append((slit, track))
    roi_pixels_per_class["training_bombs"] = bomb_pixels

    # Dark: One cluster
    dark_pixels = []
    for track in range(1000, 1050):
        for slit in range(400, 450):
            dark_pixels.append((slit, track))
    roi_pixels_per_class["training_dark"] = dark_pixels

    print(f"\n📊 Input ROIs:")
    print(f"   training_sediment: {len(sediment_pixels)} pixels")
    print(f"   training_bombs: {len(bomb_pixels)} pixels")
    print(f"   training_dark: {len(dark_pixels)} pixels")

    # Create a simple mock object with the methods we need
    class MockCube:
        def __init__(self):
            pass

        # Copy the methods from the real class
        _create_spatial_groups = georef.CombinedTransectCube._create_spatial_groups
        _create_grid_groups_for_sediment = (
            georef.CombinedTransectCube._create_grid_groups_for_sediment
        )

    cube = MockCube()

    # Call the spatial grouping function
    groups, unique_group_names = cube._create_spatial_groups(
        roi_pixels_per_class,
        datacube_shape,
        closing_radius=3,
        min_group_size=20,
        use_grid_for_sediment=True,
        sediment_grid_tile_slit=100,
        sediment_grid_tile_track=200,
        sediment_min_group_size=10,
        quiet=False,
    )

    print(f"\n✅ Grouping complete!")
    print(f"   Total pixels: {len(groups)}")
    print(f"   Unique groups: {len(unique_group_names)}")
    print(f"   Group names: {sorted(unique_group_names)}")

    # Verify sediment has many groups (grid-based)
    sediment_groups = [g for g in unique_group_names if "sed#" in g]
    bomb_groups = [g for g in unique_group_names if "bomb#" in g]
    dark_groups = [g for g in unique_group_names if "dark#" in g]

    print(f"\n🔍 Group distribution:")
    print(f"   Sediment groups (grid): {len(sediment_groups)}")
    print(f"   Bomb groups (connected): {len(bomb_groups)}")
    print(f"   Dark groups (connected): {len(dark_groups)}")

    # Assertions
    assert (
        len(sediment_groups) > 5
    ), f"Expected >5 sediment groups (grid), got {len(sediment_groups)}"
    assert len(bomb_groups) == 2, f"Expected 2 bomb groups, got {len(bomb_groups)}"
    assert len(dark_groups) == 1, f"Expected 1 dark group, got {len(dark_groups)}"

    print(f"\n✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_sediment_grid_grouping()
