"""
Visualize DEM coverage and HSI ray intersection locations from above (2D plot).
This helps diagnose why some files georeference successfully while others fail.

The DEM shown is the MERGED/COMBINED DEM created from altimeter data of ALL files.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import rasterio
from pathlib import Path
import sys
import pandas as pd


def plot_dem_and_altimeter_contributions(data_dir):
    """
    Create a 2D plot showing:
    1. DEM extent and elevation
    2. Ray intersection points from each H5 file
    3. Spatial overlap analysis
    """
    data_dir = Path(data_dir)

    # Paths
    dem_path = data_dir / "Input" / "GIS" / "dummy_dem.tif"
    h5_dir = data_dir / "Input" / "H5"
    output_dir = data_dir / "Output"

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ==================== Plot 1: DEM Coverage ====================
    ax1 = axes[0]

    # Load and plot DEM
    try:
        with rasterio.open(dem_path) as dem:
            dem_data = dem.read(1)
            dem_bounds = dem.bounds
            transform = dem.transform

            # Get pixel coordinates for plotting
            rows, cols = dem_data.shape
            x_coords = [dem_bounds.left + col * transform.a for col in range(cols)]
            y_coords = [dem_bounds.top + row * transform.e for row in range(rows)]

            # Plot DEM as heatmap
            im = ax1.imshow(
                dem_data,
                extent=[
                    dem_bounds.left,
                    dem_bounds.right,
                    dem_bounds.bottom,
                    dem_bounds.top,
                ],
                cmap="terrain",
                aspect="auto",
                alpha=0.7,
            )
            plt.colorbar(im, ax=ax1, label="Elevation (m)")

            ax1.set_title("DEM Coverage and Elevation", fontsize=14, fontweight="bold")
            ax1.set_xlabel("East (m)", fontsize=12)
            ax1.set_ylabel("North (m)", fontsize=12)
            ax1.grid(True, alpha=0.3)

            print(f"✅ DEM loaded successfully")
            print(
                f"   Bounds: East [{dem_bounds.left:.1f}, {dem_bounds.right:.1f}], "
                f"North [{dem_bounds.bottom:.1f}, {dem_bounds.top:.1f}]"
            )
            print(
                f"   Elevation range: {np.nanmin(dem_data):.1f} to {np.nanmax(dem_data):.1f} m"
            )

    except Exception as e:
        print(f"❌ Error loading DEM: {e}")
        ax1.text(
            0.5,
            0.5,
            f"DEM not found\n{dem_path}",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    # ==================== Plot 2: Ray Intersections ====================
    ax2 = axes[1]

    # Plot DEM outline on second plot too
    try:
        with rasterio.open(dem_path) as dem:
            dem_bounds = dem.bounds
            # Draw DEM boundary rectangle
            rect = plt.Rectangle(
                (dem_bounds.left, dem_bounds.bottom),
                dem_bounds.right - dem_bounds.left,
                dem_bounds.top - dem_bounds.bottom,
                fill=False,
                edgecolor="black",
                linewidth=2,
                linestyle="--",
                label="DEM Boundary",
            )
            ax2.add_patch(rect)
    except:
        pass

    # Load intersection points from georeferenced H5 files
    colors = ["blue", "red", "green", "orange", "purple"]
    h5_files = sorted(h5_dir.glob("*.h5"))

    print(f"\n📍 Loading intersection points from {len(h5_files)} H5 files:")

    for i, h5_file in enumerate(h5_files):
        try:
            with h5py.File(h5_file, "r") as f:
                # Try to find georeferenced intersection points
                # They might be in different locations depending on processing stage
                possible_paths = [
                    "/processed/geometry/points_local",
                    "/processed/georef/points_local",
                    "/georef/points_local",
                ]

                points = None
                for path in possible_paths:
                    if path in f:
                        points = f[path][:]
                        break

                if points is not None and len(points) > 0:
                    # Points are typically in NED coordinates (North, East, Down)
                    # Plot North vs East
                    north = points[:, 0]
                    east = points[:, 1]

                    ax2.scatter(
                        east,
                        north,
                        c=colors[i % len(colors)],
                        s=1,
                        alpha=0.5,
                        label=f"File {i+1}: {h5_file.name}",
                    )

                    print(f"   File {i+1}: {len(points)} intersection points")
                    print(f"      North: [{north.min():.1f}, {north.max():.1f}] m")
                    print(f"      East:  [{east.min():.1f}, {east.max():.1f}] m")
                else:
                    print(
                        f"   File {i+1}: No intersection data found (might not be georeferenced yet)"
                    )

        except Exception as e:
            print(f"   File {i+1}: Error loading - {e}")

    ax2.set_title("Ray Intersection Points by File", fontsize=14, fontweight="bold")
    ax2.set_xlabel("East (m)", fontsize=12)
    ax2.set_ylabel("North (m)", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    # Add main title explaining what's shown
    fig.suptitle(
        "DEM Coverage Analysis (MERGED DEM from all files' altimeter data)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Display without saving
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.show(block=True)  # Block until user closes window

    print("\n� Plot displayed. Close the window to continue.")


def analyze_coverage_overlap(data_dir):
    """
    Analyze spatial overlap between DEM and HSI coverage areas
    """
    data_dir = Path(data_dir)
    dem_path = data_dir / "Input" / "GIS" / "dummy_dem.tif"

    print("\n" + "=" * 60)
    print("SPATIAL COVERAGE ANALYSIS")
    print("=" * 60)

    try:
        with rasterio.open(dem_path) as dem:
            dem_bounds = dem.bounds
            dem_area = (dem_bounds.right - dem_bounds.left) * (
                dem_bounds.top - dem_bounds.bottom
            )

            print(f"\n📐 DEM Coverage:")
            print(f"   Area: {dem_area:.1f} m²")
            print(f"   East span: {dem_bounds.right - dem_bounds.left:.1f} m")
            print(f"   North span: {dem_bounds.top - dem_bounds.bottom:.1f} m")

    except Exception as e:
        print(f"❌ Could not analyze DEM: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Default path - UPDATE THIS
        data_dir = r"E:\mjosa_new_oct_2025\use_gref4hsi\057"

    print(f"Analyzing: {data_dir}\n")
    print("=" * 80)
    print("DEM COVERAGE VISUALIZATION")
    print("Note: The DEM shown is MERGED from all H5 files' altimeter data")
    print("=" * 80 + "\n")

    plot_dem_and_altimeter_contributions(data_dir)
    analyze_coverage_overlap(data_dir)

    print("\n✅ Visualization complete! Close the plot window to exit.")
