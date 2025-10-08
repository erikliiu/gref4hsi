"""
Visualize model.ply mesh vertices and HSI ray intersection locations from above (2D plot).
This helps diagnose why some files georeference successfully while others fail.

CRITICAL: We plot the ACTUAL mesh vertices from model.ply that ray tracing uses!
Not the rasterized dummy_dem.tif, but the real 3D point cloud that forms the mesh.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import sys
import pandas as pd
import pyvista as pv
from pyproj import Transformer


def plot_dem_and_altimeter_contributions(data_dir):
    """
    Create a 2D plot showing:
    1. Actual model.ply mesh vertices (the REAL ray tracing target)
    2. Ray intersection points from each H5 file
    3. Spatial overlap analysis
    """
    data_dir = Path(data_dir)

    # Paths (use lowercase to match actual directory structure)
    mesh_path = data_dir / "input" / "GIS" / "model.ply"
    h5_dir = data_dir / "input" / "H5"

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ==================== Plot 1: Mesh Vertices ====================
    ax1 = axes[0]

    # Load and plot the actual mesh vertices from model.ply
    try:
        print("Loading model.ply mesh...")
        mesh = pv.read(mesh_path)
        
        # Load the offset from model_meta.json
        import json
        meta_path = data_dir / "input" / "GIS" / "model_meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        offset_x = meta['offset_x']
        offset_y = meta['offset_y']
        offset_z = meta['offset_z']
        epsg_code = meta['epsg_code']
        
        print(f"✅ Mesh and metadata loaded successfully")
        print(f"   Number of vertices: {len(mesh.points):,}")
        print(f"   Number of faces: {mesh.n_faces:,}")
        print(f"   Mesh coordinate system: EPSG:{epsg_code}")
        print(f"   Mesh offset (ECEF): X={offset_x:.1f}, Y={offset_y:.1f}, Z={offset_z:.1f} m")
        
        # Get local vertices and add offset to get global ECEF coordinates
        local_vertices = mesh.points
        ecef_x = local_vertices[:, 0] + offset_x
        ecef_y = local_vertices[:, 1] + offset_y
        ecef_z = local_vertices[:, 2] + offset_z
        
        print(f"   Local vertex range: X=[{local_vertices[:, 0].min():.1f}, {local_vertices[:, 0].max():.1f}] m")
        print(f"   Global ECEF range: X=[{ecef_x.min():.1f}, {ecef_x.max():.1f}] m")
        
        # Convert from ECEF (EPSG:4978) to UTM Zone 32N (EPSG:32632)
        transformer = Transformer.from_crs("EPSG:4978", "EPSG:32632", always_xy=True)
        east, north, elevation = transformer.transform(ecef_x, ecef_y, ecef_z)
        
        print(f"   UTM coordinates:")
        print(f"   East range:  [{east.min():.1f}, {east.max():.1f}] m")
        print(f"   North range: [{north.min():.1f}, {north.max():.1f}] m")
        print(f"   Elevation range: [{elevation.min():.1f}, {elevation.max():.1f}] m")
        
        # Plot mesh vertices as scatter with elevation coloring
        scatter = ax1.scatter(
            east,
            north,
            c=elevation,
            cmap="terrain",
            s=1,
            alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax1, label="Elevation (m)")

        ax1.set_title("model.ply Mesh Vertices (Ray Tracing Target)", fontsize=14, fontweight="bold")
        ax1.set_xlabel("East (m)", fontsize=12)
        ax1.set_ylabel("North (m)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

    except Exception as e:
        print(f"❌ Error loading mesh: {e}")
        ax1.text(
            0.5,
            0.5,
            f"model.ply not found\n{mesh_path}",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    # ==================== Plot 2: Ray Intersections ====================
    ax2 = axes[1]

    # Plot mesh boundary on second plot too (for reference)
    try:
        mesh = pv.read(mesh_path)
        
        # Load offset and convert to global coordinates
        import json
        meta_path = data_dir / "input" / "GIS" / "model_meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        local_vertices = mesh.points
        ecef_x = local_vertices[:, 0] + meta['offset_x']
        ecef_y = local_vertices[:, 1] + meta['offset_y']
        ecef_z = local_vertices[:, 2] + meta['offset_z']
        
        # Convert to UTM
        transformer = Transformer.from_crs("EPSG:4978", "EPSG:32632", always_xy=True)
        east, north, elevation = transformer.transform(ecef_x, ecef_y, ecef_z)
        
        # Draw bounding box around mesh vertices
        from matplotlib.patches import Rectangle
        bbox = Rectangle(
            (east.min(), north.min()),
            east.max() - east.min(),
            north.max() - north.min(),
            fill=False,
            edgecolor="black",
            linewidth=2,
            linestyle="--",
            label="Mesh Boundary",
        )
        ax2.add_patch(bbox)
        
        # Also plot the mesh vertices in light gray as background
        ax2.scatter(
            east,
            north,
            c='lightgray',
            s=0.5,
            alpha=0.3,
            label="Mesh vertices",
        )
    except:
        pass

    # Load intersection points from georeferenced H5 files
    colors = ["blue", "red", "green", "orange", "purple"]
    h5_files = sorted(h5_dir.glob("*.h5"))

    print(f"\n📍 Loading intersection points from {len(h5_files)} H5 files:")

    for i, h5_file in enumerate(h5_files):
        try:
            with h5py.File(h5_file, "r") as f:
                # Load georeferenced intersection points in ECEF coordinates
                # These are saved by georeference.py after successful ray tracing
                ecef_path = "processed/georef/points_ecef_crs"
                
                if ecef_path not in f:
                    print(f"   ⚠️ File {i+1}: No intersection data found (not georeferenced yet)")
                    continue
                
                points_ecef = f[ecef_path][:]
                
                if len(points_ecef) == 0:
                    print(f"   ⚠️ File {i+1}: Empty intersection data")
                    continue
                
                # IMPORTANT: points_ecef is shaped (tracks, slits, 3), not (n_points, 3)!
                # We need to extract the X, Y, Z coordinates properly
                print(f"   File {i+1}: ECEF data shape = {points_ecef.shape}")
                
                # Extract ECEF coordinates from the 3D grid
                X_ecef = points_ecef[:, :, 0]  # All X coordinates
                Y_ecef = points_ecef[:, :, 1]  # All Y coordinates
                Z_ecef = points_ecef[:, :, 2]  # All Z coordinates
                
                # Flatten to 1D arrays for transformation
                X_flat = X_ecef.flatten()
                Y_flat = Y_ecef.flatten()
                Z_flat = Z_ecef.flatten()
                
                # Convert ECEF (EPSG:4978) to UTM Zone 32N (EPSG:32632)
                # Direct transformation from ECEF to UTM Zone 32N
                transformer = Transformer.from_crs("EPSG:4978", "EPSG:32632", always_xy=True)
                east, north, alt = transformer.transform(X_flat, Y_flat, Z_flat)
                
                # Plot the intersection points
                ax2.scatter(
                    east,
                    north,
                    c=colors[i % len(colors)],
                    s=1,
                    alpha=0.5,
                    label=f"File {i+1}: {h5_file.name}",
                )

                n_points = len(east)
                print(f"   ✅ File {i+1}: {n_points} intersection points")
                print(f"      North: [{north.min():.1f}, {north.max():.1f}] m (UTM)")
                print(f"      East:  [{east.min():.1f}, {east.max():.1f}] m (UTM)")
                print(f"      ECEF X range: [{X_flat.min():.1f}, {X_flat.max():.1f}] m")
                print(f"      ECEF Y range: [{Y_flat.min():.1f}, {Y_flat.max():.1f}] m")

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
    Analyze spatial overlap between mesh and HSI coverage areas
    """
    data_dir = Path(data_dir)
    mesh_path = data_dir / "input" / "GIS" / "model.ply"

    print("\n" + "=" * 60)
    print("SPATIAL COVERAGE ANALYSIS")
    print("=" * 60)

    try:
        mesh = pv.read(mesh_path)
        
        # Load offset and convert to global coordinates
        import json
        meta_path = data_dir / "input" / "GIS" / "model_meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        local_vertices = mesh.points
        ecef_x = local_vertices[:, 0] + meta['offset_x']
        ecef_y = local_vertices[:, 1] + meta['offset_y']
        ecef_z = local_vertices[:, 2] + meta['offset_z']
        
        # Convert to UTM
        transformer = Transformer.from_crs("EPSG:4978", "EPSG:32632", always_xy=True)
        east, north, elevation = transformer.transform(ecef_x, ecef_y, ecef_z)
        
        east_span = east.max() - east.min()
        north_span = north.max() - north.min()
        mesh_area = east_span * north_span

        print(f"\n📐 Mesh Coverage:")
        print(f"   Bounding box area: {mesh_area:.1f} m²")
        print(f"   East span: {east_span:.1f} m")
        print(f"   North span: {north_span:.1f} m")
        print(f"   Number of vertices: {len(local_vertices):,}")

    except Exception as e:
        print(f"❌ Could not analyze mesh: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Default path - UPDATE THIS
        data_dir = r"E:\mjosa_new_oct_2025\use_gref4hsi\057_all"

    print(f"Analyzing: {data_dir}\n")
    print("=" * 80)
    print("DEM COVERAGE VISUALIZATION")
    print("Note: The DEM shown is MERGED from all H5 files' altimeter data")
    print("=" * 80 + "\n")

    plot_dem_and_altimeter_contributions(data_dir)
    analyze_coverage_overlap(data_dir)

    print("\n✅ Visualization complete! Close the plot window to exit.")
