"""
MBES GeoTIFF loading and ray tracing for HSI georeferencing
"""

import numpy as np
import rasterio
import pyvista as pv
import trimesh
from tqdm import tqdm
from pathlib import Path


def load_mbes_geotiff(geotiff_path, verbose=True):
    """
    Load MBES bathymetry data from GeoTIFF.

    Parameters:
    -----------
    geotiff_path : str
        Path to MBES GeoTIFF file
    verbose : bool
        Print GeoTIFF metadata

    Returns:
    --------
    tuple
        (elevation_data, transform, crs_epsg)
        - elevation_data: 2D array of elevation values (meters)
        - transform: Affine transform for georeferencing
        - crs_epsg: EPSG code of the GeoTIFF CRS
    """
    with rasterio.open(geotiff_path) as src:
        elevation_data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs

        if verbose:
            print(f"\n=== MBES GeoTIFF Metadata ===")
            print(f"File: {geotiff_path}")
            print(f"Shape: {elevation_data.shape} (rows × cols)")
            print(f"CRS: {crs}")
            print(f"EPSG: {crs.to_epsg()}")
            print(f"Transform: {transform}")
            print(f"Bounds: {src.bounds}")
            print(
                f"Elevation range: {np.nanmin(elevation_data):.2f} to {np.nanmax(elevation_data):.2f} m"
            )
            print(f"NoData value: {src.nodata}")
            print(f"Resolution: {transform.a:.4f} × {-transform.e:.4f} m\n")

        # Handle NoData values
        if src.nodata is not None:
            elevation_data = np.where(
                elevation_data == src.nodata, np.nan, elevation_data
            )

        return elevation_data, transform, crs.to_epsg()


def geotiff_to_mesh(elevation_data, transform, simplify=False, reduction_factor=0.5):
    """
    Convert GeoTIFF elevation grid to PyVista mesh.

    Parameters:
    -----------
    elevation_data : ndarray (H, W)
        2D array of elevation values
    transform : Affine
        Rasterio affine transform for georeferencing
    simplify : bool
        Whether to simplify mesh by reducing face count
    reduction_factor : float
        Target reduction factor (0.5 = keep 50% of faces)

    Returns:
    --------
    pv.PolyData
        PyVista mesh representation of bathymetry
    """
    rows, cols = elevation_data.shape
    print(f"Converting GeoTIFF ({rows} × {cols} = {rows*cols:,} pixels) to mesh...")

    # Create grid of (x, y) coordinates from transform (VECTORIZED - much faster!)
    print("  [1/3] Building coordinate grids...")

    # Create row and column indices
    row_indices, col_indices = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing="ij"
    )

    # Apply affine transform vectorized
    # Transform formula: x = transform.c + col * transform.a + row * transform.b
    #                    y = transform.f + col * transform.d + row * transform.e
    x_coords = transform.c + col_indices * transform.a + row_indices * transform.b
    y_coords = transform.f + col_indices * transform.d + row_indices * transform.e

    # Offset to pixel center (rasterio xy() uses center by default)
    x_coords += transform.a / 2
    y_coords += transform.e / 2

    # Create mesh vertices (X, Y, Z)
    print("  [2/3] Creating structured grid...")
    valid_mask = ~np.isnan(elevation_data)
    n_valid = np.sum(valid_mask)
    print(
        f"    Valid points: {n_valid:,} / {rows*cols:,} ({n_valid/(rows*cols)*100:.1f}%)"
    )

    # Fill NaN for mesh creation (PyVista needs complete grid)
    grid_x = x_coords
    grid_y = y_coords
    grid_z = np.where(valid_mask, elevation_data, np.nanmean(elevation_data))

    # Create structured surface
    mesh = pv.StructuredGrid(grid_x, grid_y, grid_z)

    # Extract surface (converts to PolyData)
    print("  [3/3] Extracting surface mesh...")
    surface = mesh.extract_surface()

    if simplify:
        print(f"Simplifying mesh (target reduction: {reduction_factor*100:.0f}%)...")
        original_faces = surface.n_cells  # Use n_cells instead of deprecated n_faces
        surface = surface.decimate(reduction_factor)
        print(
            f"Reduced from {original_faces} to {surface.n_cells} faces ({surface.n_cells/original_faces*100:.1f}%)"
        )

    print(f"Mesh created: {surface.n_points} vertices, {surface.n_cells} faces")
    print(
        f"Mesh bounds: X=[{surface.bounds[0]:.2f}, {surface.bounds[1]:.2f}], "
        f"Y=[{surface.bounds[2]:.2f}, {surface.bounds[3]:.2f}], "
        f"Z=[{surface.bounds[4]:.2f}, {surface.bounds[5]:.2f}]"
    )

    return surface


def raytrace_hsi_to_mbes(
    ray_origins,
    ray_directions,
    mesh,
    max_ray_length=100,
    early_failure_threshold=50.0,
    max_retry_rays=10000,
):
    """
    Ray trace from HSI camera to MBES mesh.

    Parameters:
    -----------
    ray_origins : ndarray (N, 3)
        Ray origin positions in ECEF or UTM coordinates (matching mesh CRS)
    ray_directions : ndarray (N, 3)
        Ray direction vectors (normalized)
    mesh : pv.PolyData
        MBES bathymetry mesh
    max_ray_length : float
        Maximum ray length for intersection (meters)
    early_failure_threshold : float
        Fail early if this percentage of rays miss (default: 50%)
    max_retry_rays : int
        Maximum number of rays to retry with PyVista (default: 10000)

    Returns:
    --------
    dict
        Results dictionary with keys:
        - 'points': ndarray (M, 3) - intersection points
        - 'ray_indices': ndarray (M,) - indices of successful rays
        - 'cell_indices': ndarray (M,) - mesh cell indices at intersections
        - 'success_rate': float - percentage of successful intersections
    """
    n_rays = ray_origins.shape[0]

    # Scale ray directions by max length
    ray_ends = ray_origins + ray_directions * max_ray_length

    print(f"\n=== Ray Tracing ===")
    print(f"Total rays: {n_rays:,}")
    print(f"Max ray length: {max_ray_length:.1f} m")

    # Convert PyVista mesh to Trimesh for fast intersection
    faces = mesh.regular_faces
    tri_mesh = trimesh.Trimesh(vertices=mesh.points, faces=faces)
    ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
        geometry=tri_mesh
    )

    # Perform bulk intersection with Trimesh
    print("Running Trimesh ray intersection...")
    cells, rays, points = ray_mesh_intersector.intersects_id(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,
        return_locations=True,
    )

    n_hits = len(points)
    n_missing = n_rays - n_hits
    success_rate = n_hits / n_rays * 100

    print(f"Trimesh results: {n_hits:,}/{n_rays:,} rays hit ({success_rate:.2f}%)")

    if n_missing == 0:
        print("✓ All rays intersected successfully!")
        return {
            "points": points,
            "ray_indices": rays,
            "cell_indices": cells,
            "success_rate": success_rate,
        }

    # Check early failure condition
    failure_rate = n_missing / n_rays * 100
    if failure_rate > early_failure_threshold:
        print(f"\n❌ CRITICAL FAILURE: {failure_rate:.1f}% of rays missed the mesh")
        print(f"   Threshold: {early_failure_threshold}%")
        print(f"   This indicates MBES mesh doesn't cover HSI field of view.")
        print(f"   Possible causes:")
        print(f"     - HSI looking outside MBES coverage area")
        print(f"     - Coordinate system mismatch (ECEF vs UTM)")
        print(f"     - Navigation interpolation error")
        raise ValueError(
            f"Ray tracing failed: {failure_rate:.1f}% of rays missed mesh "
            f"(threshold: {early_failure_threshold}%). Cannot proceed."
        )

    # Retry missing rays with PyVista
    print(f"\n{n_missing} rays missed - attempting PyVista retry...")

    missing_rays = np.array(list(set(range(n_rays)) - set(rays)))

    if len(missing_rays) > max_retry_rays:
        print(
            f"⚠️  WARNING: {len(missing_rays)} missing rays exceeds retry limit ({max_retry_rays})"
        )
        print(
            f"   Skipping retry. Continuing with {n_hits:,} successful intersections."
        )
        failed_rays_list = list(missing_rays)
    else:
        successful_retries = 0
        failed_rays_list = []

        for ray_idx in tqdm(missing_rays, desc="PyVista retry", unit="rays"):
            # Retry with PyVista's single ray trace
            point, cell = mesh.ray_trace(
                ray_origins[ray_idx], ray_ends[ray_idx], first_point=True
            )

            if point.size > 0 and cell.size > 0:
                # Success - append to results
                cells = np.append(cells, cell)
                points = np.vstack([points, point.reshape(1, 3)])
                rays = np.append(rays, ray_idx)
                successful_retries += 1
            else:
                failed_rays_list.append(ray_idx)

        print(f"PyVista retry: {successful_retries}/{len(missing_rays)} rays recovered")

    # Final statistics
    n_final_hits = len(points)
    final_success_rate = n_final_hits / n_rays * 100
    n_failed = len(failed_rays_list)

    print(f"\n=== Final Results ===")
    print(
        f"Successful intersections: {n_final_hits:,}/{n_rays:,} ({final_success_rate:.2f}%)"
    )
    print(f"Failed rays: {n_failed:,} ({n_failed/n_rays*100:.2f}%)")

    if n_failed > 0:
        # Check failure threshold
        FAILURE_THRESHOLD = 30.0
        if (n_failed / n_rays * 100) > FAILURE_THRESHOLD:
            raise ValueError(
                f"Ray tracing failed: {n_failed/n_rays*100:.1f}% of rays missed mesh "
                f"(threshold: {FAILURE_THRESHOLD}%). Too many missing intersections."
            )
        else:
            print(
                f"⚠️  Warning: {n_failed} rays failed, but within acceptable threshold ({FAILURE_THRESHOLD}%)"
            )

    return {
        "points": points,
        "ray_indices": rays,
        "cell_indices": cells,
        "success_rate": final_success_rate,
    }


def save_mesh_ply(mesh, output_path):
    """
    Save PyVista mesh to PLY file for visualization.

    Parameters:
    -----------
    mesh : pv.PolyData
        Mesh to save
    output_path : str
        Output PLY file path
    """
    mesh.save(output_path)
    print(f"Saved mesh to {output_path}")
