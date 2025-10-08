"""
Main processing script for UHI + MBES georeferencing
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

import config
from x_gref4hsi_by_liu.utils import utils
import georeference_mbes


def process_h5_file(h5_path, nav_data, mbes_mesh, camera_calib, output_dir):
    """
    Process a single UHI H5 file: load HSI data, interpolate navigation, ray trace to MBES mesh.

    Parameters:
    -----------
    h5_path : str
        Path to UHI H5 file
    nav_data : pd.DataFrame
        Navigation data (timestamp, lat, lon, depth, roll, pitch, yaw)
    mbes_mesh : pv.PolyData
        MBES bathymetry mesh
    camera_calib : dict
        Camera calibration parameters
    output_dir : Path
        Output directory for results

    Returns:
    --------
    bool
        Success status
    """
    print(f"\n{'='*80}")
    print(f"Processing: {Path(h5_path).name}")
    print(f"{'='*80}")

    try:
        # Step 1: Load HSI timestamps from H5 file
        hsi_timestamps = utils.load_h5_timestamps(h5_path)
        n_frames = len(hsi_timestamps)
        print(f"\nLoaded {n_frames} HSI frames")
        print(f"Time range: {hsi_timestamps.min():.2f} to {hsi_timestamps.max():.2f}")

        # Step 2: Interpolate navigation to HSI frame times
        print("\nInterpolating navigation to HSI frames...")
        interp_nav = utils.interpolate_navigation(
            nav_data, hsi_timestamps, time_offset=config.TIME_OFFSET_SEC
        )

        # Step 3: Convert geographic navigation to ECEF
        print("\nConverting navigation to ECEF...")
        pos_ecef_x, pos_ecef_y, pos_ecef_z = utils.geographic_to_ecef(
            interp_nav["longitude"],
            interp_nav["latitude"],
            -interp_nav["depth"],  # Depth is negative below water surface
            epsg_geo=config.EPSG_GEOGRAPHIC,
            epsg_ecef=config.EPSG_ECEF,
        )
        positions_ecef = np.column_stack([pos_ecef_x, pos_ecef_y, pos_ecef_z])

        # Step 4: Convert Euler/heading to BODY->WORLD rotation matrices (ENU)
        print("Converting attitude to rotation matrices...")
        orientations = utils.euler_to_rotation_matrix_nav(
            interp_nav["roll"],
            interp_nav["pitch"],
            interp_nav["yaw"],
            yaw_convention=getattr(config, "YAW_CONVENTION", "heading_from_north_cw"),
        )

        # Step 5: Apply sensor transform (body frame → HSI camera frame)
        print("Applying sensor transform (body → HSI)...")
        hsi_positions_ecef, hsi_orientations = utils.apply_sensor_transform(
            positions_ecef,
            orientations,
            config.ROTATION_HSI_TO_BODY,
            config.TRANSLATION_BODY_TO_HSI,
        )

        # Step 6: Convert HSI positions from ECEF to MBES CRS (usually UTM)
        print(
            f"Converting HSI positions from ECEF (EPSG:{config.EPSG_ECEF}) to MBES CRS (EPSG:{config.EPSG_MBES})..."
        )
        hsi_pos_mbes_x, hsi_pos_mbes_y, hsi_pos_mbes_z = utils.ecef_to_utm(
            hsi_positions_ecef[:, 0],
            hsi_positions_ecef[:, 1],
            hsi_positions_ecef[:, 2],
            epsg_utm=config.EPSG_MBES,
            epsg_ecef=config.EPSG_ECEF,
        )
        hsi_positions_mbes = np.column_stack(
            [hsi_pos_mbes_x, hsi_pos_mbes_y, hsi_pos_mbes_z]
        )

        # Step 7: Build ray directions in camera frame
        print("\nBuilding ray directions for HSI camera...")
        with h5py.File(h5_path, "r") as h5f:
            # Get number of pixels (slits) from H5 file
            # Try common dataset paths (UHI format: processed/radiance/dataCube)
            possible_paths = [
                "processed/radiance/dataCube",  # Standard UHI calibrated radiance
            ]

            found_data = False
            for path in possible_paths:
                if path in h5f:
                    obj = h5f[path]
                    # Check if it's a dataset (not a group)
                    if isinstance(obj, h5py.Dataset):
                        data_shape = obj.shape
                        n_slits = data_shape[
                            1
                        ]  # Assuming shape is (frames, slits, bands)
                        print(f"Found radiance data: {path}")
                        print(f"  Shape: {data_shape} (frames × slits × bands)")
                        print(f"  Using n_slits = {n_slits}")
                        found_data = True
                        break

            if not found_data:
                # Default to camera width from calibration
                n_slits = int(camera_calib["w"])
                print(
                    f"WARNING: Could not find radiance datacube in H5, using camera width: {n_slits}"
                )

        ray_directions_camera = utils.build_ray_directions(camera_calib, n_slits)

        # Step 8: Transform ray directions from camera frame to world frame
        print("Transforming ray directions to world frame...")
        n_rays_per_frame = ray_directions_camera.shape[0]
        total_rays = n_frames * n_rays_per_frame

        ray_origins_mbes = np.zeros((total_rays, 3), dtype=float)
        ray_directions_world = np.zeros((total_rays, 3), dtype=float)

        for i in range(n_frames):
            idx_start = i * n_rays_per_frame
            idx_end = (i + 1) * n_rays_per_frame

            # ray origins: camera positions (already in MBES CRS)
            ray_origins_mbes[idx_start:idx_end] = hsi_positions_mbes[i]

            # orientations: R_world_from_body @ R_body_from_cam  => R_world_from_cam
            R_world_from_cam = hsi_orientations[
                i
            ]  # already computed as R_world_from_body @ ROTATION_HSI_TO_BODY

            dirs_world = (R_world_from_cam @ ray_directions_camera.T).T
            dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)  # normalize

            ray_directions_world[idx_start:idx_end] = dirs_world

        print(
            f"Prepared {total_rays:,} rays ({n_frames} frames × {n_rays_per_frame} slits)"
        )

        # Sanity check: print mean boresight direction (should point down, negative Z)
        up = np.array([0.0, 0.0, 1.0])
        mean_boresight_dot_up = float(np.mean(ray_directions_world @ up))
        print(
            f"Mean boresight·up = {mean_boresight_dot_up:.4f} (should be negative, pointing down)"
        )

        # Step 9: Ray trace to MBES mesh
        print("\n" + "=" * 80)
        print("RAY TRACING TO MBES MESH")
        print("=" * 80)

        results = georeference_mbes.raytrace_hsi_to_mbes(
            ray_origins_mbes,
            ray_directions_world,
            mbes_mesh,
            max_ray_length=config.MAX_RAY_LENGTH,
            early_failure_threshold=config.EARLY_FAILURE_THRESHOLD,
            max_retry_rays=10000,
        )

        # Step 10: Convert intersection points back to ECEF for saving
        print("\nConverting intersection points back to ECEF...")
        intersect_ecef_x, intersect_ecef_y, intersect_ecef_z = utils.utm_to_ecef(
            results["points"][:, 0],
            results["points"][:, 1],
            results["points"][:, 2],
            epsg_utm=config.EPSG_MBES,
            epsg_ecef=config.EPSG_ECEF,
        )
        intersections_ecef = np.column_stack(
            [intersect_ecef_x, intersect_ecef_y, intersect_ecef_z]
        )

        # Step 11: Compute frame indices from ray indices
        frame_indices = results["ray_indices"] // n_rays_per_frame
        pixel_indices = results["ray_indices"] % n_rays_per_frame

        # Step 12: Save results to H5 file
        print("\nSaving results to H5 file...")
        output_h5_path = output_dir / Path(h5_path).name

        # Copy original H5 file to output directory if not already there
        if not output_h5_path.exists():
            import shutil

            shutil.copy2(h5_path, output_h5_path)
            print(f"Copied H5 file to output directory")

        utils.save_intersection_to_h5(
            str(output_h5_path),
            intersections_ecef,
            pixel_indices,
            frame_indices,
            n_frames=n_frames,
            n_slits=n_slits,
            gridded=True,  # Save as (T, S, 3) gridded format like test_eely
        )

        # Print summary statistics
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Input file: {Path(h5_path).name}")
        print(f"Total HSI frames: {n_frames:,}")
        print(f"Total rays: {total_rays:,}")
        print(f"Successful intersections: {len(intersections_ecef):,}")
        print(f"Success rate: {results['success_rate']:.2f}%")
        print(f"Output file: {output_h5_path}")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ ERROR processing {Path(h5_path).name}:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Main workflow:
    1. Load navigation CSV
    2. Load MBES GeoTIFF and convert to mesh
    3. Load camera calibration
    4. Process each H5 file: interpolate nav, ray trace, save results
    """
    print("\n" + "=" * 80)
    print("UHI + MBES GEOREFERENCING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Step 1: Load navigation CSV
    print("=" * 80)
    print("STEP 1: Loading Navigation CSV")
    print("=" * 80)
    nav_data = utils.load_csv_navigation(config.NAV_CSV, config.CSV_COLUMNS)

    # Step 2: Load MBES GeoTIFF and convert to mesh
    print("\n" + "=" * 80)
    print("STEP 2: Loading MBES GeoTIFF")
    print("=" * 80)
    elevation_data, transform, crs_epsg = georeference_mbes.load_mbes_geotiff(
        config.MBES_GEOTIFF, verbose=config.VERBOSE
    )

    # Update EPSG_MBES in case it wasn't set correctly in config
    if crs_epsg != config.EPSG_MBES:
        print(
            f"⚠️  WARNING: GeoTIFF CRS (EPSG:{crs_epsg}) differs from config (EPSG:{config.EPSG_MBES})"
        )
        print(f"   Using GeoTIFF CRS: EPSG:{crs_epsg}")
        config.EPSG_MBES = crs_epsg

    print("\n" + "=" * 80)
    print("STEP 3: Converting GeoTIFF to Mesh")
    print("=" * 80)
    mbes_mesh = georeference_mbes.geotiff_to_mesh(
        elevation_data,
        transform,
        simplify=config.MESH_SIMPLIFICATION,
        reduction_factor=config.MESH_REDUCTION_FACTOR,
    )

    # Optionally save mesh to PLY
    if config.SAVE_MESH_PLY:
        output_dir = Path(config.OUTPUT_FOLDER)
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_ply_path = output_dir / "mbes_mesh.ply"
        georeference_mbes.save_mesh_ply(mbes_mesh, str(mesh_ply_path))

    # Step 3: Load camera calibration
    print("\n" + "=" * 80)
    print("STEP 4: Loading Camera Calibration")
    print("=" * 80)
    camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
    print(
        f"Camera: f={camera_calib['f']:.2f}, cx={camera_calib['cx']:.2f}, w={camera_calib['w']:.0f}"
    )

    # Step 4: Find all H5 files in input directory
    print("\n" + "=" * 80)
    print("STEP 5: Finding H5 Files")
    print("=" * 80)
    h5_folder = Path(config.H5_FOLDER)
    h5_files = sorted(h5_folder.glob("*.h5"))

    if len(h5_files) == 0:
        print(f"❌ ERROR: No H5 files found in {h5_folder}")
        return

    print(f"Found {len(h5_files)} H5 files:")
    for i, h5_file in enumerate(h5_files, 1):
        print(f"  {i}. {h5_file.name}")

    # Step 5: Process each H5 file
    print("\n" + "=" * 80)
    print("STEP 6: Processing H5 Files")
    print("=" * 80)

    output_dir = Path(config.OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for i, h5_file in enumerate(h5_files, 1):
        print(f"\n[File {i}/{len(h5_files)}]")

        success = process_h5_file(
            str(h5_file), nav_data, mbes_mesh, camera_calib, output_dir
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(h5_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output directory: {output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
