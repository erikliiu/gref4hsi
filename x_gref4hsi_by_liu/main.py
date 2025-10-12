"""
Main processing script for UHI + MBES georeferencing
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
import json

import config
from utils import utils
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
    dict or None
        Statistics dictionary if successful, None if failed
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
        n_bands = None  # Initialize for stats collection
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
                        n_bands = data_shape[2] if len(data_shape) > 2 else None
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
            max_retry_rays=200000,
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

        # ===== COMPUTE ADDITIONAL COVERAGE METRICS =====
        # Load gridded georeferenced data back from H5 for statistics
        with h5py.File(str(output_h5_path), "r") as h5f:
            georef_grid = h5f["processed/georef/points_ecef_crs"][
                :
            ]  # (n_frames, n_slits, 3)

        # Calculate path distance and speed
        positions_utm = np.column_stack(
            [
                hsi_positions_mbes[:, 0],  # X (Easting)
                hsi_positions_mbes[:, 1],  # Y (Northing)
                hsi_positions_mbes[:, 2],  # Z (Elevation)
            ]
        )

        # Path distance (3D)
        if n_frames > 1:
            position_diffs = np.diff(positions_utm, axis=0)
            segment_distances = np.sqrt(np.sum(position_diffs**2, axis=1))
            total_distance_m = float(np.sum(segment_distances))

            # Time differences
            time_diffs = np.diff(hsi_timestamps)
            total_duration_sec = float(hsi_timestamps.max() - hsi_timestamps.min())

            # Speed statistics (only for segments with reasonable time diff)
            valid_speeds = (
                segment_distances[time_diffs > 0.01] / time_diffs[time_diffs > 0.01]
            )
            avg_speed_ms = (
                float(np.mean(valid_speeds)) if len(valid_speeds) > 0 else 0.0
            )
            max_speed_ms = float(np.max(valid_speeds)) if len(valid_speeds) > 0 else 0.0
            min_speed_ms = float(np.min(valid_speeds)) if len(valid_speeds) > 0 else 0.0
        else:
            total_distance_m = 0.0
            total_duration_sec = 0.0
            avg_speed_ms = 0.0
            max_speed_ms = 0.0
            min_speed_ms = 0.0

        # Calculate swath width statistics (per frame)
        swath_widths = []
        swath_areas = []

        for frame_idx in range(n_frames):
            frame_points = georef_grid[frame_idx, :, :]
            valid_mask = np.isfinite(frame_points[:, 0])

            if valid_mask.sum() >= 2:
                valid_points = frame_points[valid_mask, :]
                # Convert to UTM for metric calculations
                valid_x_utm, valid_y_utm, valid_z_utm = utils.ecef_to_utm(
                    valid_points[:, 0],
                    valid_points[:, 1],
                    valid_points[:, 2],
                    epsg_utm=config.EPSG_MBES,
                    epsg_ecef=config.EPSG_ECEF,
                )

                # Swath width (cross-track extent)
                if len(valid_x_utm) >= 2:
                    # Width is max distance between any two points in swath
                    x_range = float(valid_x_utm.max() - valid_x_utm.min())
                    y_range = float(valid_y_utm.max() - valid_y_utm.min())
                    width = float(np.sqrt(x_range**2 + y_range**2))
                    swath_widths.append(width)

                    # Approximate swath area (width × along-track distance to next frame)
                    if frame_idx < n_frames - 1:
                        along_track = (
                            segment_distances[frame_idx]
                            if frame_idx < len(segment_distances)
                            else 0
                        )
                        swath_areas.append(width * along_track)

        # Coverage area statistics
        if len(swath_widths) > 0:
            mean_swath_width_m = float(np.mean(swath_widths))
            min_swath_width_m = float(np.min(swath_widths))
            max_swath_width_m = float(np.max(swath_widths))
            std_swath_width_m = float(np.std(swath_widths))
        else:
            mean_swath_width_m = 0.0
            min_swath_width_m = 0.0
            max_swath_width_m = 0.0
            std_swath_width_m = 0.0

        if len(swath_areas) > 0:
            total_coverage_area_m2 = float(np.sum(swath_areas))
        else:
            total_coverage_area_m2 = 0.0

        # Operational area bounds (bounding box of survey)
        if n_frames > 1:
            survey_x_min = float(positions_utm[:, 0].min())
            survey_x_max = float(positions_utm[:, 0].max())
            survey_y_min = float(positions_utm[:, 1].min())
            survey_y_max = float(positions_utm[:, 1].max())
            survey_extent_x = survey_x_max - survey_x_min
            survey_extent_y = survey_y_max - survey_y_min
            survey_bounding_area_m2 = float(survey_extent_x * survey_extent_y)

            # Coverage efficiency (covered area / bounding box area)
            coverage_efficiency_pct = (
                (total_coverage_area_m2 / survey_bounding_area_m2 * 100)
                if survey_bounding_area_m2 > 0
                else 0.0
            )
        else:
            survey_x_min = survey_x_max = survey_y_min = survey_y_max = 0.0
            survey_extent_x = survey_extent_y = survey_bounding_area_m2 = 0.0
            coverage_efficiency_pct = 0.0

        # Collect comprehensive statistics
        file_stats = {
            "filename": Path(h5_path).name,
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hsi_data": {
                "n_frames": int(n_frames),
                "n_slits": int(n_slits),
                "n_bands": int(n_bands) if n_bands is not None else None,
                "time_range": {
                    "start": float(hsi_timestamps.min()),
                    "end": float(hsi_timestamps.max()),
                    "duration_sec": float(hsi_timestamps.max() - hsi_timestamps.min()),
                },
            },
            "navigation": {
                "interpolated_positions": int(n_frames),
                "lat_range": {
                    "min": float(interp_nav["latitude"].min()),
                    "max": float(interp_nav["latitude"].max()),
                },
                "lon_range": {
                    "min": float(interp_nav["longitude"].min()),
                    "max": float(interp_nav["longitude"].max()),
                },
                "depth_range": {
                    "min": float(interp_nav["depth"].min()),
                    "max": float(interp_nav["depth"].max()),
                    "mean": float(interp_nav["depth"].mean()),
                    "std": float(interp_nav["depth"].std()),
                },
                "attitude_ranges": {
                    "roll": {
                        "min": float(interp_nav["roll"].min()),
                        "max": float(interp_nav["roll"].max()),
                        "mean": float(interp_nav["roll"].mean()),
                        "std": float(interp_nav["roll"].std()),
                    },
                    "pitch": {
                        "min": float(interp_nav["pitch"].min()),
                        "max": float(interp_nav["pitch"].max()),
                        "mean": float(interp_nav["pitch"].mean()),
                        "std": float(interp_nav["pitch"].std()),
                    },
                    "yaw": {
                        "min": float(interp_nav["yaw"].min()),
                        "max": float(interp_nav["yaw"].max()),
                        "mean": float(interp_nav["yaw"].mean()),
                        "std": float(interp_nav["yaw"].std()),
                    },
                },
            },
            "mission_metrics": {
                "total_distance_m": total_distance_m,
                "survey_duration_sec": total_duration_sec,
                "avg_speed_ms": avg_speed_ms,
                "avg_speed_kmh": avg_speed_ms * 3.6,
                "max_speed_ms": max_speed_ms,
                "min_speed_ms": min_speed_ms,
            },
            "coverage_metrics": {
                "swath_width": {
                    "mean_m": mean_swath_width_m,
                    "min_m": min_swath_width_m,
                    "max_m": max_swath_width_m,
                    "std_m": std_swath_width_m,
                },
                "survey_bounds_utm": {
                    "x_min": survey_x_min,
                    "x_max": survey_x_max,
                    "y_min": survey_y_min,
                    "y_max": survey_y_max,
                    "extent_x_m": survey_extent_x,
                    "extent_y_m": survey_extent_y,
                },
                "coverage_area": {
                    "total_covered_m2": total_coverage_area_m2,
                    "bounding_box_m2": survey_bounding_area_m2,
                    "efficiency_pct": float(coverage_efficiency_pct),
                },
                "frames_with_valid_data": len(swath_widths),
            },
            "ray_tracing": {
                "total_rays": int(total_rays),
                "successful_hits": int(len(intersections_ecef)),
                "failed_rays": int(total_rays - len(intersections_ecef)),
                "success_rate_pct": float(results["success_rate"]),
                "trimesh_hits": int(results.get("trimesh_hits", 0)),
                "pyvista_retries": int(results.get("pyvista_retries", 0)),
                "pyvista_recoveries": int(results.get("pyvista_recoveries", 0)),
                "max_ray_length_m": float(config.MAX_RAY_LENGTH),
            },
            "georef_coverage": {
                "frames_with_data": int(
                    np.sum(
                        np.any(
                            np.isfinite(georef_grid[:, :, 0]),
                            axis=1,
                        )
                    )
                ),
                "coverage_pct": float(len(intersections_ecef) / total_rays * 100),
            },
            "output": {
                "h5_file": str(output_h5_path),
                "format": "gridded",
                "shape": [int(n_frames), int(n_slits), 3],
            },
        }

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

        return file_stats

    except Exception as e:
        print(f"\n❌ ERROR processing {Path(h5_path).name}:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

        # Return error stats
        return {
            "filename": Path(h5_path).name,
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "failed",
            "error": {"type": type(e).__name__, "message": str(e)},
        }


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
    all_file_stats = []
    start_time = datetime.now()

    for i, h5_file in enumerate(h5_files, 1):
        print(f"\n[File {i}/{len(h5_files)}]")

        file_stats = process_h5_file(
            str(h5_file), nav_data, mbes_mesh, camera_calib, output_dir
        )

        if file_stats is not None:
            all_file_stats.append(file_stats)
            if file_stats.get("status") != "failed":
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1

        # Explicitly trigger garbage collection after each file to free memory
        import gc

        gc.collect()

    end_time = datetime.now()
    processing_duration = (end_time - start_time).total_seconds()

    # Compile comprehensive processing statistics
    summary_stats = {
        "processing_session": {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": processing_duration,
            "duration_human": f"{int(processing_duration//60)}m {int(processing_duration%60)}s",
        },
        "configuration": {
            "nav_csv": str(config.NAV_CSV),
            "mbes_geotiff": str(config.MBES_GEOTIFF),
            "camera_calib_xml": str(config.CAMERA_CALIB_XML),
            "output_folder": str(config.OUTPUT_FOLDER),
            "max_ray_length_m": float(config.MAX_RAY_LENGTH),
            "early_failure_threshold_pct": float(config.EARLY_FAILURE_THRESHOLD),
            "epsg_mbes": int(config.EPSG_MBES),
            "sensor_translation": config.TRANSLATION_BODY_TO_HSI.tolist(),
            "sensor_rotation": config.ROTATION_HSI_TO_BODY.tolist(),
        },
        "navigation_data": {
            "total_records": int(len(nav_data)),
            "time_range": {
                "start": float(nav_data["timestamp"].min()),
                "end": float(nav_data["timestamp"].max()),
                "duration_sec": float(
                    nav_data["timestamp"].max() - nav_data["timestamp"].min()
                ),
            },
        },
        "mbes_mesh": {
            "n_vertices": int(mbes_mesh.n_points),
            "n_faces": int(mbes_mesh.n_cells),
            "bounds": {
                "x_min": float(mbes_mesh.bounds[0]),
                "x_max": float(mbes_mesh.bounds[1]),
                "y_min": float(mbes_mesh.bounds[2]),
                "y_max": float(mbes_mesh.bounds[3]),
                "z_min": float(mbes_mesh.bounds[4]),
                "z_max": float(mbes_mesh.bounds[5]),
            },
        },
        "camera_calibration": {
            "focal_length": float(camera_calib["f"]),
            "principal_point": float(camera_calib["cx"]),
            "width_pixels": float(camera_calib["w"]),
            "distortion": {
                "k1": float(camera_calib["k1"]),
                "k2": float(camera_calib["k2"]),
                "k3": float(camera_calib["k3"]),
            },
        },
        "files_processed": {
            "total": len(h5_files),
            "successful": success_count,
            "failed": fail_count,
            "success_rate_pct": (
                (success_count / len(h5_files) * 100) if len(h5_files) > 0 else 0
            ),
        },
        "aggregated_statistics": {
            "total_files": len(h5_files),
            "successful_files": success_count,
            "failed_files": fail_count,
            "total_frames": sum(
                s.get("hsi_data", {}).get("n_frames", 0)
                for s in all_file_stats
                if s.get("status") != "failed"
            ),
            "total_rays": sum(
                s.get("ray_tracing", {}).get("total_rays", 0)
                for s in all_file_stats
                if s.get("status") != "failed"
            ),
            "total_hits": sum(
                s.get("ray_tracing", {}).get("successful_hits", 0)
                for s in all_file_stats
                if s.get("status") != "failed"
            ),
            "overall_success_rate_pct": (
                (
                    sum(
                        s.get("ray_tracing", {}).get("successful_hits", 0)
                        for s in all_file_stats
                        if s.get("status") != "failed"
                    )
                    / sum(
                        s.get("ray_tracing", {}).get("total_rays", 1)
                        for s in all_file_stats
                        if s.get("status") != "failed"
                    )
                    * 100
                )
                if sum(
                    s.get("ray_tracing", {}).get("total_rays", 0)
                    for s in all_file_stats
                    if s.get("status") != "failed"
                )
                > 0
                else 0
            ),
            "mission_metrics": {
                "total_distance_m": sum(
                    s.get("mission_metrics", {}).get("total_distance_m", 0)
                    for s in all_file_stats
                    if s.get("status") != "failed"
                ),
                "total_duration_sec": sum(
                    s.get("mission_metrics", {}).get("survey_duration_sec", 0)
                    for s in all_file_stats
                    if s.get("status") != "failed"
                ),
                "avg_speed_ms": (
                    sum(
                        s.get("mission_metrics", {}).get("avg_speed_ms", 0)
                        for s in all_file_stats
                        if s.get("status") != "failed"
                    )
                    / success_count
                    if success_count > 0
                    else 0
                ),
                "avg_speed_kmh": (
                    sum(
                        s.get("mission_metrics", {}).get("avg_speed_kmh", 0)
                        for s in all_file_stats
                        if s.get("status") != "failed"
                    )
                    / success_count
                    if success_count > 0
                    else 0
                ),
            },
            "coverage_metrics": {
                "total_coverage_area_m2": sum(
                    s.get("coverage_metrics", {})
                    .get("coverage_area", {})
                    .get("total_covered_m2", 0)
                    for s in all_file_stats
                    if s.get("status") != "failed"
                ),
                "mean_swath_width_m": (
                    sum(
                        s.get("coverage_metrics", {})
                        .get("swath_width", {})
                        .get("mean_m", 0)
                        for s in all_file_stats
                        if s.get("status") != "failed"
                    )
                    / success_count
                    if success_count > 0
                    else 0
                ),
                "min_swath_width_m": (
                    min(
                        [
                            s.get("coverage_metrics", {})
                            .get("swath_width", {})
                            .get("min_m", float("inf"))
                            for s in all_file_stats
                            if s.get("status") != "failed"
                        ]
                    )
                    if success_count > 0
                    else 0
                ),
                "max_swath_width_m": (
                    max(
                        [
                            s.get("coverage_metrics", {})
                            .get("swath_width", {})
                            .get("max_m", 0)
                            for s in all_file_stats
                            if s.get("status") != "failed"
                        ]
                    )
                    if success_count > 0
                    else 0
                ),
                "mean_coverage_efficiency_pct": (
                    sum(
                        s.get("coverage_metrics", {})
                        .get("coverage_area", {})
                        .get("efficiency_pct", 0)
                        for s in all_file_stats
                        if s.get("status") != "failed"
                    )
                    / success_count
                    if success_count > 0
                    else 0
                ),
            },
        },
        "individual_files": all_file_stats,
    }

    # Save statistics to JSON file
    stats_file = output_dir / "processing_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\n📊 Saved processing statistics to: {stats_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(h5_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(
        f"Total processing time: {int(processing_duration//60)}m {int(processing_duration%60)}s"
    )
    print(f"Output directory: {output_dir}")
    print(f"Statistics file: {stats_file}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
