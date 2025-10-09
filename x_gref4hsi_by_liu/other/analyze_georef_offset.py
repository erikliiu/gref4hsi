"""
Compare actual georeferenced positions with expected positions based on navigation
"""

import numpy as np
import h5py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import utils

print("=" * 80)
print("ANALYZING GEOREFERENCED DATA vs NAVIGATION")
print("=" * 80)

# Find first H5 file in output folder
output_folder = Path(config.OUTPUT_FOLDER)
h5_files = sorted(output_folder.glob("*.h5"))

if not h5_files:
    print(f"No H5 files found in {output_folder}")
    sys.exit(1)

h5_path = h5_files[0]
print(f"\nAnalyzing: {h5_path.name}")

# Load georef data
with h5py.File(h5_path, "r") as f:
    # Check available georef datasets
    if "processed/georef" in f:
        print("\nAvailable georef datasets:")
        for key in f["processed/georef"].keys():
            print(f"  - {key}")

    if "processed/georef/points_ecef_crs" in f:
        points_ecef = f["processed/georef/points_ecef_crs"][:]

        # Try different possible names for indices
        if "processed/georef/frame_nr_grid" in f:
            # Grid format: (frames, pixels, 3)
            frame_nr_grid = f["processed/georef/frame_nr_grid"][:]
            pixel_nr_grid = f["processed/georef/pixel_nr_grid"][:]
            print(f"\nGrid format detected:")
            print(f"  points_ecef shape: {points_ecef.shape}")
            print(f"  frame_nr_grid shape: {frame_nr_grid.shape}")
            print(f"  pixel_nr_grid shape: {pixel_nr_grid.shape}")

            # For grid format, points are (T, S, 3)
            # We need to iterate over the grid
            is_grid_format = True
        elif "processed/georef/frame_indices" in f:
            frame_indices = f["processed/georef/frame_indices"][:]
            pixel_indices = f["processed/georef/ray_indices"][:]
            is_grid_format = False
        else:
            print("Could not find frame/pixel indices in H5 file")
            sys.exit(1)

        if is_grid_format:
            T, S, _ = points_ecef.shape
            print(f"\nGeoreferenced grid: {T} frames × {S} pixels")
            # Flatten for analysis
            points_ecef_flat = points_ecef.reshape(-1, 3)
            frame_indices = np.repeat(np.arange(T), S)
            pixel_indices = np.tile(np.arange(S), T)
            # Only keep valid points (finite coordinates)
            valid_mask = np.all(np.isfinite(points_ecef_flat), axis=1)
            points_ecef = points_ecef_flat[valid_mask]
            frame_indices = frame_indices[valid_mask]
            pixel_indices = pixel_indices[valid_mask]
            print(
                f"Valid georeferenced points: {len(points_ecef)} ({100*len(points_ecef)/(T*S):.1f}%)"
            )
        else:
            print(f"\nGeoreferenced points: {len(points_ecef)}")

        print(f"Frames: {frame_indices.min()} to {frame_indices.max()}")
        print(f"Pixels: {pixel_indices.min()} to {pixel_indices.max()}")

        # Convert to UTM for easier visualization
        x_utm, y_utm, z_utm = utils.ecef_to_utm(
            points_ecef[:, 0],
            points_ecef[:, 1],
            points_ecef[:, 2],
            epsg_utm=config.EPSG_MBES,
            epsg_ecef=config.EPSG_ECEF,
        )

        # Load navigation
        nav_df = utils.load_csv_navigation(config.NAV_CSV, config.CSV_COLUMNS)
        hsi_timestamps = utils.load_h5_timestamps(str(h5_path))

        # Interpolate nav to HSI frames
        interp_nav = utils.interpolate_navigation(
            nav_df, hsi_timestamps, time_offset=config.TIME_OFFSET_SEC
        )

        # Convert nav to UTM
        nav_x_ecef, nav_y_ecef, nav_z_ecef = utils.geographic_to_ecef(
            interp_nav["longitude"],
            interp_nav["latitude"],
            -interp_nav["depth"],
            epsg_geo=config.EPSG_GEOGRAPHIC,
            epsg_ecef=config.EPSG_ECEF,
        )

        nav_x_utm, nav_y_utm, nav_z_utm = utils.ecef_to_utm(
            nav_x_ecef,
            nav_y_ecef,
            nav_z_ecef,
            epsg_utm=config.EPSG_MBES,
            epsg_ecef=config.EPSG_ECEF,
        )

        # Apply body->camera transform to get camera positions
        positions_ecef = np.column_stack([nav_x_ecef, nav_y_ecef, nav_z_ecef])
        orientations = utils.euler_to_rotation_matrix_nav(
            interp_nav["roll"],
            interp_nav["pitch"],
            interp_nav["yaw"],
            yaw_convention=config.YAW_CONVENTION,
        )

        cam_pos_ecef, cam_orientations = utils.apply_sensor_transform(
            positions_ecef,
            orientations,
            config.ROTATION_HSI_TO_BODY,
            config.TRANSLATION_BODY_TO_HSI,
        )

        cam_x_utm, cam_y_utm, cam_z_utm = utils.ecef_to_utm(
            cam_pos_ecef[:, 0],
            cam_pos_ecef[:, 1],
            cam_pos_ecef[:, 2],
            epsg_utm=config.EPSG_MBES,
            epsg_ecef=config.EPSG_ECEF,
        )

        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)

        # Get unique frames
        unique_frames = np.unique(frame_indices)
        print(f"\nFrames with georef hits: {len(unique_frames)}")

        # For each frame, compute stats
        print("\n" + "-" * 80)
        print("Frame-by-frame analysis (first 10 frames):")
        print("-" * 80)

        for frame_idx in unique_frames[:10]:
            mask = frame_indices == frame_idx
            frame_points_x = x_utm[mask]
            frame_points_y = y_utm[mask]

            # Camera position for this frame
            cam_x = cam_x_utm[frame_idx]
            cam_y = cam_y_utm[frame_idx]

            # Nav position for this frame (body)
            nav_x = nav_x_utm[frame_idx]
            nav_y = nav_y_utm[frame_idx]

            # Compute offset between camera and nav
            cam_offset_x = cam_x - nav_x
            cam_offset_y = cam_y - nav_y

            # Compute center of georef swath
            swath_center_x = np.mean(frame_points_x)
            swath_center_y = np.mean(frame_points_y)

            # Offset from camera to swath center
            swath_offset_x = swath_center_x - cam_x
            swath_offset_y = swath_center_y - cam_y

            # Offset from nav to swath center
            nav_to_swath_x = swath_center_x - nav_x
            nav_to_swath_y = swath_center_y - nav_y

            # Compute heading (yaw) for this frame
            yaw = interp_nav["yaw"][frame_idx]

            print(f"\nFrame {frame_idx}:")
            print(f"  Yaw (compass): {yaw:.1f}°")
            print(f"  Nav position: ({nav_x:.2f}, {nav_y:.2f})")
            print(f"  Cam position: ({cam_x:.2f}, {cam_y:.2f})")
            print(f"  Cam offset from nav: ({cam_offset_x:.2f}, {cam_offset_y:.2f})")
            print(f"  Swath center: ({swath_center_x:.2f}, {swath_center_y:.2f})")
            print(
                f"  Swath offset from cam: ({swath_offset_x:.2f}, {swath_offset_y:.2f})"
            )
            print(
                f"  Swath offset from nav: ({nav_to_swath_x:.2f}, {nav_to_swath_y:.2f})"
            )
            print(f"  Number of hits: {mask.sum()}")

        # Overall lateral offset analysis
        print("\n" + "=" * 80)
        print("OVERALL LATERAL OFFSET ANALYSIS")
        print("=" * 80)

        # Compute mean heading direction
        mean_yaw = np.mean(interp_nav["yaw"])
        print(f"\nMean heading: {mean_yaw:.1f}° (compass)")

        # Convert to ENU yaw
        yaw_enu_rad = np.deg2rad(90.0 - mean_yaw)
        flight_dir = np.array([np.cos(yaw_enu_rad), np.sin(yaw_enu_rad)])
        perpendicular_dir = np.array([-np.sin(yaw_enu_rad), np.cos(yaw_enu_rad)])

        print(f"Flight direction (ENU): ({flight_dir[0]:.3f}, {flight_dir[1]:.3f})")
        print(
            f"Perpendicular (right): ({perpendicular_dir[0]:.3f}, {perpendicular_dir[1]:.3f})"
        )

        # For each georef point, compute along-track and cross-track offset from nav
        all_offsets_x = []
        all_offsets_y = []
        all_along_track = []
        all_cross_track = []

        for i, frame_idx in enumerate(frame_indices):
            # Point position
            point_x = x_utm[i]
            point_y = y_utm[i]

            # Nav position
            nav_x = nav_x_utm[frame_idx]
            nav_y = nav_y_utm[frame_idx]

            # Offset
            offset_x = point_x - nav_x
            offset_y = point_y - nav_y

            all_offsets_x.append(offset_x)
            all_offsets_y.append(offset_y)

            # Project onto flight direction (along-track)
            # and perpendicular (cross-track)
            yaw_frame = interp_nav["yaw"][frame_idx]
            yaw_frame_enu_rad = np.deg2rad(90.0 - yaw_frame)
            flight_dir_frame = np.array(
                [np.cos(yaw_frame_enu_rad), np.sin(yaw_frame_enu_rad)]
            )
            perp_dir_frame = np.array(
                [-np.sin(yaw_frame_enu_rad), np.cos(yaw_frame_enu_rad)]
            )

            offset_vec = np.array([offset_x, offset_y])
            along_track = np.dot(offset_vec, flight_dir_frame)
            cross_track = np.dot(offset_vec, perp_dir_frame)

            all_along_track.append(along_track)
            all_cross_track.append(cross_track)

        all_offsets_x = np.array(all_offsets_x)
        all_offsets_y = np.array(all_offsets_y)
        all_along_track = np.array(all_along_track)
        all_cross_track = np.array(all_cross_track)

        print(f"\nOffset from navigation (mean ± std):")
        print(f"  East:  {np.mean(all_offsets_x):.2f} ± {np.std(all_offsets_x):.2f} m")
        print(f"  North: {np.mean(all_offsets_y):.2f} ± {np.std(all_offsets_y):.2f} m")
        print(
            f"\nAlong-track offset: {np.mean(all_along_track):.2f} ± {np.std(all_along_track):.2f} m"
        )
        print(
            f"Cross-track offset: {np.mean(all_cross_track):.2f} ± {np.std(all_cross_track):.2f} m"
        )

        if abs(np.mean(all_cross_track)) > 1.0:
            print(f"\n⚠️  WARNING: Large cross-track offset detected!")
            print(f"   Mean cross-track offset: {np.mean(all_cross_track):.2f} m")
            print(
                f"   This indicates georeferenced data is laterally offset from trajectory!"
            )

    else:
        print("No georef data found in H5 file")
