"""
Quick check: Did the fix actually change anything?
Compare old vs new georef positions for the same frame.
"""

import numpy as np
import h5py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import utils

print("=" * 80)
print("CHECKING IF FIX CHANGED THE GEOREFERENCING")
print("=" * 80)

output_folder = Path(config.OUTPUT_FOLDER)
h5_files = sorted(output_folder.glob("*.h5"))

if not h5_files:
    print(f"\n⚠️  No H5 files found in {output_folder}")
    print("You need to re-run main.py to regenerate the georeferenced data!")
    sys.exit(1)

h5_path = h5_files[0]
print(f"\nAnalyzing: {h5_path.name}")

# Check file modification time
import os

mod_time = os.path.getmtime(h5_path)
from datetime import datetime

mod_datetime = datetime.fromtimestamp(mod_time)
print(f"File last modified: {mod_datetime}")

# Check current config
print(f"\nCurrent ROTATION_HSI_TO_BODY:")
print(config.ROTATION_HSI_TO_BODY)

# Load one frame of georef data
with h5py.File(h5_path, "r") as f:
    points_ecef = f["processed/georef/points_ecef_crs"][:]
    T, S, _ = points_ecef.shape

    # Get frame 100 (middle of dataset)
    frame_idx = min(100, T // 2)
    frame_points = points_ecef[frame_idx, :, :]

    # Convert to UTM
    valid_mask = np.all(np.isfinite(frame_points), axis=1)
    if valid_mask.sum() > 0:
        x_utm, y_utm, z_utm = utils.ecef_to_utm(
            frame_points[valid_mask, 0],
            frame_points[valid_mask, 1],
            frame_points[valid_mask, 2],
            epsg_utm=config.EPSG_MBES,
            epsg_ecef=config.EPSG_ECEF,
        )

        # Get nav for this frame
        nav_df = utils.load_csv_navigation(config.NAV_CSV, config.CSV_COLUMNS)
        hsi_timestamps = utils.load_h5_timestamps(str(h5_path))
        interp_nav = utils.interpolate_navigation(
            nav_df, hsi_timestamps, time_offset=config.TIME_OFFSET_SEC
        )

        nav_lon = interp_nav["longitude"][frame_idx]
        nav_lat = interp_nav["latitude"][frame_idx]
        nav_depth = interp_nav["depth"][frame_idx]
        yaw = interp_nav["yaw"][frame_idx]

        # Convert nav to UTM
        nav_x_ecef, nav_y_ecef, nav_z_ecef = utils.geographic_to_ecef(
            np.array([nav_lon]),
            np.array([nav_lat]),
            np.array([-nav_depth]),
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

        # Compute swath center
        swath_center_x = np.mean(x_utm)
        swath_center_y = np.mean(y_utm)
        swath_width_x = np.max(x_utm) - np.min(x_utm)
        swath_width_y = np.max(y_utm) - np.min(y_utm)

        # Offset from nav
        offset_x = swath_center_x - nav_x_utm[0]
        offset_y = swath_center_y - nav_y_utm[0]

        # Convert to along-track / cross-track
        yaw_enu_rad = np.deg2rad(90.0 - yaw)
        flight_dir = np.array([np.cos(yaw_enu_rad), np.sin(yaw_enu_rad)])
        perp_dir = np.array([-np.sin(yaw_enu_rad), np.cos(yaw_enu_rad)])

        offset_vec = np.array([offset_x, offset_y])
        along_track_offset = np.dot(offset_vec, flight_dir)
        cross_track_offset = np.dot(offset_vec, perp_dir)

        print(f"\n{'='*80}")
        print(f"FRAME {frame_idx} ANALYSIS")
        print(f"{'='*80}")
        print(f"\nHeading: {yaw:.1f}°")
        print(f"\nNav position (UTM): ({nav_x_utm[0]:.2f}, {nav_y_utm[0]:.2f})")
        print(f"Swath center: ({swath_center_x:.2f}, {swath_center_y:.2f})")
        print(f"Swath dimensions: {swath_width_x:.1f}m × {swath_width_y:.1f}m")
        print(f"\nOffset from nav:")
        print(f"  East: {offset_x:.2f}m")
        print(f"  North: {offset_y:.2f}m")
        print(f"  Total: {np.sqrt(offset_x**2 + offset_y**2):.2f}m")
        print(f"\n  Along-track: {along_track_offset:.2f}m")
        print(f"  Cross-track: {cross_track_offset:.2f}m")

        if abs(cross_track_offset) > 2.0:
            print(f"\n⚠️  LARGE CROSS-TRACK OFFSET: {cross_track_offset:.2f}m")
            print(f"   The swath is offset to the ", end="")
            if cross_track_offset > 0:
                print("RIGHT (starboard)")
            else:
                print("LEFT (port)")
            print(f"\n   If this is the SAME as before the fix, then:")
            print(f"   1. You may not have regenerated the H5 files (run main.py)")
            print(f"   2. OR the fix needs to be reversed (flip back the sign)")
        else:
            print(f"\n✓ Cross-track offset is acceptable: {cross_track_offset:.2f}m")
            print(f"  The fix appears to be working!")
    else:
        print("\n⚠️  No valid georef points found in this frame")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")
print("\nIf you haven't done so already:")
print("1. Make sure config.py has the fix applied (check line 31)")
print("2. Re-run: python main.py")
print("3. Reload the data in your notebook")
print("4. Re-plot and check the offset again")
