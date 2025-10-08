"""
Utility functions for simplified UHI + MBES georeferencing
"""

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as RotLib
import xmltodict
import h5py
from pathlib import Path


def load_csv_navigation(csv_path, column_map):
    """
    Load navigation data from CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to CSV navigation file
    column_map : dict
        Dictionary mapping standard keys to CSV column names
        Example: {'timestamp': 'timestamp [unix epoch s]', 'latitude': 'latitude [deg]', ...}

    Returns:
    --------
    pd.DataFrame
        Navigation data with standardized column names
    """
    nav_data = pd.read_csv(csv_path)

    # Verify required columns exist
    for key, col_name in column_map.items():
        if col_name not in nav_data.columns:
            raise ValueError(
                f"Required column '{col_name}' not found in CSV. Available: {list(nav_data.columns)}"
            )

    # Rename columns to standard names
    rename_map = {v: k for k, v in column_map.items()}
    nav_data = nav_data.rename(columns=rename_map)

    print(f"Loaded {len(nav_data)} navigation records from {csv_path}")
    print(
        f"Time range: {nav_data['timestamp'].min():.2f} to {nav_data['timestamp'].max():.2f}"
    )

    return nav_data


def load_camera_calibration(xml_path):
    """
    Load HSI camera calibration from XML file.

    Parameters:
    -----------
    xml_path : str
        Path to camera calibration XML file

    Returns:
    --------
    dict
        Camera calibration parameters (f, cx, w, distortion, etc.)
    """
    with open(xml_path, "r", encoding="utf-8") as file:
        my_xml = file.read()
    xml_dict = xmltodict.parse(my_xml)
    calib = xml_dict["calibration"]

    return {
        "w": float(calib["width"]),
        "f": float(calib["f"]),
        "cx": float(calib["cx"]),
        "rx": float(calib["rx"]),
        "ry": float(calib["ry"]),
        "rz": float(calib["rz"]),
        "tx": float(calib["tx"]),
        "ty": float(calib["ty"]),
        "tz": float(calib["tz"]),
        "k1": float(calib["k1"]),
        "k2": float(calib["k2"]),
        "k3": float(calib["k3"]),
    }


def geographic_to_ecef(lon, lat, height, epsg_geo=4326, epsg_ecef=4978):
    """
    Convert geographic coordinates to ECEF.

    Parameters:
    -----------
    lon, lat, height : array-like
        Geographic coordinates (degrees, meters)
    epsg_geo : int
        EPSG code for geographic CRS (default: 4326 - WGS84)
    epsg_ecef : int
        EPSG code for ECEF CRS (default: 4978 - WGS84 ECEF)

    Returns:
    --------
    tuple
        (x, y, z) in ECEF coordinates
    """
    transformer = Transformer.from_crs(epsg_geo, epsg_ecef, always_xy=True)
    x, y, z = transformer.transform(lon, lat, height)
    return x, y, z


def ecef_to_utm(x, y, z, epsg_utm=32632, epsg_ecef=4978):
    """
    Convert ECEF coordinates to UTM.

    Parameters:
    -----------
    x, y, z : array-like
        ECEF coordinates (meters)
    epsg_utm : int
        EPSG code for UTM zone (default: 32632 - UTM Zone 32N)
    epsg_ecef : int
        EPSG code for ECEF (default: 4978)

    Returns:
    --------
    tuple
        (easting, northing, height) in UTM coordinates
    """
    transformer = Transformer.from_crs(epsg_ecef, epsg_utm, always_xy=True)
    easting, northing, height = transformer.transform(x, y, z)
    return easting, northing, height


def utm_to_ecef(easting, northing, height, epsg_utm=32632, epsg_ecef=4978):
    """
    Convert UTM coordinates to ECEF.

    Parameters:
    -----------
    easting, northing, height : array-like
        UTM coordinates (meters)
    epsg_utm : int
        EPSG code for UTM zone
    epsg_ecef : int
        EPSG code for ECEF

    Returns:
    --------
    tuple
        (x, y, z) in ECEF coordinates
    """
    transformer = Transformer.from_crs(epsg_utm, epsg_ecef, always_xy=True)
    x, y, z = transformer.transform(easting, northing, height)
    return x, y, z


def interpolate_navigation(nav_data, hsi_timestamps, time_offset=0):
    """
    Interpolate navigation data to HSI frame timestamps.

    Parameters:
    -----------
    nav_data : pd.DataFrame
        Navigation data with columns: timestamp, latitude, longitude, depth, roll, pitch, yaw
    hsi_timestamps : array-like
        Target timestamps for interpolation (HSI frame times)
    time_offset : float
        Time offset to add to HSI timestamps (seconds)

    Returns:
    --------
    dict
        Interpolated navigation data with keys: 'latitude', 'longitude', 'depth', 'roll', 'pitch', 'yaw'
    """
    # Apply time offset to HSI timestamps
    hsi_times_corrected = hsi_timestamps + time_offset

    # Check interpolation range
    nav_min = nav_data["timestamp"].min()
    nav_max = nav_data["timestamp"].max()
    hsi_min = hsi_times_corrected.min()
    hsi_max = hsi_times_corrected.max()

    if hsi_min < nav_min or hsi_max > nav_max:
        print(
            f"WARNING: HSI time range [{hsi_min:.2f}, {hsi_max:.2f}] exceeds navigation range [{nav_min:.2f}, {nav_max:.2f}]"
        )
        print("Extrapolation will be used - results may be unreliable!")

    # Interpolate each navigation component
    interp_data = {}
    base_cols = ["latitude", "longitude", "depth", "roll", "pitch", "yaw"]
    extra_cols = ["altitude"] if "altitude" in nav_data.columns else []

    for col in base_cols + extra_cols:
        f_interp = interp1d(
            nav_data["timestamp"],
            nav_data[col],
            kind="linear",
            fill_value="extrapolate",
        )
        interp_data[col] = f_interp(hsi_times_corrected)

    print(f"Interpolated navigation for {len(hsi_times_corrected)} HSI frames")

    return interp_data


def apply_sensor_transform(
    positions_ecef, orientations, rotation_matrix, translation_vector
):
    """
    Apply rotation and translation to transform from body frame to sensor frame.

    Parameters:
    -----------
    positions_ecef : ndarray (N, 3)
        Vehicle positions in ECEF coordinates
    orientations : ndarray (N, 3, 3)
        Vehicle orientation rotation matrices (body to world)
    rotation_matrix : ndarray (3, 3)
        Rotation matrix from sensor frame to body frame
    translation_vector : ndarray (3,)
        Translation vector from body origin to sensor origin (in body frame)

    Returns:
    --------
    tuple
        (sensor_positions_ecef, sensor_orientations)
        - sensor_positions_ecef: ndarray (N, 3) - sensor positions in ECEF
        - sensor_orientations: ndarray (N, 3, 3) - sensor orientations (sensor to world)
    """
    N = positions_ecef.shape[0]
    sensor_positions_ecef = np.zeros((N, 3))
    sensor_orientations = np.zeros((N, 3, 3))

    for i in range(N):
        # Rotation from body to world
        R_body_to_world = orientations[i]

        # Transform translation vector from body frame to world frame
        translation_world = R_body_to_world @ translation_vector

        # Sensor position = vehicle position + translated offset
        sensor_positions_ecef[i] = positions_ecef[i] + translation_world

        # Sensor orientation = R_body_to_world @ R_sensor_to_body
        sensor_orientations[i] = R_body_to_world @ rotation_matrix

    return sensor_positions_ecef, sensor_orientations


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """
    Convert Euler angles to rotation matrices.

    DEPRECATED: Use euler_to_rotation_matrix_nav() for navigation data with proper yaw convention handling.

    Parameters:
    -----------
    roll_deg, pitch_deg, yaw_deg : array-like
        Euler angles in degrees

    Returns:
    --------
    ndarray (N, 3, 3)
        Rotation matrices for each set of angles
    """
    # Convert degrees to radians
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    # Stack angles for batch processing
    angles = np.stack([roll_rad, pitch_rad, yaw_rad], axis=-1)

    # Create rotation objects (using ZYX convention - yaw, pitch, roll)
    rotations = RotLib.from_euler("ZYX", angles)

    # Return rotation matrices
    return rotations.as_matrix()


def euler_to_rotation_matrix_nav(
    roll_deg, pitch_deg, yaw_input_deg, yaw_convention="heading_from_north_cw"
):
    """
    Convert navigation RPY to rotation matrices mapping BODY->WORLD (ENU).

    This function handles different yaw conventions correctly, ensuring that the
    rotation matrices match between simulation and production code.

    Parameters:
    -----------
    roll_deg, pitch_deg : array-like
        Roll and pitch angles in degrees
        Body frame convention: X forward, Y right/starboard, Z up
    yaw_input_deg : array-like
        Yaw angle in degrees, interpretation depends on yaw_convention
    yaw_convention : str
        - 'heading_from_north_cw': compass heading (0=North, 90=East, clockwise positive)
        - 'enu_yaw_from_east_ccw': ENU yaw (0=East, 90=North, counter-clockwise positive)

    Returns:
    --------
    ndarray (N, 3, 3)
        Rotation matrices R_world_from_body (BODY->WORLD in ENU frame)
    """
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)

    if yaw_convention == "heading_from_north_cw":
        # Convert compass heading to ENU yaw (from East CCW)
        yaw_rad = np.deg2rad(90.0 - np.asarray(yaw_input_deg, dtype=float))
    else:
        yaw_rad = np.deg2rad(yaw_input_deg)

    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    # Build rotation matrices: Rz(yaw) @ Ry(pitch) @ Rx(roll)  → BODY->WORLD(ENU)
    Rz = np.stack(
        [
            np.stack([cy, -sy, 0 * np.ones_like(cy)], axis=-1),
            np.stack([sy, cy, 0 * np.ones_like(cy)], axis=-1),
            np.stack(
                [0 * np.ones_like(cy), 0 * np.ones_like(cy), np.ones_like(cy)], axis=-1
            ),
        ],
        axis=-2,
    )

    Ry = np.stack(
        [
            np.stack([cp, 0 * np.ones_like(cp), sp], axis=-1),
            np.stack(
                [0 * np.ones_like(cp), np.ones_like(cp), 0 * np.ones_like(cp)], axis=-1
            ),
            np.stack([-sp, 0 * np.ones_like(cp), cp], axis=-1),
        ],
        axis=-2,
    )

    Rx = np.stack(
        [
            np.stack(
                [np.ones_like(cr), 0 * np.ones_like(cr), 0 * np.ones_like(cr)], axis=-1
            ),
            np.stack([0 * np.ones_like(cr), cr, -sr], axis=-1),
            np.stack([0 * np.ones_like(cr), sr, cr], axis=-1),
        ],
        axis=-2,
    )

    return Rz @ Ry @ Rx


def build_ray_directions(camera_calib, num_pixels):
    """
    Build ray directions in camera frame using test_eely convention.

    This follows the exact same formula as gref4hsi/scripts/georeference.py:cal_file_to_rays()
    to ensure consistency with the official pipeline.

    Camera frame convention (matching test_eely):
      X: across-track (perpendicular to flight direction)
      Y: along-track (flight direction) - always 0 for pushbroom
      Z: down (nadir/boresight) - always 1 (unnormalized)

    Distortion model:
      x_norm = x_norm_lin + x_norm_nonlin
      x_norm_lin = (u - cx) / f
      x_norm_nonlin = -(k1*r^5 + k2*r^3 + k3*r^2) / f
      where r = (u - cx) / 1000

    Parameters
    ----------
    camera_calib : dict
        Camera calibration with keys: f, cx, k1, k2, k3
    num_pixels : int
        Number of pixels along the scan line

    Returns
    -------
    rays_cam : ndarray (num_pixels, 3)
        Ray directions in camera frame: [x_norm, 0, 1]
        NOTE: Not normalized! Matches test_eely convention.
    """
    f = float(camera_calib["f"])
    cx = float(camera_calib["cx"])
    k1 = float(camera_calib["k1"])
    k2 = float(camera_calib["k2"])
    k3 = float(camera_calib["k3"])

    # Pixel indices (1-based, matching test_eely)
    u = np.arange(1, num_pixels + 1)

    # Linear component
    x_norm_lin = (u - cx) / f

    # Nonlinear distortion component
    # NOTE: Negative sign and scaling by /1000 are critical!
    r = (u - cx) / 1000.0  # Scale for numerical stability
    x_norm_nonlin = -(k1 * r**5 + k2 * r**3 + k3 * r**2) / f

    # Total normalized x-coordinate
    x_norm = x_norm_lin + x_norm_nonlin

    # Build ray directions (test_eely convention)
    p_dir = np.zeros((len(x_norm), 3))
    p_dir[:, 0] = x_norm  # Across-track on X-axis
    p_dir[:, 1] = 0  # Along-track on Y-axis (always zero for pushbroom)
    p_dir[:, 2] = 1  # Down on Z-axis (boresight)

    # DO NOT normalize here - test_eely keeps them as [x_norm, 0, 1]
    # This matches the official gref4hsi pipeline behavior

    return p_dir


def load_h5_timestamps(h5_path):
    """
    Load HSI frame timestamps from H5 file.

    Parameters:
    -----------
    h5_path : str
        Path to H5 file

    Returns:
    --------
    ndarray
        Array of timestamps for each HSI frame
    """
    with h5py.File(h5_path, "r") as h5f:
        # Check for common timestamp dataset names (UHI format)
        possible_paths = [
            "processed/radiance/timestamp",  # Standard UHI path for HSI timestamps
        ]

        for path in possible_paths:
            if path in h5f:
                timestamps = h5f[path][:]
                print(f"Loaded {len(timestamps)} timestamps from {Path(h5_path).name}")
                print(f"  Dataset path: {path}")
                print(f"  Time range: {timestamps.min():.2f} to {timestamps.max():.2f}")
                return timestamps

        # If no standard path found, list available datasets
        print(f"WARNING: Could not find timestamp dataset in {h5_path}")
        print(f"Available top-level groups: {list(h5f.keys())}")

        # Try to list all datasets to help user
        def list_datasets(name, obj):
            if isinstance(obj, h5py.Dataset) and "time" in name.lower():
                print(f"  Found dataset with 'time': {name}")

        print("Searching for datasets containing 'time':")
        h5f.visititems(list_datasets)

        raise ValueError(f"No timestamp dataset found in {h5_path}")


def save_intersection_to_h5(
    h5_path,
    intersection_points,
    ray_indices,
    frame_indices,
    n_frames,
    n_slits,
    gridded=True,
):
    """
    Save ray intersection results to H5 file.

    Parameters:
    -----------
    h5_path : str
        Path to H5 file to save results
    intersection_points : ndarray (N, 3)
        Intersection points in ECEF coordinates (flattened)
    ray_indices : ndarray (N,)
        Pixel/slit indices for each intersection
    frame_indices : ndarray (N,)
        Frame indices for each intersection
    n_frames : int
        Total number of frames (T)
    n_slits : int
        Total number of slits/pixels (S)
    gridded : bool
        If True, save as (T, S, 3) gridded format (test_eely compatible)
        If False, save as (N, 3) flattened format
    """
    with h5py.File(h5_path, "a") as h5f:
        # Follow gref4hsi convention: store in processed/georef/
        # Delete old datasets if they exist (clean up both old and new format)
        datasets_to_remove = [
            "processed/georef/points_ecef_crs",
            "processed/georef/pixel_nr_grid",
            "processed/georef/frame_nr_grid",
            "processed/georef/ray_indices",  # Old flattened format
            "processed/georef/frame_indices",  # Old flattened format
        ]
        for dset_name in datasets_to_remove:
            if dset_name in h5f:
                del h5f[dset_name]

        # Create group if doesn't exist
        if "processed/georef" not in h5f:
            h5f.create_group("processed/georef")

        if gridded:
            # GRIDDED FORMAT: (T, S, 3) - preserves hypercube structure
            print(f"Saving in GRIDDED format: ({n_frames}, {n_slits}, 3)")

            # Create gridded arrays filled with NaN
            points_grid = np.full((n_frames, n_slits, 3), np.nan, dtype=np.float64)
            pixel_grid = np.full((n_frames, n_slits), -1, dtype=np.int32)
            frame_grid = np.full((n_frames, n_slits), -1, dtype=np.int32)

            # Fill in successful intersections
            points_grid[frame_indices, ray_indices, :] = intersection_points
            pixel_grid[frame_indices, ray_indices] = ray_indices
            frame_grid[frame_indices, ray_indices] = frame_indices

            # Save gridded data (test_eely compatible)
            h5f.create_dataset(
                "processed/georef/points_ecef_crs",
                data=points_grid,
                compression="gzip",
                compression_opts=4,
            )
            h5f.create_dataset(
                "processed/georef/pixel_nr_grid",
                data=pixel_grid,
                compression="gzip",
                compression_opts=4,
            )
            h5f.create_dataset(
                "processed/georef/frame_nr_grid",
                data=frame_grid,
                compression="gzip",
                compression_opts=4,
            )

            # Calculate statistics
            num_hits = len(intersection_points)
            num_total = n_frames * n_slits
            hit_rate = 100.0 * num_hits / num_total

            print(f"  Successful rays: {num_hits:,} / {num_total:,} ({hit_rate:.2f}%)")
            print(f"  Failed rays filled with NaN")

        else:
            # FLATTENED FORMAT: (N, 3) - only successful rays
            print(f"Saving in FLATTENED format: ({len(intersection_points)}, 3)")

            h5f.create_dataset(
                "processed/georef/points_ecef_crs",
                data=intersection_points,
                compression="gzip",
                compression_opts=4,
            )
            h5f.create_dataset(
                "processed/georef/pixel_nr_grid",
                data=ray_indices,
                compression="gzip",
                compression_opts=4,
            )
            h5f.create_dataset(
                "processed/georef/frame_nr_grid",
                data=frame_indices,
                compression="gzip",
                compression_opts=4,
            )

        # Save metadata as attributes
        h5f["processed/georef"].attrs["epsg_code"] = 4978  # ECEF
        h5f["processed/georef"].attrs["num_intersections"] = len(intersection_points)
        h5f["processed/georef"].attrs["num_frames"] = n_frames
        h5f["processed/georef"].attrs["num_slits"] = n_slits
        h5f["processed/georef"].attrs["gridded"] = gridded
        h5f["processed/georef"].attrs["timestamp"] = pd.Timestamp.now().isoformat()
        h5f["processed/georef"].attrs[
            "description"
        ] = "Ray-traced intersection points from x_gref4hsi_by_liu"

    print(f"Saved {len(intersection_points):,} intersections to {Path(h5_path).name}")
    print(f"  Dataset: processed/georef/points_ecef_crs")
