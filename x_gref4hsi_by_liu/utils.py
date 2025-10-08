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
    for col in ["latitude", "longitude", "depth", "roll", "pitch", "yaw"]:
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


def build_ray_directions(camera_calib, num_pixels):
    """
    Build ray direction vectors for all pixels in HSI line camera.

    Parameters:
    -----------
    camera_calib : dict
        Camera calibration parameters (f, cx, w, k1, k2, k3)
    num_pixels : int
        Number of pixels across the line (typically matches 'w' in calibration)

    Returns:
    --------
    ndarray (num_pixels, 3)
        Ray direction vectors in camera frame (normalized)
    """
    f = camera_calib["f"]
    cx = camera_calib["cx"]

    # Pixel coordinates for push-broom line camera
    pixel_coords = np.arange(num_pixels)

    # Normalized image coordinates
    # Note: pixel 0 = starboard (+Y), pixel_max = port (-Y)
    # So we need to flip: increasing pixel index = decreasing Y
    x_norm = (pixel_coords - cx) / f

    # Apply distortion (radial distortion model)
    k1, k2, k3 = camera_calib["k1"], camera_calib["k2"], camera_calib["k3"]
    r2 = x_norm**2
    distortion_factor = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_distorted = x_norm * distortion_factor

    # Ray directions in BODY frame (NED-aligned):
    # X = Forward (along-track) → 0 for push-broom perpendicular scan
    # Y = Starboard/Right (across-track) → varies with pixel, NEGATIVE because:
    #     pixel 0 (starboard) = +Y, pixel_max (port) = -Y
    # Z = Down (nadir) → +1 pointing at seafloor
    rays = np.zeros((num_pixels, 3))
    rays[:, 0] = 0  # No forward component (perpendicular scan)
    rays[:, 1] = -x_distorted  # Across-track, NEGATED for starboard→port convention
    rays[:, 2] = 1  # Downward (nadir pointing)

    # Normalize ray vectors
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

    return rays


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


def save_intersection_to_h5(h5_path, intersection_points, ray_indices, frame_indices):
    """
    Save ray intersection results to H5 file.

    Parameters:
    -----------
    h5_path : str
        Path to H5 file to save results
    intersection_points : ndarray (N, 3)
        Intersection points in ECEF coordinates
    ray_indices : ndarray (N,)
        Pixel indices for each intersection
    frame_indices : ndarray (N,)
        Frame indices for each intersection
    """
    with h5py.File(h5_path, "a") as h5f:
        # Follow gref4hsi convention: store in processed/georef/
        # Delete old datasets if they exist (not entire group, to preserve other data)
        if "processed/georef/points_ecef_crs" in h5f:
            del h5f["processed/georef/points_ecef_crs"]
        if "processed/georef/ray_indices" in h5f:
            del h5f["processed/georef/ray_indices"]
        if "processed/georef/frame_indices" in h5f:
            del h5f["processed/georef/frame_indices"]

        # Create group if doesn't exist
        if "processed/georef" not in h5f:
            h5f.create_group("processed/georef")

        # Save intersection data (standard gref4hsi path)
        h5f.create_dataset(
            "processed/georef/points_ecef_crs",
            data=intersection_points,
            compression="gzip",
            compression_opts=4,
        )

        # Save additional index data (custom, for debugging/reference)
        h5f.create_dataset(
            "processed/georef/ray_indices",
            data=ray_indices,
            compression="gzip",
            compression_opts=4,
        )
        h5f.create_dataset(
            "processed/georef/frame_indices",
            data=frame_indices,
            compression="gzip",
            compression_opts=4,
        )

        # Save metadata as attributes
        h5f["processed/georef"].attrs["epsg_code"] = 4978  # ECEF
        h5f["processed/georef"].attrs["num_intersections"] = len(intersection_points)
        h5f["processed/georef"].attrs["timestamp"] = pd.Timestamp.now().isoformat()
        h5f["processed/georef"].attrs[
            "description"
        ] = "Ray-traced intersection points from x_gref4hsi_by_liu"

    print(f"Saved {len(intersection_points):,} intersections to {Path(h5_path).name}")
    print(f"  Dataset: processed/georef/points_ecef_crs")
