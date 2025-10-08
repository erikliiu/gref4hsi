"""
Configuration for simplified UHI georeferencing with MBES GeoTIFF
"""

import numpy as np
from pathlib import Path

# ===== INPUT PATHS =====
NAV_CSV = r"E:\mjosa_new\navigation_data\nav_data_merged.csv"
MBES_GEOTIFF = r"E:\mjosa_new\DTM\geotiff_2.tif"
H5_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\057_own_code_1to2\input"

# ===== OUTPUT PATHS =====
OUTPUT_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\057_own_code_1to2\output"

# ===== SENSOR CONFIGURATION =====
# HSI camera relative to vehicle body frame
# EELY configuration: UHI and MBES both looking straight down (push-broom)
# NO ROTATION - camera axes aligned with body axes (both in NED frame)
ROTATION_HSI_TO_BODY = np.eye(3)  # Identity matrix - no rotation
TRANSLATION_BODY_TO_HSI = np.array([2.5, 0, 0])  # 2.5m forward offset

# Time synchronization
TIME_OFFSET_SEC = 508  # Same as in test_eely.py

# Camera calibration file
CAMERA_CALIB_XML = r"E:\mjosa_new_oct_2025\anxillary_data\HSI_2_body.xml"

# ===== COORDINATE SYSTEMS =====
# Origin for local NED frame (Mjøsa area)
LON0 = 10.7122345
LAT0 = 60.8011575
H0 = 0.0

# EPSG codes
EPSG_GEOGRAPHIC = 4326  # WGS84 lat/lon
EPSG_UTM = 32632  # UTM Zone 32N
EPSG_ECEF = 4978  # Earth-Centered Earth-Fixed

# MBES GeoTIFF EPSG (check your GeoTIFF metadata!)
EPSG_MBES = 32632  # Usually UTM Zone 32N

# ===== PROCESSING PARAMETERS =====
# Ray tracing
MAX_RAY_LENGTH = 100  # meters (increased from 20 for deeper water)
EARLY_FAILURE_THRESHOLD = 50.0  # Cancel if >50% rays fail

# CSV column names (adjust if your CSV has different headers)
CSV_COLUMNS = {
    "timestamp": "timestamp [unix epoch s]",
    "latitude": "latitude [deg]",
    "longitude": "longitude [deg]",
    "depth": "depth [m]",
    "roll": "roll [deg]",
    "pitch": "pitch [deg]",
    "yaw": "yaw [deg]",
    "altitude": "altitude [m]",
}

# ===== MESH CREATION =====
# When converting MBES GeoTIFF to mesh
MESH_SIMPLIFICATION = False  # Set True to reduce mesh size
MESH_REDUCTION_FACTOR = 0.5  # If simplification enabled, keep 50% of faces

# ===== DEBUG OPTIONS =====
VERBOSE = True
SAVE_MESH_PLY = True  # Save converted MBES mesh to PLY
SAVE_INTERSECTION_STATS = True
