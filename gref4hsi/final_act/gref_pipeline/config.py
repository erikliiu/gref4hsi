"""
Configuration for simplified UHI georeferencing with MBES GeoTIFF
"""

import numpy as np
from pathlib import Path

# ===== INPUT PATHS =====
NAV_CSV = r"E:\mjosa_new\navigation_data\nav_data_merged.csv"
DB_PATH = r"E:\mjosa\29\log_files\LOG_2024-10-29_10-13-32.db3"  # ROS bag database for DVL data
MBES_GEOTIFF = r"E:\mjosa_new\DTM\geotiff_2.tif"  # used for 057
# MBES_GEOTIFF = (
#     r"E:\mjosa_new_oct_2025\anxillary_data\EIVA\all.tif"  # full MBES (used for 028)
# )
# MBES_GEOTIFF = r"E:\mjosa_new\DTM\104921.tif"  # the fucked one to check orientation

# ===== WORKING FOLDER =====
# Change this to switch between different datasets
WORKING_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\057_final"  # this is the final version with all 6 .h5 files
# WORKING_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\028"  # for 028


# Derived paths - no need to change these
H5_FOLDER = WORKING_FOLDER + r"\input"
OUTPUT_FOLDER = WORKING_FOLDER + r"\output"

# ===== SENSOR CONFIGURATION =====
# HSI camera relative to vehicle body frame
# Body frame: +X forward, +Y starboard/right, +Z up
# Camera frame (test_eely): +X across-track, +Y along-track, +Z down
# Mapping cam->body: Xc→+Yb, Yc→+Xb, Zc→−Zb
ROTATION_HSI_TO_BODY = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float
)
TRANSLATION_BODY_TO_HSI = np.array([0, -2.5, 0.0], dtype=float)

# Navigation yaw convention used in the CSV ('yaw [deg]' is a compass heading)
YAW_CONVENTION = "heading_from_north_cw"  # or "enu_yaw_from_east_ccw"

# Time synchronization
TIME_OFFSET_SEC = 508  # Same as in test_eely.py

# Camera calibration file
CAMERA_CALIB_XML = r"E:\mjosa_new_oct_2025\anxillary_data\HSI_2_body.xml"

# ===== COORDINATE SYSTEMS =====
# Start of the mission (not UHI transect but at the very first start of the whole survey)
# Origin for local NED frame (Mjøsa area)
LON0 = 10.705125
LAT0 = 60.801146
H0 = 0.0

# UHI alignment adjustment (for matching with MBES data in NED coordinates)
# These offsets shift the UHI footprint to better align with ground truth
UHI_ALIGNMENT_DX = -0.05  # East offset in meters (positive = shift east)
UHI_ALIGNMENT_DY = -3.0  # North offset in meters (positive = shift north)

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

# ===== UHI FOOTPRINT OVERLAY =====
# Default UHI files and track range for MBES detrending footprint overlay
UHI_FILES = ["rad_uhi_20241029_115057_4", "rad_uhi_20241029_115057_5"]
UHI_TRACK_RANGE_4TO5 = (3039, 4029)  # (track_start, track_end) #this is the focus area
UHI_TRACK_RANGE_5 = (576, 1566)
# ===== MESH CREATION =====
# When converting MBES GeoTIFF to mesh
MESH_SIMPLIFICATION = False  # Set True to reduce mesh size
MESH_REDUCTION_FACTOR = 0.5  # If simplification enabled, keep 50% of faces

# ===== DEBUG OPTIONS =====
VERBOSE = True
SAVE_MESH_PLY = True  # Save converted MBES mesh to PLY
SAVE_INTERSECTION_STATS = True
