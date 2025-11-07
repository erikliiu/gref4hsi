"""
Navigation Processing Configuration for Lake Mjøsa UHI Mission
==============================================================

This is a PURE DATA configuration file - no functions, only constants.
User-adjustable settings for navigation processing.

Helper functions for this config are in config_utils.py

Date: 2024-10-29
Mission: Eelume robot navigation in Lake Mjøsa
"""

import datetime
from pathlib import Path

# ==============================================================================
# DATA PATHS (RELATIVE TO THIS CONFIG FILE)
# ==============================================================================

# Get the mjosa_complete root
# config.py is at: E:\mjosa_complete\gref4hsi\mjosa_code\utils\common\config.py
# We need to go up to: E:\mjosa_complete\
_CONFIG_DIR = Path(__file__).parent  # .../mjosa_code/utils/common/
_MJOSA_CODE_ROOT = _CONFIG_DIR.parent.parent  # .../mjosa_code/
_GREF4HSI_ROOT = _MJOSA_CODE_ROOT.parent  # .../gref4hsi/
_MJOSA_COMPLETE_ROOT = _GREF4HSI_ROOT.parent  # .../mjosa_complete/

# Root data directory (relative paths)
RAW_DATA_DIR = _MJOSA_COMPLETE_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = _MJOSA_COMPLETE_ROOT / "data" / "processed"

# Specific data paths
RAW_NAVIGATION_DIR = RAW_DATA_DIR / "navigation"
PROCESSED_NAVIGATION_DIR = PROCESSED_DATA_DIR / "navigation"

# Input files (adjust filenames as needed)
# The .db3 file is in the navigation folder
LOG_DB3_FILE = RAW_NAVIGATION_DIR / "LOG_2024-10-29_10-13-32.db3"

# Output files
MAIN_CSV_FILE = PROCESSED_NAVIGATION_DIR / "nav_data_from_logfile.csv"
MERGED_CSV_FILE = PROCESSED_NAVIGATION_DIR / "nav_data_merged.csv"
EIVA_TXT_FILE = PROCESSED_NAVIGATION_DIR / "merged_data_for_EIVA.txt"

# Alias for visualization scripts (they expect NAV_CSV)
NAV_CSV = str(MERGED_CSV_FILE)  # Use the merged/corrected navigation data

# ==============================================================================
# VISUALIZATION SCRIPT DATA PATHS
# ==============================================================================
# These paths are used by create_html.py, create_sim.py, and create_stability_plot.py

# MBES Bathymetry GeoTIFF Options:
# Full MBES coverage from EIVA (recommended for most visualizations)
MBES_GEOTIFF = _MJOSA_COMPLETE_ROOT / "data" / "anxilliary" / "EIVA_MBES" / "all.tif"

# Transect-specific MBES (for detailed analysis of transect 057)
MBES_GEOTIFF_057 = (
    _MJOSA_COMPLETE_ROOT / "data" / "anxilliary" / "EIVA" / "geotiff_2.tif"
)

# Georeferenced hyperspectral data (H5 files)
H5_FOLDER = _MJOSA_COMPLETE_ROOT / "use_gref4hsi" / "057_final" / "input"

# Output folder for visualization plots
OUTPUT_FOLDER = _MJOSA_COMPLETE_ROOT / "use_gref4hsi" / "057_final" / "output"

# Transect-specific folders (for SVM classification notebooks)
# Transect 057
TRANSECT_057_ROOT = _MJOSA_COMPLETE_ROOT / "use_gref4hsi" / "057_final"
TRANSECT_057_INPUT = TRANSECT_057_ROOT / "input"
TRANSECT_057_OUTPUT = TRANSECT_057_ROOT / "output"

# Transect 028
TRANSECT_028_ROOT = _MJOSA_COMPLETE_ROOT / "use_gref4hsi" / "028_final"
TRANSECT_028_INPUT = TRANSECT_028_ROOT / "input"
TRANSECT_028_OUTPUT = TRANSECT_028_ROOT / "output"

# ROIs and Models folders
ROI_FOLDER = _MJOSA_COMPLETE_ROOT / "data" / "anxilliary" / "ROIs"
SVM_MODELS_FOLDER = _MJOSA_COMPLETE_ROOT / "data" / "anxilliary" / "SVM_models"

# Camera calibration XML
CAMERA_CALIB_XML = _MJOSA_COMPLETE_ROOT / "data" / "anxilliary" / "HSI_2_body.xml"

# Optional: ROS database for DVL data (used by create_stability_plot.py)
DB_PATH = (
    _MJOSA_COMPLETE_ROOT / "data" / "raw" / "navigation" / "LOG_2024-10-29_10-13-32.db3"
)

# ==============================================================================
# COORDINATE SYSTEMS (for visualization scripts)
# ==============================================================================

# EPSG codes
EPSG_GEOGRAPHIC = 4326  # WGS84 lat/lon
EPSG_UTM = 32632  # UTM Zone 32N (for Mjøsa area)
EPSG_ECEF = 4978  # Earth-Centered Earth-Fixed
EPSG_MBES = 32632  # MBES GeoTIFF coordinate system (usually same as UTM)

# ==============================================================================
# SENSOR CONFIGURATION (for visualization scripts)
# ==============================================================================

# HSI camera rotation relative to vehicle body frame
# Body frame: +X forward, +Y starboard/right, +Z up
# Camera frame: +X across-track, +Y along-track, +Z down
ROTATION_HSI_TO_BODY = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
]

# Translation from body to HSI camera (meters)
TRANSLATION_BODY_TO_HSI = [0, -2.5, 0.0]

# Time synchronization offset (seconds)
TIME_OFFSET_SEC = 508

# Navigation yaw convention
YAW_CONVENTION = "heading_from_north_cw"

# ==============================================================================
# UHI-SPECIFIC CONFIGURATION (for create_stability_plot.py)
# ==============================================================================

# UHI alignment adjustment for matching with MBES data
UHI_ALIGNMENT_DX = -0.05  # East offset in meters
UHI_ALIGNMENT_DY = -3.0  # North offset in meters

# Default UHI files and track range for MBES detrending (file 5 only)
UHI_FILES = ["rad_uhi_20241029_115057_5"]
UHI_TRACK_RANGE = (576, 1566)  # Track range within file 5

# Alternative: Files 4+5 combined (if needed)
UHI_FILES_4_AND_5 = ["rad_uhi_20241029_115057_4", "rad_uhi_20241029_115057_5"]
UHI_TRACK_RANGE_4TO5 = (3039, 4029)  # Focus area spanning files 4+5

# Convenient aliases for backward compatibility
track_start, track_end = UHI_TRACK_RANGE

# ==============================================================================
# RAY TRACING PARAMETERS (for create_sim.py)
# ==============================================================================

MAX_RAY_LENGTH = 100  # meters (for deeper water)
EARLY_FAILURE_THRESHOLD = 50.0  # Cancel if >50% rays fail

# ==============================================================================
# LAKE MJØSA COORDINATE SYSTEM
# ==============================================================================

# Origin point for lat/lon to meters conversion
# This is the reference point for converting geographic coordinates to local metric coordinates
MJOSA_ORIGIN_LAT = 60.801146  # degrees North
MJOSA_ORIGIN_LON = 10.705125  # degrees East
MJOSA_ORIGIN = (MJOSA_ORIGIN_LAT, MJOSA_ORIGIN_LON)

# Aliases for visualization scripts (they expect LAT0/LON0)
LAT0 = MJOSA_ORIGIN_LAT
LON0 = MJOSA_ORIGIN_LON
H0 = 0.0

# Earth radius for coordinate conversion (WGS-84)
EARTH_RADIUS_M = 6_378_137.0  # meters

# ==============================================================================
# MISSION TIME RANGE
# ==============================================================================

# Mission date
MISSION_DATE = datetime.date(2024, 10, 29)

# Overall mission time range
MISSION_START_TIME = datetime.datetime(
    2024, 10, 29, 10, 10, 0, tzinfo=datetime.timezone.utc
)
MISSION_END_TIME = datetime.datetime(
    2024, 10, 29, 13, 50, 0, tzinfo=datetime.timezone.utc
)

# Comments:
# - 10:10 UTC: Robot starts to move
# - 13:50 UTC: Extended to include all transects (last one ends at 13:48:29)

# ==============================================================================
# TRANSECT TIME INTERVALS
# ==============================================================================

# IMU/Camera transect time intervals (HH:MM:SS format)
# These are the time segments where we want corrected navigation using dead reckoning
TRANSECT_TIME_INTERVALS = [
    ("10:50:51", "10:59:06"),  # Transect 1
    ("11:23:50", "11:25:04"),  # Transect 2
    ("11:26:32", "11:31:35"),  # Transect 3
    ("11:59:27", "12:08:17"),  # Transect 4
    ("12:25:09", "12:33:33"),  # Transect 5
    ("12:58:58", "13:07:20"),  # Transect 6
    ("13:10:05", "13:13:22"),  # Transect 7
    # ("13:43:34", "13:45:00"),  # Transect 8
    # ("13:45:18", "13:48:29"),  # Transect 9
]

# ==============================================================================
# DEAD RECKONING PARAMETERS
# ==============================================================================

# Constant velocity dead reckoning settings
CONSTANT_VELOCITY_M_S = (
    0.2  # meters per second (proven reliable for Eelume in this mission)
)
ADJUST_SPEED_TO_HIT_END = True  # Adjust speed to ensure DR ends at actual end position

# DVL (Doppler Velocity Log) settings
USE_DVL_VELOCITY = True  # Use DVL measurements instead of constant velocity
DVL_FALLBACK_TO_CONSTANT = (
    True  # Fall back to constant velocity if DVL data unavailable
)

# ==============================================================================
# CSV COLUMN NAMES
# ==============================================================================

# Standard column names used in navigation CSV files
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

# ==============================================================================
# EIVA OUTPUT FORMAT
# ==============================================================================

# EIVA expects this specific text format for navigation data
EIVA_HEADER = "Date     Time          Lat [deg]         Long [deg]        Depth [Meter]"
EIVA_DATE_FORMAT = "%Y%m%d"  # YYYYMMDD
EIVA_TIME_FORMAT = "%H%M%S.%f"  # HHMMSS.ssssss

# ==============================================================================
# PROCESSING OPTIONS
# ==============================================================================

# Logging
VERBOSE_LOGGING = True  # Print detailed progress messages
LOG_STEP_TIMING = True  # Log time taken for each processing step

# Data validation
VALIDATE_ALTITUDE = True  # Filter out rows with NaN altitude values
MIN_ALTITUDE_M = 0.0  # Minimum valid altitude in meters
MAX_ALTITUDE_M = 100.0  # Maximum valid altitude in meters (sanity check)

# Dead reckoning method selection
# Options: 'constant_velocity', 'dvl_velocity'
DEFAULT_DR_METHOD = "constant_velocity"  # Use proven constant velocity as default
ALTERNATIVE_DR_METHOD = "dvl_velocity"  # Alternative DVL-based method
