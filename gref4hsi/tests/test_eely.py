from collections import namedtuple
import configparser
import os
import sys
import argparse
import numpy as np
import pandas as pd

# Force reload of modules to get latest changes
import importlib
import gref4hsi.utils.uhi_parsing_utils as uhi_parsing_utils

importlib.reload(uhi_parsing_utils)

from gref4hsi.utils import parsing_utils
from gref4hsi.scripts import georeference, orthorectification
from gref4hsi.utils import visualize
from gref4hsi.utils.config_utils import (
    prepend_data_dir_to_relative_paths,
    customize_config,
)


def load_csv_navigation(csv_path):
    """
    Load navigation data from CSV file.
    Expected columns: timestamp, lat, lon, depth, roll, pitch, yaw, altitude
    """
    nav_data = pd.read_csv(csv_path)
    required_columns = [
        "timestamp [unix epoch s]",
        "latitude [deg]",
        "longitude [deg]",
        "depth [m]",
        "roll [deg]",
        "pitch [deg]",
        "yaw [deg]",
        "altitude [m]",
    ]

    for col in required_columns:
        if col not in nav_data.columns:
            print(
                f"Warning: Column '{col}' not found in CSV. Available columns: {list(nav_data.columns)}"
            )

    return nav_data


def main(args):
    # DATA_DIR is a string that points to the directory where the data is stored
    DATA_DIR = args.data_dir

    # The configuration file stores the settings for georeferencing
    config_file_mission = os.path.join(DATA_DIR, "configuration.ini")

    config_path_template = os.path.join(DATA_DIR, "configuration_uhi.ini")

    """
    EELY WORKFLOW MODIFICATION:
    
    Since you already have a CSV with IMU data (timestamp, lat, lon, depth, attitude, altitude),
    you can skip or modify certain steps:
    
    SKIP:
    - uhi_dbe() - This reads raw sensor data from H5 files, but you have processed CSV data
    
    KEEP:
    - Configuration setup
    - export_pose() - Still needed to transform poses to camera frame and save to H5
    - export_model() - If you want to create DEM from your altitude data
    - georeference.main() - Core georeferencing step
    - orthorectification.main() - Final processing step
    
    MODIFY:
    - Add CSV loading functionality
    - Update configuration to point to your CSV data
    - Potentially create a custom function to inject your CSV data into H5 files
    """

    # Load your CSV navigation data
    csv_nav_path = args.csv_nav_path
    nav_data = load_csv_navigation(csv_nav_path)
    print(f"Loaded {len(nav_data)} navigation records from CSV")

    # Copies the template "configuration_uhi.ini" to "config_file_mission" and sets up the necessary directories
    prepend_data_dir_to_relative_paths(
        config_path=config_path_template, DATA_DIR=DATA_DIR
    )

    # Custom configuration for EELY workflow
    custom_config = {
        "Orthorectification": {
            "resample_rgb_only": False,
            "resample_ancillary": False,
            "resolutionhyperspectralmosaic": args.resolution,
            "raster_transform_method": args.raster_transform,
            "mask_pixel_by_footprint": args.interpolation,
        },
        "HDF.raw_nav": {
            # MODIFY: Point to your CSV data source instead of H5 datasets
            "file_type": "csv",  # Custom flag to indicate CSV source
            "csv_path": csv_nav_path,  # Path to your CSV file
        },
        "HDF.hyperspectral": {"is_calibrated": False},
        "Absolute Paths": {
            "geoid_path": os.path.join(
                r"C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\gref4hsi\data\world\geoids\egm08_25.gtx"
            )
        },
        "Coordinate Reference Systems": {
            "proj_epsg": 32632,  # Update for your survey area
            "dem_epsg": 32632,
        },
    }

    # Customizes the config file with the dict. defined above
    customize_config(config_path=config_file_mission, dict_custom=custom_config)

    # EELY-specific preprocessing settings
    SettingsPreprocess = namedtuple(
        "SettingsPreprocessing",
        [
            "dtype_datacube",
            "rotation_matrix_hsi_to_body",
            "translation_body_to_hsi",
            "rotation_matrix_alt_to_body",
            "translation_alt_to_body",
            "config_file_name",
            "time_offset_sec",
            "lon_lat_alt_origin",
            "resolution_dem",
            "agisoft_process",
            "csv_nav_data",  # NEW: Add CSV data to settings
        ],
    )

    config_uhi_preprocess = SettingsPreprocess(
        dtype_datacube=np.float32,
        # Rotation matrices for DOWNWARD-LOOKING sensors (both HSI and altimeter)
        # This matches test_main_dbe.py configuration for nadir-looking setup
        # rz = -π/2: 90° rotation to align camera frame with body frame
        rotation_matrix_hsi_to_body=np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        translation_body_to_hsi=np.array([2.5, 0, 0]),
        # Altimeter has same orientation as HSI (both pointing down)
        rotation_matrix_alt_to_body=np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        translation_alt_to_body=np.array([0, 0, 0]),
        time_offset_sec=args.time_offset_sec,
        lon_lat_alt_origin=np.array([10.7122345, 60.8011575, 0]),
        config_file_name="configuration.ini",
        resolution_dem=args.dem_resolution,
        agisoft_process=False,
        csv_nav_data=nav_data,  # NEW: Pass CSV data to processing functions
    )

    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_file_mission)

    """
    EELY WORKFLOW DECISION POINT:
    
    Option 1: Skip uhi_dbe() entirely and create custom CSV-to-H5 injection function
    Option 2: Modify uhi_dbe() to accept CSV data instead of raw sensor data
    Option 3: Convert your CSV to the expected H5 format first
    
    For now, we'll try Option 1 - skip uhi_dbe() and go directly to export_pose()
    if your CSV data is already processed and synchronized.
    """

    print("\n \n Running uhi_eely() - processing CSV navigation data")
    uhi_parsing_utils.uhi_eely(config=config, config_uhi=config_uhi_preprocess)

    """
    KEEP THESE STEPS - They work with any navigation data format:
    """

    # Step 2: Export pose - transforms navigation to camera frame and saves to H5
    # This step should work regardless of whether nav data came from H5 or CSV
    config = parsing_utils.export_pose(config_file_mission)

    # Reload config after pose export
    config = configparser.ConfigParser()
    config.read(config_file_mission)

    # # Step 3: Export model - creates 3D mesh from altitude/depth data
    # # DECISION: Do you want to create DEM from your CSV altitude data?
    # if args.create_dem_from_csv:
    #     print("Creating DEM from CSV altitude data \n")
    #     parsing_utils.export_model(config_file_mission)
    # else:
    #     print("Skipping DEM creation - expecting external DEM file \n")
    #     # TODO: Point config to your existing DEM file if you have one

    parsing_utils.export_model(config_file_mission)

    # Step 4: Georeference - projects hyperspectral pixels to 3D coordinates
    # This step should work regardless of data source
    try:
        georeference.main(config_file_mission)
        print("\n✅ Georeferencing completed successfully!")
    except Exception as e:
        print(f"\n❌ Georeferencing failed with error: {e}")
        print(
            "Stopping pipeline - cannot proceed to orthorectification without valid georeferencing."
        )
        print("\nTroubleshooting tips:")
        print("  1. Check if DEM covers all HSI field of view areas")
        print("  2. Verify time_offset_sec is correct")
        print("  3. Check rotation matrices are correct for your sensor mounting")
        print(
            "  4. Consider processing H5 files separately if they cover different areas"
        )
        raise SystemExit(1)

    # Step 5: Orthorectification - creates final hyperspectral maps
    print("\n################ Starting Orthorectification ################")
    orthorectification.main(config_file_mission)
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--csv_nav_path",
        type=str,
    )
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--interpolation", type=bool, default=True)
    parser.add_argument("--raster_transform", type=str, default="north_east")
    parser.add_argument("--time_offset_sec", type=int, default=0)
    parser.add_argument("--dem_resolution", type=float, default=0.2)
    parser.add_argument("--create_dem_from_csv", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
