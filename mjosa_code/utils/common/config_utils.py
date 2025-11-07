"""
Configuration utilities for navigation processing.

Helper functions for working with config values - validation, path generation, etc.
"""

import datetime
from pathlib import Path
from typing import List, Tuple

from . import config


def parse_transect_intervals() -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Convert transect time strings to datetime objects.

    Returns:
        List of (start_datetime, end_datetime) tuples
    """
    intervals = []
    for start_str, end_str in config.TRANSECT_TIME_INTERVALS:
        start_dt = datetime.datetime.combine(
            config.MISSION_DATE,
            datetime.datetime.strptime(start_str, "%H:%M:%S").time(),
            tzinfo=datetime.timezone.utc,
        )
        end_dt = datetime.datetime.combine(
            config.MISSION_DATE,
            datetime.datetime.strptime(end_str, "%H:%M:%S").time(),
            tzinfo=datetime.timezone.utc,
        )
        intervals.append((start_dt, end_dt))
    return intervals


def get_transect_output_dir(transect_index: int) -> Path:
    """
    Get the output directory for a specific transect.

    Args:
        transect_index: Zero-based index of the transect

    Returns:
        Path object for the transect output directory
    """
    start_str, _ = config.TRANSECT_TIME_INTERVALS[transect_index]
    folder_name = start_str.replace(":", "")  # e.g., "105051"
    return config.PROCESSED_NAVIGATION_DIR / folder_name


def get_transect_csv_path(transect_index: int, suffix: str = "") -> Path:
    """
    Get the CSV file path for a specific transect.

    Args:
        transect_index: Zero-based index of the transect
        suffix: Optional suffix like "_dr" or "_dr_dvl_v2"

    Returns:
        Path object for the transect CSV file
    """
    start_str, _ = config.TRANSECT_TIME_INTERVALS[transect_index]
    folder_name = start_str.replace(":", "")
    filename = f"nav_data_{folder_name}{suffix}.csv"
    return config.PROCESSED_NAVIGATION_DIR / folder_name / filename


def create_output_directories():
    """
    Create all necessary output directories if they don't exist.
    """
    config.PROCESSED_NAVIGATION_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(len(config.TRANSECT_TIME_INTERVALS)):
        transect_dir = get_transect_output_dir(i)
        transect_dir.mkdir(parents=True, exist_ok=True)


def validate_config() -> bool:
    """
    Validate configuration settings and print warnings for potential issues.

    Returns:
        True if validation passed, False if issues found
    """
    issues = []

    # Check if input file directory exists
    if not config.LOG_DB3_FILE.parent.exists():
        issues.append(
            f"⚠️  Navigation directory does not exist: {config.LOG_DB3_FILE.parent}"
        )

    # Check if input file exists
    if not config.LOG_DB3_FILE.exists():
        issues.append(f"⚠️  Log file does not exist: {config.LOG_DB3_FILE}")

    # Check transect intervals
    if len(config.TRANSECT_TIME_INTERVALS) == 0:
        issues.append("⚠️  No transect intervals defined!")

    # Check time ordering
    transect_datetimes = parse_transect_intervals()
    for i, (start_dt, end_dt) in enumerate(transect_datetimes):
        if start_dt >= end_dt:
            issues.append(f"⚠️  Transect {i+1}: Start time is not before end time!")

        if start_dt < config.MISSION_START_TIME or end_dt > config.MISSION_END_TIME:
            issues.append(f"⚠️  Transect {i+1}: Outside mission time range!")

    if issues:
        print("❌ Configuration validation found issues:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ Configuration validation passed!")
        return True


def print_config_summary():
    """
    Print a summary of the current configuration.
    """
    print("=" * 70)
    print("MJØSA NAVIGATION PROCESSING CONFIGURATION")
    print("=" * 70)
    print(f"Mission Date:        {config.MISSION_DATE}")
    print(
        f"Mission Duration:    {config.MISSION_START_TIME.time()} - {config.MISSION_END_TIME.time()} UTC"
    )
    print(f"Number of Transects: {len(config.TRANSECT_TIME_INTERVALS)}")
    print(
        f"Origin Point:        ({config.MJOSA_ORIGIN_LAT:.7f}°N, {config.MJOSA_ORIGIN_LON:.7f}°E)"
    )
    print(f"Default DR Method:   {config.DEFAULT_DR_METHOD}")
    print(f"Constant Velocity:   {config.CONSTANT_VELOCITY_M_S} m/s")
    print(f"Use DVL Data:        {config.USE_DVL_VELOCITY}")
    print()
    print(f"Input Log File:      {config.LOG_DB3_FILE}")
    print(f"Output Main CSV:     {config.MAIN_CSV_FILE}")
    print(f"Output Merged CSV:   {config.MERGED_CSV_FILE}")
    print(f"Output EIVA TXT:     {config.EIVA_TXT_FILE}")
    print("=" * 70)
