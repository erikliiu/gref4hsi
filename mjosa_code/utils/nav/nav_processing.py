"""
Navigation Data Processing Functions for Lake Mjøsa UHI Mission
================================================================

This module provides high-level functions for processing Eelume robot navigation data:
1. Load .db3 ROS2 log file → CSV
2. Extract transect segments
3. Apply dead reckoning (constant velocity or DVL-based)
4. Merge corrected transects back into main CSV
5. Convert to EIVA format

Author: Auto-generated from original notebook
Date: 2025-11-07
"""

import sys
import os
from pathlib import Path
import datetime
import time

from ..common import config_utils
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

# Add external_libs to path for Eelume import
_current_dir = Path(__file__).resolve().parent
_mjosa_root = _current_dir.parent.parent
_external_libs = _mjosa_root / "external_libs" / "eelume_pypost"
sys.path.insert(0, str(_external_libs))

from Eelume import PyPost as pp

# Import config
from ..common import config


# ==============================================================================
# STEP 1: LOAD .DB3 FILE TO CSV
# ==============================================================================


def load_db3_to_csv(
    db3_file_path: str,
    output_csv_path: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    verbose: bool = True,
) -> Tuple[pp.DatabaseHandler, pp.MotionAnalyzer]:
    """
    Load ROS2 .db3 log file and create aligned CSV with 6DOF navigation data.

    This function:
    1. Opens the .db3 database file
    2. Creates a MotionAnalyzer for the specified time range
    3. Extracts aligned 6DOF pose data (lat, lon, depth, roll, pitch, yaw, altitude)
    4. Saves to CSV file

    Args:
        db3_file_path: Path to ROS2 .db3 log file
        output_csv_path: Path for output CSV file
        start_time: Start time for data extraction (datetime with timezone)
        end_time: End time for data extraction (datetime with timezone)
        verbose: Print progress messages

    Returns:
        Tuple of (DatabaseHandler, MotionAnalyzer) for further processing

    Example:
        >>> db, analyzer = load_db3_to_csv(
        ...     "LOG_2024-10-29_10-13-32.db3",
        ...     "nav_data.csv",
        ...     start_time=datetime.datetime(2024, 10, 29, 10, 10, 0, tzinfo=datetime.timezone.utc),
        ...     end_time=datetime.datetime(2024, 10, 29, 13, 15, 0, tzinfo=datetime.timezone.utc)
        ... )
    """
    if verbose:
        print("=" * 70)
        print("STEP 1: Loading .db3 file to CSV")
        print("=" * 70)
        print(f"Input:  {db3_file_path}")
        print(f"Output: {output_csv_path}")
        print(f"Time:   {start_time} to {end_time}")
        print(f"Duration: {(end_time - start_time).total_seconds():.1f} seconds")

    t_start = time.time()

    # Open database
    if verbose:
        print("\n📂 Opening database...")
    db = pp.DatabaseHandler(db3_file_path)

    # Create motion analyzer
    if verbose:
        print("📊 Creating motion analyzer...")
    analyzer = pp.MotionAnalyzer(db, start_time=start_time, end_time=end_time)

    # Create aligned CSV file
    if verbose:
        print("💾 Extracting and aligning 6DOF data...")

    # Ensure output directory exists
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analyzer.create_aligned_csv_file(str(output_csv_path))

    t_end = time.time()

    if verbose:
        print(f"✅ CSV created successfully!")
        print(f"⏱️  Time elapsed: {t_end - t_start:.2f} seconds")

        # Quick stats
        df = pd.read_csv(output_csv_path)
        print(f"📊 Data points: {len(df)}")
        print(
            f"   Lat range: {df['latitude [deg]'].min():.6f} to {df['latitude [deg]'].max():.6f}"
        )
        print(
            f"   Lon range: {df['longitude [deg]'].min():.6f} to {df['longitude [deg]'].max():.6f}"
        )
        print(
            f"   Depth range: {df['depth [m]'].min():.2f} to {df['depth [m]'].max():.2f} m"
        )

    return db, analyzer


# ==============================================================================
# STEP 2: EXTRACT TRANSECT SEGMENTS
# ==============================================================================


def extract_transect_segments(
    main_csv_path: str,
    transect_intervals: List[Tuple[str, str]],
    output_base_dir: str,
    analyzer: pp.MotionAnalyzer,
    verbose: bool = True,
) -> List[str]:
    """
    Extract individual transect segments from main CSV file.

    Uses the Eelume MotionAnalyzer's get_individual_transect_csvs method to
    extract time-specific segments from the main navigation CSV.

    Args:
        main_csv_path: Path to main navigation CSV file
        transect_intervals: List of (start_time_str, end_time_str) tuples in "HH:MM:SS" format
        output_base_dir: Base directory for transect output files
        analyzer: Eelume MotionAnalyzer object
        verbose: Print progress messages

    Returns:
        List of paths to extracted transect CSV files

    Example:
        >>> transect_files = extract_transect_segments(
        ...     "nav_data.csv",
        ...     [("10:50:51", "10:59:06"), ("11:23:50", "11:25:04")],
        ...     "transects/",
        ...     analyzer
        ... )
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: Extracting Transect Segments")
        print("=" * 70)
        print(f"Input:  {main_csv_path}")
        print(f"Output: {output_base_dir}")
        print(f"Number of transects: {len(transect_intervals)}")

    t_start = time.time()

    # Ensure output directory exists
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

    # Use Eelume's built-in method
    if verbose:
        print("\n📂 Extracting transect CSV files...")

    analyzer.get_individual_transect_csvs(
        main_csv_path, output_base_dir, transect_intervals
    )

    # Build list of output files
    transect_files = []
    for start_str, _ in transect_intervals:
        folder_name = start_str.replace(":", "")
        csv_filename = f"nav_data_{folder_name}.csv"
        csv_path = Path(output_base_dir) / folder_name / csv_filename

        if csv_path.exists():
            transect_files.append(str(csv_path))
            if verbose:
                df = pd.read_csv(csv_path)
                print(f"   ✅ {folder_name}: {len(df)} points")
        else:
            if verbose:
                print(f"   ❌ {folder_name}: File not created!")

    t_end = time.time()

    if verbose:
        print(
            f"\n✅ Extracted {len(transect_files)}/{len(transect_intervals)} transects"
        )
        print(f"⏱️  Time elapsed: {t_end - t_start:.2f} seconds")

    return transect_files


# ==============================================================================
# STEP 3A: CONSTANT VELOCITY DEAD RECKONING
# ==============================================================================


def apply_constant_velocity_dr(
    transect_csv_files: List[str],
    analyzer: pp.MotionAnalyzer,
    output_suffix: str = "_dr",
    adjust_speed: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    Apply constant velocity dead reckoning to transect CSV files.

    This method:
    1. Calculates constant bearing from start to end position
    2. Uses constant speed (default 0.2 m/s)
    3. Optionally adjusts speed to hit end position exactly
    4. Creates new CSV files with dead-reckoned positions

    Args:
        transect_csv_files: List of paths to transect CSV files
        analyzer: Eelume MotionAnalyzer object
        output_suffix: Suffix for output files (default "_dr")
        adjust_speed: Adjust speed to ensure end position matches (default True)
        verbose: Print progress messages

    Returns:
        List of paths to dead-reckoned CSV files

    Example:
        >>> dr_files = apply_constant_velocity_dr(
        ...     ["transect1.csv", "transect2.csv"],
        ...     analyzer,
        ...     output_suffix="_dr"
        ... )
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3A: Constant Velocity Dead Reckoning")
        print("=" * 70)
        print(f"Method: Constant bearing from start→end, constant speed")
        print(f"Speed:  {config.CONSTANT_VELOCITY_M_S} m/s")
        print(f"Adjust: {adjust_speed}")
        print(f"Files:  {len(transect_csv_files)}")

    t_start = time.time()
    dr_files = []

    for i, csv_file in enumerate(transect_csv_files, 1):
        if not os.path.exists(csv_file):
            if verbose:
                print(f"   ❌ {i}. File not found: {os.path.basename(csv_file)}")
            continue

        if verbose:
            print(f"\n📁 {i}. Processing: {os.path.basename(csv_file)}")

        try:
            dr_path = analyzer.create_constant_body_vel_dead_reckoned_csv(
                csv_file,
                output_suffix=output_suffix,
                adjust_speed_to_hit_end=adjust_speed,
            )

            if dr_path and os.path.exists(dr_path):
                dr_files.append(dr_path)
                if verbose:
                    df_orig = pd.read_csv(csv_file)
                    df_dr = pd.read_csv(dr_path)
                    print(f"   ✅ Created: {os.path.basename(dr_path)}")
                    print(f"      Points: {len(df_orig)} → {len(df_dr)}")
            else:
                if verbose:
                    print(f"   ❌ Failed to create DR file")

        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")

    t_end = time.time()

    if verbose:
        print(f"\n✅ Completed: {len(dr_files)}/{len(transect_csv_files)} files")
        print(f"⏱️  Time elapsed: {t_end - t_start:.2f} seconds")

    return dr_files


# ==============================================================================
# STEP 3B: DVL VELOCITY DEAD RECKONING
# ==============================================================================


def apply_dvl_velocity_dr(
    transect_csv_files: List[str],
    analyzer: pp.MotionAnalyzer,
    output_suffix: str = "_dr_dvl_v2",
    adjust_speed: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Apply DVL velocity dead reckoning to transect CSV files.

    This method:
    1. Calculates constant bearing from start to end position (like constant velocity)
    2. Uses actual DVL velocity measurements instead of constant 0.2 m/s
    3. Falls back to constant velocity if DVL data unavailable
    4. Creates new CSV files with dead-reckoned positions

    Args:
        transect_csv_files: List of paths to transect CSV files
        analyzer: Eelume MotionAnalyzer object
        output_suffix: Suffix for output files (default "_dr_dvl_v2")
        adjust_speed: Adjust speed to ensure end position matches (default False)
        verbose: Print progress messages

    Returns:
        List of paths to DVL dead-reckoned CSV files

    Example:
        >>> dvl_dr_files = apply_dvl_velocity_dr(
        ...     ["transect1.csv", "transect2.csv"],
        ...     analyzer,
        ...     output_suffix="_dr_dvl"
        ... )
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3B: DVL Velocity Dead Reckoning")
        print("=" * 70)
        print(f"Method: Constant bearing from start→end, actual DVL speeds")
        print(
            f"Fallback: Constant {config.CONSTANT_VELOCITY_M_S} m/s if DVL unavailable"
        )
        print(f"Adjust: {adjust_speed}")
        print(f"Files:  {len(transect_csv_files)}")

    t_start = time.time()
    dr_files = []

    for i, csv_file in enumerate(transect_csv_files, 1):
        if not os.path.exists(csv_file):
            if verbose:
                print(f"   ❌ {i}. File not found: {os.path.basename(csv_file)}")
            continue

        if verbose:
            print(f"\n📁 {i}. Processing: {os.path.basename(csv_file)}")

        try:
            t_file_start = time.time()

            dr_path = analyzer.create_dvl_velocity_dead_reckoned_csv_v2(
                csv_file,
                output_suffix=output_suffix,
                adjust_speed_to_hit_end=adjust_speed,
            )

            t_file_end = time.time()

            if dr_path and os.path.exists(dr_path):
                dr_files.append(dr_path)
                if verbose:
                    df_dr = pd.read_csv(dr_path)
                    print(
                        f"   ✅ Created: {os.path.basename(dr_path)} ({t_file_end - t_file_start:.2f}s)"
                    )

                    if "dvl_speed [m/s]" in df_dr.columns:
                        avg_speed = df_dr["dvl_speed [m/s]"].mean()
                        print(
                            f"      Avg DVL speed: {avg_speed:.3f} m/s (vs {config.CONSTANT_VELOCITY_M_S} m/s constant)"
                        )

                    if "dr_bearing [deg]" in df_dr.columns:
                        bearing = df_dr["dr_bearing [deg]"].iloc[0]
                        print(f"      Bearing: {bearing:.1f}°")
            else:
                if verbose:
                    print(f"   ❌ Failed to create DVL DR file")

        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")

    t_end = time.time()

    if verbose:
        print(f"\n✅ Completed: {len(dr_files)}/{len(transect_csv_files)} files")
        print(f"⏱️  Time elapsed: {t_end - t_start:.2f} seconds")

    return dr_files


# ==============================================================================
# STEP 4: MERGE TRANSECTS
# ==============================================================================


def merge_transect_fixes(
    main_csv_path: str,
    transect_dr_paths: List[str],
    output_csv_path: str,
    timestamp_col: str = "timestamp [unix epoch s]",
    verbose: bool = True,
) -> str:
    """
    Merge corrected transect CSV files back into main navigation file.

    This function:
    1. Loads main navigation CSV
    2. For each corrected transect: removes overlapping timestamps from main
    3. Appends corrected transect data
    4. Sorts by timestamp
    5. Saves merged result

    Args:
        main_csv_path: Path to original full-track CSV
        transect_dr_paths: List of paths to corrected "_dr.csv" transect files
        output_csv_path: Path for merged output CSV
        timestamp_col: Name of timestamp column (default "timestamp [unix epoch s]")
        verbose: Print progress messages

    Returns:
        Path to merged CSV file

    Example:
        >>> merged_path = merge_transect_fixes(
        ...     "nav_data.csv",
        ...     ["transect1_dr.csv", "transect2_dr.csv"],
        ...     "nav_data_merged.csv"
        ... )
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 4: Merging Corrected Transects")
        print("=" * 70)
        print(f"Main file:     {main_csv_path}")
        print(f"Transect DRs:  {len(transect_dr_paths)}")
        print(f"Output:        {output_csv_path}")

    t_start = time.time()

    # Load main track
    if verbose:
        print("\n📂 Loading main navigation file...")
    main_df = pd.read_csv(main_csv_path)
    original_count = len(main_df)

    if timestamp_col not in main_df.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_col}' not found in {main_csv_path!r}"
        )

    if verbose:
        print(f"   Original: {original_count} data points")

    # For each corrected transect: drop overlapping rows, then append
    replaced_count = 0
    for i, dr_path in enumerate(transect_dr_paths, 1):
        if not os.path.exists(dr_path):
            if verbose:
                print(f"   ⚠️  {i}. File not found: {os.path.basename(dr_path)}")
            continue

        dr_df = pd.read_csv(dr_path)

        if timestamp_col not in dr_df.columns:
            if verbose:
                print(
                    f"   ⚠️  {i}. Timestamp column not found in: {os.path.basename(dr_path)}"
                )
            continue

        # Remove any rows in main_df that share timestamps with dr_df
        ts_to_replace = set(dr_df[timestamp_col])
        before_len = len(main_df)
        main_df = main_df[~main_df[timestamp_col].isin(ts_to_replace)]
        after_len = len(main_df)
        removed = before_len - after_len
        replaced_count += removed

        # Append the corrected transect rows
        main_df = pd.concat([main_df, dr_df], ignore_index=True)

        if verbose:
            print(
                f"   ✅ {i}. {os.path.basename(dr_path)}: Replaced {removed} points, added {len(dr_df)}"
            )

    # Sort by timestamp
    if verbose:
        print("\n🔄 Sorting by timestamp...")
    main_df.sort_values(by=timestamp_col, inplace=True)

    # Ensure output directory exists
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    if verbose:
        print("💾 Writing merged CSV...")
    main_df.to_csv(output_csv_path, index=False)

    t_end = time.time()

    if verbose:
        print(f"\n✅ Merged CSV created successfully!")
        print(f"   Original:  {original_count} points")
        print(f"   Replaced:  {replaced_count} points")
        print(f"   Final:     {len(main_df)} points")
        print(f"⏱️  Time elapsed: {t_end - t_start:.2f} seconds")

    return output_csv_path


# ==============================================================================
# STEP 5: CONVERT TO EIVA FORMAT
# ==============================================================================


def convert_to_eiva_format(
    csv_path: str, output_txt_path: str, verbose: bool = True
) -> str:
    """
    Convert navigation CSV to EIVA-compatible text format.

    EIVA format:
    Date     Time          Lat [deg]         Long [deg]        Depth [Meter]
    20241029 101000.123456 60.80115750000000 10.71223450000000 5.1234567890

    Args:
        csv_path: Path to input CSV file
        output_txt_path: Path for output TXT file
        verbose: Print progress messages

    Returns:
        Path to EIVA-formatted text file

    Example:
        >>> eiva_path = convert_to_eiva_format(
        ...     "nav_data_merged.csv",
        ...     "nav_data_for_EIVA.txt"
        ... )
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 5: Converting to EIVA Format")
        print("=" * 70)
        print(f"Input:  {csv_path}")
        print(f"Output: {output_txt_path}")

    t_start = time.time()

    # Read CSV file
    if verbose:
        print("\n📂 Reading CSV file...")
    df = pd.read_csv(csv_path)

    # Convert unix timestamp to datetime
    if verbose:
        print("🔄 Converting timestamps...")
    df["datetime"] = pd.to_datetime(df["timestamp [unix epoch s]"], unit="s")

    # Format output lines
    if verbose:
        print("📝 Formatting EIVA output...")
    output_lines = []

    # Header with exact spacing
    output_lines.append(config.EIVA_HEADER)

    # Format each data row
    for _, row in df.iterrows():
        date_str = row["datetime"].strftime(config.EIVA_DATE_FORMAT)
        time_str = row["datetime"].strftime(config.EIVA_TIME_FORMAT)
        lat_str = f"{row['latitude [deg]']:.14f}"
        lon_str = f"{row['longitude [deg]']:.14f}"
        depth_str = f"{row['depth [m]']:.10f}"

        line = f"{date_str} {time_str} {lat_str} {lon_str} {depth_str}"
        output_lines.append(line)

    # Ensure output directory exists
    output_path = Path(output_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    if verbose:
        print("💾 Writing EIVA text file...")
    with open(output_txt_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    t_end = time.time()

    if verbose:
        print(f"\n✅ EIVA file created successfully!")
        print(f"   Data points: {len(df)}")
        print(f"   Output file: {output_txt_path}")
        print(f"⏱️  Time elapsed: {t_end - t_start:.2f} seconds")
        print("\nFirst few lines of output:")
        for i, line in enumerate(output_lines[:4]):
            if i == 0:
                print(f"   {line}")  # Header
            else:
                print(f"   {line}")

    return output_txt_path


# ==============================================================================
# COMPLETE PIPELINE
# ==============================================================================


def run_complete_pipeline(
    db3_file: Optional[str] = None, use_dvl: bool = False, verbose: bool = True
) -> dict:
    """
    Run the complete navigation processing pipeline.

    This function orchestrates all steps:
    1. Load .db3 → CSV
    2. Extract transect segments
    3. Apply dead reckoning (constant velocity or DVL)
    4. Merge corrected transects
    5. Convert to EIVA format

    Args:
        db3_file: Path to .db3 file (None = use config default)
        use_dvl: Use DVL velocity instead of constant velocity (default False)
        verbose: Print detailed progress messages (default True)

    Returns:
        Dictionary with paths to all output files:
        {
            'main_csv': path to main CSV,
            'transect_csvs': list of transect CSV paths,
            'dr_csvs': list of dead-reckoned CSV paths,
            'merged_csv': path to merged CSV,
            'eiva_txt': path to EIVA text file
        }

    Example:
        >>> results = run_complete_pipeline(use_dvl=False, verbose=True)
        >>> print(f"Final merged CSV: {results['merged_csv']}")
        >>> print(f"EIVA format file: {results['eiva_txt']}")
    """
    if verbose:
        print("\n" + "=" * 80)
        print(" " * 20 + "MJØSA NAVIGATION PROCESSING PIPELINE")
        print("=" * 80)
        config_utils.print_config_summary()
        print("\n🚀 Starting complete processing pipeline...\n")

    pipeline_start = time.time()

    # Use config defaults if not provided
    if db3_file is None:
        db3_file = str(config.LOG_DB3_FILE)

    main_csv = str(config.MAIN_CSV_FILE)

    # Use different output filenames for DVL method
    if use_dvl:
        merged_csv = str(config.MERGED_CSV_FILE).replace(".csv", "_dvl.csv")
        eiva_txt = str(config.EIVA_TXT_FILE).replace(".txt", "_dvl.txt")
    else:
        merged_csv = str(config.MERGED_CSV_FILE)
        eiva_txt = str(config.EIVA_TXT_FILE)

    transect_base_dir = str(config.PROCESSED_NAVIGATION_DIR)

    # Create output directories
    config_utils.create_output_directories()

    try:
        # Step 1: Load .db3 to CSV
        db, analyzer = load_db3_to_csv(
            db3_file,
            main_csv,
            config.MISSION_START_TIME,
            config.MISSION_END_TIME,
            verbose=verbose,
        )

        # Step 2: Extract transect segments
        transect_csvs = extract_transect_segments(
            main_csv,
            config.TRANSECT_TIME_INTERVALS,
            transect_base_dir,
            analyzer,
            verbose=verbose,
        )

        # Step 3: Apply dead reckoning
        if use_dvl:
            dr_csvs = apply_dvl_velocity_dr(
                transect_csvs,
                analyzer,
                output_suffix="_dr_dvl_v2",
                adjust_speed=False,
                verbose=verbose,
            )
        else:
            dr_csvs = apply_constant_velocity_dr(
                transect_csvs,
                analyzer,
                output_suffix="_dr",
                adjust_speed=config.ADJUST_SPEED_TO_HIT_END,
                verbose=verbose,
            )

        # Step 4: Merge corrected transects
        merge_transect_fixes(main_csv, dr_csvs, merged_csv, verbose=verbose)

        # Step 5: Convert to EIVA format
        convert_to_eiva_format(merged_csv, eiva_txt, verbose=verbose)

        pipeline_end = time.time()

        if verbose:
            print("\n" + "=" * 80)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"⏱️  Total time: {pipeline_end - pipeline_start:.2f} seconds")
            print("\n📁 Output files:")
            print(f"   Main CSV:    {main_csv}")
            print(f"   Merged CSV:  {merged_csv}")
            print(f"   EIVA TXT:    {eiva_txt}")
            print(f"   Transects:   {len(transect_csvs)} files")
            print(f"   DR files:    {len(dr_csvs)} files")
            print("=" * 80)

        return {
            "main_csv": main_csv,
            "transect_csvs": transect_csvs,
            "dr_csvs": dr_csvs,
            "merged_csv": merged_csv,
            "eiva_txt": eiva_txt,
        }

    except Exception as e:
        if verbose:
            print(f"\n❌ Pipeline failed with error: {e}")
            import traceback

            print(traceback.format_exc())
        raise


# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================


def test_load_db3():
    """Test loading .db3 file to CSV."""
    print("\n🧪 Testing: load_db3_to_csv()")
    db, analyzer = load_db3_to_csv(
        str(config.LOG_DB3_FILE),
        str(config.MAIN_CSV_FILE),
        config.MISSION_START_TIME,
        config.MISSION_END_TIME,
        verbose=True,
    )
    assert os.path.exists(config.MAIN_CSV_FILE), "Main CSV not created!"
    print("✅ Test passed: load_db3_to_csv")
    return db, analyzer


def test_extract_transects(analyzer):
    """Test extracting transect segments."""
    print("\n🧪 Testing: extract_transect_segments()")
    transect_csvs = extract_transect_segments(
        str(config.MAIN_CSV_FILE),
        config.TRANSECT_TIME_INTERVALS,
        str(config.PROCESSED_NAVIGATION_DIR),
        analyzer,
        verbose=True,
    )
    assert len(transect_csvs) > 0, "No transects extracted!"
    print(f"✅ Test passed: extract_transect_segments ({len(transect_csvs)} files)")
    return transect_csvs


def test_constant_velocity_dr(transect_csvs, analyzer):
    """Test constant velocity dead reckoning."""
    print("\n🧪 Testing: apply_constant_velocity_dr()")
    dr_csvs = apply_constant_velocity_dr(
        transect_csvs[:2],  # Test with first 2 transects
        analyzer,
        output_suffix="_dr_test",
        adjust_speed=True,
        verbose=True,
    )
    assert len(dr_csvs) > 0, "No DR files created!"
    print(f"✅ Test passed: apply_constant_velocity_dr ({len(dr_csvs)} files)")
    return dr_csvs


if __name__ == "__main__":
    # When run directly, execute the complete pipeline
    print("Running navigation processing pipeline...")
    results = run_complete_pipeline(
        use_dvl=False, verbose=True  # Use constant velocity by default
    )
    print("\nDone!")
