"""
Plotting utilities for navigation data visualization and validation.

This module provides functions for:
- Plotting transect overlays on main navigation data
- 3D trajectory visualization
- DVL velocity plots
- Dead reckoning accuracy comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
from datetime import datetime

from .analyze_log_file import LogData


def deg_to_meter_simple(latitudes, longitudes, origin=None):
    """
    Convert lat/lon → metres using simple approximation.

    Parameters:
    -----------
    latitudes : array-like
        Latitude values in degrees
    longitudes : array-like
        Longitude values in degrees
    origin : tuple(lat, lon), optional
        Reference point. If None, uses first point.

    Returns:
    --------
    east, north : tuple of arrays
        Coordinates in meters
    """
    lat0, lon0 = (latitudes[0], longitudes[0]) if origin is None else origin
    R = 6_378_137.0  # WGS-84 equatorial radius [m]

    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    east = (lon_rad - lon0_rad) * R * np.cos(lat0_rad)
    north = (lat_rad - lat0_rad) * R
    return east, north


def plot_transects_comparison(
    main_csv: str,
    transect_csv_paths: List[str],
    transect_labels: Optional[List[str]] = None,
    use_meters: bool = False,
    origin: Optional[Tuple[float, float]] = None,
):
    """
    Plot main navigation data with overlaid transect segments.

    Parameters:
    -----------
    main_csv : str
        Path to main navigation CSV file
    transect_csv_paths : list
        List of paths to transect CSV files to overlay
    transect_labels : list, optional
        Custom labels for each transect
    use_meters : bool
        If True, converts lat/lon to meters
    origin : tuple(lat, lon), optional
        Reference point for meter conversion (required if use_meters=True)
    """
    # Load main data
    log_data = LogData(main_csv)

    # Convert spatial coords for main data
    if use_meters:
        if origin is None:
            raise ValueError("origin must be provided when use_meters=True")
        east, north = deg_to_meter_simple(
            log_data.latitude, log_data.longitude, origin=origin
        )
        x_coords, y_coords = east, north
        x_label, y_label = "East [m]", "North [m]"
        x3_label, y3_label = "North [m]", "East [m]"
    else:
        x_coords, y_coords = log_data.longitude, log_data.latitude
        x_label, y_label = "Lon [deg]", "Lat [deg]"
        x3_label, y3_label = "Lat [deg]", "Lon [deg]"

    # Load transect data
    transects = []
    labels = []
    colors = [
        "red",
        "orange",
        "green",
        "blue",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    for i, csv_path in enumerate(transect_csv_paths):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.rename(
                columns={
                    "timestamp [unix epoch s]": "timestamp",
                    "latitude [deg]": "latitude",
                    "longitude [deg]": "longitude",
                    "depth [m]": "depth",
                    "roll [deg]": "roll",
                    "pitch [deg]": "pitch",
                    "yaw [deg]": "yaw",
                    "altitude [m]": "altitude",
                }
            )
            df = df[df["altitude"].notna()]

            # Convert transect coordinates
            if use_meters:
                t_east, t_north = deg_to_meter_simple(
                    df["latitude"].values, df["longitude"].values, origin=origin
                )
                df["x_coords"] = t_east
                df["y_coords"] = t_north
                df["x3_coords"] = t_north
                df["y3_coords"] = t_east
            else:
                df["x_coords"] = df["longitude"]
                df["y_coords"] = df["latitude"]
                df["x3_coords"] = df["latitude"]
                df["y3_coords"] = df["longitude"]

            transects.append(df)

            # Create label
            if transect_labels and i < len(transect_labels):
                labels.append(transect_labels[i])
            else:
                filename = os.path.basename(csv_path)
                labels.append(filename.replace(".csv", ""))
        else:
            print(f"Warning: {csv_path} not found")

    if not transects:
        print("No valid transect files found.")
        return

    # 1. Position scatter with transects overlaid
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(x_coords, y_coords, s=5, alpha=0.7, color="lightblue", label="All data")

    for i, (transect, label) in enumerate(zip(transects, labels)):
        color = colors[i % len(colors)]
        ax.scatter(
            transect["x_coords"],
            transect["y_coords"],
            s=3,
            alpha=0.8,
            color=color,
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Position")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    plt.show()

    # 2. 3D trajectory
    min_length = min(len(y_coords), len(x_coords), len(log_data.depth))
    main_x3 = y_coords[:min_length]
    main_y3 = x_coords[:min_length]
    main_depth = log_data.depth[:min_length]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        main_x3,
        main_y3,
        main_depth,
        color="lightblue",
        alpha=0.5,
        linewidth=2,
        label="All data",
    )

    for i, (transect, label) in enumerate(zip(transects, labels)):
        color = colors[i % len(colors)]
        ax.plot(
            transect["x3_coords"],
            transect["y3_coords"],
            transect["depth"],
            color=color,
            linewidth=3,
            alpha=0.9,
            label=label,
        )

    ax.set_xlabel(x3_label)
    ax.set_ylabel(y3_label)
    ax.set_zlabel("Depth [m]")
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.set_title("3D Trajectory")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    plt.show()


def calculate_endpoint_errors(
    transect_csvs: List[str],
    dr_csvs_const: List[str],
    dr_csvs_dvl: List[str],
    origin: Tuple[float, float],
    constant_velocity: float,
) -> None:
    """
    Calculate and print endpoint errors for dead reckoning methods.

    Parameters:
    -----------
    transect_csvs : list
        Original transect CSV files (real GNSS)
    dr_csvs_const : list
        Constant velocity DR files
    dr_csvs_dvl : list
        DVL velocity DR files
    origin : tuple(lat, lon)
        Reference point for meter conversion
    constant_velocity : float
        Constant velocity used in m/s
    """
    print("\n🔍 Detailed comparison of dead reckoning accuracy...")
    print(
        "   Comparing end positions: how close does each DR method get to real GNSS end position?"
    )
    print()

    def calc_error(original_csv, dr_csv):
        """Helper to calculate endpoint error"""
        try:
            df_orig = pd.read_csv(original_csv)
            df_dr = pd.read_csv(dr_csv)

            # Get final positions
            orig_lat, orig_lon = (
                df_orig["latitude [deg]"].iloc[-1],
                df_orig["longitude [deg]"].iloc[-1],
            )
            dr_lat, dr_lon = (
                df_dr["latitude [deg]"].iloc[-1],
                df_dr["longitude [deg]"].iloc[-1],
            )

            # Convert to meters
            orig_east, orig_north = deg_to_meter_simple(
                [orig_lat], [orig_lon], origin=origin
            )
            dr_east, dr_north = deg_to_meter_simple([dr_lat], [dr_lon], origin=origin)

            # Calculate distance error
            error_m = np.sqrt(
                (dr_east[0] - orig_east[0]) ** 2 + (dr_north[0] - orig_north[0]) ** 2
            )
            return error_m
        except Exception as e:
            print(f"Error: {e}")
            return None

    print("📊 Endpoint accuracy comparison:")
    print("   (How far does each DR method end up from the real GNSS endpoint?)")
    print()

    for i, (orig_file, dr_const_file, dr_dvl_file) in enumerate(
        zip(transect_csvs, dr_csvs_const, dr_csvs_dvl)
    ):
        transect_name = (
            os.path.basename(orig_file).replace("nav_data_", "").replace(".csv", "")
        )
        print(f"🎯 Transect {transect_name}:")

        # Constant velocity DR
        if os.path.exists(dr_const_file):
            error = calc_error(orig_file, dr_const_file)
            if error is not None:
                print(f"   🟠 Constant velocity DR error: {error:.1f} m")
        else:
            print(f"   🟠 Constant velocity DR file missing")

        # DVL DR
        if os.path.exists(dr_dvl_file):
            error = calc_error(orig_file, dr_dvl_file)
            if error is not None:
                print(f"   🟢 DVL DR error: {error:.1f} m")

                # Load DVL stats
                df_dvl = pd.read_csv(dr_dvl_file)
                if "dvl_speed [m/s]" in df_dvl.columns:
                    avg_speed = df_dvl["dvl_speed [m/s]"].mean()
                    print(
                        f"      📊 DVL avg speed: {avg_speed:.3f} m/s (vs {constant_velocity:.1f} m/s constant)"
                    )
        else:
            print(f"   🟢 DVL DR file missing")
        print()

    print("💡 Interpretation:")
    print("   • Lower error = more accurate dead reckoning")
    print(
        "   • DVL method should be more accurate if vehicle speed varied significantly from constant velocity"
    )
    print("   • Constant velocity method is proven and reliable baseline!")


def plot_all_methods_comparison(
    main_csv: str,
    transect_csvs: List[str],
    dr_csvs: List[str],
    dr_csvs_dvl: List[str],
    transect_intervals: List[Tuple[datetime, datetime]],
    origin: Tuple[float, float],
) -> None:
    """
    Compare all three approaches: original GNSS, constant velocity DR, and DVL DR.
    Includes 2D position plot and DOF time series plots.

    Parameters:
    -----------
    main_csv : str
        Path to main navigation CSV
    transect_csvs : list
        Original transect CSV files (GNSS)
    dr_csvs : list
        Constant velocity DR files
    dr_csvs_dvl : list
        DVL velocity DR files
    transect_intervals : list of tuples
        List of (start_datetime, end_datetime) tuples
    origin : tuple(lat, lon)
        Reference point for meter conversion
    """
    print("\n📊 Now let's compare all three approaches:")
    print("   🔵 Original: Simple extraction from main CSV (real GNSS positions)")
    print("   🟠 _dr: Constant velocity dead reckoning")
    print("   🟢 _dr_dvl: DVL velocity dead reckoning")
    print()

    # Combine all files
    all_comparison_files = []
    all_comparison_files.extend(transect_csvs)  # Original GNSS
    all_comparison_files.extend(dr_csvs)  # Constant velocity DR
    all_comparison_files.extend(dr_csvs_dvl)  # DVL DR

    # Create labels from timestamps
    comparison_labels = []
    # Handle both string tuples and datetime tuples
    transect_times = []
    for start, end in transect_intervals:
        if isinstance(start, str):
            # Already a string like "10:50:51", extract HHMMSS
            transect_times.append(start.replace(":", ""))
        else:
            # datetime object, format it
            transect_times.append(start.strftime("%H%M%S"))

    for t_time in transect_times:
        comparison_labels.append(f"{t_time} (GNSS)")
    for t_time in transect_times:
        comparison_labels.append(f"{t_time} (DR_const)")
    for t_time in transect_times:
        comparison_labels.append(f"{t_time} (DR_DVL)")

    # Filter to existing files
    existing_files = []
    existing_labels = []
    for file_path, label in zip(all_comparison_files, comparison_labels):
        if os.path.exists(file_path):
            existing_files.append(file_path)
            existing_labels.append(label)
        else:
            print(f"⚠️  Missing: {label}")

    print(f"\n🎯 Plotting {len(existing_files)} files for comparison...")

    # 1. Plot 2D position comparison
    if existing_files:
        plot_transects_comparison(
            main_csv,
            existing_files,
            transect_labels=existing_labels,
            use_meters=True,
            origin=origin,
        )
    else:
        print("❌ No files available for comparison!")
        return

    # 2. Plot DOF time series (lat, lon, depth, roll, pitch, yaw)
    print("\n📊 Plotting DOF time series comparisons...")

    # Load main CSV for background
    main_df = pd.read_csv(main_csv)
    main_df = main_df.rename(
        columns={
            "timestamp [unix epoch s]": "timestamp",
            "latitude [deg]": "latitude",
            "longitude [deg]": "longitude",
            "depth [m]": "depth",
            "roll [deg]": "roll",
            "pitch [deg]": "pitch",
            "yaw [deg]": "yaw",
        }
    )

    # Define DOF signals to plot
    dof_signals = [
        ("latitude", "Latitude [deg]"),
        ("longitude", "Longitude [deg]"),
        ("depth", "Depth [m]"),
        ("roll", "Roll [deg]"),
        ("pitch", "Pitch [deg]"),
        ("yaw", "Yaw [deg]"),
    ]

    # Load all transect data
    transect_data = []
    for file_path, label in zip(existing_files, existing_labels):
        df = pd.read_csv(file_path)
        df = df.rename(
            columns={
                "timestamp [unix epoch s]": "timestamp",
                "latitude [deg]": "latitude",
                "longitude [deg]": "longitude",
                "depth [m]": "depth",
                "roll [deg]": "roll",
                "pitch [deg]": "pitch",
                "yaw [deg]": "yaw",
            }
        )
        transect_data.append((df, label))

    # Create color map for different methods
    colors = []
    for label in existing_labels:
        if "(GNSS)" in label:
            colors.append("blue")
        elif "(DR_const)" in label:
            colors.append("orange")
        elif "(DR_DVL)" in label:
            colors.append("green")
        else:
            colors.append("gray")

    # Plot each DOF
    for attr, ylabel in dof_signals:
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot main data in light gray background
        ax.plot(
            main_df["timestamp"],
            main_df[attr],
            color="lightgray",
            alpha=0.5,
            linewidth=1,
            label="All data",
        )

        # Plot transect segments
        for (df, label), color in zip(transect_data, colors):
            ax.plot(
                df["timestamp"],
                df[attr],
                color=color,
                linewidth=1.5,
                alpha=0.7,
                label=label,
            )

        ax.set_xlabel("Time [unix epoch s]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel} over Time", fontsize=12, fontweight="bold")

        # Invert y-axis for depth
        if attr == "depth":
            ax.invert_yaxis()

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        plt.show()

    print("✅ DOF time series plots complete!")


def plot_merged_data(
    main_csv: str,
    merged_csv: str,
    origin: Tuple[float, float],
) -> None:
    """
    Plot the final merged navigation data with corrections applied.

    Parameters:
    -----------
    main_csv : str
        Path to main navigation CSV
    merged_csv : str
        Path to merged CSV with corrections
    origin : tuple(lat, lon)
        Reference point for meter conversion
    """
    print("📊 Plotting merged navigation data with all corrections applied...")
    print(f"   File: {merged_csv}")
    print()

    plot_transects_comparison(
        main_csv,
        [merged_csv],
        transect_labels=["Merged (corrected)"],
        use_meters=True,
        origin=origin,
    )


def plot_dvl_velocities(
    analyzer,
    transect_intervals: List[Tuple[str, str]],
) -> None:
    """
    Plot DVL velocity data with highlighted transect intervals.

    Parameters:
    -----------
    analyzer : MotionAnalyzer
        The analyzer object with DVL data
    transect_intervals : list of tuples
        List of (start_time_str, end_time_str) tuples like ("10:50:51", "10:59:06")
    """
    print("📊 Plotting DVL velocity data...")
    print(f"   Transects: {len(transect_intervals)}")
    print()

    print("🔹 Full mission DVL velocity with highlighted transects:")
    analyzer.plot_dvl_velocity(highlight_intervals=transect_intervals)

    print("\n🔹 Individual transect DVL velocity plots:")
    analyzer.plot_dvl_velocity_intervals(transect_intervals)


def plot_highlighted_transects_2d(
    main_csv_path: str,
    dr_csv_paths: List[str],
    highlight_indices: List[int],
    highlight_labels: List[str],
    highlight_colors: List[str],
    origin: Tuple[float, float],
    figsize: Tuple[float, float] = (8, 8),
    aspect_ratio: str = "equal",
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
) -> None:
    """
    Create a 2D position plot (NED frame) with highlighted transects.

    Parameters:
    -----------
    main_csv_path : str
        Path to main CSV (unfiltered GNSS data)
    dr_csv_paths : list of str
        List of paths to DR corrected CSVs (all transects)
    highlight_indices : list of int
        Indices of transects to highlight (0-based)
    highlight_labels : list of str
        Labels for highlighted transects (e.g., ["Transect A", "Transect B"])
    highlight_colors : list of str
        Colors for highlighted transects (e.g., ["red", "blue"])
    origin : tuple(lat, lon)
        Origin point for NED coordinate conversion
    figsize : tuple(float, float)
        Figure size in inches (default: (8, 8) for square plot)
    aspect_ratio : str or float
        Axis scaling ratio. Options:
        - "equal" (default): 1 meter East = 1 meter North (true scale)
        - "auto": Let matplotlib auto-scale axes independently
        - float (e.g., 1.5): Custom ratio (East unit / North unit)
    xlim : tuple(float, float), optional
        East axis limits (min, max) in meters. If None, auto-scale.
        Example: xlim=(-10, 100) sets East axis from -10m to 100m
    ylim : tuple(float, float), optional
        North axis limits (min, max) in meters. If None, auto-scale.
        Example: ylim=(-150, 50) sets North axis from -150m to 50m
    """
    print("\n📊 Creating 2D position plot with highlighted transects...")
    print(f"   Origin: ({origin[0]:.7f}°N, {origin[1]:.7f}°E)")
    print(f"   Highlighting {len(highlight_indices)} transects")
    print()

    # Load main CSV (unfiltered baseline)
    main_df = pd.read_csv(main_csv_path)
    main_df = main_df.rename(
        columns={
            "latitude [deg]": "latitude",
            "longitude [deg]": "longitude",
        }
    )

    # Convert main data to NED (meters)
    main_east, main_north = deg_to_meter_simple(
        main_df["latitude"].values, main_df["longitude"].values, origin=origin
    )

    # Create square figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot main baseline in grey (NO LABEL - exclude from legend)
    ax.plot(
        main_east,
        main_north,
        color="darkgray",
        linewidth=1.5,
        alpha=1,
        zorder=1,
    )

    # Load and categorize DR transects
    orange_transects = []  # Regular transects
    highlighted_transects = {}  # {index: (east, north, label, color)}

    for i, dr_path in enumerate(dr_csv_paths):
        if os.path.exists(dr_path):
            df = pd.read_csv(dr_path)
            df = df.rename(
                columns={
                    "latitude [deg]": "latitude",
                    "longitude [deg]": "longitude",
                }
            )

            # Convert to NED
            east, north = deg_to_meter_simple(
                df["latitude"].values, df["longitude"].values, origin=origin
            )

            # Check if this transect should be highlighted
            if i in highlight_indices:
                idx_in_highlight = highlight_indices.index(i)
                highlighted_transects[i] = (
                    east,
                    north,
                    highlight_labels[idx_in_highlight],
                    highlight_colors[idx_in_highlight],
                )
            else:
                orange_transects.append((east, north))
        else:
            print(f"   ⚠️  Warning: {os.path.basename(dr_path)} not found")

    # Plot highlighted transects FIRST (so they appear first in legend)
    for i, (east, north, label, color) in highlighted_transects.items():
        # Add "Transect" prefix if not already present
        if not label.startswith("Transect"):
            label = f"Transect {label}"
        ax.plot(east, north, color=color, linewidth=3, alpha=0.9, label=label, zorder=3)
        print(f"   ✅ {label}: {len(east)} points, color={color}")

    # Plot orange transects (non-highlighted) without labels
    for east, north in orange_transects:
        ax.plot(
            east,
            north,
            color="orange",
            linewidth=2.5,
            alpha=0.8,
            zorder=2,
        )

    # Add single legend entry for orange transects LAST (bottom of legend)
    if orange_transects:
        ax.plot(
            [],
            [],
            color="orange",
            linewidth=2.5,
            alpha=0.8,
            label="Other transects",
        )

    # Configure plot
    ax.set_xlabel("East [m]", fontsize=12)
    ax.set_ylabel("North [m]", fontsize=12)
    ax.set_title("Navigation Track - NED Frame", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Set axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set aspect ratio for axis scaling
    if aspect_ratio == "equal":
        ax.set_aspect("equal", adjustable="datalim")
    elif aspect_ratio == "auto":
        ax.set_aspect("auto")
    else:
        ax.set_aspect(aspect_ratio, adjustable="datalim")

    plt.tight_layout()
    plt.show()

    print("\n✅ Plot complete!")
    print(
        f"   Total DR transects: {len(dr_csv_paths)} ({len(orange_transects)} orange + {len(highlighted_transects)} highlighted)"
    )


def plot_highlighted_transects_3d(
    main_csv_path: str,
    dr_csv_paths: List[str],
    highlight_indices: List[int],
    highlight_labels: List[str],
    highlight_colors: List[str],
    origin: Tuple[float, float],
    figsize: Tuple[float, float] = (12, 10),
    elev: float = 20,
    azim: float = -60,
) -> None:
    """
    Create a 3D trajectory plot (NED frame with depth) with highlighted transects.

    Parameters:
    -----------
    main_csv_path : str
        Path to main CSV (unfiltered GNSS data)
    dr_csv_paths : list of str
        List of paths to DR corrected CSVs (all transects)
    highlight_indices : list of int
        Indices of transects to highlight (0-based)
    highlight_labels : list of str
        Labels for highlighted transects (e.g., ["Transect A", "Transect B"])
    highlight_colors : list of str
        Colors for highlighted transects (e.g., ["red", "blue"])
    origin : tuple(lat, lon)
        Origin point for NED coordinate conversion
    figsize : tuple(float, float)
        Figure size in inches (default: (12, 10))
    elev : float
        Elevation viewing angle in degrees (default: 20)
    azim : float
        Azimuth viewing angle in degrees (default: -60)
    """
    print("\n📊 Creating 3D trajectory plot with highlighted transects...")
    print(f"   Origin: ({origin[0]:.7f}°N, {origin[1]:.7f}°E)")
    print(f"   Highlighting {len(highlight_indices)} transects")
    print()

    # Load main CSV (unfiltered baseline)
    main_df = pd.read_csv(main_csv_path)
    main_df = main_df.rename(
        columns={
            "latitude [deg]": "latitude",
            "longitude [deg]": "longitude",
            "depth [m]": "depth",
        }
    )

    # Convert main data to NED (meters)
    main_east, main_north = deg_to_meter_simple(
        main_df["latitude"].values, main_df["longitude"].values, origin=origin
    )
    main_depth = main_df["depth"].values

    # Create 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot main baseline in grey (NO LABEL - exclude from legend)
    ax.plot(
        main_north,  # X-axis: North
        main_east,  # Y-axis: East
        main_depth,  # Z-axis: Depth
        color="darkgray",
        linewidth=1.5,
        alpha=0.5,
        zorder=1,
    )

    # Load and categorize DR transects
    orange_transects = []  # Regular transects
    highlighted_transects = {}  # {index: (north, east, depth, label, color)}

    for i, dr_path in enumerate(dr_csv_paths):
        if os.path.exists(dr_path):
            df = pd.read_csv(dr_path)
            df = df.rename(
                columns={
                    "latitude [deg]": "latitude",
                    "longitude [deg]": "longitude",
                    "depth [m]": "depth",
                }
            )

            # Convert to NED
            east, north = deg_to_meter_simple(
                df["latitude"].values, df["longitude"].values, origin=origin
            )
            depth = df["depth"].values

            # Check if this transect should be highlighted
            if i in highlight_indices:
                idx_in_highlight = highlight_indices.index(i)
                highlighted_transects[i] = (
                    north,
                    east,
                    depth,
                    highlight_labels[idx_in_highlight],
                    highlight_colors[idx_in_highlight],
                )
            else:
                orange_transects.append((north, east, depth))
        else:
            print(f"   ⚠️  Warning: {os.path.basename(dr_path)} not found")

    # Plot highlighted transects FIRST (so they appear first in legend)
    for i, (north, east, depth, label, color) in highlighted_transects.items():
        # Add "Transect" prefix if not already present
        if not label.startswith("Transect"):
            label = f"Transect {label}"
        ax.plot(
            north,
            east,
            depth,
            color=color,
            linewidth=4,
            alpha=0.95,
            label=label,
            zorder=3,
        )
        print(f"   ✅ {label}: {len(north)} points, color={color}")

    # Plot orange transects (non-highlighted) without labels
    for north, east, depth in orange_transects:
        ax.plot(
            north,
            east,
            depth,
            color="orange",
            linewidth=3,
            alpha=0.8,
            zorder=2,
        )

    # Add single legend entry for orange transects LAST (bottom of legend)
    if orange_transects:
        ax.plot(
            [],
            [],
            [],
            color="orange",
            linewidth=3,
            alpha=0.8,
            label="Other transects",
        )

    # Configure plot
    ax.set_xlabel("North [m]", fontsize=11)
    ax.set_ylabel("East [m]", fontsize=11)
    ax.set_zlabel("Depth [m]", fontsize=11)
    ax.set_title(
        "3D Navigation Track - NED Frame", fontsize=14, fontweight="bold", pad=20
    )
    ax.invert_zaxis()  # Depth increases downward
    ax.invert_yaxis()  # Standard NED convention
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()

    print("\n✅ 3D Plot complete!")
    print(
        f"   Total DR transects: {len(dr_csv_paths)} ({len(orange_transects)} orange + {len(highlighted_transects)} highlighted)"
    )
    print(f"   View angles: elevation={elev}°, azimuth={azim}°")
