"""
6DOF (Position + Orientation) Visualization for UHI Transect Section

Plots navigation data for the specific transect section defined by UHI_TRACK_RANGE:
- Position: East/North scatter plot (with alignment shift applied)
- Orientation: Roll, Pitch, Yaw time series
- Depth and Altitude time series

Uses config values:
- NAV_CSV: Navigation data source
- UHI_FILES, UHI_TRACK_RANGE: Define transect section
- UHI_ALIGNMENT_DX, UHI_ALIGNMENT_DY: Coordinate alignment
- LAT0, LON0: NED frame origin
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
from datetime import datetime, timezone

# Add paths for imports - use gref4hsi utils but mjosa_code config
gref4hsi_root = Path(__file__).parent.parent.parent  # Up to gref4hsi root
mjosa_code_root = Path(__file__).parent.parent  # Up to mjosa_code
sys.path.insert(0, str(gref4hsi_root / "gref4hsi" / "final_act"))
sys.path.insert(0, str(mjosa_code_root))

from utils.common import config  # Use mjosa_code config

# Add PyPost for DVL data access
sys.path.append(str(mjosa_code_root / "external_libs" / "eelume_pypost"))
from Eelume import PyPost as pp

# WGS84 ellipsoid parameters for lat/lon to meters conversion
A_WGS84 = 6378137.0  # semi-major axis (m)
E2_WGS84 = 0.00669437999014  # first eccentricity squared


def lat_lon_to_ned(lat, lon, lat0, lon0):
    """
    Convert geodetic coordinates to local NED (North-East-Down) frame.

    Parameters:
    -----------
    lat, lon : array-like
        Latitude and longitude in degrees
    lat0, lon0 : float
        Origin of NED frame in degrees

    Returns:
    --------
    north, east : array-like
        North and East coordinates in meters
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    # Radius of curvature in the prime vertical
    N = A_WGS84 / np.sqrt(1 - E2_WGS84 * np.sin(lat0_rad) ** 2)

    # Approximate NED conversion (valid for small distances)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    north = dlat * (N * (1 - E2_WGS84) / (1 - E2_WGS84 * np.sin(lat0_rad) ** 2))
    east = dlon * N * np.cos(lat0_rad)

    return north, east


def get_transect_time_range():
    """
    Get time range for the transect by reading timestamps directly from H5 files.

    The track range in config refers to frame indices across ALL specified H5 files combined.
    For example, if file_4 has 2463 frames and file_5 has 2462 frames:
    - Tracks 0-2462 are in file_4
    - Tracks 2463-4924 are in file_5

    Returns:
    --------
    t_start, t_end : float
        Unix timestamps for the transect section
    """
    h5_folder = Path(config.H5_FOLDER)
    track_start, track_end = config.UHI_TRACK_RANGE

    print(
        f"\n🔍 Reading timestamps from H5 files for tracks {track_start}-{track_end}..."
    )

    cumulative_tracks = 0
    t_start = None
    t_end = None

    for uhi_file in config.UHI_FILES:
        h5_path = h5_folder / f"{uhi_file}.h5"

        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as hf:
            # Read timestamps from the H5 file
            timestamps = hf["processed/radiance/timestamp"][:]
            n_tracks = len(timestamps)

            file_track_start = cumulative_tracks
            file_track_end = cumulative_tracks + n_tracks

            print(
                f"  📁 {uhi_file}: tracks {file_track_start}-{file_track_end-1} ({n_tracks} frames)"
            )

            # Check if this file overlaps with our target range
            if file_track_end > track_start and file_track_start < track_end:
                # Calculate overlap indices within this file
                overlap_start = max(track_start, file_track_start)
                overlap_end = min(track_end, file_track_end)

                # Convert to local indices within this file
                local_start = overlap_start - file_track_start
                local_end = overlap_end - file_track_start

                # Get timestamps for this overlap
                if t_start is None:
                    t_start = timestamps[local_start]
                t_end = timestamps[local_end - 1]  # -1 because end is exclusive

                print(
                    f"    ✓ Overlap: global tracks {overlap_start}-{overlap_end-1}, local indices {local_start}-{local_end-1}"
                )
                print(
                    f"    ⏰ Times: {datetime.fromtimestamp(timestamps[local_start], tz=timezone.utc).strftime('%H:%M:%S.%f')[:-3]} to {datetime.fromtimestamp(timestamps[local_end-1], tz=timezone.utc).strftime('%H:%M:%S.%f')[:-3]} UTC"
                )

            cumulative_tracks = file_track_end

    if t_start is None or t_end is None:
        raise ValueError(
            f"Could not find time range for tracks {track_start}-{track_end}"
        )

    # Apply time offset from config
    t_start += config.TIME_OFFSET_SEC
    t_end += config.TIME_OFFSET_SEC

    print(f"\n✅ Final time range (after {config.TIME_OFFSET_SEC}s offset):")
    print(f"   Start: {datetime.fromtimestamp(t_start, tz=timezone.utc)}")
    print(f"   End:   {datetime.fromtimestamp(t_end, tz=timezone.utc)}")
    print(f"   Duration: {t_end - t_start:.2f} seconds")

    return t_start, t_end


def load_navigation_data(t_start, t_end):
    """
    Load navigation data and filter to transect time range.
    Also returns full mission data for context plotting.

    Parameters:
    -----------
    t_start, t_end : float
        Unix timestamps defining time range

    Returns:
    --------
    nav_df : pd.DataFrame
        Filtered navigation data for transect
    nav_full : pd.DataFrame
        Full mission navigation data for context
    """
    print(f"\n📊 Loading navigation data from {config.NAV_CSV}...")

    # Load full navigation CSV
    nav_full = pd.read_csv(config.NAV_CSV)

    # Get column names from config
    col_time = config.CSV_COLUMNS["timestamp"]
    col_lat = config.CSV_COLUMNS["latitude"]
    col_lon = config.CSV_COLUMNS["longitude"]
    col_depth = config.CSV_COLUMNS["depth"]
    col_roll = config.CSV_COLUMNS["roll"]
    col_pitch = config.CSV_COLUMNS["pitch"]
    col_yaw = config.CSV_COLUMNS["yaw"]
    col_alt = config.CSV_COLUMNS["altitude"]

    print(f"   Total nav records: {len(nav_full)}")

    # Convert full dataset to NED
    north_full, east_full = lat_lon_to_ned(
        nav_full[col_lat].values, nav_full[col_lon].values, config.LAT0, config.LON0
    )
    nav_full["east"] = east_full
    nav_full["north"] = north_full
    nav_full["east_aligned"] = east_full + config.UHI_ALIGNMENT_DX
    nav_full["north_aligned"] = north_full + config.UHI_ALIGNMENT_DY

    # Filter to transect time range
    mask = (nav_full[col_time] >= t_start) & (nav_full[col_time] <= t_end)
    nav_df = nav_full[mask].copy()
    print(f"   Transect records: {len(nav_df)}")

    if len(nav_df) == 0:
        raise ValueError("No navigation data found in transect time range!")

    # Convert lat/lon to NED
    north, east = lat_lon_to_ned(
        nav_df[col_lat].values, nav_df[col_lon].values, config.LAT0, config.LON0
    )

    # Apply alignment shift
    east_aligned = east + config.UHI_ALIGNMENT_DX
    north_aligned = north + config.UHI_ALIGNMENT_DY

    # Add to dataframe with simpler column names
    nav_df["time"] = nav_df[col_time]
    nav_df["east"] = east
    nav_df["north"] = north
    nav_df["east_aligned"] = east_aligned
    nav_df["north_aligned"] = north_aligned
    nav_df["depth"] = nav_df[col_depth]
    nav_df["roll"] = nav_df[col_roll]
    nav_df["pitch"] = nav_df[col_pitch]
    nav_df["yaw"] = nav_df[col_yaw]
    nav_df["altitude"] = nav_df[col_alt]

    print(f"   ✓ Coordinate conversion complete")
    print(
        f"   Alignment shift applied: ΔE = {config.UHI_ALIGNMENT_DX:.3f} m, ΔN = {config.UHI_ALIGNMENT_DY:.3f} m"
    )

    return nav_df, nav_full


def plot_attitude_and_dvl_stacked(nav_df, db_path, t_start, t_end):
    """
    Create stacked plots: Attitude (top) and DVL velocity (bottom).
    Both share the same time axis for clean comparison.

    Parameters:
    -----------
    nav_df : pd.DataFrame
        Navigation data for the transect
    db_path : str
        Path to database file for DVL data
    t_start, t_end : float
        Unix timestamps for time range
    """
    print("\n� Creating stacked attitude and DVL velocity plots...")

    # Get navigation times
    utc_times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in nav_df["time"]]

    # Get DVL data from database
    db = pp.DatabaseHandler(db_path)
    sig_x = db.get_signal(
        "/sensors/dvl/velocity/D__x",
        start_time=t_start,
        end_time=t_end,
        reset_time_stamp_to_zero=False,
    )
    sig_y = db.get_signal(
        "/sensors/dvl/velocity/D__y",
        start_time=t_start,
        end_time=t_end,
        reset_time_stamp_to_zero=False,
    )
    sig_z = db.get_signal(
        "/sensors/dvl/velocity/D__z",
        start_time=t_start,
        end_time=t_end,
        reset_time_stamp_to_zero=False,
    )

    # Convert DVL timestamps to UTC datetime
    times_x = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in sig_x.axis]
    times_y = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in sig_y.axis]
    times_z = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in sig_z.axis]

    # Create stacked subplots: 5 inches wide, 4.5 inches tall total (more space for legends)
    fig, (ax_att, ax_dvl) = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True)

    # ===== TOP SUBPLOT: ATTITUDE (with dual Y-axes) =====
    ax_att_left = ax_att
    ax_att_right = ax_att.twinx()

    # Left Y-axis: Roll and Pitch
    ax_att_left.plot(
        utc_times, nav_df["roll"], "g-", linewidth=1, label="Roll", alpha=0.9
    )
    ax_att_left.plot(
        utc_times, nav_df["pitch"], "b-", linewidth=1, label="Pitch", alpha=0.9
    )
    ax_att_left.set_ylabel("Roll / Pitch [deg]", fontsize=10)
    ax_att_left.tick_params(axis="y", labelsize=9)
    ax_att_left.grid(True, alpha=0.3)

    # Right Y-axis: Yaw
    ax_att_right.plot(
        utc_times, nav_df["yaw"], "r-", linewidth=1, label="Yaw", alpha=0.9
    )
    ax_att_right.set_ylabel("Yaw [deg]", fontsize=10, color="red")
    ax_att_right.tick_params(axis="y", labelcolor="red", labelsize=9)

    # === LEGEND OPTIONS - Choose one by uncommenting ===

    # OPTION 1: Above the plot (Best for papers) - CURRENTLY ACTIVE
    lines1, labels1 = ax_att_left.get_legend_handles_labels()
    lines2, labels2 = ax_att_right.get_legend_handles_labels()
    ax_att_left.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    # OPTION 2: Inside upper left (Compact inline) - UNCOMMENT TO USE
    # lines1, labels1 = ax_att_left.get_legend_handles_labels()
    # lines2, labels2 = ax_att_right.get_legend_handles_labels()
    # ax_att_left.legend(lines1 + lines2, labels1 + labels2,
    #                    loc="upper left", frameon=False, fontsize=9)

    # OPTION 3: Shared global legend at figure top - UNCOMMENT TO USE
    # (Implemented at the end after both subplots are created)

    # ===== BOTTOM SUBPLOT: DVL VELOCITY =====
    ax_dvl.plot(times_x, sig_x.data, "g-", linewidth=1, label="D__x", alpha=0.9)
    ax_dvl.plot(times_y, sig_y.data, "b-", linewidth=1, label="D__y", alpha=0.9)
    ax_dvl.plot(times_z, sig_z.data, "r-", linewidth=1, label="D__z", alpha=0.9)

    ax_dvl.set_xlabel("Time (UTC)", fontsize=10)
    ax_dvl.set_ylabel("Velocity [m/s]", fontsize=10)
    ax_dvl.grid(True, alpha=0.3)
    ax_dvl.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax_dvl.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # OPTION 1: Above the plot (Best for papers) - CURRENTLY ACTIVE
    ax_dvl.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    # OPTION 2: Inside upper left (Compact inline) - UNCOMMENT TO USE
    # ax_dvl.legend(loc="upper left", frameon=False, fontsize=9)

    # === OPTION 3: Shared global legend - UNCOMMENT TO USE ===
    # Remove individual legends first:
    # ax_att_left.get_legend().remove()
    # ax_dvl.get_legend().remove()
    # Then add one shared legend:
    # fig.legend(labels=["Roll", "Pitch", "Yaw", "D__x", "D__y", "D__z"],
    #            loc="upper center", bbox_to_anchor=(0.5, 0.98),
    #            ncol=6, frameon=False, fontsize=9)

    plt.tight_layout()

    # Save stacked plot
    output_path = Path(config.OUTPUT_FOLDER) / "6dof_attitude_dvl_stacked.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Saved stacked attitude+DVL plot to: {output_path}")

    # Print statistics
    roll_mean, roll_std = nav_df["roll"].mean(), nav_df["roll"].std()
    pitch_mean, pitch_std = nav_df["pitch"].mean(), nav_df["pitch"].std()
    yaw_mean, yaw_std = nav_df["yaw"].mean(), nav_df["yaw"].std()

    print(f"   Attitude Statistics:")
    print(f"      Roll: μ={roll_mean:.2f}°, σ={roll_std:.2f}°")
    print(f"      Pitch: μ={pitch_mean:.2f}°, σ={pitch_std:.2f}°")
    print(f"      Yaw: μ={yaw_mean:.2f}°, σ={yaw_std:.2f}°")
    print(f"   DVL Statistics:")
    print(
        f"      D__x: μ={np.mean(sig_x.data):.3f} m/s, σ={np.std(sig_x.data):.3f} m/s"
    )
    print(
        f"      D__y: μ={np.mean(sig_y.data):.3f} m/s, σ={np.std(sig_y.data):.3f} m/s"
    )
    print(
        f"      D__z: μ={np.mean(sig_z.data):.3f} m/s, σ={np.std(sig_z.data):.3f} m/s"
    )


def plot_6dof(nav_df, nav_full):
    """
    Create 6DOF plots: position, orientation (separate), depth, altitude.

    Parameters:
    -----------
    nav_df : pd.DataFrame
        Filtered navigation data for transect
    nav_full : pd.DataFrame
        Full mission navigation data for context
    """
    print(f"\n📈 Creating 6DOF plots...")

    # Create figure with 2x4 subplot layout
    fig = plt.figure(figsize=(24, 12))

    # Convert Unix timestamps to UTC datetime objects for plotting
    utc_times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in nav_df["time"]]

    # ===== Row 1: Position =====

    # 1. East/North scatter (with alignment) - show full mission context
    ax1 = plt.subplot(2, 4, 1)

    # Plot full mission trajectory in gray (background)
    ax1.plot(
        nav_full["east_aligned"],
        nav_full["north_aligned"],
        "gray",
        linewidth=1,
        alpha=0.4,
        label="Full Mission",
        zorder=1,
    )

    # Plot transect section highlighted in RED (no colormap scale)
    ax1.scatter(
        nav_df["east_aligned"],
        nav_df["north_aligned"],
        c="red",
        s=15,
        alpha=0.9,
        edgecolors="darkred",
        linewidths=0.5,
        label="Transect Section",
        zorder=2,
    )

    ax1.set_xlabel("East [m]", fontsize=12)
    ax1.set_ylabel("North [m]", fontsize=12)
    ax1.set_title(
        "Position - Transect in Context (NED with Alignment)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", "box")
    ax1.legend(loc="best", fontsize=9)

    # Add alignment info
    ax1.text(
        0.02,
        0.98,
        f"Shift: ΔE={config.UHI_ALIGNMENT_DX:.2f}m, ΔN={config.UHI_ALIGNMENT_DY:.2f}m",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 2. East vs Time
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(
        utc_times, nav_df["east_aligned"], "b-", linewidth=1.5, label="With Alignment"
    )
    ax2.plot(utc_times, nav_df["east"], "b--", linewidth=1, alpha=0.5, label="Original")
    ax2.set_xlabel("Time (UTC)", fontsize=12)
    ax2.set_ylabel("East [m]", fontsize=12)
    ax2.set_title("East Position vs Time", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 3. North vs Time
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(
        utc_times, nav_df["north_aligned"], "r-", linewidth=1.5, label="With Alignment"
    )
    ax3.plot(
        utc_times, nav_df["north"], "r--", linewidth=1, alpha=0.5, label="Original"
    )
    ax3.set_xlabel("Time (UTC)", fontsize=12)
    ax3.set_ylabel("North [m]", fontsize=12)
    ax3.set_title("North Position vs Time", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 4. Depth
    ax4 = plt.subplot(2, 4, 4)
    ax4.plot(utc_times, nav_df["depth"], "b-", linewidth=1.5)
    ax4.set_xlabel("Time (UTC)", fontsize=12)
    ax4.set_ylabel("Depth [m]", fontsize=12)
    ax4.set_title("Depth vs Time", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()  # Depth increases downward
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats
    depth_mean = nav_df["depth"].mean()
    depth_std = nav_df["depth"].std()
    ax4.text(
        0.02,
        0.02,
        f"Mean: {depth_mean:.2f}m\nStd: {depth_std:.2f}m",
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # ===== Row 2: Orientation (separate plots) + Altitude =====

    # 5. Roll
    ax5 = plt.subplot(2, 4, 5)
    ax5.plot(utc_times, nav_df["roll"], "g-", linewidth=1.5)
    ax5.set_xlabel("Time (UTC)", fontsize=12)
    ax5.set_ylabel("Roll [deg]", fontsize=12)
    ax5.set_title("Roll vs Time", fontsize=13, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats
    roll_mean = nav_df["roll"].mean()
    roll_std = nav_df["roll"].std()
    ax5.text(
        0.02,
        0.98,
        f"Mean: {roll_mean:.2f}°\nStd: {roll_std:.2f}°",
        transform=ax5.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    # 6. Pitch
    ax6 = plt.subplot(2, 4, 6)
    ax6.plot(utc_times, nav_df["pitch"], "b-", linewidth=1.5)
    ax6.set_xlabel("Time (UTC)", fontsize=12)
    ax6.set_ylabel("Pitch [deg]", fontsize=12)
    ax6.set_title("Pitch vs Time", fontsize=13, fontweight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats
    pitch_mean = nav_df["pitch"].mean()
    pitch_std = nav_df["pitch"].std()
    ax6.text(
        0.02,
        0.98,
        f"Mean: {pitch_mean:.2f}°\nStd: {pitch_std:.2f}°",
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # 7. Yaw
    ax7 = plt.subplot(2, 4, 7)
    ax7.plot(utc_times, nav_df["yaw"], "r-", linewidth=1.5)
    ax7.set_xlabel("Time (UTC)", fontsize=12)
    ax7.set_ylabel("Yaw [deg]", fontsize=12)
    ax7.set_title("Yaw vs Time", fontsize=13, fontweight="bold")
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats
    yaw_mean = nav_df["yaw"].mean()
    yaw_std = nav_df["yaw"].std()
    ax7.text(
        0.02,
        0.98,
        f"Mean: {yaw_mean:.2f}°\nStd: {yaw_std:.2f}°",
        transform=ax7.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )

    # 8. Altitude
    ax8 = plt.subplot(2, 4, 8)
    ax8.plot(utc_times, nav_df["altitude"], "orange", linewidth=1.5)
    ax8.set_xlabel("Time (UTC)", fontsize=12)
    ax8.set_ylabel("Altitude [m]", fontsize=12)
    ax8.set_title("Altitude vs Time", fontsize=13, fontweight="bold")
    ax8.grid(True, alpha=0.3)
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats
    alt_mean = nav_df["altitude"].mean()
    alt_std = nav_df["altitude"].std()
    ax8.text(
        0.02,
        0.98,
        f"Mean: {alt_mean:.2f}m\nStd: {alt_std:.2f}m",
        transform=ax8.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="moccasin", alpha=0.5),
    )

    # Overall title
    duration = nav_df["time"].iloc[-1] - nav_df["time"].iloc[0]
    fig.suptitle(
        f"6DOF Navigation Data - UHI Transect (Tracks {config.UHI_TRACK_RANGE[0]}-{config.UHI_TRACK_RANGE[1]}, {len(nav_df)} samples, {duration:.1f}s)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = Path(config.OUTPUT_FOLDER) / "6dof_transect_navigation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Saved to: {output_path}")

    # ===== ADDITIONAL PLOT: Combined Roll/Pitch/Yaw with dual Y-axes =====
    print("\n📈 Creating combined orientation plot with dual Y-axes...")
    fig2, ax_left = plt.subplots(figsize=(5, 2))  # Smaller size for paper column

    # Left Y-axis: Roll and Pitch (smaller range, ~0-1 degrees)
    ax_left.plot(utc_times, nav_df["roll"], "g-", linewidth=1, label="Roll", alpha=0.8)
    ax_left.plot(
        utc_times, nav_df["pitch"], "b-", linewidth=1, label="Pitch", alpha=0.8
    )
    ax_left.set_xlabel("Time (UTC)", fontsize=10)
    ax_left.set_ylabel("Roll / Pitch [deg]", fontsize=10, color="black")
    ax_left.tick_params(axis="y", labelcolor="black", labelsize=9)
    ax_left.grid(True, alpha=0.3)
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # Right Y-axis: Yaw (larger range, ~180 degrees)
    ax_right = ax_left.twinx()
    ax_right.plot(utc_times, nav_df["yaw"], "r-", linewidth=1, label="Yaw", alpha=0.8)
    ax_right.set_ylabel("Yaw [deg]", fontsize=10, color="red")
    ax_right.tick_params(axis="y", labelcolor="red", labelsize=9)

    # Title
    fig2.suptitle(
        "Attitude - UHI Transect",
        fontsize=11,
        fontweight="bold",
    )

    # Combine legends
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    plt.tight_layout()

    # Save combined orientation plot
    output_path_combined = Path(config.OUTPUT_FOLDER) / "6dof_combined_orientation.png"
    plt.savefig(output_path_combined, dpi=150, bbox_inches="tight")
    print(f"   ✓ Saved combined orientation plot to: {output_path_combined}")

    # Print orientation statistics (removed from plot)
    print(f"   Attitude Statistics:")
    print(f"      Roll: μ={roll_mean:.2f}°, σ={roll_std:.2f}°")
    print(f"      Pitch: μ={pitch_mean:.2f}°, σ={pitch_std:.2f}°")
    print(f"      Yaw: μ={yaw_mean:.2f}°, σ={yaw_std:.2f}°")

    # Print summary statistics
    print(f"\n📊 Navigation Statistics:")
    print(f"   Position range:")
    print(
        f"      East:  {nav_df['east_aligned'].min():.2f} to {nav_df['east_aligned'].max():.2f} m (range: {nav_df['east_aligned'].max()-nav_df['east_aligned'].min():.2f} m)"
    )
    print(
        f"      North: {nav_df['north_aligned'].min():.2f} to {nav_df['north_aligned'].max():.2f} m (range: {nav_df['north_aligned'].max()-nav_df['north_aligned'].min():.2f} m)"
    )
    print(f"   Orientation:")
    print(
        f"      Roll:  {nav_df['roll'].min():.2f}° to {nav_df['roll'].max():.2f}° (mean: {nav_df['roll'].mean():.2f}°, std: {nav_df['roll'].std():.2f}°)"
    )
    print(
        f"      Pitch: {nav_df['pitch'].min():.2f}° to {nav_df['pitch'].max():.2f}° (mean: {nav_df['pitch'].mean():.2f}°, std: {nav_df['pitch'].std():.2f}°)"
    )
    print(
        f"      Yaw:   {nav_df['yaw'].min():.2f}° to {nav_df['yaw'].max():.2f}° (mean: {nav_df['yaw'].mean():.2f}°, std: {nav_df['yaw'].std():.2f}°)"
    )
    print(
        f"   Depth:    {nav_df['depth'].min():.2f} to {nav_df['depth'].max():.2f} m (mean: {depth_mean:.2f}m, std: {depth_std:.2f}m)"
    )
    print(
        f"   Altitude: {nav_df['altitude'].min():.2f} to {nav_df['altitude'].max():.2f} m (mean: {alt_mean:.2f}m, std: {alt_std:.2f}m)"
    )


def main():
    """Main execution function."""
    print("=" * 80)
    print("6DOF NAVIGATION VISUALIZATION FOR UHI TRANSECT SECTION")
    print("=" * 80)

    try:
        # Step 1: Get time range for transect from statistics
        t_start, t_end = get_transect_time_range()

        # Step 2: Load and filter navigation data
        nav_df, nav_full = load_navigation_data(t_start, t_end)

        # Step 3: Create navigation plots
        plot_6dof(nav_df, nav_full)

        # Step 4: Create stacked attitude and DVL velocity plot
        db_path = config.DB_PATH
        plot_attitude_and_dvl_stacked(nav_df, db_path, t_start, t_end)

        # Show all figures at once
        print("\n📊 Displaying all plots...")
        plt.show()

        print("\n✅ Complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
