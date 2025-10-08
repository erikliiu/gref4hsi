"""Analyze altitude differences between IMU and camera frames.

This script mirrors the geometric configuration used in `gref4hsi.tests.test_eely`
so that the computed camera pose is consistent with the production pipeline.
It loads the merged navigation CSV, transforms the body pose into the camera
frame using the same rotation matrices and lever-arm offsets, and produces plots
comparing the vehicle (IMU) altitude, the altimeter readings, and the derived
camera altitude.

QUICK START - Just run the script directly:
    python camera_imu_altitude_comparison.py

CONFIGURATION (edit these constants below):
    RUN_DIRECTLY: Set to True to run without CLI arguments
    DEFAULT_STRIDE: Subsampling factor for navigation data (5 = every 5th point)
    SHOW_PLOT: Set to True to display interactive plot window
    SAVE_PLOT: Set to True to save PNG file
    H5_FOLDER: Path to folder containing H5 files
    H5_PATTERN: Glob pattern to match H5 files (e.g., "rad_uhi_*.h5")
    FILTER_BY_H5_TIME: Set to True to only plot nav data during H5 acquisitions

The output figure can be shown interactively and/or saved alongside a brief CSV
report summarising the statistics of the differences.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymap3d as pm
from scipy.spatial.transform import Rotation as R
import h5py
import glob

# ---------------------------------------------------------------------------
# Defaults aligned with gref4hsi/tests/test_eely.py
# ---------------------------------------------------------------------------
DEFAULT_NAV_PATH = Path(r"E:\mjosa_new\navigation_data\nav_data_merged.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_ORIGIN = (
    10.7122345,
    60.8011575,
    0.0,
)  # (lon, lat, alt) as in SettingsPreprocess
DEFAULT_TIME_OFFSET = 0.0
DEFAULT_TRANSLATION_BODY_TO_HSI = np.array([2.5, 0.0, 0.0])
DEFAULT_ROTATION_HSI_TO_BODY = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)

# Configuration for direct execution (no CLI arguments)
RUN_DIRECTLY = True  # Set to True to run without command-line arguments
DEFAULT_STRIDE = 5
SHOW_PLOT = True  # Set to True to display interactive plot window
SAVE_PLOT = True  # Set to True to also save PNG

# H5 file configuration - set paths to filter time range based on H5 timestamps
H5_FOLDER = Path(r"E:\mjosa_new_oct_2025\use_gref4hsi\057_own_code_1to2\input")
H5_PATTERN = "rad_uhi_*.h5"  # Pattern to match H5 files
FILTER_BY_H5_TIME = True  # Set to True to only show nav data during H5 acquisitions

REQUIRED_COLUMNS = [
    "timestamp [unix epoch s]",
    "latitude [deg]",
    "longitude [deg]",
    "depth [m]",
    "roll [deg]",
    "pitch [deg]",
    "yaw [deg]",
    "altitude [m]",
]


@dataclass
class AltitudeSeries:
    time: pd.Series
    imu_altitude_m: np.ndarray
    camera_altitude_m: np.ndarray
    range_altitude_m: np.ndarray

    @property
    def difference_m(self) -> np.ndarray:
        return self.camera_altitude_m - self.imu_altitude_m


def _ensure_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Navigation CSV is missing required columns: " + ", ".join(missing)
        )


def _load_navigation(csv_path: Path, stride: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _ensure_required_columns(df, REQUIRED_COLUMNS)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    return df


def _get_h5_time_ranges(h5_folder: Path, h5_pattern: str) -> list[tuple[float, float]]:
    """Extract time ranges from all H5 files matching the pattern."""
    search_path = h5_folder / h5_pattern
    h5_files = glob.glob(str(search_path))

    time_ranges = []
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, "r") as f:
                # Try common timestamp paths
                timestamp_paths = [
                    "processed/radiance/timestamp",
                    "raw/timestamp",
                    "timestamp",
                ]
                timestamps = None
                for path in timestamp_paths:
                    try:
                        timestamps = f[path][()]
                        break
                    except KeyError:
                        continue

                if timestamps is not None and len(timestamps) > 0:
                    time_ranges.append(
                        (float(timestamps.min()), float(timestamps.max()))
                    )
                    print(
                        f"  {Path(h5_file).name}: {timestamps.min():.2f} to {timestamps.max():.2f}"
                    )
        except Exception as e:
            print(f"Warning: Could not read {h5_file}: {e}")

    return time_ranges


def _filter_by_time_ranges(
    df: pd.DataFrame, time_ranges: list[tuple[float, float]], buffer_sec: float = 10.0
) -> pd.DataFrame:
    """Filter navigation data to only include times within H5 time ranges (with buffer)."""
    if not time_ranges:
        return df

    timestamps = df["timestamp [unix epoch s]"].to_numpy()
    mask = np.zeros(len(timestamps), dtype=bool)

    for t_min, t_max in time_ranges:
        mask |= (timestamps >= t_min - buffer_sec) & (timestamps <= t_max + buffer_sec)

    filtered_df = df[mask].reset_index(drop=True)
    print(
        f"\nFiltered navigation: {len(filtered_df)} / {len(df)} records within H5 time ranges"
    )
    return filtered_df


def _compute_camera_altitude(
    nav_df: pd.DataFrame,
    origin_lon_lat_alt: tuple[float, float, float],
    translation_body_to_hsi: np.ndarray,
) -> AltitudeSeries:
    lon0, lat0, h0 = origin_lon_lat_alt

    timestamps = pd.to_datetime(nav_df["timestamp [unix epoch s]"], unit="s", utc=True)

    lat = nav_df["latitude [deg]"].to_numpy(dtype=float)
    lon = nav_df["longitude [deg]"].to_numpy(dtype=float)
    depth = nav_df["depth [m]"].to_numpy(dtype=float)

    roll_deg = nav_df["roll [deg]"].to_numpy(dtype=float)
    pitch_deg = nav_df["pitch [deg]"].to_numpy(dtype=float)
    yaw_deg = nav_df["yaw [deg]"].to_numpy(dtype=float)

    # Position of the IMU (body origin) expressed in a local NED frame
    north, east, down = pm.geodetic2ned(
        lat, lon, depth, lat0=lat0, lon0=lon0, h0=h0, deg=True
    )

    imu_altitude = -down  # convert Down (positive down) to altitude (positive up)

    # Body-to-NED rotation using the same convention as altimeter_data_to_point_cloud
    euler_triplets = np.column_stack((yaw_deg, pitch_deg, roll_deg))
    r_body_to_ned = R.from_euler("ZYX", euler_triplets, degrees=True).as_matrix()

    lever_arm_ned = np.einsum("ijk,k->ij", r_body_to_ned, translation_body_to_hsi)
    camera_down = down + lever_arm_ned[:, 2]
    camera_altitude = -camera_down

    range_altitude = nav_df["altitude [m]"].to_numpy(dtype=float)

    return AltitudeSeries(
        time=timestamps,
        imu_altitude_m=imu_altitude,
        camera_altitude_m=camera_altitude,
        range_altitude_m=range_altitude,
    )


def _make_plots(
    series: AltitudeSeries,
    output_dir: Path,
    title_suffix: str,
    show: bool = True,
    save: bool = True,
) -> Path | None:
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )

    axes[0].plot(
        series.time, series.imu_altitude_m, label="IMU altitude", linewidth=1.2
    )
    axes[0].plot(
        series.time, series.camera_altitude_m, label="Camera altitude", linewidth=1.2
    )
    axes[0].plot(
        series.time,
        series.range_altitude_m,
        label="Range sensor",
        linewidth=1.0,
        alpha=0.6,
    )
    axes[0].set_ylabel("Altitude above seabed [m]")
    axes[0].set_title(f"Altitude comparison {title_suffix}")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(series.time, series.difference_m, color="tab:red", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Camera - IMU [m]")
    axes[1].set_xlabel("UTC time")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        "Camera vs. IMU altitude (matching test_eely configuration)", fontsize=14
    )

    output_path = None
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "camera_vs_imu_altitude.png"
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _export_summary(series: AltitudeSeries, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metric": [
            "mean(camera-imu)",
            "min(camera-imu)",
            "max(camera-imu)",
            "std(camera-imu)",
        ],
        "value_m": [
            float(np.mean(series.difference_m)),
            float(np.min(series.difference_m)),
            float(np.max(series.difference_m)),
            float(np.std(series.difference_m)),
        ],
    }
    df = pd.DataFrame(summary)
    output_path = output_dir / "camera_vs_imu_altitude_stats.csv"
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    # Check if running directly or from command line
    if RUN_DIRECTLY:
        print("Running in direct mode with default settings...")
        nav_csv = DEFAULT_NAV_PATH
        output_dir = DEFAULT_OUTPUT_DIR
        stride = DEFAULT_STRIDE
        time_offset = DEFAULT_TIME_OFFSET
    else:
        parser = argparse.ArgumentParser(description="Compare camera and IMU altitude.")
        parser.add_argument(
            "--nav-csv",
            type=Path,
            default=DEFAULT_NAV_PATH,
            help="Path to navigation CSV (defaults to merged nav file)",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help="Directory where plots and summaries will be stored",
        )
        parser.add_argument(
            "--stride",
            type=int,
            default=1,
            help="Subsample the input navigation data by this factor for quicker plotting.",
        )
        parser.add_argument(
            "--time-offset",
            type=float,
            default=DEFAULT_TIME_OFFSET,
            help="Optional time offset to subtract from the sensor timestamps (s).",
        )
        args = parser.parse_args()
        nav_csv = args.nav_csv
        output_dir = args.output_dir
        stride = args.stride
        time_offset = args.time_offset

    if not nav_csv.exists():
        raise FileNotFoundError(f"Navigation CSV not found: {nav_csv}")

    nav_df = _load_navigation(nav_csv, stride=max(1, stride))

    # Filter by H5 time ranges if requested
    if FILTER_BY_H5_TIME and H5_FOLDER.exists():
        print(f"\nSearching for H5 files in {H5_FOLDER}...")
        time_ranges = _get_h5_time_ranges(H5_FOLDER, H5_PATTERN)
        if time_ranges:
            nav_df = _filter_by_time_ranges(nav_df, time_ranges)
        else:
            print("Warning: No H5 files found or readable, using full nav data")

    series = _compute_camera_altitude(
        nav_df=nav_df,
        origin_lon_lat_alt=DEFAULT_ORIGIN,
        translation_body_to_hsi=DEFAULT_TRANSLATION_BODY_TO_HSI,
    )

    # Apply a global time offset if requested to align with HSI timeline
    if time_offset:
        series.time = series.time - pd.to_timedelta(time_offset, unit="s")

    plot_path = _make_plots(
        series,
        output_dir=output_dir,
        title_suffix="(stride {})".format(stride),
        show=SHOW_PLOT,
        save=SAVE_PLOT,
    )

    if SAVE_PLOT:
        summary_path = _export_summary(series, output_dir=output_dir)
        print(f"✓ Plot saved to {plot_path}")
        print(f"✓ Summary saved to {summary_path}")

    print(
        "Camera minus IMU altitude stats [m]: mean={:.3f}, min={:.3f}, max={:.3f}, std={:.3f}".format(
            float(np.mean(series.difference_m)),
            float(np.min(series.difference_m)),
            float(np.max(series.difference_m)),
            float(np.std(series.difference_m)),
        )
    )


if __name__ == "__main__":
    main()
