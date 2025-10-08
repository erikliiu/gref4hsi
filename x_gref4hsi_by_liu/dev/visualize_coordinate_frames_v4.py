"""
Static 3D mission plot with UHI window highlighted.
Prints HSI start and end times in epoch and UTC, and the offset corrected lookup range.

Requirements from your repo:
- config: NAV_CSV, CSV_COLUMNS, H5_FOLDER, TIME_OFFSET_SEC,
          EPSG_GEOGRAPHIC, EPSG_ECEF, EPSG_MBES
- utils: load_csv_navigation, load_h5_timestamps, interpolate_navigation,
         geographic_to_ecef, ecef_to_utm

Run:
    python visualize_coordinate_frames_static_3d.py
    python visualize_coordinate_frames_static_3d.py --frames 400
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# Make local config and utils importable
sys.path.append(str(Path(__file__).parent.parent))
import config  # noqa: E402
import utils  # noqa: E402


def _fmt_utc(ts: float) -> str:
    """Unix seconds to 'YYYY-mm-dd HH:MM:SS UTC'."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def load_static_highlight_data(num_frames: int | None = None):
    """
    Load
      • Full mission path from NAV CSV to UTM
      • HSI timestamps from first H5 in config.H5_FOLDER
      • Interpolate NAV at (HSI + TIME_OFFSET_SEC), convert to UTM for highlight
    """
    # 1) Full mission from CSV
    nav_df = utils.load_csv_navigation(config.NAV_CSV, config.CSV_COLUMNS)

    lon_all = nav_df["longitude"].to_numpy()
    lat_all = nav_df["latitude"].to_numpy()
    depth_all = nav_df["depth"].to_numpy()

    x_ecef, y_ecef, z_ecef = utils.geographic_to_ecef(
        lon_all,
        lat_all,
        -depth_all,
        epsg_geo=config.EPSG_GEOGRAPHIC,
        epsg_ecef=config.EPSG_ECEF,
    )
    x_all, y_all, z_all = utils.ecef_to_utm(
        x_ecef,
        y_ecef,
        z_ecef,
        epsg_utm=config.EPSG_MBES,
        epsg_ecef=config.EPSG_ECEF,
    )

    # 2) HSI timestamps from H5
    h5_folder = Path(config.H5_FOLDER)
    h5_files = sorted(h5_folder.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_folder}")
    h5_path = str(h5_files[0])

    hsi_ts = utils.load_h5_timestamps(h5_path)  # raw HSI frame times
    if len(hsi_ts) == 0:
        raise ValueError(f"HSI has zero timestamps in {h5_path}")

    if num_frames is not None and len(hsi_ts) > num_frames:
        idx = np.linspace(0, len(hsi_ts) - 1, num_frames, dtype=int)
        hsi_ts = hsi_ts[idx]

    # 3) Interpolate NAV at lookup times = HSI + offset
    offset = float(config.TIME_OFFSET_SEC)
    interp_nav = utils.interpolate_navigation(nav_df, hsi_ts, time_offset=offset)

    # 4) Convert highlighted path to UTM
    xe, ye, ze = utils.geographic_to_ecef(
        interp_nav["longitude"],
        interp_nav["latitude"],
        -interp_nav["depth"],
        epsg_geo=config.EPSG_GEOGRAPHIC,
        epsg_ecef=config.EPSG_ECEF,
    )
    x_uhi, y_uhi, z_uhi = utils.ecef_to_utm(
        xe,
        ye,
        ze,
        epsg_utm=config.EPSG_MBES,
        epsg_ecef=config.EPSG_ECEF,
    )

    # Time bounds
    hsi_start = float(hsi_ts.min())
    hsi_end = float(hsi_ts.max())
    lookup_start = hsi_start + offset
    lookup_end = hsi_end + offset

    return {
        "full": {"x": x_all, "y": y_all, "z": z_all},
        "uhi": {"x": x_uhi, "y": y_uhi, "z": z_uhi},
        "h5_path": h5_path,
        "hsi_start_epoch": hsi_start,
        "hsi_end_epoch": hsi_end,
        "hsi_start_utc": _fmt_utc(hsi_start),
        "hsi_end_utc": _fmt_utc(hsi_end),
        "lookup_start_epoch": lookup_start,
        "lookup_end_epoch": lookup_end,
        "lookup_start_utc": _fmt_utc(lookup_start),
        "lookup_end_utc": _fmt_utc(lookup_end),
        "offset_sec": offset,
        "n_hsi_frames": int(len(hsi_ts)),
    }


def plot_static_trajectory_highlight_3d(data: dict):
    """
    Static 3D plot.
      • Full mission in gray
      • UHI window highlighted as a thicker line
    """
    xa, ya, za = data["full"]["x"], data["full"]["y"], data["full"]["z"]
    xu, yu, zu = data["uhi"]["x"], data["uhi"]["y"], data["uhi"]["z"]

    title = (
        f"Mission with UHI window highlighted\n"
        f"HSI UTC {data['hsi_start_utc']} to {data['hsi_end_utc']}, "
        f"offset {data['offset_sec']:+.1f}s, {Path(data['h5_path']).name}"
    )

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xa, ya, za, linewidth=1.0, color="gray", alpha=0.45, label="Full mission")
    ax.plot(xu, yu, zu, linewidth=2.2, label="UHI window (offset corrected)")

    rng = np.array([xa.max() - xa.min(), ya.max() - ya.min(), za.max() - za.min()])
    half = rng.max() / 2.0
    cx = (xa.max() + xa.min()) / 2.0
    cy = (ya.max() + ya.min()) / 2.0
    cz = (za.max() + za.min()) / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title(title, pad=14)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Static 3D mission plot with UHI window highlighted"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="Optional HSI frame subsample count"
    )
    args = parser.parse_args()

    data = load_static_highlight_data(num_frames=args.frames)

    # Print clear time bounds
    print("\n=== HSI time bounds (raw H5) ===")
    print(f"Frames: {data['n_hsi_frames']}")
    print(f"Epoch: {data['hsi_start_epoch']:.3f} to {data['hsi_end_epoch']:.3f}")
    print(f"UTC:   {data['hsi_start_utc']} to {data['hsi_end_utc']}")

    print("\n=== Lookup time bounds used for interpolation (HSI + offset) ===")
    print(f"Offset applied: {data['offset_sec']:+.3f} s")
    print(f"Epoch: {data['lookup_start_epoch']:.3f} to {data['lookup_end_epoch']:.3f}")
    print(f"UTC:   {data['lookup_start_utc']} to {data['lookup_end_utc']}")

    print(f"\nUsing H5: {data['h5_path']}\n")

    plot_static_trajectory_highlight_3d(data)


if __name__ == "__main__":
    main()
