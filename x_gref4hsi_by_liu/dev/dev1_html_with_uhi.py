"""
HSI vs MBES overview with UHI swath scanlines.

What it shows:
  - Full navigation trajectory (grey)
  - Active UHI transect(s) corresponding to the provided H5 files (red)
  - MBES GeoTIFF coverage polygon (blue)
  - UHI swath scanlines along the active transect(s), sized by altitude & FOV

Requires: pandas, numpy, folium, rasterio, h5py, pyproj
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium.plugins import MeasureControl, MousePosition, LocateControl
import rasterio
import h5py
from pyproj import Transformer, Geod


# ==================== USER INPUTS (edit these) ====================

NAV_CSV = r"E:\mjosa_new\navigation_data\nav_data_merged.csv"  # CSV with nav (timestamp, lat, lon, depth, altitude?, yaw?)
H5_FOLDER_OR_FILES = r"E:\mjosa_new_oct_2025\use_gref4hsi\057"  # folder containing *.h5 or a single file path
MBES_GEOTIFF = r"E:\mjosa_new\DTM\geotiff_2.tif"  # MBES GeoTIFF path
OUTPUT_HTML = "hsi_mbes_overview.html"

# If nav timestamps and H5 timestamps need alignment
TIME_OFFSET_SEC = 0  # positive means H5 timestamps are later than nav; shift nav forward by this amount

# Camera field-of-view (across-track) used for swath width
CAMERA_FOV_DEG = 45.0

# How many swath lines to draw per transect (decimated)
TARGET_SWATH_LINES = 50

# Column names in NAV_CSV (adjust if your headers differ)
COL_TS = "timestamp [unix epoch s]"
COL_LAT = "latitude [deg]"
COL_LON = "longitude [deg]"
COL_DEPTH = "depth [m]"
COL_ALT = "altitude [m]"  # if missing, we’ll fallback per-frame to NaN

# Optional yaw/heading column (degrees, 0..360, nav frame). If missing, we’ll compute bearing from lat/lon.
COL_YAW = "yaw [deg]"

# ==================== END USER INPUTS ====================


def list_h5_files(path_like):
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".h5":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No .h5 files found in folder: {p}")
        return files
    raise FileNotFoundError(f"Not a file or folder: {p}")


def load_h5_timestamps(h5_path):
    """
    Return 1D numpy array of timestamps (seconds, float64) for this H5.
    Tries common locations: processed/radiance/timestamp, timestamps, etc.
    """
    candidates = [
        "processed/radiance/timestamp",
        "processed/radiance/timestamps",
        "processed/timestamp",
        "processed/timestamps",
        "timestamp",
        "timestamps",
    ]
    with h5py.File(h5_path, "r") as f:
        for ds in candidates:
            if ds in f:
                ts = f[ds][()]
                ts = np.array(ts).astype(np.float64).ravel()
                # Handle ms vs s if needed (heuristic)
                # If average is huge (e.g., 1.7e12), assume milliseconds:
                if np.nanmean(ts) > 1e11:
                    ts = ts / 1000.0
                return ts
    raise KeyError(
        f"No timestamp dataset found in {h5_path}. Tried: {', '.join(candidates)}"
    )


def load_all_hsi_timestamp_ranges(h5_files):
    """
    Load per-file timestamp ranges and the concatenated timestamps (for union plotting).
    Returns:
      ranges: list of dicts {file, tmin, tmax, n}
      all_ts: np.ndarray of all timestamps concatenated
    """
    ranges = []
    all_ts_list = []
    for fp in h5_files:
        ts = load_h5_timestamps(fp)
        if ts.size == 0:
            continue
        ranges.append(
            {
                "file": Path(fp).name,
                "tmin": float(np.nanmin(ts)),
                "tmax": float(np.nanmax(ts)),
                "n": int(ts.size),
            }
        )
        all_ts_list.append(ts)
    if not all_ts_list:
        raise ValueError("No timestamps found in the provided H5 files.")
    all_ts = np.concatenate(all_ts_list)
    return ranges, all_ts


def load_nav_csv(nav_csv):
    df = pd.read_csv(nav_csv)
    # normalize columns
    missing = [c for c in [COL_TS, COL_LAT, COL_LON, COL_DEPTH] if c not in df.columns]
    if missing:
        raise ValueError(f"Nav CSV is missing columns: {missing}")

    # Make a copy with standardized names
    out = pd.DataFrame(
        {
            "ts": df[COL_TS].astype(float),
            "lat": df[COL_LAT].astype(float),
            "lon": df[COL_LON].astype(float),
            "depth": df[COL_DEPTH].astype(float),
        }
    )
    if COL_ALT in df.columns:
        out["alt"] = df[COL_ALT].astype(float)
    else:
        out["alt"] = np.nan

    if COL_YAW in df.columns:
        out["yaw"] = df[COL_YAW].astype(float) % 360.0
    else:
        out["yaw"] = np.nan

    # apply time offset (shift nav forward)
    out["ts"] = out["ts"] + float(TIME_OFFSET_SEC)
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def compute_bearing_from_path(lat, lon):
    """
    Compute forward bearing (deg) between consecutive points. Fallback for first point.
    """
    g = Geod(ellps="WGS84")
    bearings = np.full_like(lat, np.nan, dtype=float)
    if len(lat) < 2:
        return np.zeros_like(lat)
    # Using line azimuth between points
    for i in range(len(lat) - 1):
        fwd_az, back_az, _ = g.inv(lon[i], lat[i], lon[i + 1], lat[i + 1])
        bearings[i] = (fwd_az + 360.0) % 360.0
    # Last bearing = previous
    bearings[-1] = bearings[-2]
    # First bearing fallback if NaN
    if np.isnan(bearings[0]):
        bearings[0] = bearings[1]
    return bearings


def interpolate_nav_to_times(nav_df, ts_query):
    """
    Interpolate lat/lon/depth/alt/yaw onto arbitrary timestamps.
    Returns dict of arrays.
    """
    base = nav_df[["ts", "lat", "lon", "depth", "alt", "yaw"]].to_numpy()
    t = base[:, 0]
    lat = base[:, 1]
    lon = base[:, 2]
    depth = base[:, 3]
    alt = base[:, 4]
    yaw = base[:, 5]

    # Compute missing yaw from path if needed
    if np.all(np.isnan(yaw)):
        yaw = compute_bearing_from_path(lat, lon)

    # Ensure t strictly increasing for np.interp
    order = np.argsort(t)
    t = t[order]
    lat = lat[order]
    lon = lon[order]
    depth = depth[order]
    alt = alt[order]
    yaw = yaw[order]

    # Clip query into nav time range to avoid edge NaNs in interp
    tmin, tmax = t[0], t[-1]
    tsq = np.clip(np.asarray(ts_query, dtype=float), tmin, tmax)

    def interp_or_const(vals, const=np.nan):
        if np.all(np.isnan(vals)):
            return np.full_like(tsq, const, dtype=float)
        return np.interp(tsq, t, vals)

    lat_i = interp_or_const(lat)
    lon_i = interp_or_const(lon)
    depth_i = interp_or_const(depth, const=np.nan)
    alt_i = interp_or_const(alt, const=np.nan)
    yaw_i = interp_or_const(yaw, const=np.nan)
    # Ensure 0..360 wrap
    yaw_i = np.mod(yaw_i, 360.0)

    return {
        "ts": tsq,
        "lat": lat_i,
        "lon": lon_i,
        "depth": depth_i,
        "alt": alt_i,
        "yaw": yaw_i,
    }


def load_mbes_bounds(geotiff_path):
    """
    Return: corners in (lat,lon) list and bbox (minLon, minLat, maxLon, maxLat).
    """
    with rasterio.open(geotiff_path) as src:
        b = src.bounds  # left,bottom,right,top
        tf = Transformer.from_crs(src.crs, 4326, always_xy=True)
        corners_xy = [
            (b.left, b.bottom),
            (b.right, b.bottom),
            (b.right, b.top),
            (b.left, b.top),
        ]
        corners_ll = []
        for x, y in corners_xy:
            lon, lat = tf.transform(x, y)
            corners_ll.append((lat, lon))
        lats = [c[0] for c in corners_ll]
        lons = [c[1] for c in corners_ll]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        return corners_ll, bbox


def swath_half_width_m(altitude_m, fov_deg):
    """
    Across-track half width (meters) for a symmetric FOV at given altitude above bottom.
    """
    if np.isnan(altitude_m) or altitude_m <= 0:
        return np.nan
    return float(altitude_m * np.tan(np.deg2rad(fov_deg) / 2.0))


def draw_swath_lines(mapobj, lat, lon, yaw_deg, alt, fov_deg, step):
    """
    Draw short line segments perpendicular to heading, sized by swath half-width.
    """
    g = Geod(ellps="WGS84")
    n = len(lat)
    idxs = np.arange(0, n, max(1, n // max(1, step)))
    for i in idxs:
        a = swath_half_width_m(alt[i], fov_deg)
        if not np.isfinite(a) or a <= 0:
            continue
        bearing = yaw_deg[i]  # along-track
        # perpendicular bearings:
        left_brg = (bearing - 90.0) % 360.0
        right_brg = (bearing + 90.0) % 360.0
        # endpoints at +/- half swath
        _, _, lonL, latL = g.fwd(lon[i], lat[i], left_brg, a)
        _, _, lonR, latR = g.fwd(lon[i], lat[i], right_brg, a)
        folium.PolyLine(
            [(latL, lonL), (latR, lonR)],
            color="yellow",
            weight=2,
            opacity=0.7,
            tooltip=f"Swath ~{2*a:.1f} m (alt={alt[i]:.1f} m)",
        ).add_to(mapobj)


def build_map(nav_df, h5_ranges, interp_union, mbes_corners, mbes_bbox, out_html):
    # Center
    lat_c = nav_df["lat"].mean()
    lon_c = nav_df["lon"].mean()
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=14,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    # Satellite layer
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Tools
    MousePosition(
        position="bottomright",
        separator=" , ",
        prefix="Lat/Lng:",
        lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 6);}",
    ).add_to(m)
    MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    LocateControl(position="topright").add_to(m)

    # MBES coverage polygon
    mbes_fg = folium.FeatureGroup(name="MBES Coverage", show=True)
    folium.Polygon(
        locations=mbes_corners,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.15,
        weight=2,
        tooltip=f"MBES Bounds\nLat [{mbes_bbox[1]:.6f}, {mbes_bbox[3]:.6f}]\nLon [{mbes_bbox[0]:.6f}, {mbes_bbox[2]:.6f}]",
    ).add_to(mbes_fg)
    mbes_fg.add_to(m)

    # Full navigation trajectory (grey)
    coords_all = list(zip(nav_df["lat"], nav_df["lon"]))
    folium.PolyLine(
        coords_all,
        color="#888888",
        weight=2,
        opacity=0.7,
        tooltip=f"Full navigation track ({len(coords_all)} pts)",
    ).add_to(folium.FeatureGroup(name="Full Track", show=True).add_to(m))

    # Active transect(s) from H5 ranges
    tran_fg = folium.FeatureGroup(name="Active UHI Transect(s)", show=True)
    swath_fg = folium.FeatureGroup(name="UHI Swath Lines", show=True)

    # For each H5 file range, extract the corresponding segment
    for r in h5_ranges:
        mask = (interp_union["ts"] >= r["tmin"]) & (interp_union["ts"] <= r["tmax"])
        if not np.any(mask):
            continue
        seg = {k: v[mask] for k, v in interp_union.items()}
        coords_seg = list(zip(seg["lat"], seg["lon"]))

        # Colored transect line
        folium.PolyLine(
            coords_seg,
            color="red",
            weight=4,
            opacity=0.9,
            tooltip=f"{r['file']}  (frames: {r['n']})\n{pd.to_datetime(r['tmin'], unit='s')} → {pd.to_datetime(r['tmax'], unit='s')}",
        ).add_to(tran_fg)

        # Start/End
        if len(coords_seg) >= 1:
            folium.CircleMarker(
                coords_seg[0],
                radius=6,
                color="green",
                fill=True,
                fill_opacity=1.0,
                tooltip=f"START: {r['file']}",
            ).add_to(tran_fg)
            folium.CircleMarker(
                coords_seg[-1],
                radius=6,
                color="red",
                fill=True,
                fill_opacity=1.0,
                tooltip=f"END: {r['file']}",
            ).add_to(tran_fg)

        # Swath lines along this segment
        draw_swath_lines(
            swath_fg,
            seg["lat"],
            seg["lon"],
            seg["yaw"],
            seg["alt"],
            CAMERA_FOV_DEG,
            TARGET_SWATH_LINES,
        )

    tran_fg.add_to(m)
    swath_fg.add_to(m)

    # Stats overlay
    stats_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; width: 360px;
                background-color: white; border: 1px solid #777; z-index: 9999;
                font-size: 12px; padding: 10px; border-radius: 6px;">
      <h4 style="margin: 0 0 6px 0;">HSI / MBES Overview</h4>
      <b>H5 Files:</b> {len(h5_ranges)}<br>
      {"<br>".join([f"• {r['file']} ({r['n']} frames)" for r in h5_ranges])}<br><br>
      <b>FOV:</b> {CAMERA_FOV_DEG}° (across-track)<br>
      <b>Time offset applied:</b> {TIME_OFFSET_SEC} s<br>
      <b>Nav points:</b> {len(nav_df)}<br>
      <b>Output:</b> {os.path.abspath(out_html)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    print(f"\n✅ Saved: {out_html}\nOpen it in your browser.")


def main():
    # 1) H5 files & timestamp ranges
    h5_files = list_h5_files(H5_FOLDER_OR_FILES)
    print(f"Found {len(h5_files)} H5 files:")
    for f in h5_files:
        print("  -", f)

    h5_ranges, all_ts = load_all_hsi_timestamp_ranges(h5_files)
    tmin_all = float(np.nanmin(all_ts))
    tmax_all = float(np.nanmax(all_ts))
    print(
        f"\nHSI union time: {pd.to_datetime(tmin_all, unit='s')} → {pd.to_datetime(tmax_all, unit='s')}"
    )

    # 2) Navigation CSV
    nav_df = load_nav_csv(NAV_CSV)

    # 3) Interpolate nav → union of all HSI timestamps (for a smooth union path)
    interp_union = interpolate_nav_to_times(nav_df, all_ts)

    # 4) MBES bounds
    mbes_corners, mbes_bbox = load_mbes_bounds(MBES_GEOTIFF)
    print(f"\nMBES bbox (lon/lat): {mbes_bbox}")

    # 5) Build interactive map
    build_map(nav_df, h5_ranges, interp_union, mbes_corners, mbes_bbox, OUTPUT_HTML)


if __name__ == "__main__":
    main()
