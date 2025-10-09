"""
3D UHI simulation + overlay of *georeferenced HSI* footprints saved by main.py,
with a synced 2D top-down window.

What it shows
-------------
• 3D: MBES surface, mission track, per-H5 colored segments, IMU & camera,
      camera body axes, and the CURRENT FRAME georeferenced footprint (from H5).
      Optional: show decimated "all frames" georef footprints for each H5.

• 2D: MBES raster (UTM), full mission track, CURRENT FRAME georef footprint.
      Stays in sync with 3D slider / play/pause.

Inputs/Conventions
------------------
• Uses config.py (NAV_CSV, MBES_GEOTIFF, H5_FOLDER, TIME_OFFSET_SEC, EPSG_*,
  ROTATION_HSI_TO_BODY, etc.)
• Uses utils.py for CSV load, timestamp load, interpolation, ECEF/UTM transforms,
  camera calibration and ray model (for orientation consistency only).
• Orientation math matches your sims: body +X forward, +Y starboard, +Z up.
  Yaw convention read from config.YAW_CONVENTION if present.

Run
---
    python dev5_sim_georef_overlay.py
    python dev5_sim_georef_overlay.py --frames 400 --mbes-step 8 --mbes-alpha 0.7
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from datetime import datetime, timezone

import h5py
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Make widgets clickable when run as a script
for _backend in ("Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        pass

# Project imports
sys.path.append(str(Path(__file__).parent.parent))
import config  # noqa: E402
import x_gref4hsi_by_liu.utils.utils as utils  # noqa: E402


# ========================= Tunables =========================
ORI_CONVENTION = getattr(config, "YAW_CONVENTION", "heading_from_north_cw")
RPY_UNITS = "deg"
CAMERA_FWD_OFFSET_M = (
    float(config.TRANSLATION_BODY_TO_HSI[0])
    if hasattr(config, "TRANSLATION_BODY_TO_HSI")
    else 2.5
)
AXIS_ARROW_LEN_M = 1.5
R_B_FROM_CAM = np.array(config.ROTATION_HSI_TO_BODY, dtype=float)

MBES_CMAP_3D = "Greys"
MBES_ALPHA_3D_DEFAULT = 0.7  # slightly more visible per your request

# Georef visualization
GEOREF_COLOR = (0.95, 0.0, 0.95)
GEOREF_ALL_ALPHA = 0.25  # all-frames (decimated) faint overlay
GEOREF_ALL_DECIM = 40  # decimate factor for plotting "all" points
# ===========================================================


def _fmt_utc(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _extract_rpy(nav_like):
    if isinstance(nav_like, dict):
        cols = nav_like.keys()
        get = lambda k: np.asarray(nav_like[k])
    else:
        cols = nav_like.columns
        get = lambda k: nav_like[k].to_numpy()
    candidates = [
        ("roll", "pitch", "yaw"),
        ("roll_deg", "pitch_deg", "yaw_deg"),
        ("phi", "theta", "psi"),
    ]
    for r, p, y in candidates:
        if r in cols and p in cols and y in cols:
            return get(r), get(p), get(y)
    raise KeyError("roll/pitch/yaw columns not found.")


def _to_radians(arr):
    return np.deg2rad(arr) if RPY_UNITS == "deg" else arr


def _body_to_enu_rotation(roll_rad, pitch_rad, yaw_in):
    """
    R_ENU_from_BODY with ENU yaw from East CCW; supports heading input.
    """
    if ORI_CONVENTION == "heading_from_north_cw":
        yaw_val = (
            np.deg2rad(90.0 - yaw_in) if RPY_UNITS == "deg" else (np.pi / 2.0) - yaw_in
        )
    else:
        yaw_val = _to_radians(yaw_in)

    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_val), np.sin(yaw_val)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _full_mission_xyz(nav_df):
    lon = nav_df["longitude"].to_numpy()
    lat = nav_df["latitude"].to_numpy()
    depth = nav_df["depth"].to_numpy()
    x_ecef, y_ecef, z_ecef = utils.geographic_to_ecef(
        lon, lat, -depth, epsg_geo=config.EPSG_GEOGRAPHIC, epsg_ecef=config.EPSG_ECEF
    )
    return utils.ecef_to_utm(
        x_ecef, y_ecef, z_ecef, epsg_utm=config.EPSG_MBES, epsg_ecef=config.EPSG_ECEF
    )


def _load_segments(num_frames=None):
    """
    Build timeline (possibly subsampled) and keep mapping to original H5 frame ids.
    """
    nav_df = utils.load_csv_navigation(config.NAV_CSV, config.CSV_COLUMNS)
    x_all, y_all, z_all = _full_mission_xyz(nav_df)

    h5_folder = Path(config.H5_FOLDER)
    h5_files = sorted(h5_folder.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_folder}")

    offset = float(config.TIME_OFFSET_SEC)
    seg_list = []
    for p in h5_files:
        ts = utils.load_h5_timestamps(str(p))  # HSI frame times
        if len(ts) == 0:
            continue

        # Decide which frames we're visualizing (subsample optionally)
        if num_frames is not None and len(ts) > num_frames:
            idx_used = np.linspace(0, len(ts) - 1, num_frames, dtype=int)
        else:
            idx_used = np.arange(len(ts))
        ts_used = ts[idx_used]

        # Interpolate NAV to lookup times = ts_used + offset
        interp_nav = utils.interpolate_navigation(nav_df, ts_used, time_offset=offset)

        # IMU positions (UTM)
        xe, ye, ze = utils.geographic_to_ecef(
            interp_nav["longitude"],
            interp_nav["latitude"],
            -interp_nav["depth"],
            epsg_geo=config.EPSG_GEOGRAPHIC,
            epsg_ecef=config.EPSG_ECEF,
        )
        x_imu, y_imu, z_imu = utils.ecef_to_utm(
            xe, ye, ze, epsg_utm=config.EPSG_MBES, epsg_ecef=config.EPSG_ECEF
        )

        roll, pitch, yaw = _extract_rpy(interp_nav)
        lookup_ts_used = ts_used + offset

        seg_list.append(
            {
                "name": p.name,
                "path": str(p),
                "frame_ids": idx_used,  # <- mapping to original H5 frame index
                "x_imu": np.asarray(x_imu),
                "y_imu": np.asarray(y_imu),
                "z_imu": np.asarray(z_imu),
                "roll": np.asarray(roll),
                "pitch": np.asarray(pitch),
                "yaw": np.asarray(yaw),
                "lookup_ts_used": lookup_ts_used,
                "hsi_start_utc": _fmt_utc(float(ts_used.min())),
                "hsi_end_utc": _fmt_utc(float(ts_used.max())),
                "lookup_start_utc": _fmt_utc(float(lookup_ts_used.min())),
                "lookup_end_utc": _fmt_utc(float(lookup_ts_used.max())),
            }
        )

    if not seg_list:
        raise ValueError("All H5 files had zero timestamps")

    # Sort by first lookup time
    seg_list.sort(key=lambda s: s["lookup_ts_used"].min())

    # Colors
    cmap = matplotlib.colormaps.get_cmap("tab20")
    cols = cmap(np.linspace(0, 1, len(seg_list)))
    for i, s in enumerate(seg_list):
        s["color"] = tuple(cols[i, :3])

    # Combine for slider
    xs, ys, zs, tl, edges, names, colors = [], [], [], [], [], [], []
    rolls, pitchs, yaws, frame_ids_all = [], [], [], []
    acc = 0
    for s in seg_list:
        n = len(s["x_imu"])
        xs.append(s["x_imu"])
        ys.append(s["y_imu"])
        zs.append(s["z_imu"])
        tl.append(s["lookup_ts_used"])
        rolls.append(s["roll"])
        pitchs.append(s["pitch"])
        yaws.append(s["yaw"])
        frame_ids_all.append(s["frame_ids"])
        acc += n
        edges.append(acc)
        names.append(s["name"])
        colors.append(s["color"])

    combined = {
        "x_imu": np.concatenate(xs),
        "y_imu": np.concatenate(ys),
        "z_imu": np.concatenate(zs),
        "lookup_ts": np.concatenate(tl),
        "roll": np.concatenate(rolls),
        "pitch": np.concatenate(pitchs),
        "yaw": np.concatenate(yaws),
        "frame_ids": np.concatenate(
            frame_ids_all
        ),  # aligned with concatenated timeline
        "seg_edges": np.array(edges, dtype=int),
        "seg_names": names,
        "seg_colors": colors,
    }
    return (x_all, y_all, z_all), seg_list, combined, nav_df


# --------------------------- MBES helpers ---------------------------


def _read_mbes_geotiff_as_target_epsg(geotiff_path, target_epsg):
    with rasterio.open(geotiff_path) as src:
        src_epsg = src.crs.to_epsg() if src.crs else None
        if src_epsg == target_epsg:
            arr = src.read(1, masked=True).filled(np.nan)
            return arr, src.transform, src_epsg
        dst_crs = rasterio.crs.CRS.from_epsg(target_epsg)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        return dst, transform, target_epsg


def _surface_from_raster(arr, transform, step=8):
    h, w = arr.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    X = transform.c + cc * transform.a + rr * transform.b
    Y = transform.f + cc * transform.d + rr * transform.e
    X = X + transform.a * 0.5
    Y = Y + transform.e * 0.5
    return X[::step, ::step], Y[::step, ::step], arr[::step, ::step]


def _extent_from_transform(arr, transform):
    h, w = arr.shape
    x0 = transform.c
    y0 = transform.f
    x1 = x0 + transform.a * w + transform.b * h
    y1 = y0 + transform.d * w + transform.e * h
    xmin, xmax = (min(x0, x1), max(x0, x1))
    ymin, ymax = (min(y0, y1), max(y0, y1))
    return (xmin, xmax, ymin, ymax)


# ---------------------- Georef H5 (from main.py) ----------------------


def _load_georef_grouped_utm(h5_path, epsg_utm, epsg_ecef):
    """
    Read georeferenced hits saved by main.py and group by frame id.

    Returns a dict:
      {
        'frames': unique_frame_ids (sorted),
        'starts': start indices per frame (into *_sorted arrays),
        'counts': counts per frame,
        'ex': easting_sorted, 'ey': northing_sorted, 'ez': height_sorted,
        'px': pixel_sorted
      }
    or None if datasets not found.
    """
    ds_points = "processed/georef/points_ecef_crs"
    ds_frames = "processed/georef/frame_indices"
    ds_pixels = "processed/georef/ray_indices"  # contains pixel indices (as saved)

    try:
        with h5py.File(h5_path, "r") as f:
            if ds_points not in f or ds_frames not in f or ds_pixels not in f:
                return None
            pts_ecef = f[ds_points][()]  # (M,3)
            frames = f[ds_frames][()].astype(int)  # (M,)
            pixels = f[ds_pixels][()].astype(int)  # (M,)
    except Exception:
        return None

    if (
        pts_ecef.ndim != 2
        or pts_ecef.shape[1] != 3
        or frames.ndim != 1
        or pixels.ndim != 1
    ):
        return None
    M = pts_ecef.shape[0]
    if len(frames) != M or len(pixels) != M:
        return None

    # Convert all points ECEF -> UTM once
    ex, ey, ez = utils.ecef_to_utm(
        pts_ecef[:, 0],
        pts_ecef[:, 1],
        pts_ecef[:, 2],
        epsg_utm=epsg_utm,
        epsg_ecef=epsg_ecef,
    )

    # Sort by (frame, pixel) so each frame’s slits are in pixel order
    order = np.lexsort((pixels, frames))
    fs = frames[order]
    pxs = pixels[order]
    exs, eys, ezs = np.asarray(ex)[order], np.asarray(ey)[order], np.asarray(ez)[order]

    uniq, first_idx, counts = np.unique(fs, return_index=True, return_counts=True)

    return {
        "frames": uniq,
        "starts": first_idx,
        "counts": counts,
        "ex": exs,
        "ey": eys,
        "ez": ezs,
        "px": pxs,
    }


def _build_georef_for_segments(seg_list):
    """
    Load georef (grouped) for each H5 in seg_list (aligned order).
    Returns a list (same length as seg_list) with dicts or None.
    """
    out = []
    for s in seg_list:
        g = _load_georef_grouped_utm(s["path"], config.EPSG_MBES, config.EPSG_ECEF)
        if g is None:
            print(
                f"⚠️  No georef dataset in {Path(s['path']).name} (run main.py first?)"
            )
        else:
            print(
                f"✓ Georef loaded for {Path(s['path']).name}: {len(g['frames'])} frames with hits"
            )
        out.append(g)
    return out


# --------------------------- Plotting ---------------------------


def _segment_index_from_global(i, seg_edges):
    return int(np.searchsorted(seg_edges, i, side="right"))


def _local_index_in_segment(i, seg_idx, seg_edges):
    start = 0 if seg_idx == 0 else seg_edges[seg_idx - 1]
    return int(i - start)


def plot_with_georef(
    full_xyz,
    segments,
    combined,
    camera_calib,
    mbes_surface=None,
    mbes_raster=None,
    mbes_alpha=MBES_ALPHA_3D_DEFAULT,
    georef_grouped=None,
    title_extra="",
):
    """
    3D + 2D viewer with current-frame georeferenced footprint overlay.
    """
    xa, ya, za = full_xyz
    xi, yi, zi = combined["x_imu"], combined["y_imu"], combined["z_imu"]
    t_lookup = combined["lookup_ts"]
    roll_arr, pitch_arr, yaw_arr = combined["roll"], combined["pitch"], combined["yaw"]
    frame_ids = combined["frame_ids"]
    seg_edges = combined["seg_edges"]
    seg_names = combined["seg_names"]
    seg_colors = combined["seg_colors"]

    n_total = len(xi)
    global_start = _fmt_utc(float(t_lookup.min()))
    global_end = _fmt_utc(float(t_lookup.max()))

    # Figure 1: 3D
    fig3d = plt.figure(figsize=(13, 10))
    ax3d = fig3d.add_subplot(111, projection="3d", position=[0.05, 0.18, 0.9, 0.77])

    # MBES surface
    if mbes_surface is not None:
        Xm, Ym, Zm = mbes_surface
        ax3d.plot_surface(
            Xm,
            Ym,
            Zm,
            cmap=MBES_CMAP_3D,
            linewidth=0,
            antialiased=False,
            alpha=mbes_alpha,
            zorder=0,
        )

    # Full mission + segments
    ax3d.plot(xa, ya, za, linewidth=1.0, color="gray", alpha=0.45, label="Full mission")
    for s in segments:
        ax3d.plot(
            s["x_imu"],
            s["y_imu"],
            s["z_imu"],
            linewidth=2.0,
            color=s["color"],
            alpha=0.95,
            label=s["name"],
        )

    # Markers & axes
    (imu_pt,) = ax3d.plot(
        [], [], [], "o", markersize=9, color="black", zorder=12, label="IMU"
    )
    (cam_pt,) = ax3d.plot(
        [], [], [], "o", markersize=8, color="magenta", zorder=12, label="Camera"
    )
    (x_axis_line,) = ax3d.plot(
        [], [], [], linewidth=2.0, color="red", zorder=11, label="Body X"
    )
    (y_axis_line,) = ax3d.plot(
        [], [], [], linewidth=2.0, color="green", zorder=11, label="Body Y"
    )
    (z_axis_line,) = ax3d.plot(
        [], [], [], linewidth=2.0, color="blue", zorder=11, label="Body Z"
    )

    # Current-frame georef overlay (3D)
    georef_line3d = ax3d.plot(
        [], [], [], "-", color=GEOREF_COLOR, linewidth=1.5, label="HSI georef (current)"
    )[0]
    georef_pts3d = ax3d.plot([], [], [], ".", color=GEOREF_COLOR, markersize=3)[0]

    # Optional: decimated "all frames georef" (per segment color)
    show_all_georef = [False]
    all_georef_scats = []  # one per segment
    for s, g in zip(segments, georef_grouped or []):
        if g is None:
            all_georef_scats.append(None)
            continue
        # decimate
        mask = np.arange(len(g["ex"])) % max(1, int(GEOREF_ALL_DECIM)) == 0
        scat = ax3d.plot(
            g["ex"][mask],
            g["ey"][mask],
            g["ez"][mask],
            ".",
            color=s["color"],
            alpha=GEOREF_ALL_ALPHA,
            markersize=1,
        )[0]
        scat.set_visible(False)
        all_georef_scats.append(scat)

    # Info
    info3d = ax3d.text2D(
        0.02,
        0.985,
        "",
        transform=ax3d.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9),
        family="monospace",
    )

    # Limits
    rng = np.array([xa.max() - xa.min(), ya.max() - ya.min(), za.max() - za.min()])
    half = rng.max() / 2.0
    cxm = (xa.max() + xa.min()) / 2.0
    cym = (ya.max() + ya.min()) / 2.0
    czm = (za.max() + za.min()) / 2.0
    ax3d.set_xlim(cxm - half, cxm + half)
    ax3d.set_ylim(cym - half, cym + half)
    ax3d.set_zlim(czm - half, czm + half)

    ttl = f"Mission + MBES + HSI Georef  {title_extra}\nGlobal lookup: {global_start} – {global_end}"
    ax3d.set_title(ttl, pad=10)
    ax3d.set_xlabel("Easting (m)")
    ax3d.set_ylabel("Northing (m)")
    ax3d.set_zlabel("Height (m)")
    ax3d.legend(loc="upper right", fontsize=8)

    # Figure 2: 2D top-down
    fig2d, ax2d = plt.subplots(figsize=(10, 9))
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_title("Top-down (UTM) — MBES + Georeferenced HSI (current frame)")
    if mbes_raster is not None:
        arr, tf = mbes_raster
        extent = _extent_from_transform(arr, tf)
        im = ax2d.imshow(
            arr,
            extent=extent,
            origin="upper" if tf.e < 0 else "lower",
            cmap="cividis",
            alpha=0.9,
        )
        fig2d.colorbar(im, ax=ax2d, shrink=0.8, label="MBES elevation (m)")
    ax2d.plot(xa, ya, color="cyan", linewidth=1.2, alpha=0.7, label="Mission track")
    (georef_line2d,) = ax2d.plot(
        [], [], "-", color=GEOREF_COLOR, linewidth=1.5, label="HSI georef (current)"
    )
    (georef_pts2d,) = ax2d.plot([], [], ".", color=GEOREF_COLOR, markersize=3)
    ax2d.legend(loc="lower right")

    # Controls
    ax_slider = fig3d.add_axes([0.15, 0.06, 0.45, 0.03])
    slider = Slider(ax_slider, "Index", 0, n_total - 1, valinit=0, valstep=1)
    ax_btn_play = fig3d.add_axes([0.62, 0.06, 0.1, 0.04])
    btn_play = Button(ax_btn_play, "Play/Pause")
    ax_btn_all = fig3d.add_axes([0.74, 0.06, 0.1, 0.04])
    btn_all = Button(ax_btn_all, "All HSI On/Off")

    anim = FuncAnimation(
        fig3d, lambda i: None, frames=n_total, interval=60, blit=False, repeat=True
    )
    fig3d._anim = anim

    def _update(i):
        i = int(i)
        # IMU pose
        px, py, pz = xi[i], yi[i], zi[i]
        R_bw = _body_to_enu_rotation(
            _to_radians(roll_arr[i]), _to_radians(pitch_arr[i]), yaw_arr[i]
        )

        # Camera position
        cam_off_body = np.array([CAMERA_FWD_OFFSET_M, 0.0, 0.0])
        cam_enu = R_bw @ cam_off_body
        cx, cy, cz = px + cam_enu[0], py + cam_enu[1], pz + cam_enu[2]

        # Body axes lines
        L = AXIS_ARROW_LEN_M
        x_tip = np.array([px, py, pz]) + (R_bw @ np.array([L, 0.0, 0.0]))
        y_tip = np.array([px, py, pz]) + (R_bw @ np.array([0.0, L, 0.0]))
        z_tip = np.array([px, py, pz]) + (R_bw @ np.array([0.0, 0.0, L]))

        imu_pt.set_data([px], [py])
        imu_pt.set_3d_properties([pz])
        cam_pt.set_data([cx], [cy])
        cam_pt.set_3d_properties([cz])
        x_axis_line.set_data([px, x_tip[0]], [py, x_tip[1]])
        x_axis_line.set_3d_properties([pz, x_tip[2]])
        y_axis_line.set_data([px, y_tip[0]], [py, y_tip[1]])
        y_axis_line.set_3d_properties([pz, y_tip[2]])
        z_axis_line.set_data([px, z_tip[0]], [py, z_tip[1]])
        z_axis_line.set_3d_properties([pz, z_tip[2]])

        # Which segment & local frame is this?
        seg_idx = _segment_index_from_global(i, seg_edges)
        local_idx = _local_index_in_segment(i, seg_idx, seg_edges)
        imu_pt.set_color(seg_colors[seg_idx])

        # Current frame id in ORIGINAL H5:
        frame_id = segments[seg_idx]["frame_ids"][local_idx]

        # Pull this frame's georef from grouped store
        g = (
            georef_grouped[seg_idx]
            if (georef_grouped is not None and seg_idx < len(georef_grouped))
            else None
        )
        if g is not None:
            # Locate frame_id among g['frames']
            pos = np.searchsorted(g["frames"], frame_id)
            has = (pos < len(g["frames"])) and (g["frames"][pos] == frame_id)
            if has:
                start = int(g["starts"][pos])
                cnt = int(g["counts"][pos])
                xs = g["ex"][start : start + cnt]
                ys = g["ey"][start : start + cnt]
                zs = g["ez"][start : start + cnt]
                # 3D & 2D
                georef_line3d.set_data(xs, ys)
                georef_line3d.set_3d_properties(zs)
                georef_pts3d.set_data(xs, ys)
                georef_pts3d.set_3d_properties(zs)
                georef_line2d.set_data(xs, ys)
                georef_pts2d.set_data(xs, ys)
            else:
                # No hits recorded for this frame
                georef_line3d.set_data([], [])
                georef_line3d.set_3d_properties([])
                georef_pts3d.set_data([], [])
                georef_pts3d.set_3d_properties([])
                georef_line2d.set_data([], [])
                georef_pts2d.set_data([], [])
        else:
            georef_line3d.set_data([], [])
            georef_line3d.set_3d_properties([])
            georef_pts3d.set_data([], [])
            georef_pts3d.set_3d_properties([])
            georef_line2d.set_data([], [])
            georef_pts2d.set_data([], [])

        info3d.set_text(
            f"Idx {i}/{n_total-1} | {seg_names[seg_idx]}\n"
            f"Lookup UTC: {_fmt_utc(float(t_lookup[i]))}\n"
            f"RPY ({RPY_UNITS}): roll={roll_arr[i]:.3f}, pitch={pitch_arr[i]:.3f}, yaw={yaw_arr[i]:.3f}\n"
            f"Frame id: {int(frame_id)}"
        )
        return (
            imu_pt,
            cam_pt,
            x_axis_line,
            y_axis_line,
            z_axis_line,
            georef_line3d,
            georef_pts3d,
            georef_line2d,
            georef_pts2d,
            info3d,
        )

    def on_slider_change(val):
        anim.event_source.stop()
        _update(int(slider.val))
        fig3d.canvas.draw_idle()
        fig2d.canvas.draw_idle()

    def on_play_pause(event):
        if anim.event_source.running:
            anim.event_source.stop()
        else:
            anim.event_source.start()

    def on_toggle_all(event):
        show_all_georef[0] = not show_all_georef[0]
        for scat in all_georef_scats:
            if scat is not None:
                scat.set_visible(show_all_georef[0])
        fig3d.canvas.draw_idle()

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)
    btn_all.on_clicked(on_toggle_all)

    # Initialize
    _update(0)
    plt.show()
    return fig3d, ax3d, fig2d, ax2d


# ------------------------------- Main -------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="3D UHI + MBES + georeferenced HSI overlay (from main.py output)"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="Optional per-H5 subsample count"
    )
    parser.add_argument(
        "--mbes-step",
        type=int,
        default=8,
        help="MBES surface decimation step for 3D view",
    )
    parser.add_argument(
        "--mbes-alpha",
        type=float,
        default=MBES_ALPHA_3D_DEFAULT,
        help="MBES surface alpha",
    )
    args = parser.parse_args()

    full_xyz, segments, combined, _ = _load_segments(num_frames=args.frames)

    print("\n=== Per-file HSI time bounds (used frames) ===")
    for s in segments:
        print(f"{s['name']}")
        print(f"  HSI UTC:    {s['hsi_start_utc']} to {s['hsi_end_utc']}")
        print(f"  Lookup UTC: {s['lookup_start_utc']} to {s['lookup_end_utc']}")
    print("")
    print(f"Orientation convention (from config): {ORI_CONVENTION}")
    print(f"Camera offset: {CAMERA_FWD_OFFSET_M} m (BODY +X)")
    print("R_B_FROM_CAM = config.ROTATION_HSI_TO_BODY")
    print(R_B_FROM_CAM)

    # Camera intrinsics (not directly needed for georef draw, but keep printed for sanity)
    camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
    print(
        f"Camera calib: f={camera_calib['f']:.3f}, cx={camera_calib['cx']:.3f}, w={int(camera_calib['w'])}"
    )

    # MBES for 3D & 2D
    mbes_surface = None
    mbes_raster = None
    try:
        arr, tf, used_epsg = _read_mbes_geotiff_as_target_epsg(
            config.MBES_GEOTIFF, config.EPSG_MBES
        )
        Xm, Ym, Zm = _surface_from_raster(arr, tf, step=max(1, int(args.mbes_step)))
        mbes_surface = (Xm, Ym, Zm)
        mbes_raster = (arr, tf)
        print(
            f"MBES loaded: EPSG:{used_epsg} | surface {Xm.shape} (decimated), raster {arr.shape}"
        )
    except Exception as e:
        print(f"⚠️  MBES load failed: {e}")
        print("    Continuing without MBES.")

    # Load georef data produced by main.py for each H5 (aligned with segments)
    georef_grouped = _build_georef_for_segments(segments)

    title_extra = (
        f"(offset {float(config.TIME_OFFSET_SEC):+.1f}s; MBES step={args.mbes_step})"
    )
    plot_with_georef(
        full_xyz,
        segments,
        combined,
        camera_calib,
        mbes_surface=mbes_surface,
        mbes_raster=mbes_raster,
        mbes_alpha=float(args.mbes_alpha),
        georef_grouped=georef_grouped,
        title_extra=title_extra,
    )


if __name__ == "__main__":
    main()
