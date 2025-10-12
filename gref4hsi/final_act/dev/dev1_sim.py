"""
3D UHI simulation with multi-H5 segments, IMU & Camera frames, camera rays, and MBES surface.

Edits requested:
- Use the term "Altitude [m]" for units and axis labels.
- In the static time plot, show only two curves in the legend: "IMU" and "UHI".
- Remove the fan-range band; define UHI as the MINIMUM ray-to-bottom distance across the sampled fan.

Run:
    python dev3_sim_with_mbes.py
    python dev3_sim_with_mbes.py --frames 400 --mbes-step 8 --mbes-alpha 0.65
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
from matplotlib.collections import LineCollection  # 2D rays
from datetime import datetime, timezone

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Try GUI backends so Slider/Button are clickable
for _backend in ("Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        pass

# Import local project modules
# This file is in final_act/other/, so we go up 2 levels to reach final_act/
sys.path.append(str(Path(__file__).parent.parent))
from gref_pipeline import config  # noqa: E402
from utils.gref_pipeline import utils  # noqa: E402


# ================== User-tunable orientation & visualization ==================
# Read yaw convention from config to stay in sync with main.py
ORI_CONVENTION = getattr(
    config, "YAW_CONVENTION", "heading_from_north_cw"
)  # or "enu_yaw_from_east_ccw"
RPY_UNITS = "deg"  # "deg" or "rad"

CAMERA_FWD_OFFSET_M = 2.5  # Camera is 2.5 m ahead along BODY +X
AXIS_ARROW_LEN_M = 1.5  # Body-axis arrows length

# Camera frame (utils.build_ray_directions "test_eely" convention):
#   cam X = across-track, cam Y = along-track, cam Z = down
# Body frame:
#   body X = forward, body Y = starboard, body Z = up
# Mapping cam->body: X_c->+Y_b, Y_c->+X_b, Z_c->-Z_b
# NOW READING FROM CONFIG TO KEEP IN SYNC WITH MAIN.PY
R_B_FROM_CAM = np.array(config.ROTATION_HSI_TO_BODY, dtype=float)

# Rays
SHOW_RAYS_BY_DEFAULT = True
RAY_LENGTH_M = 8.0
MAX_RAYS_DRAWN = 61
RAYS_COLOR = (0.6, 0.1, 0.9)  # magenta-ish

# MBES surface
SHOW_MBES_BY_DEFAULT = True
MBES_CMAP = "Greys"  # "terrain", "cividis", etc.
# ==============================================================================


def _fmt_utc(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _extract_rpy(nav_like):
    """Return roll, pitch, yaw arrays. Accepts dict or DataFrame-like."""
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
    R_ENU_from_BODY for one sample.
    Order: Rz(yaw_enu) @ Ry(pitch) @ Rx(roll) mapping body->ENU.
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
        lon,
        lat,
        -depth,
        epsg_geo=config.EPSG_GEOGRAPHIC,
        epsg_ecef=config.EPSG_ECEF,
    )
    return utils.ecef_to_utm(
        x_ecef,
        y_ecef,
        z_ecef,
        epsg_utm=config.EPSG_MBES,
        epsg_ecef=config.EPSG_ECEF,
    )


def _load_segments(num_frames=None):
    """
    Returns:
      full_xyz: (x_all, y_all, z_all)
      segments: list of dicts
      combined: concatenated arrays and per-seg edges
      nav_df: DataFrame
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
        ts = utils.load_h5_timestamps(str(p))
        if len(ts) == 0:
            continue

        ts_used = ts
        if num_frames is not None and len(ts) > num_frames:
            idx = np.linspace(0, len(ts) - 1, num_frames, dtype=int)
            ts_used = ts[idx]

        interp_nav = utils.interpolate_navigation(nav_df, ts_used, time_offset=offset)

        # Prefer explicit CSV altitude if available
        if "altitude" in interp_nav:
            alt_imu = np.asarray(interp_nav["altitude"])
        elif "altitude_m" in interp_nav:
            alt_imu = np.asarray(interp_nav["altitude_m"])
        elif "depth" in interp_nav:
            # if depth is positive-down, altitude ~ -depth relative to surface ref
            alt_imu = -np.asarray(interp_nav["depth"])
        else:
            alt_imu = None

        xe, ye, ze = utils.geographic_to_ecef(
            interp_nav["longitude"],
            interp_nav["latitude"],
            -interp_nav["depth"],
            epsg_geo=config.EPSG_GEOGRAPHIC,
            epsg_ecef=config.EPSG_ECEF,
        )
        x_imu, y_imu, z_imu = utils.ecef_to_utm(
            xe,
            ye,
            ze,
            epsg_utm=config.EPSG_MBES,
            epsg_ecef=config.EPSG_ECEF,
        )

        roll, pitch, yaw = _extract_rpy(interp_nav)
        lookup_ts_used = ts_used + offset

        seg_list.append(
            {
                "name": p.name,
                "x_imu": np.asarray(x_imu),
                "y_imu": np.asarray(y_imu),
                "z_imu": np.asarray(z_imu),
                "roll": np.asarray(roll),
                "pitch": np.asarray(pitch),
                "yaw": np.asarray(yaw),
                "altitude_imu": alt_imu,
                "lookup_ts_used": lookup_ts_used,
                "hsi_start_utc": _fmt_utc(float(ts_used.min())),
                "hsi_end_utc": _fmt_utc(float(ts_used.max())),
                "lookup_start_utc": _fmt_utc(float(lookup_ts_used.min())),
                "lookup_end_utc": _fmt_utc(float(lookup_ts_used.max())),
            }
        )

    if not seg_list:
        raise ValueError("All H5 files had zero timestamps")

    seg_list.sort(key=lambda s: s["lookup_ts_used"].min())

    cmap = matplotlib.colormaps.get_cmap("tab20")
    cols = cmap(np.linspace(0, 1, len(seg_list)))
    for i, s in enumerate(seg_list):
        s["color"] = tuple(cols[i, :3])

    xs, ys, zs, tl, edges, names, colors = [], [], [], [], [], [], []
    rolls, pitchs, yaws, alts = [], [], [], []
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
        alts.append(
            np.asarray(s["altitude_imu"])
            if s["altitude_imu"] is not None
            else np.asarray(s["z_imu"])
        )
        acc += n
        edges.append(acc)
        names.append(s["name"])
        colors.append(s["color"])

    combined = {
        "x_imu": np.concatenate(xs),
        "y_imu": np.concatenate(ys),
        "z_imu": np.concatenate(zs),
        "altitude_imu": np.concatenate(alts),
        "lookup_ts": np.concatenate(tl),
        "roll": np.concatenate(rolls),
        "pitch": np.concatenate(pitchs),
        "yaw": np.concatenate(yaws),
        "seg_edges": np.array(edges, dtype=int),
        "seg_names": names,
        "seg_colors": colors,
    }
    return (x_all, y_all, z_all), seg_list, combined, nav_df


# -------------------------- MBES utilities --------------------------


def _read_mbes_geotiff_as_target_epsg(geotiff_path, target_epsg):
    """
    Read band-1 and reproject to target_epsg (e.g., EPSG:32632) if needed.
    Returns (arr, transform, epsg).
    """
    with rasterio.open(geotiff_path) as src:
        src_epsg = src.crs.to_epsg() if src.crs else None
        if src_epsg == target_epsg:
            arr = src.read(1, masked=True).filled(np.nan)
            return arr, src.transform, src_epsg
        # reproject to target
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
    """
    Build decimated X,Y,Z arrays (UTM meters) for plotting with plot_surface.
    Affine: x = c + a*col + b*row, y = f + d*col + e*row
    """
    h, w = arr.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    X = transform.c + cc * transform.a + rr * transform.b
    Y = transform.f + cc * transform.d + rr * transform.e
    # center of pixel
    X = X + transform.a * 0.5
    Y = Y + transform.e * 0.5
    # decimate
    Xd = X[::step, ::step]
    Yd = Y[::step, ::step]
    Zd = arr[::step, ::step]
    return Xd, Yd, Zd


# -------------------------- Plotting --------------------------


def _segment_index_from_global(i, seg_edges):
    return int(np.searchsorted(seg_edges, i, side="right"))


def plot_with_mbes_and_rays(
    full_xyz,
    segments,
    combined,
    camera_calib,
    mbes_surface=None,
    mbes_alpha=0.6,
    title_extra="",
):
    """
    3D plot:
      - MBES surface (optional)
      - Full mission in gray
      - Static colored segments
      - IMU + Camera markers
      - Body axes
      - Camera rays (line segments)
      - Slider, Play/Pause, Rays on/off, MBES on/off

    Side-view (separate figure):
      - 2D starboard view in body frame, vehicle fixed at (0,0)
      - x forward to the right, z down
      - camera and rays projected to XZ
    """
    xa, ya, za = full_xyz
    xi, yi, zi = combined["x_imu"], combined["y_imu"], combined["z_imu"]
    t_lookup = combined["lookup_ts"]
    roll_arr, pitch_arr, yaw_arr = combined["roll"], combined["pitch"], combined["yaw"]
    seg_edges = combined["seg_edges"]
    seg_names = combined["seg_names"]
    seg_colors = combined["seg_colors"]

    n_total = len(xi)
    global_start = _fmt_utc(float(t_lookup.min()))
    global_end = _fmt_utc(float(t_lookup.max()))

    # Prepare camera rays in camera frame once
    n_slits = int(camera_calib["w"])
    rays_cam_full = utils.build_ray_directions(camera_calib, n_slits)  # (W,3)
    if n_slits <= MAX_RAYS_DRAWN:
        sel = np.arange(n_slits)
    else:
        sel = np.linspace(0, n_slits - 1, MAX_RAYS_DRAWN, dtype=int)
    rays_cam = rays_cam_full[sel, :]

    # Figure/axes (3D main)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d", position=[0.05, 0.18, 0.9, 0.77])

    # MBES surface
    mbes_state = [SHOW_MBES_BY_DEFAULT and (mbes_surface is not None)]
    mbes_artist = [None]
    if mbes_surface is not None and mbes_state[0]:
        Xm, Ym, Zm = mbes_surface
        mbes_artist[0] = ax.plot_surface(
            Xm,
            Ym,
            Zm,
            cmap=MBES_CMAP,
            linewidth=0,
            antialiased=False,
            alpha=mbes_alpha,
            zorder=0,
        )

    # Full mission in gray
    ax.plot(xa, ya, za, linewidth=1.0, color="gray", alpha=0.45, label="Full mission")

    # Colored segments
    for s in segments:
        ax.plot(
            s["x_imu"],
            s["y_imu"],
            s["z_imu"],
            linewidth=2.0,
            color=s["color"],
            alpha=0.95,
            label=s["name"],
        )

    # Moving markers
    (imu_pt,) = ax.plot(
        [], [], [], "o", markersize=9, color="black", zorder=12, label="IMU"
    )
    (cam_pt,) = ax.plot(
        [], [], [], "o", markersize=8, color="magenta", zorder=12, label="Camera"
    )

    # Body-axis arrows
    (x_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="red", zorder=11, label="Body X"
    )
    (y_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="green", zorder=11, label="Body Y"
    )
    (z_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="blue", zorder=11, label="Body Z"
    )

    # Rays
    rays_coll = [None]
    rays_state = [SHOW_RAYS_BY_DEFAULT]

    # Info box
    info = ax.text2D(
        0.02,
        0.985,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9),
        family="monospace",
    )

    # Cube-like limits
    rng = np.array([xa.max() - xa.min(), ya.max() - ya.min(), za.max() - za.min()])
    half = rng.max() / 2.0
    cxm = (xa.max() + xa.min()) / 2.0
    cym = (ya.max() + ya.min()) / 2.0
    czm = (za.max() + za.min()) / 2.0
    ax.set_xlim(cxm - half, cxm + half)
    ax.set_ylim(cym - half, cym + half)
    ax.set_zlim(czm - half, czm + half)

    ttl = f"Mission + MBES + UHI rays  {title_extra}\nGlobal lookup: {global_start} – {global_end}"
    ax.set_title(ttl, pad=10)
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_zlabel("Altitude [m]")  # <- unit wording updated
    ax.legend(loc="upper right", fontsize=8)

    # ----------------- 2D side view figure (vehicle at center) -----------------
    SIDE_X_HALF = 20.0
    SIDE_Z_HALF = 15.0
    SIDE_Y_SLICE = 1.0
    SIDE_MAX_POINTS = 6000

    side_fig, side_ax = plt.subplots(figsize=(10, 6))
    side_ax.set_title("Starboard side view — body frame (x→ forward, z↓ down)")
    side_ax.set_xlabel("Body X forward [m]")
    side_ax.set_ylabel("Body Z down [m]")
    side_ax.set_xlim(-SIDE_X_HALF, SIDE_X_HALF)
    side_ax.set_ylim(-SIDE_Z_HALF, SIDE_Z_HALF)
    side_ax.invert_yaxis()
    side_ax.set_aspect("equal", adjustable="box")

    (imu2_pt,) = side_ax.plot([], [], "o", markersize=7, color="black", label="IMU")
    (cam2_pt,) = side_ax.plot(
        [], [], "o", markersize=7, color="magenta", label="Camera"
    )
    (x2_axis_line,) = side_ax.plot([], [], linewidth=2.0, color="red", label="Body X")
    (z2_axis_line,) = side_ax.plot([], [], linewidth=2.0, color="blue", label="Body Z")
    rays2_coll = [None]
    side_mbes_scatter = [None]
    side_mbes_points = None
    if mbes_surface is not None:
        Xm, Ym, Zm = mbes_surface
        Xf = Xm.ravel()
        Yf = Ym.ravel()
        Zf = Zm.ravel()
        ok = np.isfinite(Zf)
        side_mbes_points = np.column_stack([Xf[ok], Yf[ok], Zf[ok]])
    side_ax.legend(loc="upper right", fontsize=8)

    def _update_frame(i):
        i = int(i)
        px, py, pz = xi[i], yi[i], zi[i]

        R_bw = _body_to_enu_rotation(
            _to_radians(roll_arr[i]), _to_radians(pitch_arr[i]), yaw_arr[i]
        )

        cam_off_body = np.array([CAMERA_FWD_OFFSET_M, 0.0, 0.0])
        cam_off_enu = R_bw @ cam_off_body
        cx, cy, cz = px + cam_off_enu[0], py + cam_off_enu[1], pz + cam_off_enu[2]

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

        seg_idx = _segment_index_from_global(i, seg_edges)
        imu_pt.set_color(seg_colors[seg_idx])

        # Rays: r_world = (R_bw @ R_B_FROM_CAM) @ r_cam
        if rays_coll[0] is not None:
            try:
                rays_coll[0].remove()
            except Exception:
                pass
            rays_coll[0] = None

        if rays_state[0]:
            R_wc = R_bw @ R_B_FROM_CAM
            d_world = (R_wc @ rays_cam.T).T
            d_world = d_world / np.linalg.norm(d_world, axis=1, keepdims=True)

            starts = np.column_stack(
                [[cx] * len(d_world), [cy] * len(d_world), [cz] * len(d_world)]
            )
            ends = starts + d_world * RAY_LENGTH_M
            segments = np.stack([starts, ends], axis=1)

            up = np.array([0.0, 0.0, 1.0])
            dots = d_world @ up
            bad = dots > 0  # upward
            colors = np.repeat([RAYS_COLOR], len(d_world), axis=0).astype(float)
            if np.any(bad):
                colors[bad] = (1.0, 0.0, 0.0)

            coll = Line3DCollection(segments, colors=colors, linewidths=0.8, zorder=10)
            rays_coll[0] = coll
            ax.add_collection3d(coll)
            bad_count = int(np.sum(bad))
            rays_line = f"Rays: ON  (count={len(d_world)})  | bad(up)={bad_count}"
        else:
            rays_line = f"Rays: OFF (count={len(rays_cam)})"

        mbes_line = (
            "MBES: ON" if mbes_state[0] and (mbes_surface is not None) else "MBES: OFF"
        )

        info.set_text(
            f"Idx {i}/{len(xi)-1} | {seg_names[seg_idx]}\n"
            f"Lookup UTC: {_fmt_utc(float(t_lookup[i]))}\n"
            f"RPY ({RPY_UNITS}): roll={roll_arr[i]:.3f}, pitch={pitch_arr[i]:.3f}, yaw={yaw_arr[i]:.3f}\n"
            f"{rays_line}   |   {mbes_line}"
        )

        # -------- 2D side view --------
        imu2_pt.set_data([0.0], [0.0])
        cam2_pt.set_data([CAMERA_FWD_OFFSET_M], [0.0])
        x2_axis_line.set_data([0.0, AXIS_ARROW_LEN_M], [0.0, 0.0])
        z2_axis_line.set_data([0.0, 0.0], [0.0, AXIS_ARROW_LEN_M])

        if rays2_coll[0] is not None:
            try:
                rays2_coll[0].remove()
            except Exception:
                pass
            rays2_coll[0] = None

        if rays_state[0]:
            d_body = (R_B_FROM_CAM @ rays_cam.T).T
            d_body = d_body / np.linalg.norm(d_body, axis=1, keepdims=True)
            starts_b = np.column_stack(
                [np.full(len(d_body), CAMERA_FWD_OFFSET_M), np.zeros(len(d_body))]
            )
            ends_b = np.column_stack(
                [
                    starts_b[:, 0] + d_body[:, 0] * RAY_LENGTH_M,
                    0.0 - d_body[:, 2] * RAY_LENGTH_M,
                ]
            )
            segs2 = np.stack([starts_b, ends_b], axis=1)
            rays2 = LineCollection(
                segs2, colors=[RAYS_COLOR], linewidths=1.0, alpha=0.95
            )
            rays2_coll[0] = rays2
            side_ax.add_collection(rays2)

        if (
            (mbes_surface is not None)
            and mbes_state[0]
            and (side_mbes_points is not None)
        ):
            R_wb = R_bw.T
            rel = side_mbes_points - np.array([px, py, pz])[None, :]
            p_body = rel @ R_wb.T
            yb = p_body[:, 1]
            zb_down = -p_body[:, 2]
            xb = p_body[:, 0]
            mask = (
                (np.abs(yb) <= 1.0) & (np.abs(xb) <= 20.0) & (np.abs(zb_down) <= 15.0)
            )
            xb = xb[mask]
            zb_down = zb_down[mask]
            if xb.size > 6000:
                step = max(1, xb.size // 6000)
                xb = xb[::step]
                zb_down = zb_down[::step]
            if side_mbes_scatter[0] is None:
                side_mbes_scatter[0] = side_ax.scatter(
                    xb, zb_down, s=6, c="gray", alpha=0.8, edgecolors="none"
                )
            else:
                side_mbes_scatter[0].set_offsets(np.column_stack([xb, zb_down]))

        return imu_pt, cam_pt, x_axis_line, y_axis_line, z_axis_line, info

    # Controls
    ax_slider = fig.add_axes([0.15, 0.06, 0.45, 0.03])
    slider = Slider(ax_slider, "Index", 0, n_total - 1, valinit=0, valstep=1)
    ax_btn_play = fig.add_axes([0.62, 0.06, 0.1, 0.04])
    btn_play = Button(ax_btn_play, "Play/Pause")
    ax_btn_rays = fig.add_axes([0.74, 0.06, 0.1, 0.04])
    btn_rays = Button(ax_btn_rays, "Rays On/Off")
    ax_btn_mbes = fig.add_axes([0.86, 0.06, 0.1, 0.04])
    btn_mbes = Button(ax_btn_mbes, "MBES On/Off")

    anim_running = [True]
    anim = FuncAnimation(
        fig, _update_frame, frames=n_total, interval=60, blit=False, repeat=True
    )
    fig._anim = anim  # keep ref

    def on_slider_change(val):
        anim.event_source.stop()
        anim_running[0] = False
        _update_frame(int(slider.val))
        fig.canvas.draw_idle()
        side_fig.canvas.draw_idle()

    def on_play_pause(event):
        if anim_running[0]:
            anim.event_source.stop()
            anim_running[0] = False
        else:
            anim.event_source.start()
            anim_running[0] = True

    def on_toggle_rays(event):
        rays_state[0] = not rays_state[0]
        _update_frame(int(slider.val))
        fig.canvas.draw_idle()
        side_fig.canvas.draw_idle()

    def on_toggle_mbes(event):
        mbes_state[0] = not mbes_state[0]
        if mbes_surface is None:
            return
        if mbes_artist[0] is not None:
            try:
                mbes_artist[0].remove()
            except Exception:
                pass
            mbes_artist[0] = None
        if mbes_state[0]:
            Xm, Ym, Zm = mbes_surface
            mbes_artist[0] = ax.plot_surface(
                Xm,
                Ym,
                Zm,
                cmap=MBES_CMAP,
                linewidth=0,
                antialiased=False,
                alpha=mbes_alpha,
                zorder=0,
            )
        fig.canvas.draw_idle()
        side_fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)
    btn_rays.on_clicked(on_toggle_rays)
    btn_mbes.on_clicked(on_toggle_mbes)

    _update_frame(0)
    plt.show()
    return fig, ax


# ===================== DEM helpers & ray intersection =====================


def _rc_from_xy(transform, x, y):
    a, b, c, d, e, f = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )
    X = x - (c + 0.5 * (a + b))
    Y = y - (f + 0.5 * (d + e))
    M = np.array([[a, b], [d, e]], dtype=float)
    try:
        col_f, row_f = np.linalg.solve(M, np.array([X, Y]))
    except np.linalg.LinAlgError:
        col_f = X / (a if a != 0 else 1.0)
        row_f = Y / (e if e != 0 else -1.0)
    return float(row_f), float(col_f)


def _bilinear_z_at(x, y, dem, transform):
    row_f, col_f = _rc_from_xy(transform, x, y)
    r0 = int(np.floor(row_f))
    c0 = int(np.floor(col_f))
    r1 = r0 + 1
    c1 = c0 + 1
    h, w = dem.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return np.nan
    z00 = dem[r0, c0]
    z01 = dem[r0, c1]
    z10 = dem[r1, c0]
    z11 = dem[r1, c1]
    if not (
        np.isfinite(z00) and np.isfinite(z01) and np.isfinite(z10) and np.isfinite(z11)
    ):
        return np.nan
    fr = row_f - r0
    fc = col_f - c0
    z0 = z00 * (1 - fc) + z01 * fc
    z1 = z10 * (1 - fc) + z11 * fc
    return z0 * (1 - fr) + z1 * fr


def _intersect_ray_dem(
    cam_xyz, d_world, dem, transform, t_max=60.0, step=0.5, bisect_iters=8
):
    cx, cy, cz = cam_xyz
    dx, dy, dz = d_world
    if dz >= 0:  # ray must go down
        return np.nan

    z_surf0 = _bilinear_z_at(cx, cy, dem, transform)
    if not np.isfinite(z_surf0):
        return np.nan
    f_prev = cz - z_surf0
    if f_prev <= 0:
        return 0.0

    t = step
    while t <= t_max:
        x = cx + dx * t
        y = cy + dy * t
        z = cz + dz * t
        z_surf = _bilinear_z_at(x, y, dem, transform)
        if not np.isfinite(z_surf):
            t += step
            continue
        f = z - z_surf
        if f <= 0.0:
            lo = t - step
            hi = t
            f_lo = f_prev
            f_hi = f
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                xm = cx + dx * mid
                ym = cy + dy * mid
                zm = cz + dz * mid
                zsm = _bilinear_z_at(xm, ym, dem, transform)
                if not np.isfinite(zsm):
                    hi = mid
                    continue
                fm = zm - zsm
                if fm > 0:
                    lo, f_lo = mid, fm
                else:
                    hi, f_hi = mid, fm
            return 0.5 * (lo + hi)
        f_prev = f
        t += step
    return np.nan


def _camera_pose_series(combined):
    xi, yi, zi = combined["x_imu"], combined["y_imu"], combined["z_imu"]
    roll_arr, pitch_arr, yaw_arr = combined["roll"], combined["pitch"], combined["yaw"]
    N = len(xi)
    cam_xyz = np.zeros((N, 3), dtype=float)
    R_bw_all = []
    for i in range(N):
        R_bw = _body_to_enu_rotation(
            _to_radians(roll_arr[i]), _to_radians(pitch_arr[i]), yaw_arr[i]
        )
        R_bw_all.append(R_bw)
        cam_off_body = np.array([CAMERA_FWD_OFFSET_M, 0.0, 0.0])
        cam_off_world = R_bw @ cam_off_body
        cam_xyz[i, :] = np.array([xi[i], yi[i], zi[i]]) + cam_off_world
    return cam_xyz, R_bw_all


def plot_altitude_vs_time(
    segments,
    combined,
    mbes_arr,
    mbes_transform,
    camera_calib,
    n_rays_sample=31,
    t_max=60.0,
    step=0.5,
):
    """
    STATIC figure:
      - "IMU": altitude from CSV (fallback to z_imu), in meters.
      - "UHI": MINIMUM ray-to-bottom distance across the sampled fan, in meters.
      - No fan-range band, legend shows only "IMU" and "UHI".
    """
    import matplotlib.dates as mdates

    times = combined["lookup_ts"]
    t_dt = [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in times]

    # 1) IMU altitude
    alt_imu = np.asarray(combined["altitude_imu"], dtype=float)

    # 2) Camera positions + orientation per frame
    cam_xyz, R_bw_all = _camera_pose_series(combined)

    # 3) Rays in camera frame and subset
    W = int(camera_calib["w"])
    rays_cam = utils.build_ray_directions(camera_calib, W)  # (W,3)
    sel = (
        np.arange(W, dtype=int)
        if n_rays_sample >= W
        else np.linspace(0, W - 1, n_rays_sample, dtype=int)
    )
    rays_cam = rays_cam[sel, :]
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)

    # 4) For each frame, rotate rays, intersect, and take MIN
    uhi_min = np.full(len(times), np.nan, dtype=float)
    for i in range(len(times)):
        R_bw = R_bw_all[i]
        R_wc = R_bw @ R_B_FROM_CAM
        d_world = (R_wc @ rays_cam.T).T
        d_world = d_world / np.linalg.norm(d_world, axis=1, keepdims=True)
        ranges = []
        for d in d_world:
            r = _intersect_ray_dem(
                cam_xyz[i, :],
                d,
                mbes_arr,
                mbes_transform,
                t_max=float(t_max),
                step=float(step),
                bisect_iters=8,
            )
            if np.isfinite(r):
                ranges.append(r)
        uhi_min[i] = np.nanmin(ranges) if len(ranges) else np.nan

    # 5) Plot, legend = only two entries
    fig, ax = plt.subplots(figsize=(12, 5))
    (imu_line,) = ax.plot(t_dt, alt_imu, lw=1.8, color="black", label="IMU")
    (uhi_line,) = ax.plot(t_dt, uhi_min, lw=1.8, color="magenta", label="UHI")

    # File boundaries for readability (no legend handles created)
    edges = combined["seg_edges"]
    names = combined["seg_names"]
    for k, e in enumerate(edges[:-1]):  # skip final end edge
        t_v = t_dt[e - 1] if e - 1 < len(t_dt) else t_dt[-1]
        ax.axvline(t_v, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax.text(
            t_v,
            ax.get_ylim()[1],
            f" {names[k+1]}",
            va="top",
            ha="left",
            fontsize=8,
            rotation=90,
            color="gray",
        )

    ax.set_title("Altitude vs Time — IMU and UHI")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude [m]")  # <- unit wording updated
    ax.grid(True, alpha=0.3)

    # Legend only IMU and UHI
    ax.legend(handles=[imu_line, uhi_line], labels=["IMU", "UHI"], loc="best")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ------------------------------ Main ------------------------------


def main():
    parser = argparse.ArgumentParser(description="3D UHI simulation with MBES & rays")
    parser.add_argument(
        "--frames", type=int, default=None, help="Optional per-H5 subsample count"
    )
    parser.add_argument(
        "--mbes-step",
        type=int,
        default=8,
        help="MBES surface decimation step (bigger = lighter)",
    )
    parser.add_argument(
        "--mbes-alpha", type=float, default=0.6, help="MBES surface transparency (0..1)"
    )
    args = parser.parse_args()

    # Load segments / trajectory
    full_xyz, segments, combined, nav_df = _load_segments(num_frames=args.frames)

    # Print per-file ranges
    print("\n=== Per-file HSI time bounds (used frames) ===")
    for s in segments:
        print(f"{s['name']}")
        print(f"  HSI UTC:    {s['hsi_start_utc']} to {s['hsi_end_utc']}")
        print(f"  Lookup UTC: {s['lookup_start_utc']} to {s['lookup_end_utc']}")
    print("")
    print(f"Orientation convention: {ORI_CONVENTION} (from config), units: {RPY_UNITS}")
    print(f"Camera offset: {CAMERA_FWD_OFFSET_M} m along body +X")
    print("Camera->Body rotation (R_B_FROM_CAM) from config.ROTATION_HSI_TO_BODY:")
    print(R_B_FROM_CAM)

    # Camera intrinsics for rays
    camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
    print(
        f"Camera calib: f={camera_calib['f']:.3f}, cx={camera_calib['cx']:.3f}, w={int(camera_calib['w'])}"
    )

    # MBES surface in UTM (EPSG_MBES)
    mbes_surface = None
    mbes_arr, mbes_tf = None, None
    try:
        mbes_arr, mbes_tf, used_epsg = _read_mbes_geotiff_as_target_epsg(
            config.MBES_GEOTIFF, config.EPSG_MBES
        )
        Xm, Ym, Zm = _surface_from_raster(
            mbes_arr, mbes_tf, step=max(1, int(args.mbes_step))
        )
        mbes_surface = (Xm, Ym, Zm)
        print(
            f"MBES: loaded GeoTIFF '{config.MBES_GEOTIFF}' (EPSG:{used_epsg}), surface size {Xm.shape} after decimation"
        )
    except Exception as e:
        print(f"⚠️  MBES load failed: {e}")
        print("    Continuing without MBES surface.")

    # Static Altitude vs Time (only if DEM available)
    if (mbes_arr is not None) and (mbes_tf is not None):
        try:
            plot_altitude_vs_time(
                segments=segments,
                combined=combined,
                mbes_arr=mbes_arr,
                mbes_transform=mbes_tf,
                camera_calib=camera_calib,
                n_rays_sample=31,
                t_max=60.0,
                step=0.5,
            )
        except Exception as e:
            print(f"⚠️  Altitude-vs-time plot failed: {e}")
    else:
        print("ℹ️  Skipping altitude-vs-time plot (no MBES DEM loaded).")

    title_extra = f"(offset {float(config.TIME_OFFSET_SEC):+.1f}s; rays ≤ {MAX_RAYS_DRAWN}; MBES step={args.mbes_step})"
    plot_with_mbes_and_rays(
        full_xyz,
        segments,
        combined,
        camera_calib,
        mbes_surface=mbes_surface,
        mbes_alpha=float(args.mbes_alpha),
        title_extra=title_extra,
    )


if __name__ == "__main__":
    main()
