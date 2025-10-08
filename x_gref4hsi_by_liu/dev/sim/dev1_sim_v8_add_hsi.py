"""
3D UHI simulation with multi-H5 segments, IMU & Camera frames, camera rays, and MBES surface.
Now with a toggleable top-down RGB plot of georeferenced HSI (separate figure).

Requires:
  - config.py  (NAV_CSV, MBES_GEOTIFF, H5_FOLDER, TIME_OFFSET_SEC, EPSG_* ...)
  - utils.py   (load_csv_navigation, load_h5_timestamps, interpolate_navigation,
                geographic_to_ecef, ecef_to_utm, load_camera_calibration,
                build_ray_directions)
  - utils/plot_gref.py providing load_transect (optional, used for HSI RGB plot)

Run:
    python dev3_sim_with_mbes.py
    python dev3_sim_with_mbes.py --frames 400 --mbes-step 8 --mbes-alpha 0.65
    # optional HSI:
    python dev3_sim_with_mbes.py --hsi-folder "E:/.../output" --hsi-coords NED
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
import os

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ---- replace your import block with this ----
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make project root first on sys.path

import config  # noqa: E402

try:
    import utils.utils as utils  # <- pulls functions like load_csv_navigation
except ModuleNotFoundError:
    import utils  # fallback if there is no utils/utils.py

# HSI loader lives in utils/georef.py (your second code block)
try:
    from utils.georef import load_transect as _load_transect_rgb

    _HAVE_HSI = True
except Exception as e:
    print(f"HSI loader unavailable: {e}")
    _load_transect_rgb = None
    _HAVE_HSI = False
# --------------------------------------------


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

    # Colormap (Matplotlib 3.7+)
    cmap = matplotlib.colormaps.get_cmap("tab20")
    cols = cmap(np.linspace(0, 1, len(seg_list)))
    for i, s in enumerate(seg_list):
        s["color"] = tuple(cols[i, :3])

    # Combine for slider playback
    xs, ys, zs, tl, edges, names, colors = [], [], [], [], [], [], []
    rolls, pitchs, yaws = [], [], []
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


# -------------------------- HSI RGB helpers (top-down 2D figure) --------------------------


def _ecef_of_geodetic_via_utils(lat_deg: float, lon_deg: float, h_m: float):
    x, y, z = utils.geographic_to_ecef(
        np.array([lon_deg]),
        np.array([lat_deg]),
        np.array([h_m]),
        epsg_geo=config.EPSG_GEOGRAPHIC,
        epsg_ecef=config.EPSG_ECEF,
    )
    return float(x[0]), float(y[0]), float(z[0])


def _ecef_to_ned_arrays(X, Y, Z, lat0, lon0, h0):
    """Vectorized ECEF -> NED relative to (lat0, lon0, h0)."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    x0, y0, z0 = _ecef_of_geodetic_via_utils(lat0, lon0, h0)
    dx, dy, dz = X - x0, Y - y0, Z - z0
    lat = np.radians(lat0)
    lon = np.radians(lon0)
    sL, cL = np.sin(lat), np.cos(lat)
    sO, cO = np.sin(lon), np.cos(lon)
    n = (-sL * cO) * dx + (-sL * sO) * dy + (cL) * dz
    e = (-sO) * dx + (cO) * dy
    d = (-cL * cO) * dx + (-cL * sO) * dy + (-sL) * dz
    return n, e, d


def _prepare_hsi_topdown_ctx(
    hsi_folder, hsi_names, coord_sys, origin_lat, origin_lon, origin_h
):
    """
    Return dict with Xc, Yc, RGB and labels for a fast pcolormesh figure.
    None if unavailable.
    """
    if not _HAVE_HSI or not hsi_folder or not os.path.isdir(hsi_folder):
        return None

    try:
        tds = _load_transect_rgb(hsi_folder, use_corrected=False)
        if hsi_names:
            cube = tds.select_files(hsi_names)
        else:
            cube = tds.select_all_files()

        X_ecef, Y_ecef, Z_ecef, R, G, B = cube._rgb_from_wavelengths(
            654.2, 560.0, 440.3, normalize=True
        )
        T, S = X_ecef.shape

        # Build alpha and RGB, set NaN pixels to NaN for transparency
        RGB = np.dstack([R, G, B]).astype(np.float64)  # (T,S,3)
        alpha = np.ones((T, S), dtype=np.float64)
        alpha[~(np.isfinite(R) & np.isfinite(G) & np.isfinite(B))] = 0.0

        if coord_sys.upper() == "NED":
            N, E, D = _ecef_to_ned_arrays(
                X_ecef, Y_ecef, Z_ecef, origin_lat, origin_lon, origin_h
            )
            Xp, Yp = E, N
            xlabel = f"East (m) from {origin_lat}°, {origin_lon}°"
            ylabel = "North (m)"
            title = f"RGB georeferenced composite [NED]\norigin @ {origin_lat:.6f}°, {origin_lon:.6f}°"
        else:
            x0, y0, z0 = _ecef_of_geodetic_via_utils(origin_lat, origin_lon, origin_h)
            Xp, Yp = X_ecef - x0, Y_ecef - y0
            xlabel = f"ΔECEF X (m) from {origin_lat}°, {origin_lon}°"
            ylabel = "ΔECEF Y (m)"
            title = f"RGB georeferenced composite [ECEF Δ]\norigin @ {origin_lat:.6f}°, {origin_lon:.6f}°"

        alpha[~(np.isfinite(Xp) & np.isfinite(Yp))] = 0.0
        RGB[alpha == 0.0] = np.nan

        # pcolormesh cell corners
        Xc = np.pad(Xp, ((0, 1), (0, 1)), mode="edge")
        Yc = np.pad(Yp, ((0, 1), (0, 1)), mode="edge")

        return {
            "Xc": Xc,
            "Yc": Yc,
            "RGB": RGB,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "title": title,
            "T": T,
            "S": S,
        }
    except Exception as e:
        print(f"⚠️  HSI RGB preparation failed: {e}")
        return None


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
    hsi_ctx=None,  # <-- optional HSI top-down context for toggle figure
):
    """
    3D plot:
      - MBES surface (optional)
      - Full mission in gray
      - Static colored segments
      - IMU + Camera markers
      - Body axes
      - Camera rays (line segments)
      - Slider, Play/Pause, Rays on/off, MBES on/off, HSI RGB on/off (new)
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
    rays_cam_full = utils.build_ray_directions(
        camera_calib, n_slits
    )  # (W,3), not normalized
    if n_slits <= MAX_RAYS_DRAWN:
        sel = np.arange(n_slits)
    else:
        sel = np.linspace(0, n_slits - 1, MAX_RAYS_DRAWN, dtype=int)
    rays_cam = rays_cam_full[sel, :]

    # Figure/axes
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

    # Rays as a Line3DCollection
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

    # Cube-like limits around full mission
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
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Height (m)")
    ax.legend(loc="upper right", fontsize=8)

    # HSI toggle figure state
    hsi_state = [False]
    hsi_fig = [None]
    hsi_ax = [None]

    def _open_or_close_hsi_fig():
        if hsi_ctx is None:
            print(
                "⚠️  HSI RGB not available. Provide --hsi-folder or ensure utils/plot_gref.py is importable."
            )
            return
        if hsi_state[0]:
            # close
            if hsi_fig[0] is not None:
                try:
                    plt.close(hsi_fig[0])
                except Exception:
                    pass
                hsi_fig[0] = None
                hsi_ax[0] = None
            hsi_state[0] = False
        else:
            # open
            hsi_fig[0], hsi_ax[0] = plt.subplots(figsize=(10, 8))
            hsi_ax[0].pcolormesh(
                hsi_ctx["Xc"], hsi_ctx["Yc"], hsi_ctx["RGB"], shading="flat"
            )
            hsi_ax[0].set_aspect("equal", adjustable="box")
            hsi_ax[0].set_xlabel(hsi_ctx["xlabel"])
            hsi_ax[0].set_ylabel(hsi_ctx["ylabel"])
            hsi_ax[0].set_title(hsi_ctx["title"])
            hsi_fig[0].tight_layout()
            hsi_state[0] = True
            hsi_fig[0].canvas.draw_idle()

    def _update_frame(i):
        i = int(i)
        # IMU position
        px, py, pz = xi[i], yi[i], zi[i]

        # BODY->ENU rotation
        R_bw = _body_to_enu_rotation(
            _to_radians(roll_arr[i]), _to_radians(pitch_arr[i]), yaw_arr[i]
        )

        # Camera position: IMU + R_bw * [2.5, 0, 0]_body
        cam_off_body = np.array([CAMERA_FWD_OFFSET_M, 0.0, 0.0])
        cam_off_enu = R_bw @ cam_off_body
        cx, cy, cz = px + cam_off_enu[0], py + cam_off_enu[1], pz + cam_off_enu[2]

        # Body axes from IMU
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

        # Color IMU by current segment
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
                [
                    np.full(len(d_world), cx),
                    np.full(len(d_world), cy),
                    np.full(len(d_world), cz),
                ]
            )
            ends = starts + d_world * RAY_LENGTH_M
            segments = np.stack([starts, ends], axis=1)

            # sanity: rays must point down (dot with +Z < 0)
            up = np.array([0.0, 0.0, 1.0])
            dots = d_world @ up
            bad = dots > 0  # upward
            colors = np.repeat([RAYS_COLOR], len(d_world), axis=0).astype(float)
            if np.any(bad):
                colors[bad] = (1.0, 0.0, 0.0)  # red for wrong sign

            coll = Line3DCollection(segments, colors=colors, linewidths=0.8, zorder=10)
            rays_coll[0] = coll
            ax.add_collection3d(coll)

        mbes_line = (
            "MBES: ON" if mbes_state[0] and (mbes_surface is not None) else "MBES: OFF"
        )
        rays_line = "Rays: ON" if rays_state[0] else "Rays: OFF"
        info.set_text(
            f"Idx {i}/{n_total-1} | {seg_names[seg_idx]}\n"
            f"Lookup UTC: {_fmt_utc(float(t_lookup[i]))}\n"
            f"RPY ({RPY_UNITS}): roll={roll_arr[i]:.3f}, pitch={pitch_arr[i]:.3f}, yaw={yaw_arr[i]:.3f}\n"
            f"{rays_line}   |   {mbes_line}"
        )
        return imu_pt, cam_pt, x_axis_line, y_axis_line, z_axis_line, info

    # Controls
    ax_slider = fig.add_axes([0.15, 0.06, 0.45, 0.03])
    slider = Slider(ax_slider, "Index", 0, n_total - 1, valinit=0, valstep=1)

    # Shifted to make space for HSI button
    ax_btn_play = fig.add_axes([0.52, 0.06, 0.1, 0.04])
    btn_play = Button(ax_btn_play, "Play/Pause")
    ax_btn_rays = fig.add_axes([0.64, 0.06, 0.1, 0.04])
    btn_rays = Button(ax_btn_rays, "Rays On/Off")
    ax_btn_mbes = fig.add_axes([0.76, 0.06, 0.1, 0.04])
    btn_mbes = Button(ax_btn_mbes, "MBES On/Off")
    ax_btn_hsi = fig.add_axes([0.88, 0.06, 0.1, 0.04])
    btn_hsi = Button(
        ax_btn_hsi, "HSI RGB On/Off" if hsi_ctx is not None else "HSI RGB N/A"
    )

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
        if hsi_state[0] and hsi_fig[0] is not None:
            hsi_fig[0].canvas.draw_idle()

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

    def on_toggle_mbes(event):
        mbes_state[0] = not mbes_state[0]
        # (re-)draw/remove MBES surface
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

    def on_toggle_hsi(event):
        _open_or_close_hsi_fig()

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)
    btn_rays.on_clicked(on_toggle_rays)
    btn_mbes.on_clicked(on_toggle_mbes)
    btn_hsi.on_clicked(on_toggle_hsi)

    # Initialize frame 0
    _update_frame(0)
    plt.show()
    return fig, ax


# ------------------------------ Main ------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="3D UHI simulation with MBES & rays + HSI RGB toggle"
    )
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
    # HSI options (all optional)
    parser.add_argument(
        "--hsi-folder",
        type=str,
        default=getattr(config, "OUTPUT_FOLDER", None),
        help="Folder with georeferenced H5 outputs to build an RGB footprint (optional).",
    )
    parser.add_argument(
        "--hsi-files",
        type=str,
        default=None,
        help="Comma separated list of H5 base names to include for HSI RGB. If omitted, uses all in folder.",
    )
    parser.add_argument(
        "--hsi-coords",
        type=str,
        default="NED",
        choices=["NED", "ECEF"],
        help="Coordinate system for the HSI RGB top-down figure.",
    )
    parser.add_argument(
        "--hsi-origin",
        type=float,
        nargs=3,
        default=[
            float(getattr(config, "LAT0", 60.8011575)),
            float(getattr(config, "LON0", 10.7122345)),
            float(getattr(config, "H0", 0.0)),
        ],
        metavar=("LAT", "LON", "H"),
        help="Origin (lat lon h) used for NED or ECEF-Δ HSI RGB figure.",
    )

    args = parser.parse_args()

    # Load segments / trajectory
    full_xyz, segments, combined, _nav_df = _load_segments(num_frames=args.frames)

    # Print per-file ranges
    print("\n=== Per-file HSI time bounds (used frames) ===")
    for s in segments:
        print(f"{s['name']}")
        print(f"  HSI UTC:    {s['hsi_start_utc']} to {s['hsi_end_utc']}")
        print(f"  Lookup UTC: {s['lookup_start_utc']} to {s['lookup_end_utc']}")
    print("")
    print(f"Orientation convention: {ORI_CONVENTION} (from config), units: {RPY_UNITS}")
    print(f"Camera offset: {CAMERA_FWD_OFFSET_M} m along body +X")
    print(
        "Camera->Body rotation (R_B_FROM_CAM) - loaded from config.ROTATION_HSI_TO_BODY:"
    )
    print(R_B_FROM_CAM)
    print("✓ Simulation now uses SAME rotation matrix as main.py!")

    # Camera intrinsics for rays
    camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
    print(
        f"Camera calib: f={camera_calib['f']:.3f}, cx={camera_calib['cx']:.3f}, w={int(camera_calib['w'])}"
    )

    # MBES surface in UTM (EPSG_MBES)
    mbes_surface = None
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

    # Prepare HSI RGB top-down context (optional)
    hsi_ctx = None
    if _HAVE_HSI and args.hsi_folder:
        names = (
            [n.strip() for n in args.hsi_files.split(",")] if args.hsi_files else None
        )
        hsi_ctx = _prepare_hsi_topdown_ctx(
            args.hsi_folder,
            names,
            args.hsi_coords,
            float(args.hsi_origin[0]),
            float(args.hsi_origin[1]),
            float(args.hsi_origin[2]),
        )
        if hsi_ctx is None:
            print("⚠️  HSI RGB context unavailable. The HSI button will show 'N/A'.")

    title_extra = f"(offset {float(config.TIME_OFFSET_SEC):+.1f}s; rays ≤ {MAX_RAYS_DRAWN}; MBES step={args.mbes_step})"
    plot_with_mbes_and_rays(
        full_xyz,
        segments,
        combined,
        camera_calib,
        mbes_surface=mbes_surface,
        mbes_alpha=float(args.mbes_alpha),
        title_extra=title_extra,
        hsi_ctx=hsi_ctx,
    )


if __name__ == "__main__":
    main()
