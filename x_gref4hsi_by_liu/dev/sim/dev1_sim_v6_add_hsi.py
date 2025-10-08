"""
3D UHI simulation with MBES, camera rays, and OVERLAID georeferenced HSI points (from main.py output).
Also opens a top-down 2D window with MBES raster + all georef points.

Inputs: config.py, utils.py; main.py must have produced georef datasets in OUTPUT_FOLDER/*.h5

Run examples:
    python dev6_sim_georef_mbes.py
    python dev6_sim_georef_mbes.py --frames 400 --mbes-step 8 --mbes-alpha 0.65 --georef-stride 2 --georef-color pixel
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from datetime import datetime, timezone

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Try GUI backends so widgets are clickable
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


# ===================== Tunables =====================
ORI_CONVENTION = getattr(config, "YAW_CONVENTION", "heading_from_north_cw")
RPY_UNITS = "deg"
CAMERA_FWD_OFFSET_M = float(getattr(config, "TRANSLATION_BODY_TO_HSI", [2.5, 0, 0])[0])
AXIS_ARROW_LEN_M = 1.5

# Camera rays
R_B_FROM_CAM = np.array(config.ROTATION_HSI_TO_BODY, dtype=float)  # same as main.py
SHOW_RAYS_BY_DEFAULT = True
RAY_LENGTH_M = 8.0
MAX_RAYS_DRAWN = 61
RAYS_COLOR = (0.6, 0.1, 0.9)  # magenta-ish

# MBES surface
SHOW_MBES_BY_DEFAULT = True
MBES_CMAP = "Greys"

# Georef points (from OUTPUT_FOLDER H5)
SHOW_GEOREF_BY_DEFAULT = True
GEOREF_CMAP = "turbo"  # used for height/pixel/frame coloring
GEOREF_POINT_SIZE = 4
# ====================================================


# ----------------- Helpers -----------------
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
    """R_ENU_from_BODY. Order: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
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


def _segment_index_from_global(i, seg_edges):
    return int(np.searchsorted(seg_edges, i, side="right"))


# ----------------- Load segments & nav -----------------
def _load_segments(num_frames=None):
    """
    Returns:
      full_xyz
      segments: list of dict per H5 (has in_path/out_path, frame_ids, colors, etc.)
      combined: concatenated arrays + global frame_ids
      nav_df
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

        if num_frames is not None and len(ts) > num_frames:
            idx_used = np.linspace(0, len(ts) - 1, num_frames, dtype=int)
        else:
            idx_used = np.arange(len(ts))
        ts_used = ts[idx_used]

        interp_nav = utils.interpolate_navigation(nav_df, ts_used, time_offset=offset)
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

        in_path = str(p)
        out_path = str(Path(config.OUTPUT_FOLDER) / p.name)

        seg_list.append(
            {
                "name": p.name,
                "in_path": in_path,
                "out_path": out_path,
                "frame_ids": idx_used,  # mapping to original H5 frame index
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

    seg_list.sort(key=lambda s: s["lookup_ts_used"].min())

    cmap = matplotlib.colormaps.get_cmap("tab20")
    cols = cmap(np.linspace(0, 1, len(seg_list)))
    for i, s in enumerate(seg_list):
        s["color"] = tuple(cols[i, :3])

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
        "frame_ids": np.concatenate(frame_ids_all),
        "seg_edges": np.array(edges, dtype=int),
        "seg_names": names,
        "seg_colors": colors,
    }
    return (x_all, y_all, z_all), seg_list, combined, nav_df


# ----------------- MBES utils -----------------
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
    x1 = x0 + transform.a * w
    y1 = y0 + transform.e * h
    xmin, xmax = (min(x0, x1), max(x0, x1))
    ymin, ymax = (min(y0, y1), max(y0, y1))
    return [xmin, xmax, ymin, ymax]


# ----------------- Georef loader -----------------
def _load_georef_grouped_utm(h5_path, epsg_utm, epsg_ecef):
    CAND = [
        "processed/georef/points_ecef_crs",
        "processed/georef/points_ecef",
        "georef/points_ecef_crs",
        "georef/points_ecef",
    ]
    DS_FR = "processed/georef/frame_indices"
    DS_PX = "processed/georef/ray_indices"

    with h5py.File(h5_path, "r") as f:
        pts = None
        used = None
        for p in CAND:
            if p in f:
                a = f[p][()]
                if a.ndim == 2 and a.shape[1] == 3:
                    pts = a
                    used = p
                    break
        if pts is None or DS_FR not in f or DS_PX not in f:
            return None
        frames = f[DS_FR][()].astype(int)
        pixels = f[DS_PX][()].astype(int)

    if len(frames) != len(pts) or len(pixels) != len(pts):
        return None

    ex, ey, ez = utils.ecef_to_utm(
        pts[:, 0], pts[:, 1], pts[:, 2], epsg_utm=epsg_utm, epsg_ecef=epsg_ecef
    )
    order = np.lexsort((pixels, frames))
    fs = frames[order]
    pxs = pixels[order]
    exs = np.asarray(ex)[order]
    eys = np.asarray(ey)[order]
    ezs = np.asarray(ez)[order]
    uniq, first, counts = np.unique(fs, return_index=True, return_counts=True)

    print(f"  Using georef: {h5_path}")
    print(f"    dataset: {used} | hits: {len(exs)} | frames with hits: {len(uniq)}")
    return {
        "frames": uniq,
        "starts": first,
        "counts": counts,
        "ex": exs,
        "ey": eys,
        "ez": ezs,
        "px": pxs,
    }


def _build_georef_for_segments(seg_list):
    out = []
    print("\n=== Georef search ===")
    for s in seg_list:
        out_path = Path(s["out_path"])
        in_path = Path(s["in_path"])
        if out_path.exists():
            print(f"{s['name']}: try OUTPUT -> {out_path}")
            g = _load_georef_grouped_utm(
                str(out_path), config.EPSG_MBES, config.EPSG_ECEF
            )
            if g is not None:
                out.append(g)
                continue
            print("  ⚠ no georef in output; fallback to INPUT")
        else:
            print(f"{s['name']}: output not found -> {out_path}")
        print(f"{s['name']}: try INPUT  -> {in_path}")
        g = _load_georef_grouped_utm(str(in_path), config.EPSG_MBES, config.EPSG_ECEF)
        if g is None:
            print("  ❌ no georef datasets found")
        out.append(g)
    print("=== End search ===\n")
    return out


# ----------------- Color mapping -----------------
def _map_colors(vals, cmap_name=GEOREF_CMAP, vmin=None, vmax=None, alpha=1.0):
    vals = np.asarray(vals)
    if vmin is None:
        vmin = float(np.nanmin(vals))
    if vmax is None:
        vmax = float(np.nanmax(vals) if np.nanmax(vals) > vmin else vmin + 1.0)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(norm(vals))
    rgba[..., 3] = alpha
    return rgba


# ----------------- 3D plot with MBES + Rays + Georef -----------------
def plot_with_mbes_rays_georef(
    full_xyz,
    segments,
    combined,
    camera_calib,
    georef_sets,
    mbes_surface=None,
    mbes_alpha=0.65,
    georef_stride=1,
    georef_color_mode="height",
    title_extra="",
):
    xa, ya, za = full_xyz
    xi, yi, zi = combined["x_imu"], combined["y_imu"], combined["z_imu"]
    t_lookup = combined["lookup_ts"]
    roll_arr, pitch_arr, yaw_arr = combined["roll"], combined["pitch"], combined["yaw"]
    seg_edges = combined["seg_edges"]
    seg_names = combined["seg_names"]
    seg_colors = combined["seg_colors"]
    frame_ids_global = combined["frame_ids"]
    n_total = len(xi)

    global_start = _fmt_utc(float(t_lookup.min()))
    global_end = _fmt_utc(float(t_lookup.max()))

    # Rays in camera frame (test_eely convention) – not normalized
    n_slits = int(camera_calib["w"])
    rays_cam_full = utils.build_ray_directions(camera_calib, n_slits)
    sel = (
        np.arange(n_slits)
        if n_slits <= MAX_RAYS_DRAWN
        else np.linspace(0, n_slits - 1, MAX_RAYS_DRAWN, dtype=int)
    )
    rays_cam = rays_cam_full[sel, :]

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d", position=[0.05, 0.18, 0.9, 0.77])

    # MBES
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

    # Mission & segments
    ax.plot(xa, ya, za, linewidth=1.0, color="gray", alpha=0.45, label="Full mission")
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

    # Markers & body axes
    (imu_pt,) = ax.plot(
        [], [], [], "o", markersize=9, color="black", zorder=12, label="IMU"
    )
    (cam_pt,) = ax.plot(
        [], [], [], "o", markersize=8, color="magenta", zorder=12, label="Camera"
    )
    (x_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="red", zorder=11, label="Body X"
    )
    (y_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="green", zorder=11, label="Body Y"
    )
    (z_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="blue", zorder=11, label="Body Z"
    )

    # Rays collection
    rays_coll = [None]
    rays_state = [SHOW_RAYS_BY_DEFAULT]

    # Georef scatter (per-frame, togglable)
    georef_scatter = [None]
    georef_state = [SHOW_GEOREF_BY_DEFAULT]

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

    # Cube-ish limits
    rng = np.array([xa.max() - xa.min(), ya.max() - ya.min(), za.max() - za.min()])
    half = rng.max() / 2.0
    cxm = (xa.max() + xa.min()) / 2.0
    cym = (ya.max() + ya.min()) / 2.0
    czm = (za.max() + za.min()) / 2.0
    ax.set_xlim(cxm - half, cxm + half)
    ax.set_ylim(cym - half, cym + half)
    ax.set_zlim(czm - half, czm + half)

    ttl = f"Mission + MBES + UHI rays + GEOREF  {title_extra}\nGlobal lookup: {global_start} – {global_end}"
    ax.set_title(ttl, pad=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Height (m)")
    ax.legend(loc="upper right", fontsize=8)

    def _update_frame(i):
        i = int(i)
        # IMU pose
        px, py, pz = xi[i], yi[i], zi[i]
        R_bw = _body_to_enu_rotation(
            _to_radians(roll_arr[i]), _to_radians(pitch_arr[i]), yaw_arr[i]
        )

        # Camera position = IMU + R_bw * [2.5,0,0]_body
        cam_off_body = np.array([CAMERA_FWD_OFFSET_M, 0.0, 0.0])
        cam_off_enu = R_bw @ cam_off_body
        cx, cy, cz = px + cam_off_enu[0], py + cam_off_enu[1], pz + cam_off_enu[2]

        # Update markers
        imu_pt.set_data([px], [py])
        imu_pt.set_3d_properties([pz])
        cam_pt.set_data([cx], [cy])
        cam_pt.set_3d_properties([cz])

        # Axes at IMU
        L = AXIS_ARROW_LEN_M
        x_tip = np.array([px, py, pz]) + (R_bw @ np.array([L, 0, 0]))
        y_tip = np.array([px, py, pz]) + (R_bw @ np.array([0, L, 0]))
        z_tip = np.array([px, py, pz]) + (R_bw @ np.array([0, 0, L]))
        x_axis_line.set_data([px, x_tip[0]], [py, x_tip[1]])
        x_axis_line.set_3d_properties([pz, x_tip[2]])
        y_axis_line.set_data([px, y_tip[0]], [py, y_tip[1]])
        y_axis_line.set_3d_properties([pz, y_tip[2]])
        z_axis_line.set_data([px, z_tip[0]], [py, z_tip[1]])
        z_axis_line.set_3d_properties([pz, z_tip[2]])

        # Color IMU by segment
        seg_idx = _segment_index_from_global(i, seg_edges)
        imu_pt.set_color(seg_colors[seg_idx])

        # Rays from CAMERA
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

            # sanity: downwards (dot with up < 0)
            up = np.array([0.0, 0.0, 1.0])
            dots = d_world @ up
            bad = dots > 0
            colors = np.repeat([RAYS_COLOR], len(d_world), axis=0).astype(float)
            if np.any(bad):
                colors[bad] = (1.0, 0.0, 0.0)

            coll = Line3DCollection(segments, colors=colors, linewidths=0.8, zorder=10)
            rays_coll[0] = coll
            ax.add_collection3d(coll)

        # Georef points for THIS frame (from segment’s georef set)
        if georef_scatter[0] is not None:
            try:
                georef_scatter[0].remove()
            except Exception:
                pass
            georef_scatter[0] = None

        if georef_state[0] and georef_sets[seg_idx] is not None:
            g = georef_sets[seg_idx]
            # original H5 frame id for this global index:
            frame_id_local = frame_ids_global[i]
            # locate in georef unique frames
            idx = np.searchsorted(g["frames"], frame_id_local)
            if 0 <= idx < len(g["frames"]) and g["frames"][idx] == frame_id_local:
                s = int(g["starts"][idx])
                c = int(g["counts"][idx])
                if c > 0:
                    ex = g["ex"][s : s + c : georef_stride]
                    ey = g["ey"][s : s + c : georef_stride]
                    ez = g["ez"][s : s + c : georef_stride]
                    px = g["px"][s : s + c : georef_stride]

                    if georef_color_mode == "height":
                        col = _map_colors(ez, GEOREF_CMAP, alpha=0.9)
                    elif georef_color_mode == "pixel":
                        col = _map_colors(
                            px, GEOREF_CMAP, vmin=np.min(px), vmax=np.max(px), alpha=0.9
                        )
                    elif georef_color_mode == "frame":
                        # constant color for the frame (or map tiny range)
                        col = _map_colors(
                            np.full_like(ex, frame_id_local, dtype=float),
                            GEOREF_CMAP,
                            vmin=frame_id_local - 0.5,
                            vmax=frame_id_local + 0.5,
                            alpha=0.9,
                        )
                    else:
                        col = _map_colors(ez, GEOREF_CMAP, alpha=0.9)

                    georef_scatter[0] = ax.scatter(
                        ex,
                        ey,
                        ez,
                        s=GEOREF_POINT_SIZE,
                        c=col,
                        depthshade=False,
                        zorder=9,
                    )

        info.set_text(
            f"Idx {i}/{n_total-1} | {seg_names[seg_idx]}\n"
            f"Lookup UTC: {_fmt_utc(float(t_lookup[i]))}\n"
            f"RPY ({RPY_UNITS}): roll={roll_arr[i]:.3f}, pitch={pitch_arr[i]:.3f}, yaw={yaw_arr[i]:.3f}\n"
            f"Rays: {'ON' if rays_state[0] else 'OFF'}   |   MBES: {'ON' if mbes_state[0] else 'OFF'}   |   Georef: {'ON' if georef_state[0] else 'OFF'}"
        )
        return imu_pt, cam_pt, x_axis_line, y_axis_line, z_axis_line, info

    # Controls
    ax_slider = fig.add_axes([0.15, 0.06, 0.37, 0.03])
    slider = Slider(ax_slider, "Index", 0, n_total - 1, valinit=0, valstep=1)
    ax_btn_play = fig.add_axes([0.53, 0.06, 0.1, 0.04])
    btn_play = Button(ax_btn_play, "Play/Pause")
    ax_btn_rays = fig.add_axes([0.65, 0.06, 0.1, 0.04])
    btn_rays = Button(ax_btn_rays, "Rays On/Off")
    ax_btn_mbes = fig.add_axes([0.77, 0.06, 0.1, 0.04])
    btn_mbes = Button(ax_btn_mbes, "MBES On/Off")
    ax_btn_geo = fig.add_axes([0.89, 0.06, 0.1, 0.04])
    btn_geo = Button(ax_btn_geo, "Georef On/Off")

    anim_running = [True]
    anim = FuncAnimation(
        fig, _update_frame, frames=n_total, interval=60, blit=False, repeat=True
    )
    fig._anim = anim  # keep reference

    def on_slider_change(val):
        anim.event_source.stop()
        anim_running[0] = False
        _update_frame(int(slider.val))
        fig.canvas.draw_idle()

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

    def on_toggle_georef(event):
        georef_state[0] = not georef_state[0]
        _update_frame(int(slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)
    btn_rays.on_clicked(on_toggle_rays)
    btn_mbes.on_clicked(on_toggle_mbes)
    btn_geo.on_clicked(on_toggle_georef)

    _update_frame(0)
    plt.show()
    return fig, ax


# ----------------- 2D Top-down view -----------------
def plot_topdown_georef_all(
    mbes_arr, mbes_tf, georef_sets, color_mode="height", georef_stride=2
):
    if mbes_arr is None or mbes_tf is None:
        print("⚠️  Top-down: MBES not available; plotting georef only.")
    plt.figure(figsize=(11, 9))
    if mbes_arr is not None:
        extent = _extent_from_transform(mbes_arr, mbes_tf)
        vmin = np.nanpercentile(mbes_arr, 2)
        vmax = np.nanpercentile(mbes_arr, 98)
        plt.imshow(
            mbes_arr,
            extent=extent,
            origin="upper",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,
        )

    # concat all georef hits (decimated)
    xs, ys, zs, px, fr = [], [], [], [], []
    for g in georef_sets:
        if g is None:
            continue
        ex, ey, ez, p = (
            g["ex"][::georef_stride],
            g["ey"][::georef_stride],
            g["ez"][::georef_stride],
            g["px"][::georef_stride],
        )
        frames_rep = np.repeat(g["frames"], g["counts"])[::georef_stride]
        xs.append(ex)
        ys.append(ey)
        zs.append(ez)
        px.append(p)
        fr.append(frames_rep)
    if xs:
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        Z = np.concatenate(zs)
        P = np.concatenate(px)
        F = np.concatenate(fr)
        if color_mode == "height":
            C = _map_colors(Z, GEOREF_CMAP, alpha=0.9)
            cb_label = "Height (m)"
        elif color_mode == "pixel":
            C = _map_colors(
                P, GEOREF_CMAP, vmin=float(np.min(P)), vmax=float(np.max(P)), alpha=0.9
            )
            cb_label = "Pixel index"
        else:  # frame
            C = _map_colors(
                F.astype(float),
                GEOREF_CMAP,
                vmin=float(np.min(F)),
                vmax=float(np.max(F)),
                alpha=0.9,
            )
            cb_label = "Frame id"
        sc = plt.scatter(X, Y, s=2, c=C)
        plt.title(f"Top-down georef points (color by {color_mode})")
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.gca().set_aspect("equal", adjustable="box")
        # Add a colorbar proxy for readability
        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colormaps.get_cmap(GEOREF_CMAP),
            norm=Normalize(
                vmin=(
                    np.min(Z)
                    if color_mode == "height"
                    else (np.min(P) if color_mode == "pixel" else np.min(F))
                ),
                vmax=(
                    np.max(Z)
                    if color_mode == "height"
                    else (np.max(P) if color_mode == "pixel" else np.max(F))
                ),
            ),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.8)
        cbar.set_label(cb_label)
    else:
        plt.title("Top-down georef points: none found")
    plt.show()


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(
        description="3D UHI simulation + MBES + georef overlay + top-down view"
    )
    ap.add_argument(
        "--frames", type=int, default=None, help="Optional per-H5 subsample count"
    )
    ap.add_argument(
        "--mbes-step",
        type=int,
        default=8,
        help="MBES surface decimation step (bigger=faster)",
    )
    ap.add_argument("--mbes-alpha", type=float, default=0.65, help="MBES surface alpha")
    ap.add_argument(
        "--georef-stride",
        type=int,
        default=1,
        help="Decimation for per-frame georef scatter",
    )
    ap.add_argument(
        "--georef-color",
        choices=["height", "pixel", "frame"],
        default="height",
        help="How to color georef points",
    )
    args = ap.parse_args()

    full_xyz, segments, combined, _nav_df = _load_segments(num_frames=args.frames)

    print("\n=== Per-file HSI time bounds (used frames) ===")
    for s in segments:
        print(f"{s['name']}")
        print(f"  HSI UTC:    {s['hsi_start_utc']} to {s['hsi_end_utc']}")
        print(f"  Lookup UTC: {s['lookup_start_utc']} to {s['lookup_end_utc']}")
    print("")
    print(f"Yaw convention: {ORI_CONVENTION} (from config), units: {RPY_UNITS}")
    print(f"Camera offset (body +X): {CAMERA_FWD_OFFSET_M} m")
    print("Camera->Body rotation (from config.ROTATION_HSI_TO_BODY):")
    print(R_B_FROM_CAM)

    camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
    print(
        f"Camera calib: f={camera_calib['f']:.3f}, cx={camera_calib['cx']:.3f}, w={int(camera_calib['w'])}"
    )

    # MBES
    mbes_surface = None
    mbes_arr = None
    mbes_tf = None
    try:
        mbes_arr, mbes_tf, used_epsg = _read_mbes_geotiff_as_target_epsg(
            config.MBES_GEOTIFF, config.EPSG_MBES
        )
        Xm, Ym, Zm = _surface_from_raster(
            mbes_arr, mbes_tf, step=max(1, int(args.mbes_step))
        )
        mbes_surface = (Xm, Ym, Zm)
        print(
            f"MBES GeoTIFF: {config.MBES_GEOTIFF} (EPSG:{used_epsg}), surface size {Xm.shape} after decimation"
        )
    except Exception as e:
        print(f"⚠️  MBES load failed: {e}")

    # Georef from OUTPUT_FOLDER (preferred), else input H5
    georef_sets = _build_georef_for_segments(segments)

    title_extra = f"(offset {float(config.TIME_OFFSET_SEC):+.1f}s; rays≤{MAX_RAYS_DRAWN}; georef stride={args.georef_stride})"
    plot_with_mbes_rays_georef(
        full_xyz,
        segments,
        combined,
        camera_calib,
        georef_sets,
        mbes_surface=mbes_surface,
        mbes_alpha=float(args.mbes_alpha),
        georef_stride=max(1, int(args.georef_stride)),
        georef_color_mode=args.georef_color,
        title_extra=title_extra,
    )

    # Top-down 2D window
    plot_topdown_georef_all(
        mbes_arr,
        mbes_tf,
        georef_sets,
        color_mode=args.georef_color,
        georef_stride=max(1, int(args.georef_stride)),
    )


if __name__ == "__main__":
    main()
