"""
Multi-H5 UHI segments with slider, showing IMU & Camera frames, body axes, and CAMERA RAYS.

Repo expectations:
- config: NAV_CSV, CSV_COLUMNS, H5_FOLDER, TIME_OFFSET_SEC,
          EPSG_GEOGRAPHIC, EPSG_ECEF, EPSG_MBES, CAMERA_CALIB_XML
- utils:  load_csv_navigation, load_h5_timestamps, interpolate_navigation,
          geographic_to_ecef, ecef_to_utm, load_camera_calibration, build_ray_directions

Run:
    python dev1_simulation.py
    python dev1_simulation.py --frames 400
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from datetime import datetime, timezone

# Optional GUI backends so widgets are clickable when run as a script
for _backend in ("Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        pass

# Make local config and utils importable
sys.path.append(str(Path(__file__).parent.parent))
import config  # noqa: E402
import x_gref4hsi_by_liu.utils.utils as utils  # noqa: E402

# For loading MBES mesh
try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("WARNING: PyVista not available. MBES mesh will not be loaded.")

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("WARNING: Rasterio not available. MBES GeoTIFF loading may fail.")


# ========= User-tunable orientation & ray-draw settings =========
# Yaw convention:
#   "heading_from_north_cw" means yaw is a compass heading (deg), 0=N, 90=E, clockwise positive.
#   "enu_yaw_from_east_ccw" means yaw is yaw about +Z ENU from +X (East), CCW (math/ROS).
ORI_CONVENTION = "heading_from_north_cw"  # or "enu_yaw_from_east_ccw"
RPY_UNITS = "deg"  # "deg" or "rad"

# Camera mounting relative to BODY:
CAMERA_FWD_OFFSET_M = 2.5  # camera is 2.5 m ahead of IMU along BODY +X (your spec)
# Camera frame axes (as in utils.build_ray_directions "test_eely" convention):
#   cam X = across-track, cam Y = along-track, cam Z = down
# Body frame axes:
#   body X = forward, body Y = starboard, body Z = up
# Mapping cam->body: X_c->+Y_b, Y_c->+X_b, Z_c->-Z_b
R_B_FROM_CAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

# Body-axis arrow visuals
AXIS_ARROW_LEN_M = 1.5

# Ray visuals
SHOW_RAYS_BY_DEFAULT = True
RAY_LENGTH_M = 8.0  # how long each drawn ray appears
MAX_RAYS_DRAWN = 41  # draw at most this many per frame (subsample evenly across line)
RAYS_COLOR = (0.6, 0.1, 0.9)  # magenta-ish; matplotlib RGB tuple

# MBES mesh visuals
SHOW_MBES_BY_DEFAULT = True
MBES_POINT_SIZE = 0.5  # point size for scatter plot
MBES_ALPHA = 0.3  # transparency (0=invisible, 1=opaque)
MBES_COLOR = (0.6, 0.4, 0.2)  # brown-ish color for seafloor
MBES_SUBSAMPLE = 50  # subsample factor (use every Nth point to reduce clutter)

# ================================================================


def _fmt_utc(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _load_mbes_mesh():
    """
    Load MBES mesh from GeoTIFF and return subsampled points in UTM coordinates.

    Returns:
        tuple: (x_mbes, y_mbes, z_mbes) as numpy arrays, or (None, None, None) if loading fails
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available - skipping MBES mesh loading")
        return None, None, None

    try:
        # Try to load mesh from GeoTIFF
        from pathlib import Path

        geotiff_path = Path(config.MBES_GEOTIFF)

        if not geotiff_path.exists():
            print(f"MBES GeoTIFF not found: {geotiff_path}")
            return None, None, None

        print(f"\nLoading MBES mesh from: {geotiff_path.name}")

        # Try loading as mesh (if already converted)
        ply_path = geotiff_path.parent / (geotiff_path.stem + "_mesh.ply")
        if ply_path.exists():
            print(f"Loading pre-converted mesh: {ply_path.name}")
            mesh = pv.read(str(ply_path))
        else:
            # Load and convert GeoTIFF to mesh
            if not RASTERIO_AVAILABLE:
                print("Rasterio not available - cannot load GeoTIFF")
                return None, None, None

            with rasterio.open(str(geotiff_path)) as src:
                data = src.read(1)
                transform = src.transform

                # Create mesh grid
                rows, cols = data.shape
                x = np.zeros((rows, cols))
                y = np.zeros((rows, cols))

                for row in range(rows):
                    for col in range(cols):
                        x[row, col], y[row, col] = rasterio.transform.xy(
                            transform, row, col
                        )

                # Flatten arrays
                x_flat = x.flatten()
                y_flat = y.flatten()
                z_flat = data.flatten()

                # Remove invalid values
                valid = ~np.isnan(z_flat) & ~np.isinf(z_flat)
                x_flat = x_flat[valid]
                y_flat = y_flat[valid]
                z_flat = z_flat[valid]

                # Subsample for visualization
                if len(x_flat) > 0:
                    subsample_idx = np.arange(0, len(x_flat), MBES_SUBSAMPLE)
                    x_mbes = x_flat[subsample_idx]
                    y_mbes = y_flat[subsample_idx]
                    z_mbes = z_flat[subsample_idx]

                    print(
                        f"Loaded {len(x_mbes):,} MBES points (subsampled by {MBES_SUBSAMPLE})"
                    )
                    print(
                        f"MBES depth range: {z_mbes.min():.1f} to {z_mbes.max():.1f} m"
                    )

                    return x_mbes, y_mbes, z_mbes
                else:
                    print("No valid MBES data points found")
                    return None, None, None

    except Exception as e:
        print(f"Error loading MBES mesh: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


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


def _extract_rpy(interp_nav):
    """
    Return roll, pitch, yaw arrays as numpy, raising if missing.
    Tries common column names.
    """
    if isinstance(interp_nav, dict):
        cols = interp_nav.keys()
        get = lambda k: np.asarray(interp_nav[k])
    else:
        cols = interp_nav.columns
        get = lambda k: interp_nav[k].to_numpy()

    candidates = [
        ("roll", "pitch", "yaw"),
        ("roll_deg", "pitch_deg", "yaw_deg"),
        ("phi", "theta", "psi"),
    ]
    for r, p, y in candidates:
        if r in cols and p in cols and y in cols:
            return get(r), get(p), get(y)

    raise KeyError("Could not find roll/pitch/yaw in interpolated nav.")


def _to_radians(arr):
    return np.deg2rad(arr) if RPY_UNITS == "deg" else arr


def _body_to_enu_rotation(roll_rad, pitch_rad, yaw_in):
    """
    Build R_ENU_from_BODY for a single sample given:
      - roll_rad, pitch_rad in radians
      - yaw_in depending on ORI_CONVENTION (see header)
    Rotation order: Rz(yaw_enu) @ Ry(pitch) @ Rx(roll) mapping body->ENU.
    """
    # Normalize yaw to ENU yaw (from East CCW, radians)
    yaw_val = yaw_in
    if ORI_CONVENTION == "heading_from_north_cw":
        if RPY_UNITS == "deg":
            yaw_val = np.deg2rad(90.0 - yaw_in)  # heading (N=0 cw) -> ENU yaw (E=0 ccw)
        else:
            yaw_val = (np.pi / 2.0) - yaw_in
    else:
        yaw_val = _to_radians(yaw_in)

    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_val), np.sin(yaw_val)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx  # body->ENU


def _load_segments(num_frames=None):
    """
    Returns:
      full_xyz: (x_all, y_all, z_all)
      segments: list of dicts (per H5), each with keys:
        name, color, x_imu, y_imu, z_imu, roll, pitch, yaw, lookup_ts_used,
        hsi_start_utc, hsi_end_utc, lookup_start_utc, lookup_end_utc
      combined: dict with concatenated arrays and segment indexing
      nav_df: full navigation dataframe (for context if needed)
    """
    # Full mission from CSV
    nav_df = utils.load_csv_navigation(config.NAV_CSV, config.CSV_COLUMNS)
    x_all, y_all, z_all = _full_mission_xyz(nav_df)

    # H5 files
    h5_folder = Path(config.H5_FOLDER)
    h5_files = sorted(h5_folder.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_folder}")

    offset = float(config.TIME_OFFSET_SEC)
    seg_list = []
    for p in h5_files:
        h5_path = str(p)
        hsi_ts_all = utils.load_h5_timestamps(h5_path)  # raw HSI times for this file
        if len(hsi_ts_all) == 0:
            continue

        # Subsample evenly per file if requested
        hsi_ts_used = hsi_ts_all
        if num_frames is not None and len(hsi_ts_all) > num_frames:
            idx = np.linspace(0, len(hsi_ts_all) - 1, num_frames, dtype=int)
            hsi_ts_used = hsi_ts_all[idx]

        # Interpolate NAV at lookup times = HSI + offset
        interp_nav = utils.interpolate_navigation(
            nav_df, hsi_ts_used, time_offset=offset
        )

        # positions (IMU)
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

        # orientations
        roll, pitch, yaw = _extract_rpy(interp_nav)

        # time bounds for used frames
        lookup_ts_used = hsi_ts_used + offset
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
                "hsi_start_utc": _fmt_utc(float(hsi_ts_used.min())),
                "hsi_end_utc": _fmt_utc(float(hsi_ts_used.max())),
                "lookup_start_utc": _fmt_utc(float(lookup_ts_used.min())),
                "lookup_end_utc": _fmt_utc(float(lookup_ts_used.max())),
            }
        )

    if not seg_list:
        raise ValueError("All H5 files had zero timestamps")

    # Sort by first lookup time for a global play order
    seg_list.sort(key=lambda s: s["lookup_ts_used"].min())

    # Colors
    cmap = plt.cm.get_cmap("tab20", len(seg_list))
    for i, s in enumerate(seg_list):
        s["color"] = (
            cmap.colors[i]
            if hasattr(cmap, "colors")
            else cmap(i / max(1, len(seg_list) - 1))
        )

    # Combine arrays for slider playback
    xs_i, ys_i, zs_i, t_lookup, edges, names, colors = [], [], [], [], [], [], []
    rolls, pitchs, yaws = [], [], []
    acc = 0
    for s in seg_list:
        n = len(s["x_imu"])
        xs_i.append(s["x_imu"])
        ys_i.append(s["y_imu"])
        zs_i.append(s["z_imu"])
        t_lookup.append(s["lookup_ts_used"])
        rolls.append(s["roll"])
        pitchs.append(s["pitch"])
        yaws.append(s["yaw"])
        acc += n
        edges.append(acc)  # exclusive end index
        names.append(s["name"])
        colors.append(s["color"])

    combined = {
        "x_imu": np.concatenate(xs_i),
        "y_imu": np.concatenate(ys_i),
        "z_imu": np.concatenate(zs_i),
        "lookup_ts": np.concatenate(t_lookup),
        "roll": np.concatenate(rolls),
        "pitch": np.concatenate(pitchs),
        "yaw": np.concatenate(yaws),
        "seg_edges": np.array(edges, dtype=int),
        "seg_names": names,
        "seg_colors": colors,
    }

    return (x_all, y_all, z_all), seg_list, combined, nav_df


def _segment_index_from_global(i, seg_edges):
    # seg_edges are exclusive ends like [n1, n1+n2, ...]
    return int(np.searchsorted(seg_edges, i, side="right"))


from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_multi_segments_with_slider_and_frames(
    full_xyz, segments, combined, camera_calib, mbes_xyz=None, title_extra=""
):
    """
    3D plot:
      - Full mission in gray
      - Each H5 segment in its color (static lines)
      - MBES seafloor points (optional, brown scatter)
      - Slider + Play/Pause to scrub globally
      - Moving IMU point (colored by segment) and Camera point (magenta)
      - Body-axis arrows at the current pose (X forward, Y starboard, Z up)
      - CAMERA RAYS drawn as clean line segments (no arrowheads):
          r_world = (R_body_to_world @ R_B_FROM_CAM) @ r_cam
      - Sanity check: rays must point down (d_world · [0,0,1] < 0); wrong-sign rays shown red.
    """
    xa, ya, za = full_xyz
    xi, yi, zi = combined["x_imu"], combined["y_imu"], combined["z_imu"]
    t_lookup = combined["lookup_ts"]
    roll_arr, pitch_arr, yaw_arr = combined["roll"], combined["pitch"], combined["yaw"]
    seg_edges = combined["seg_edges"]
    seg_names = combined["seg_names"]
    seg_colors = combined["seg_colors"]

    # Build one line-scan worth of camera-frame rays
    n_slits = int(camera_calib["w"])
    rays_cam_full = utils.build_ray_directions(camera_calib, n_slits)  # shape (W,3)
    # Subsample for drawing
    if n_slits <= MAX_RAYS_DRAWN:
        sel = np.arange(n_slits)
    else:
        sel = np.linspace(0, n_slits - 1, MAX_RAYS_DRAWN, dtype=int)
    rays_cam = rays_cam_full[sel, :]

    global_start = _fmt_utc(float(t_lookup.min()))
    global_end = _fmt_utc(float(t_lookup.max()))

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d", position=[0.05, 0.18, 0.9, 0.77])

    # MBES seafloor points (if available)
    mbes_scatter = None
    mbes_visible = [SHOW_MBES_BY_DEFAULT]
    if mbes_xyz is not None:
        x_mbes, y_mbes, z_mbes = mbes_xyz
        if x_mbes is not None and len(x_mbes) > 0:
            mbes_scatter = ax.scatter(
                x_mbes,
                y_mbes,
                z_mbes,
                c=[MBES_COLOR],
                s=MBES_POINT_SIZE,
                alpha=MBES_ALPHA,
                label="MBES seafloor",
                depthshade=True,
            )
            print(f"Added {len(x_mbes):,} MBES points to plot")

    # Background full mission
    ax.plot(xa, ya, za, linewidth=1.0, color="gray", alpha=0.45, label="Full mission")

    # Static colored segments (IMU track)
    for s in segments:
        ax.plot(
            s["x_imu"],
            s["y_imu"],
            s["z_imu"],
            linewidth=2.0,
            label=s["name"],
            color=s["color"],
            alpha=0.95,
        )

    # Moving points
    (imu_pt,) = ax.plot(
        [], [], [], "o", markersize=9, color="black", zorder=12, label="IMU"
    )
    (cam_pt,) = ax.plot(
        [], [], [], "o", markersize=8, color="magenta", zorder=12, label="Camera"
    )

    # Body-axis arrows as 3 line segments
    (x_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="red", zorder=11, label="Body X"
    )
    (y_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="green", zorder=11, label="Body Y"
    )
    (z_axis_line,) = ax.plot(
        [], [], [], linewidth=2.0, color="blue", zorder=11, label="Body Z"
    )

    # Rays as a Line3DCollection (we re-create per frame)
    rays_coll = [None]

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

    ttl = f"Mission with UHI segments + rays   {title_extra}\nGlobal lookup range: {global_start} to {global_end}"
    ax.set_title(ttl, pad=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Height (m)")
    ax.legend(loc="upper right", fontsize=8)

    n_total = len(xi)
    rays_visible = [SHOW_RAYS_BY_DEFAULT]

    def _update_frame(i):
        i = int(i)
        # IMU position
        px, py, pz = xi[i], yi[i], zi[i]

        # BODY->ENU rotation for this pose
        roll_rad = _to_radians(roll_arr[i])
        pitch_rad = _to_radians(pitch_arr[i])
        yaw_in = yaw_arr[i]
        R_bw = _body_to_enu_rotation(roll_rad, pitch_rad, yaw_in)

        # Camera position: IMU + R_bw * [2.5, 0, 0]_body
        cam_offset_body = np.array([CAMERA_FWD_OFFSET_M, 0.0, 0.0])
        cam_offset_enu = R_bw @ cam_offset_body
        cx, cy, cz = (
            px + cam_offset_enu[0],
            py + cam_offset_enu[1],
            pz + cam_offset_enu[2],
        )

        # Body axes arrows from IMU
        ax_len = AXIS_ARROW_LEN_M
        x_tip = np.array([px, py, pz]) + (R_bw @ np.array([ax_len, 0.0, 0.0]))
        y_tip = np.array([px, py, pz]) + (R_bw @ np.array([0.0, ax_len, 0.0]))
        z_tip = np.array([px, py, pz]) + (R_bw @ np.array([0.0, 0.0, ax_len]))

        # Update points/axes
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

        if rays_visible[0]:
            R_wc = R_bw @ R_B_FROM_CAM
            d_world = (R_wc @ rays_cam.T).T  # (K,3)
            # Normalize direction for clean length
            d_world = d_world / np.linalg.norm(d_world, axis=1, keepdims=True)

            # Build segments [start, end] for Line3DCollection
            starts = np.column_stack(
                [
                    np.full(len(d_world), cx),
                    np.full(len(d_world), cy),
                    np.full(len(d_world), cz),
                ]
            )
            ends = starts + d_world * RAY_LENGTH_M
            segments = np.stack([starts, ends], axis=1)

            # Sanity check: Z should point down -> dot with +Z should be negative
            up = np.array([0.0, 0.0, 1.0])
            dots = d_world @ up
            bad = dots > 0  # these point up
            colors = np.repeat([RAYS_COLOR], len(d_world), axis=0)
            if np.any(bad):
                colors = np.array(colors, dtype=float)
                colors[bad] = (1.0, 0.0, 0.0)  # highlight wrong-sign in red

            coll = Line3DCollection(segments, colors=colors, linewidths=0.8)
            rays_coll[0] = coll
            ax.add_collection3d(coll)

        mbes_status = ""
        if mbes_scatter is not None:
            mbes_status = f"\nMBES: {'ON' if mbes_visible[0] else 'OFF'}  (points={len(mbes_xyz[0]):,})"

        info.set_text(
            f"Global idx: {i}/{n_total-1}\n"
            f"Lookup UTC: {_fmt_utc(float(t_lookup[i]))}\n"
            f"Segment: {seg_names[seg_idx]}\n"
            f"RPY ({RPY_UNITS}): roll={roll_arr[i]:.3f}, pitch={pitch_arr[i]:.3f}, yaw={yaw_arr[i]:.3f}\n"
            f"Rays: {'ON' if rays_visible[0] else 'OFF'}  (count={len(rays_cam)})"
            + (
                f"  | bad(up)={int((d_world @ np.array([0,0,1])).gt(0).sum())}"
                if rays_visible[0]
                else ""
            )
            + mbes_status
        )
        return imu_pt, cam_pt, x_axis_line, y_axis_line, z_axis_line, info

    # Slider
    ax_slider = fig.add_axes([0.15, 0.06, 0.55, 0.03])
    slider = Slider(ax_slider, "Index", 0, n_total - 1, valinit=0, valstep=1)

    # Play/Pause
    ax_btn_play = fig.add_axes([0.72, 0.06, 0.1, 0.04])
    btn_play = Button(ax_btn_play, "Play/Pause")

    # Toggle rays
    ax_btn_rays = fig.add_axes([0.84, 0.06, 0.1, 0.04])
    btn_rays = Button(ax_btn_rays, "Rays On/Off")

    # Toggle MBES
    ax_btn_mbes = fig.add_axes([0.72, 0.01, 0.1, 0.04])
    btn_mbes = Button(ax_btn_mbes, "MBES On/Off")

    anim_running = [True]
    rays_visible[0] = SHOW_RAYS_BY_DEFAULT

    anim = FuncAnimation(
        fig, _update_frame, frames=n_total, interval=60, blit=False, repeat=True
    )

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
        rays_visible[0] = not rays_visible[0]
        _update_frame(int(slider.val))
        fig.canvas.draw_idle()

    def on_toggle_mbes(event):
        if mbes_scatter is not None:
            mbes_visible[0] = not mbes_visible[0]
            mbes_scatter.set_visible(mbes_visible[0])
            fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)
    btn_rays.on_clicked(on_toggle_rays)
    btn_mbes.on_clicked(on_toggle_mbes)

    # Initialize frame 0
    _update_frame(0)
    plt.show()
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Multi-H5 UHI with IMU/Cam frames, body axes, and camera rays"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="Optional per-H5 subsample count"
    )
    args = parser.parse_args()

    full_xyz, segments, combined, _nav_df = _load_segments(num_frames=args.frames)

    # Print per-file ranges
    print("\n=== Per-file HSI time bounds (used frames) ===")
    for s in segments:
        print(f"{s['name']}")
        print(f"  HSI UTC:    {s['hsi_start_utc']} to {s['hsi_end_utc']}")
        print(f"  Lookup UTC: {s['lookup_start_utc']} to {s['lookup_end_utc']}")
    print("")
    print(f"Orientation convention: {ORI_CONVENTION}, units: {RPY_UNITS}")
    print(f"Camera offset: {CAMERA_FWD_OFFSET_M} m along body +X")
    print("Camera->Body rotation (R_B_FROM_CAM):")
    print(R_B_FROM_CAM)

    # Load camera intrinsics for ray construction
    camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
    print(
        f"Camera calib: f={camera_calib['f']:.3f}, cx={camera_calib['cx']:.3f}, w={camera_calib['w']:.0f}"
    )

    # Load MBES mesh
    print("\n=== Loading MBES seafloor mesh ===")
    mbes_xyz = _load_mbes_mesh()
    if mbes_xyz[0] is not None:
        print(f"MBES loaded successfully - will be displayed in plot")
    else:
        print("MBES not loaded - continuing without seafloor visualization")

    title_extra = f"(offset {float(config.TIME_OFFSET_SEC):+.1f}s; rays shown up to {MAX_RAYS_DRAWN})"
    plot_multi_segments_with_slider_and_frames(
        full_xyz,
        segments,
        combined,
        camera_calib,
        mbes_xyz=mbes_xyz,
        title_extra=title_extra,
    )


if __name__ == "__main__":
    main()
