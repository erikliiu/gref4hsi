"""
Pure matplotlib visualization of coordinate frames and ray directions.
Interactive animation with proper controls - no HTML nonsense.

Usage:
    python visualize_coordinate_frames_v3.py --frames 100
    python visualize_coordinate_frames_v3.py --frames 200 --view 2d
    python visualize_coordinate_frames_v3.py --view 3d
"""

# --- GUI backend selection (must be before importing pyplot) ---
import matplotlib

for _bk in ("QtAgg", "Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(_bk)
        break
    except Exception:
        pass
print("Matplotlib backend:", matplotlib.get_backend())

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import proj3d, Axes3D
from pathlib import Path
import sys
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
import x_gref4hsi_by_liu.utils.utils as utils


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def load_and_process_trajectory(h5_path, nav_csv, num_frames=None):
    """
    Load trajectory data and prepare for visualization.
    """
    print(f"\nLoading data from {Path(h5_path).name}...")

    # Load navigation
    nav_data = utils.load_csv_navigation(nav_csv, config.CSV_COLUMNS)

    # Load HSI timestamps
    hsi_timestamps = utils.load_h5_timestamps(h5_path)
    print(f"Total frames available: {len(hsi_timestamps)}")

    # Limit to requested number of frames
    if num_frames is not None and len(hsi_timestamps) > num_frames:
        indices = np.linspace(0, len(hsi_timestamps) - 1, num_frames, dtype=int)
        hsi_timestamps = hsi_timestamps[indices]
        print(f"Using {num_frames} evenly-spaced frames")
    else:
        print(f"Using all {len(hsi_timestamps)} frames")

    # Interpolate navigation
    interp_nav = utils.interpolate_navigation(
        nav_data, hsi_timestamps, time_offset=config.TIME_OFFSET_SEC
    )

    # Convert to ECEF
    pos_ecef_x, pos_ecef_y, pos_ecef_z = utils.geographic_to_ecef(
        interp_nav["longitude"],
        interp_nav["latitude"],
        -interp_nav["depth"],
        epsg_geo=config.EPSG_GEOGRAPHIC,
        epsg_ecef=config.EPSG_ECEF,
    )
    positions_ecef = np.column_stack([pos_ecef_x, pos_ecef_y, pos_ecef_z])

    # Convert to rotation matrices
    orientations = utils.euler_to_rotation_matrix(
        interp_nav["roll"], interp_nav["pitch"], interp_nav["yaw"]
    )

    # Apply sensor transform
    camera_positions_ecef, camera_orientations = utils.apply_sensor_transform(
        positions_ecef,
        orientations,
        config.ROTATION_HSI_TO_BODY,
        config.TRANSLATION_BODY_TO_HSI,
    )

    # Convert to UTM
    imu_pos_utm_x, imu_pos_utm_y, imu_pos_utm_z = utils.ecef_to_utm(
        positions_ecef[:, 0],
        positions_ecef[:, 1],
        positions_ecef[:, 2],
        epsg_utm=config.EPSG_MBES,
        epsg_ecef=config.EPSG_ECEF,
    )
    imu_positions_utm = np.column_stack([imu_pos_utm_x, imu_pos_utm_y, imu_pos_utm_z])

    cam_pos_utm_x, cam_pos_utm_y, cam_pos_utm_z = utils.ecef_to_utm(
        camera_positions_ecef[:, 0],
        camera_positions_ecef[:, 1],
        camera_positions_ecef[:, 2],
        epsg_utm=config.EPSG_MBES,
        epsg_ecef=config.EPSG_ECEF,
    )
    camera_positions_utm = np.column_stack(
        [cam_pos_utm_x, cam_pos_utm_y, cam_pos_utm_z]
    )

    # Load camera calibration
    camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)

    # Build ray directions
    with h5py.File(h5_path, "r") as h5f:
        if "processed/radiance/dataCube" in h5f:
            n_slits = h5f["processed/radiance/dataCube"].shape[1]
        else:
            n_slits = int(camera_calib["w"])

    ray_directions_camera = utils.build_ray_directions(camera_calib, n_slits)

    print(f"Loaded {len(hsi_timestamps)} frames with {n_slits} pixels each")

    return {
        "timestamps": hsi_timestamps,
        "imu_positions": imu_positions_utm,
        "imu_orientations": orientations,
        "camera_positions": camera_positions_utm,
        "camera_orientations": camera_orientations,
        "ray_directions_camera": ray_directions_camera,
        "n_slits": n_slits,
        "interp_nav": interp_nav,
    }


def create_2d_animation(trajectory_data, full_mission_nav=None):
    """
    Create 2D top-down matplotlib animation.

    Parameters:
    -----------
    trajectory_data : dict
        Trajectory data for H5 frames
    full_mission_nav : dict, optional
        Complete mission navigation from CSV (all points)
    """
    n_frames = len(trajectory_data["timestamps"])
    imu_traj = trajectory_data["imu_positions"]
    cam_traj = trajectory_data["camera_positions"]

    # Create figure with room for controls
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(bottom=0.15)  # Make room for slider

    # Set up plot limits
    all_x = np.concatenate([imu_traj[:, 0], cam_traj[:, 0]])
    all_y = np.concatenate([imu_traj[:, 1], cam_traj[:, 1]])

    # If we have full mission data, include it in bounds
    if full_mission_nav is not None:
        all_x = np.concatenate([all_x, full_mission_nav["x"]])
        all_y = np.concatenate([all_y, full_mission_nav["y"]])

    margin = 50  # Larger margin to see full context
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Easting (m)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Northing (m)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Coordinate Frames - Top-Down View (Complete Mission + H5 Data)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Initialize plot elements
    # Complete mission trajectory (gray, all data from CSV)
    if full_mission_nav is not None:
        (mission_path,) = ax.plot(
            full_mission_nav["x"],
            full_mission_nav["y"],
            "gray",
            linewidth=0.8,
            label="Complete Mission (All CSV Data)",
            alpha=0.3,
            zorder=0,
        )

    # H5 trajectory paths (subset corresponding to H5 frames)
    (full_imu_path,) = ax.plot(
        imu_traj[:, 0],
        imu_traj[:, 1],
        "b-",
        linewidth=1.2,
        label="H5 IMU Path",
        alpha=0.3,
        zorder=1,
    )
    (full_cam_path,) = ax.plot(
        cam_traj[:, 0],
        cam_traj[:, 1],
        "orange",
        linewidth=1.2,
        linestyle="--",
        label="H5 Camera Path",
        alpha=0.3,
        zorder=1,
    )

    # Animated trails (builds as animation progresses)
    (imu_trail,) = ax.plot(
        [], [], "b-", linewidth=2.5, label="IMU Trail", alpha=0.8, zorder=3
    )
    (cam_trail,) = ax.plot(
        [],
        [],
        "orange",
        linewidth=2.5,
        linestyle="--",
        label="Camera Trail",
        alpha=0.8,
        zorder=3,
    )

    # Current position markers
    (imu_marker,) = ax.plot([], [], "bo", markersize=14, label="Current IMU", zorder=10)
    (cam_marker,) = ax.plot(
        [], [], "o", color="orange", markersize=14, label="Current Camera", zorder=10
    )

    # Ray lines
    ray_lines = []
    num_rays_to_show = 12

    # Text annotation
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="wheat",
            alpha=0.9,
            edgecolor="black",
            linewidth=1.5,
        ),
        family="monospace",
    )

    ax.legend(
        loc="upper right", fontsize=11, framealpha=0.9, edgecolor="black", fancybox=True
    )

    # Store arrow patches
    arrows = []

    def init():
        imu_trail.set_data([], [])
        cam_trail.set_data([], [])
        imu_marker.set_data([], [])
        cam_marker.set_data([], [])
        info_text.set_text("")
        return [
            full_imu_path,
            full_cam_path,
            imu_trail,
            cam_trail,
            imu_marker,
            cam_marker,
            info_text,
        ]

    def update(frame):
        # Update trails
        imu_trail.set_data(imu_traj[: frame + 1, 0], imu_traj[: frame + 1, 1])
        cam_trail.set_data(cam_traj[: frame + 1, 0], cam_traj[: frame + 1, 1])

        # Update markers
        imu_marker.set_data([imu_traj[frame, 0]], [imu_traj[frame, 1]])
        cam_marker.set_data([cam_traj[frame, 0]], [cam_traj[frame, 1]])

        # Remove old arrows and rays
        for arrow in arrows:
            arrow.remove()
        arrows.clear()

        for line in ray_lines:
            line.remove()
        ray_lines.clear()

        # Draw coordinate frames
        arrow_scale = 6.0
        arrow_width = 2.0
        head_width = 2.5
        head_length = 2.0

        # IMU frame
        R_imu = trajectory_data["imu_orientations"][frame]

        # X-axis (red, forward)
        arrow = ax.arrow(
            imu_traj[frame, 0],
            imu_traj[frame, 1],
            R_imu[0, 0] * arrow_scale,
            R_imu[1, 0] * arrow_scale,
            head_width=head_width,
            head_length=head_length,
            fc="red",
            ec="darkred",
            linewidth=arrow_width,
            alpha=0.9,
            zorder=5,
        )
        arrows.append(arrow)

        # Y-axis (green, starboard)
        arrow = ax.arrow(
            imu_traj[frame, 0],
            imu_traj[frame, 1],
            R_imu[0, 1] * arrow_scale,
            R_imu[1, 1] * arrow_scale,
            head_width=head_width,
            head_length=head_length,
            fc="green",
            ec="darkgreen",
            linewidth=arrow_width,
            alpha=0.9,
            zorder=5,
        )
        arrows.append(arrow)

        # Camera frame
        R_cam = trajectory_data["camera_orientations"][frame]

        # X-axis (dark red, forward)
        arrow = ax.arrow(
            cam_traj[frame, 0],
            cam_traj[frame, 1],
            R_cam[0, 0] * arrow_scale,
            R_cam[1, 0] * arrow_scale,
            head_width=head_width,
            head_length=head_length,
            fc="darkred",
            ec="red",
            linewidth=arrow_width,
            alpha=0.7,
            linestyle="--",
            zorder=4,
        )
        arrows.append(arrow)

        # Y-axis (dark green, starboard)
        arrow = ax.arrow(
            cam_traj[frame, 0],
            cam_traj[frame, 1],
            R_cam[0, 1] * arrow_scale,
            R_cam[1, 1] * arrow_scale,
            head_width=head_width,
            head_length=head_length,
            fc="darkgreen",
            ec="green",
            linewidth=arrow_width,
            alpha=0.7,
            linestyle="--",
            zorder=4,
        )
        arrows.append(arrow)

        # Draw rays (XY projection)
        ray_dirs_world = (R_cam @ trajectory_data["ray_directions_camera"].T).T
        ray_length = 18.0
        indices = np.linspace(0, len(ray_dirs_world) - 1, num_rays_to_show, dtype=int)

        for idx in indices:
            ray_end_x = cam_traj[frame, 0] + ray_dirs_world[idx, 0] * ray_length
            ray_end_y = cam_traj[frame, 1] + ray_dirs_world[idx, 1] * ray_length
            (line,) = ax.plot(
                [cam_traj[frame, 0], ray_end_x],
                [cam_traj[frame, 1], ray_end_y],
                "c-",
                linewidth=1.5,
                alpha=0.5,
                zorder=3,
            )
            ray_lines.append(line)

        # Update info text
        info_text.set_text(
            f"Frame: {frame:3d}/{n_frames-1:3d}\n"
            f"Time:  {trajectory_data['timestamps'][frame]:8.2f}s\n"
            f"Roll:  {trajectory_data['interp_nav']['roll'][frame]:6.1f}°\n"
            f"Pitch: {trajectory_data['interp_nav']['pitch'][frame]:6.1f}°\n"
            f"Yaw:   {trajectory_data['interp_nav']['yaw'][frame]:6.1f}°"
        )

        return (
            [imu_trail, cam_trail, imu_marker, cam_marker, info_text]
            + arrows
            + ray_lines
        )

    print(f"\nCreating 2D animation with {n_frames} frames...")
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=True,
        interval=100,
        repeat=True,
    )

    # Add custom controls
    # Slider for frame selection
    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

    # Play/Pause button
    ax_button = plt.axes([0.82, 0.05, 0.1, 0.04])
    btn_play = Button(ax_button, "Play/Pause")

    # Control state
    anim_running = [True]

    def on_slider_change(val):
        frame = int(slider.val)
        anim.event_source.stop()
        anim_running[0] = False
        update(frame)
        fig.canvas.draw_idle()

    def on_play_pause(event):
        if anim_running[0]:
            anim.event_source.stop()
            anim_running[0] = False
        else:
            anim.event_source.start()
            anim_running[0] = True

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)

    plt.tight_layout()
    return fig, anim


def create_3d_animation(trajectory_data, full_mission_nav=None):
    """
    Create 3D matplotlib animation.

    Parameters:
    -----------
    trajectory_data : dict
        Trajectory data for H5 frames
    full_mission_nav : dict, optional
        Complete mission navigation from CSV (all points)
    """
    n_frames = len(trajectory_data["timestamps"])
    imu_traj = trajectory_data["imu_positions"]
    cam_traj = trajectory_data["camera_positions"]

    # Create figure with 3D axes
    fig = plt.figure(figsize=(14, 10))
    # Position the 3D axes to leave room at bottom for controls
    ax = fig.add_subplot(111, projection="3d", position=[0.05, 0.15, 0.9, 0.8])

    # Set up plot limits
    all_x = np.concatenate([imu_traj[:, 0], cam_traj[:, 0]])
    all_y = np.concatenate([imu_traj[:, 1], cam_traj[:, 1]])
    all_z = np.concatenate([imu_traj[:, 2], cam_traj[:, 2]])

    # If we have full mission data, include it in bounds
    if full_mission_nav is not None:
        all_x = np.concatenate([all_x, full_mission_nav["x"]])
        all_y = np.concatenate([all_y, full_mission_nav["y"]])
        all_z = np.concatenate([all_z, full_mission_nav["z"]])

    margin = 50  # Larger margin for full context
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_zlim(all_z.min() - margin, all_z.max() + margin)

    ax.set_xlabel("Easting (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Northing (m)", fontsize=12, fontweight="bold")
    ax.set_zlabel("Height (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Coordinate Frames - 3D View (Complete Mission + H5 Data)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Initialize plot elements
    # Complete mission trajectory (gray, all data from CSV)
    if full_mission_nav is not None:
        (mission_path,) = ax.plot(
            full_mission_nav["x"],
            full_mission_nav["y"],
            full_mission_nav["z"],
            "gray",
            linewidth=0.8,
            label="Complete Mission (All CSV Data)",
            alpha=0.3,
            zorder=0,
        )

    # H5 trajectory paths (subset corresponding to H5 frames)
    (full_imu_path,) = ax.plot(
        imu_traj[:, 0],
        imu_traj[:, 1],
        imu_traj[:, 2],
        "b-",
        linewidth=1.2,
        label="H5 IMU Path",
        alpha=0.3,
        zorder=1,
    )
    (full_cam_path,) = ax.plot(
        cam_traj[:, 0],
        cam_traj[:, 1],
        cam_traj[:, 2],
        "orange",
        linewidth=1.2,
        linestyle="--",
        label="H5 Camera Path",
        alpha=0.3,
        zorder=1,
    )

    # Animated trails
    (imu_trail,) = ax.plot(
        [], [], [], "b-", linewidth=2, label="IMU Trail", alpha=0.8, zorder=3
    )
    (cam_trail,) = ax.plot(
        [],
        [],
        [],
        "orange",
        linewidth=2,
        linestyle="--",
        label="Camera Trail",
        alpha=0.8,
        zorder=3,
    )

    # Current position markers
    (imu_marker,) = ax.plot(
        [], [], [], "bo", markersize=10, label="Current IMU", zorder=10
    )
    (cam_marker,) = ax.plot(
        [],
        [],
        [],
        "o",
        color="orange",
        markersize=10,
        label="Current Camera",
        zorder=10,
    )

    # Ray lines
    ray_lines = []
    num_rays_to_show = 10

    # Coordinate frame arrows
    arrows = []

    # Info text
    info_text = ax.text2D(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="wheat", alpha=0.9),
        family="monospace",
    )

    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    def init():
        imu_trail.set_data([], [])
        imu_trail.set_3d_properties([])
        cam_trail.set_data([], [])
        cam_trail.set_3d_properties([])
        imu_marker.set_data([], [])
        imu_marker.set_3d_properties([])
        cam_marker.set_data([], [])
        cam_marker.set_3d_properties([])
        info_text.set_text("")
        return [
            full_imu_path,
            full_cam_path,
            imu_trail,
            cam_trail,
            imu_marker,
            cam_marker,
            info_text,
        ]

    def update(frame):
        # Update trails
        imu_trail.set_data(imu_traj[: frame + 1, 0], imu_traj[: frame + 1, 1])
        imu_trail.set_3d_properties(imu_traj[: frame + 1, 2])
        cam_trail.set_data(cam_traj[: frame + 1, 0], cam_traj[: frame + 1, 1])
        cam_trail.set_3d_properties(cam_traj[: frame + 1, 2])

        # Update markers
        imu_marker.set_data([imu_traj[frame, 0]], [imu_traj[frame, 1]])
        imu_marker.set_3d_properties([imu_traj[frame, 2]])
        cam_marker.set_data([cam_traj[frame, 0]], [cam_traj[frame, 1]])
        cam_marker.set_3d_properties([cam_traj[frame, 2]])

        # Remove old arrows and rays
        for arrow in arrows:
            arrow.remove()
        arrows.clear()

        for line in ray_lines:
            line.remove()
        ray_lines.clear()

        # Draw coordinate frames
        arrow_scale = 6.0
        arrow_props = dict(arrowstyle="->", lw=3, mutation_scale=20)

        # IMU frame (solid colors)
        R_imu = trajectory_data["imu_orientations"][frame]
        colors_imu = ["red", "green", "blue"]
        for i, color in enumerate(colors_imu):
            direction = R_imu[:, i] * arrow_scale
            endpoint = imu_traj[frame] + direction
            arrow = Arrow3D(
                [imu_traj[frame, 0], endpoint[0]],
                [imu_traj[frame, 1], endpoint[1]],
                [imu_traj[frame, 2], endpoint[2]],
                color=color,
                **arrow_props,
                alpha=0.9,
            )
            ax.add_artist(arrow)
            arrows.append(arrow)

        # Camera frame (darker colors)
        R_cam = trajectory_data["camera_orientations"][frame]
        colors_cam = ["darkred", "darkgreen", "darkblue"]
        for i, color in enumerate(colors_cam):
            direction = R_cam[:, i] * arrow_scale
            endpoint = cam_traj[frame] + direction
            arrow = Arrow3D(
                [cam_traj[frame, 0], endpoint[0]],
                [cam_traj[frame, 1], endpoint[1]],
                [cam_traj[frame, 2], endpoint[2]],
                color=color,
                **arrow_props,
                alpha=0.7,
                linestyle="--",
            )
            ax.add_artist(arrow)
            arrows.append(arrow)

        # Draw rays
        ray_dirs_world = (R_cam @ trajectory_data["ray_directions_camera"].T).T
        ray_length = 18.0
        indices = np.linspace(0, len(ray_dirs_world) - 1, num_rays_to_show, dtype=int)

        for idx in indices:
            endpoint = cam_traj[frame] + ray_dirs_world[idx] * ray_length
            (line,) = ax.plot(
                [cam_traj[frame, 0], endpoint[0]],
                [cam_traj[frame, 1], endpoint[1]],
                [cam_traj[frame, 2], endpoint[2]],
                "c-",
                linewidth=1.5,
                alpha=0.5,
            )
            ray_lines.append(line)

        # Update info text
        info_text.set_text(
            f"Frame: {frame:3d}/{n_frames-1:3d}\n"
            f"Time:  {trajectory_data['timestamps'][frame]:8.2f}s\n"
            f"Roll:  {trajectory_data['interp_nav']['roll'][frame]:6.1f}°\n"
            f"Pitch: {trajectory_data['interp_nav']['pitch'][frame]:6.1f}°\n"
            f"Yaw:   {trajectory_data['interp_nav']['yaw'][frame]:6.1f}°"
        )

        return (
            [imu_trail, cam_trail, imu_marker, cam_marker, info_text]
            + arrows
            + ray_lines
        )

    print(f"\nCreating 3D animation with {n_frames} frames...")
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=False,
        interval=100,
        repeat=True,
    )

    # Add custom controls (positioned at bottom, with enough clearance)
    # Slider for frame selection - positioned lower to avoid 3D axes overlap
    ax_slider = fig.add_axes([0.15, 0.05, 0.65, 0.03], facecolor="lightgray")
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

    # Play/Pause button
    ax_button = fig.add_axes([0.82, 0.05, 0.1, 0.04], facecolor="lightgray")
    btn_play = Button(ax_button, "Play/Pause")

    # Control state
    anim_running = [True]

    def on_slider_change(val):
        frame = int(slider.val)
        # Stop animation
        anim.event_source.stop()
        anim_running[0] = False
        # Update the plot manually
        update(frame)
        # Force complete redraw for 3D - need to explicitly call canvas methods
        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()
        except:
            pass

    def on_play_pause(event):
        if anim_running[0]:
            anim.event_source.stop()
            anim_running[0] = False
        else:
            anim.event_source.start()
            anim_running[0] = True

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)

    return fig, anim


def main():
    parser = argparse.ArgumentParser(
        description="Pure matplotlib coordinate frame visualization"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to visualize (default: 100)",
    )
    parser.add_argument(
        "--view",
        choices=["2d", "3d"],
        default="3d",
        help="View type: 2d (top-down) or 3d (default: 2d)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("COORDINATE FRAMES VISUALIZATION - PURE MATPLOTLIB")
    print("=" * 80)
    print(f"View: {args.view.upper()}")
    print(f"Frames: {args.frames}")

    # Find first H5 file
    h5_folder = Path(config.H5_FOLDER)
    h5_files = sorted(h5_folder.glob("*.h5"))

    if not h5_files:
        print(f"❌ No H5 files found in {h5_folder}")
        return

    h5_path = h5_files[0]
    print(f"\nUsing: {h5_path.name}")

    # Load and process trajectory (H5 subset)
    trajectory_data = load_and_process_trajectory(
        str(h5_path), config.NAV_CSV, num_frames=args.frames
    )

    # Load complete mission navigation from CSV (all points)
    print(f"\n📍 Loading complete mission trajectory from CSV...")
    full_nav_df = utils.load_csv_navigation(config.NAV_CSV, config.CSV_COLUMNS)
    print(f"   Total navigation points: {len(full_nav_df)}")

    # Convert all points to UTM
    print(f"   Converting to UTM...")
    # DataFrame columns have been renamed to standard names (keys from CSV_COLUMNS dict)
    lon_all = full_nav_df["longitude"].values
    lat_all = full_nav_df["latitude"].values
    height_all = full_nav_df["altitude"].values

    # Convert to ECEF first
    x_ecef_all, y_ecef_all, z_ecef_all = utils.geographic_to_ecef(
        lat_all, lon_all, height_all
    )

    # Then ECEF to UTM
    x_utm_all, y_utm_all, z_utm_all = utils.ecef_to_utm(
        x_ecef_all, y_ecef_all, z_ecef_all, config.EPSG_UTM
    )

    full_mission_nav = {"x": x_utm_all, "y": y_utm_all, "z": z_utm_all}
    print(f"   ✓ Complete mission trajectory loaded ({len(x_utm_all)} points)")

    # Create visualization
    if args.view == "2d":
        fig, anim = create_2d_animation(trajectory_data, full_mission_nav)
    else:
        fig, anim = create_3d_animation(trajectory_data, full_mission_nav)

    print("\n" + "=" * 80)
    print("ANIMATION READY!")
    print("=" * 80)
    print("\nControls:")
    print("  • Use the matplotlib animation toolbar at bottom")
    print("  • Play/Pause button to start/stop")
    print("  • Slider to scrub through frames")
    print("  • Pan/zoom with mouse (won't reset on play!)")
    print("  • Close window when done")
    print("\nWhat to check:")
    print("  1. Camera (orange) should be ~2.5m ahead of IMU (blue)")
    print("  2. Coordinate axes should align (same directions)")
    print("  3. Rays should fan out in Y direction (starboard-port)")
    print("  4. No fanning in X direction (perpendicular scan)")
    if args.view == "3d":
        print("  5. Rays should point downward (negative Z)")
    print("=" * 80)

    plt.show()


if __name__ == "__main__":
    main()
