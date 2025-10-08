"""
Interactive visualization of coordinate frames and ray directions.
Supports both 2D (matplotlib - top-down map view) and 3D (plotly - interactive).

Usage:
    python visualize_coordinate_frames.py --mode 2d --frames 100
    python visualize_coordinate_frames.py --mode 3d --frames 50
    python visualize_coordinate_frames.py --mode both --frames 100
"""

import numpy as np
import pandas as pd
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow
from pathlib import Path
import sys
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
import utils


def load_and_process_trajectory(h5_path, nav_csv, num_frames=None):
    """
    Load trajectory data and prepare for visualization.

    Parameters:
    -----------
    h5_path : str
        Path to H5 file
    nav_csv : str
        Path to navigation CSV
    num_frames : int or None
        Number of frames to load (None = all frames)

    Returns:
    --------
    dict
        Dictionary with trajectory data
    """
    print(f"\nLoading data from {Path(h5_path).name}...")

    # Load navigation
    nav_data = utils.load_csv_navigation(nav_csv, config.CSV_COLUMNS)

    # Load HSI timestamps
    hsi_timestamps = utils.load_h5_timestamps(h5_path)
    print(f"Total frames available: {len(hsi_timestamps)}")

    # Limit to requested number of frames
    if num_frames is not None and len(hsi_timestamps) > num_frames:
        # Take evenly spaced frames
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

    # Apply sensor transform (get camera positions and orientations)
    camera_positions_ecef, camera_orientations = utils.apply_sensor_transform(
        positions_ecef,
        orientations,
        config.ROTATION_HSI_TO_BODY,
        config.TRANSLATION_BODY_TO_HSI,
    )

    # Convert to UTM for better visualization (local coordinates)
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

    # Build ray directions in camera frame
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


def create_2d_matplotlib_animation(
    trajectory_data, output_file="coordinate_frames_2d.gif"
):
    """
    Create 2D top-down map view animation using matplotlib.

    Parameters:
    -----------
    trajectory_data : dict
        Trajectory data from load_and_process_trajectory
    output_file : str
        Output filename for animation
    """
    n_frames = len(trajectory_data["timestamps"])
    imu_traj = trajectory_data["imu_positions"]
    cam_traj = trajectory_data["camera_positions"]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Set up plot limits with padding
    all_x = np.concatenate([imu_traj[:, 0], cam_traj[:, 0]])
    all_y = np.concatenate([imu_traj[:, 1], cam_traj[:, 1]])
    margin = 20
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Easting (m)", fontsize=12)
    ax.set_ylabel("Northing (m)", fontsize=12)
    ax.set_title("Coordinate Frames - Top-Down View", fontsize=14, fontweight="bold")

    # Initialize plot elements
    (imu_trail,) = ax.plot([], [], "b-", linewidth=2, label="IMU Trail", alpha=0.6)
    (cam_trail,) = ax.plot(
        [], [], "orange", linewidth=2, linestyle="--", label="Camera Trail", alpha=0.6
    )
    (imu_marker,) = ax.plot([], [], "bo", markersize=12, label="Current IMU")
    (cam_marker,) = ax.plot(
        [], [], "o", color="orange", markersize=12, label="Current Camera"
    )

    # Coordinate frame arrows (will be updated each frame)
    imu_x_arrow = None
    imu_y_arrow = None
    cam_x_arrow = None
    cam_y_arrow = None

    # Ray lines (show subset)
    ray_lines = []
    num_rays_to_show = 10

    # Text annotation
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.legend(loc="upper right", fontsize=10)

    def init():
        imu_trail.set_data([], [])
        cam_trail.set_data([], [])
        imu_marker.set_data([], [])
        cam_marker.set_data([], [])
        info_text.set_text("")
        return [imu_trail, cam_trail, imu_marker, cam_marker, info_text]

    def update(frame):
        nonlocal imu_x_arrow, imu_y_arrow, cam_x_arrow, cam_y_arrow, ray_lines

        # Update trails
        imu_trail.set_data(imu_traj[: frame + 1, 0], imu_traj[: frame + 1, 1])
        cam_trail.set_data(cam_traj[: frame + 1, 0], cam_traj[: frame + 1, 1])

        # Update current positions
        imu_marker.set_data([imu_traj[frame, 0]], [imu_traj[frame, 1]])
        cam_marker.set_data([cam_traj[frame, 0]], [cam_traj[frame, 1]])

        # Remove old arrows and rays
        if imu_x_arrow:
            imu_x_arrow.remove()
        if imu_y_arrow:
            imu_y_arrow.remove()
        if cam_x_arrow:
            cam_x_arrow.remove()
        if cam_y_arrow:
            cam_y_arrow.remove()
        for line in ray_lines:
            line.remove()
        ray_lines = []

        # Draw IMU coordinate frame (X=red, Y=green)
        R_imu = trajectory_data["imu_orientations"][frame]
        arrow_scale = 5.0

        # X-axis (red, forward)
        imu_x_arrow = ax.arrow(
            imu_traj[frame, 0],
            imu_traj[frame, 1],
            R_imu[0, 0] * arrow_scale,
            R_imu[1, 0] * arrow_scale,
            head_width=1.5,
            head_length=1.5,
            fc="red",
            ec="red",
            linewidth=2,
            alpha=0.8,
        )

        # Y-axis (green, starboard)
        imu_y_arrow = ax.arrow(
            imu_traj[frame, 0],
            imu_traj[frame, 1],
            R_imu[0, 1] * arrow_scale,
            R_imu[1, 1] * arrow_scale,
            head_width=1.5,
            head_length=1.5,
            fc="green",
            ec="green",
            linewidth=2,
            alpha=0.8,
        )

        # Draw Camera coordinate frame
        R_cam = trajectory_data["camera_orientations"][frame]

        # X-axis (red, forward)
        cam_x_arrow = ax.arrow(
            cam_traj[frame, 0],
            cam_traj[frame, 1],
            R_cam[0, 0] * arrow_scale,
            R_cam[1, 0] * arrow_scale,
            head_width=1.5,
            head_length=1.5,
            fc="darkred",
            ec="darkred",
            linewidth=2,
            alpha=0.6,
            linestyle="--",
        )

        # Y-axis (green, starboard)
        cam_y_arrow = ax.arrow(
            cam_traj[frame, 0],
            cam_traj[frame, 1],
            R_cam[0, 1] * arrow_scale,
            R_cam[1, 1] * arrow_scale,
            head_width=1.5,
            head_length=1.5,
            fc="darkgreen",
            ec="darkgreen",
            linewidth=2,
            alpha=0.6,
            linestyle="--",
        )

        # Draw rays (subset, projected onto XY plane)
        ray_dirs_world = (R_cam @ trajectory_data["ray_directions_camera"].T).T
        ray_length = 15.0
        indices = np.linspace(0, len(ray_dirs_world) - 1, num_rays_to_show, dtype=int)

        for idx in indices:
            ray_end_x = cam_traj[frame, 0] + ray_dirs_world[idx, 0] * ray_length
            ray_end_y = cam_traj[frame, 1] + ray_dirs_world[idx, 1] * ray_length
            (line,) = ax.plot(
                [cam_traj[frame, 0], ray_end_x],
                [cam_traj[frame, 1], ray_end_y],
                "c-",
                linewidth=1,
                alpha=0.4,
            )
            ray_lines.append(line)

        # Update info text
        info_text.set_text(
            f"Frame: {frame}/{n_frames-1}\n"
            f"Time: {trajectory_data['timestamps'][frame]:.2f}s\n"
            f"Roll: {trajectory_data['interp_nav']['roll'][frame]:.1f}°\n"
            f"Pitch: {trajectory_data['interp_nav']['pitch'][frame]:.1f}°\n"
            f"Yaw: {trajectory_data['interp_nav']['yaw'][frame]:.1f}°"
        )

        return [
            imu_trail,
            cam_trail,
            imu_marker,
            cam_marker,
            info_text,
            imu_x_arrow,
            imu_y_arrow,
            cam_x_arrow,
            cam_y_arrow,
        ] + ray_lines

    print(f"\nCreating 2D animation with {n_frames} frames...")
    print("This may take a minute...")
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=False,
        interval=100,
        repeat=True,
    )

    # Save animation
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer="pillow", fps=10, dpi=100)
    print(f"✅ 2D animation saved: {output_file}")

    plt.close()

    return output_file


def create_3d_plotly_visualization(
    trajectory_data, output_file="coordinate_frames_3d.html"
):
    """
    Create 3D interactive plotly visualization with tracking camera.

    Parameters:
    -----------
    trajectory_data : dict
        Trajectory data from load_and_process_trajectory
    output_file : str
        Output HTML filename
    """
    n_frames = len(trajectory_data["timestamps"])
    imu_traj = trajectory_data["imu_positions"]
    cam_traj = trajectory_data["camera_positions"]

    fig = go.Figure()

    # Prepare frames for animation
    frames = []

    for i in range(n_frames):
        frame_traces = []

        # Trajectory trails
        frame_traces.append(
            go.Scatter3d(
                x=imu_traj[: i + 1, 0],
                y=imu_traj[: i + 1, 1],
                z=imu_traj[: i + 1, 2],
                mode="lines",
                line=dict(color="blue", width=4),
                name="IMU Trail",
                showlegend=bool(i == 0),
            )
        )

        frame_traces.append(
            go.Scatter3d(
                x=cam_traj[: i + 1, 0],
                y=cam_traj[: i + 1, 1],
                z=cam_traj[: i + 1, 2],
                mode="lines",
                line=dict(color="orange", width=4, dash="dash"),
                name="Camera Trail",
                showlegend=bool(i == 0),
            )
        )

        # Current position markers
        frame_traces.append(
            go.Scatter3d(
                x=[imu_traj[i, 0]],
                y=[imu_traj[i, 1]],
                z=[imu_traj[i, 2]],
                mode="markers",
                marker=dict(size=12, color="blue", symbol="diamond"),
                name="Current IMU",
                showlegend=bool(i == 0),
            )
        )

        frame_traces.append(
            go.Scatter3d(
                x=[cam_traj[i, 0]],
                y=[cam_traj[i, 1]],
                z=[cam_traj[i, 2]],
                mode="markers",
                marker=dict(size=12, color="orange", symbol="circle"),
                name="Current Camera",
                showlegend=bool(i == 0),
            )
        )

        # Coordinate frames (X=red, Y=green, Z=blue)
        for name, pos, R in [
            ("IMU", imu_traj[i], trajectory_data["imu_orientations"][i]),
            ("Camera", cam_traj[i], trajectory_data["camera_orientations"][i]),
        ]:
            colors = ["red", "green", "blue"]
            labels = ["X", "Y", "Z"]
            scale = 5.0

            for j, (color, label) in enumerate(zip(colors, labels)):
                direction = R[:, j] * scale
                endpoint = pos + direction
                frame_traces.append(
                    go.Scatter3d(
                        x=[pos[0], endpoint[0]],
                        y=[pos[1], endpoint[1]],
                        z=[pos[2], endpoint[2]],
                        mode="lines",
                        line=dict(color=color, width=6),
                        name=f"{name} {label}",
                        showlegend=bool(i == 0 and name == "IMU"),
                    )
                )

        # Rays
        ray_dirs_world = (
            trajectory_data["camera_orientations"][i]
            @ trajectory_data["ray_directions_camera"].T
        ).T
        ray_origins = np.tile(cam_traj[i], (len(ray_dirs_world), 1))
        ray_length = 20.0
        num_rays = 15
        indices = np.linspace(0, len(ray_dirs_world) - 1, num_rays, dtype=int)

        for idx in indices:
            endpoint = ray_origins[idx] + ray_dirs_world[idx] * ray_length
            frame_traces.append(
                go.Scatter3d(
                    x=[ray_origins[idx, 0], endpoint[0]],
                    y=[ray_origins[idx, 1], endpoint[1]],
                    z=[ray_origins[idx, 2], endpoint[2]],
                    mode="lines",
                    line=dict(color="cyan", width=2),
                    opacity=0.6,
                    showlegend=bool(i == 0 and idx == indices[0]),
                    name="Rays",
                )
            )

        # Camera tracking
        cam_pos = cam_traj[i]
        look_offset = 30.0
        height_offset = 15.0

        if i > 0:
            heading = cam_traj[i] - cam_traj[i - 1]
            heading_norm = np.linalg.norm(heading[:2])
            heading_unit = (
                heading / heading_norm if heading_norm > 0.1 else np.array([1, 0, 0])
            )
        else:
            heading_unit = np.array([1, 0, 0])

        eye_offset = -heading_unit * look_offset
        eye_x = (cam_pos[0] + eye_offset[0] - cam_traj[0, 0]) / 50.0
        eye_y = (cam_pos[1] + eye_offset[1] - cam_traj[0, 1]) / 50.0
        eye_z = (cam_pos[2] + height_offset - cam_traj[0, 2]) / 50.0

        center_x = (cam_pos[0] - cam_traj[0, 0]) / 50.0
        center_y = (cam_pos[1] - cam_traj[0, 1]) / 50.0
        center_z = (cam_pos[2] - cam_traj[0, 2]) / 50.0

        frames.append(
            go.Frame(
                data=frame_traces,
                name=str(i),
                layout=go.Layout(
                    title_text=f"Frame {i}/{n_frames-1} | Time: {trajectory_data['timestamps'][i]:.2f}s | "
                    f"Roll: {trajectory_data['interp_nav']['roll'][i]:.1f}° | "
                    f"Pitch: {trajectory_data['interp_nav']['pitch'][i]:.1f}° | "
                    f"Yaw: {trajectory_data['interp_nav']['yaw'][i]:.1f}°",
                    scene=dict(
                        camera=dict(
                            eye=dict(x=eye_x, y=eye_y, z=eye_z),
                            center=dict(x=center_x, y=center_y, z=center_z),
                        )
                    ),
                ),
            )
        )

    fig.add_traces(frames[0].data)
    fig.frames = frames

    # Controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 100},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
                x=0.1,
                y=1.15,
            )
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "y": 0.0,
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ],
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="Height (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2), center=dict(x=0, y=0, z=0)),
        ),
        title="Coordinate Frames - 3D Interactive View",
        height=900,
        showlegend=True,
        legend=dict(x=0.7, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    )

    fig.write_html(output_file)
    print(f"✅ 3D visualization saved: {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Visualize coordinate frames and ray directions"
    )
    parser.add_argument(
        "--mode",
        choices=["2d", "3d", "both"],
        default="3d",
        help="Visualization mode: 2d (matplotlib), 3d (plotly), or both",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to visualize (default: all frames)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("COORDINATE FRAMES & RAY DIRECTIONS VISUALIZATION")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Frames: {'All' if args.frames is None else args.frames}")

    # Find first H5 file
    h5_folder = Path(config.H5_FOLDER)
    h5_files = sorted(h5_folder.glob("*.h5"))

    if not h5_files:
        print(f"❌ No H5 files found in {h5_folder}")
        return

    h5_path = h5_files[0]
    print(f"\nUsing: {h5_path.name}")

    # Load and process trajectory
    trajectory_data = load_and_process_trajectory(
        str(h5_path), config.NAV_CSV, num_frames=args.frames
    )

    # Create visualizations based on mode
    if args.mode in ["2d", "both"]:
        print("\n" + "=" * 80)
        print("Creating 2D matplotlib animation...")
        print("=" * 80)
        create_2d_matplotlib_animation(trajectory_data)

    if args.mode in ["3d", "both"]:
        print("\n" + "=" * 80)
        print("Creating 3D plotly visualization...")
        print("=" * 80)
        create_3d_plotly_visualization(trajectory_data)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print("\nOutput files:")
    if args.mode in ["2d", "both"]:
        print("  • coordinate_frames_2d.gif (2D top-down map animation)")
    if args.mode in ["3d", "both"]:
        print("  • coordinate_frames_3d.html (3D interactive visualization)")

    print("\nWhat to check:")
    print("  1. Camera should be 2.5m ahead of IMU in forward direction")
    print("  2. Coordinate axes should align (identity rotation)")
    print("  3. Rays should point downward and spread in Y direction")
    print("  4. No spread in X direction (push-broom perpendicular)")


if __name__ == "__main__":
    main()
