"""
Interactive 3D visualization of coordinate frames and ray directions.

Shows:
- IMU/Body coordinate system (origin)
            opacity=0.6,
            name=f'{name} pixel {idx}',
            showlegend=bool(idx == indices[0]),  # Only show one legend entry
            legendgroup='rays',
            hovertemplate=f'Pixel {idx}<br>Origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})<br>Direction: ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})<extra></extra>'era coordinate system (2.5m forward offset)
- Ray directions from camera pixels
- Vehicle trajectory
- Time slider to animate through frames

This helps debug the georeferencing pipeline by visualizing:
1. Where the camera is relative to IMU
2. Which direction each pixel is looking
3. How rays transform from camera frame to world frame
"""

import numpy as np
import pandas as pd
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
import utils


def create_coordinate_frame_arrows(origin, rotation_matrix, scale=10.0, name="Frame"):
    """
    Create 3D arrows representing a coordinate frame (X=red, Y=green, Z=blue).

    Parameters:
    -----------
    origin : array-like (3,)
        Origin of coordinate frame
    rotation_matrix : ndarray (3, 3)
        Rotation matrix defining frame orientation
    scale : float
        Length of axis arrows in meters
    name : str
        Name prefix for traces

    Returns:
    --------
    list
        List of plotly traces (3 arrows)
    """
    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]

    traces = []
    for i, (color, label) in enumerate(zip(colors, labels)):
        # Axis direction in world frame
        direction = rotation_matrix[:, i] * scale
        endpoint = origin + direction

        # Create arrow line
        trace = go.Scatter3d(
            x=[origin[0], endpoint[0]],
            y=[origin[1], endpoint[1]],
            z=[origin[2], endpoint[2]],
            mode="lines+text",
            line=dict(color=color, width=6),
            text=["", f"{name}-{label}"],
            textposition="top center",
            textfont=dict(size=10, color=color),
            name=f"{name} {label}-axis",
            showlegend=True,
            hovertemplate=f"{name} {label}-axis<br>Direction: {direction}<extra></extra>",
        )
        traces.append(trace)

    return traces


def create_ray_visualization(
    ray_origins,
    ray_directions,
    ray_length=50.0,
    num_rays_to_show=20,
    name="Rays",
    color="cyan",
):
    """
    Create visualization of ray directions from camera.

    Parameters:
    -----------
    ray_origins : ndarray (N, 3)
        Starting points of rays
    ray_directions : ndarray (N, 3)
        Direction vectors (should be normalized)
    ray_length : float
        Length to draw each ray
    num_rays_to_show : int
        Number of rays to visualize (evenly spaced)
    name : str
        Name for traces
    color : str
        Color of rays

    Returns:
    --------
    list
        List of plotly traces
    """
    # Select subset of rays to visualize (evenly spaced across slit)
    total_rays = len(ray_origins)
    indices = np.linspace(0, total_rays - 1, num_rays_to_show, dtype=int)

    traces = []
    for idx in indices:
        origin = ray_origins[idx]
        direction = ray_directions[idx]
        endpoint = origin + direction * ray_length

        trace = go.Scatter3d(
            x=[origin[0], endpoint[0]],
            y=[origin[1], endpoint[1]],
            z=[origin[2], endpoint[2]],
            mode="lines",
            line=dict(color=color, width=2),
            opacity=0.6,
            name=f"{name} pixel {idx}",
            showlegend=bool(idx == indices[0]),  # Only show one legend entry
            legendgroup="rays",
            hovertemplate=f"Pixel {idx}<br>Origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})<br>Direction: ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})<extra></extra>",
        )
        traces.append(trace)

    return traces


def load_and_process_trajectory(h5_path, nav_csv, num_frames=50):
    """
    Load trajectory data and prepare for visualization.

    Parameters:
    -----------
    h5_path : str
        Path to H5 file
    nav_csv : str
        Path to navigation CSV
    num_frames : int
        Number of frames to load

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

    # Limit to requested number of frames
    if len(hsi_timestamps) > num_frames:
        # Take evenly spaced frames
        indices = np.linspace(0, len(hsi_timestamps) - 1, num_frames, dtype=int)
        hsi_timestamps = hsi_timestamps[indices]

    print(f"Processing {len(hsi_timestamps)} frames...")

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


def create_interactive_visualization(trajectory_data):
    """
    Create interactive plotly visualization with time slider that follows the camera.

    Parameters:
    -----------
    trajectory_data : dict
        Dictionary with trajectory data from load_and_process_trajectory

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    n_frames = len(trajectory_data["timestamps"])
    imu_traj = trajectory_data["imu_positions"]
    cam_traj = trajectory_data["camera_positions"]

    # Create figure
    fig = go.Figure()

    # Prepare frames for animation
    frames = []

    for i in range(n_frames):
        frame_traces = []

        # Trajectory trail up to current frame (IMU path)
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

        # Trajectory trail up to current frame (Camera path)
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

        # Current IMU position marker
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

        # Current camera position marker
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

        # IMU coordinate frame
        imu_frame = create_coordinate_frame_arrows(
            imu_traj[i], trajectory_data["imu_orientations"][i], scale=5.0, name="IMU"
        )
        frame_traces.extend(imu_frame)

        # Camera coordinate frame
        cam_frame = create_coordinate_frame_arrows(
            cam_traj[i],
            trajectory_data["camera_orientations"][i],
            scale=5.0,
            name="Camera",
        )
        frame_traces.extend(cam_frame)

        # Transform ray directions to world frame
        ray_dirs_world = (
            trajectory_data["camera_orientations"][i]
            @ trajectory_data["ray_directions_camera"].T
        ).T

        # Create ray origins (all from camera position)
        ray_origins = np.tile(cam_traj[i], (len(ray_dirs_world), 1))

        # Visualize rays
        rays = create_ray_visualization(
            ray_origins,
            ray_dirs_world,
            ray_length=20.0,
            num_rays_to_show=15,
            name="Rays",
            color="cyan",
        )
        frame_traces.extend(rays)

        # Calculate camera position to follow the vehicle
        # Camera looks at current position from behind and above
        cam_pos = cam_traj[i]
        look_offset = 30.0  # Distance behind vehicle
        height_offset = 15.0  # Height above vehicle

        # Get heading direction (approximate from previous position)
        if i > 0:
            heading = cam_traj[i] - cam_traj[i - 1]
            heading_norm = np.linalg.norm(heading[:2])  # Only XY plane
            if heading_norm > 0.1:
                heading_unit = heading / heading_norm
            else:
                heading_unit = np.array([1, 0, 0])  # Default forward
        else:
            heading_unit = np.array([1, 0, 0])

        # Camera eye position: behind and above
        eye_offset = -heading_unit * look_offset
        eye_x = (
            cam_pos[0] + eye_offset[0] - cam_traj[0, 0]
        ) / 50.0  # Normalize relative to start
        eye_y = (cam_pos[1] + eye_offset[1] - cam_traj[0, 1]) / 50.0
        eye_z = (cam_pos[2] + height_offset - cam_traj[0, 2]) / 50.0

        # Center on current camera position
        center_x = (cam_pos[0] - cam_traj[0, 0]) / 50.0
        center_y = (cam_pos[1] - cam_traj[0, 1]) / 50.0
        center_z = (cam_pos[2] - cam_traj[0, 2]) / 50.0

        # Create frame with updated camera position
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

    # Add first frame data
    fig.add_traces(frames[0].data)

    # Add frames to figure
    fig.frames = frames

    # Add slider and play/pause buttons
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
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        title="Coordinate Frames and Ray Directions Visualization",
        height=900,
        showlegend=True,
        legend=dict(x=0.7, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    )

    return fig


def main():
    print("=" * 80)
    print("COORDINATE FRAMES & RAY DIRECTIONS VISUALIZATION")
    print("=" * 80)

    # Find first H5 file
    h5_folder = Path(config.H5_FOLDER)
    h5_files = sorted(h5_folder.glob("*.h5"))

    if not h5_files:
        print(f"❌ No H5 files found in {h5_folder}")
        return

    h5_path = h5_files[0]  # Use first file
    print(f"\nUsing: {h5_path.name}")

    # Load and process trajectory
    trajectory_data = load_and_process_trajectory(
        str(h5_path), config.NAV_CSV, num_frames=50  # Limit frames for performance
    )

    # Create visualization
    print("\nCreating interactive visualization...")
    fig = create_interactive_visualization(trajectory_data)

    # Save to HTML
    output_html = "coordinate_frames_visualization.html"
    fig.write_html(output_html)
    print(f"\n✅ Visualization saved: {output_html}")

    # Show in browser
    print("\nOpening visualization in browser...")
    fig.show()

    print("\n" + "=" * 80)
    print("INSTRUCTIONS:")
    print("=" * 80)
    print("• Blue line: IMU/Body trajectory")
    print("• Orange dashed line: Camera trajectory (2.5m forward offset)")
    print("• Red/Green/Blue arrows: Coordinate axes (X/Y/Z)")
    print("• Cyan lines: Ray directions from camera pixels")
    print("• Use slider to step through frames")
    print("• Click 'Play' to animate")
    print("• Rotate/zoom with mouse")
    print("=" * 80)

    print("\nWhat to check:")
    print("1. Camera should be 2.5m ahead of IMU in X direction")
    print("2. Coordinate axes should align (identity rotation)")
    print("3. Rays should point downward (negative Z in UTM)")
    print("4. Rays should spread in Y direction (starboard-port)")
    print("5. No spread in X direction (push-broom perpendicular)")


if __name__ == "__main__":
    main()
