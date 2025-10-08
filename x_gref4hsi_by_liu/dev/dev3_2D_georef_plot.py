"""
Enhanced plot_georef with navigation trajectory overlay
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import utils


def plot_georef_with_trajectory(
    cube,
    nav_csv_path=None,
    red_wl=654.2,
    green_wl=560.0,
    blue_wl=440.3,
    normalize=True,
    figsize=(11, 9),
    coordinate_system=None,
    use_local_origin=True,
    origin=None,
    show_file_boundaries=True,
    show_trajectory=True,
    trajectory_color="red",
    trajectory_linewidth=2.0,
    trajectory_alpha=0.8,
    trajectory_label="Navigation trajectory",
    interactive=True,
    **pcolor_kwargs,
):
    """
    Render an RGB composite with navigation trajectory overlay.

    This function extends the standard plot_georef() by adding the full
    navigation trajectory from the merged CSV file.

    Parameters:
    -----------
    cube : CombinedTransectCube
        The georeferenced HSI data cube
    nav_csv_path : str, optional
        Path to navigation CSV file. If None, uses config.NAV_CSV
    red_wl, green_wl, blue_wl : float
        Wavelengths for RGB composite (default: 654.2, 560.0, 440.3 nm)
    normalize : bool
        Whether to normalize RGB values (default: True)
    figsize : tuple
        Figure size (default: (11, 9))
    coordinate_system : str or None
        "LATLON" (default), "NED", or "ECEF"
    use_local_origin : bool
        For ECEF mode, use local origin offset (default: True)
    origin : tuple or None
        (lat, lon, h) for NED/ECEF modes. Auto-computed for LATLON
    show_file_boundaries : bool
        Show H5 file boundaries (default: True)
    show_trajectory : bool
        Show navigation trajectory overlay (default: True)
    trajectory_color : str
        Color for trajectory line (default: "red")
    trajectory_linewidth : float
        Width of trajectory line (default: 2.0)
    trajectory_alpha : float
        Alpha transparency for trajectory (default: 0.8)
    trajectory_label : str
        Label for trajectory in legend (default: "Navigation trajectory")
    interactive : bool
        Enable click-to-show coordinates (default: True)
    **pcolor_kwargs : dict
        Additional arguments passed to pcolormesh

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """

    # Default to LATLON if not specified
    if coordinate_system is None:
        coordinate_system = "LATLON"

    # Get navigation CSV path
    if nav_csv_path is None:
        nav_csv_path = config.NAV_CSV

    # First, call the standard plot_georef but capture the figure
    # We need to temporarily disable interactive mode to modify the plot
    original_interactive = plt.isinteractive()
    plt.ioff()  # Turn off interactive mode temporarily

    # Call the original plot_georef method
    cube.plot_georef(
        red_wl=red_wl,
        green_wl=green_wl,
        blue_wl=blue_wl,
        normalize=normalize,
        figsize=figsize,
        coordinate_system=coordinate_system,
        use_local_origin=use_local_origin,
        origin=origin,
        show_file_boundaries=show_file_boundaries,
        alpha_for_nodata=0.0,
        interactive=False,  # We'll add interactivity after adding trajectory
        **pcolor_kwargs,
    )

    # Get current figure and axes
    fig = plt.gcf()
    ax = plt.gca()

    # Add trajectory if requested
    if show_trajectory:
        try:
            # Load navigation data
            nav_df = utils.load_csv_navigation(nav_csv_path, config.CSV_COLUMNS)

            # Extract coordinates
            lon = nav_df["longitude"].to_numpy()
            lat = nav_df["latitude"].to_numpy()
            depth = nav_df["depth"].to_numpy()

            # Convert to ECEF first
            x_ecef, y_ecef, z_ecef = utils.geographic_to_ecef(
                lon,
                lat,
                -depth,  # depth is negative for underwater
                epsg_geo=config.EPSG_GEOGRAPHIC,
                epsg_ecef=config.EPSG_ECEF,
            )

            # Transform to the same coordinate system as the plot
            from pyproj import Transformer

            if coordinate_system.upper() == "LATLON":
                # Already have lat/lon
                x_traj, y_traj = lon, lat

            elif coordinate_system.upper() == "NED":
                # Convert ECEF to NED
                if origin is None:
                    # Use config origin or compute from data
                    if hasattr(config, "LAT0") and hasattr(config, "LON0"):
                        origin = (
                            float(config.LAT0),
                            float(config.LON0),
                            float(getattr(config, "H0", 0.0)),
                        )
                    else:
                        origin = (60.8011575, 10.7122345, 0.0)

                lat0, lon0, h0 = origin
                from utils.georef import _ecef_to_ned_arrays

                N, E, D = _ecef_to_ned_arrays(x_ecef, y_ecef, z_ecef, lat0, lon0, h0)
                x_traj, y_traj = E, N

            elif coordinate_system.upper() == "ECEF":
                # ECEF or ECEF-local
                if origin is None:
                    if hasattr(config, "LAT0") and hasattr(config, "LON0"):
                        origin = (
                            float(config.LAT0),
                            float(config.LON0),
                            float(getattr(config, "H0", 0.0)),
                        )
                    else:
                        origin = (60.8011575, 10.7122345, 0.0)

                if use_local_origin:
                    from utils.georef import _ecef_of_geodetic

                    lat0, lon0, h0 = origin
                    x0, y0, z0 = _ecef_of_geodetic(lat0, lon0, h0)
                    x_traj, y_traj = x_ecef - x0, y_ecef - y0
                else:
                    x_traj, y_traj = x_ecef, y_ecef
            else:
                raise ValueError(f"Unknown coordinate system: {coordinate_system}")

            # Plot trajectory
            ax.plot(
                x_traj,
                y_traj,
                color=trajectory_color,
                linewidth=trajectory_linewidth,
                alpha=trajectory_alpha,
                label=trajectory_label,
                zorder=10,  # Draw on top
            )

            # Update legend
            ax.legend()

            print(f"✅ Added trajectory with {len(lon)} navigation points")

        except Exception as e:
            print(f"⚠️  Could not add trajectory: {e}")

    # Add interactive click handler if requested
    if interactive:
        from pyproj import Transformer

        # Get the coordinate data from cube
        X_ecef, Y_ecef, Z_ecef = cube.X_ecef, cube.Y_ecef, cube.Z_ecef
        R, G, B = cube.R, cube.G, cube.B

        # Recompute display coordinates (same logic as in plot_georef)
        if coordinate_system.upper() == "LATLON":
            tf_ecef_to_geo = Transformer.from_crs(
                "EPSG:4978", "EPSG:4979", always_xy=True
            )
            lon_grid, lat_grid, height = tf_ecef_to_geo.transform(
                X_ecef, Y_ecef, Z_ecef
            )
            Xp, Yp = lon_grid, lat_grid
        elif coordinate_system.upper() == "NED":
            from utils.georef import _ecef_to_ned_arrays

            lat0, lon0, h0 = origin
            N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)
            Xp, Yp = E, N
        elif coordinate_system.upper() == "ECEF":
            if use_local_origin:
                from utils.georef import _ecef_of_geodetic

                lat0, lon0, h0 = origin
                x0, y0, z0 = _ecef_of_geodetic(lat0, lon0, h0)
                Xp, Yp = X_ecef - x0, Y_ecef - y0
            else:
                Xp, Yp = X_ecef, Y_ecef

        # Store data for click handler
        click_data = {
            "Xp": Xp,
            "Yp": Yp,
            "X_ecef": X_ecef,
            "Y_ecef": Y_ecef,
            "Z_ecef": Z_ecef,
            "R": R,
            "G": G,
            "B": B,
            "origin": origin,
            "coord_system": coordinate_system.upper(),
            "use_local_origin": use_local_origin,
        }

        # Create transformer for ECEF to lat/lon
        tf_ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

        def on_click(event):
            if event.inaxes is None:
                return

            # Get click position
            x_click, y_click = event.xdata, event.ydata

            # Find nearest grid point
            dist = (click_data["Xp"] - x_click) ** 2 + (click_data["Yp"] - y_click) ** 2
            min_idx = np.nanargmin(dist)
            track_idx, slit_idx = np.unravel_index(min_idx, click_data["Xp"].shape)

            # Get coordinates at this point
            x_plot = click_data["Xp"][track_idx, slit_idx]
            y_plot = click_data["Yp"][track_idx, slit_idx]
            x_ecef = click_data["X_ecef"][track_idx, slit_idx]
            y_ecef = click_data["Y_ecef"][track_idx, slit_idx]
            z_ecef = click_data["Z_ecef"][track_idx, slit_idx]

            # Get RGB values
            r_val = click_data["R"][track_idx, slit_idx]
            g_val = click_data["G"][track_idx, slit_idx]
            b_val = click_data["B"][track_idx, slit_idx]

            # Check if valid point
            if not (
                np.isfinite(x_ecef) and np.isfinite(y_ecef) and np.isfinite(z_ecef)
            ):
                print(f"⚠️  Invalid point at track={track_idx}, slit={slit_idx}")
                return

            # Convert ECEF to lat/lon/height
            lon_deg, lat_deg, height = tf_ecef_to_geo.transform(x_ecef, y_ecef, z_ecef)

            # Print info
            print("\n" + "=" * 70)
            print(f"📍 Clicked at track={track_idx}, slit={slit_idx}")

            # Show coordinates based on current display mode
            if click_data["coord_system"] == "LATLON":
                print(f"   Lat/Lon: {y_plot:.6f}°, {x_plot:.6f}°")
            elif click_data["coord_system"] == "NED":
                print(f"   NED: E={x_plot:.2f}m, N={y_plot:.2f}m")
            else:
                print(f"   Plot coords: ({x_plot:.2f}, {y_plot:.2f})")

            # Always show ECEF and WGS84 for reference
            print(f"   ECEF: X={x_ecef:.2f}m, Y={y_ecef:.2f}m, Z={z_ecef:.2f}m")
            print(f"   WGS84: Lat={lat_deg:.6f}°, Lon={lon_deg:.6f}°, h={height:.2f}m")

            if np.isfinite(r_val):
                print(f"   RGB: R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")
            else:
                print(f"   RGB: No data")
            print("=" * 70)

        # Connect click event
        fig.canvas.mpl_connect("button_press_event", on_click)
        print("💡 Interactive mode: Click on the plot to display coordinates")

    # Restore interactive mode and show
    if original_interactive:
        plt.ion()

    plt.tight_layout()
    plt.show()

    return fig, ax
