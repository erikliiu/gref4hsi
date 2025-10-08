"""
Diagnostic visualization: HSI track vs MBES coverage
Shows why ray tracing is missing (97% failure rate)
"""

import numpy as np
import pandas as pd
import folium
from folium.features import DivIcon
from folium.plugins import MeasureControl, MousePosition
import h5py
import rasterio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
import config
import utils


def load_full_mission_track(nav_csv):
    """
    Load complete mission trajectory from merged CSV.
    Returns DataFrame with all navigation data.
    """
    nav_data = utils.load_csv_navigation(nav_csv, config.CSV_COLUMNS)
    print(f"\nTotal mission trajectory points: {len(nav_data)}")
    return nav_data


def load_hsi_track(h5_files, nav_csv):
    """
    Load HSI camera positions for all H5 files (subset of mission).
    Returns DataFrame with columns: timestamp, lat, lon, depth, altitude
    """
    # Load navigation CSV
    nav_data = utils.load_csv_navigation(nav_csv, config.CSV_COLUMNS)

    all_positions = []

    for h5_path in h5_files:
        print(f"\nProcessing {Path(h5_path).name}...")

        # Load HSI timestamps
        hsi_timestamps = utils.load_h5_timestamps(h5_path)

        # Interpolate navigation
        interp_nav = utils.interpolate_navigation(
            nav_data, hsi_timestamps, time_offset=config.TIME_OFFSET_SEC
        )

        # Store positions
        for i in range(len(hsi_timestamps)):
            all_positions.append(
                {
                    "timestamp": hsi_timestamps[i],
                    "latitude": interp_nav["latitude"][i],
                    "longitude": interp_nav["longitude"][i],
                    "depth": interp_nav["depth"][i],
                    "altitude": (
                        nav_data["altitude"].iloc[0]
                        if "altitude" in nav_data
                        else np.nan
                    ),
                }
            )

    df = pd.DataFrame(all_positions)
    print(f"\nHSI-specific positions: {len(df)}")
    return df


def load_mbes_bounds(geotiff_path):
    """
    Load MBES GeoTIFF bounds in geographic coordinates.
    Returns: (min_lon, min_lat, max_lon, max_lat)
    """
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds  # (left, bottom, right, top) in UTM

        # Convert UTM corners to geographic
        from pyproj import Transformer

        transformer = Transformer.from_crs(src.crs, 4326, always_xy=True)

        # Transform all 4 corners
        corners_utm = [
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top),
        ]

        corners_geo = []
        for x, y in corners_utm:
            lon, lat = transformer.transform(x, y)
            corners_geo.append((lat, lon))

        # Get bounding box
        lats = [c[0] for c in corners_geo]
        lons = [c[1] for c in corners_geo]

        print(f"\n=== MBES Coverage ===")
        print(f"UTM bounds: {bounds}")
        print(f"Geographic bounds:")
        print(f"  Lat: {min(lats):.6f} to {max(lats):.6f}")
        print(f"  Lon: {min(lons):.6f} to {max(lons):.6f}")

        return corners_geo, (min(lons), min(lats), max(lons), max(lats))


def estimate_swath_width(altitude, camera_fov_deg=45):
    """
    Estimate HSI swath width based on altitude and field of view.
    Returns swath width in meters.
    """
    fov_rad = np.deg2rad(camera_fov_deg)
    swath = 2 * altitude * np.tan(fov_rad / 2)
    return swath


def create_diagnostic_map(
    full_mission_df, hsi_track_df, mbes_corners, mbes_bbox, output_html
):
    """
    Create interactive map showing:
    1. Full mission trajectory (gray line)
    2. HSI subset track (red line)
    3. MBES coverage (blue rectangle)
    4. Estimated UHI swath width and coverage area
    """
    # Center map on mission trajectory
    lat0 = full_mission_df["latitude"].mean()
    lon0 = full_mission_df["longitude"].mean()

    m = folium.Map(
        location=[lat0, lon0],
        zoom_start=15,
        control_scale=True,
    )

    # Add satellite layer
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Add mouse position
    MousePosition(
        position="bottomright",
        separator=" , ",
        prefix="Lat/Lng:",
        lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 6);}",
    ).add_to(m)

    # Add measure tool
    MeasureControl(
        position="topleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
    ).add_to(m)

    # 1. Full Mission Trajectory (gray line)
    mission_fg = folium.FeatureGroup(name="Full Mission Trajectory", show=True)
    mission_coords = list(
        zip(full_mission_df["latitude"], full_mission_df["longitude"])
    )
    folium.PolyLine(
        mission_coords,
        color="gray",
        weight=2,
        opacity=0.6,
        tooltip=f"Full Mission ({len(mission_coords)} points)",
    ).add_to(mission_fg)
    mission_fg.add_to(m)

    # 2. MBES Coverage (polygon)
    mbes_fg = folium.FeatureGroup(name="MBES Coverage", show=True)
    folium.Polygon(
        locations=mbes_corners,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.2,
        weight=2,
        tooltip=f"MBES GeoTIFF Coverage<br>Bounds: {mbes_bbox}",
    ).add_to(mbes_fg)
    mbes_fg.add_to(m)

    # 3. HSI Subset Track (red line)
    hsi_track_fg = folium.FeatureGroup(name="UHI HSI Track", show=True)
    track_coords = list(zip(hsi_track_df["latitude"], hsi_track_df["longitude"]))
    folium.PolyLine(
        track_coords,
        color="red",
        weight=3,
        opacity=0.8,
        tooltip=f"UHI HSI Track ({len(track_coords)} frames)",
    ).add_to(hsi_track_fg)

    # Start/End markers
    folium.CircleMarker(
        location=track_coords[0],
        radius=8,
        color="green",
        fill=True,
        fill_opacity=1.0,
        tooltip="HSI Start",
    ).add_to(hsi_track_fg)

    folium.CircleMarker(
        location=track_coords[-1],
        radius=8,
        color="red",
        fill=True,
        fill_opacity=1.0,
        tooltip="HSI End",
    ).add_to(hsi_track_fg)

    hsi_track_fg.add_to(m)

    # 4. Estimated UHI swath width and coverage area
    swath_fg = folium.FeatureGroup(name="UHI Swath Coverage", show=True)

    avg_altitude = (
        hsi_track_df["altitude"].mean()
        if not hsi_track_df["altitude"].isna().all()
        else 20.0
    )
    avg_depth = abs(hsi_track_df["depth"].mean())

    # Use altitude from vehicle to seabed (depth + altitude reading)
    effective_altitude = (
        avg_depth + avg_altitude if not np.isnan(avg_altitude) else avg_depth
    )

    swath_width_m = estimate_swath_width(effective_altitude, camera_fov_deg=45)

    print(f"\n=== HSI Swath Estimate ===")
    print(f"Average depth: {avg_depth:.1f} m")
    print(f"Average altitude (to seabed): {avg_altitude:.1f} m")
    print(f"Effective altitude: {effective_altitude:.1f} m")
    print(f"Estimated swath width: {swath_width_m:.1f} m")

    # Draw swath rectangles every N points
    step = max(1, len(track_coords) // 20)  # Show ~20 swath indicators
    for i in range(0, len(track_coords), step):
        lat, lon = track_coords[i]

        # Approximate swath as perpendicular rectangle
        # This is rough - real swath depends on heading
        # For visualization, just show a circle
        folium.Circle(
            location=[lat, lon],
            radius=swath_width_m / 2,  # Half swath on each side
            color="yellow",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.1,
            weight=1,
            tooltip=f"Swath ~{swath_width_m:.0f}m",
        ).add_to(swath_fg)

    swath_fg.add_to(m)

    # 5. Statistics overlay
    stats_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 60px; width: 300px; 
                background-color: white; border: 2px solid grey; 
                z-index: 9999; font-size: 12px; padding: 10px;
                border-radius: 5px;">
    <h4 style="margin-top:0;">Coverage Diagnostic</h4>
    <b>Full Mission:</b><br>
    • Total points: {len(full_mission_df):,}<br>
    • Lat range: {full_mission_df['latitude'].min():.6f} to {full_mission_df['latitude'].max():.6f}<br>
    • Lon range: {full_mission_df['longitude'].min():.6f} to {full_mission_df['longitude'].max():.6f}<br>
    <br>
    <b>UHI HSI Subset:</b><br>
    • Frames: {len(hsi_track_df):,}<br>
    • Lat: {hsi_track_df['latitude'].min():.6f} to {hsi_track_df['latitude'].max():.6f}<br>
    • Lon: {hsi_track_df['longitude'].min():.6f} to {hsi_track_df['longitude'].max():.6f}<br>
    • Depth: {hsi_track_df['depth'].min():.1f} to {hsi_track_df['depth'].max():.1f} m<br>
    <br>
    <b>MBES Coverage:</b><br>
    • Lat: {mbes_bbox[1]:.6f} to {mbes_bbox[3]:.6f}<br>
    • Lon: {mbes_bbox[0]:.6f} to {mbes_bbox[2]:.6f}<br>
    <br>
    <b>UHI Swath:</b><br>
    • Altitude: {effective_altitude:.1f} m<br>
    • Est. width: {swath_width_m:.1f} m<br>
    <br>
    <b>Overlap Check:</b><br>
    • UHI in MBES: <span style="color:{'green' if (hsi_track_df['latitude'].min() >= mbes_bbox[1] and hsi_track_df['latitude'].max() <= mbes_bbox[3] and hsi_track_df['longitude'].min() >= mbes_bbox[0] and hsi_track_df['longitude'].max() <= mbes_bbox[2]) else 'red'};">
    {'YES ✓' if (hsi_track_df['latitude'].min() >= mbes_bbox[1] and hsi_track_df['latitude'].max() <= mbes_bbox[3] and hsi_track_df['longitude'].min() >= mbes_bbox[0] and hsi_track_df['longitude'].max() <= mbes_bbox[2]) else 'NO ✗'}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    m.save(output_html)
    print(f"\n✅ Diagnostic map saved: {output_html}")


def main():
    print("=" * 80)
    print("Mission Coverage Visualization")
    print("=" * 80)

    # Load full mission trajectory
    print("\n" + "=" * 80)
    print("Loading full mission trajectory from merged CSV...")
    print("=" * 80)
    full_mission_df = load_full_mission_track(config.NAV_CSV)

    # Find H5 files
    h5_folder = Path(config.H5_FOLDER)
    h5_files = sorted(h5_folder.glob("*.h5"))

    if not h5_files:
        print(f"❌ No H5 files found in {h5_folder}")
        return

    print(f"\nFound {len(h5_files)} H5 files:")
    for h5 in h5_files:
        print(f"  • {h5.name}")

    # Load HSI track (subset)
    print("\n" + "=" * 80)
    print("Loading UHI HSI track positions (subset)...")
    print("=" * 80)
    hsi_track_df = load_hsi_track(h5_files, config.NAV_CSV)

    # Load MBES bounds
    print("\n" + "=" * 80)
    print("Loading MBES coverage...")
    print("=" * 80)
    mbes_corners, mbes_bbox = load_mbes_bounds(config.MBES_GEOTIFF)

    # Create map
    print("\n" + "=" * 80)
    print("Creating coverage visualization map...")
    print("=" * 80)
    output_html = "hsi_mbes_diagnostic.html"
    create_diagnostic_map(
        full_mission_df, hsi_track_df, mbes_corners, mbes_bbox, output_html
    )

    # Print analysis
    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    # Check if HSI track is within MBES bounds
    hsi_in_mbes_lat = (
        hsi_track_df["latitude"].min() >= mbes_bbox[1]
        and hsi_track_df["latitude"].max() <= mbes_bbox[3]
    )
    hsi_in_mbes_lon = (
        hsi_track_df["longitude"].min() >= mbes_bbox[0]
        and hsi_track_df["longitude"].max() <= mbes_bbox[2]
    )

    if hsi_in_mbes_lat and hsi_in_mbes_lon:
        print("✓ HSI track IS within MBES coverage bounds")
        print("  → Ray tracing SHOULD work")
        print("  → Problem likely: coordinate transform or rotation matrix")
    else:
        print("✗ HSI track is OUTSIDE MBES coverage bounds")
        if not hsi_in_mbes_lat:
            print(f"  → Latitude mismatch:")
            print(
                f"    HSI: {hsi_track_df['latitude'].min():.6f} to {hsi_track_df['latitude'].max():.6f}"
            )
            print(f"    MBES: {mbes_bbox[1]:.6f} to {mbes_bbox[3]:.6f}")
        if not hsi_in_mbes_lon:
            print(f"  → Longitude mismatch:")
            print(
                f"    HSI: {hsi_track_df['longitude'].min():.6f} to {hsi_track_df['longitude'].max():.6f}"
            )
            print(f"    MBES: {mbes_bbox[0]:.6f} to {mbes_bbox[2]:.6f}")

    print("\n" + "=" * 80)
    print(f"Open '{output_html}' in your browser to visualize the issue!")
    print("=" * 80)


if __name__ == "__main__":
    print("bob")

    main()
