"""
Quick script to check DEM and trajectory alignment
Run this to diagnose intersection failures
"""

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

# Paths - UPDATE THESE
DATA_DIR = r"E:\mjosa_new_oct_2025\use_gref4hsi\057"
CSV_PATH = r"E:\mjosa_new\navigation_data\nav_data_merged.csv"
DEM_PATH = f"{DATA_DIR}/Input/GIS/dummy_dem.tif"

print("=" * 60)
print("DEM and Trajectory Alignment Check")
print("=" * 60)

# 1. Check DEM
print("\n1. DEM Information:")
try:
    with rasterio.open(DEM_PATH) as dem:
        print(f"   CRS: {dem.crs}")
        print(f"   Bounds: {dem.bounds}")
        print(f"   Shape: {dem.shape}")
        print(f"   Resolution: {dem.res}")

        dem_data = dem.read(1)
        print(
            f"   Height range: {np.nanmin(dem_data):.2f} to {np.nanmax(dem_data):.2f} m"
        )
        print(f"   Valid pixels: {np.sum(~np.isnan(dem_data))} / {dem_data.size}")

        dem_bounds = dem.bounds
except Exception as e:
    print(f"   ERROR reading DEM: {e}")
    dem_bounds = None

# 2. Check Navigation CSV
print("\n2. Navigation CSV Information:")
try:
    nav_data = pd.read_csv(CSV_PATH)
    print(f"   Total records: {len(nav_data)}")
    print(
        f"   Time range: {nav_data['timestamp [unix epoch s]'].min():.1f} to {nav_data['timestamp [unix epoch s]'].max():.1f}"
    )

    # Get coordinate columns
    lat_col = "latitude [deg]"
    lon_col = "longitude [deg]"

    print(
        f"   Lat range: {nav_data[lat_col].min():.6f} to {nav_data[lat_col].max():.6f}"
    )
    print(
        f"   Lon range: {nav_data[lon_col].min():.6f} to {nav_data[lon_col].max():.6f}"
    )

    # Transform to UTM for comparison with DEM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    nav_x, nav_y = transformer.transform(
        nav_data[lon_col].values, nav_data[lat_col].values
    )

    print(f"\n   In UTM 32N:")
    print(f"   X range: {nav_x.min():.2f} to {nav_x.max():.2f} m")
    print(f"   Y range: {nav_y.min():.2f} to {nav_y.max():.2f} m")

except Exception as e:
    print(f"   ERROR reading CSV: {e}")
    nav_x = nav_y = None

# 3. Check Overlap
print("\n3. Spatial Overlap Check:")
if dem_bounds is not None and nav_x is not None:
    print(f"   DEM X: {dem_bounds.left:.2f} to {dem_bounds.right:.2f}")
    print(f"   Nav X: {nav_x.min():.2f} to {nav_x.max():.2f}")

    print(f"   DEM Y: {dem_bounds.bottom:.2f} to {dem_bounds.top:.2f}")
    print(f"   Nav Y: {nav_y.min():.2f} to {nav_y.max():.2f}")

    # Check if navigation is within DEM bounds
    nav_in_dem_x = (nav_x.min() >= dem_bounds.left) and (
        nav_x.max() <= dem_bounds.right
    )
    nav_in_dem_y = (nav_y.min() >= dem_bounds.bottom) and (
        nav_y.max() <= dem_bounds.top
    )

    if nav_in_dem_x and nav_in_dem_y:
        print("\n   ✅ Navigation trajectory is WITHIN DEM bounds")
    else:
        print("\n   ❌ Navigation trajectory OUTSIDE DEM bounds!")
        print(f"      X overlap: {nav_in_dem_x}")
        print(f"      Y overlap: {nav_in_dem_y}")

        # Calculate offset
        offset_x = (nav_x.min() + nav_x.max()) / 2 - (
            dem_bounds.left + dem_bounds.right
        ) / 2
        offset_y = (nav_y.min() + nav_y.max()) / 2 - (
            dem_bounds.bottom + dem_bounds.top
        ) / 2
        print(f"\n      Offset: X={offset_x:.2f}m, Y={offset_y:.2f}m")
        print(f"      Distance: {np.sqrt(offset_x**2 + offset_y**2):.2f}m")

print("\n" + "=" * 60)
