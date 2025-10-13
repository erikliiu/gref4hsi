import sys
import numpy as np
import rasterio

sys.path.insert(0, "..")
sys.path.insert(0, "../../gref4hsi/final_act")
from utils.gref_pipeline import georef
from utils.gref_pipeline.georef import _ecef_to_ned_arrays
from utils.gref_pipeline import utils as gref_utils
import config

print("=" * 70)
print("CHECKING: Which coordinate system should we use?")
print("=" * 70)

# 1. Check MBES GeoTIFF native coordinates
print("\n📍 MBES GeoTIFF Native Coordinates:")
with rasterio.open(config.MBES_GEOTIFF) as src:
    bounds = src.bounds
    epsg = src.crs.to_epsg() if src.crs else None
    print(f"   EPSG: {epsg}")
    print(f"   Bounds: left={bounds.left:.2f}, right={bounds.right:.2f}")
    print(f"           bottom={bounds.bottom:.2f}, top={bounds.top:.2f}")
    print(f"   Range X: {bounds.right - bounds.left:.2f} m")
    print(f"   Range Y: {bounds.top - bounds.bottom:.2f} m")

    # Get center point
    center_x = (bounds.left + bounds.right) / 2
    center_y = (bounds.bottom + bounds.top) / 2
    print(f"   Center: ({center_x:.2f}, {center_y:.2f})")

# 2. Check UHI coordinates in ECEF and NED
print("\n📍 UHI Footprint Coordinates:")
transect = georef.load_transect(config.OUTPUT_FOLDER)
cube = transect.select_files(["rad_uhi_20241029_115057_4", "rad_uhi_20241029_115057_5"])
track_start = 3039
track_end = 4029

X_ecef = cube.X_ecef[track_start:track_end, :]
Y_ecef = cube.Y_ecef[track_start:track_end, :]
Z_ecef = cube.Z_ecef[track_start:track_end, :]

# Convert to NED
lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)

valid = np.isfinite(E) & np.isfinite(N)
print(f"   NED East range: [{np.nanmin(E[valid]):.2f}, {np.nanmax(E[valid]):.2f}]")
print(f"   NED North range: [{np.nanmin(N[valid]):.2f}, {np.nanmax(N[valid]):.2f}]")
print(f"   NED Center: ({np.nanmean(E[valid]):.2f}, {np.nanmean(N[valid]):.2f})")

# 3. Convert NED origin to UTM for comparison
from pyproj import Transformer

transformer = Transformer.from_crs(
    f"EPSG:{config.EPSG_GEOGRAPHIC}", f"EPSG:{epsg}", always_xy=True
)
origin_utm_x, origin_utm_y = transformer.transform(lon0, lat0)
print(f"\n📍 NED Origin in UTM:")
print(f"   Origin UTM: ({origin_utm_x:.2f}, {origin_utm_y:.2f})")

# 4. Check if UHI in UTM would overlap with MBES
print("\n🔍 Checking UHI position in UTM coordinates:")
# Convert a few UHI points from ECEF to UTM
uhi_utm_x, uhi_utm_y, _ = gref_utils.ecef_to_utm(
    X_ecef[valid], Y_ecef[valid], Z_ecef[valid], epsg_utm=epsg, epsg_ecef=4978
)
print(f"   UHI in UTM X range: [{uhi_utm_x.min():.2f}, {uhi_utm_x.max():.2f}]")
print(f"   UHI in UTM Y range: [{uhi_utm_y.min():.2f}, {uhi_utm_y.max():.2f}]")
print(f"   UHI in UTM Center: ({uhi_utm_x.mean():.2f}, {uhi_utm_y.mean():.2f})")

# 5. Compare
print("\n" + "=" * 70)
print("VERDICT:")
print("=" * 70)
mbes_center_x = (bounds.left + bounds.right) / 2
mbes_center_y = (bounds.bottom + bounds.top) / 2
uhi_utm_center_x = uhi_utm_x.mean()
uhi_utm_center_y = uhi_utm_y.mean()

distance = np.sqrt(
    (mbes_center_x - uhi_utm_center_x) ** 2 + (mbes_center_y - uhi_utm_center_y) ** 2
)
print(f"Distance between MBES center and UHI center in UTM: {distance:.2f} m")

if distance < 1000:
    print("✅ UHI and MBES are close in UTM coordinates!")
    print("   → The NOTEBOOK approach (converting MBES to NED) is CORRECT")
    print("   → This ensures both datasets are in the same local coordinate frame")
else:
    print("⚠️  UHI and MBES are far apart in UTM coordinates")
    print("   → Need to check coordinate transformations")

print("\n💡 EXPLANATION:")
print("   - MBES GeoTIFF is stored in UTM coordinates (native)")
print("   - UHI data is in ECEF, then converted to NED")
print("   - To overlay them correctly, we must:")
print("     OPTION 1: Convert MBES from UTM → NED (what notebook does)")
print("     OPTION 2: Convert UHI from NED → UTM (what dev14 tried)")
print("   - Both should work IF the coordinate transformation is correct")
print("   - The notebook's approach (OPTION 1) is cleaner because:")
print("     • It uses a local coordinate frame (NED) centered at the survey origin")
print("     • Smaller numbers, easier to work with")
print("     • Less numerical precision issues")
