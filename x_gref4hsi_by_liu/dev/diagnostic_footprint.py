import sys
import numpy as np

sys.path.insert(0, "..")
from utils import georef
from utils.georef import _ecef_to_ned_arrays
import config

print("=" * 70)
print("DIAGNOSTIC: Comparing dev14_cont.py vs Notebook")
print("=" * 70)

# Load and select files (same as both scripts)
transect = georef.load_transect(config.OUTPUT_FOLDER)
cube = transect.select_files(["rad_uhi_20241029_115057_4", "rad_uhi_20241029_115057_5"])

track_start = 3039
track_end = 4029

# Extract coordinates (same as both scripts)
X_ecef = cube.X_ecef[track_start:track_end, :]
Y_ecef = cube.Y_ecef[track_start:track_end, :]
Z_ecef = cube.Z_ecef[track_start:track_end, :]
R = cube.R[track_start:track_end, :]
G = cube.G[track_start:track_end, :]
B = cube.B[track_start:track_end, :]

print(f"\n📊 Data shapes:")
print(f"   X_ecef: {X_ecef.shape}")
print(f"   R: {R.shape}")

# Build valid mask (same as both scripts)
coords_valid = np.isfinite(X_ecef) & np.isfinite(Y_ecef) & np.isfinite(Z_ecef)
rgb_valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
full_valid = coords_valid & rgb_valid

print(
    f"\n✅ Valid pixels: {full_valid.sum():,} / {full_valid.size:,} ({100*full_valid.mean():.1f}%)"
)

# Convert to NED (same as both scripts)
lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)

print(f"\n📍 NED coordinates:")
print(f"   N shape: {N.shape}")
print(f"   E shape: {E.shape}")
print(f"   N range: [{np.nanmin(N):.2f}, {np.nanmax(N):.2f}]")
print(f"   E range: [{np.nanmin(E):.2f}, {np.nanmax(E):.2f}]")

# Check valid region bounds
valid_N = N[full_valid]
valid_E = E[full_valid]
print(f"\n🎯 Valid region bounds:")
print(f"   N: [{valid_N.min():.2f}, {valid_N.max():.2f}]")
print(f"   E: [{valid_E.min():.2f}, {valid_E.max():.2f}]")

print(f"\n" + "=" * 70)
print("If the footprints look different, check:")
print("1. Are the N, E coordinates the same?")
print("2. Is the valid mask the same?")
print("3. Is the MBES coordinate system the same (NED vs UTM)?")
print("=" * 70)
