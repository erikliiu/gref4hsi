"""
Debug: Check ray direction after all transforms
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config
import x_gref4hsi_by_liu.utils.utils as utils

# Test with a simple scenario
print("=" * 80)
print("RAY DIRECTION DEBUG")
print("=" * 80)

# 1. Build camera ray (center pixel, pointing "forward" in camera frame)
camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
rays_camera = utils.build_ray_directions(camera_calib, 3)  # Just 3 pixels

print("\n1. Ray directions in CAMERA frame:")
print(f"   Left pixel:   {rays_camera[0]}")
print(f"   Center pixel: {rays_camera[1]}")
print(f"   Right pixel:  {rays_camera[2]}")
print(f"   → Camera Z-axis points: {rays_camera[1]}")  # Should be ~[0, 0, 1]

# 2. Apply rotation from camera to body
R_camera_to_body = config.ROTATION_HSI_TO_BODY
ray_in_body = R_camera_to_body @ rays_camera[1]

print(f"\n2. After rotation (camera → body):")
print(f"   Rotation matrix:\n{R_camera_to_body}")
print(f"   Center ray in BODY frame: {ray_in_body}")

# Check what direction this points
if ray_in_body[2] > 0.5:
    print(f"   ✓ Points DOWN (Z > 0)")
elif ray_in_body[2] < -0.5:
    print(f"   ✗ Points UP (Z < 0) - WRONG FOR UNDERWATER!")
else:
    print(f"   ? Points sideways (Z ≈ 0)")

# 3. Simulate vehicle orientation (level flight, no roll/pitch/yaw)
roll, pitch, yaw = 0, 0, 0
R_body_to_world = utils.euler_to_rotation_matrix(
    np.array([roll]), np.array([pitch]), np.array([yaw])
)[0]

print(f"\n3. Vehicle orientation (roll=pitch=yaw=0):")
print(f"   R_body_to_world:\n{R_body_to_world}")

# 4. Full transform: camera → body → world
R_camera_to_world = R_body_to_world @ R_camera_to_body
ray_in_world = R_camera_to_world @ rays_camera[1]

print(f"\n4. Final ray in WORLD frame (NED):")
print(f"   Ray direction: {ray_in_world}")
print(f"   Components:")
print(f"     North (X): {ray_in_world[0]:+.3f}")
print(f"     East  (Y): {ray_in_world[1]:+.3f}")
print(f"     Down  (Z): {ray_in_world[2]:+.3f}")

if ray_in_world[2] > 0.5:
    print(f"   ✓ Points DOWNWARD - correct for underwater camera!")
elif ray_in_world[2] < -0.5:
    print(f"   ✗ Points UPWARD - rays will miss seafloor!")
else:
    print(f"   ? Points horizontally - rays parallel to surface!")

# 5. Test with non-zero vehicle orientation
print(f"\n5. Test with vehicle pitched down 10°:")
roll, pitch, yaw = 0, 10, 0  # 10° nose down
R_body_to_world_pitched = utils.euler_to_rotation_matrix(
    np.array([roll]), np.array([pitch]), np.array([yaw])
)[0]
R_camera_to_world_pitched = R_body_to_world_pitched @ R_camera_to_body
ray_pitched = R_camera_to_world_pitched @ rays_camera[1]

print(f"   Ray direction: {ray_pitched}")
print(f"   Down component: {ray_pitched[2]:+.3f}")
if ray_pitched[2] > 0:
    print(f"   ✓ Still points downward")
else:
    print(f"   ✗ Now points upward - rotation matrix WRONG!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Check what the rotation matrix actually does
print("\nWhat does ROTATION_HSI_TO_BODY do?")
test_vectors = {
    "X (right in camera)": np.array([1, 0, 0]),
    "Y (up in camera)": np.array([0, 1, 0]),
    "Z (forward in camera)": np.array([0, 0, 1]),
}

for name, vec in test_vectors.items():
    transformed = R_camera_to_body @ vec
    print(
        f"  {name:25s} → Body: [{transformed[0]:+.0f}, {transformed[1]:+.0f}, {transformed[2]:+.0f}]"
    )

print("\nFor underwater downward-looking camera:")
print("  Expected: Camera Z should map to Body +Z (downward)")
print(f"  Actual:   Camera Z maps to Body: {R_camera_to_body[2, :]}")
if R_camera_to_body[2, 2] > 0.9:
    print("  ✓ CORRECT - Camera forward = Body down")
elif R_camera_to_body[2, 2] < -0.9:
    print("  ✗ INVERTED - Camera forward = Body up (WRONG!)")
else:
    print(f"  ? SIDEWAYS - Camera forward = Body horizontal")

print("\n" + "=" * 80)
