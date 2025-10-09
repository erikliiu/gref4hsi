"""
Debug script to diagnose lateral offset between simulation and main.py georeferencing
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import utils

# Test parameters
print("=" * 80)
print("DEBUGGING LATERAL OFFSET ISSUE")
print("=" * 80)

# Test case: vehicle at origin, heading North (yaw=0), level (roll=pitch=0)
print("\n### TEST 1: Vehicle at origin, heading North, level")
print("-" * 80)

roll_deg = 0.0
pitch_deg = 0.0
yaw_deg = 0.0  # Heading North (compass convention)

print(f"Input: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}° (compass heading)")

# Compute rotation using utils function (used in main.py)
R_body_to_world = utils.euler_to_rotation_matrix_nav(
    np.array([roll_deg]),
    np.array([pitch_deg]),
    np.array([yaw_deg]),
    yaw_convention=config.YAW_CONVENTION,
)[0]

print("\nR_body_to_world (from euler_to_rotation_matrix_nav):")
print(R_body_to_world)

# Body frame unit vectors in world frame
body_x_in_world = R_body_to_world @ np.array([1, 0, 0])
body_y_in_world = R_body_to_world @ np.array([0, 1, 0])
body_z_in_world = R_body_to_world @ np.array([0, 0, 1])

print(f"\nBody X (forward) in world: {body_x_in_world}")
print(f"Body Y (starboard) in world: {body_y_in_world}")
print(f"Body Z (up) in world: {body_z_in_world}")

# Now check camera frame
print("\n" + "=" * 80)
print("CAMERA FRAME TRANSFORMATION")
print("=" * 80)

print("\nROTATION_HSI_TO_BODY matrix:")
print(config.ROTATION_HSI_TO_BODY)

# Camera frame unit vectors in body frame
cam_x_in_body = config.ROTATION_HSI_TO_BODY @ np.array([1, 0, 0])
cam_y_in_body = config.ROTATION_HSI_TO_BODY @ np.array([0, 1, 0])
cam_z_in_body = config.ROTATION_HSI_TO_BODY @ np.array([0, 0, 1])

print(f"\nCam X (across-track) in body: {cam_x_in_body}")
print(f"Cam Y (along-track) in body: {cam_y_in_body}")
print(f"Cam Z (down/nadir) in body: {cam_z_in_body}")

# Now in world frame
R_world_from_cam = R_body_to_world @ config.ROTATION_HSI_TO_BODY
cam_x_in_world = R_world_from_cam @ np.array([1, 0, 0])
cam_y_in_world = R_world_from_cam @ np.array([0, 1, 0])
cam_z_in_world = R_world_from_cam @ np.array([0, 0, 1])

print(f"\nCam X (across-track) in world: {cam_x_in_world}")
print(f"Cam Y (along-track) in world: {cam_y_in_world}")
print(f"Cam Z (down/nadir) in world: {cam_z_in_world}")

# Build a sample ray (center pixel, should point straight down)
print("\n" + "=" * 80)
print("RAY DIRECTION TEST")
print("=" * 80)

# Load camera calibration
camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
n_slits = int(camera_calib["w"])
print(f"\nCamera width: {n_slits} slits")
print(f"Focal length: {camera_calib['f']}")
print(f"Principal point: {camera_calib['cx']}")

# Build ray directions
ray_dirs_cam = utils.build_ray_directions(camera_calib, n_slits)
print(f"\nBuilt {len(ray_dirs_cam)} ray directions")

# Check center ray (should point mostly down)
center_idx = n_slits // 2
center_ray_cam = ray_dirs_cam[center_idx]
print(f"\nCenter ray (pixel {center_idx}) in camera frame:")
print(f"  {center_ray_cam}")
print(f"  Normalized: {center_ray_cam / np.linalg.norm(center_ray_cam)}")

# Transform to world
center_ray_world = R_world_from_cam @ center_ray_cam
center_ray_world_norm = center_ray_world / np.linalg.norm(center_ray_world)
print(f"\nCenter ray in world frame:")
print(f"  {center_ray_world}")
print(f"  Normalized: {center_ray_world_norm}")

# Check if pointing down (negative Z in world if ENU, or check against up vector)
print(f"\nIs ray pointing down? (Z component should be negative in ENU)")
print(f"  World Z component: {center_ray_world_norm[2]:.4f}")

# Check edge rays (left and right)
print("\n" + "=" * 80)
print("EDGE RAY TEST (checking for lateral offset)")
print("=" * 80)

left_ray_cam = ray_dirs_cam[0]
right_ray_cam = ray_dirs_cam[-1]

left_ray_world = R_world_from_cam @ left_ray_cam
right_ray_world = R_world_from_cam @ right_ray_cam

left_ray_world_norm = left_ray_world / np.linalg.norm(left_ray_world)
right_ray_world_norm = right_ray_world / np.linalg.norm(right_ray_world)

print(f"\nLeft edge ray (pixel 0) in camera frame:")
print(f"  {left_ray_cam}")
print(f"\nLeft edge ray in world frame (normalized):")
print(f"  {left_ray_world_norm}")
print(f"  East component: {left_ray_world_norm[0]:.4f}")
print(f"  North component: {left_ray_world_norm[1]:.4f}")
print(f"  Up component: {left_ray_world_norm[2]:.4f}")

print(f"\nRight edge ray (pixel {n_slits-1}) in camera frame:")
print(f"  {right_ray_cam}")
print(f"\nRight edge ray in world frame (normalized):")
print(f"  {right_ray_world_norm}")
print(f"  East component: {right_ray_world_norm[0]:.4f}")
print(f"  North component: {right_ray_world_norm[1]:.4f}")
print(f"  Up component: {right_ray_world_norm[2]:.4f}")

# Check if the swath is perpendicular to flight direction
print("\n" + "=" * 80)
print("SWATH ORIENTATION CHECK")
print("=" * 80)

# Across-track vector (left to right)
across_track_world = right_ray_world_norm - left_ray_world_norm
across_track_world /= np.linalg.norm(across_track_world)

# Flight direction is body X in world
flight_dir = body_x_in_world

# They should be perpendicular (dot product ~0)
dot_product = np.dot(across_track_world, flight_dir)
print(f"\nAcross-track direction: {across_track_world}")
print(f"Flight direction (body X): {flight_dir}")
print(f"Dot product: {dot_product:.6f} (should be ~0 for perpendicular)")

if abs(dot_product) > 0.1:
    print("\n⚠️  WARNING: Swath is NOT perpendicular to flight direction!")
    print("   This will cause lateral offset!")
else:
    print("\n✓ Swath is perpendicular to flight direction")

# Test with different yaw angle
print("\n" + "=" * 80)
print("TEST 2: Vehicle heading East (yaw=90°)")
print("=" * 80)

yaw_deg = 90.0
R_body_to_world_east = utils.euler_to_rotation_matrix_nav(
    np.array([0.0]),
    np.array([0.0]),
    np.array([yaw_deg]),
    yaw_convention=config.YAW_CONVENTION,
)[0]

body_x_in_world_east = R_body_to_world_east @ np.array([1, 0, 0])
print(f"\nBody X (forward) when heading East: {body_x_in_world_east}")
print(f"Expected: [1, 0, 0] (East)")

R_world_from_cam_east = R_body_to_world_east @ config.ROTATION_HSI_TO_BODY
cam_x_in_world_east = R_world_from_cam_east @ np.array([1, 0, 0])
print(f"\nCam X (across-track) when heading East: {cam_x_in_world_east}")
print(f"Expected: [0, ±1, 0] (North or South, perpendicular to East)")
