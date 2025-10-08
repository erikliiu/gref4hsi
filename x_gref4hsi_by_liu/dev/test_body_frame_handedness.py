"""
Simple test: If body Y is truly "starboard" (right), then with the current rotation,
camera should be offset to the RIGHT (positive cross-track) when we add a Y offset.

If the current code has Body Y mapped incorrectly, adding +Y offset will move camera LEFT.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import utils

print("=" * 80)
print("SIMPLE BODY FRAME HANDEDNESS TEST")
print("=" * 80)

# Scenario: Vehicle heading SOUTH (yaw=180°), level flight
roll = 0.0
pitch = 0.0
yaw = 180.0  # Heading South

print(f"\nVehicle heading: {yaw}° (South)")
print(f"Roll: {roll}°, Pitch: {pitch}°")

# Get rotation matrix
R_body_to_world = utils.euler_to_rotation_matrix_nav(
    np.array([roll]),
    np.array([pitch]),
    np.array([yaw]),
    yaw_convention=config.YAW_CONVENTION,
)[0]

print("\nR_body_to_world:")
print(R_body_to_world)

# Test: If body +Y is truly starboard (right when heading south),
# then body +Y should map to EAST (+X in ENU/UTM)
body_y = np.array([0, 1, 0])  # Unit vector along body Y axis
world_y = R_body_to_world @ body_y

print(f"\nBody +Y (starboard/right) in world: {world_y}")
print(f"  East component: {world_y[0]:.4f}")
print(f"  North component: {world_y[1]:.4f}")

if world_y[0] > 0.9:
    print("  ✓ CORRECT: Starboard maps to East when heading South")
elif world_y[0] < -0.9:
    print("  ✗ WRONG: Starboard maps to West when heading South")
    print("  This means the body frame Y-axis is flipped (port/starboard reversed)")
else:
    print(f"  ? UNCLEAR: East component is {world_y[0]:.4f}")

# Test body +X (forward)
body_x = np.array([1, 0, 0])
world_x = R_body_to_world @ body_x

print(f"\nBody +X (forward) in world: {world_x}")
print(f"  East component: {world_x[0]:.4f}")
print(f"  North component: {world_x[1]:.4f}")

if world_x[1] < -0.9:
    print("  ✓ CORRECT: Forward maps to South when heading South")
elif world_x[1] > 0.9:
    print("  ✗ WRONG: Forward maps to North when heading South")
else:
    print(f"  ? UNCLEAR: North component is {world_x[1]:.4f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if world_y[0] < -0.9 and world_x[1] < -0.9:
    print("\n⚠️  The body frame Y-axis is FLIPPED!")
    print("Current: Body +Y maps to port (left) instead of starboard (right)")
    print("\nFIX: In config.py, change ROTATION_HSI_TO_BODY first row from:")
    print("  [0.0, 1.0, 0.0]  # Cam X → Body +Y")
    print("to:")
    print("  [0.0, -1.0, 0.0]  # Cam X → Body -Y")
elif world_y[0] > 0.9 and world_x[1] < -0.9:
    print("\n✓ Body frame orientation is CORRECT")
else:
    print("\n? Results are unclear, check rotation matrix")
