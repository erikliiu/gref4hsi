# GEOREFERENCING PIPELINE - COMPLETE BREAKDOWN

This document explains **exactly** how each spatial pixel of the UHI gets georeferenced to the seafloor.

---

## STEP 1: CAMERA FRAME DEFINITION

### What is the camera frame?
- **Origin**: At the UHI sensor
- **Orientation**: We assume it's ALIGNED with body frame (identity rotation)
  - X-axis: Forward (along vehicle motion)
  - Y-axis: Starboard/Right (perpendicular to motion)
  - Z-axis: Down (toward seafloor)

### How do we know where each pixel is looking?

This happens in `build_ray_directions()` in `utils.py`:

```python
# For each of the 968 pixels (slits) across the line:
pixel_coords = np.arange(num_pixels)  # [0, 1, 2, ..., 967]

# Convert pixel index to angle using CAMERA CALIBRATION
x_norm = (pixel_coords - cx) / f
# Where:
#   f = 930.63   (focal length in pixels)
#   cx = 484.75  (principal point - center of sensor)

# Apply lens distortion correction
r2 = x_norm**2
distortion_factor = 1 + k1*r2 + k2*r2^2 + k3*r2^3
x_distorted = x_norm * distortion_factor

# Build 3D ray direction for each pixel:
rays[:, 0] = 0              # X: No forward component (push-broom scans perpendicular)
rays[:, 1] = -x_distorted   # Y: Across-track angle (starboard → port)
rays[:, 2] = 1              # Z: Downward (looking at seafloor)
```

**YES, we have FOV information!** It's encoded in the focal length `f` and sensor width `w`:
- FOV ≈ 2 × arctan(w / (2×f)) ≈ 2 × arctan(968 / (2×930.63)) ≈ **55 degrees total**

**Key point**: The camera calibration XML tells us EXACTLY what angle each pixel is looking at!

---

## STEP 2: CAMERA TO BODY FRAME

You're right - the camera is **2.5m forward** of the body frame origin (where IMU is).

This happens in `apply_sensor_transform()` in `utils.py`:

```python
ROTATION_HSI_TO_BODY = np.eye(3)         # Identity - aligned orientation
TRANSLATION_BODY_TO_HSI = [2.5, 0, 0]   # 2.5m forward in X

# For each frame:
R_body_to_world = orientations[i]  # From IMU (roll, pitch, yaw)

# Camera position in world = body position + rotated translation
translation_world = R_body_to_world @ TRANSLATION_BODY_TO_HSI
sensor_positions_ecef[i] = positions_ecef[i] + translation_world

# Camera orientation in world = body orientation (since identity rotation)
sensor_orientations[i] = R_body_to_world @ ROTATION_HSI_TO_BODY
                       = R_body_to_world  # (since identity)
```

**Result**: We now know where the camera is in world coordinates (ECEF), and its orientation.

---

## STEP 3: BODY TO WORLD FRAME (6DOF from IMU)

This happens in `interpolate_navigation()` and `euler_to_rotation_matrix()`:

```python
# For each HSI frame timestamp:
# 1. Interpolate navigation to exact frame time (with -508 sec offset)
interp_nav = interpolate_navigation(nav_data, hsi_timestamps, time_offset=-508)

# 2. Convert lat/lon/depth to ECEF (X,Y,Z in Earth-Centered Earth-Fixed)
positions_ecef = geographic_to_ecef(longitude, latitude, -depth)

# 3. Convert roll/pitch/yaw to rotation matrix
orientations = euler_to_rotation_matrix(roll, pitch, yaw)
```

**Result**: For each frame, we have:
- **Position**: (X, Y, Z) in ECEF coordinates
- **Orientation**: 3×3 rotation matrix describing which way the vehicle is pointing

---

## STEP 4: RAY DIRECTIONS IN WORLD FRAME

This is the KEY step where we combine everything:

```python
# For each frame i:
R_camera_to_world = hsi_orientations[i]  # 3×3 rotation matrix

# For each pixel j in that frame:
ray_direction_camera = ray_directions_camera[j]  # [0, -x_distorted, 1]

# Transform to world frame:
ray_direction_world = R_camera_to_world @ ray_direction_camera

# Ray origin (same for all pixels in frame):
ray_origin = hsi_positions_mbes[i]  # Camera position in UTM
```

**Result**: Each pixel now has:
- A 3D starting point (camera position in UTM)
- A 3D direction vector (where that pixel is looking in world coordinates)

---

## STEP 5: RAY TRACING TO SEAFLOOR (MBES MESH)

This happens in `raytrace_hsi_to_mbes()` in `georeference_mbes.py`:

```python
# For each ray (total: 968 pixels × 1333 frames = 1,290,344 rays):
intersection_point = ray_trace(origin, direction, mbes_mesh)

# The ray trace solver finds WHERE the ray intersects the 3D MBES mesh
# Returns: (X, Y, Z) in UTM coordinates where ray hit seafloor
```

**THIS IS WHERE EACH PIXEL HITS THE SEAFLOOR!**

The MBES mesh is a 3D surface model of the seafloor. The ray tracer calculates the geometric intersection of the ray (camera pixel look direction) with this surface.

**If ray tracing fails (97% miss rate), it means**:
- Rays are pointing in wrong direction (coordinate system error)
- Rays start from wrong position (transform error)
- Ray directions not properly rotated to world frame

---

## STEP 6: SAVE GEOREFERENCED POINTS

```python
# Convert intersection points from UTM back to ECEF
intersections_ecef = utm_to_ecef(intersection_points_utm)

# Save to H5 file
h5f['processed/georef/points_ecef_crs'] = intersections_ecef
```

**Result**: Each pixel now has a 3D point on the seafloor in ECEF coordinates!

---

## SUMMARY: PIXEL → SEAFLOOR MAPPING

**For pixel #j in frame #i:**

1. **Camera ray**: `[0, -x_distorted[j], 1]` ← From calibration (FOV encoded)
2. **Camera position**: IMU position + 2.5m forward ← Known offset
3. **Camera orientation**: Same as body (roll, pitch, yaw from IMU) ← Identity rotation
4. **World ray direction**: `R_camera_to_world @ camera_ray` ← Rotation to world
5. **Ray origin**: Camera position in UTM ← From step 2
6. **Seafloor hit**: Ray trace intersection with MBES mesh ← Geometric calculation
7. **Georeferenced point**: (X, Y, Z) in ECEF ← Final output

---

## YOUR QUESTIONS ANSWERED:

**Q: "Do you have information about the FOV of the UHI?"**
**A: YES!** FOV = 2×arctan(968/(2×930.63)) ≈ 55°. Encoded in focal length `f=930.63` and width `w=968`.

**Q: "Where does each pixel hit the seafloor?"**
**A: In Step 5** - ray tracing calculates geometric intersection of pixel look direction with MBES 3D mesh.

**Q: "Is the code actually doing this?"**
**A: YES** - see `main.py` lines 50-180. The pipeline is:
- Load timestamps
- Interpolate navigation (6DOF)
- Convert to ECEF
- Apply sensor offset (2.5m forward)
- Build ray directions (with FOV from calibration)
- Transform rays to world frame
- Ray trace to MBES mesh
- Save georeferenced points

---

## POTENTIAL BUGS TO CHECK:

1. ✅ **Ray directions** - Fixed: Now `[0, -x_distorted, 1]` in body frame
2. ✅ **Rotation matrix** - Fixed: Now identity (camera aligned with body)
3. ⚠️ **Coordinate system conventions** - Need to verify:
   - Is IMU roll/pitch/yaw in NED or ENU?
   - Is MBES mesh Z-up or Z-down?
   - Are ray directions properly normalized?
4. ⚠️ **Ray tracing mesh orientation** - Check if MBES mesh normals point up or down

---

## NEXT DEBUGGING STEPS:

1. **Print sample ray directions** after world transform - should point downward
2. **Visualize rays in 3D** - plot camera position + ray vectors
3. **Check mesh normals** - verify MBES surface orientation
4. **Test single frame** - debug one frame before processing all
