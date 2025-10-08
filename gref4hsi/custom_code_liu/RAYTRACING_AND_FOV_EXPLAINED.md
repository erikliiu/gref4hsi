# How Raytracing Works in test_eely.py

This document explains how the georeferencing pipeline determines the field of view (FOV) and performs raytracing to georeference hyperspectral images.

## Overview: The Complete Pipeline

```
1. Load H5 file → Extract FOV data
2. Build camera model → Create ray directions for each pixel
3. Load navigation → Get camera position & orientation for each frame
4. Ray geometry → Transform rays from camera to world coordinates
5. Ray-mesh intersection → Find where each ray hits the DEM
6. Compute angles → Calculate view angles and sun angles
```

---

## Step 1: How the FOV is Extracted from H5 Files

### Location in Code
**File:** `gref4hsi/utils/uhi_parsing_utils.py`, function `uhi_eely()`

### What Happens (Lines 1375-1403)

```python
# Read the .h5 data
hyp = HyperspectralLite(h5_filename=H5_FILE_PATH, h5_tree_dict=h5_dict_read)

# Extract FOV from the hyperspectral object
fov_arr = hyp.fov  # Array of view angles in degrees for each pixel column

# Creates an XML calibration file
set_camera_model(
    config=config,
    config_file_path=config_file_path,
    config_uhi=config_uhi,
    model_type="embedded",  # Use FOV array embedded in H5 file
    binning_spatial=binning_spatial,
    fov_arr=fov_arr,
)
```

### FOV Data Format
- **Source:** `processed/radiance/calibration/geometric/fieldOfView` in H5 file
- **Format:** 1D array of angles in degrees, one value per spatial pixel
- **Example:** `[-17.5, -17.3, ..., 0.0, ..., +17.3, +17.5]` (for ~1000 pixels)
- **Meaning:** Each value is the viewing angle of that pixel column relative to camera center

---

## Step 2: Converting FOV to Camera Model

### Location in Code
**File:** `gref4hsi/utils/specim_parsing_utils.py`, function `fov_2_param()`

### What Happens

The FOV angles are converted to a **pinhole camera model with distortion** using optimization:

```python
def fov_2_param(fov):
    """
    Converts field-of-view angles to camera intrinsic parameters
    Input: fov array in degrees (e.g., [-17.5, ..., +17.5])
    Output: Camera parameters (focal length, distortion coefficients, etc.)
    """
    
    # 1. Convert angles to normalized image coordinates
    theta_true = fov * np.pi / 180
    x_true = np.tan(theta_true)
    
    # 2. Optimize to find best-fit camera model
    # Variables to solve for:
    #   f   = focal length (pixels)
    #   k1  = 5th order radial distortion
    #   k2  = 3rd order radial distortion  
    #   k3  = 2nd order radial distortion
    #   c_x = principal point (pixel offset)
    
    # 3. Fit model using least-squares optimization
    res = least_squares(optimize_func, param_0, args=(x_true, n_pix))
    
    # 4. Return dictionary with camera parameters
    return {
        'f': f,          # Focal length
        'cx': c_x,       # Principal point
        'k1': k1,        # Distortion coefficients
        'k2': k2,
        'k3': k3,
        'width': n_pix,  # Image width in pixels
        'rx': 0,         # Rotation (set later)
        'ry': 0,
        'rz': 0,
        'tx': 0,         # Translation (set later)
        'ty': 0,
        'tz': 0
    }
```

### Camera Model Equation

For each pixel column `u` (1 to n_pix):

```
x_normalized = (u - c_x) / f + distortion_term

distortion_term = -(k1*r^5 + k2*r^3 + k3*r^2) / f
where r = (u - c_x) / 1000

ray_direction = [x_normalized, 0, 1]  # In camera frame (Z points down)
```

This defines the 3D ray direction for each pixel column.

---

## Step 3: Adding Camera Position & Orientation

### Location in Code
**File:** `gref4hsi/utils/uhi_parsing_utils.py`, function `set_camera_model()`

The camera model gets extended with **extrinsic parameters** (position & orientation):

```python
# From test_eely.py config:
rotation_matrix_hsi_to_body = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
translation_body_to_hsi = np.array([2.5, 0, 0])

# Convert rotation matrix to Euler angles
R_hsi_body = config_uhi.rotation_matrix_hsi_to_body
r_zyx = RotLib.from_matrix(R_hsi_body).as_euler("ZYX", degrees=False)

# Update camera parameters
param_dict["rz"] = r_zyx[0]  # Rotation around Z-axis
param_dict["ry"] = r_zyx[1]  # Rotation around Y-axis
param_dict["rx"] = r_zyx[2]  # Rotation around X-axis
param_dict["tx"] = translation_body_to_hsi[0]  # 2.5 m forward
param_dict["ty"] = translation_body_to_hsi[1]  # 0 m
param_dict["tz"] = translation_body_to_hsi[2]  # 0 m

# Write to XML file
CalibHSI(file_name_cal_xml=xml_cal_write_path, mode="w", param_dict=param_dict)
```

The XML file (e.g., `HSI_2b.xml`) now contains the complete camera calibration.

---

## Step 4: Building Ray Geometry

### Location in Code
**File:** `gref4hsi/scripts/georeference.py`, function `cal_file_to_rays()`

This function reads the XML calibration and generates **ray directions**:

```python
def cal_file_to_rays(filename_cal):
    # Load camera parameters from XML
    calHSI = CalibHSI(file_name_cal_xml=filename_cal)
    f = calHSI.f
    u_c = calHSI.cx
    k1, k2, k3 = calHSI.k1, calHSI.k2, calHSI.k3
    
    # Define pixel coordinates
    n_pix = calHSI.w
    u = np.arange(1, n_pix + 1)
    
    # Compute normalized image coordinates with distortion
    x_norm_lin = (u - u_c) / f
    x_norm_nonlin = -(k1*((u-u_c)/1000)**5 + k2*((u-u_c)/1000)**3 + k3*((u-u_c)/1000)**2) / f
    x_norm = x_norm_lin + x_norm_nonlin
    
    # Build 3D ray directions in camera frame
    p_dir = np.zeros((len(x_norm), 3))
    p_dir[:, 0] = x_norm  # X component (across-track)
    p_dir[:, 2] = 1       # Z component (down)
    # p_dir[:, 1] = 0     # Y component (along-track) - always zero for pushbroom
    
    # Load extrinsic parameters
    rot_hsi_ref_eul = np.array([rot_z, rot_y, rot_x])
    rot_hsi_ref_obj = RotLib.from_euler(seq="ZYX", angles=rot_hsi_ref_eul, degrees=False)
    translation_ref_hsi = np.array([trans_x, trans_y, trans_z])
    
    return {
        "translation_ref_hsi": translation_ref_hsi,
        "rot_hsi_ref_obj": rot_hsi_ref_obj,
        "ray_directions_local": p_dir,  # Shape: (n_pixels, 3)
    }
```

At this point, we have:
- **Ray directions** for each pixel column (in camera frame)
- **Camera position offset** from IMU (2.5 m forward)
- **Camera orientation** relative to body frame

---

## Step 5: Transforming Rays to World Coordinates

### Location in Code
**File:** `gref4hsi/scripts/georeference.py`, function `define_hsi_ray_geometry()`

For each HSI frame (scan line):

```python
def define_hsi_ray_geometry(pos_ref_ecef, quat_ref_ecef, time_pose, intrinsic_geometry_dict):
    """
    Inputs:
    - pos_ref_ecef: IMU positions in ECEF coordinates (N_frames x 3)
    - quat_ref_ecef: IMU orientations as quaternions (N_frames x 4)
    - intrinsic_geometry_dict: Camera model with ray directions
    
    Returns:
    - CameraGeometry object with rays in world coordinates
    """
    
    # Extract camera offset and rotation
    translation_ref_hsi = intrinsic_geometry_dict["translation_ref_hsi"]  # [2.5, 0, 0]
    rot_hsi_ref_obj = intrinsic_geometry_dict["rot_hsi_ref_obj"]
    ray_directions_local = intrinsic_geometry_dict["ray_directions_local"]  # (n_pix, 3)
    
    # For each frame:
    for frame_idx in range(N_frames):
        # 1. Get IMU position & orientation at this time
        imu_pos_ecef = pos_ref_ecef[frame_idx]
        imu_quat_ecef = quat_ref_ecef[frame_idx]
        R_imu_to_ecef = Rotation.from_quat(imu_quat_ecef).as_matrix()
        
        # 2. Transform camera offset from body to ECEF
        camera_offset_ecef = R_imu_to_ecef @ translation_ref_hsi
        camera_pos_ecef = imu_pos_ecef + camera_offset_ecef
        
        # 3. Compute camera orientation in ECEF
        R_camera_to_body = rot_hsi_ref_obj.as_matrix()
        R_camera_to_ecef = R_imu_to_ecef @ R_camera_to_body
        
        # 4. Transform ray directions from camera frame to ECEF
        for pixel_idx in range(n_pixels):
            ray_direction_camera = ray_directions_local[pixel_idx]  # [x_norm, 0, 1]
            ray_direction_ecef = R_camera_to_ecef @ ray_direction_camera
            
            # Store ray origin and direction
            rays[frame_idx, pixel_idx] = {
                'origin': camera_pos_ecef,
                'direction': ray_direction_ecef / np.linalg.norm(ray_direction_ecef)
            }
    
    return CameraGeometry(...)
```

Now we have a **grid of rays** (N_frames × N_pixels) where each ray has:
- **Origin:** Camera position in ECEF
- **Direction:** Unit vector pointing from camera through pixel toward ground

---

## Step 6: Ray-Mesh Intersection

### Location in Code
**File:** `gref4hsi/utils/geometry_utils.py`, method `intersect_with_mesh()`

This is where the actual **raytracing** happens:

```python
def intersect_with_mesh(self, mesh, max_ray_length, mesh_trans):
    """
    Intersects all camera rays with the DEM mesh
    
    Inputs:
    - mesh: 3D triangular mesh (DEM converted to triangles)
    - max_ray_length: Maximum distance to search (e.g., 20 meters)
    - mesh_trans: Mesh offset for numerical stability
    
    Outputs:
    - points_ecef_crs: Intersection points (N_frames x N_pixels x 3)
    - normals_ecef_crs: Surface normals at intersection points
    """
    
    n_frames = self.rayDirectionsGlobal.shape[0]
    n_pixels = self.rayDirectionsGlobal.shape[1]
    
    # Prepare ray origins and directions for batch intersection
    start_ECEF = self.position_ecef  # Camera positions (N_frames, 3)
    start_ECEF = np.repeat(start_ECEF[:, None, :], n_pixels, axis=1)  # (N_frames, N_pixels, 3)
    
    start = start_ECEF.reshape((-1, 3)) - mesh_trans  # Flatten and offset
    directions = (self.rayDirectionsGlobal * max_ray_length).reshape((-1, 3))
    
    # Perform ray-mesh intersection using PyEmbree (fast) or Trimesh (fallback)
    try:
        # Fast method: PyEmbree multi-ray trace
        points, rays, cells = mesh.multi_ray_trace(
            origins=start,
            directions=directions,
            first_point=True,
            retry=True
        )
    except:
        # Fallback: Trimesh intersection
        faces = mesh.regular_faces
        tri_mesh = trimesh.Trimesh(vertices=mesh.points, faces=faces)
        ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tri_mesh)
        
        cells, rays, points = ray_mesh_intersector.intersects_id(
            ray_origins=start,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=True
        )
    
    # Check for failed intersections
    n_rays = start.shape[0]
    n_hits = len(points) // 3
    
    if n_hits != n_rays:
        print(f"Warning: {n_rays - n_hits} rays did not intersect DEM")
        # This usually means DEM doesn't cover the FOV
    
    # Reshape intersection points back to grid
    self.points_ecef_crs = points.reshape((n_frames, n_pixels, 3)) + mesh_trans
    
    # Extract surface normals at intersection points
    self.normals_ecef_crs = mesh.compute_normals()[cells].reshape((n_frames, n_pixels, 3))
```

### Ray-Mesh Intersection Algorithm

The intersection uses **Möller-Trumbore algorithm** internally:

```
For each ray (origin O, direction D):
    For each triangle in mesh (vertices V0, V1, V2):
        1. Compute ray-plane intersection
        2. Check if intersection point is inside triangle
        3. If yes, compute distance t
        4. Return closest intersection point: P = O + t*D
```

**PyEmbree** accelerates this using:
- **Bounding Volume Hierarchy (BVH):** Quickly reject triangles far from ray
- **SIMD instructions:** Process multiple rays in parallel
- **Early termination:** Stop at first intersection (since we only need closest point)

---

## Summary: How the System Knows the FOV

1. **H5 file contains FOV data** stored during lab calibration or from sensor specs
   - Path: `processed/radiance/calibration/geometric/fieldOfView`
   - Format: Array of viewing angles in degrees

2. **FOV → Camera model** conversion happens in `fov_2_param()`
   - Fits pinhole + distortion model to match the measured FOV angles
   - Outputs focal length, distortion coefficients, principal point

3. **Camera model → Ray directions** in `cal_file_to_rays()`
   - Uses camera intrinsics to compute 3D ray direction for each pixel
   - Ray directions are defined in camera frame: `[x_norm, 0, 1]`

4. **Ray directions → World coordinates** in `define_hsi_ray_geometry()`
   - Transforms rays using IMU position + camera offset (2.5m forward)
   - Rotates rays using IMU orientation + camera mounting angle

5. **Rays → Ground intersections** in `intersect_with_mesh()`
   - Casts each ray into the DEM mesh
   - Finds exact 3D point where ray hits terrain
   - These points become the geolocation for each HSI pixel

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│  H5 File                                                     │
│  └─ FOV array: [-17.5°, -17.3°, ..., 0°, ..., +17.5°]      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Camera Model (pinhole + distortion)                        │
│  └─ f=500 pixels, cx=512, k1, k2, k3                       │
│  └─ Ray directions: [[x₁,0,1], [x₂,0,1], ..., [xₙ,0,1]]   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  IMU Navigation Data                                         │
│  └─ Position: [lat, lon, depth]                            │
│  └─ Orientation: [roll, pitch, yaw]                        │
│  └─ Camera offset: [2.5m, 0, 0] forward                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  World-Space Rays (ECEF coordinates)                        │
│  └─ Origin: Camera position = IMU pos + 2.5m forward       │
│  └─ Direction: Rotated ray direction (unit vector)         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  DEM Mesh (triangulated terrain)                            │
│  └─ Ray-triangle intersection using PyEmbree/Trimesh        │
│  └─ Result: 3D point on ground for each pixel              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files Reference

| File | Function | Purpose |
|------|----------|---------|
| `gref4hsi/utils/uhi_parsing_utils.py` | `uhi_eely()` | Extracts FOV from H5 files |
| `gref4hsi/utils/specim_parsing_utils.py` | `fov_2_param()` | Converts FOV angles to camera model |
| `gref4hsi/utils/uhi_parsing_utils.py` | `set_camera_model()` | Adds extrinsic parameters (position/rotation) |
| `gref4hsi/scripts/georeference.py` | `cal_file_to_rays()` | Builds ray directions from camera model |
| `gref4hsi/scripts/georeference.py` | `define_hsi_ray_geometry()` | Transforms rays to world coordinates |
| `gref4hsi/utils/geometry_utils.py` | `intersect_with_mesh()` | Performs raytracing to find ground points |

---

## Example: Tracing a Single Pixel

Let's trace pixel column 512 (center pixel) through the entire pipeline:

```python
# Step 1: FOV extraction
fov_512 = 0.0°  # Center pixel has 0° viewing angle (nadir)

# Step 2: Camera model
x_norm = (512 - 512.5) / 500.0 = -0.001  # Normalized coordinate
ray_camera = [-0.001, 0, 1]  # Ray direction in camera frame

# Step 3: Navigation data (example frame)
imu_position = [100, 200, -110]  # North, East, Down (meters in NED)
imu_orientation = [roll=0°, pitch=-10°, yaw=295°]

# Step 4: Transform to body frame
camera_offset_body = [2.5, 0, 0]  # 2.5m forward
R_body_to_ned = rotation_from_euler(yaw=295°, pitch=-10°, roll=0°)
camera_offset_ned = R_body_to_ned @ [2.5, 0, 0] = [2.4, -0.5, -0.4]
camera_position = [100+2.4, 200-0.5, -110-0.4] = [102.4, 199.5, -110.4]

# Step 5: Rotate ray to NED frame
R_camera_to_body = [[0,1,0], [-1,0,0], [0,0,1]]  # 90° rotation
ray_body = R_camera_to_body @ [-0.001, 0, 1] = [0, -0.001, 1]
ray_ned = R_body_to_ned @ [0, -0.001, 1] = [0.17, -0.98, -0.11]  # Pointing down-ish

# Step 6: Raytracing
ray_origin = [102.4, 199.5, -110.4]
ray_direction = [0.17, -0.98, -0.11]  # Unit vector
max_length = 20m
intersection_point = ray_trace(origin, direction, dem_mesh)
# Result: [104, 180, -122]  # Ground point 11.6m below camera
```

This intersection point `[104, 180, -122]` becomes the geolocation for pixel 512 in this HSI frame.

---

## Common Issues & Debugging

### "No intersections found"
- **Cause:** DEM doesn't cover the area where camera is looking
- **Fix:** Check DEM time range matches HSI time range
- **Debug:** Plot `camera_position` and `DEM extent` to verify overlap

### "Too many rays missing"
- **Cause:** Camera pointing outside DEM boundary or max_ray_length too short
- **Fix:** Increase `max_ray_length` or extend DEM coverage
- **Debug:** Check `camera_altitude` vs `range_sensor_altitude` (should be ~equal)

### "Camera altitude > IMU altitude"
- **Cause:** Incorrect lever arm or rotation matrix
- **Fix:** Verify `translation_body_to_hsi` sign and `rotation_matrix_hsi_to_body`
- **Debug:** Use `camera_imu_altitude_comparison.py` to visualize the difference

---

**Last Updated:** Oct 2025  
**Author:** Liu (with AI assistance)  
**Related:** See also `GEOREFERENCING_PIPELINE_EXPLAINED.md` in `x_gref4hsi_by_liu/`
