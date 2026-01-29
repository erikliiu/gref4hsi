# Hyperspectral Georeferencing Pipeline (Modified gref4hsi)

## Overview

This pipeline georef erences underwater hyperspectral (UHI) imagery by **ray tracing from the HSI camera to a surveyed MBES bathymetry mesh**. Each HSI pixel gets a 3D position where its ray intersects the actual seafloor.

**Key Difference from Original gref4hsi:**
- **Original:** Creates a DEM (Digital Elevation Model) from vehicle altitude/depth data → assumes flat or smoothly varying seafloor
- **This Version:** Uses actual MBES GeoTIFF bathymetry → handles real terrain variation (rocks, slopes, munitions)

---

## What Problem Does This Solve?

**Problem:** HSI cameras capture spectral data, but without georeferencing, you don't know WHERE on the seafloor each pixel is looking.

**Solution:** Ray trace from camera position through each pixel to find where it hits the seafloor (using MBES ground truth bathymetry).

**Output:** Gridded H5 file with `processed/georef/points_ecef_crs` dataset containing (T, S, 3) array:
- T = number of HSI frames (track points)
- S = number of spatial pixels (968 slits across swath)
- 3 = ECEF coordinates (X, Y, Z) of seafloor intersection

**Why MBES?** The original gref4hsi created a synthetic DEM from altitude measurements, which doesn't capture actual seafloor features. MBES provides millimeter-accurate bathymetry, enabling:
- Correct georeferencing over uneven terrain
- Co-location of HSI spectral anomalies with MBES bathymetric anomalies
- Multi-sensor fusion for target detection

---

## Pipeline Workflow

### **Inputs Required:**

1. **Navigation CSV** (`nav_data_merged.csv`)
   - Columns: timestamp, lat, lon, depth, roll, pitch, yaw, altitude
   - From corrected navigation (see notebook 1: navigation preprocessing)

2. **MBES GeoTIFF** (`geotiff_2.tif`)
   - Bathymetry raster (depth in meters)
   - Coordinate system: UTM Zone 32N (EPSG:32632)
   - Resolution: ~0.05-0.10 m/pixel

3. **HSI H5 Files** (`rad_uhi_*.h5`)
   - UHI calibrated radiance: `processed/radiance/dataCube` (T, S, B)
   - Timestamps: `processed/radiance/timestamp` (T,)
   - T = frames, S = spatial pixels (968), B = spectral bands (210)

4. **Camera Calibration XML** (`HSI_2_body.xml`)
   - Focal length `f`, principal point `cx`, distortion `k1, k2, k3`
   - Sensor mounting: rotation and translation relative to vehicle body

### **Processing Steps:**

#### **Step 1: Load Navigation CSV**
```python
nav_data = utils.load_csv_navigation(config.NAV_CSV, config.CSV_COLUMNS)
```
- Reads corrected navigation data
- Columns: timestamp, latitude, longitude, depth, roll, pitch, yaw, altitude
- Time range: Full survey duration (~2-3 hours)

---

#### **Step 2: Load MBES GeoTIFF**
```python
elevation_data, transform, crs_epsg = georeference_mbes.load_mbes_geotiff(config.MBES_GEOTIFF)
```
**What it does:**
- Reads bathymetry raster (2D array of depths in meters)
- Extracts affine transform (pixel → UTM coordinates)
- Handles NoData values (NaN)

**MBES Data Structure:**
- Shape: e.g., (12000, 8000) pixels
- CRS: UTM Zone 32N (EPSG:32632)
- Resolution: 0.05 m/pixel
- Depth range: ~120-122 m below sea surface

---

#### **Step 3: Convert GeoTIFF to Mesh**
```python
mbes_mesh = georeference_mbes.geotiff_to_mesh(elevation_data, transform)
```

**Why a mesh?**
- Ray tracing requires a surface (not a raster)
- PyVista mesh = vertices + triangular faces
- Enables fast ray-surface intersection

**Mesh Creation Process:**
1. Build (X, Y, Z) grid from affine transform
2. Create PyVista StructuredGrid
3. Extract surface as PolyData (triangular mesh)
4. Optionally simplify (decimate) to reduce face count

**Output:**
- Mesh vertices: ~96 million points
- Mesh faces: ~192 million triangles
- Bounds: UTM coordinates matching GeoTIFF extent

---

#### **Step 4: Load Camera Calibration**
```python
camera_calib = utils.load_camera_calibration(config.CAMERA_CALIB_XML)
```

**Calibration Parameters:**
- `f`: Focal length (pixels)
- `cx`: Principal point (center pixel)
- `w`: Image width (968 pixels for UHI)
- `k1, k2, k3`: Radial distortion coefficients

**Camera Frame Convention:**
- X-axis: Across-track (perpendicular to flight)
- Y-axis: Along-track (always 0 for pushbroom scanner)
- Z-axis: Down (nadir/boresight direction)

---

#### **Step 5: Process Each H5 File**

For each HSI H5 file, the `process_h5_file()` function:

##### **5a. Load HSI Timestamps**
```python
hsi_timestamps = utils.load_h5_timestamps(h5_path)
```
- Reads `processed/radiance/timestamp` from H5
- One timestamp per frame (T values)
- Example: 2461 frames over ~123 seconds

##### **5b. Interpolate Navigation to HSI Frames**
```python
interp_nav = utils.interpolate_navigation(nav_data, hsi_timestamps, time_offset=508)
```

**Why interpolate?**
- Navigation logged at ~10 Hz
- HSI captures at ~20 Hz
- Need vehicle pose (lat, lon, depth, roll, pitch, yaw) at EXACT HSI frame times

**Method:** Linear interpolation with time offset correction (508 seconds)

**Output:** DataFrame with 2461 rows (one per HSI frame):
```
timestamp, latitude, longitude, depth, roll, pitch, yaw, altitude
```

---

##### **5c. Coordinate Transformations**

**Geographic → ECEF:**
```python
positions_ecef = utils.geographic_to_ecef(lon, lat, -depth, epsg_geo=4326, epsg_ecef=4978)
```
- Converts lat/lon/depth to Earth-Centered Earth-Fixed (ECEF)
- ECEF origin at Earth's center
- All coordinates in meters

**Euler Angles → Rotation Matrices:**
```python
orientations = utils.euler_to_rotation_matrix_nav(roll, pitch, yaw, yaw_convention="heading_from_north_cw")
```
- Converts roll/pitch/yaw (degrees) to 3×3 rotation matrices
- Yaw convention: Compass heading (0° = North, 90° = East, clockwise)
- Orientation matrices: **BODY → WORLD (ENU frame)**

**Apply Sensor Transform:**
```python
hsi_positions_ecef, hsi_orientations = utils.apply_sensor_transform(
    positions_ecef, orientations, 
    config.ROTATION_HSI_TO_BODY, 
    config.TRANSLATION_BODY_TO_HSI
)
```

**What this does:**
- Transforms from vehicle body frame to HSI camera frame
- Translation: HSI camera mounted 2.5 m to port (negative Y in body frame)
- Rotation: Aligns camera axes with body axes

**Coordinate frame mappings:**
- **Body frame:** +X forward, +Y starboard (right), +Z up
- **Camera frame:** +X across-track, +Y along-track, +Z down
- **Mapping:** Xₒₐₘ → +Yᵦₒdy, Yₒₐₘ → +Xᵦₒdy, Zₒₐₘ → −Zᵦₒdy

**ECEF → UTM (MBES CRS):**
```python
hsi_positions_mbes = utils.ecef_to_utm(hsi_positions_ecef, epsg_utm=32632, epsg_ecef=4978)
```
- Converts HSI camera positions to same CRS as MBES mesh
- UTM Zone 32N (EPSG:32632)
- Now camera positions and mesh are in same reference frame!

---

##### **5d. Build Ray Directions**

**Read HSI Data Shape:**
```python
with h5py.File(h5_path) as h5f:
    datacube = h5f['processed/radiance/dataCube']  # (T, S, B)
    n_slits = datacube.shape[1]  # 968 spatial pixels
```

**Build Rays in Camera Frame:**
```python
ray_directions_camera = utils.build_ray_directions(camera_calib, n_slits=968)
```

**Ray Direction Formula:**
For each pixel `u` (1 to 968):

1. **Linear component:**
   ```
   x_norm_lin = (u - cx) / f
   ```

2. **Distortion component:**
   ```
   r = (u - cx) / 1000.0
   x_norm_nonlin = -(k1·r⁵ + k2·r³ + k3·r²) / f
   ```

3. **Total normalized X:**
   ```
   x_norm = x_norm_lin + x_norm_nonlin
   ```

4. **Ray direction (unnormalized):**
   ```python
   ray_dir = [x_norm, 0, 1]  # X=across-track, Y=0 (pushbroom), Z=down
   ```

**Output:** Array of 968 ray directions in camera frame

**NOTE:** Rays are **NOT normalized** — this matches original gref4hsi convention!

---

##### **5e. Transform Rays to World Frame**

```python
for i in range(n_frames):  # For each HSI frame
    R_world_from_cam = hsi_orientations[i]  # Already includes sensor transform
    dirs_world = (R_world_from_cam @ ray_directions_camera.T).T
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)  # Normalize
```

**What this does:**
- Rotates each ray from camera frame to world frame (ENU)
- Uses orientation matrix computed earlier
- Normalizes ray directions

**Total rays:** n_frames × n_slits = 2461 × 968 = **2,382,248 rays**

**Sanity check:**
```python
mean_boresight_dot_up = np.mean(ray_directions_world @ [0, 0, 1])
# Should be negative (rays pointing down)
```

---

##### **5f. Ray Trace to MBES Mesh**

```python
results = georeference_mbes.raytrace_hsi_to_mbes(
    ray_origins_mbes,      # (N, 3) - Camera positions in UTM
    ray_directions_world,  # (N, 3) - Ray directions in world frame
    mbes_mesh,             # PyVista mesh
    max_ray_length=100     # meters
)
```

**Ray Tracing Algorithm:**

**Phase 1: Bulk intersection (Trimesh + Embree)**
- Uses Embree ray tracer (Intel's fast ray-triangle intersection library)
- Processes all 2.4 million rays in parallel
- Typical result: ~95% hit rate (2.26M successful intersections)

**Phase 2: Retry missed rays (PyVista)**
- For rays that missed in Phase 1, retry individually with PyVista
- PyVista uses different intersection algorithm (may catch edge cases)
- Typically recovers 1-5% of missed rays

**Output:**
```python
results = {
    'points': ndarray (M, 3),        # Intersection points in UTM (M successful rays)
    'ray_indices': ndarray (M,),     # Which ray hit (0 to N-1)
    'cell_indices': ndarray (M,),    # Which mesh triangle was hit
    'success_rate': float            # Percentage of successful intersections
}
```

**Example stats:**
```
Total rays: 2,382,248
Trimesh hits: 2,266,912 (95.2%)
PyVista retries: 115,336
PyVista recoveries: 98,221
Final success rate: 99.3% (2,365,133 intersections)
```

**Why some rays fail:**
- Ray shoots past mesh edge (outside MBES coverage)
- Numerical precision issues at mesh boundaries
- Very shallow angles (grazing incidence)

---

##### **5g. Convert Back to ECEF**

```python
intersections_ecef = utils.utm_to_ecef(results['points'], epsg_utm=32632, epsg_ecef=4978)
```

**Why convert back?**
- H5 file stores georeferenced data in ECEF (matches original gref4hsi)
- ECEF is a global coordinate system (not zone-specific like UTM)
- Consistent with other processing steps

---

##### **5h. Save to H5 File (Gridded Format)**

```python
utils.save_intersection_to_h5(
    output_h5_path,
    intersections_ecef,  # (M, 3) - Successful intersections
    pixel_indices,        # (M,) - Which slit (0-967)
    frame_indices,        # (M,) - Which frame (0-2460)
    n_frames=2461,
    n_slits=968,
    gridded=True  # Save as (T, S, 3) array
)
```

**Gridded Format:**

Creates 3D array: `(n_frames, n_slits, 3)`

Example: `(2461, 968, 3)` array

- Failed rays filled with `NaN`
- Successful intersections filled with ECEF coordinates `[X, Y, Z]`

**H5 File Structure:**
```
processed/
  ├── radiance/
  │   ├── dataCube (T, S, B) - Spectral data
  │   └── timestamp (T,) - Frame times
  └── georef/  ← ADDED BY THIS PIPELINE
      ├── points_ecef_crs (T, S, 3) - ECEF coordinates of each pixel
      ├── pixel_nr_grid (T, S) - Pixel indices
      └── frame_nr_grid (T, S) - Frame indices
```

**Why gridded?**
- Preserves hypercube structure (frames × pixels × coordinates)
- Easy to align with spectral data (same T × S dimensions)
- Simplifies downstream processing (no need to reconstruct grid)

---

#### **Step 6: Compute Statistics**

The pipeline computes comprehensive statistics:

**Mission Metrics:**
- Total distance traveled (3D path length)
- Survey duration (seconds)
- Average/min/max speed (m/s and km/h)

**Coverage Metrics:**
- Mean/min/max/std swath width (meters)
- Total coverage area (m²)
- Bounding box area (m²)
- Coverage efficiency (% of bounding box covered)

**Ray Tracing Metrics:**
- Total rays cast
- Successful intersections
- Success rate (%)
- Trimesh hits / PyVista retries / PyVista recoveries

**Output:** `processing_statistics.json` with all metrics

---

## Key Advantages Over Original gref4hsi

| Aspect | Original gref4hsi | This Pipeline |
|--------|------------------|---------------|
| **Bathymetry Source** | DEM from altitude data | MBES GeoTIFF (surveyed) |
| **Terrain Handling** | Assumes smooth seafloor | Handles rocks, slopes, munitions |
| **Accuracy** | ±0.5-1.0 m (altitude noise) | ±0.05-0.10 m (MBES resolution) |
| **Multi-Sensor Fusion** | HSI only | HSI + MBES co-registered |
| **Target Validation** | Spectral-only detection | Spectral + bathymetric confirmation |
| **Use Case** | Flat or gently sloping seafloor | Complex terrain, munition surveys |

---

## Coordinate Systems Summary

The pipeline juggles multiple coordinate systems:

1. **Geographic (EPSG:4326):** Lat/lon/height (input navigation)
2. **ECEF (EPSG:4978):** Earth-centered Cartesian (intermediate, H5 storage)
3. **UTM Zone 32N (EPSG:32632):** Projected coordinates (MBES mesh, ray tracing)
4. **NED (Local):** North-East-Down relative to origin (visualization, MBES detrending)

**Transformation Flow:**
```
Navigation CSV (Geographic)
    ↓ geographic_to_ecef()
ECEF (vehicle positions)
    ↓ apply_sensor_transform()
ECEF (camera positions)
    ↓ ecef_to_utm()
UTM (camera positions) ← Ray tracing happens here! → UTM (MBES mesh)
    ↓ utm_to_ecef()
ECEF (intersection points) → Saved to H5
```

---

## Camera Frame Conventions

**Body Frame:**
- +X: Forward (vehicle heading)
- +Y: Starboard/right
- +Z: Up

**HSI Camera Frame:**
- +X: Across-track (perpendicular to flight)
- +Y: Along-track (parallel to flight, **always 0 for pushbroom**)
- +Z: Down (nadir/boresight)

**Sensor Mounting:**
```python
ROTATION_HSI_TO_BODY = [[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, -1]]

TRANSLATION_BODY_TO_HSI = [0, -2.5, 0]  # 2.5 m to port
```

**Orientation Matrices:**
- `R_body_to_world`: From navigation (roll, pitch, yaw)
- `R_cam_to_body`: Fixed sensor mounting (config)
- `R_cam_to_world = R_body_to_world @ R_cam_to_body`

---

## Configuration Parameters

Key parameters in `config.py`:

**Sensor Transform:**
```python
ROTATION_HSI_TO_BODY = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
TRANSLATION_BODY_TO_HSI = np.array([0, -2.5, 0])  # meters
```

**Time Synchronization:**
```python
TIME_OFFSET_SEC = 508  # Offset between nav and HSI clocks
```

**Ray Tracing:**
```python
MAX_RAY_LENGTH = 100  # meters (depth range)
EARLY_FAILURE_THRESHOLD = 50.0  # Abort if >50% rays fail
```

**Coordinate Systems:**
```python
EPSG_GEOGRAPHIC = 4326   # WGS84
EPSG_UTM = 32632         # UTM Zone 32N
EPSG_ECEF = 4978         # ECEF
EPSG_MBES = 32632        # MBES GeoTIFF CRS
```

**NED Origin:**
```python
LON0 = 10.705125  # Mjøsa survey area
LAT0 = 60.801146
H0 = 0.0
```

---

## Mathematical Foundations

This section explains the mathematical transformations that convert HSI sensor measurements into georeferenced 3D positions. The georeferencing pipeline performs a series of coordinate transformations:

**Camera Model → Body Frame → World Frame → Ray-Mesh Intersection → Final Position**

### 1. Camera Model: Pixel to Ray Direction

Each pixel in the HSI pushbroom sensor generates a ray direction in the camera frame. The camera calibration parameters define how pixel coordinates map to 3D ray directions.

**Camera Frame Convention:**
- **X-axis**: Across-track direction (perpendicular to flight)
- **Y-axis**: Along-track direction (flight direction) - always 0 for pushbroom
- **Z-axis**: Down (nadir/boresight direction) - always 1 (unnormalized)

**Distortion Model:**

The camera has both linear (pinhole) and nonlinear (radial distortion) components:

```
x_norm = x_norm_linear + x_norm_nonlinear
```

Where:
```
x_norm_linear = (u - cx) / f

r = (u - cx) / 1000                                    (scaled for numerical stability)
x_norm_nonlinear = -(k1·r⁵ + k2·r³ + k3·r²) / f       (radial distortion)
```

**Parameters:**
- `u`: Pixel index (1-based, from 1 to 968)
- `cx`: Principal point x-coordinate (pixel 484.5, sensor center)
- `f`: Focal length (in pixel units, typically ~1000)
- `k1, k2, k3`: Radial distortion coefficients

**Ray Direction in Camera Frame:**

For pixel `u`, the ray direction vector is:

```
ray_camera = [x_norm, 0, 1]
```

This is **not normalized** yet - it follows the `test_eely` convention where Z=1 always.

**Example:**
```python
# Center pixel (u=484)
u = 484
cx = 484.5
f = 1000.0
k1, k2, k3 = -0.001, 0.002, -0.0005  # Example distortion coefficients

# Linear component
x_norm_lin = (484 - 484.5) / 1000.0 = -0.0005

# Nonlinear component
r = (484 - 484.5) / 1000.0 = -0.0005
x_norm_nonlin = -(k1*r**5 + k2*r**3 + k3*r**2) / f ≈ 0.0

# Total
x_norm = -0.0005 + 0.0 = -0.0005

# Ray direction (unnormalized)
ray_camera = [-0.0005, 0, 1]
```

This represents a ray pointing nearly straight down with a tiny leftward tilt.

### 2. Body Frame Transform: Vehicle to Sensor

The HSI camera is mounted on the UAV body with a fixed offset and rotation. The sensor's position and orientation must be transformed from the vehicle body frame to the world frame.

**Sensor Position in World Frame:**

```
P_sensor_world = P_vehicle_world + R_body_to_world @ T_body_to_sensor
```

Where:
- `P_vehicle_world`: Vehicle position in ECEF coordinates (from navigation system)
- `R_body_to_world`: 3×3 rotation matrix from vehicle body frame to world frame
- `T_body_to_sensor`: Translation vector from vehicle body to sensor (e.g., [0, -2.5, 0] meters)

**Sensor Orientation in World Frame:**

```
R_sensor_to_world = R_body_to_world @ R_sensor_to_body
```

Where:
- `R_sensor_to_body`: Fixed rotation matrix defining camera mounting orientation

For the Mjøsa deployment, the camera is rotated 90° such that:
```python
R_sensor_to_body = [[0,  1,  0],
                     [1,  0,  0],
                     [0,  0, -1]]
```

This rotation:
- Maps camera X (across-track) → body Y (port/starboard)
- Maps camera Y (along-track) → body X (nose/tail)
- Maps camera Z (down) → body -Z (flip vertical)

**Example:**
```python
# Vehicle position (ECEF meters)
P_vehicle_world = [3154123.45, 591234.56, 5789123.78]

# Vehicle orientation (roll, pitch, yaw converted to rotation matrix)
# Assume level flight: roll=0°, pitch=0°, yaw=45°
R_body_to_world = [[0.707, -0.707, 0],
                    [0.707,  0.707, 0],
                    [0,      0,     1]]

# Sensor offset (2.5m to port side)
T_body_to_sensor = [0, -2.5, 0]

# Transform translation to world frame
translation_world = R_body_to_world @ [0, -2.5, 0]
                  = [0.707*0 + (-0.707)*(-2.5) + 0*0,
                     0.707*0 +   0.707*(-2.5) + 0*0,
                     0*0     +   0*(-2.5)     + 1*0]
                  = [1.768, -1.768, 0]

# Sensor position in world frame
P_sensor_world = [3154123.45, 591234.56, 5789123.78] + [1.768, -1.768, 0]
               = [3154125.218, 591232.792, 5789123.78]
```

The sensor is offset 1.768 m east and 1.768 m south from the vehicle position.

### 3. Ray Transformation: Camera Frame to World Frame

Once we have ray directions in the camera frame and the sensor's orientation in the world frame, we transform the rays to world coordinates.

**Ray Direction in World Frame:**

```
ray_world = (R_sensor_to_world @ ray_camera.T).T
ray_world = ray_world / ||ray_world||                 (normalize)
```

The transpose operations `(R @ rays.T).T` efficiently transform all rays at once in a vectorized NumPy operation.

**Example:**
```python
# Ray in camera frame (center pixel)
ray_camera = [-0.0005, 0, 1]

# Sensor orientation in world frame
R_sensor_to_world = R_body_to_world @ R_sensor_to_body
                  = [[0.707, -0.707, 0],     [[0,  1,  0],
                     [0.707,  0.707, 0],  @   [1,  0,  0],
                     [0,      0,     1]]      [0,  0, -1]]
                  = [[-0.707,  0.707, 0],
                     [ 0.707,  0.707, 0],
                     [ 0,      0,    -1]]

# Transform ray to world frame
ray_world_unnorm = R_sensor_to_world @ [-0.0005, 0, 1]
                 = [(-0.707)*(-0.0005) + 0.707*0 + 0*1,
                    0.707*(-0.0005)    + 0.707*0 + 0*1,
                    0*(-0.0005)        + 0*0     + (-1)*1]
                 = [0.0003535, -0.0003535, -1]

# Normalize
||ray_world_unnorm|| = sqrt(0.0003535² + 0.0003535² + 1²) ≈ 1.0
ray_world = [0.0003535, -0.0003535, -1.0]
```

This ray points almost straight down (-Z direction) with tiny eastward and southward tilts.

### 4. Ray-MBES Intersection

The final step is finding where each ray intersects the MBES bathymetry mesh.

**Ray Parametric Equation:**

A ray is defined by an origin and direction:

```
P(t) = P_origin + t · d_ray
```

Where:
- `P_origin`: Ray origin (sensor position in ECEF or UTM)
- `d_ray`: Ray direction (normalized)
- `t`: Distance parameter (t ≥ 0)

**Intersection Algorithm:**

The pipeline uses **Trimesh with Embree ray tracer** for fast ray-triangle intersection:

1. **Convert mesh**: MBES bathymetry (PyVista PolyData) → Trimesh format
2. **Ray trace**: For each ray, find the first triangle it intersects
3. **Solve for t**: Find parameter `t` where ray intersects triangle plane
4. **Compute point**: `P_intersection = P_origin + t · d_ray`
5. **Retry failures**: If ray misses mesh, retry with PyVista ray tracer

**Intersection Formula:**

For a triangle with vertices `v0, v1, v2`:

```
P = P_origin + t·d_ray                                    (ray equation)
P = v0 + u·(v1 - v0) + v·(v2 - v0)                       (barycentric on triangle)

where u, v ≥ 0 and u + v ≤ 1                              (inside triangle)
```

The Embree library solves this system efficiently using the Möller-Trumbore algorithm.

**Example:**
```python
# Ray origin (sensor position in UTM)
P_origin = [599234.5, 6745123.8, 125.3]  # (Easting, Northing, Height)

# Ray direction (world frame, normalized)
d_ray = [0.0003535, -0.0003535, -1.0]

# Ray equation for t=0 to t=100 meters
t_max = 100
P_end = P_origin + t_max * d_ray
      = [599234.5, 6745123.8, 125.3] + 100*[0.0003535, -0.0003535, -1.0]
      = [599234.5354, 6745123.7646, 25.3]

# Embree finds intersection at t=93.7 meters
t_intersection = 93.7
P_intersection = [599234.5, 6745123.8, 125.3] + 93.7*[0.0003535, -0.0003535, -1.0]
               = [599234.5331, 6745123.7669, 31.6]

# This point is at depth 125.3 - 31.6 = 93.7 meters below sensor
```

**Failure Handling:**

- If more than 50% of rays miss the mesh → **abort processing** (MBES doesn't cover HSI field of view)
- If fewer rays miss → **retry with PyVista** ray tracer (slower but more robust)
- Missing rays are logged for debugging

### 5. Complete Transformation Chain

Here's the full pipeline for a single pixel:

```
1. Pixel u → Ray in camera frame:
   ray_cam = [x_norm, 0, 1]  where x_norm = (u - cx)/f + distortion

2. Sensor position in world frame:
   P_sensor_world = P_vehicle_world + R_body_to_world @ T_body_to_sensor

3. Sensor orientation in world frame:
   R_sensor_to_world = R_body_to_world @ R_sensor_to_body

4. Ray in world frame:
   ray_world = normalize(R_sensor_to_world @ ray_cam)

5. Ray-mesh intersection:
   P_intersection = P_sensor_world + t·ray_world  (solve for t using Embree)

6. Convert back to ECEF (if working in UTM):
   P_ecef = UTM_to_ECEF(P_intersection)

7. Result: Georeferenced 3D position [X_ecef, Y_ecef, Z_ecef]
```

### 6. Coordinate System Summary

The pipeline uses multiple coordinate reference systems:

| **CRS** | **EPSG** | **Usage** | **Units** |
|---------|----------|-----------|-----------|
| Geographic WGS84 | 4326 | Navigation input (lat, lon, height) | Degrees, meters |
| ECEF | 4978 | 3D Cartesian world frame | Meters (X, Y, Z) |
| UTM Zone 32N | 32632 | Ray tracing (flat approximation) | Meters (Easting, Northing, Height) |
| NED Local | Custom | Body frame orientation reference | Meters (North, East, Down) |

**Conversion Order:**
```
Geographic (4326) → ECEF (4978) → UTM (32632) [ray tracing] → ECEF (4978) [output]
```

**Why use UTM for ray tracing?**
- UTM is a flat, metric coordinate system (easier for distance calculations)
- MBES mesh is in UTM (loaded from GeoTIFF with UTM CRS)
- Ray directions and distances are more intuitive in meters
- Final output is converted back to ECEF for georeferencing

---

## Output Files

**Per H5 File:**
- `output/rad_uhi_*.h5` - Copy of input with added `processed/georef/` datasets

**Processing Statistics:**
- `output/processing_statistics.json` - Comprehensive metrics

**Optional:**
- `output/mbes_mesh.ply` - MBES mesh for visualization (if `SAVE_MESH_PLY=True`)

---

## Usage Example

```python
# Run the pipeline
python main.py

# Output:
# ✅ Loaded 2461 HSI frames
# ✅ Interpolated navigation
# ✅ Ray tracing: 2,365,133/2,382,248 rays hit (99.3%)
# ✅ Saved to output/rad_uhi_20241029_115057_4.h5
```

**Load georeferenced data:**
```python
import h5py

with h5py.File('output/rad_uhi_20241029_115057_4.h5', 'r') as h5f:
    # Spectral data
    radiance = h5f['processed/radiance/dataCube'][:]  # (2461, 968, 210)
    
    # Georeferenced positions
    positions = h5f['processed/georef/points_ecef_crs'][:]  # (2461, 968, 3)
    
    # Each pixel (t, s) has:
    #   - Spectral signature: radiance[t, s, :]
    #   - 3D position: positions[t, s, :]
```

---

## Multi-Sensor Fusion Workflow

**Step 1:** Run this gref4hsi pipeline
- Output: HSI with 3D positions

**Step 2:** Run MBES detrending (notebook 2)
- Output: MBES residuals showing bathymetric anomalies

**Step 3:** Convert georeferenced HSI to NED coordinates
```python
from utils.gref_pipeline import utils

# Convert ECEF positions to NED
x_ned, y_ned, z_ned = utils.ecef_to_ned(
    positions[:, :, 0], 
    positions[:, :, 1], 
    positions[:, :, 2],
    ned_origin=(config.LON0, config.LAT0, config.H0)
)
```

**Step 4:** Overlay HSI footprint on MBES residuals
- See notebook 2: `plot_zoomed_residuals()`
- Compare HSI spectral anomalies with MBES bathymetric anomalies

**Step 5:** Validate targets
- **HSI detection + MBES residual** → High-confidence munition
- **HSI detection + no MBES residual** → Surface feature only
- **No HSI detection + MBES residual** → Buried/covered target

---

## Troubleshooting

**Low ray tracing success rate (<90%):**
- Check MBES coverage: Does mesh cover HSI field of view?
- Verify coordinate systems: EPSG codes match?
- Inspect sensor transform: Camera orientation correct?

**Time synchronization errors:**
- Adjust `TIME_OFFSET_SEC` in config
- Check HSI timestamps vs navigation timestamps

**Memory issues:**
- Enable mesh simplification: `MESH_SIMPLIFICATION = True`
- Reduce mesh resolution: `MESH_REDUCTION_FACTOR = 0.5`
- Process fewer H5 files at once

**Coordinate mismatches:**
- Verify GeoTIFF CRS: Check `rasterio.open(geotiff).crs`
- Check yaw convention: `heading_from_north_cw` vs `enu_yaw_from_east_ccw`
- Inspect sensor transform: Right-handed or left-handed?

---

## Technical Notes

**Why Embree Ray Tracer?**
- Intel Embree = optimized ray-triangle intersection library
- ~100x faster than naive Python loops
- Handles millions of rays efficiently

**Why Gridded Format?**
- Matches original gref4hsi output structure
- Easy to align with spectral datacube (same shape)
- Simplifies downstream processing (no index lookups)

**Why ECEF for Storage?**
- Global coordinate system (no zone boundaries)
- Consistent with original gref4hsi
- Easy to convert to any local CRS later

**Numerical Precision:**
- Ray directions normalized to unit vectors
- ECEF coordinates in meters (double precision)
- UTM coordinates also in meters (avoids degree/meter confusion)

---

## References

**Original gref4hsi:**
- GitHub: `https://github.com/havardlovas/gref4hsi`
- Uses altitude-based DEM creation
- Designed for airborne/shallow water surveys

**This Modified Pipeline:**
- MBES-based georeferencing
- Designed for underwater AUV surveys
- Handles complex terrain (munition detection)

**Key Dependencies:**
- PyVista: Mesh representation and visualization
- Trimesh + Embree: Fast ray tracing
- Rasterio: GeoTIFF I/O
- PyProj: Coordinate transformations
- h5py: HDF5 file handling

---

## Summary

**Input:** Navigation CSV + MBES GeoTIFF + HSI H5 files + Camera calibration

**Process:**
1. Load navigation and MBES mesh
2. Interpolate nav to HSI frame times
3. Transform coordinates (Geographic → ECEF → UTM)
4. Build ray directions (camera frame → world frame)
5. Ray trace to MBES mesh (2.4M rays)
6. Save georeferenced positions to H5 (gridded format)

**Output:** HSI H5 files with `processed/georef/points_ecef_crs` dataset

**Result:** Every HSI pixel has a 3D position on the actual surveyed seafloor

**Use:** Multi-sensor fusion, target validation, georeferenced mosaics, spectral mapping
