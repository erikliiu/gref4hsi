# Body Transformation Verification: x_gref4hsi_by_liu vs test_eely

## ✅ CONFIRMED: Your transformations match test_eely exactly

I've verified that your code in `x_gref4hsi_by_liu` uses **the same body transformations and orientation conventions** as `test_eely.py`.

---

## Key Transformation Components

### 1. Rotation Matrix (HSI to Body)

**test_eely.py (line 144):**
```python
rotation_matrix_hsi_to_body = np.array([[0, 1, 0], 
                                         [-1, 0, 0], 
                                         [0, 0, 1]])
```

**config.py (your code):**
```python
ROTATION_HSI_TO_BODY = np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]])
```

✅ **MATCH** - Identical rotation matrix

**Physical Meaning:**
- This is a 90° rotation around the Z-axis (rz = -π/2 in XML)
- Aligns the camera frame with the body frame
- For downward-looking (nadir) camera configuration

---

### 2. Translation Vector (Body to HSI)

**test_eely.py (line 145):**
```python
translation_body_to_hsi = np.array([2.5, 0, 0])
```

**config.py (your code):**
```python
TRANSLATION_BODY_TO_HSI = np.array([2.5, 0, 0])
```

✅ **MATCH** - Identical translation (2.5m forward along body X-axis)

**Physical Meaning:**
- HSI camera is mounted 2.5 meters ahead of the IMU reference point
- In body frame coordinates
- This is the lever arm offset

---

## Transformation Pipeline Comparison

### test_eely Pipeline (via gref4hsi library)

```python
# In georeference.py: define_hsi_ray_geometry()
hsi_geometry.intrinsicTransformHSI(
    translation_ref_hsi=translation_body_to_hsi,  # [2.5, 0, 0]
    rot_hsi_ref_obj=rotation_hsi_to_body_obj      # [[0,1,0],[-1,0,0],[0,0,1]]
)

# In geometry_utils.py: intrinsicTransformHSI()
self.position_ecef = (
    self.position_nav_interpolated + 
    self.rotation_nav_interpolated.apply(translation_ref_hsi)
)
self.rotation_hsi = self.rotation_nav_interpolated * self.rotation_hsi_rgb
```

### Your Pipeline (x_gref4hsi_by_liu)

```python
# In main.py (line 76-80)
hsi_positions_ecef, hsi_orientations = utils.apply_sensor_transform(
    positions_ecef,
    orientations,
    config.ROTATION_HSI_TO_BODY,      # [[0,1,0],[-1,0,0],[0,0,1]]
    config.TRANSLATION_BODY_TO_HSI    # [2.5, 0, 0]
)

# In utils.py: apply_sensor_transform() (line 238-246)
R_body_to_world = orientations[i]
translation_world = R_body_to_world @ translation_vector
sensor_positions_ecef[i] = positions_ecef[i] + translation_world
sensor_orientations[i] = R_body_to_world @ rotation_matrix
```

✅ **MATCH** - Same mathematical operations:
1. Both rotate translation vector from body frame to world frame
2. Both add translated offset to vehicle position
3. Both compose rotations: `R_world = R_body_to_world @ R_sensor_to_body`

---

## Ray Direction Transformation

### test_eely (via gref4hsi)

```python
# In geometry_utils.py: defineRayDirections()
for i in range(n):
    self.rayDirectionsGlobal[i, :, :] = self.rotation_hsi[i].apply(dir_local)
```

### Your Code

```python
# In main.py (line 159-162)
R_camera_to_world = hsi_orientations[i]
ray_directions_world[idx_start:idx_end] = (
    R_camera_to_world @ ray_directions_camera.T
).T
```

✅ **MATCH** - Same operation:
- `rotation_hsi[i].apply(dir_local)` ≡ `R_camera_to_world @ ray_directions_camera`
- Both transform rays from camera frame to world (ECEF) frame using the composed rotation

---

## Complete Transformation Chain

Both implementations follow this identical sequence:

```
1. Vehicle Navigation (ECEF position + body orientation)
           ↓
2. Apply Body → HSI Transform
   - Position: pos_hsi = pos_body + R_body_to_world @ translation_body_to_hsi
   - Orientation: R_hsi_to_world = R_body_to_world @ R_hsi_to_body
           ↓
3. Build Ray Directions in Camera Frame
   - Using same formula: x_norm = (u-cx)/f - (k1*r^5+k2*r^3+k3*r^2)/f
   - Ray vectors: [x_norm, 0, 1]
           ↓
4. Transform Rays to World Frame
   - ray_world = R_hsi_to_world @ ray_camera
           ↓
5. Ray-Mesh Intersection (PyEmbree/Trimesh)
```

---

## Key Verification Points

| Component | test_eely | Your Code | Status |
|-----------|-----------|-----------|--------|
| Rotation matrix HSI→Body | `[[0,1,0],[-1,0,0],[0,0,1]]` | Same | ✅ |
| Translation Body→HSI | `[2.5, 0, 0]` | Same | ✅ |
| Position transform | `pos + R @ trans` | Same | ✅ |
| Orientation composition | `R_body @ R_sensor` | Same | ✅ |
| Ray direction formula | `(u-cx)/f - distortion` | Same | ✅ |
| Ray transform to world | `R @ ray_local` | Same | ✅ |
| Camera frame convention | X=across, Y=0, Z=down | Same | ✅ |

---

## Differences (Intentional)

The only difference between your pipeline and test_eely is **the intersection target**:

- **test_eely**: Uses altimeter-derived DEM (from altitude measurements)
- **Your code**: Uses MBES bathymetry mesh (underwater terrain)

This is an **intentional design choice** for your underwater application and does NOT affect the coordinate transformations.

---

## Conclusion

✅ **All body transformations and orientations match test_eely exactly**

Your implementation in `x_gref4hsi_by_liu` correctly applies:
1. The same rotation matrix `[[0,1,0],[-1,0,0],[0,0,1]]`
2. The same translation vector `[2.5, 0, 0]`
3. The same transformation sequence (position offset + rotation composition)
4. The same ray direction formula and transformation to world frame

The geometric pipeline is mathematically equivalent to test_eely. Any differences in final results would come from:
- The MBES mesh itself (not the transformations)
- Time synchronization between HSI and navigation
- Navigation data quality/interpolation

---

**Bottom Line:** Your code uses the correct body transformations matching test_eely! 🎯
