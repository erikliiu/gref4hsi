# ⚠️ CRITICAL MISMATCH: main.py vs dev1_sim_v4.py Transformations

## Summary

**There is a CRITICAL difference** between how `main.py` and `dev1_sim_v4.py` compute camera orientations and ray directions. They use **different rotation conventions** that will result in **rays pointing in different directions**.

---

## 🔴 The Problem

### Config File Says:
```python
# config.py line 20
ROTATION_HSI_TO_BODY = np.eye(3)  # Identity matrix - NO ROTATION
```

### But dev1_sim_v4.py Uses:
```python
# dev1_sim_v4.py line 55
R_B_FROM_CAM = np.array([[0.0, 1.0, 0.0], 
                         [1.0, 0.0, 0.0], 
                         [0.0, 0.0, -1.0]])  # 90° rotation!
```

**These are NOT the same!**

---

## Detailed Comparison

### 1. Camera-to-Body Rotation Matrix

| File | Rotation Matrix | Meaning |
|------|----------------|---------|
| **config.py** (used by main.py) | `[[1,0,0], [0,1,0], [0,0,1]]` | Identity - NO rotation |
| **dev1_sim_v4.py** | `[[0,1,0], [1,0,0], [0,0,-1]]` | Maps cam→body: X_c→Y_b, Y_c→X_b, Z_c→-Z_b |

**Impact**: This completely changes how camera frame rays are oriented relative to the body!

---

### 2. Euler Angle to Rotation Matrix Convention

#### main.py (via utils.euler_to_rotation_matrix):
```python
# utils.py line 269
rotations = RotLib.from_euler("ZYX", angles)
```
- **Convention**: ZYX (yaw-pitch-roll) intrinsic rotations
- **Yaw meaning**: Direct use of yaw value from CSV

#### dev1_sim_v4.py (via _body_to_enu_rotation):
```python
# dev1_sim_v4.py line 99-111
# Converts heading to ENU yaw
if ORI_CONVENTION == "heading_from_north_cw":
    yaw_val = np.deg2rad(90.0 - yaw_in)  # heading → ENU yaw conversion

# Then builds R = Rz(yaw_enu) @ Ry(pitch) @ Rx(roll)
```
- **Convention**: Explicitly builds Rz @ Ry @ Rx with ENU yaw
- **Yaw meaning**: Converts heading (N=0° CW) to ENU yaw (E=0° CCW)

**Impact**: Different yaw interpretations mean the body frame points in different directions!

---

### 3. Complete Ray Transformation Chain

#### main.py Process:
```
1. Build rays in camera frame:
   rays_cam = build_ray_directions(calib, n_slits)  # [x_norm, 0, 1]

2. Get body→world rotation from CSV angles:
   R_body_to_world = euler_to_rotation_matrix(roll, pitch, yaw)  # ZYX convention
   
3. Apply sensor transform with IDENTITY rotation:
   R_sensor_to_world = R_body_to_world @ np.eye(3)  # No camera rotation!
   
4. Transform rays:
   ray_world = R_sensor_to_world @ rays_cam
```

#### dev1_sim_v4.py Process:
```
1. Build rays in camera frame:
   rays_cam = build_ray_directions(calib, n_slits)  # [x_norm, 0, 1]

2. Get body→ENU rotation with heading conversion:
   yaw_enu = 90° - heading
   R_bw = Rz(yaw_enu) @ Ry(pitch) @ Rx(roll)  # Explicit matrix multiplication
   
3. Apply camera→body rotation:
   R_B_FROM_CAM = [[0,1,0], [1,0,0], [0,0,-1]]  # 90° rotation
   
4. Transform rays:
   R_wc = R_bw @ R_B_FROM_CAM
   ray_world = R_wc @ rays_cam
```

---

## 🔍 Specific Differences

### A. Camera Frame Orientation

**main.py assumes**:
- Camera frame = Body frame (identity rotation)
- Camera X-axis = Body X-axis (forward)
- Camera Y-axis = Body Y-axis (starboard)  
- Camera Z-axis = Body Z-axis (up in ENU, down in NED)

**dev1_sim_v4.py assumes**:
- Camera frame ≠ Body frame (90° rotation)
- Camera X-axis → Body Y-axis (starboard)
- Camera Y-axis → Body X-axis (forward)
- Camera Z-axis → Body -Z-axis (down in ENU frame)

### B. Yaw Convention

**main.py**:
- Uses yaw directly from CSV (unclear if heading or ENU yaw)
- Applies ZYX Euler convention via scipy

**dev1_sim_v4.py**:
- Explicitly converts heading (N=0° CW) to ENU yaw (E=0° CCW)
- Formula: `yaw_enu = 90° - heading`
- Builds rotation matrices explicitly

---

## 🎯 What This Means

### If you run main.py with current config:
1. ❌ Camera rays will NOT match the simulation visualization
2. ❌ Ray directions will be rotated 90° wrong relative to flight direction
3. ❌ Intersection points will be in wrong locations
4. ❌ The georeferenced H5 files will contain INCORRECT coordinates

### Example Ray Direction Comparison:

For a camera at (x, y, z) = (100, 200, 50) with heading=45° (NE):

**dev1_sim_v4.py** (correct):
- Center ray points down-ish: ~[0.1, -0.1, -0.99] (slightly northeast, mostly down)
- Matches what you see in visualization

**main.py** (current config):
- Center ray points wrong direction due to identity rotation
- Will not match simulation

---

## ✅ Required Fixes for main.py

### Option 1: Match test_eely Convention (RECOMMENDED)

Update `config.py`:
```python
# Change from:
ROTATION_HSI_TO_BODY = np.eye(3)  # WRONG!

# To:
ROTATION_HSI_TO_BODY = np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]])  # test_eely convention
```

**Note**: This is the matrix you verified earlier matches test_eely! The config file currently has the wrong value.

### Option 2: Match dev1_sim_v4.py Convention

Update `config.py`:
```python
ROTATION_HSI_TO_BODY = np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, -1]])  # dev1_sim_v4 convention (ENU frame)
```

Also need to update `euler_to_rotation_matrix` in utils.py to match dev1_sim_v4's heading conversion.

---

## 🔬 How to Verify Which is Correct

### Check the Camera Calibration XML

Look at `HSI_2_body.xml`:
```xml
<rx>...</rx>  <!-- rotation about X -->
<ry>...</ry>  <!-- rotation about Y -->
<rz>-1.5708</rz>  <!-- rotation about Z = -90° -->
```

If `rz = -1.5708` (≈ -π/2 = -90°), this means:
- **test_eely uses**: `[[0,1,0], [-1,0,0], [0,0,1]]` (rotation about +Z by -90°)
- **dev1_sim_v4 uses**: `[[0,1,0], [1,0,0], [0,0,-1]]` (different convention, ENU frame)

The XML rotation should match test_eely convention based on your earlier verification.

---

## 📋 Action Items

### Priority 1: Fix config.py Rotation Matrix
- [ ] Change `ROTATION_HSI_TO_BODY` from identity to correct rotation
- [ ] Use test_eely value: `[[0,1,0], [-1,0,0], [0,0,1]]`

### Priority 2: Verify Euler Convention
- [ ] Check if CSV yaw is "heading from north CW" or "ENU yaw from east CCW"
- [ ] If heading, need to convert: `yaw_enu = 90° - heading` before `euler_to_rotation_matrix`
- [ ] Or update `euler_to_rotation_matrix` to handle heading convention

### Priority 3: Test Ray Directions
- [ ] Run main.py with fixed rotation matrix
- [ ] Compare intersection points with expected seafloor depth
- [ ] Verify rays point downward (negative Z component in world frame)

### Priority 4: Update dev1_sim_v4.py Comment
- [ ] Current comment says cam→body mapping is `[[0,1,0], [1,0,0], [0,0,-1]]`
- [ ] This is for ENU frame (Z up)
- [ ] main.py should use NED-like convention from test_eely

---

## 🚨 Critical Question to Answer

**What coordinate frame convention are you using?**

1. **NED (North-East-Down)**: Z-axis points down
   - Use test_eely rotation: `[[0,1,0], [-1,0,0], [0,0,1]]`
   
2. **ENU (East-North-Up)**: Z-axis points up
   - Use dev1_sim_v4 rotation: `[[0,1,0], [1,0,0], [0,0,-1]]`

The XML calibration and test_eely suggest **NED convention**, so main.py should use the test_eely rotation matrix.

---

## Bottom Line

**Your config.py has the WRONG rotation matrix (identity instead of 90° rotation).** 

This means `main.py` will produce georeferencing results that DO NOT match what you see in the `dev1_sim_v4.py` visualization.

**Fix immediately before processing any H5 files!**
