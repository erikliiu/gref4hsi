# Comparison: x_gref4hsi_by_liu vs test_eely Raytracing

## Critical Differences Found

### 1. ❌ **RAY DIRECTION CONSTRUCTION - MAJOR ISSUE**

**test_eely (CORRECT):**
```python
# In cal_file_to_rays() - gref4hsi/scripts/georeference.py
u = np.arange(1, n_pix + 1)  # Pixel indices start at 1

x_norm_lin = (u - u_c) / f

x_norm_nonlin = (
    -(k1 * ((u - u_c) / 1000) ** 5
      + k2 * ((u - u_c) / 1000) ** 3
      + k3 * ((u - u_c) / 1000) ** 2)
    / f
)

x_norm = x_norm_lin + x_norm_nonlin

p_dir = np.zeros((len(x_norm), 3))
p_dir[:, 0] = x_norm  # X axis (across-track)
p_dir[:, 2] = 1       # Z axis (down)
# p_dir[:, 1] = 0     # Y axis (along-track) implicitly zero
```

**x_gref4hsi_by_liu (WRONG):**
```python
# In build_ray_directions() - utils.py
u = np.arange(int(num_pixels), dtype=float)  # ❌ Starts at 0, not 1

x = (u - cx) / f
r2 = x * x
x = x * (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)  # ❌ Wrong distortion formula

rays[:, 1] = -x   # ❌ Wrong axis (Y instead of X)
rays[:, 2] = 1.0  # ✓ Correct (Z down)
```

### Problems:
1. **Pixel indexing:** Starting at 0 vs 1 causes a half-pixel shift in all ray directions
2. **Distortion formula:** test_eely uses **negative** distortion with division by 1000 for scaling
3. **Axis assignment:** test_eely puts across-track on **X-axis**, your code uses **Y-axis**
4. **Normalization:** test_eely does NOT normalize rays in camera frame (kept as [x_norm, 0, 1])

---

### 2. ❌ **DISTORTION MODEL - COMPLETELY DIFFERENT**

**test_eely formula:**
```
x_norm_nonlin = -(k1*r^5 + k2*r^3 + k3*r^2) / f
where r = (u - u_c) / 1000
```

**Your formula:**
```
x = x * (1 + k1*r^2 + k2*r^4 + k3*r^6)
where r^2 = x^2
```

These are **incompatible**:
- test_eely: Uses **5th, 3rd, 2nd order** terms
- You: Uses **2nd, 4th, 6th order** terms (Brown-Conrady standard model)
- test_eely: **Negative** sign and scales by `/f`
- You: **Positive** sign (multiplicative correction)
- test_eely: Scales input by `/1000` before computing powers
- You: No scaling

---

### 3. ❌ **CAMERA FRAME AXIS CONVENTION**

**test_eely:**
```
p_dir[:, 0] = x_norm  # Across-track on X
p_dir[:, 1] = 0       # Along-track on Y (flight direction)
p_dir[:, 2] = 1       # Down on Z (boresight)
```

**Your code:**
```
rays[:, 0] = 0        # Along-track on X
rays[:, 1] = -x       # Across-track on Y  
rays[:, 2] = 1        # Down on Z
```

**This is a 90° rotation difference in camera frame conventions!**

---

### 4. ⚠️ **NORMALIZATION TIMING**

**test_eely:**
- Rays in camera frame are **NOT normalized**: `[x_norm, 0, 1]`
- Normalization likely happens later during transformation or in ray-mesh intersection

**Your code:**
- Normalizes immediately: `rays /= np.linalg.norm(rays, axis=1, keepdims=True)`
- This is fine but different timing

---

### 5. ⚠️ **EXTRINSIC PARAMETERS (ROTATION/TRANSLATION)**

**test_eely approach:**
```python
# Reads rotation from XML calibration file
rot_x, rot_y, rot_z = from XML
rot_hsi_ref_eul = np.array([rot_z, rot_y, rot_x])
rot_hsi_ref_obj = RotLib.from_euler(seq="ZYX", angles=rot_hsi_ref_eul, degrees=False)

# Reads translation from XML
translation_ref_hsi = np.array([trans_x, trans_y, trans_z])
```

**Your approach:**
```python
# Hardcoded in config.py
ROTATION_HSI_TO_BODY = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
TRANSLATION_BODY_TO_HSI = np.array([2.5, 0, 0])
```

**This is OK** if your hardcoded values match the XML, but test_eely is more flexible.

---

## Summary of Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| Pixel indexing (0 vs 1) | HIGH | ~0.5 pixel systematic error in all intersections |
| Distortion formula | CRITICAL | Completely wrong ray directions (wrong powers, sign, scaling) |
| Axis convention (X vs Y) | CRITICAL | 90° rotation error in camera frame |
| Normalization timing | LOW | Probably OK, just different |
| Extrinsic parameter source | LOW | OK if hardcoded values match XML |

---

## What Needs to be Fixed

### Priority 1: Ray Direction Construction

Replace your `build_ray_directions()` function with test_eely's exact formula:

```python
def build_ray_directions(camera_calib, num_pixels):
    """
    Build ray directions in camera frame following test_eely convention.
    
    Camera frame:
      X: across-track (perpendicular to flight)
      Y: along-track (flight direction)  
      Z: down (nadir/boresight)
    
    Returns: (num_pixels, 3) array of ray directions [x_norm, 0, 1]
    """
    f = float(camera_calib['f'])
    cx = float(camera_calib['cx'])
    k1 = float(camera_calib['k1'])
    k2 = float(camera_calib['k2'])
    k3 = float(camera_calib['k3'])
    
    # Pixel indices (1-based, like test_eely)
    u = np.arange(1, num_pixels + 1)
    
    # Linear component
    x_norm_lin = (u - cx) / f
    
    # Nonlinear distortion component (negative sign!)
    r = (u - cx) / 1000.0  # Scale factor for numerical stability
    x_norm_nonlin = -(k1 * r**5 + k2 * r**3 + k3 * r**2) / f
    
    # Total normalized x-coordinate
    x_norm = x_norm_lin + x_norm_nonlin
    
    # Build ray directions
    p_dir = np.zeros((len(x_norm), 3))
    p_dir[:, 0] = x_norm  # Across-track on X
    p_dir[:, 1] = 0       # Along-track on Y (always zero for pushbroom)
    p_dir[:, 2] = 1       # Down on Z (boresight)
    
    # DO NOT normalize here - test_eely keeps them as [x_norm, 0, 1]
    # Normalization happens during transformation or intersection
    
    return p_dir
```

### Priority 2: Verify Extrinsic Parameters

Check that your hardcoded rotation/translation match the XML calibration:

```bash
# Look at your XML file to verify
cat E:\mjosa_new_oct_2025\use_gref4hsi\057_own_code_1to2\input\Calib\HSI_2b.xml
```

Compare:
- XML `<rx>, <ry>, <rz>` → should match your rotation matrix
- XML `<tx>, <ty>, <tz>` → should match `[2.5, 0, 0]`

### Priority 3: FOV Consistency

Your code currently doesn't use FOV from H5 files. test_eely does:

```python
# test_eely extracts FOV from H5
fov_arr = hyp.fov  # From H5: processed/radiance/calibration/geometric/fieldOfView
param_dict = Specim.fov_2_param(fov=fov_arr)  # Fits camera model to FOV
```

**Recommendation:** Either:
1. Use FOV from H5 files (like test_eely), OR
2. Verify your XML calibration matches the H5 FOV data

---

## Testing Strategy

After fixing the ray direction formula:

1. **Visual test:** Plot ray directions for center pixel and edge pixels
   - Center pixel should give `[~0, 0, 1]` (nearly straight down)
   - Edge pixels should fan out symmetrically

2. **Comparison test:** Load same H5 file in both pipelines and compare:
   - Ray directions for pixel 512 (center)
   - Intersection points for a few test frames
   
3. **Sanity check:** After raytracing, check:
   - Camera-to-ground distance should match altimeter reading
   - Footprint should be reasonable (not way too wide/narrow)

---

## Code Fix Summary

**Files to modify:**
1. `x_gref4hsi_by_liu/utils.py` → Replace `build_ray_directions()` function
2. `x_gref4hsi_by_liu/config.py` → Verify rotation/translation match XML

**Critical changes:**
- Start pixel indexing at 1: `u = np.arange(1, num_pixels + 1)`
- Use test_eely distortion formula: `-(k1*r^5 + k2*r^3 + k3*r^2) / f` where `r = (u-cx)/1000`
- Put across-track on X-axis: `p_dir[:, 0] = x_norm`
- Keep Y-axis zero: `p_dir[:, 1] = 0`
- Do NOT normalize in camera frame

---

**Impact:** These fixes will ensure your raytracing uses the **exact same** camera model and FOV interpretation as test_eely, giving consistent georeferencing results.
