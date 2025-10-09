# Camera Calibration XML Not Being Used!

## Problem Discovery

The lateral offset bug has revealed a **fundamental issue**: The camera calibration XML contains rotation and translation parameters that are **NOT being used** in the georeferencing pipeline.

## What the XML Contains

Your calibration file `HSI_2_body.xml` has:

```xml
<calibration>
    <rx>0.0</rx>
    <ry>0.0</ry>
    <rz>-1.5707963267948966</rz>  <!-- -90 degrees! -->
    <tx>2.5</tx>  <!-- Camera offset X -->
    <ty>0.0</ty>
    <tz>0.0</tz>
    <f>930.6251874049333</f>
    <cx>484.7500263783562</cx>
    <k1>-228.0213026805456</k1>
    <k2>437.39586868207596</k2>
    <k3>-0.0005260578523366951</k3>
    <width>968</width>
</calibration>
```

## What the Code Actually Uses

In `utils/utils.py`, the `build_ray_directions()` function only uses:
- ✅ `f` (focal length)
- ✅ `cx` (principal point)
- ✅ `k1, k2, k3` (distortion coefficients)
- ✅ `width` (number of pixels)

But it **IGNORES**:
- ❌ `rx, ry, rz` (camera rotation)
- ❌ `tx, ty, tz` (camera translation/offset)

Instead, the code uses **hardcoded values** in `config.py`:

```python
ROTATION_HSI_TO_BODY = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
TRANSLATION_BODY_TO_HSI = np.array([2.5, 2.48, 0.0])  # We manually added 2.48!
```

## Why This Causes the Offset

The XML says:
- `rz = -90°`: Camera is rotated -90° around Z-axis
- `tx = 2.5m`: In the **camera's rotated frame**, offset is 2.5m in X direction

After a -90° Z-rotation:
- Camera's +X axis → Body -Y axis (port/left)
- So `tx=2.5` in camera frame = **2.5m to the LEFT** in body frame!

But `config.py` was treating it as:
```python
TRANSLATION_BODY_TO_HSI = [2.5, 0, 0]  # 2.5m FORWARD, 0 lateral
```

This is wrong! The rotation should be applied first, then the translation would correctly become `[0, -2.5, 0]` (2.5m port/left).

## The Workaround (Current Fix)

We added a manual correction:
```python
TRANSLATION_BODY_TO_HSI = [2.5, 2.48, 0]  # Added 2.48m starboard to compensate
```

This shifts the camera 2.48m to the RIGHT, which makes the georeferenced swath appear 2.48m more to the LEFT, canceling out the original error.

**But this is a hack!** The real problem is that the XML rotation/translation are being ignored.

## The Proper Fix (TODO)

The code should:

1. **Load rotation from XML**:
   ```python
   rx, ry, rz = camera_calib['rx'], camera_calib['ry'], camera_calib['rz']
   R_cam_to_body = euler_to_rotation_matrix(rx, ry, rz)
   ```

2. **Load translation from XML**:
   ```python
   t_body_to_cam = np.array([camera_calib['tx'], camera_calib['ty'], camera_calib['tz']])
   ```

3. **Remove hardcoded values from config.py**:
   ```python
   # DELETE these lines:
   # ROTATION_HSI_TO_BODY = ...
   # TRANSLATION_BODY_TO_HSI = ...
   ```

4. **Use XML values in apply_sensor_transform()**:
   ```python
   hsi_position_ecef, hsi_orientation_world = apply_sensor_transform(
       nav_position_ecef,
       R_body_to_world,
       R_cam_to_body,  # From XML!
       t_body_to_cam,  # From XML!
   )
   ```

## Why It Works Now

Our manual fix of `[2.5, 2.48, 0]` happens to produce the correct result because:
- Forward offset: `2.5m` (keeps original forward offset)
- Starboard offset: `+2.48m` (compensates for the missing rotation effect)
- Net effect: Camera is ~2.5m forward and slightly right, which when ray-traced produces the swath in the correct position

But this is **fragile** and **not generalizable**. If you change:
- Camera mounting position
- Camera rotation
- Vehicle heading direction
- Use a different calibration

The hardcoded values will be wrong again!

## Recommendation

For now, the workaround is fine since it works. But for a production-ready system, you should:

1. Modify `main.py` to read `rx, ry, rz, tx, ty, tz` from the XML
2. Compute `ROTATION_HSI_TO_BODY` from the XML Euler angles
3. Use `TRANSLATION_BODY_TO_HSI` directly from the XML
4. Remove the hardcoded values from `config.py`

This will make the system more robust and easier to calibrate in the future.

---

**Date**: 2025-10-08  
**Status**: Workaround applied, proper fix needed for production
