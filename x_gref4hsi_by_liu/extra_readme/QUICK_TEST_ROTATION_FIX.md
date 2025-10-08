# Quick Test: Verify Rotation Fix Works

## Before Running main.py

Run these quick checks to verify the fixes are working:

### 1. Check config.py has correct values

```powershell
cd x_gref4hsi_by_liu
python -c "import config; import numpy as np; print('Rotation matrix:'); print(config.ROTATION_HSI_TO_BODY); print('Yaw convention:', config.YAW_CONVENTION)"
```

**Expected output:**
```
Rotation matrix:
[[ 0.  1.  0.]
 [ 1.  0.  0.]
 [ 0.  0. -1.]]
Yaw convention: heading_from_north_cw
```

### 2. Run simulation to verify rays point down

```powershell
cd dev
python dev1_sim_v5_redo_rotation.py --frames 50
```

**Look for in console:**
- ✅ "✓ Simulation now uses SAME rotation matrix as main.py!"
- ✅ Rotation matrix printed should match above

**In visualization:**
- ✅ Rays should NOT be red (red = pointing up, wrong!)
- ✅ Rays should point downward toward MBES surface

### 3. Test main.py on one H5 file (dry run)

```powershell
cd x_gref4hsi_by_liu
```

Edit `main.py` temporarily to process only 1 file with first 50 frames:
```python
# In main() function, after loading h5_files:
h5_files = h5_files[:1]  # Only first file

# In process_h5_file(), after loading timestamps:
hsi_timestamps = hsi_timestamps[:50]  # Only first 50 frames
```

Then run:
```powershell
python main.py
```

**Expected console output should include:**
```
Converting attitude to rotation matrices...
Transforming ray directions to world frame...
Prepared 502,250 rays (50 frames × 10045 slits)
Mean boresight·up = -0.9234 (should be negative, pointing down)  ← CHECK THIS!
```

**✅ If `Mean boresight·up` is NEGATIVE → Fix is working!**
**❌ If `Mean boresight·up` is POSITIVE → Something still wrong!**

### 4. Quick sanity checks

After processing completes:

```python
# Open the output H5 file and check intersection points
import h5py
import numpy as np

with h5py.File("output/your_file.h5", "r") as f:
    pts = f["georeferencing/intersection_points"][:]
    
    # Check if points are in reasonable UTM range
    print(f"X range: {pts[:,0].min():.1f} to {pts[:,0].max():.1f}")  # Should be ~100k-800k
    print(f"Y range: {pts[:,1].min():.1f} to {pts[:,1].max():.1f}")  # Should be ~6M-7M range
    print(f"Z range: {pts[:,2].min():.1f} to {pts[:,2].max():.1f}")  # Should match MBES depth
```

---

## Red Flags to Watch For

### 🚩 Problem: Rays showing as RED in simulation
**Cause**: Rays pointing upward instead of downward
**Fix**: Check that config.ROTATION_HSI_TO_BODY is correct (not identity!)

### 🚩 Problem: `Mean boresight·up` is POSITIVE
**Cause**: Yaw convention wrong or rotation matrix inverted
**Fix**: Double-check config.YAW_CONVENTION and rotation matrix

### 🚩 Problem: Low intersection success rate (<50%)
**Cause**: Rays not hitting MBES mesh
**Possible causes**:
- MBES mesh not loaded correctly
- Coordinate system mismatch (check EPSG codes)
- Time synchronization wrong (TIME_OFFSET_SEC)

### 🚩 Problem: Intersection points way off
**Cause**: Something still wrong with transformations
**Debug**:
```python
# In main.py after computing ray_directions_world:
print("Sample ray directions (first 5, center pixel):")
center_idx = n_rays_per_frame // 2
for i in range(5):
    ray_idx = i * n_rays_per_frame + center_idx
    print(f"  Frame {i}: {ray_directions_world[ray_idx]}")
# All Z components should be NEGATIVE (pointing down in ENU)
```

---

## All Good? Proceed to Full Processing

If all checks pass:
1. ✅ Remove the test limits from main.py
2. ✅ Run full processing on all H5 files
3. ✅ Monitor console for reasonable success rates
4. ✅ Spot-check a few output H5 files for sensible coordinates

---

## Emergency Rollback

If something goes wrong, the OLD (broken) values were:

```python
# config.py OLD VALUES (DO NOT USE!)
ROTATION_HSI_TO_BODY = np.eye(3)  # WRONG!
# (no YAW_CONVENTION)

# utils.py: use old euler_to_rotation_matrix() function
# main.py Step 4: use old function call
```

But you really shouldn't need to roll back - the new approach is proven correct!
