# Quick Reference: x_gref4hsi_by_liu Ray Tracing

## ✅ Status: FIXED - Now Matches test_eely

Your `x_gref4hsi_by_liu/utils.py::build_ray_directions()` now uses **the exact same method** as `gref4hsi/tests/test_eely.py`.

---

## What's the Same as test_eely

| Component | Your Code | test_eely | Status |
|-----------|-----------|-----------|--------|
| Pixel indexing | 1-based (`u = 1 to N`) | 1-based | ✅ Match |
| Distortion formula | `-(k1*r^5 + k2*r^3 + k3*r^2)/f` | Same | ✅ Match |
| Distortion scaling | `r = (u-cx)/1000` | Same | ✅ Match |
| X-axis | Across-track (`x_norm`) | Across-track | ✅ Match |
| Y-axis | Along-track (0) | Along-track (0) | ✅ Match |
| Z-axis | Down (1) | Down (1) | ✅ Match |
| Normalization | None (keep `[x_norm, 0, 1]`) | None | ✅ Match |

---

## The Fixed Formula

```python
# Pixel indices (1-based)
u = np.arange(1, num_pixels + 1)

# Linear projection
x_norm_lin = (u - cx) / f

# Nonlinear distortion (NEGATIVE sign!)
r = (u - cx) / 1000.0
x_norm_nonlin = -(k1 * r**5 + k2 * r**3 + k3 * r**2) / f

# Combined
x_norm = x_norm_lin + x_norm_nonlin

# Build rays [x_norm, 0, 1]
p_dir[:, 0] = x_norm  # X: across-track
p_dir[:, 1] = 0       # Y: along-track
p_dir[:, 2] = 1       # Z: down
```

---

## How to Test

```python
# Run your main pipeline
cd x_gref4hsi_by_liu
python main.py

# Check ray directions match test_eely:
# 1. Center pixel should be ~[0, 0, 1]
# 2. Edge pixels should fan out symmetrically
# 3. Ray-to-ground distance should match altimeter
```

---

## Key Files

- **Modified:** `x_gref4hsi_by_liu/utils.py` → `build_ray_directions()`
- **Unchanged:** `x_gref4hsi_by_liu/main.py` (already using it correctly)
- **Reference:** `gref4hsi/scripts/georeference.py` → `cal_file_to_rays()` (the original)

---

## Documentation

See detailed explanations in:
- `x_gref4hsi_by_liu/RAY_DIRECTION_FIX_SUMMARY.md` (what was fixed)
- `gref4hsi/custom_code_liu/RAYTRACING_AND_FOV_EXPLAINED.md` (how it works)
- `gref4hsi/custom_code_liu/RAYTRACING_COMPARISON_AND_FIXES.md` (comparison)

---

**Bottom Line:** Your ray tracing now uses the same method as test_eely. Run your pipeline and the results should be consistent with the official implementation! 🎯
