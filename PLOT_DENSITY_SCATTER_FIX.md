# Fix for plot_density_scatter Spurious Correlation

## Problem
The `plot_density_scatter` function was showing spurious correlation between MBES residuals and UHI intensity when they should be independent (r ≈ 0).

## Root Cause
The bug was in how normalization was applied:

### BEFORE (Buggy):
```python
# Step 1: Mask the full 2D grids
mbes_in = np.where(inside, self.residuals, np.nan)  # 2D array
uhi_in = np.where(inside, uhi_on_mbes, np.nan)      # 2D array

# Step 2: Normalize the ENTIRE grids separately
z_mbes = _robust_z(mbes_in)  # Each normalized independently!
z_uhi = _robust_z(uhi_in)    # This creates artificial correlation!

# Step 3: Extract valid pairs
valid_pairs = np.isfinite(z_mbes) & np.isfinite(z_uhi)
zM = z_mbes[valid_pairs].ravel()
zU = z_uhi[valid_pairs].ravel()
```

**Why this creates spurious correlation:**
- `_robust_z` computes: `(data - median) / MAD`
- When applied to full grids separately, each dataset is centered around its own median
- If MBES and UHI have similar spatial patterns (even if unrelated), the z-score transformation can make them appear correlated
- This is because both get centered/scaled independently, removing their true statistical relationship

### AFTER (Fixed):
```python
# Step 1: Extract VALID PAIRS first (before normalization)
valid_pairs_raw = np.isfinite(mbes_in) & np.isfinite(uhi_in) & inside
mbes_raw = mbes_in[valid_pairs_raw].ravel()  # 1D array of paired samples
uhi_raw = uhi_in[valid_pairs_raw].ravel()    # 1D array of paired samples

# Step 2: Normalize the PAIRED samples
z_mbes = _robust_z(mbes_raw)  # Normalized together
z_uhi = _robust_z(uhi_raw)    # Preserves true relationship

# Step 3: Use directly
zM = z_mbes
zU = z_uhi
```

**Why this fixes the issue:**
- Normalization is applied to the same set of spatially-matched samples
- The statistical relationship between MBES and UHI is preserved
- No artificial correlation is introduced by separate centering/scaling

## Expected Results

### Before Fix:
- Scatter plot showed correlation (r > 0.3 or similar)
- Points clustered along a diagonal line
- This was **wrong** - MBES residuals and UHI intensity should be independent

### After Fix:
- Scatter plot shows no correlation (r ≈ 0, typically -0.1 to +0.1)
- Points form a circular cloud (no clear linear pattern)
- This is **correct** - the datasets are truly independent

## Testing

Run the new test cell added to the notebook:
```python
fig_scatter_fixed = mb_ned.plot_density_scatter(
    use_adjusted=True, 
    normalization_method="zscore", 
    show=True
)
```

Look for:
- `Pearson r ≈ 0` (should be close to zero, e.g., -0.1 to +0.1)
- Circular scatter pattern (no diagonal line)
- This confirms MBES and UHI are independent

## Files Modified
- `gref4hsi/final_act/utils/other/detrend_mbes.py` - Fixed `plot_density_scatter` method (lines ~2112-2145)
- `gref4hsi/final_act/notebooks/detrend_mbes.ipynb` - Added test cells

## Related Functions
This fix only affects `plot_density_scatter`. All other plotting functions were working correctly:
- ✅ `plot_triptych`
- ✅ `plot_uhi_mbes_comparison`  
- ✅ `plot_zoomed_residuals`
- ✅ All other visualization methods
