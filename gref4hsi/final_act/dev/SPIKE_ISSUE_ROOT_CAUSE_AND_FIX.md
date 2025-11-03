# Root Cause Analysis: Spike Reappearance Issue

## Problem Summary
Spectral spikes at ~490nm, ~565nm, and ~620nm reappeared after training, despite applying `apply_wavelength_interpolation()`.

## Root Cause Identified ✅

**The spikes never disappeared in the first place!**

### Why the Interpolation Failed:

1. **Original approach**:
   ```python
   # Step 1: Interpolate on 210-band cube at indices [64, 107, 138]
   cube.apply_wavelength_interpolation([64, 107, 138])
   
   # Step 2: Crop to 490-680nm (creates 108-band cube)
   cube.apply_wavelength_filter((490, 680))
   ```

2. **The problem**:
   - Indices `[64, 107, 138]` correspond to wavelengths in the ORIGINAL 210-band cube
   - But the spike at **~490nm is at the EDGE of the cropping range**!
   - When you crop to `490-680nm`, you **include the 490nm spike**
   - After cropping, the 490nm spike is now at **index 0** (the first wavelength!)
   - The 565nm and 620nm spikes are at **different indices** in the cropped cube

3. **Result**:
   - The interpolation fixed indices [64, 107, 138] in the 210-band cube
   - But cropping to 490-680nm brought back the 490nm spike at the start
   - The spikes at 565nm and 620nm are still present at their NEW indices

### Diagnostic Evidence:

From your test output:
```
📍 Testing 5 pixels from training_bombs ROI...
  Pixel 1: first/mid=1.243, last/mid=0.845  ⚠️ SPIKES DETECTED!
  Pixel 2: first/mid=1.420, last/mid=0.821  ⚠️ SPIKES DETECTED!
  ...
```

The `first/mid` ratios of 1.24-1.43 indicate **spikes at the beginning of the spectrum** (index 0-4), which is exactly where 490nm is after cropping!

## The Fix ✅

**Interpolate AFTER cropping**, using the correct indices in the cropped cube:

```python
# Step 1: Interpolate on original 210-band cube
cube.apply_wavelength_interpolation([64, 107, 138])

# Step 2: Crop to 490-680nm
cube.apply_wavelength_filter((490, 680))

# Step 3: RE-INTERPOLATE on the cropped cube! ⭐ NEW STEP
# Find indices of ~490, 565, 620 nm in the CROPPED 108-band cube
target_wavelengths_nm = [490, 565, 620]
bad_indices_cropped = []

for target_wl in target_wavelengths_nm:
    idx = np.argmin(np.abs(cube.wavelengths - target_wl))
    actual_wl = cube.wavelengths[idx]
    if abs(actual_wl - target_wl) < 3:  # Within 3nm
        bad_indices_cropped.append(idx)

# Interpolate at the correct indices in the cropped cube
cube.apply_wavelength_interpolation(interpolate_wavelengths=bad_indices_cropped)
```

## Alternative Solutions:

### Option 1: Crop to exclude the 490nm spike
```python
# Crop to 495-680nm instead of 490-680nm
cube.apply_wavelength_filter((495, 680))
```

### Option 2: Interpolate only after cropping
```python
# Skip the first interpolation, do it all after cropping
cube.apply_wavelength_filter((490, 680))
cube.apply_wavelength_interpolation(bad_indices_cropped)  # Calculated indices
```

### Option 3: Use wavelength-based interpolation (future enhancement)
Modify `apply_wavelength_interpolation()` to accept wavelength values instead of indices:
```python
# Hypothetical improved API
cube.apply_wavelength_interpolation(wavelengths_nm=[490, 565, 620])
```

## Why Training Didn't Cause The Issue:

The training function (`train_svm_with_cv()`) **did NOT reload or modify the data**. The diagnostic showed:
- ✅ Wavelengths unchanged (same 108 bands, same object ID)
- ✅ No reloading from disk
- ❌ Spikes were already present BEFORE training

The training function simply exposed the pre-existing spikes that were never properly removed.

## Implementation Status:

✅ **Fix implemented** in cells 24-26 of the notebook:
- Cell 24: Diagnostic to find bad wavelength indices in cropped cube
- Cell 25: Re-interpolate at correct indices after cropping  
- Cell 26: Verify spikes are removed

## Testing:

Run these cells in order:
1. Run cells 1-23 (setup, load, first interpolation, crop)
2. Run cells 24-26 (re-interpolation fix)
3. Check cell 26 output - should show "✅ SUCCESS: All pixels spike-free!"
4. Run cell 38 (spike detection test) - should show "✅ PASS: Data appears spike-free"

## Lessons Learned:

1. **Index-based operations are fragile** when followed by cropping/filtering
2. **Always verify** that preprocessing worked (don't assume!)
3. **Spikes at range boundaries** are tricky - they can survive cropping
4. **Wavelength-based APIs** would be more robust than index-based

---

**Status**: Fix implemented and ready for testing ✅
