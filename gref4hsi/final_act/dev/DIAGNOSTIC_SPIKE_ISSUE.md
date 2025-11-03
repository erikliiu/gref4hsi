# Investigation: Wavelength Interpolation Spike Reappearance Issue

## Problem Description
After running `apply_wavelength_interpolation()` to fix spikes at bad wavelengths, the spikes reappear after calling `train_svm_with_cv()`.

## Initial Analysis

### Suspected Root Causes:
1. **Data reloading**: Training function might reload original data from HDF5 files
2. **Reference vs Copy issue**: Data might be getting reset through shared references
3. **Method calls**: Some method within training might call `apply_illumination_correction_v2()` which reloads data from disk

### Key Code Locations:

**Preprocessing functions** (`georef.py`):
- `apply_wavelength_interpolation()` (line ~3125): Modifies `self.data_corrected` in-place
- `apply_wavelength_filter()` (line ~3205): Creates NEW `self.data_corrected` with `.copy()`
- `apply_illumination_correction_v2()` (line ~2720): **RELOADS data from HDF5 files!**

**Training function** (`georef.py` line ~6880):
- Gets reference to `self.data_corrected` (line ~7003)
- Stores wavelength info in `self.svm_wavelength_indices` and `self.svm_wavelengths`
- Does NOT appear to call any reloading functions

## Diagnostic Cells Added to Notebook

I've added several diagnostic cells to the notebook to identify exactly when and how the data gets corrupted:

### 1. Pre-training checkpoint (after cell 31)
- Saves state of `data_corrected` and `wavelengths` before training
- Records array IDs, shapes, and hash of data

### 2. Pre-training ROI save (before training cell)
- Saves copy of original ROIs for comparison

### 3. Post-training diagnostics (after cell 35)
- Detailed wavelength state check
- Data integrity check looking for spikes
- Verification of checkpoint

### 4. Spike detection test function
- Tests random pixels for presence of edge spikes
- Reports pass/fail

## Next Steps

### For You (User):
1. **Run the notebook with "Run All"**
2. **Check the diagnostic outputs**, especially:
   - Does the checkpoint verification show any changes?
   - Does the spike detection test fail after training?
   - Are the array IDs different (indicating reassignment)?
3. **Report back with the diagnostic output**

### Potential Fixes (to be implemented after diagnostics):

#### Fix Option 1: Protect preprocessed data
Make arrays read-only after preprocessing to catch any accidental modification:
```python
cube.data_corrected.flags.writeable = False
cube.wavelengths.flags.writeable = False
```

#### Fix Option 2: Deep copy in training
Modify `train_svm_with_cv()` to create a deep copy of data:
```python
cube_data = self.data_corrected.copy()  # Force copy, not reference
```

#### Fix Option 3: Add safety checks
Add verification before/after training:
```python
assert cube.data_corrected.shape[2] == 108, "Data was reset to original!"
```

#### Fix Option 4: Cache preprocessed state
Save preprocessed data to a separate attribute:
```python
self.data_preprocessed = self.data_corrected.copy()
# Always use data_preprocessed after preprocessing
```

## Files Created:
- `test_fix_wavelength_persistence.py`: Helper functions for checkpoint/verification
- This diagnostic document

## Questions to Answer:
1. ✅ When exactly do spikes reappear? (After training call)
2. ❓ Is `data_corrected` being reassigned? (Check array ID)
3. ❓ Is data being reloaded from disk? (Check hash)
4. ❓ Are wavelengths being reset? (Check length/values)
5. ❓ Does training call any reloading functions? (Needs trace)

---

**Status**: Awaiting diagnostic output from user to confirm root cause and implement targeted fix.
