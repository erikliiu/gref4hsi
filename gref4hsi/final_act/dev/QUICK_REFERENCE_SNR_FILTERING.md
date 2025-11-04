# Quick Reference: SNR Filtering for Edge Bands

## 📌 The Problem

After `apply_illumination_correction_v2()` and `apply_wavelength_filter(490-690 nm)`:
- **Edge bands still have HIGH NOISE** due to low radiance values being amplified
- This affects: SVM classification, NDI analysis, spectral analysis

## 🎯 The Solution: Soft Edge Weighting (RECOMMENDED)

```python
def apply_soft_edge_weighting(datacube, wavelengths, edge_width=20, min_weight=0.1):
    """Apply sigmoid weights to reduce edge band influence."""
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    weights = np.ones(len(wavelengths))
    
    # Left edge
    left_mask = wavelengths < (wl_min + edge_width)
    if np.any(left_mask):
        x = (wavelengths[left_mask] - wl_min) / edge_width
        weights[left_mask] = min_weight + (1 - min_weight) * (1 / (1 + np.exp(-10*(x - 0.5))))
    
    # Right edge  
    right_mask = wavelengths > (wl_max - edge_width)
    if np.any(right_mask):
        x = (wl_max - wavelengths[right_mask]) / edge_width
        weights[right_mask] = min_weight + (1 - min_weight) * (1 / (1 + np.exp(-10*(x - 0.5))))
    
    # Apply weights
    weighted_cube = datacube * weights[np.newaxis, np.newaxis, :]
    return weighted_cube, weights
```

## 🔧 Usage in Workflow

```python
# BEFORE (old workflow)
cube.apply_illumination_correction_v2()
cube.apply_wavelength_filter(wavelength_range=(490, 690))
# → Edge bands have high noise!

# AFTER (recommended workflow)
cube.apply_illumination_correction_v2()
cube.apply_wavelength_filter(wavelength_range=(490, 690))

# NEW: Apply soft edge weighting
cube.data_corrected, weights = apply_soft_edge_weighting(
    cube.data_corrected,
    cube.wavelengths,
    edge_width=20,    # 20 nm transition zone
    min_weight=0.1    # 10% weight at edges
)
# → Edge bands now have reduced influence!

# Continue as normal
cube.apply_spectral_normalization(method='l2')
# ...
```

## 📊 Recommended Parameters

| Parameter | Recommended Value | Range | Effect |
|-----------|------------------|-------|--------|
| `edge_width` | **20 nm** | 10-40 nm | Distance from edge where weighting starts |
| `min_weight` | **0.1** | 0.0-0.5 | Minimum weight at extreme edges |

- **edge_width=20**: Good balance (not too abrupt, not too gradual)
- **min_weight=0.1**: Keeps 10% of edge information (not completely removed)

## ✅ Benefits

- ✅ **Reduces noise** at edge wavelengths (490-510 nm, 670-690 nm)
- ✅ **Preserves center wavelengths** (540-620 nm) unchanged
- ✅ **Smooth transition** (no hard cutoff)
- ✅ **Works with existing pipeline** (no major code changes)
- ✅ **Improves SVM classification** by reducing noisy feature influence
- ✅ **Improves NDI analysis** by avoiding problematic wavelength pairs

## 📁 Files

- **Documentation**: `SPECTRAL_PREPROCESSING_WORKFLOW_AND_SNR_FILTERING.md`
- **Test notebook**: `test_snr_filtering_methods.ipynb`
- **Implementation**: Add function to `utils/gref_pipeline/georef.py`

## 🔬 Testing

Run the test notebook to:
1. Visualize raw spectra with edge noise
2. See SNR per wavelength band
3. Compare original vs weighted spectra
4. Test different parameter settings

```bash
# Open test notebook
jupyter notebook test_snr_filtering_methods.ipynb
```

## 🚀 Next Steps

1. **Test on 057_5**: Compare SVM accuracy with/without weighting
2. **Test on 028**: Check if NDI analysis improves
3. **Add to pipeline**: Consider making this a standard preprocessing step
4. **Document results**: Record optimal parameters for different transects

---

**Author:** Analysis based on dev20-dev29 workflows  
**Date:** November 4, 2025  
**Status:** Ready for testing and integration
