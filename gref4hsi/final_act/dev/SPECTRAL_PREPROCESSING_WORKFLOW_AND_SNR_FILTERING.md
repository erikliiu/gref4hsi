# Spectral Preprocessing Workflow & SNR-Based Filtering

## Date: November 4, 2025
## Author: Analysis of dev20-dev29 workflow with recommendations for edge band filtering

---

## 📋 TABLE OF CONTENTS

1. [General Workflow Overview](#general-workflow-overview)
2. [The SNR Problem at Wavelength Edges](#the-snr-problem-at-wavelength-edges)
3. [Recommended Filtering/Smoothing Methods](#recommended-filteringsmoothing-methods)
4. [Implementation Strategy](#implementation-strategy)
5. [Code Examples](#code-examples)

---

## 🔄 GENERAL WORKFLOW OVERVIEW

### Workflow from dev20-dev29 Analysis

The typical hyperspectral processing pipeline follows these steps:

```
RAW DATA (Digital Counts in HDF5)
    ↓
1. LOAD RADIANCE DATA
   - Read from HDF5: datacube (tracks, slits, bands)
   - Shape: typically (2000-2500, 968, 210)
   - Units: W/(m² sr μm) or similar radiometric units
    ↓
2. ILLUMINATION CORRECTION (apply_illumination_correction_v2)
   - Converts radiance to PSEUDO-REFLECTANCE
   - Per-slit, per-band rolling median normalization
   - Window size: typically 500-1000 tracks
   - Removes illumination variations along track
   - OUTPUT: data_corrected (normalized radiance ≈ pseudo-reflectance)
    ↓
3. WAVELENGTH FILTERING (apply_wavelength_filter)
   - Crop to useful spectral range: typically 490-690 nm
   - Removes noisy edge bands
   - PROBLEM: Even within this range, edges have poor SNR!
    ↓
4. OPTIONAL: SPECTRAL SMOOTHING (apply_spectral_smoothing)
   - Moving average or Savitzky-Golay filter
   - Window: 10-30 bands
   - Reduces high-frequency noise
    ↓
5. OPTIONAL: SPECTRAL NORMALIZATION (apply_spectral_normalization)
   - L2 normalization (unit vector)
   - SNV (Standard Normal Variate)
   - Mean centering
    ↓
6. ANALYSIS
   - SVM classification (dev20-23)
   - NDI (Normalized Difference Index) analysis (dev24-26)
   - Greyscale/single wavelength (dev29)
   - Band ratios and derivatives (dev25)
```

### Key Files in Workflow:
- **dev20**: SVM classification with raw spectral data
- **dev21**: Intensity-only normalization tests
- **dev22**: Full spectral normalization for SVM
- **dev23**: Transfer learning (train on 057, test on 028)
- **dev24-26**: NDI heatmap analysis for UXO detection
- **dev27-28**: ROI extraction and cropping
- **dev29**: Single wavelength greyscale analysis

---

## ⚠️ THE SNR PROBLEM AT WAVELENGTH EDGES

### The Issue

**EVEN AFTER** applying `apply_wavelength_filter(wavelength_range=(490, 690))`, the **edges of the clipped range still have poor SNR** due to:

1. **Low Radiance Values**
   - Sensor sensitivity drops at spectral edges
   - Water absorption increases dramatically at red wavelengths (>650 nm)
   - Blue wavelengths (<500 nm) suffer from low illumination

2. **Sensor Noise**
   - Fixed read noise becomes dominant when signal is weak
   - Dark current becomes significant
   - Quantization noise at low digital counts

3. **Illumination Correction Amplifies Noise**
   - Division by low reference values amplifies noise
   - Edge bands can have artificially inflated values
   - Creates "spikes" or "bands of noise" at edges

### Visualization of the Problem

```
Wavelength (nm):  490 -------- 550 -------- 620 -------- 690
                   ↑                                        ↑
SNR:          [LOW SNR]    [GOOD SNR]    [GOOD SNR]   [LOW SNR]
              High noise    Stable data   Stable data   High noise
              Low signal                                Low signal
```

### Why This Happens in Your Workflow

```python
# Step 1: Load radiance (raw data has low values at edges)
cube = transect.select_files(["rad_uhi_20241029_125028_5"])

# Step 2: Illumination correction (DIVIDES by reference, amplifies noise!)
cube.apply_illumination_correction_v2()
# → Now edge bands have HIGH values but HIGH NOISE

# Step 3: Wavelength filter (crops, but doesn't fix SNR at new edges!)
cube.apply_wavelength_filter(wavelength_range=(490, 690))
# → Edge bands (490-510 nm, 670-690 nm) still have poor SNR
```

---

## 🛠️ RECOMMENDED FILTERING/SMOOTHING METHODS

### Method 1: **Soft Edge Weighting** (RECOMMENDED)

Apply weights that gradually reduce the influence of edge bands.

**Advantages:**
- ✅ Preserves spectral information
- ✅ Smooth transition (no hard cutoff)
- ✅ Works well with SVM and NDI analysis
- ✅ Physically motivated

**Implementation:**
```python
def apply_soft_edge_weighting(datacube, wavelengths, 
                               edge_width=20,  # nm from each edge
                               min_weight=0.1):
    """
    Apply sigmoid weights to reduce edge band influence.
    
    Parameters:
    -----------
    edge_width : float
        Distance (in nm) from edge where weighting starts
    min_weight : float
        Minimum weight at the very edge (0-1)
    """
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    wl_range = wl_max - wl_min
    
    # Create sigmoid weights
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
    
    # Apply weights (broadcast across tracks and slits)
    weighted_cube = datacube * weights[np.newaxis, np.newaxis, :]
    
    return weighted_cube, weights
```

### Method 2: **SNR-Based Adaptive Filtering**

Calculate SNR for each wavelength and apply filtering based on actual noise levels.

**Advantages:**
- ✅ Data-driven (uses actual noise measurements)
- ✅ Automatically identifies problematic bands
- ✅ Can be computed per-transect

**Implementation:**
```python
def compute_snr_per_band(datacube, method='std'):
    """
    Compute SNR for each spectral band.
    
    Parameters:
    -----------
    method : str
        'std': SNR = mean / std (simple)
        'mad': SNR based on Median Absolute Deviation (robust)
    """
    T, S, B = datacube.shape
    snr = np.zeros(B)
    
    for b in range(B):
        band_data = datacube[:, :, b].flatten()
        band_data = band_data[np.isfinite(band_data)]
        
        if method == 'std':
            snr[b] = np.mean(band_data) / (np.std(band_data) + 1e-10)
        elif method == 'mad':
            median = np.median(band_data)
            mad = np.median(np.abs(band_data - median))
            snr[b] = median / (1.4826 * mad + 1e-10)  # 1.4826 for normal distribution
    
    return snr

def apply_snr_based_weighting(datacube, wavelengths, 
                               snr_threshold=10,
                               smooth_transition=5):
    """
    Weight bands based on their SNR.
    
    Parameters:
    -----------
    snr_threshold : float
        SNR below which weighting starts
    smooth_transition : float
        SNR range over which transition occurs
    """
    snr = compute_snr_per_band(datacube, method='mad')
    
    # Create weights: 1.0 for SNR > threshold, smooth decay below
    weights = np.ones(len(wavelengths))
    low_snr_mask = snr < snr_threshold
    
    if np.any(low_snr_mask):
        # Sigmoid transition
        x = (snr[low_snr_mask] - snr_threshold) / smooth_transition
        weights[low_snr_mask] = 1 / (1 + np.exp(-5*x))
    
    # Apply weights
    weighted_cube = datacube * weights[np.newaxis, np.newaxis, :]
    
    return weighted_cube, weights, snr
```

### Method 3: **Histogram-Based Outlier Clipping**

Clip extreme values that occur predominantly at edge wavelengths.

**Advantages:**
- ✅ Removes outliers caused by noise amplification
- ✅ Preserves normal spectral features
- ✅ Simple to implement

**Implementation:**
```python
def apply_percentile_clipping(datacube, wavelengths,
                               lower_percentile=1,
                               upper_percentile=99,
                               edge_only=True,
                               edge_width=20):
    """
    Clip extreme values based on histogram percentiles.
    
    Parameters:
    -----------
    lower_percentile, upper_percentile : float
        Percentiles for clipping (0-100)
    edge_only : bool
        If True, only apply to edge bands
    edge_width : float
        Width of edge region (nm) if edge_only=True
    """
    clipped_cube = datacube.copy()
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    
    # Determine which bands to process
    if edge_only:
        edge_mask = (wavelengths < wl_min + edge_width) | (wavelengths > wl_max - edge_width)
        band_indices = np.where(edge_mask)[0]
    else:
        band_indices = range(len(wavelengths))
    
    # Clip each band
    for b in band_indices:
        band_data = datacube[:, :, b]
        lower = np.percentile(band_data[np.isfinite(band_data)], lower_percentile)
        upper = np.percentile(band_data[np.isfinite(band_data)], upper_percentile)
        clipped_cube[:, :, b] = np.clip(band_data, lower, upper)
    
    return clipped_cube
```

### Method 4: **Spectral Smoothing with Edge Emphasis**

Apply stronger smoothing to edge bands, lighter smoothing to center.

**Advantages:**
- ✅ Reduces noise where it's worst
- ✅ Preserves spectral features in good SNR regions
- ✅ Can combine with existing smoothing methods

**Implementation:**
```python
def apply_adaptive_smoothing(datacube, wavelengths,
                              center_window=5,
                              edge_window=30,
                              edge_width=20,
                              method='savgol'):
    """
    Apply variable smoothing: strong at edges, light in center.
    
    Parameters:
    -----------
    center_window : int
        Smoothing window for center bands
    edge_window : int
        Smoothing window for edge bands
    edge_width : float
        Width (nm) defining edge region
    method : str
        'savgol' or 'moving_average'
    """
    from scipy.signal import savgol_filter
    
    T, S, B = datacube.shape
    smoothed_cube = np.zeros_like(datacube)
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    
    # Determine window size for each wavelength
    windows = np.zeros(B, dtype=int)
    for i, wl in enumerate(wavelengths):
        # Distance to nearest edge
        dist_to_edge = min(wl - wl_min, wl_max - wl)
        
        # Interpolate window size
        if dist_to_edge < edge_width:
            alpha = dist_to_edge / edge_width
            windows[i] = int(edge_window - alpha * (edge_window - center_window))
        else:
            windows[i] = center_window
    
    # Ensure odd window sizes
    windows = windows // 2 * 2 + 1
    
    # Apply smoothing track-by-track, slit-by-slit
    for t in range(T):
        for s in range(S):
            spectrum = datacube[t, s, :]
            
            if method == 'savgol':
                # Variable-width Savitzky-Golay
                smoothed = np.zeros(B)
                for b in range(B):
                    w = windows[b]
                    half_w = w // 2
                    start = max(0, b - half_w)
                    end = min(B, b + half_w + 1)
                    local_window = end - start
                    if local_window >= 5 and local_window % 2 == 1:
                        try:
                            smoothed[b] = savgol_filter(
                                spectrum[start:end],
                                window_length=local_window,
                                polyorder=2
                            )[b - start]
                        except:
                            smoothed[b] = spectrum[b]
                    else:
                        smoothed[b] = spectrum[b]
                
                smoothed_cube[t, s, :] = smoothed
                
            elif method == 'moving_average':
                # Variable-width moving average
                smoothed = np.zeros(B)
                for b in range(B):
                    w = windows[b]
                    half_w = w // 2
                    start = max(0, b - half_w)
                    end = min(B, b + half_w + 1)
                    smoothed[b] = np.mean(spectrum[start:end])
                
                smoothed_cube[t, s, :] = smoothed
    
    return smoothed_cube, windows
```

---

## 📊 IMPLEMENTATION STRATEGY

### Recommended Workflow Integration

```python
# ===== STANDARD WORKFLOW =====

# 1. Load data
transect = load_transect(config.OUTPUT_FOLDER)
cube = transect.select_files(["rad_uhi_20241029_125028_5"])

# 2. Illumination correction (converts to pseudo-reflectance)
cube.apply_illumination_correction_v2(window_size=500, strength=1.0)

# 3. Wavelength filter (initial crop)
cube.apply_wavelength_filter(wavelength_range=(490, 690))

# ===== NEW: SNR-AWARE PROCESSING =====

# METHOD 1: Soft edge weighting (RECOMMENDED FOR MOST CASES)
cube.data_corrected, weights = apply_soft_edge_weighting(
    cube.data_corrected,
    cube.wavelengths,
    edge_width=20,  # 20 nm transition zone
    min_weight=0.1   # 10% weight at extreme edges
)

# METHOD 2: SNR-based weighting (IF YOU WANT DATA-DRIVEN APPROACH)
cube.data_corrected, weights, snr = apply_snr_based_weighting(
    cube.data_corrected,
    cube.wavelengths,
    snr_threshold=10,
    smooth_transition=5
)

# Plot SNR to diagnose issues
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cube.wavelengths, snr)
plt.xlabel('Wavelength (nm)')
plt.ylabel('SNR')
plt.title('SNR per Band')
plt.subplot(1, 2, 2)
plt.plot(cube.wavelengths, weights)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Weight')
plt.title('Applied Weights')
plt.tight_layout()

# METHOD 3: Histogram clipping (ADDITIONAL STEP IF NEEDED)
cube.data_corrected = apply_percentile_clipping(
    cube.data_corrected,
    cube.wavelengths,
    lower_percentile=1,
    upper_percentile=99,
    edge_only=True
)

# METHOD 4: Adaptive smoothing (IF USING SMOOTHING ANYWAY)
cube.data_corrected, windows = apply_adaptive_smoothing(
    cube.data_corrected,
    cube.wavelengths,
    center_window=5,
    edge_window=30,
    edge_width=20,
    method='savgol'
)

# ===== CONTINUE WITH STANDARD WORKFLOW =====

# 4. Optional: L2 normalization
cube.apply_spectral_normalization(method='l2')

# 5. Analysis (SVM, NDI, etc.)
# ...
```

### When to Use Each Method

| Method | Best For | Avoid When |
|--------|----------|------------|
| **Soft Edge Weighting** | General purpose, SVM classification, NDI analysis | Need to preserve absolute intensities |
| **SNR-Based Weighting** | Unknown data quality, multi-transect analysis | Computationally expensive for large datasets |
| **Histogram Clipping** | Extreme outliers, visualization | May remove real features |
| **Adaptive Smoothing** | Already using smoothing, want to enhance edges | Need high spectral resolution |

### Combining Methods

You can combine multiple methods for robust processing:

```python
# 1. First clip extreme outliers
cube.data_corrected = apply_percentile_clipping(
    cube.data_corrected, cube.wavelengths,
    edge_only=True, edge_width=20
)

# 2. Then apply soft edge weighting
cube.data_corrected, weights = apply_soft_edge_weighting(
    cube.data_corrected, cube.wavelengths,
    edge_width=20, min_weight=0.2
)

# 3. Finally apply adaptive smoothing
cube.data_corrected, windows = apply_adaptive_smoothing(
    cube.data_corrected, cube.wavelengths,
    center_window=5, edge_window=20, edge_width=15
)
```

---

## 💻 CODE EXAMPLES

### Example 1: Add Methods to CombinedTransectCube Class

Add these methods to `gref4hsi/final_act/utils/gref_pipeline/georef.py`:

```python
def apply_soft_edge_weighting(self, edge_width=20, min_weight=0.1, quiet=False):
    """
    Apply sigmoid weights to reduce edge band influence.
    
    This should be called AFTER apply_wavelength_filter() to handle
    the new edge bands that may have poor SNR.
    
    Parameters:
    -----------
    edge_width : float, default=20
        Distance (in nm) from each edge where weighting starts
    min_weight : float, default=0.1
        Minimum weight at the very edge (0-1)
    quiet : bool, default=False
        If True, suppresses progress messages
    
    Returns:
    --------
    tuple : (weighted_datacube, weights)
    """
    if self.data_corrected is None:
        raise ValueError("No corrected data available.")
    
    wl_min, wl_max = self.wavelengths[0], self.wavelengths[-1]
    
    # Create sigmoid weights
    weights = np.ones(len(self.wavelengths))
    
    # Left edge
    left_mask = self.wavelengths < (wl_min + edge_width)
    if np.any(left_mask):
        x = (self.wavelengths[left_mask] - wl_min) / edge_width
        weights[left_mask] = min_weight + (1 - min_weight) * (1 / (1 + np.exp(-10*(x - 0.5))))
    
    # Right edge  
    right_mask = self.wavelengths > (wl_max - edge_width)
    if np.any(right_mask):
        x = (wl_max - self.wavelengths[right_mask]) / edge_width
        weights[right_mask] = min_weight + (1 - min_weight) * (1 / (1 + np.exp(-10*(x - 0.5))))
    
    if not quiet:
        print("=" * 60)
        print("🎚️  SOFT EDGE WEIGHTING")
        print("=" * 60)
        print(f"📊 Wavelength range: {wl_min:.1f}-{wl_max:.1f} nm")
        print(f"🔀 Edge width: {edge_width} nm")
        print(f"⬇️  Minimum weight: {min_weight}")
        print(f"📉 Left edge affected: {np.sum(left_mask)} bands")
        print(f"📉 Right edge affected: {np.sum(right_mask)} bands")
    
    # Apply weights
    self.data_corrected = self.data_corrected * weights[np.newaxis, np.newaxis, :]
    
    if not quiet:
        print("✅ Edge weighting applied")
        print("=" * 60)
    
    return self.data_corrected, weights


def compute_and_plot_snr(self, method='mad', figsize=(12, 5)):
    """
    Compute and plot SNR for each spectral band.
    
    Useful for diagnosing which bands have poor quality.
    
    Parameters:
    -----------
    method : str, default='mad'
        'std': SNR = mean / std (simple)
        'mad': SNR based on Median Absolute Deviation (robust)
    figsize : tuple, default=(12, 5)
        Figure size
    
    Returns:
    --------
    np.ndarray : SNR values for each band
    """
    if self.data_corrected is None:
        raise ValueError("No corrected data available.")
    
    T, S, B = self.data_corrected.shape
    snr = np.zeros(B)
    
    print(f"Computing SNR using {method.upper()} method...")
    
    for b in range(B):
        band_data = self.data_corrected[:, :, b].flatten()
        band_data = band_data[np.isfinite(band_data)]
        
        if method == 'std':
            snr[b] = np.mean(band_data) / (np.std(band_data) + 1e-10)
        elif method == 'mad':
            median = np.median(band_data)
            mad = np.median(np.abs(band_data - median))
            snr[b] = median / (1.4826 * mad + 1e-10)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # SNR plot
    ax1.plot(self.wavelengths, snr, 'b-', linewidth=2)
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('SNR', fontsize=12)
    ax1.set_title(f'Signal-to-Noise Ratio ({method.upper()})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=10, color='r', linestyle='--', label='Threshold=10')
    ax1.legend()
    
    # Histogram
    ax2.hist(snr, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('SNR', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('SNR Distribution', fontsize=14, fontweight='bold')
    ax2.axvline(x=10, color='r', linestyle='--', label='Threshold=10')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSNR Statistics:")
    print(f"  Min:    {np.min(snr):.2f}")
    print(f"  25%:    {np.percentile(snr, 25):.2f}")
    print(f"  Median: {np.median(snr):.2f}")
    print(f"  75%:    {np.percentile(snr, 75):.2f}")
    print(f"  Max:    {np.max(snr):.2f}")
    print(f"  Bands with SNR < 10: {np.sum(snr < 10)}/{len(snr)}")
    
    return snr
```

### Example 2: Complete Processing Script

Create a new file `gref4hsi/final_act/dev/test_snr_filtering.py`:

```python
"""
Test SNR-aware filtering methods on UHI hyperspectral data.

This script demonstrates various approaches to handling poor SNR
at wavelength edges after illumination correction.
"""

import importlib
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))

from utils.gref_pipeline import georef
from gref_pipeline import config

importlib.reload(georef)
from utils.gref_pipeline.georef import *

# ===== LOAD DATA =====

transect = load_transect(config.OUTPUT_FOLDER)
transect.list_files()

cube = transect.select_files(["rad_uhi_20241029_115057_5"])
cube.describe()

# ===== STANDARD PREPROCESSING =====

# 1. Illumination correction
cube.apply_illumination_correction_v2(window_size=500, strength=1.0)
print(f"✅ Illumination correction complete")

# 2. Wavelength filter
cube.apply_wavelength_filter(wavelength_range=(490, 690))
print(f"✅ Wavelength filtering complete")
print(f"   Wavelength range: {cube.wavelengths[0]:.1f} - {cube.wavelengths[-1]:.1f} nm")

# ===== DIAGNOSE SNR ISSUES =====

print("\n" + "="*80)
print("DIAGNOSING SNR ISSUES")
print("="*80)

# Compute and plot SNR
snr = cube.compute_and_plot_snr(method='mad')

# ===== TEST DIFFERENT FILTERING METHODS =====

# Save original for comparison
data_original = cube.data_corrected.copy()

# Test Method 1: Soft Edge Weighting
print("\n" + "="*80)
print("METHOD 1: SOFT EDGE WEIGHTING")
print("="*80)

cube.data_corrected = data_original.copy()
cube.apply_soft_edge_weighting(edge_width=20, min_weight=0.1)

# Plot a sample spectrum
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original spectrum
ax = axes[0, 0]
spectrum_orig = data_original[1200, 500, :]
ax.plot(cube.wavelengths, spectrum_orig, 'b-', linewidth=2, label='Original')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Pseudo-reflectance')
ax.set_title('Original Spectrum (track=1200, slit=500)')
ax.legend()
ax.grid(True, alpha=0.3)

# Weighted spectrum
ax = axes[0, 1]
spectrum_weighted = cube.data_corrected[1200, 500, :]
ax.plot(cube.wavelengths, spectrum_weighted, 'r-', linewidth=2, label='Weighted')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Pseudo-reflectance')
ax.set_title('After Soft Edge Weighting')
ax.legend()
ax.grid(True, alpha=0.3)

# Comparison
ax = axes[1, 0]
ax.plot(cube.wavelengths, spectrum_orig, 'b-', linewidth=2, alpha=0.7, label='Original')
ax.plot(cube.wavelengths, spectrum_weighted, 'r-', linewidth=2, alpha=0.7, label='Weighted')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Pseudo-reflectance')
ax.set_title('Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Difference
ax = axes[1, 1]
diff = spectrum_weighted - spectrum_orig
ax.plot(cube.wavelengths, diff, 'g-', linewidth=2)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Difference')
ax.set_title('Weighted - Original')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('test_snr_filtering_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Processing complete!")
print(f"   Figure saved: test_snr_filtering_comparison.png")
```

---

## 🎯 SUMMARY & RECOMMENDATIONS

### For SVM Classification (dev20-23 workflow):
1. **Use soft edge weighting** with `edge_width=20, min_weight=0.1`
2. **Apply L2 normalization** as usual
3. Edge bands will contribute less to classification without being completely removed

### For NDI Analysis (dev24-26 workflow):
1. **Use SNR-based weighting** to identify problematic wavelength pairs
2. **Avoid computing NDI** with bands below SNR threshold
3. Consider **adaptive smoothing** before computing derivatives

### For Greyscale/Single Wavelength (dev29 workflow):
1. **Avoid edge wavelengths** entirely for greyscale conversion
2. Use center wavelengths (540-600 nm) where SNR is best
3. If must use edges, apply **histogram clipping** first

### General Best Practice:
```python
# RECOMMENDED STANDARD WORKFLOW
cube.apply_illumination_correction_v2()
cube.apply_wavelength_filter(wavelength_range=(490, 690))
cube.apply_soft_edge_weighting(edge_width=20, min_weight=0.1)  # NEW!
cube.apply_spectral_normalization(method='l2')
# Then proceed with analysis...
```

---

## 📚 REFERENCES

- Dev20-29 notebooks in `gref4hsi/final_act/dev/`
- `georef.py`: `apply_illumination_correction_v2()`, `apply_wavelength_filter()`
- `ndi_analysis_utils.py`: NDI computation functions
- Signal processing: Savitzky-Golay filtering, rolling medians

---

## ✅ NEXT STEPS

1. **Implement** `apply_soft_edge_weighting()` in `georef.py`
2. **Test** on a single transect (057_5 or 028_4)
3. **Compare** classification accuracy with/without edge weighting
4. **Document** optimal parameters for different transects
5. **Consider** adding to preprocessing pipeline as default step

---

**Author Notes:**
- This document synthesizes findings from dev20-29 workflow analysis
- All methods are designed to integrate seamlessly with existing pipeline
- Focus is on practical implementation with minimal code changes
- Methods preserve spectral information while reducing edge noise impact
