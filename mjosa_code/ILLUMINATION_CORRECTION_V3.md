# Illumination Correction Algorithm (v3)

## Overview

The illumination correction algorithm removes along-track illumination gradients caused by changing sun angles during flight acquisition. The algorithm normalizes the radiance data by dividing each pixel by a rolling median baseline computed along the flight direction (time axis).

**Version:** v3 (latest)  
**Location:** `mjosa_code/utils/uhi/georef.py` (lines 4261-4750)  
**Method:** `apply_illumination_correction_v3()`  
**Default Parameters:** `window_size=500`, `strength=1.0`, `strategy="memory"`

---

## Physical Motivation

**Problem:** As the aircraft flies, the sun angle relative to the water surface changes continuously. This causes a multiplicative gradient in the measured radiance across the flightline (brightest when sun angle is optimal, darker at flight start/end).

**Solution:** Compute a smooth baseline representing the along-track illumination trend, then normalize each pixel by this baseline. This removes the sun angle gradient while preserving spatial features.

**Why Rolling Median?** 
- Robust to outliers (waves, whitecaps, ships)
- Smooths out along-track illumination changes
- Preserves median radiance level (doesn't introduce bias)
- Works per (slit, band) combination to handle spectral and cross-track variations

---

## Mathematical Formulation

Let $\mathbf{R}(t, s, \lambda)$ be the raw radiance datacube, where:
- $t \in [1, T]$ is the along-track (time) index
- $s \in [1, S]$ is the across-track (slit/pixel) index  
- $\lambda \in [1, B]$ is the spectral band index

### Step 1: Compute Rolling Median Baseline

For each (slit, band) pair $(s, \lambda)$, compute the rolling median along the time axis:

$$
\mathbf{R}_{\text{ref}}(t, s, \lambda) = \text{median}\left(\mathbf{R}(t', s, \lambda) : |t' - t| \leq \frac{w}{2}\right)
$$

where $w$ is the `window_size` parameter (default: 500 pixels ≈ 25 m along-track).

**Implementation details:**
- Uses `pandas.Series.rolling(window=w, center=True, min_periods=1).median()`
- `center=True`: Window is centered at current pixel
- `min_periods=1`: Use available data at boundaries (no NaN padding)
- Computed independently for each of the $S \times B$ time series

### Step 2: Normalize by Baseline

Divide the raw radiance by the rolling median baseline:

$$
\mathbf{R}_{\text{norm}}(t, s, \lambda) = \frac{\mathbf{R}(t, s, \lambda)}{\mathbf{R}_{\text{ref}}(t, s, \lambda)}
$$

**Zero handling:** If $\mathbf{R}_{\text{ref}}(t, s, \lambda) = 0$, set $\mathbf{R}_{\text{ref}}(t, s, \lambda) = 1$ to avoid division by zero.

### Step 3: Blend with Original Data

Apply a strength parameter $\alpha \in [0, 1]$ to blend the corrected data with the original:

$$
\mathbf{R}_{\text{corrected}}(t, s, \lambda) = (1 - \alpha) \cdot \mathbf{R}(t, s, \lambda) + \alpha \cdot \mathbf{R}_{\text{norm}}(t, s, \lambda)
$$

where:
- $\alpha = 0$: No correction (output = input)
- $\alpha = 1$: Full correction (output = normalized data)
- $0 < \alpha < 1$: Partial correction (blend of original and normalized)

**Default:** $\alpha = 1.0$ (full correction)

---

## Processing Strategies

The algorithm offers three processing strategies with different RAM/speed trade-offs:

### Strategy 1: Memory (Default)
**Description:** Load all data into memory, compute rolling median vectorized, save corrected data.

**Pros:**
- Fastest (single-pass, vectorized operations)
- Simplest implementation

**Cons:**
- Requires RAM for full datacube: $T \times S \times B \times 4$ bytes (float32)
- Example: 20,000 × 968 × 462 ≈ 34 GB RAM

**Use when:** Sufficient RAM available (typical for small-medium datasets)

### Strategy 2: Two-Pass
**Description:** Pass 1 computes rolling median and saves to disk. Pass 2 loads data and applies correction.

**Pros:**
- Moderate RAM usage (only one file in memory at a time)
- Still relatively fast (two sequential passes)

**Cons:**
- 2× file I/O overhead
- Requires disk space for intermediate rolling median file

**Use when:** Limited RAM but fast disk I/O available

### Strategy 3: Streaming
**Description:** Process data in chunks with boundary overlap for rolling median continuity.

**Pros:**
- Minimal RAM (only chunk + padding in memory)
- Can process arbitrarily large datasets

**Cons:**
- Slowest (repeated boundary handling)
- Complex implementation (overlap management)

**Use when:** Very limited RAM or extremely large datasets

---

## Algorithm Details

### Vectorization Approach

To efficiently compute rolling medians for $S \times B$ independent time series:

1. **Reshape:** Transform $(T, S, B) \rightarrow (T, S \cdot B)$
2. **Compute:** Apply rolling median column-wise (each column is one time series)
3. **Reshape back:** $(T, S \cdot B) \rightarrow (T, S, B)$

This leverages pandas' vectorized rolling operations, avoiding slow Python loops.

### Boundary Handling

**At file boundaries:**
- **Memory strategy:** Concatenate all files before computing rolling median (seamless)
- **Two-pass strategy:** Concatenate all files in pass 1, split in pass 2
- **Streaming strategy:** Load padding from adjacent files (half-window overlap)

**At dataset start/end:**
- `min_periods=1`: Use available data within window (no NaN values)
- Example: At $t=1$, window only has $[1, 250]$ available (not full 500)

### Output Format

Corrected data saved to H5 file as new dataset:

**Dataset name:** `"radiance/reflectance_cube_dn_smoothed_raw_corrected_wXXX_sY.YY"`
- `XXX`: Window size (e.g., 500)
- `Y.YY`: Strength parameter (e.g., 1.00)

**Attributes:**
- `window_size`: Rolling median window width
- `strength`: Blending parameter $\alpha$
- `correction_method`: "rolling_v3_memory" / "rolling_v3_twopass" / "rolling_v3_streaming"

**Compression:** gzip level 1 (fast compression, ~2× size reduction)

---

## Code Example

```python
from mjosa_code.utils.uhi.georef import GeorefImages

# Initialize georeferenced image collection
geofiles = GeorefImages(...)

# Apply illumination correction
geofiles.apply_illumination_correction_v3(
    window_size=500,    # 500 pixels ≈ 25 m along-track
    strength=1.0,       # Full correction (α = 1.0)
    strategy="memory",  # Use memory strategy (fastest)
    smoothed_raw_dset="radiance/reflectance_cube_dn_smoothed_raw"  # Input dataset
)
```

---

## Paper Methods Text (Suggested)

> Illumination correction was applied to remove along-track radiance gradients caused by changing solar zenith angles during acquisition. For each spectral band and across-track pixel, a rolling median baseline was computed along the flight direction using a 500-pixel window (~25 m). The radiance data were then normalized by dividing each pixel by its corresponding baseline value, effectively removing the multiplicative sun angle gradient while preserving spatial features. The rolling median was chosen for its robustness to outliers (e.g., waves, whitecaps) and its ability to smooth along-track illumination trends without introducing bias.

---

## Key Differences from Earlier Versions

**v3 improvements:**
- ✅ Removed global mode (only rolling window, more accurate)
- ✅ Three processing strategies for different RAM constraints
- ✅ Vectorized implementation (10× faster than v2)
- ✅ Proper boundary handling at file edges
- ✅ Per-(slit, band) baseline (accounts for cross-track and spectral variations)

**v2 limitations:**
- Had global mode option (removed in v3 for accuracy)
- Single processing strategy (no RAM flexibility)
- Slower loop-based implementation

---

## Validation

**Visual check:** Corrected radiance should have uniform brightness along flight direction, with no visible gradient from bright (middle) to dark (edges).

**Quantitative check:** Compute along-track mean radiance profile:
$$
\bar{R}(t) = \frac{1}{S \cdot B} \sum_{s=1}^{S} \sum_{\lambda=1}^{B} R(t, s, \lambda)
$$

Before correction: $\bar{R}(t)$ should show parabolic or gradient pattern.  
After correction: $\bar{R}(t)$ should be approximately flat (small variance).

---

## References

- **Implementation:** `mjosa_code/utils/uhi/georef.py` lines 4261-4750
- **Notebook example:** `mjosa_code/notebooks/17_compare_dark_vs_empty_section.ipynb`
- **Related:** See `SEGMENT_STATISTICS_GUIDE.md` for per-segment analysis after correction
