# Custom Class Weights + Brightness Feature Implementation

## 📋 Overview

Implemented two new features to improve SVM bomb detection:
1. **Custom Class Weights**: Allows manual control over class importance (e.g., 2x weight for bombs)
2. **Brightness Feature**: Adds mean intensity across wavelength-filtered spectrum as an additional feature

## 🎯 Problem

Current SVM classification was **too conservative** - missing some bombs because:
- Balanced class weights treat all classes equally
- Only spectral features used (no brightness/intensity information)

## ✅ Solution

### 1. Custom Class Weights (`class_weight_dict`)

**What it does:**
- Allows you to specify custom weights for each class
- Higher weight = model pays more attention to that class
- Increases recall (catches more instances) at potential cost of precision

**Example usage:**
```python
custom_weights = {
    "training_bombs": 2.0,  # 2x weight - catch more bombs!
    "training_dark": 1.0,
    "training_sediment": 1.0,
}

cv_results = cube.train_svm_with_cv(
    ...,
    class_weight_dict=custom_weights,  # 🔥 NEW
)
```

**Default behavior:**
- If `class_weight_dict=None` (default), uses `class_weight="balanced"` (automatic)

### 2. Brightness Feature (`add_brightness_feature`)

**What it does:**
- Calculates mean intensity across wavelength-filtered spectrum (490-680nm)
- Uses **corrected** pseudo-reflectance data
- **NO normalization** - raw mean value
- Appends as the last feature column

**Example usage:**
```python
cv_results = cube.train_svm_with_cv(
    ...,
    wavelength_range=(490, 680),  # Filter wavelengths first
    add_brightness_feature=True,  # 🔥 NEW - calculate AFTER filtering
)
```

**Implementation details:**
```python
# After wavelength filtering (490-680nm)
brightness = np.mean(spectrum_490_680nm, axis=1, keepdims=True)  # Shape: (n_pixels, 1)
X_train = np.hstack([X_train, brightness])  # Append as last column
```

**Applies everywhere:**
- ✅ Training data (`train_svm_with_cv`)
- ✅ Cross-validation
- ✅ Classification (`classify_segment_with_validation`)

## 📁 Files Modified

### 1. `georef.py` - Main implementation

**Function: `train_svm_with_cv`** (line ~6485)
- Added parameters:
  - `class_weight_dict=None`
  - `add_brightness_feature=False`
- Stores `self.svm_add_brightness_feature` for classification

**Changes:**
1. **Lines ~6505**: Added new parameters to function signature
2. **Lines ~6625**: Print brightness feature status, store `svm_add_brightness_feature`
3. **Lines ~6688-6700**: Calculate and append brightness to `X_train`
4. **Lines ~7065-7090**: Convert `class_weight_dict` to encoded labels, pass to SVC
5. **Lines ~7095-7120**: Use `class_weight_setting` in both GridSearchCV branches

**Function: `classify_segment_with_validation`** (line ~7406)
- **Lines ~7555-7560**: Check if brightness was used during training, calculate and append to `X_classify`

### 2. `test_class_weights_and_brightness.py` - Test script

**Purpose:** Compare SVM performance with/without custom weights and brightness

**Structure:**
1. Load data and ROIs
2. **TEST 1 (Baseline)**: Train with balanced weights, no brightness
3. **TEST 2 (Improved)**: Train with custom weights (2x bombs), brightness enabled
4. **Comparison**: Print side-by-side metrics
5. **Visualization**: Plot before/after classification maps

**Run command:**
```bash
cd gref4hsi/final_act/dev
python test_class_weights_and_brightness.py
```

### 3. `dev17_test_SVM.ipynb` - Notebook cells

**New cells added after cell 6 (training cell):**

**Cell 7 (Markdown):**
- Header explaining the new test

**Cell 8 (Code):**
- Train SVM with custom weights + brightness
- Compare CV metrics with baseline

**Cell 9 (Code):**
- Classify segment with custom weights + brightness
- Print validation metrics comparison

**Cell 10 (Code):**
- Visualize before/after side-by-side
- Print pixel count changes

## 🧪 Testing

### Option 1: Run standalone test script
```bash
cd gref4hsi/final_act/dev
python test_class_weights_and_brightness.py
```

### Option 2: Run notebook cells
1. Open `dev17_test_SVM.ipynb`
2. Run cells 1-6 (load data, baseline training)
3. Run cells 7-10 (custom weights + brightness testing)

## 📊 Expected Results

**Bomb recall should increase** (catch more bombs):
- More bomb pixels classified correctly
- Fewer false negatives (missed bombs)

**Potential trade-offs:**
- Precision may decrease slightly (more false positives)
- Dark spots or sediment misclassified as bombs

**Brightness feature benefits:**
- Distinguishes bright bombs from dark spots
- Adds intensity information to spectral features

## 🔧 Parameters Reference

### `train_svm_with_cv`

```python
def train_svm_with_cv(
    ...,
    class_weight_dict=None,        # Dict: {"class_name": weight_value}
    add_brightness_feature=False,  # Bool: Enable brightness feature
):
```

**`class_weight_dict`:**
- Type: `dict` or `None`
- Example: `{"training_bombs": 2.0, "training_dark": 1.0, "training_sediment": 1.0}`
- Default: `None` (uses `class_weight="balanced"`)

**`add_brightness_feature`:**
- Type: `bool`
- Default: `False`
- Calculation: `brightness = mean(spectrum_490_680nm)` (corrected data)

## ⚠️ Important Notes

1. **Wavelength filtering is CRITICAL:**
   - ALWAYS use `wavelength_range=(490, 680)`
   - Brightness is calculated AFTER wavelength filtering
   - Never use full spectrum for classification!

2. **Post-classification filtering unchanged:**
   - `cc_min_area=20` for all classes (uniform threshold)
   - Custom weights affect SVM training, NOT post-filtering

3. **Brightness units:**
   - Same units as corrected pseudo-reflectance
   - NO normalization applied
   - Raw mean value

4. **Storage:**
   - `self.svm_add_brightness_feature` stored in cube object
   - Automatically applied during classification
   - No need to specify again in `classify_segment_with_validation`

## 🚀 Usage Example

```python
# Define custom weights (2x for bombs)
custom_weights = {
    "training_bombs": 2.0,
    "training_dark": 1.0,
    "training_sediment": 1.0,
}

# Train SVM with custom weights + brightness
cv_results = cube.train_svm_with_cv(
    training_rois=["training_dark", "training_sediment", "training_bombs"],
    segment_start=100,
    segment_end=500,
    wavelength_range=(490, 680),       # ✅ CRITICAL: Filter wavelengths
    class_weight_dict=custom_weights,  # 🔥 Custom weights
    add_brightness_feature=True,       # 🔥 Enable brightness
    use_corrected=True,
    optimize_params=True,
)

# Classify segment (brightness automatically applied)
classification_results = cube.classify_segment_with_validation(
    segment_start=100,
    segment_end=500,
    validation_rois=["sediment", "dark spots", "all bombs"],
    validation_class_mapping={
        "sediment": "training_sediment",
        "dark spots": "training_dark",
        "all bombs": "training_bombs",
    },
    use_corrected=True,
    save_to_h5=True,
    apply_post_filtering=True,
    cc_min_area=20,  # Same for all classes
)

# Compare metrics
print(f"Bomb recall: {classification_results['validation_metrics']['recall_per_class']['training_bombs']:.3f}")
```

## 📈 Next Steps

1. Run test script or notebook cells to evaluate improvement
2. Adjust class weights if needed (try 1.5x, 2.5x, 3.0x for bombs)
3. Compare before/after visualizations
4. Check confusion matrix for misclassification patterns

## ✅ Implementation Complete

All features are implemented and ready to test! 🎉
