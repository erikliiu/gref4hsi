# Clean Function Separation - SVM Classification Pipeline

## 🎯 Problem Solved

**Before:** Messy monolithic functions that did too much
- `train_svm_with_cv()` - Did training + spatial clustering (confusing name)
- `classify_segment_with_validation()` - Did classification + filtering + validation (too much!)

**After:** Clean separation of concerns with descriptive names

---

## 📋 New Clean API

### **1. Training Phase**

```python
# Train SVM with custom weights + brightness
cv_results = cube.train_svm_with_cv(
    training_rois=training_rois,
    segment_start=100,
    segment_end=500,
    wavelength_range=(490, 680),
    class_weight_dict={"training_bombs": 2.0},  # Custom weights
    add_brightness_feature=True,                 # Brightness feature
    use_corrected=True,
    optimize_params=True,
)
```

**What it does:**
- Creates spatial training groups (bomb#1, bomb#2, etc.)
- Trains SVM with Leave-One-Bomb-Out CV
- Returns model + CV metrics + spatial groups

---

### **2. Classification Phase (3 separate steps)**

#### **Step 1: Classify** (pure classification, no filtering)

```python
classification_results = cube.classify_segment(
    segment_start=100,
    segment_end=500,
    use_corrected=True,
)
```

**Returns:**
- `classification_map` - 2D array of class names
- `classification_map_encoded` - 2D array of encoded labels
- `class_names` - List of class names
- `segment_shape` - Shape of segment
- `track_range` - (start, end)

---

#### **Step 2: Filter** (optional post-processing)

```python
filtering_results = cube.filter_classification(
    classification_map=classification_results["classification_map"],
    class_names=classification_results["class_names"],
    filter_bombs=True,      # Apply filtering to bombs
    filter_dark=True,       # Apply filtering to dark spots
    filter_sediment=False,  # Don't filter sediment
    min_area_px=20,         # Minimum pixels per group
    connectivity=8,         # 8-connectivity
)
```

**What it does:**
- Removes isolated pixels (connected-component analysis)
- Fills small holes (morphological closing)
- Removes small protrusions (morphological opening)
- Optionally merges nearby components

**Returns:**
- `filtered_map` - Cleaned classification map
- `original_map` - Copy of original (for comparison)
- `pixels_removed_per_class` - Dict of removed counts

**Parameters explained:**
- `min_area_px` - Groups smaller than this are removed
- `connectivity` - 4 (orthogonal only) or 8 (diagonal OK)
- `morph_close_radius` - Fill holes this size (0 = disabled)
- `morph_open_radius` - Remove protrusions this size (0 = disabled)
- `merge_proximity_px` - Merge groups closer than this (0 = disabled)

---

#### **Step 3: Validate** (calculate accuracy metrics)

```python
validation_results = cube.validate_classification(
    classification_map=filtering_results["filtered_map"],
    segment_start=100,
    segment_end=500,
    validation_rois=validation_rois,
    validation_class_mapping={
        "sediment": "training_sediment",
        "dark spots": "training_dark",
        "all bombs": "training_bombs",
    },
)
```

**Returns:**
- `accuracy` - Overall accuracy
- `confusion_matrix` - Confusion matrix
- `precision_per_class` - Dict {class_name: precision}
- `recall_per_class` - Dict {class_name: recall}
- `f1_per_class` - Dict {class_name: F1 score}
- `support_per_class` - Dict {class_name: pixel count}
- `filtered_validation_rois` - Dict of validation pixels used

---

## 🔄 Complete Workflow Example

```python
# ============================================================================
# TRAINING
# ============================================================================
custom_weights = {
    "training_bombs": 2.0,
    "training_dark": 1.0,
    "training_sediment": 1.0,
}

cv_results = cube.train_svm_with_cv(
    training_rois=["training_dark", "training_sediment", "training_bombs"],
    segment_start=100,
    segment_end=500,
    wavelength_range=(490, 680),
    class_weight_dict=custom_weights,
    add_brightness_feature=True,
    use_corrected=True,
)

# ============================================================================
# CLASSIFICATION (3 clean steps)
# ============================================================================

# Step 1: Classify
classification_results = cube.classify_segment(
    segment_start=100,
    segment_end=500,
    use_corrected=True,
)

# Step 2: Filter (optional)
filtering_results = cube.filter_classification(
    classification_map=classification_results["classification_map"],
    class_names=classification_results["class_names"],
    filter_bombs=True,
    filter_dark=True,
    filter_sediment=False,
    min_area_px=20,
)

# Step 3: Validate (optional)
validation_results = cube.validate_classification(
    classification_map=filtering_results["filtered_map"],
    segment_start=100,
    segment_end=500,
    validation_rois=["sediment", "dark spots", "all bombs"],
    validation_class_mapping={
        "sediment": "training_sediment",
        "dark spots": "training_dark",
        "all bombs": "training_bombs",
    },
)

# ============================================================================
# VISUALIZATION
# ============================================================================
cube.classification_map = filtering_results["filtered_map"]
cube.plot_classification_map(figsize=(40, 10))
cube.plot_classification_overlay(figsize=(40, 10), alpha=0.1)
```

---

## 🆚 Old vs New

### **Old Way (messy, all-in-one):**

```python
# Everything jammed into one function!
results = cube.classify_segment_with_validation(
    segment_start=100,
    segment_end=500,
    validation_rois=validation_rois,
    validation_class_mapping={...},
    save_to_h5=True,
    apply_post_filtering=True,      # ← Confusing
    post_cc_bomb=True,              # ← What is cc?
    post_cc_dark=True,              # ← What is cc?
    cc_connectivity=8,              # ← What is cc?
    cc_min_area=20,                 # ← What is cc?
    morph_close_radius=1,           # ← What is morph?
    morph_open_radius=1,            # ← What is morph?
    merge_proximity_px=0,           # ← Too many parameters!
)
```

### **New Way (clean, separated):**

```python
# Step 1: Classify (clear purpose)
classification_results = cube.classify_segment(...)

# Step 2: Filter (clear purpose)
filtering_results = cube.filter_classification(
    classification_map=classification_results["classification_map"],
    class_names=classification_results["class_names"],
    filter_bombs=True,              # ← Clear!
    filter_dark=True,               # ← Clear!
    min_area_px=20,                 # ← Clear!
)

# Step 3: Validate (clear purpose)
validation_results = cube.validate_classification(...)
```

---

## 💡 Benefits

**1. Clear separation of concerns**
- Each function does ONE thing
- Easier to understand what's happening

**2. Flexible workflow**
- Skip filtering if you don't want it
- Skip validation if you don't need it
- Compare filtered vs unfiltered easily

**3. Better parameter names**
- `filter_bombs` instead of `post_cc_bomb`
- `min_area_px` instead of `cc_min_area`
- `connectivity` instead of `cc_connectivity`

**4. Backwards compatible**
- Old `classify_segment_with_validation()` still works!
- Just uses new functions internally

---

## 📝 Parameter Glossary

### **Filtering Parameters:**

| Parameter | What It Does | Common Values |
|-----------|--------------|---------------|
| `filter_bombs` | Apply filtering to bomb pixels | `True` |
| `filter_dark` | Apply filtering to dark pixels | `True` |
| `filter_sediment` | Apply filtering to sediment pixels | `False` |
| `min_area_px` | Minimum pixels per group (smaller = removed) | `10-30` |
| `connectivity` | How pixels connect: 4 (orthogonal) or 8 (diagonal) | `8` |
| `morph_close_radius` | Fill holes this size (0 = disabled) | `1-3` or `0` |
| `morph_open_radius` | Remove protrusions this size (0 = disabled) | `1-3` or `0` |
| `merge_proximity_px` | Merge groups closer than this (0 = disabled) | `0-5` or `0` |

### **Training Parameters:**

| Parameter | What It Does | Common Values |
|-----------|--------------|---------------|
| `class_weight_dict` | Custom class weights | `{"training_bombs": 2.0}` |
| `add_brightness_feature` | Add mean intensity feature | `True` or `False` |
| `wavelength_range` | Filter wavelengths | `(490, 680)` |
| `use_spatial_groups` | Enable spatial clustering for CV | `True` |
| `optimize_params` | GridSearchCV for C and gamma | `True` |

---

## ✅ Summary

**You now have:**
- ✅ Clean function names that describe what they do
- ✅ Separation of classification, filtering, and validation
- ✅ Flexible workflow (skip steps if needed)
- ✅ Better parameter names (no more "cc" or "morph" confusion)
- ✅ Old code still works (backwards compatible)

**Updated notebook (dev19) uses the new clean API! 🎉**
