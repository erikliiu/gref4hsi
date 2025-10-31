# SVM Classification Workflow Guide

## 🎯 Overview

This workflow trains an SVM classifier on labeled pixels **outside** a target segment, then classifies pixels **inside** the segment, and evaluates performance using validation ROIs.

---

## 📋 Complete Workflow

### Step 1: Load and Prepare Data
```python
# Cell 1: Import modules
import importlib
import sys
import os

sys.path.append(os.path.abspath("../"))
from utils.gref_pipeline import georef
from gref_pipeline import config

importlib.reload(georef)
from utils.gref_pipeline.georef import *

# Cell 2: Load transect and select file
transect = load_transect(config.OUTPUT_FOLDER)
transect.list_files()

cube = transect.select_files(["rad_uhi_20241029_115057_5"])
cube.describe()

# Cell 3: Apply illumination correction
cube.apply_illumination_correction_v2()

# Cell 4: Load ROIs from JSON
cube.import_rois("./ROIs/057_5_combined.json")

# Cell 5: Define ROI lists
training_rois = [
    "training_dark",      # Pixels OUTSIDE segment
    "training_sediment",  # Pixels OUTSIDE segment
    "training_bombs",     # Pixels OUTSIDE segment
]

validation_rois = [
    "sediment",    # Pixels INSIDE segment
    "dark spots",  # Pixels INSIDE segment
    "all bombs",   # Pixels INSIDE segment
]
```

---

### Step 2: Load Full Datacube
```python
# Cell 6: Load the ENTIRE datacube (needed to extract training pixels from anywhere)
cube.load_data()  # Loads all tracks

print(f"Loaded datacube shape: {cube.data.shape}")
```

---

### Step 3: Train SVM with Cross-Validation
```python
# Cell 7: Train SVM on pixels OUTSIDE the segment
cv_results = cube.train_svm_with_cv(
    training_rois=training_rois,
    segment_start=config.UHI_TRACK_RANGE_5[0],
    segment_end=config.UHI_TRACK_RANGE_5[1],
    cv_folds=5,
    use_corrected=True,
    svm_kernel='rbf',
    optimize_params=True,
    quiet=False
)

# Print summary
print(f"\n📊 Cross-Validation Summary:")
print(f"Accuracy:  {cv_results['cv_mean_metrics']['accuracy_mean']:.3f} ± {cv_results['cv_mean_metrics']['accuracy_std']:.3f}")
print(f"Precision: {cv_results['cv_mean_metrics']['precision_mean']:.3f} ± {cv_results['cv_mean_metrics']['precision_std']:.3f}")
print(f"Recall:    {cv_results['cv_mean_metrics']['recall_mean']:.3f} ± {cv_results['cv_mean_metrics']['recall_std']:.3f}")
print(f"F1 Score:  {cv_results['cv_mean_metrics']['f1_mean']:.3f} ± {cv_results['cv_mean_metrics']['f1_std']:.3f}")

print(f"\n🎯 Best hyperparameters:")
print(f"C = {cv_results['best_params']['C']}")
print(f"gamma = {cv_results['best_params']['gamma']}")

print(f"\n📍 Training pixels used:")
for class_name, count in cv_results['training_pixels_per_class'].items():
    print(f"  {class_name}: {count} pixels")

print(f"\n🔍 Pixel filtering:")
print(f"  Kept (outside segment): {cv_results['filtered_pixels_outside']}")
print(f"  Rejected (inside segment): {cv_results['filtered_pixels_inside']}")
```

---

### Step 4: Visualize Training ROIs (After Filtering)
```python
# Cell 8: Plot the FILTERED training ROIs to see what was actually used
%matplotlib inline

cube.plot_georef(
    use_corrected=True,
    figsize=(40, 10),
    roi_collection=cv_results['filtered_training_rois'],  # ✅ Use filtered ROIs!
    roi_marker_size=3,
    roi_legend_loc="outside",
    roi_marker_edgewidth=0,
    roi_legend_markersize=40,
)
```

**This plot shows you EXACTLY which pixels were used for training** (after filtering out any pixels that were accidentally inside the segment).

---

### Step 5: Classify Segment with Validation
```python
# Cell 9: Classify the segment and validate
classification_results = cube.classify_segment_with_validation(
    segment_start=config.UHI_TRACK_RANGE_5[0],
    segment_end=config.UHI_TRACK_RANGE_5[1],
    validation_rois=validation_rois,
    use_corrected=True,
    save_to_h5=True,
    dataset_name="svm_classification_validated",
    quiet=False
)

# Print validation results
if classification_results['validation_metrics']:
    val_metrics = classification_results['validation_metrics']
    
    print(f"\n📈 Validation Results:")
    print(f"Overall Accuracy: {val_metrics['accuracy']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(val_metrics['confusion_matrix'])
    
    print(f"\nPer-Class Metrics:")
    for class_name in classification_results['class_names']:
        print(f"{class_name}:")
        print(f"  Precision: {val_metrics['precision_per_class'][class_name]:.3f}")
        print(f"  Recall:    {val_metrics['recall_per_class'][class_name]:.3f}")
        print(f"  F1 Score:  {val_metrics['f1_per_class'][class_name]:.3f}")
        print(f"  Support:   {val_metrics['support_per_class'][class_name]} pixels")
```

---

### Step 6: Visualize Validation ROIs (After Filtering)
```python
# Cell 10: Plot the FILTERED validation ROIs
cube.plot_georef(
    use_corrected=True,
    track_start=config.UHI_TRACK_RANGE_5[0],
    track_end=config.UHI_TRACK_RANGE_5[1],
    figsize=(40, 10),
    roi_collection=classification_results['filtered_validation_rois'],  # ✅ Use filtered ROIs!
    roi_marker_size=3,
    roi_legend_loc="outside",
    roi_marker_edgewidth=0,
    roi_legend_markersize=40,
)
```

**This plot shows you EXACTLY which validation pixels were used** (after filtering out any pixels outside the segment).

---

### Step 7: Visualize Classification Results
```python
# Cell 11: Plot pure classification map
cube.plot_classification_map(
    coordinate_system="NED",
    figsize=(40, 10),
    cmap="tab10",
    show_legend=True
)

# Cell 12: Plot RGB + classification overlay
cube.plot_classification_overlay(
    coordinate_system="NED",
    use_corrected=True,
    figsize=(40, 10),
    alpha=0.5,  # Transparency
    cmap="tab10",
    show_legend=True
)
```

---

## 📊 Understanding the Results

### Cross-Validation Results
- **Accuracy, Precision, Recall, F1**: Mean ± std across K folds
- **Confusion Matrices**: One per fold (stored in `cv_results['cv_results']['confusion_matrices']`)
- **Best Parameters**: Optimal C and gamma found by GridSearchCV

### Validation Results
- **Confusion Matrix**: 3×3 matrix comparing true vs predicted classes
- **Per-Class Metrics**: Precision, recall, F1 for each class
- **Support**: Number of validation pixels per class

### Filtered ROIs
- `filtered_training_rois`: Dict of {class_name: [(track, slit), ...]} for training
- `filtered_validation_rois`: Dict of {class_name: [(track, slit), ...]} for validation
- These can be passed directly to `plot_georef(roi_collection=...)`

---

## 🔍 Debugging Tips

### Check Pixel Filtering
```python
# See how many pixels were filtered
print(f"Training pixels kept: {cv_results['filtered_pixels_outside']}")
print(f"Training pixels rejected: {cv_results['filtered_pixels_inside']}")

print(f"\nValidation pixels kept: {classification_results['validation_pixels_inside']}")
print(f"Validation pixels rejected: {classification_results['validation_pixels_outside']}")
```

### Inspect Confusion Matrix
```python
import pandas as pd

# Create a nice confusion matrix DataFrame
cm = classification_results['validation_metrics']['confusion_matrix']
classes = classification_results['class_names']

cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print("\nConfusion Matrix:")
print(cm_df)
```

### Plot Individual Fold Results
```python
import matplotlib.pyplot as plt

# Plot accuracy across folds
folds = range(1, len(cv_results['cv_results']['accuracy']) + 1)
plt.figure(figsize=(10, 6))
plt.plot(folds, cv_results['cv_results']['accuracy'], marker='o', label='Accuracy')
plt.plot(folds, cv_results['cv_results']['precision'], marker='s', label='Precision')
plt.plot(folds, cv_results['cv_results']['recall'], marker='^', label='Recall')
plt.plot(folds, cv_results['cv_results']['f1'], marker='d', label='F1')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Cross-Validation Performance per Fold')
plt.legend()
plt.grid(True)
plt.ylim([0, 1])
plt.show()
```

---

## 🚀 Key Features

1. **Automatic Pixel Filtering**: Rejects training pixels inside segment and validation pixels outside segment
2. **Cross-Validation**: K-fold CV on training data for robust performance estimation
3. **Hyperparameter Optimization**: GridSearchCV finds best C and gamma
4. **Comprehensive Metrics**: Accuracy, precision, recall, F1, confusion matrices
5. **Visualization Support**: Filtered ROI coordinates can be plotted with `plot_georef()`
6. **H5 Storage**: Classification maps saved to H5 files for later use

---

## ❓ FAQ

**Q: What if I accidentally picked training pixels inside the segment?**  
A: The code automatically filters them out and prints a warning. Check `filtered_pixels_inside`.

**Q: Can I use different data sources for training and classification?**  
A: Yes, but use the same preprocessing! Both use `use_corrected=True/False`.

**Q: How do I choose the segment range?**  
A: Use `config.UHI_TRACK_RANGE_5[0]` and `config.UHI_TRACK_RANGE_5[1]` or specify custom track indices.

**Q: What if GridSearchCV is too slow?**  
A: Set `optimize_params=False` and provide manual `svm_C` and `svm_gamma` values.

**Q: Can I skip validation?**  
A: Yes, pass `validation_rois=[]` to `classify_segment_with_validation()`.
