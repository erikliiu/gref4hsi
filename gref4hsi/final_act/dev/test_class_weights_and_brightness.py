"""
Test script to compare SVM classification results:
- BEFORE: balanced class weights, no brightness feature
- AFTER: custom class weights (2x for bombs), brightness feature enabled

This demonstrates the improvement in bomb detection with custom weights and brightness.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))

from utils.gref_pipeline import georef
from gref_pipeline import config

# Import georef to access classes
from utils.gref_pipeline.georef import *

print("=" * 80)
print("🧪 TEST: Custom Class Weights + Brightness Feature")
print("=" * 80)

# Load transect and cube
print("\n📦 Loading data...")
transect = load_transect(config.OUTPUT_FOLDER)
cube = transect.select_files(["rad_uhi_20241029_115057_5"])
cube.apply_illumination_correction_v2()
cube.import_rois("./ROIs/057_5_combined.json")

# Define ROIs
training_rois = ["training_dark", "training_sediment", "training_bombs"]
validation_rois = ["sediment", "dark spots", "all bombs"]
validation_class_mapping = {
    "sediment": "training_sediment",
    "dark spots": "training_dark",
    "all bombs": "training_bombs",
}

segment_start = config.UHI_TRACK_RANGE_5[0]
segment_end = config.UHI_TRACK_RANGE_5[1]
wavelength_range = (490, 680)

print("\n" + "=" * 80)
print("📊 TEST 1: BASELINE (balanced weights, no brightness)")
print("=" * 80)

# Train SVM WITHOUT custom weights and brightness
cv_results_before = cube.train_svm_with_cv(
    training_rois=training_rois,
    segment_start=segment_start,
    segment_end=segment_end,
    wavelength_range=wavelength_range,
    cv_folds=5,
    use_corrected=True,
    svm_kernel="rbf",
    optimize_params=True,
    class_weight_dict=None,  # Use balanced (default)
    add_brightness_feature=False,  # No brightness
    quiet=False,
)

# Classify segment WITHOUT custom weights and brightness
classification_results_before = cube.classify_segment_with_validation(
    segment_start=segment_start,
    segment_end=segment_end,
    validation_rois=validation_rois,
    validation_class_mapping=validation_class_mapping,
    use_corrected=True,
    save_to_h5=False,  # Don't save to avoid overwriting
    apply_post_filtering=True,
    post_cc_bomb=True,
    post_cc_dark=True,
    post_cc_sediment=False,
    cc_connectivity=8,
    cc_min_area=20,
    morph_close_radius=1,
    morph_open_radius=1,
    merge_proximity_px=0,
    quiet=False,
)

print("\n" + "=" * 80)
print("📊 TEST 2: IMPROVED (custom weights 2x bombs, brightness enabled)")
print("=" * 80)

# Train SVM WITH custom weights and brightness
custom_weights = {
    "training_bombs": 2.0,  # 2x weight for bombs (higher recall)
    "training_dark": 1.0,
    "training_sediment": 1.0,
}

cv_results_after = cube.train_svm_with_cv(
    training_rois=training_rois,
    segment_start=segment_start,
    segment_end=segment_end,
    wavelength_range=wavelength_range,
    cv_folds=5,
    use_corrected=True,
    svm_kernel="rbf",
    optimize_params=True,
    class_weight_dict=custom_weights,  # 🔥 Custom weights
    add_brightness_feature=True,  # 🔥 Enable brightness
    quiet=False,
)

# Classify segment WITH custom weights and brightness
classification_results_after = cube.classify_segment_with_validation(
    segment_start=segment_start,
    segment_end=segment_end,
    validation_rois=validation_rois,
    validation_class_mapping=validation_class_mapping,
    use_corrected=True,
    save_to_h5=True,  # Save final result
    dataset_name="svm_classification_with_custom_weights",
    apply_post_filtering=True,
    post_cc_bomb=True,
    post_cc_dark=True,
    post_cc_sediment=False,
    cc_connectivity=8,
    cc_min_area=20,
    morph_close_radius=1,
    morph_open_radius=1,
    merge_proximity_px=0,
    quiet=False,
)

# ============================================================================
# COMPARISON: Print metrics side-by-side
# ============================================================================
print("\n" + "=" * 80)
print("📈 COMPARISON: Validation Metrics")
print("=" * 80)

metrics_before = classification_results_before["validation_metrics"]
metrics_after = classification_results_after["validation_metrics"]

print(f"\n{'Metric':<30} {'BEFORE':>15} {'AFTER':>15} {'CHANGE':>15}")
print("-" * 80)
print(
    f"{'Overall Accuracy':<30} {metrics_before['accuracy']:>15.3f} {metrics_after['accuracy']:>15.3f} {metrics_after['accuracy']-metrics_before['accuracy']:>+15.3f}"
)

print("\n📊 Per-Class Metrics:")
for class_name in cv_results_before["class_names"]:
    print(f"\n{class_name}:")
    print(
        f"{'  Precision':<30} {metrics_before['precision_per_class'][class_name]:>15.3f} {metrics_after['precision_per_class'][class_name]:>15.3f} {metrics_after['precision_per_class'][class_name]-metrics_before['precision_per_class'][class_name]:>+15.3f}"
    )
    print(
        f"{'  Recall':<30} {metrics_before['recall_per_class'][class_name]:>15.3f} {metrics_after['recall_per_class'][class_name]:>15.3f} {metrics_after['recall_per_class'][class_name]-metrics_before['recall_per_class'][class_name]:>+15.3f}"
    )
    print(
        f"{'  F1 Score':<30} {metrics_before['f1_per_class'][class_name]:>15.3f} {metrics_after['f1_per_class'][class_name]:>15.3f} {metrics_after['f1_per_class'][class_name]-metrics_before['f1_per_class'][class_name]:>+15.3f}"
    )
    print(
        f"{'  Support':<30} {metrics_before['support_per_class'][class_name]:>15} {metrics_after['support_per_class'][class_name]:>15}"
    )

# ============================================================================
# VISUALIZATION: Plot before/after classification maps
# ============================================================================
print("\n" + "=" * 80)
print("📊 Generating comparison visualizations...")
print("=" * 80)


def classification_map_to_roi_collection(
    classification_map, segment_start, class_names
):
    """Convert classification map to ROI collection format for plot_georef."""
    roi_collection = {}

    for class_name in class_names:
        if not class_name:  # Skip empty strings
            continue

        # Find pixels of this class
        class_mask = classification_map == class_name
        track_indices, slit_indices = np.where(class_mask)

        # Convert to absolute track indices (add segment_start offset)
        track_indices_abs = track_indices + segment_start

        # Store as (slit, track) tuples
        pixels = [(slit, track) for slit, track in zip(slit_indices, track_indices_abs)]

        if len(pixels) > 0:
            roi_collection[class_name] = pixels

    return roi_collection


# Create ROI collections for before/after
roi_before = classification_map_to_roi_collection(
    classification_results_before["classification_map"],
    segment_start,
    cv_results_before["class_names"],
)

roi_after = classification_map_to_roi_collection(
    classification_results_after["classification_map"],
    segment_start,
    cv_results_after["class_names"],
)

# Print pixel counts
print("\n📊 Pixel counts comparison:")
print(f"{'Class':<25} {'BEFORE':>10} {'AFTER':>10} {'CHANGE':>10} {'% Change':>12}")
print("-" * 75)
for class_name in cv_results_before["class_names"]:
    before_count = len(roi_before.get(class_name, []))
    after_count = len(roi_after.get(class_name, []))
    change = after_count - before_count
    pct_change = (change / before_count * 100) if before_count > 0 else 0
    print(
        f"{class_name:<25} {before_count:>10} {after_count:>10} {change:>+10} {pct_change:>+11.1f}%"
    )

# Plot BEFORE (balanced weights, no brightness)
print("\n🔍 Plotting BEFORE (balanced weights, no brightness)...")
result_before = cube.plot_georef(
    use_corrected=True,
    track_start=segment_start,
    track_end=segment_end,
    figsize=(40, 10),
    roi_collection=roi_before,
    roi_marker_size=3,
    roi_legend_loc="outside",
    roi_marker_edgewidth=0,
    roi_legend_markersize=40,
    return_fig=True,
)
fig_before, ax_before = result_before
fig_before.suptitle(
    "BEFORE: Balanced Weights + No Brightness", fontsize=16, weight="bold", y=0.98
)
plt.show()

# Plot AFTER (custom weights + brightness)
print("\n✨ Plotting AFTER (custom weights 2x bombs + brightness)...")
result_after = cube.plot_georef(
    use_corrected=True,
    track_start=segment_start,
    track_end=segment_end,
    figsize=(40, 10),
    roi_collection=roi_after,
    roi_marker_size=3,
    roi_legend_loc="outside",
    roi_marker_edgewidth=0,
    roi_legend_markersize=40,
    return_fig=True,
)
fig_after, ax_after = result_after
fig_after.suptitle(
    "AFTER: Custom Weights (2x Bombs) + Brightness Feature",
    fontsize=16,
    weight="bold",
    y=0.98,
)
plt.show()

print("\n" + "=" * 80)
print("✅ TEST COMPLETE!")
print("=" * 80)
print("\n🎯 Key Findings:")
print(
    f"   - Overall accuracy change: {metrics_after['accuracy']-metrics_before['accuracy']:+.3f}"
)
print(
    f"   - Bomb recall change: {metrics_after['recall_per_class']['training_bombs']-metrics_before['recall_per_class']['training_bombs']:+.3f}"
)
print(
    f"   - Bomb F1 change: {metrics_after['f1_per_class']['training_bombs']-metrics_before['f1_per_class']['training_bombs']:+.3f}"
)
print(
    f"   - Bomb pixel count change: {len(roi_after.get('training_bombs', []))-len(roi_before.get('training_bombs', []))}"
)
