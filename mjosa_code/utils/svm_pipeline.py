"""
SVM Classification Pipeline Wrapper Functions

This module provides clean wrapper functions for the complete SVM classification workflow,
including training, cross-validation, classification, filtering, merging, validation, and visualization.

Based on notebooks 6_svm_057.ipynb and 7_svm_057_no_unknown.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import cohen_kappa_score


def setup_and_load_data(
    transect_path: str,
    file_name: str,
    roi_file: str,
    smooth_sigma: Optional[float] = None,
    l2_normalize: bool = False,
    window_size: int = 500,
    apply_correction: bool = True,
    use_smoothed_input: bool = False,
):
    """
    Load hyperspectral cube, apply corrections, and load ROI data.

    Parameters:
    -----------
    transect_path : str
        Path to the transect directory
    file_name : str
        Name of the HDF5 file (without extension)
    roi_file : str
        Name of the ROI JSON file (e.g., 'my_rois_6.json')
    smooth_sigma : float, optional
        Gaussian smoothing sigma for visualization
    l2_normalize : bool, default=False
        Whether to apply L2 normalization for visualization
    window_size : int, default=500
        Window size for illumination correction
    apply_correction : bool, default=True
        Whether to apply illumination correction
    use_smoothed_input : bool, default=False
        Whether to smooth raw data before illumination correction

    Returns:
    --------
    dict : Dictionary containing:
        - 'cube': HyperspectralCube object
        - 'roi_data': Dictionary of ROI data
        - 'roi_file_path': Path to ROI file
    """
    from mjosa_code.utils.uhi.georef import HyperspectralCube, load_roi_data
    import os

    print("=" * 80)
    print("SETUP AND LOAD DATA")
    print("=" * 80)

    # Initialize cube
    print(f"\n[1/4] Loading hyperspectral cube: {file_name}")
    cube = HyperspectralCube(transect_path, file_name)
    print(f"      Cube shape: {cube.data.shape}")
    print(
        f"      Bands: {cube.data.shape[0]}, Pixels: {cube.data.shape[1]} x {cube.data.shape[2]}"
    )

    # Apply illumination correction if requested
    if apply_correction:
        print(
            f"\n[2/4] Applying illumination correction (window={window_size}, use_smoothed={use_smoothed_input})..."
        )
        cube.apply_illumination_correction_v2(
            window_size=window_size,
            use_cache=True,
            use_smoothed_input=use_smoothed_input,
        )
        print("      ✓ Illumination correction applied")
    else:
        print("\n[2/4] Skipping illumination correction")

    # Load ROI data
    roi_path = os.path.join(transect_path, roi_file)
    print(f"\n[3/4] Loading ROI data: {roi_file}")
    roi_data = load_roi_data(roi_path)
    print(f"      Found {len(roi_data)} ROI classes:")
    for class_name, rois in roi_data.items():
        total_pixels = sum(len(roi["pixel_indices"]) for roi in rois)
        print(f"        - {class_name}: {len(rois)} ROIs, {total_pixels} pixels")

    # Set visualization parameters
    print(f"\n[4/4] Setting visualization parameters")
    if smooth_sigma is not None:
        print(f"      Gaussian smoothing: sigma={smooth_sigma}")
    if l2_normalize:
        print(f"      L2 normalization: enabled")

    print("\n" + "=" * 80)
    print("✓ Setup complete")
    print("=" * 80 + "\n")

    return {"cube": cube, "roi_data": roi_data, "roi_file_path": roi_path}


def train_svm_with_cv(
    cube,
    roi_data: Dict,
    training_classes: List[str],
    cv_folds: int = 5,
    C: float = 10.0,
    gamma: str = "scale",
    random_state: int = 42,
):
    """
    Train SVM classifier with cross-validation.

    Parameters:
    -----------
    cube : HyperspectralCube
        Hyperspectral cube object
    roi_data : dict
        Dictionary of ROI data from load_roi_data()
    training_classes : list of str
        List of class names to use for training (e.g., ['sediment', 'macroalgae', 'water'])
    cv_folds : int, default=5
        Number of cross-validation folds
    C : float, default=10.0
        SVM regularization parameter
    gamma : str or float, default='scale'
        Kernel coefficient for RBF
    random_state : int, default=42
        Random state for reproducibility

    Returns:
    --------
    dict : Dictionary containing:
        - 'model': Trained SVM model
        - 'cv_results': Cross-validation results from train_svm_with_cv
        - 'training_classes': List of training class names
    """
    from mjosa_code.utils.uhi.georef import train_svm_with_cv

    print("=" * 80)
    print("TRAIN SVM WITH CROSS-VALIDATION")
    print("=" * 80)

    print(f"\n[1/3] Training configuration:")
    print(f"      Classes: {training_classes}")
    print(f"      CV folds: {cv_folds}")
    print(f"      SVM parameters: C={C}, gamma={gamma}")
    print(f"      Random state: {random_state}")

    print(f"\n[2/3] Training SVM with {cv_folds}-fold cross-validation...")
    model, cv_results = train_svm_with_cv(
        cube=cube,
        roi_data=roi_data,
        class_names=training_classes,
        n_folds=cv_folds,
        C=C,
        gamma=gamma,
        random_state=random_state,
    )

    print(f"\n[3/3] Cross-validation results:")
    print(
        f"      Overall Accuracy: {cv_results['cv_mean_accuracy']:.4f} ± {cv_results['cv_std_accuracy']:.4f}"
    )
    print(f"\n      Per-class metrics:")
    for class_name in training_classes:
        precision = cv_results["cv_mean_metrics"][class_name]["precision"]
        recall = cv_results["cv_mean_metrics"][class_name]["recall"]
        f1 = cv_results["cv_mean_metrics"][class_name]["f1-score"]
        print(
            f"        {class_name:12s}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
        )

    # Compute Macro F1
    macro_f1 = np.mean(
        [cv_results["cv_mean_metrics"][c]["f1-score"] for c in training_classes]
    )
    print(f"\n      Macro F1 Score: {macro_f1:.4f}")

    # Check for Kappa if available
    if "kappa" in cv_results.get("cv_mean_metrics", {}).get(training_classes[0], {}):
        kappas = [
            cv_results["cv_mean_metrics"][c].get("kappa", 0) for c in training_classes
        ]
        mean_kappa = np.mean(kappas)
        print(f"      Cohen's Kappa: {mean_kappa:.4f}")

        # Kappa interpretation
        if mean_kappa > 0.8:
            interpretation = "Strong agreement"
        elif mean_kappa > 0.6:
            interpretation = "Substantial agreement"
        elif mean_kappa > 0.4:
            interpretation = "Moderate agreement"
        elif mean_kappa > 0.2:
            interpretation = "Fair agreement"
        else:
            interpretation = "Slight agreement"
        print(f"                     ({interpretation})")

    print("\n" + "=" * 80)
    print("✓ Training complete")
    print("=" * 80 + "\n")

    return {
        "model": model,
        "cv_results": cv_results,
        "training_classes": training_classes,
        "macro_f1": macro_f1,
    }


def classify_segment(
    cube,
    model,
    confidence_threshold: float = 0.0,
    class_names: Optional[List[str]] = None,
):
    """
    Classify the entire hyperspectral cube.

    Parameters:
    -----------
    cube : HyperspectralCube
        Hyperspectral cube object
    model : sklearn SVM
        Trained SVM model
    confidence_threshold : float, default=0.0
        Confidence threshold (0.0 = force all pixels into classes)
    class_names : list of str, optional
        List of class names (for display only)

    Returns:
    --------
    dict : Dictionary containing:
        - 'classification_map': 2D array of class labels
        - 'confidence_map': 2D array of classification confidence scores
    """
    from mjosa_code.utils.uhi.georef import classify_segment

    print("=" * 80)
    print("CLASSIFY HYPERSPECTRAL CUBE")
    print("=" * 80)

    print(f"\n[1/2] Classification configuration:")
    print(f"      Confidence threshold: {confidence_threshold}")
    if class_names:
        print(f"      Expected classes: {class_names}")

    print(f"\n[2/2] Classifying {cube.data.shape[1]} x {cube.data.shape[2]} pixels...")
    classification_map, confidence_map = classify_segment(
        cube=cube, model=model, confidence_threshold=confidence_threshold
    )

    # Count pixels per class
    unique, counts = np.unique(classification_map, return_counts=True)
    print(f"\n      Classification results:")
    for label, count in zip(unique, counts):
        percentage = (count / classification_map.size) * 100
        if label == -1:
            print(
                f"        Unknown/Low confidence: {count:7d} pixels ({percentage:5.2f}%)"
            )
        else:
            class_name = (
                class_names[label]
                if class_names and label < len(class_names)
                else f"Class {label}"
            )
            print(f"        {class_name:20s}: {count:7d} pixels ({percentage:5.2f}%)")

    print("\n" + "=" * 80)
    print("✓ Classification complete")
    print("=" * 80 + "\n")

    return {"classification_map": classification_map, "confidence_map": confidence_map}


def filter_classification(
    classification_map: np.ndarray,
    min_area_px: int = 50,
    class_names: Optional[List[str]] = None,
):
    """
    Apply morphological filtering to remove small isolated regions.

    Parameters:
    -----------
    classification_map : np.ndarray
        2D array of class labels from classify_segment
    min_area_px : int, default=50
        Minimum area in pixels for keeping a region
    class_names : list of str, optional
        List of class names (for display only)

    Returns:
    --------
    dict : Dictionary containing:
        - 'filtered_map': 2D array with small regions removed (-1 for removed)
        - 'removal_stats': Dictionary with removal statistics per class
    """
    from mjosa_code.utils.uhi.georef import filter_classification

    print("=" * 80)
    print("MORPHOLOGICAL FILTERING")
    print("=" * 80)

    print(f"\n[1/2] Filtering configuration:")
    print(f"      Minimum area: {min_area_px} pixels")

    print(f"\n[2/2] Applying morphological filtering...")
    filtered_map = filter_classification(
        classification_map=classification_map, min_area_px=min_area_px
    )

    # Calculate removal statistics
    print(f"\n      Filtering results:")
    unique_classes = np.unique(classification_map[classification_map != -1])
    removal_stats = {}

    for label in unique_classes:
        original_count = np.sum(classification_map == label)
        filtered_count = np.sum(filtered_map == label)
        removed_count = original_count - filtered_count
        removed_percentage = (
            (removed_count / original_count * 100) if original_count > 0 else 0
        )

        class_name = (
            class_names[label]
            if class_names and label < len(class_names)
            else f"Class {label}"
        )
        removal_stats[class_name] = {
            "original": original_count,
            "filtered": filtered_count,
            "removed": removed_count,
            "removed_pct": removed_percentage,
        }

        print(
            f"        {class_name:20s}: {removed_count:6d} pixels removed ({removed_percentage:5.2f}%)"
        )

    total_removed = np.sum(filtered_map == -1) - np.sum(classification_map == -1)
    print(f"\n      Total pixels removed: {total_removed}")

    print("\n" + "=" * 80)
    print("✓ Filtering complete")
    print("=" * 80 + "\n")

    return {"filtered_map": filtered_map, "removal_stats": removal_stats}


def merge_filtered_to_sediment(
    filtered_map: np.ndarray,
    sediment_label: int = 0,
    class_names: Optional[List[str]] = None,
):
    """
    Merge filtered-out pixels (-1) into sediment class.

    Parameters:
    -----------
    filtered_map : np.ndarray
        2D array from filter_classification with -1 for removed regions
    sediment_label : int, default=0
        Label value for sediment class
    class_names : list of str, optional
        List of class names (for display only)

    Returns:
    --------
    dict : Dictionary containing:
        - 'merged_map': 2D array with -1 pixels merged to sediment
        - 'merge_count': Number of pixels merged
    """
    from mjosa_code.utils.uhi.georef import merge_filtered_to_sediment

    print("=" * 80)
    print("MERGE FILTERED REGIONS TO SEDIMENT")
    print("=" * 80)

    sediment_name = (
        class_names[sediment_label]
        if class_names and sediment_label < len(class_names)
        else f"Class {sediment_label}"
    )

    print(f"\n[1/2] Merge configuration:")
    print(f"      Target class: {sediment_name} (label={sediment_label})")

    pixels_to_merge = np.sum(filtered_map == -1)
    print(f"\n[2/2] Merging {pixels_to_merge} filtered pixels to {sediment_name}...")

    merged_map = merge_filtered_to_sediment(
        filtered_map=filtered_map, sediment_label=sediment_label
    )

    # Verify merge
    remaining_unknown = np.sum(merged_map == -1)
    actual_merged = pixels_to_merge - remaining_unknown

    print(f"\n      Merge results:")
    print(f"        Pixels merged: {actual_merged}")
    print(f"        Remaining -1: {remaining_unknown}")

    # Final class distribution
    unique, counts = np.unique(merged_map, return_counts=True)
    print(f"\n      Final classification distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / merged_map.size) * 100
        if label == -1:
            print(f"        Unknown: {count:7d} pixels ({percentage:5.2f}%)")
        else:
            class_name = (
                class_names[label]
                if class_names and label < len(class_names)
                else f"Class {label}"
            )
            print(f"        {class_name:20s}: {count:7d} pixels ({percentage:5.2f}%)")

    print("\n" + "=" * 80)
    print("✓ Merge complete")
    print("=" * 80 + "\n")

    return {"merged_map": merged_map, "merge_count": actual_merged}


def validate_classification(
    cube,
    classification_map: np.ndarray,
    roi_data: Dict,
    validation_classes: List[str],
    class_names: List[str],
):
    """
    Validate classification results against validation ROIs.

    Parameters:
    -----------
    cube : HyperspectralCube
        Hyperspectral cube object
    classification_map : np.ndarray
        2D array of final class labels (after filtering and merging)
    roi_data : dict
        Dictionary of ROI data from load_roi_data()
    validation_classes : list of str
        List of class names to use for validation
    class_names : list of str
        List of all class names (for label mapping)

    Returns:
    --------
    dict : Dictionary containing:
        - 'confusion_matrix': Confusion matrix
        - 'overall_accuracy': Overall accuracy
        - 'per_class_metrics': Precision, recall, F1 per class
        - 'cohens_kappa': Cohen's Kappa score
        - 'macro_f1': Macro-averaged F1 score
        - 'y_true': True labels
        - 'y_pred': Predicted labels
    """
    from mjosa_code.utils.uhi.georef import validate_classification

    print("=" * 80)
    print("VALIDATE CLASSIFICATION")
    print("=" * 80)

    print(f"\n[1/3] Validation configuration:")
    print(f"      Validation classes: {validation_classes}")

    print(f"\n[2/3] Extracting validation pixels and computing metrics...")
    confusion_matrix, overall_accuracy, per_class_metrics = validate_classification(
        cube=cube,
        classification_map=classification_map,
        roi_data=roi_data,
        validation_classes=validation_classes,
        class_names=class_names,
    )

    print(f"\n[3/3] Validation results:")
    print(
        f"\n      Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)"
    )

    # Reconstruct y_true and y_pred from confusion matrix
    y_true = []
    y_pred = []
    for true_idx in range(len(validation_classes)):
        for pred_idx in range(len(validation_classes)):
            count = confusion_matrix[true_idx, pred_idx]
            y_true.extend([true_idx] * count)
            y_pred.extend([pred_idx] * count)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"      Cohen's Kappa: {kappa:.4f}")

    # Kappa interpretation
    if kappa > 0.8:
        interpretation = "Strong agreement"
    elif kappa > 0.6:
        interpretation = "Substantial agreement"
    elif kappa > 0.4:
        interpretation = "Moderate agreement"
    elif kappa > 0.2:
        interpretation = "Fair agreement"
    else:
        interpretation = "Slight agreement"
    print(f"                     ({interpretation})")

    # Per-class metrics
    print(f"\n      Per-class metrics:")
    f1_scores = []
    for class_name in validation_classes:
        metrics = per_class_metrics[class_name]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1-score"]
        support = metrics["support"]
        f1_scores.append(f1)
        print(
            f"        {class_name:12s}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} (n={support})"
        )

    # Macro F1
    macro_f1 = np.mean(f1_scores)
    print(f"\n      Macro F1 Score: {macro_f1:.4f}")

    # Confusion matrix
    print(f"\n      Confusion Matrix:")
    print(f"        {'':12s} | " + " | ".join([f"{c:>8s}" for c in validation_classes]))
    print(f"        {'-'*12} | " + " | ".join(["-" * 8 for _ in validation_classes]))
    for i, true_class in enumerate(validation_classes):
        row_str = f"        {true_class:12s} | "
        row_str += " | ".join(
            [f"{confusion_matrix[i, j]:8d}" for j in range(len(validation_classes))]
        )
        print(row_str)

    # Unknown/rejected pixels analysis
    unknown_count = np.sum(classification_map == -1)
    if unknown_count > 0:
        unknown_pct = (unknown_count / classification_map.size) * 100
        print(
            f"\n      Note: {unknown_count} pixels ({unknown_pct:.2f}%) remain unclassified (label=-1)"
        )

    # Paper-ready summary
    print(f"\n" + "=" * 80)
    print("PAPER-READY SUMMARY")
    print("=" * 80)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f} ({interpretation})")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"\nPer-class Performance:")
    for class_name in validation_classes:
        metrics = per_class_metrics[class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1-score']:.4f}")

    print("\n" + "=" * 80)
    print("✓ Validation complete")
    print("=" * 80 + "\n")

    return {
        "confusion_matrix": confusion_matrix,
        "overall_accuracy": overall_accuracy,
        "per_class_metrics": per_class_metrics,
        "cohens_kappa": kappa,
        "kappa_interpretation": interpretation,
        "macro_f1": macro_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_all_results(
    cube,
    roi_data: Dict,
    roi_file_path: str,
    training_classes: List[str],
    validation_classes: List[str],
    classification_map: np.ndarray,
    filtered_map: np.ndarray,
    merged_map: np.ndarray,
    confusion_matrix: np.ndarray,
    class_names: List[str],
    smooth_sigma: Optional[float] = None,
    l2_normalize: bool = False,
):
    """
    Generate all visualization plots.

    Parameters:
    -----------
    cube : HyperspectralCube
        Hyperspectral cube object
    roi_data : dict
        Dictionary of ROI data
    roi_file_path : str
        Path to ROI file
    training_classes : list of str
        List of training class names
    validation_classes : list of str
        List of validation class names
    classification_map : np.ndarray
        Initial classification map
    filtered_map : np.ndarray
        Classification map after filtering
    merged_map : np.ndarray
        Final classification map after merging
    confusion_matrix : np.ndarray
        Confusion matrix from validation
    class_names : list of str
        List of all class names
    smooth_sigma : float, optional
        Gaussian smoothing sigma
    l2_normalize : bool, default=False
        Whether to apply L2 normalization

    Returns:
    --------
    None (displays plots)
    """
    from mjosa_code.utils.uhi.georef import (
        plot_georef_with_roi_overlay,
        plot_confusion_matrix,
    )

    print("=" * 80)
    print("GENERATE VISUALIZATIONS")
    print("=" * 80)

    # Plot 1: Training ROIs
    print(f"\n[1/7] Plotting training ROIs overlay...")
    plot_georef_with_roi_overlay(
        cube=cube,
        roi_file_path=roi_file_path,
        roi_classes=training_classes,
        roi_overlay_mode="solid",
        title=f'Training ROIs: {", ".join(training_classes)}',
        smooth_sigma=smooth_sigma,
        l2_normalize=l2_normalize,
    )

    # Plot 2: Initial classification
    print(f"\n[2/7] Plotting initial classification map...")
    cube.plot_georef_wavelength(
        classification_map=classification_map,
        class_names=class_names,
        title="Initial Classification",
        smooth_sigma=smooth_sigma,
        l2_normalize=l2_normalize,
    )

    # Plot 3: Filtered classification
    print(f"\n[3/7] Plotting filtered classification map...")
    cube.plot_georef_wavelength(
        classification_map=filtered_map,
        class_names=class_names,
        title="After Morphological Filtering",
        smooth_sigma=smooth_sigma,
        l2_normalize=l2_normalize,
    )

    # Plot 4: Merged classification
    print(f"\n[4/7] Plotting merged classification map...")
    cube.plot_georef_wavelength(
        classification_map=merged_map,
        class_names=class_names,
        title="Final Classification (Filtered Regions Merged to Sediment)",
        smooth_sigma=smooth_sigma,
        l2_normalize=l2_normalize,
    )

    # Plot 5: Validation ROIs
    print(f"\n[5/7] Plotting validation ROIs overlay...")
    plot_georef_with_roi_overlay(
        cube=cube,
        roi_file_path=roi_file_path,
        roi_classes=validation_classes,
        roi_overlay_mode="solid",
        title=f'Validation ROIs: {", ".join(validation_classes)}',
        smooth_sigma=smooth_sigma,
        l2_normalize=l2_normalize,
    )

    # Plot 6: Final classification with validation overlay
    print(f"\n[6/7] Plotting final classification with validation ROIs...")
    plot_georef_with_roi_overlay(
        cube=cube,
        classification_map=merged_map,
        class_names=class_names,
        roi_file_path=roi_file_path,
        roi_classes=validation_classes,
        roi_overlay_mode="solid",
        title="Final Classification with Validation ROIs",
        smooth_sigma=smooth_sigma,
        l2_normalize=l2_normalize,
    )

    # Plot 7: Confusion matrix
    print(f"\n[7/7] Plotting confusion matrix...")
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=validation_classes,
        title="Validation Confusion Matrix",
    )

    print("\n" + "=" * 80)
    print("✓ All visualizations complete")
    print("=" * 80 + "\n")
