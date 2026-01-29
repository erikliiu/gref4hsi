"""
Test script to find optimal SVM hyperparameters for 057 and 028.

This script tests an expanded C range to see if smaller C values improve performance.
Current best: C=0.1, gamma=0.1 (057) and C=10, gamma=0.1 (028)
"""

import numpy as np
import sys
from pathlib import Path

# Add mjosa_code root to path
mjosa_code_root = Path(__file__).parent.parent
sys.path.insert(0, str(mjosa_code_root))

from utils.uhi import georef
from utils.common import config
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import f1_score, accuracy_score
import time

print("=" * 80)
print("🔬 HYPERPARAMETER SEARCH TEST")
print("=" * 80)

# Expanded parameter grid with smaller C values
param_grid_expanded = {
    "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],  # Extended lower range
    "gamma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, "scale", "auto"],  # Same as before
}

param_grid_original = {
    "C": [0.1, 1, 10, 100, 1000],  # Original range
    "gamma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, "scale", "auto"],
}

print("\n📋 Parameter Grids:")
print(f"\n1. ORIGINAL grid:")
print(f"   C: {param_grid_original['C']}")
print(f"   gamma: {param_grid_original['gamma']}")

print(f"\n2. EXPANDED grid (with smaller C values):")
print(f"   C: {param_grid_expanded['C']}")
print(f"   gamma: {param_grid_expanded['gamma']}")


def test_transect_057():
    """Test hyperparameter search on transect 057."""
    print("\n" + "=" * 80)
    print("🧪 TEST 1: TRANSECT 057")
    print("=" * 80)

    # Load 057 data
    print("\n📂 Loading transect 057...")
    transect = georef.load_transect(config.TRANSECT_057_OUTPUT)
    cube = transect.select_files(["rad_uhi_20241029_115057_5"])

    # Apply preprocessing (same as notebook 6)
    print("🔧 Applying preprocessing...")
    cube.apply_illumination_correction_v2()
    cube.apply_spectral_smoothing(method="gaussian", gaussian_sigma=5.0)
    cube.apply_wavelength_filter(wavelength_range=(490, 700))

    # Import ROIs
    roi_path = Path(__file__).parent / "rois" / "057_5_combined.json"
    cube.import_rois(str(roi_path))

    training_rois = ["training_dark", "training_sediment", "training_bombs"]

    # Run grid search with EXPANDED parameter grid
    print("\n🔍 Running grid search with EXPANDED parameter grid...")
    print(
        f"   (This will take a while - testing {len(param_grid_expanded['C']) * len(param_grid_expanded['gamma'])} combinations)"
    )

    start_time = time.time()

    cv_results = cube.train_svm_with_cv(
        training_rois=training_rois,
        segment_start=config.UHI_TRACK_RANGE[0],
        segment_end=config.UHI_TRACK_RANGE[1] - 1,
        cv_folds=5,
        use_corrected=True,
        svm_kernel="rbf",
        optimize_params=True,
        add_brightness_feature=False,
        quiet=True,  # Suppress verbose output
        closing_radius=3,
    )

    elapsed_time = time.time() - start_time

    print(f"\n✅ Grid search complete! ({elapsed_time:.1f} seconds)")

    print(f"\n📊 RESULTS (057 - EXPANDED GRID):")
    print(f"   Best C:       {cv_results['best_params']['C']}")
    print(f"   Best gamma:   {cv_results['best_params']['gamma']}")
    print(
        f"   CV Accuracy:  {cv_results['cv_mean_metrics']['accuracy_mean']:.3f} ± {cv_results['cv_mean_metrics']['accuracy_std']:.3f}"
    )
    print(
        f"   CV F1 Score:  {cv_results['cv_mean_metrics']['f1_mean']:.3f} ± {cv_results['cv_mean_metrics']['f1_std']:.3f}"
    )

    # Compare with original (C=0.1, gamma=0.1)
    print(f"\n🔍 COMPARISON WITH ORIGINAL:")
    print(f"   Original best: C=0.1, gamma=0.1")
    print(
        f"   New best:      C={cv_results['best_params']['C']}, gamma={cv_results['best_params']['gamma']}"
    )

    if cv_results["best_params"]["C"] < 0.1:
        improvement = (
            cv_results["cv_mean_metrics"]["f1_mean"] - 0.593
        )  # Original F1 from notebook 6
        print(f"   ✅ SMALLER C selected! F1 improvement: {improvement:+.3f}")
    elif cv_results["best_params"]["C"] == 0.1:
        print(f"   ⚠️  C=0.1 still optimal (at boundary)")
    else:
        print(f"   📈 Larger C selected: C={cv_results['best_params']['C']}")

    return cv_results


def test_transect_028():
    """Test hyperparameter search on transect 028."""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: TRANSECT 028")
    print("=" * 80)
    print("⚠️  NOTE: 028 CV failed due to rust having only 1 spatial group")
    print("   This test will show if hyperparameters matter despite failed CV")

    # Load 028 data
    print("\n📂 Loading transect 028...")
    transect = georef.load_transect(config.TRANSECT_028_OUTPUT)
    cube = transect.select_files(
        [
            "rad_uhi_20241029_125028_1",
            "rad_uhi_20241029_125028_2",
            "rad_uhi_20241029_125028_3",
            "rad_uhi_20241029_125028_4",
            "rad_uhi_20241029_125028_5",
        ]
    )

    # Apply preprocessing (same as notebook 9)
    print("🔧 Applying preprocessing...")
    cube.apply_illumination_correction_v2()
    cube.apply_spectral_smoothing(method="gaussian", gaussian_sigma=5.0)
    cube.apply_wavelength_filter(wavelength_range=(490, 700))

    # Import ROIs
    roi_path = Path(__file__).parent / "rois" / "028_new.json"
    cube.import_rois(str(roi_path))

    # Map ROIs to classes (same as notebook 9)
    roi_mapping = {
        "sediment": "sediment",
        "rust": "rust",
        "1_dark": "dark_bomb",
        "2_dark": "dark_bomb",
        "3_dark": "dark_bomb",
        "dark_pits": "dark_pit",
        "dark_pits_multiple": "dark_pit",
        "1_halo": "halo",
        "2_halo": "halo",
        "3_halo": "halo",
    }

    training_rois_multiclass = [
        "sediment",
        "rust",
        "1_dark",
        "2_dark",
        "3_dark",
        "dark_pits",
        "dark_pits_multiple",
        "1_halo",
        "2_halo",
        "3_halo",
    ]

    # Create mapped ROI collection
    mapped_training_rois = {}
    for roi_name in training_rois_multiclass:
        if roi_name in cube.roi_collection:
            mapped_name = roi_mapping.get(roi_name, roi_name)
            if mapped_name not in mapped_training_rois:
                mapped_training_rois[mapped_name] = []
            mapped_training_rois[mapped_name].extend(cube.roi_collection[roi_name])

    # Remove duplicates
    for class_name in mapped_training_rois:
        mapped_training_rois[class_name] = list(set(mapped_training_rois[class_name]))

    # Replace ROI collection
    cube._original_roi_collection = cube.roi_collection.copy()
    cube.roi_collection = mapped_training_rois

    # Run grid search with EXPANDED parameter grid
    print("\n🔍 Running grid search with EXPANDED parameter grid...")
    print(f"   (CV will likely fail again due to rust, but grid search will still run)")

    start_time = time.time()

    cv_results = cube.train_svm_with_cv(
        training_rois=list(mapped_training_rois.keys()),
        segment_start=None,  # Use all data
        segment_end=None,
        wavelength_range=None,
        cv_folds=5,
        use_corrected=True,
        svm_kernel="rbf",
        optimize_params=True,
        add_brightness_feature=False,
        use_intensity_only=False,
        quiet=True,  # Suppress verbose output
        closing_radius=50,
    )

    elapsed_time = time.time() - start_time

    # Restore original ROI collection
    cube.roi_collection = cube._original_roi_collection
    del cube._original_roi_collection

    print(f"\n✅ Grid search complete! ({elapsed_time:.1f} seconds)")

    print(f"\n📊 RESULTS (028 - EXPANDED GRID):")
    print(f"   Best C:       {cv_results['best_params']['C']}")
    print(f"   Best gamma:   {cv_results['best_params']['gamma']}")

    # CV metrics might be nan
    if not np.isnan(cv_results["cv_mean_metrics"]["accuracy_mean"]):
        print(
            f"   CV Accuracy:  {cv_results['cv_mean_metrics']['accuracy_mean']:.3f} ± {cv_results['cv_mean_metrics']['accuracy_std']:.3f}"
        )
        print(
            f"   CV F1 Score:  {cv_results['cv_mean_metrics']['f1_mean']:.3f} ± {cv_results['cv_mean_metrics']['f1_std']:.3f}"
        )
    else:
        print(f"   CV Accuracy:  nan (CV failed - expected)")
        print(f"   CV F1 Score:  nan (CV failed - expected)")

    # Compare with original (C=10, gamma=0.1)
    print(f"\n🔍 COMPARISON WITH ORIGINAL:")
    print(f"   Original best: C=10, gamma=0.1")
    print(
        f"   New best:      C={cv_results['best_params']['C']}, gamma={cv_results['best_params']['gamma']}"
    )

    if cv_results["best_params"]["C"] != 10:
        print(f"   📈 Different C selected!")
    else:
        print(f"   ⚠️  C=10 still selected")

    return cv_results


if __name__ == "__main__":
    print("\n🚀 Starting hyperparameter search tests...")
    print("   This will test both transects with expanded C range")
    print("   Expected run time: ~5-10 minutes")

    # Test 057
    try:
        results_057 = test_transect_057()
    except Exception as e:
        print(f"\n❌ ERROR testing 057: {e}")
        import traceback

        traceback.print_exc()
        results_057 = None

    # Test 028
    try:
        results_028 = test_transect_028()
    except Exception as e:
        print(f"\n❌ ERROR testing 028: {e}")
        import traceback

        traceback.print_exc()
        results_028 = None

    # Final summary
    print("\n" + "=" * 80)
    print("📝 FINAL SUMMARY")
    print("=" * 80)

    if results_057:
        print(f"\n057 RESULTS:")
        print(f"   Best C: {results_057['best_params']['C']}")
        print(f"   Best gamma: {results_057['best_params']['gamma']}")
        print(f"   F1 Score: {results_057['cv_mean_metrics']['f1_mean']:.3f}")

        if results_057["best_params"]["C"] < 0.1:
            print(
                f"   ✅ RECOMMENDATION: Update parameter grid to include smaller C values!"
            )
            print(f"      New C range: [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]")
        else:
            print(f"   ⚠️  C=0.1 or larger still optimal")

    if results_028:
        print(f"\n028 RESULTS:")
        print(f"   Best C: {results_028['best_params']['C']}")
        print(f"   Best gamma: {results_028['best_params']['gamma']}")
        if not np.isnan(results_028["cv_mean_metrics"]["f1_mean"]):
            print(f"   F1 Score: {results_028['cv_mean_metrics']['f1_mean']:.3f}")
        else:
            print(f"   F1 Score: nan (CV failed)")

    print("\n" + "=" * 80)
    print("✅ Hyperparameter search test complete!")
    print("=" * 80)
