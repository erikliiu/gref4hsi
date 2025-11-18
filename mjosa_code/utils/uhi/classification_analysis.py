"""
Comprehensive classification analysis utilities for hyperspectral SVM results.

This module provides tools to analyze classification results including:
- Per-class pixel counts and percentages
- Before/after filtering comparisons
- Connected component cluster statistics (requires scipy)
- Visualization: bar charts, pie charts, histograms
- CSV export of summary statistics

Author: Auto-generated for gref4hsi project
Date: 2025-11-15
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict


def analyze_classification_results(
    classification_map,
    filtered_map=None,
    class_names: Optional[List[str]] = None,
    track_start: int = 0,
    figsize: Tuple[int, int] = (12, 6),
    show_plots: bool = True,
    save_csv: Optional[str] = None,
) -> Dict:
    """
    Display a comprehensive set of statistics and plots for a classification.

    This function analyzes classification results from SVM models, providing:
    1. Per-class pixel counts and percentages (before/after filtering)
    2. Unknown/unclassified pixel statistics
    3. Filtering impact analysis (pixels removed per class)
    4. Connected component cluster statistics (if scipy available)
    5. Visualizations: bar charts, pie charts, cluster size histograms
    6. Optional CSV export of summary statistics

    Parameters
    ----------
    classification_map : ndarray
        2D array (tracks × slits) of class labels (strings) from classify_segment.
    filtered_map : ndarray, optional
        2D array (tracks × slits) after filtering. If provided, before/after
        comparisons will be computed.
    class_names : list[str], optional
        Ordered list of class names to report. If None, auto-discovered from maps.
    track_start : int, default=0
        Track offset for coordinate reporting (informational only).
    figsize : tuple, default=(12, 6)
        Figure size for matplotlib plots.
    show_plots : bool, default=True
        If True, generate and display matplotlib plots. If False, only compute
        metrics and return dict.
    save_csv : str, optional
        If provided, save a CSV summary to this path.

    Returns
    -------
    dict
        Dictionary containing:
        - 'total_pixels': int, total number of pixels in map
        - 'counts_before': dict, {class_name: pixel_count} before filtering
        - 'counts_after': dict or None, {class_name: pixel_count} after filtering
        - 'filtering_impact': dict or None, per-class removal statistics
        - 'cluster_stats': dict, connected component statistics (if scipy available)

    Examples
    --------
    >>> # After running classification in a notebook:
    >>> from utils.uhi.classification_analysis import analyze_classification_results
    >>> metrics = analyze_classification_results(
    ...     classification_map=classification_results['classification_map'],
    ...     filtered_map=filtering_results['filtered_map'],
    ...     class_names=['sediment', 'rust', 'dark_bomb'],
    ...     show_plots=True,
    ...     save_csv='./saved_data/summary.csv'
    ... )
    >>> print(f"Total pixels: {metrics['total_pixels']:,}")

    Notes
    -----
    - Requires numpy and matplotlib (both standard in gref4hsi environment)
    - Cluster statistics require scipy.ndimage.label (optional dependency)
    - If scipy not available, cluster stats are skipped with a warning message
    - All visualizations use seaborn-style color palettes when available
    """

    import csv

    try:
        from scipy.ndimage import label

        _has_label = True
    except Exception:
        _has_label = False

    cls_map = np.asarray(classification_map)
    filtered = np.asarray(filtered_map) if filtered_map is not None else None

    # Determine class list (auto-discover if not provided)
    if class_names is None:
        names = np.unique(cls_map)
        if filtered is not None:
            names = np.unique(np.concatenate([names, np.unique(filtered)]))
        class_names = [str(n) for n in names]

    total_pixels = cls_map.size

    # ==================================================================
    # 1. BASIC COUNTS
    # ==================================================================
    counts_before = {cn: int(np.sum(cls_map == cn)) for cn in class_names}
    counts_after = (
        {cn: int(np.sum(filtered == cn)) for cn in class_names}
        if filtered is not None
        else None
    )

    # Identify unknown/unclassified pixels (any class with "unknown" in name)
    unknown_labels = [n for n in class_names if "unknown" in str(n).lower()]
    unknown_before = sum(counts_before.get(u, 0) for u in unknown_labels)
    unknown_after = (
        sum(counts_after.get(u, 0) for u in unknown_labels) if counts_after else 0
    )

    # ==================================================================
    # 2. FILTERING IMPACT
    # ==================================================================
    filtering_impact = None
    if counts_after is not None:
        filtering_impact = {}
        for cn in class_names:
            b = counts_before.get(cn, 0)
            a = counts_after.get(cn, 0)
            filtering_impact[cn] = {
                "before": b,
                "after": a,
                "removed": b - a,
                "removed_pct": 100.0 * (b - a) / b if b > 0 else 0.0,
            }

    # ==================================================================
    # 3. CLUSTER STATISTICS (connected components)
    # ==================================================================
    cluster_stats = {}
    if _has_label:
        for cn in class_names:
            mask = cls_map == cn
            if mask.sum() == 0:
                cluster_stats[cn] = {
                    "n_clusters": 0,
                    "largest": 0,
                    "smallest": 0,
                    "mean_size": 0.0,
                }
                continue

            labeled, ncomp = label(mask)
            if ncomp == 0:
                cluster_stats[cn] = {
                    "n_clusters": 0,
                    "largest": 0,
                    "smallest": 0,
                    "mean_size": 0.0,
                }
                continue

            sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
            cluster_stats[cn] = {
                "n_clusters": int(ncomp),
                "largest": int(sizes.max()),
                "smallest": int(sizes.min()),
                "mean_size": float(sizes.mean()),
                "sizes": sizes,  # raw cluster sizes (numpy array)
            }
    else:
        cluster_stats["__warning"] = (
            "scipy.ndimage.label not available; "
            "install scipy to enable cluster statistics"
        )

    # ==================================================================
    # 4. PRINT SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("CLASSIFICATION ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total pixels in map: {total_pixels:,}")
    print(
        f"Unknown / unclassified (before): {unknown_before:,} pixels "
        f"({100.0 * unknown_before / total_pixels:.2f}%)"
    )
    if filtered is not None:
        print(
            f"Unknown / unclassified (after): {unknown_after:,} pixels "
            f"({100.0 * unknown_after / total_pixels:.2f}%)"
        )

    print("\nPer-class counts (before filtering):")
    for cn in class_names:
        b = counts_before.get(cn, 0)
        print(f"  {cn}: {b:,} pixels ({100.0 * b / total_pixels:.2f}%)")

    if counts_after is not None:
        print("\nPer-class counts (after filtering):")
        for cn in class_names:
            a = counts_after.get(cn, 0)
            print(f"  {cn}: {a:,} pixels ({100.0 * a / total_pixels:.2f}%)")

    # Print filtering impact summary
    if filtering_impact is not None:
        print("\nFiltering impact (removed pixels):")
        for cn, v in filtering_impact.items():
            if v["removed"] > 0:
                print(f"  {cn}: removed {v['removed']:,} px ({v['removed_pct']:.2f}%)")

    # Print cluster statistics summary
    if _has_label:
        print("\nCluster statistics (connected components):")
        for cn, v in cluster_stats.items():
            if cn == "__warning":
                continue
            print(
                f"  {cn}: {v['n_clusters']} clusters, "
                f"largest={v['largest']} px, smallest={v['smallest']} px, "
                f"mean={v['mean_size']:.1f} px"
            )
    else:
        print("\n⚠️  Cluster statistics unavailable (scipy.ndimage not installed)")

    # ==================================================================
    # 5. PREPARE RETURN DICTIONARY
    # ==================================================================
    results = {
        "total_pixels": int(total_pixels),
        "counts_before": counts_before,
        "counts_after": counts_after,
        "filtering_impact": filtering_impact,
        "cluster_stats": cluster_stats,
    }

    # ==================================================================
    # 6. SAVE CSV SUMMARY
    # ==================================================================
    if save_csv is not None:
        csv_rows = []
        for cn in class_names:
            row = {
                "class": cn,
                "before": counts_before.get(cn, 0),
                "after": (counts_after.get(cn, 0) if counts_after is not None else ""),
            }
            csv_rows.append(row)

        with open(save_csv, "w", newline="") as csvfile:
            fieldnames = ["class", "before", "after"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print(f"\n💾 Saved CSV summary to: {save_csv}")

    # ==================================================================
    # 7. GENERATE PLOTS
    # ==================================================================
    if show_plots:
        labels = class_names
        before_vals = [counts_before.get(cn, 0) for cn in labels]
        after_vals = (
            [counts_after.get(cn, 0) for cn in labels]
            if counts_after is not None
            else None
        )

        x = np.arange(len(labels))
        width = 0.35

        # --- Plot 1: Bar chart (before vs after) ---
        plt.figure(figsize=figsize)
        if after_vals is not None:
            plt.bar(
                x - width / 2, before_vals, width, label="Before filtering", alpha=0.8
            )
            plt.bar(
                x + width / 2, after_vals, width, label="After filtering", alpha=0.8
            )
        else:
            plt.bar(x, before_vals, width, label="Pixel counts", alpha=0.8)

        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Pixel count")
        plt.title("Class Distribution: Before vs After Filtering")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Pie chart (final distribution) ---
        final = after_vals if after_vals is not None else before_vals
        # Filter out zero counts for cleaner pie chart
        nonzero_labels = [labels[i] for i, v in enumerate(final) if v > 0]
        nonzero_vals = [v for v in final if v > 0]

        if nonzero_vals:
            plt.figure(figsize=(8, 8))
            plt.pie(
                nonzero_vals,
                labels=nonzero_labels,
                autopct="%.1f%%",
                startangle=90,
                textprops={"fontsize": 10},
            )
            plt.title(
                "Final Class Distribution"
                + (" (After Filtering)" if after_vals else "")
            )
            plt.tight_layout()
            plt.show()

        # --- Plot 3: Cluster size histogram (if available) ---
        if _has_label:
            all_sizes = []
            for k, v in cluster_stats.items():
                if k != "__warning" and v.get("sizes") is not None:
                    all_sizes.extend(v["sizes"])

            if all_sizes:
                plt.figure(figsize=(10, 5))
                plt.hist(
                    all_sizes,
                    bins=min(50, len(all_sizes)),
                    edgecolor="black",
                    alpha=0.7,
                )
                plt.xlabel("Cluster size (pixels)")
                plt.ylabel("Frequency")
                plt.title("Distribution of Cluster Sizes (All Classes)")
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.show()

    print("\n" + "=" * 80)
    print("✅ Analysis complete. Returned metrics dictionary.")
    print("=" * 80)

    return results
