"""
Script to compare TIF files and check for differences.

Compares:
1. Original TIF from config (main MBES)
2. Multiple TIFs from the additional folder

Checks:
- CRS (coordinate reference system)
- Bounds (spatial extent)
- Resolution
- Data range (min/max values)
- Array shape
- Data statistics
- Overlap analysis

Usage:
    python check_tif_differences.py
"""

import os
import sys
import glob
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gref_pipeline import config


def analyze_tif(tif_path):
    """
    Extract comprehensive information about a TIF file.

    Returns dict with:
    - filename, path
    - CRS, bounds (in native and WGS84)
    - shape, resolution
    - data statistics (min, max, mean, std, finite count)
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)

        # Basic metadata
        info = {
            "filename": os.path.basename(tif_path),
            "path": tif_path,
            "crs": str(src.crs),
            "bounds_native": src.bounds,
            "shape": data.shape,
            "resolution": (src.res[0], src.res[1]),
            "nodata": src.nodata,
        }

        # Transform bounds to WGS84 for comparison
        if src.crs is not None:
            bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            info["bounds_wgs84"] = bounds_wgs84
        else:
            info["bounds_wgs84"] = None

        # Data statistics
        finite_mask = np.isfinite(data)
        if src.nodata is not None:
            finite_mask &= data != src.nodata

        if finite_mask.any():
            finite_data = data[finite_mask]
            info["stats"] = {
                "min": float(np.min(finite_data)),
                "max": float(np.max(finite_data)),
                "mean": float(np.mean(finite_data)),
                "std": float(np.std(finite_data)),
                "finite_count": int(np.sum(finite_mask)),
                "total_pixels": int(data.size),
                "finite_percent": float(100 * np.sum(finite_mask) / data.size),
            }
        else:
            info["stats"] = {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "finite_count": 0,
                "total_pixels": int(data.size),
                "finite_percent": 0.0,
            }

        return info


def check_overlap(bounds1, bounds2):
    """
    Check if two bounding boxes overlap.
    bounds format: (west, south, east, north)

    Returns:
    - overlap_exists: bool
    - overlap_bounds: tuple or None
    - overlap_percent_1: percent of bounds1 that overlaps
    - overlap_percent_2: percent of bounds2 that overlaps
    """
    west1, south1, east1, north1 = bounds1
    west2, south2, east2, north2 = bounds2

    # Check for overlap
    if west1 > east2 or west2 > east1 or south1 > north2 or south2 > north1:
        return False, None, 0.0, 0.0

    # Calculate overlap bounds
    overlap_west = max(west1, west2)
    overlap_south = max(south1, south2)
    overlap_east = min(east1, east2)
    overlap_north = min(north1, north2)

    overlap_bounds = (overlap_west, overlap_south, overlap_east, overlap_north)

    # Calculate areas
    area1 = (east1 - west1) * (north1 - south1)
    area2 = (east2 - west2) * (north2 - south2)
    overlap_area = (overlap_east - overlap_west) * (overlap_north - overlap_south)

    overlap_pct1 = 100 * overlap_area / area1 if area1 > 0 else 0
    overlap_pct2 = 100 * overlap_area / area2 if area2 > 0 else 0

    return True, overlap_bounds, overlap_pct1, overlap_pct2


def print_tif_info(info, prefix=""):
    """Pretty print TIF information."""
    print(f"{prefix}📄 {info['filename']}")
    print(f"{prefix}   Path: {info['path']}")
    print(f"{prefix}   CRS: {info['crs']}")
    print(f"{prefix}   Shape: {info['shape'][0]} x {info['shape'][1]} pixels")
    print(
        f"{prefix}   Resolution: {info['resolution'][0]:.6f} x {info['resolution'][1]:.6f}"
    )

    if info["bounds_wgs84"]:
        w, s, e, n = info["bounds_wgs84"]
        print(f"{prefix}   Bounds (WGS84):")
        print(f"{prefix}      West:  {w:.6f}°")
        print(f"{prefix}      South: {s:.6f}°")
        print(f"{prefix}      East:  {e:.6f}°")
        print(f"{prefix}      North: {n:.6f}°")

    stats = info["stats"]
    print(f"{prefix}   Data Statistics:")
    if stats["min"] is not None:
        print(f"{prefix}      Min:   {stats['min']:.3f}")
        print(f"{prefix}      Max:   {stats['max']:.3f}")
        print(f"{prefix}      Mean:  {stats['mean']:.3f}")
        print(f"{prefix}      Std:   {stats['std']:.3f}")
    print(
        f"{prefix}      Valid: {stats['finite_count']:,} / {stats['total_pixels']:,} ({stats['finite_percent']:.1f}%)"
    )
    print()


def compare_tifs(original_tif, new_tif_folder):
    """
    Compare original TIF with all TIFs in the new folder.
    """
    print("=" * 80)
    print("TIF FILE COMPARISON ANALYSIS")
    print("=" * 80)
    print()

    # Analyze original TIF
    print("🔍 ORIGINAL TIF (from config):")
    print("-" * 80)
    if not os.path.exists(original_tif):
        print(f"❌ ERROR: Original TIF not found: {original_tif}")
        return

    orig_info = analyze_tif(original_tif)
    print_tif_info(orig_info, prefix="   ")

    # Find new TIFs
    print("\n🔍 NEW TIF FILES (from folder):")
    print("-" * 80)
    if not os.path.isdir(new_tif_folder):
        print(f"❌ ERROR: Folder not found: {new_tif_folder}")
        return

    tif_files = sorted(glob.glob(os.path.join(new_tif_folder, "*.tif")))
    tif_files.extend(sorted(glob.glob(os.path.join(new_tif_folder, "*.tiff"))))
    tif_files = sorted(set(tif_files))

    if len(tif_files) == 0:
        print(f"❌ No TIF files found in {new_tif_folder}")
        return

    print(f"   Found {len(tif_files)} TIF file(s)\n")

    new_infos = []
    for tif_path in tif_files:
        try:
            info = analyze_tif(tif_path)
            new_infos.append(info)
            print_tif_info(info, prefix="   ")
        except Exception as e:
            print(f"   ❌ Error analyzing {os.path.basename(tif_path)}: {e}\n")

    # Comparison analysis
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)

    # 1. CRS comparison
    print("\n1️⃣  CRS (Coordinate Reference System):")
    print("-" * 80)
    print(f"   Original: {orig_info['crs']}")
    for info in new_infos:
        match = "✅ SAME" if info["crs"] == orig_info["crs"] else "⚠️  DIFFERENT"
        print(f"   {info['filename']}: {info['crs']} {match}")

    # 2. Spatial extent comparison
    print("\n2️⃣  Spatial Extent (Bounds in WGS84):")
    print("-" * 80)
    if orig_info["bounds_wgs84"]:
        w, s, e, n = orig_info["bounds_wgs84"]
        print(f"   Original: W={w:.6f}°, S={s:.6f}°, E={e:.6f}°, N={n:.6f}°")

        for info in new_infos:
            if info["bounds_wgs84"]:
                w2, s2, e2, n2 = info["bounds_wgs84"]
                print(f"   {info['filename']}:")
                print(f"      W={w2:.6f}°, S={s2:.6f}°, E={e2:.6f}°, N={n2:.6f}°")

                # Check overlap
                overlap, overlap_bounds, pct1, pct2 = check_overlap(
                    orig_info["bounds_wgs84"], info["bounds_wgs84"]
                )
                if overlap:
                    print(
                        f"      ✅ OVERLAPS with original ({pct2:.1f}% of this TIF, {pct1:.1f}% of original)"
                    )
                else:
                    print(
                        f"      ⚠️  NO OVERLAP with original - completely different area!"
                    )

    # 3. Resolution comparison
    print("\n3️⃣  Resolution:")
    print("-" * 80)
    orig_res = orig_info["resolution"]
    print(f"   Original: {orig_res[0]:.6f} x {orig_res[1]:.6f}")
    for info in new_infos:
        res = info["resolution"]
        match = (
            "✅ SAME"
            if abs(res[0] - orig_res[0]) < 1e-6 and abs(res[1] - orig_res[1]) < 1e-6
            else "⚠️  DIFFERENT"
        )
        print(f"   {info['filename']}: {res[0]:.6f} x {res[1]:.6f} {match}")

    # 4. Data range comparison
    print("\n4️⃣  Data Range (Min/Max values):")
    print("-" * 80)
    if orig_info["stats"]["min"] is not None:
        print(
            f"   Original: {orig_info['stats']['min']:.3f} to {orig_info['stats']['max']:.3f}"
        )
        for info in new_infos:
            if info["stats"]["min"] is not None:
                similar = (
                    abs(info["stats"]["min"] - orig_info["stats"]["min"]) < 1.0
                    and abs(info["stats"]["max"] - orig_info["stats"]["max"]) < 1.0
                )
                match = "✅ SIMILAR" if similar else "⚠️  DIFFERENT"
                print(
                    f"   {info['filename']}: {info['stats']['min']:.3f} to {info['stats']['max']:.3f} {match}"
                )

    # 5. Summary verdict
    print("\n" + "=" * 80)
    print("📊 SUMMARY VERDICT")
    print("=" * 80)

    all_same_crs = all(info["crs"] == orig_info["crs"] for info in new_infos)
    any_overlap = any(
        check_overlap(orig_info["bounds_wgs84"], info["bounds_wgs84"])[0]
        for info in new_infos
        if info["bounds_wgs84"]
    )

    if all_same_crs and any_overlap:
        print("✅ TIFs appear to be from SAME/SIMILAR dataset:")
        print("   - Same coordinate system")
        print("   - Overlapping spatial extent")
        print("   ➡️  Likely different time periods or processing versions")
    else:
        print("⚠️  TIFs appear to be DIFFERENT datasets:")
        if not all_same_crs:
            print("   - Different coordinate systems detected")
        if not any_overlap:
            print("   - No spatial overlap detected")
        print("   ➡️  These may be from completely different surveys/areas")

    print("=" * 80)


if __name__ == "__main__":
    print("\n")

    # Paths
    ORIGINAL_TIF = config.MBES_GEOTIFF
    NEW_TIF_FOLDER = r"E:\mjosa_new_oct_2025\all_tifs_from_eiva\relevant_tifs_only"

    # Run comparison
    compare_tifs(ORIGINAL_TIF, NEW_TIF_FOLDER)

    print("\n✅ Analysis complete!")
    print()
