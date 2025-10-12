"""
UHI Data Georeferencing and Mask Visualization

This script:
1. Loads UHI georeferenced data from H5 files
2. Extracts and visualizes the valid/invalid data mask
3. Shows spatial distribution of valid georeferenced points
"""

import importlib
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import rasterio
from scipy.ndimage import binary_dilation

# Add parent directory to path
sys.path.append(os.path.abspath("../"))

# Import and reload modules
from utils import georef
import config

# Reload to get latest changes
importlib.reload(georef)
from utils.georef import *


def plot_georef_mask(cube, track_start=None, track_end=None, coordinate_system="NED"):
    """
    Plot the valid/invalid mask of georeferenced UHI data.

    Parameters:
    -----------
    cube : CombinedTransectCube
        The georeferenced data cube
    track_start : int, optional
        Starting track index (inclusive)
    track_end : int, optional
        Ending track index (exclusive)
    coordinate_system : str
        "NED", "LATLON", or "ECEF"
    """
    # Extract coordinates
    X_ecef = cube.X_ecef
    Y_ecef = cube.Y_ecef
    Z_ecef = cube.Z_ecef
    R = cube.R
    G = cube.G
    B = cube.B

    T, S = X_ecef.shape
    print(f"📊 Original data shape: {T} tracks × {S} slits = {T*S} total points")

    # Apply track slicing if specified
    if track_start is not None or track_end is not None:
        start_idx = track_start if track_start is not None else 0
        end_idx = track_end if track_end is not None else T

        X_ecef = X_ecef[start_idx:end_idx, :]
        Y_ecef = Y_ecef[start_idx:end_idx, :]
        Z_ecef = Z_ecef[start_idx:end_idx, :]
        R = R[start_idx:end_idx, :]
        G = G[start_idx:end_idx, :]
        B = B[start_idx:end_idx, :]
        T, S = X_ecef.shape

        print(f"📊 Sliced to tracks {start_idx}–{end_idx-1}: {T} tracks × {S} slits")

    # Create validity masks
    coords_valid = np.isfinite(X_ecef) & np.isfinite(Y_ecef) & np.isfinite(Z_ecef)
    rgb_valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
    full_valid = coords_valid & rgb_valid

    # Statistics
    n_total = T * S
    n_coords_valid = np.sum(coords_valid)
    n_rgb_valid = np.sum(rgb_valid)
    n_full_valid = np.sum(full_valid)

    print(f"\n📈 Validity Statistics:")
    print(f"   Total points:          {n_total:,}")
    print(
        f"   Valid coordinates:     {n_coords_valid:,} ({100*n_coords_valid/n_total:.1f}%)"
    )
    print(f"   Valid RGB:             {n_rgb_valid:,} ({100*n_rgb_valid/n_total:.1f}%)")
    print(
        f"   Fully valid (both):    {n_full_valid:,} ({100*n_full_valid/n_total:.1f}%)"
    )
    print(
        f"   Invalid:               {n_total - n_full_valid:,} ({100*(n_total-n_full_valid)/n_total:.1f}%)"
    )

    # Convert coordinates for plotting
    if coordinate_system.upper() == "LATLON":
        # Convert ECEF to Lat/Lon
        tf_ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
        lon, lat, height = tf_ecef_to_geo.transform(X_ecef, Y_ecef, Z_ecef)
        Xp, Yp = lon, lat
        xlabel, ylabel = "Longitude (°)", "Latitude (°)"

    elif coordinate_system.upper() == "NED":
        # Use origin from config
        lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
        from utils.georef import _ecef_to_ned_arrays

        N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)
        Xp, Yp = E, N
        xlabel, ylabel = f"East (m) from {lat0:.4f}°, {lon0:.4f}°", "North (m)"

    elif coordinate_system.upper() == "ECEF":
        Xp, Yp = X_ecef, Y_ecef
        xlabel, ylabel = "ECEF X (m)", "ECEF Y (m)"
    else:
        raise ValueError("coordinate_system must be 'LATLON', 'NED', or 'ECEF'")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Prepare coordinates for pcolormesh (add padding for cell corners)
    Xc = np.pad(Xp, ((0, 1), (0, 1)), mode="edge")
    Yc = np.pad(Yp, ((0, 1), (0, 1)), mode="edge")

    # Fix non-finite coordinates
    mask_valid_coords = np.isfinite(Xc) & np.isfinite(Yc)
    if not mask_valid_coords.all():
        from scipy.ndimage import distance_transform_edt

        invalid_mask = ~mask_valid_coords
        if invalid_mask.any():
            indices = distance_transform_edt(
                invalid_mask, return_distances=False, return_indices=True
            )
            Xc[invalid_mask] = Xc[tuple(indices[:, invalid_mask])]
            Yc[invalid_mask] = Yc[tuple(indices[:, invalid_mask])]

    # 1. Coordinate validity mask
    ax = axes[0, 0]
    mask_display = coords_valid.astype(float)
    mask_display[~coords_valid] = np.nan
    im1 = ax.pcolormesh(
        Xc, Yc, mask_display, shading="flat", cmap="RdYlGn", vmin=0, vmax=1
    )
    ax.set_title(
        f"Coordinate Validity Mask\n{n_coords_valid:,}/{n_total:,} valid ({100*n_coords_valid/n_total:.1f}%)"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im1, ax=ax, label="Valid (1) / Invalid (NaN)")

    # 2. RGB validity mask
    ax = axes[0, 1]
    mask_display = rgb_valid.astype(float)
    mask_display[~rgb_valid] = np.nan
    im2 = ax.pcolormesh(
        Xc, Yc, mask_display, shading="flat", cmap="RdYlGn", vmin=0, vmax=1
    )
    ax.set_title(
        f"RGB Data Validity Mask\n{n_rgb_valid:,}/{n_total:,} valid ({100*n_rgb_valid/n_total:.1f}%)"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im2, ax=ax, label="Valid (1) / Invalid (NaN)")

    # 3. Full validity mask (both coords and RGB)
    ax = axes[1, 0]
    mask_display = full_valid.astype(float)
    mask_display[~full_valid] = np.nan
    im3 = ax.pcolormesh(
        Xc, Yc, mask_display, shading="flat", cmap="RdYlGn", vmin=0, vmax=1
    )
    ax.set_title(
        f"Full Validity Mask (Coords AND RGB)\n{n_full_valid:,}/{n_total:,} valid ({100*n_full_valid/n_total:.1f}%)"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im3, ax=ax, label="Valid (1) / Invalid (NaN)")

    # 4. Invalid points highlighted
    ax = axes[1, 1]
    # Create a mask where: 1=valid, 0=invalid, show both
    invalid_highlight = np.ones_like(full_valid, dtype=float)
    invalid_highlight[~full_valid] = 0
    im4 = ax.pcolormesh(
        Xc, Yc, invalid_highlight, shading="flat", cmap="bwr", vmin=0, vmax=1
    )
    ax.set_title(
        f"Invalid Points (Red)\n{n_total - n_full_valid:,} invalid points ({100*(n_total-n_full_valid)/n_total:.1f}%)"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im4, ax=ax, label="Valid (blue=1) / Invalid (red=0)")

    plt.tight_layout()
    plt.show()

    return full_valid


def load_mbes_and_crop_to_uhi(
    mbes_geotiff_path, uhi_x_ned, uhi_y_ned, validity_mask, buffer_m=5.0
):
    """
    Load MBES GeoTIFF and crop it to match the UHI data footprint.

    Parameters:
    -----------
    mbes_geotiff_path : str
        Path to MBES GeoTIFF file
    uhi_x_ned : np.ndarray
        UHI X coordinates in NED (East), shape (T, S)
    uhi_y_ned : np.ndarray
        UHI Y coordinates in NED (North), shape (T, S)
    validity_mask : np.ndarray
        Boolean mask of valid UHI points, shape (T, S)
    buffer_m : float
        Buffer distance in meters around UHI footprint

    Returns:
    --------
    dict with keys:
        - 'data': MBES elevation data (2D array)
        - 'mask': boolean mask of cropped region
        - 'x_coords': X coordinates of MBES pixels
        - 'y_coords': Y coordinates of MBES pixels
        - 'extent': (xmin, xmax, ymin, ymax) for plotting
        - 'transform': rasterio affine transform
        - 'crs': coordinate reference system
    """
    print(f"\n📂 Loading MBES GeoTIFF: {mbes_geotiff_path}")

    # Load MBES data
    with rasterio.open(mbes_geotiff_path) as src:
        mbes_data = src.read(1).astype(np.float64)
        mbes_transform = src.transform
        mbes_crs = src.crs
        mbes_bounds = src.bounds

        print(f"   MBES shape: {mbes_data.shape}")
        print(f"   MBES CRS: {mbes_crs}")
        print(f"   MBES bounds: {mbes_bounds}")

    # Replace nodata with NaN
    nodata_value = 9999.0
    mbes_data = np.where(mbes_data == nodata_value, np.nan, mbes_data)

    # Get MBES pixel coordinates
    H, W = mbes_data.shape
    a, b, c, d, e, f = (
        mbes_transform.a,
        mbes_transform.b,
        mbes_transform.c,
        mbes_transform.d,
        mbes_transform.e,
        mbes_transform.f,
    )

    # Pixel centers
    cols = np.arange(W)
    rows = np.arange(H)
    mbes_x_1d = c + a * (cols + 0.5)  # X coords for each column
    mbes_y_1d = f + e * (rows + 0.5)  # Y coords for each row

    px_x = abs(a)
    px_y = abs(e)

    print(f"   MBES pixel size: {px_x:.2f}m × {px_y:.2f}m")

    # Convert UHI coordinates to MBES pixel coordinates
    # First, convert UHI NED to ECEF, then to MBES CRS
    print(f"\n🔄 Converting UHI coordinates to MBES CRS...")

    # Get origin from config
    lat0, lon0, h0 = config.LAT0, config.LON0, config.H0

    # Extract valid UHI points only
    valid_x = uhi_x_ned[validity_mask]
    valid_y = uhi_y_ned[validity_mask]

    if len(valid_x) == 0:
        print("⚠️  No valid UHI points found!")
        return None

    print(f"   Valid UHI points: {len(valid_x):,}")

    # Convert NED to ECEF
    from utils.georef import _ecef_of_geodetic

    x0, y0, z0 = _ecef_of_geodetic(lat0, lon0, h0)

    # NED to ECEF transformation
    lat_rad = np.radians(lat0)
    lon_rad = np.radians(lon0)
    sL = np.sin(lat_rad)
    cL = np.cos(lat_rad)
    sO = np.sin(lon_rad)
    cO = np.cos(lon_rad)

    # NED components
    N = valid_y  # North
    E = valid_x  # East
    D = np.zeros_like(N)  # Down = 0 (assume surface level)

    # Inverse NED to ECEF transformation
    dx = -sL * cO * N - sO * E - cL * cO * D
    dy = -sL * sO * N + cO * E - cL * sO * D
    dz = cL * N - sL * D

    uhi_x_ecef = x0 + dx
    uhi_y_ecef = y0 + dy
    uhi_z_ecef = z0 + dz

    # Convert ECEF to lat/lon
    tf_ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    uhi_lon, uhi_lat, uhi_h = tf_ecef_to_geo.transform(
        uhi_x_ecef, uhi_y_ecef, uhi_z_ecef
    )

    # Convert lat/lon to MBES CRS
    if mbes_crs is not None:
        tf_geo_to_mbes = Transformer.from_crs("EPSG:4326", mbes_crs, always_xy=True)
        uhi_x_mbes, uhi_y_mbes = tf_geo_to_mbes.transform(uhi_lon, uhi_lat)
    else:
        # Assume MBES is in lat/lon
        uhi_x_mbes = uhi_lon
        uhi_y_mbes = uhi_lat

    # Find MBES pixels corresponding to UHI points
    col0_left = mbes_x_1d[0] - px_x * 0.5
    row0_top = mbes_y_1d[0] + px_y * 0.5

    uhi_cols = np.floor((uhi_x_mbes - col0_left) / px_x).astype(int)
    uhi_rows = np.floor((row0_top - uhi_y_mbes) / px_y).astype(int)

    # Filter to valid pixel indices
    valid_pixels = (uhi_rows >= 0) & (uhi_rows < H) & (uhi_cols >= 0) & (uhi_cols < W)
    uhi_rows_valid = uhi_rows[valid_pixels]
    uhi_cols_valid = uhi_cols[valid_pixels]

    print(f"   UHI points within MBES bounds: {valid_pixels.sum():,}")

    # Create footprint mask
    footprint_mask = np.zeros((H, W), dtype=bool)
    if len(uhi_rows_valid) > 0:
        footprint_mask[uhi_rows_valid, uhi_cols_valid] = True

    # Apply buffer dilation
    if buffer_m > 0:
        dx_dilate = int(np.ceil(buffer_m / px_x))
        dy_dilate = int(np.ceil(buffer_m / px_y))
        dilate_iters = max(dx_dilate, dy_dilate)
        print(f"   Applying buffer: {buffer_m}m ({dilate_iters} pixels)")
        footprint_mask = binary_dilation(footprint_mask, iterations=dilate_iters)

    # Statistics
    n_cropped = footprint_mask.sum()
    n_total = H * W
    print(f"\n📊 Cropped MBES region:")
    print(
        f"   Pixels selected: {n_cropped:,} / {n_total:,} ({100*n_cropped/n_total:.2f}%)"
    )

    # Find bounding box of cropped region
    rows_with_data = np.any(footprint_mask, axis=1)
    cols_with_data = np.any(footprint_mask, axis=0)

    if not rows_with_data.any() or not cols_with_data.any():
        print("⚠️  No MBES data in cropped region!")
        return None

    row_min = np.where(rows_with_data)[0].min()
    row_max = np.where(rows_with_data)[0].max()
    col_min = np.where(cols_with_data)[0].min()
    col_max = np.where(cols_with_data)[0].max()

    print(f"   Bounding box: rows [{row_min}, {row_max}], cols [{col_min}, {col_max}]")

    # Create coordinate grids for plotting
    mbes_x_2d, mbes_y_2d = np.meshgrid(mbes_x_1d, mbes_y_1d)

    extent = (
        mbes_x_1d[col_min] - px_x * 0.5,
        mbes_x_1d[col_max] + px_x * 0.5,
        mbes_y_1d[row_max] - px_y * 0.5,
        mbes_y_1d[row_min] + px_y * 0.5,
    )

    return {
        "data": mbes_data,
        "mask": footprint_mask,
        "x_coords": mbes_x_2d,
        "y_coords": mbes_y_2d,
        "extent": extent,
        "transform": mbes_transform,
        "crs": mbes_crs,
        "px_x": px_x,
        "px_y": px_y,
        "bbox_rows": (row_min, row_max),
        "bbox_cols": (col_min, col_max),
    }


def plot_mbes_cropped(mbes_result, coordinate_system="NED"):
    """
    Plot the cropped MBES data with mask overlay.

    Parameters:
    -----------
    mbes_result : dict
        Result from load_mbes_and_crop_to_uhi()
    coordinate_system : str
        Coordinate system for axis labels
    """
    if mbes_result is None:
        print("⚠️  No MBES data to plot!")
        return

    data = mbes_result["data"]
    mask = mbes_result["mask"]
    x_coords = mbes_result["x_coords"]
    y_coords = mbes_result["y_coords"]

    # Extract bounding box region
    row_min, row_max = mbes_result["bbox_rows"]
    col_min, col_max = mbes_result["bbox_cols"]

    data_crop = data[row_min : row_max + 1, col_min : col_max + 1]
    mask_crop = mask[row_min : row_max + 1, col_min : col_max + 1]
    x_crop = x_coords[row_min : row_max + 1, col_min : col_max + 1]
    y_crop = y_coords[row_min : row_max + 1, col_min : col_max + 1]

    # Mask out data outside footprint
    data_masked = np.where(mask_crop, data_crop, np.nan)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Full MBES data (cropped region)
    ax = axes[0]
    valid_data = data_crop[np.isfinite(data_crop)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [2, 98])
    else:
        vmin, vmax = 0, 1

    im1 = ax.pcolormesh(
        x_crop, y_crop, data_crop, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax
    )
    ax.set_title("MBES Data (Cropped Region)")
    ax.set_xlabel(f"East (m)" if coordinate_system == "NED" else "X")
    ax.set_ylabel(f"North (m)" if coordinate_system == "NED" else "Y")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im1, ax=ax, label="Elevation (m)")

    # 2. UHI Footprint Mask
    ax = axes[1]
    im2 = ax.pcolormesh(
        x_crop,
        y_crop,
        mask_crop.astype(float),
        shading="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"UHI Footprint Mask\n({mask_crop.sum():,} pixels)")
    ax.set_xlabel(f"East (m)" if coordinate_system == "NED" else "X")
    ax.set_ylabel(f"North (m)" if coordinate_system == "NED" else "Y")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im2, ax=ax, label="In Footprint")

    # 3. MBES Data (masked to UHI footprint)
    ax = axes[2]
    im3 = ax.pcolormesh(
        x_crop,
        y_crop,
        data_masked,
        shading="auto",
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        f"MBES Data (Masked to UHI Footprint)\n{np.isfinite(data_masked).sum():,} valid pixels"
    )
    ax.set_xlabel(f"East (m)" if coordinate_system == "NED" else "X")
    ax.set_ylabel(f"North (m)" if coordinate_system == "NED" else "Y")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im3, ax=ax, label="Elevation (m)")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main execution
# ============================================================

print("\n" + "=" * 70)
print("UHI GEOREFERENCING AND MASK VISUALIZATION")
print("=" * 70 + "\n")

# Load transect from the output folder specified in config
print(f"📁 Loading transect from: {config.OUTPUT_FOLDER}")
transect = load_transect(config.OUTPUT_FOLDER)
transect.list_files()

# Select specific files
print("\n🔍 Selecting files...")
cube = transect.select_files(
    [
        "rad_uhi_20241029_115057_4",
        "rad_uhi_20241029_115057_5",
    ]
)
cube.describe()

# Define track range
track_start = 3039
track_end = 4029

print(f"\n📊 Analyzing tracks {track_start}–{track_end-1}")

# Plot the mask visualization
print("\n🎨 Creating mask visualization...")
validity_mask = plot_georef_mask(
    cube, track_start=track_start, track_end=track_end, coordinate_system="NED"
)

# ============================================================
# MBES DATA CROPPING
# ============================================================

print("\n" + "=" * 70)
print("MBES DATA CROPPING TO UHI FOOTPRINT")
print("=" * 70)

# Get UHI coordinates in NED
print("\n🔄 Extracting UHI coordinates...")
X_ecef = cube.X_ecef
Y_ecef = cube.Y_ecef
Z_ecef = cube.Z_ecef

# Apply track slicing
if track_start is not None or track_end is not None:
    start_idx = track_start if track_start is not None else 0
    end_idx = track_end if track_end is not None else X_ecef.shape[0]
    X_ecef = X_ecef[start_idx:end_idx, :]
    Y_ecef = Y_ecef[start_idx:end_idx, :]
    Z_ecef = Z_ecef[start_idx:end_idx, :]

# Convert to NED
lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
from utils.georef import _ecef_to_ned_arrays

N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)

print(f"   UHI NED coordinates shape: {E.shape}")
print(f"   Valid coordinates: {np.isfinite(E).sum():,} / {E.size:,}")

# Load and crop MBES data
mbes_result = load_mbes_and_crop_to_uhi(
    mbes_geotiff_path=config.MBES_GEOTIFF,
    uhi_x_ned=E,
    uhi_y_ned=N,
    validity_mask=validity_mask,
    buffer_m=10.0,  # 10 meter buffer around UHI footprint
)

# Plot cropped MBES data
if mbes_result is not None:
    print("\n🎨 Creating MBES cropped visualization...")
    plot_mbes_cropped(mbes_result, coordinate_system="NED")
else:
    print("⚠️  Could not crop MBES data - no overlap with UHI data")

# ============================================================
# OPTIONAL: RGB georef plot for comparison
# ============================================================

print("\n🎨 Creating RGB georeferencing plot...")
cube.plot_georef(
    coordinate_system="NED",
    show_file_boundaries=True,
    interactive=True,
    track_start=track_start,
    track_end=track_end,
)

print("\n" + "=" * 70)
print("✅ PROCESSING COMPLETE!")
print("=" * 70)
