"""
dev2_html.py - Interactive map with MBES and HSI footprints

Creates an interactive Folium map showing:
- MBES bathymetry (from GeoTIFF, colorized overlay)
- Navigation track (from CSV)
- HSI footprints (from georeferenced H5 files)
- HSI RGB overlay (colored points), now heavily thinned to keep file small

All input paths are read from ../config.py
Output: mjosa_mbes_hsi_map.html in the dev/ folder

Usage:
    python dev2_html.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import folium
from folium.plugins import (
    MeasureControl,
    MousePosition,
    LocateControl,
    FastMarkerCluster,
)
from folium import raster_layers
import branca.colormap as bcm

import matplotlib
import matplotlib.colors as mcolors

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

import h5py
from pyproj import Transformer

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gref_pipeline import config


# ------------------------ General helpers ------------------------


def load_nav_df(nav_csv):
    if not os.path.exists(nav_csv):
        raise FileNotFoundError(f"Navigation CSV not found: {nav_csv}")
    df = pd.read_csv(nav_csv)
    df["datetime"] = pd.to_datetime(df["timestamp [unix epoch s]"], unit="s")
    df = df.rename(
        columns={
            "latitude [deg]": "lat",
            "longitude [deg]": "lon",
            "depth [m]": "depth",
        }
    )
    return df


def reproject_to_epsg4326(src_dataset, resampling=Resampling.bilinear):
    """
    Reproject band 1 to EPSG:4326. Returns (arr_4326, transform_4326, bounds_4326).
    bounds_4326 = (west, south, east, north)
    """
    dst_crs = "EPSG:4326"
    transform, width, height = calculate_default_transform(
        src_dataset.crs,
        dst_crs,
        src_dataset.width,
        src_dataset.height,
        *src_dataset.bounds,
    )

    dst = np.full((height, width), np.nan, dtype=np.float32)

    reproject(
        source=rasterio.band(src_dataset, 1),
        destination=dst,
        src_transform=src_dataset.transform,
        src_crs=src_dataset.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=src_dataset.nodata,
        dst_nodata=np.nan,
    )

    # Affine: x = c + a*col + b*row, y = f + d*col + e*row
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height  # e is negative for north-up datasets
    west, east = (min(left, right), max(left, right))
    south, north = (min(bottom, top), max(bottom, top))

    return dst, transform, (west, south, east, north)


def colorize_to_rgba_array(
    data,
    cmap_name="viridis",
    vmin=None,
    vmax=None,
    pct_clip=(2, 98),
    make_transparent_on_nan=True,
):
    """
    Map a single-band float array to an RGBA uint8 array (H,W,4) in memory.
    """
    arr = data.copy()
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("No finite values in MBES array to colorize.")

    if vmin is None or vmax is None:
        vmin = np.nanpercentile(arr, pct_clip[0])
        vmax = np.nanpercentile(arr, pct_clip[1])
        if vmin == vmax:
            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba_float = cmap(norm(arr))  # (H,W,4), floats 0..1

    if make_transparent_on_nan:
        alpha = rgba_float[..., 3]
        alpha[~finite] = 0.0
        rgba_float[..., 3] = alpha

    rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    return rgba_uint8, float(vmin), float(vmax), cmap


def add_mbes_overlay_from_array(
    fmap, rgba_uint8, bounds_latlon, name="MBES overlay", opacity=0.7, show=True
):
    """
    Add an RGBA uint8 numpy array as an overlay. No image is saved to disk.
    bounds_latlon = (west, south, east, north)
    """
    west, south, east, north = bounds_latlon
    overlay = raster_layers.ImageOverlay(
        name=name,
        image=rgba_uint8,
        bounds=[[south, west], [north, east]],
        opacity=opacity,
        interactive=False,
        cross_origin=False,
        zindex=2,
        show=show,
    )
    overlay.add_to(fmap)


def make_branca_colormap(cmap, vmin, vmax, n=10, caption="MBES"):
    xs = np.linspace(0, 1, n)
    hex_colors = [matplotlib.colors.to_hex(cmap(x)) for x in xs]
    cm = bcm.LinearColormap(hex_colors, vmin=vmin, vmax=vmax)
    cm.caption = caption
    return cm


# ------------------------ HSI helpers ------------------------

# Dataset paths to try in H5 files (in order of preference)
_DATASET_CANDIDATES = [
    "processed/georef/points_ecef_crs",  # Main output from main.py
    "processed/georef/points_ecef",
    "georef/points_ecef_crs",
    "georef/points_ecef",
    "intersection_points",  # Alternative naming
]


def _discover_h5_files(root_dir, recursive=True):
    if not root_dir or not os.path.isdir(root_dir):
        raise FileNotFoundError(f"HSI directory not found: {root_dir}")
    pattern = "**/*.h5" if recursive else "*.h5"
    paths = glob.glob(os.path.join(root_dir, pattern), recursive=recursive)
    paths = sorted(p for p in paths if os.path.isfile(p))
    return paths


def _read_hsi_points_ecef_from_file(path):
    """
    Try several dataset paths. Return X, Y, Z arrays, or None if not found.
    The data can be either:
    - (T, S, 3) for gridded hypercubes (T=along-track, S=across-track, 3=XYZ)
    - (N, 3) for flattened point clouds (N=total points, 3=XYZ)
    """
    with h5py.File(path, "r") as f:
        for dset_path in _DATASET_CANDIDATES:
            try:
                d = f[dset_path][()]

                if d.ndim == 3 and d.shape[-1] == 3:
                    # Gridded format (T, S, 3)
                    X = d[:, :, 0]
                    Y = d[:, :, 1]
                    Z = d[:, :, 2]
                    return X, Y, Z, dset_path

                elif d.ndim == 2 and d.shape[-1] == 3:
                    # Flattened format (N, 3) - reshape to grid or keep as 1D
                    X = d[:, 0].reshape(-1, 1)  # (N, 1)
                    Y = d[:, 1].reshape(-1, 1)
                    Z = d[:, 2].reshape(-1, 1)
                    return X, Y, Z, dset_path

            except Exception:
                continue

    return None, None, None, None


def _ecef_to_geodetic_lonlat(X, Y, Z, from_epsg=4978, to_epsg=4979):
    """
    ECEF (from_epsg) to geodetic lon, lat, h (to_epsg).
    """
    tf = Transformer.from_crs(f"EPSG:{from_epsg}", f"EPSG:{to_epsg}", always_xy=True)
    lon, lat, h = tf.transform(X, Y, Z)
    return lon, lat, h


def _extract_rgb_from_h5(
    h5_path,
    red_wl=654.2,
    green_wl=560,
    blue_wl=440.3,
    normalize=True,
    use_corrected=False,
):
    """
    Extract RGB bands from HSI datacube.
    Returns R, G, B arrays (same shape as tracks × slits), or None if failed.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # Read datacube
            dset_path = (
                "processed/radiance/dataCube_corrected"
                if use_corrected
                else "processed/radiance/dataCube"
            )
            if dset_path not in f:
                dset_path = "processed/radiance/dataCube"
            if dset_path not in f:
                return None, None, None

            datacube = f[dset_path][()]  # (T, S, B)

            # Read wavelengths
            wl_path = "processed/radiance/calibration/spectral/band2Wavelength"
            if wl_path not in f:
                return None, None, None
            wavelengths = f[wl_path][()]

            # Find closest band indices
            idx_r = np.argmin(np.abs(wavelengths - red_wl))
            idx_g = np.argmin(np.abs(wavelengths - green_wl))
            idx_b = np.argmin(np.abs(wavelengths - blue_wl))

            # Extract bands
            R = datacube[:, :, idx_r].astype(np.float64)
            G = datacube[:, :, idx_g].astype(np.float64)
            B = datacube[:, :, idx_b].astype(np.float64)

            # Normalize
            if normalize:
                for C in (R, G, B):
                    finite_mask = np.isfinite(C)
                    if finite_mask.any():
                        vmin = C[finite_mask].min()
                        vmax = C[finite_mask].max()
                        if vmax > vmin:
                            C[:] = (C - vmin) / (vmax - vmin)

            return R, G, B

    except Exception as e:
        print(f"    ⚠️  Failed to extract RGB: {e}")
        return None, None, None


def add_hsi_rgb_overlay(
    fmap,
    hsi_h5_paths=None,
    hsi_h5_dir=None,
    recursive=True,
    layer_name="HSI RGB",
    stride_tracks=10,
    stride_slits=10,
    keep_fraction=0.10,  # <--- NEW: random thin after stride (0..1)
    add_tooltip=False,  # <--- NEW: tooltips add a lot of bytes; off by default
    red_wl=654.2,
    green_wl=560,
    blue_wl=440.3,
    normalize=True,
    use_corrected=False,
    point_radius=3,
    point_opacity=0.8,
    ecef_epsg=4978,
    geodetic_epsg=4979,
    rng_seed=12345,  # deterministic thinning
):
    """
    Add HSI RGB overlay to the map - colored points based on actual spectral data.
    The number of points is reduced by:
      1) grid stride (stride_tracks × stride_slits)
      2) random thinning by keep_fraction AFTER striding

    Example: stride=10 and keep_fraction=0.10  -> ~0.1% of original cells.
    """
    files = []
    if hsi_h5_dir:
        files = _discover_h5_files(hsi_h5_dir, recursive=recursive)
        print(f"HSI RGB: found {len(files)} .h5 files under {hsi_h5_dir}")
    if hsi_h5_paths:
        files.extend([p for p in hsi_h5_paths if os.path.isfile(p)])
        files = sorted(set(files))

    if not files:
        print("⚠️  HSI RGB: no files found")
        return

    rng = np.random.default_rng(rng_seed)
    grp = folium.FeatureGroup(name=layer_name, show=True)
    ok_files = 0

    for idx, fp in enumerate(files, 1):
        basename = os.path.basename(fp)
        print(f"  [{idx}/{len(files)}] {basename}...", end=" ")

        # Get ECEF coordinates
        X, Y, Z, used_path = _read_hsi_points_ecef_from_file(fp)
        if X is None:
            print(f"❌ No ECEF data")
            continue

        # Get RGB values
        R, G, B = _extract_rgb_from_h5(
            fp, red_wl, green_wl, blue_wl, normalize, use_corrected
        )
        if R is None:
            print(f"❌ No RGB data")
            continue

        # Convert ECEF to lat/lon
        try:
            lon, lat, _ = _ecef_to_geodetic_lonlat(
                X, Y, Z, from_epsg=ecef_epsg, to_epsg=geodetic_epsg
            )
        except Exception as e:
            print(f"❌ Transform failed: {e}")
            continue

        # Add colored points with stride + thinning
        T, S = lat.shape

        track_indices = list(range(0, T, stride_tracks))
        slit_indices = list(range(0, S, stride_slits))
        total_cells = len(track_indices) * len(slit_indices)
        est_kept = int(total_cells * keep_fraction)

        n_points = 0
        progress_update = max(1, total_cells // 20)

        print(
            f"adding ~{est_kept:,} / {total_cells:,} cells (stride {stride_tracks}×{stride_slits}, keep {keep_fraction:.0%})...",
            end="",
            flush=True,
        )

        # Reduce precision of coords to shrink HTML a little
        def _round6(x):
            return float(np.round(x, 6))

        for idx_i, i in enumerate(track_indices):
            for idx_j, j in enumerate(slit_indices):
                if keep_fraction < 1.0 and rng.random() > keep_fraction:
                    continue

                la = lat[i, j]
                lo = lon[i, j]
                r = R[i, j]
                g = G[i, j]
                b = B[i, j]

                if not (
                    np.isfinite(la)
                    and np.isfinite(lo)
                    and np.isfinite(r)
                    and np.isfinite(g)
                    and np.isfinite(b)
                ):
                    continue

                # Convert RGB [0,1] to hex color
                r_byte = int(np.clip(r * 255, 0, 255))
                g_byte = int(np.clip(g * 255, 0, 255))
                b_byte = int(np.clip(b * 255, 0, 255))
                hex_color = f"#{r_byte:02x}{g_byte:02x}{b_byte:02x}"

                tooltip_txt = None
                if add_tooltip:
                    tooltip_txt = (
                        f"{basename} | track {i}, slit {j}<br>"
                        f"RGB: ({r:.3f}, {g:.3f}, {b:.3f})"
                    )

                folium.CircleMarker(
                    location=[_round6(float(la)), _round6(float(lo))],
                    radius=point_radius,
                    color=hex_color,
                    fill=True,
                    fillColor=hex_color,
                    fillOpacity=point_opacity,
                    opacity=point_opacity,
                    weight=0,
                    tooltip=tooltip_txt,
                ).add_to(grp)
                n_points += 1

                cell_num = idx_i * len(slit_indices) + idx_j
                if cell_num > 0 and cell_num % progress_update == 0:
                    pct = int(100 * cell_num / total_cells)
                    print(
                        f"\r  [{idx}/{len(files)}] {basename}... {pct}% (~{n_points:,} kept)",
                        end="",
                        flush=True,
                    )

        print(
            f"\r  [{idx}/{len(files)}] {basename}... ✅ kept {n_points:,} points     "
        )
        ok_files += 1

    if ok_files > 0:
        grp.add_to(fmap)
        print(f"✅ HSI RGB: added {ok_files}/{len(files)} files to '{layer_name}'")
    else:
        print(f"❌ HSI RGB: no usable files")


def add_hsi_from_ecef(
    fmap,
    hsi_h5_paths=None,
    hsi_h5_dir=None,
    recursive=True,
    layer_name="HSI footprint",
    stride_tracks=4,
    stride_slits=4,
    add_lines=True,
    add_points=False,
    point_cluster=False,
    color="#ff00ff",
    line_weight=1,
    line_opacity=0.6,
    point_radius=1,
    point_opacity=0.7,
    ecef_epsg=4978,
    geodetic_epsg=4979,
):
    """
    Add hyperspectral footprint to the Folium map in WGS84 lat, lon.
    You can provide a folder (hsi_h5_dir) or an explicit list of files (hsi_h5_paths).
    """
    files = []
    if hsi_h5_dir:
        files = _discover_h5_files(hsi_h5_dir, recursive=recursive)
        print(f"HSI: found {len(files)} .h5 files under {hsi_h5_dir}")
    if hsi_h5_paths:
        files.extend([p for p in hsi_h5_paths if os.path.isfile(p)])
        files = sorted(set(files))
        print(f"HSI: using {len(files)} files after merging provided list")

    if not files:
        print("⚠️  HSI: no files found, skipping HSI layer")
        return

    grp = folium.FeatureGroup(name=layer_name, show=True)

    ok_files = 0
    failed_files = 0
    for idx, fp in enumerate(files, 1):
        basename = os.path.basename(fp)
        print(f"  [{idx}/{len(files)}] Processing {basename}...", end=" ")

        X, Y, Z, used_path = _read_hsi_points_ecef_from_file(fp)
        if X is None:
            print(f"❌ No ECEF data found")
            print(f"      Tried: {', '.join(_DATASET_CANDIDATES)}")
            failed_files += 1
            continue

        try:
            lon, lat, _ = _ecef_to_geodetic_lonlat(
                X, Y, Z, from_epsg=ecef_epsg, to_epsg=geodetic_epsg
            )
            print(f"✅ {used_path}, shape={lat.shape}")
        except Exception as e:
            print(f"❌ CRS transform failed: {e}")
            failed_files += 1
            continue

        # Lines along tracks
        if add_lines:
            T, S = lat.shape
            num_lines = 0
            for i in range(0, T, stride_tracks):
                la_row = lat[i, ::stride_slits]
                lo_row = lon[i, ::stride_slits]

                valid_mask = np.isfinite(la_row) & np.isfinite(lo_row)
                la_valid = la_row[valid_mask]
                lo_valid = lo_row[valid_mask]

                if len(la_valid) < 2:
                    continue

                coords = [
                    [float(la), float(lo)]
                    for la, lo in zip(la_valid.tolist(), lo_valid.tolist())
                ]
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=line_weight,
                    opacity=line_opacity,
                    tooltip=f"{basename} | {used_path} | track {i}",
                ).add_to(grp)
                num_lines += 1

        # Points (outline) — off by default; they bloat the HTML
        if add_points:
            lat_s = lat[::stride_tracks, ::stride_slits].ravel()
            lon_s = lon[::stride_tracks, ::stride_slits].ravel()

            valid_mask = np.isfinite(lat_s) & np.isfinite(lon_s)
            lat_s = lat_s[valid_mask]
            lon_s = lon_s[valid_mask]

            pts = [[float(la), float(lo)] for la, lo in zip(lat_s, lon_s)]
            if point_cluster:
                FastMarkerCluster(pts, name=f"HSI points {basename}").add_to(grp)
            else:
                for la, lo in pts:
                    folium.CircleMarker(
                        [la, lo],
                        radius=point_radius,
                        color=color,
                        fill=False,
                        opacity=point_opacity,
                    ).add_to(grp)

        ok_files += 1

    print()
    if ok_files == 0:
        print("❌ HSI: no usable HSI files, skipping layer")
        return

    grp.add_to(fmap)
    print(f"✅ HSI: added {ok_files}/{len(files)} files to layer '{layer_name}'")
    if failed_files > 0:
        print(f"⚠️  HSI: {failed_files} files failed to load")


# ------------------------ Main builder ------------------------


def build_map_with_mbes(
    nav_csv,
    geotiff_path,
    output_html="map_with_mbes.html",
    cmap_name="viridis",
    opacity=0.7,
    pct_clip=(2, 98),
    vmin=None,
    vmax=None,
    legend_caption="MBES (depth, m)",
    # OPTIONAL: Multiple TIF overlays from folder
    tif_folder=None,
    tif_opacity=0.7,
    # HSI inputs
    hsi_h5_paths=None,
    hsi_h5_dir=None,
    hsi_recursive=True,
    hsi_layer_name="HSI footprint",
    hsi_stride_tracks=10,
    hsi_stride_slits=10,
    hsi_add_lines=True,
    hsi_add_points=False,
    hsi_point_cluster=False,
    hsi_color="#ff00ff",
    # HSI RGB overlay settings
    hsi_add_rgb=True,
    hsi_rgb_layer_name="HSI RGB",
    hsi_rgb_stride_tracks=10,
    hsi_rgb_stride_slits=10,
    hsi_rgb_keep_fraction=0.10,  # <--- NEW: keep ~10% of the strided points
    hsi_rgb_add_tooltip=False,  # <--- NEW: tooltips disabled to cut size
    hsi_rgb_red_wl=654.2,
    hsi_rgb_green_wl=560,
    hsi_rgb_blue_wl=440.3,
    hsi_rgb_normalize=True,
    hsi_rgb_use_corrected=False,
    hsi_rgb_point_radius=3,
    hsi_rgb_point_opacity=0.8,
    # HSI CRS settings
    hsi_ecef_epsg=4978,
    hsi_geodetic_epsg=4979,
):
    # 1) Load navigation data
    nav = load_nav_df(nav_csv)

    # 2) Base map centered on track
    lat_center = float(nav["lat"].mean())
    lon_center = float(nav["lon"].mean())
    fmap = folium.Map(
        location=[lat_center, lon_center],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Extras
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(fmap)
    fmap.add_child(folium.LatLngPopup())
    MousePosition(
        position="bottomright",
        separator=" , ",
        prefix="Lat/Lng:",
        lat_formatter="function(num){return L.Util.formatNum(num,8);} ",
        lng_formatter="function(num){return L.Util.formatNum(num,8);} ",
    ).add_to(fmap)
    LocateControl(auto_start=False, position="topright").add_to(fmap)
    MeasureControl(
        position="topleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
    ).add_to(fmap)

    # 3) Read and reproject MBES GeoTIFF to EPSG:4326
    if not os.path.exists(geotiff_path):
        raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")
    with rasterio.open(geotiff_path) as src:
        arr4326, transform4326, bounds4326 = reproject_to_epsg4326(src)

    # 4) Colorize to RGBA array in memory
    rgba_uint8, used_vmin, used_vmax, mpl_cmap = colorize_to_rgba_array(
        arr4326,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        pct_clip=pct_clip,
        make_transparent_on_nan=True,
    )

    # 5) Overlay RGBA array directly
    add_mbes_overlay_from_array(
        fmap,
        rgba_uint8,
        bounds4326,
        name="MBES overlay",
        opacity=opacity,
        show=True,
    )

    # 5b) OPTIONAL: Add multiple TIF overlays from folder
    if tif_folder and os.path.isdir(tif_folder):
        print(f"\n📂 Processing additional TIF files from: {tif_folder}")
        tif_files = sorted(glob.glob(os.path.join(tif_folder, "*.tif")))
        tif_files.extend(sorted(glob.glob(os.path.join(tif_folder, "*.tiff"))))
        tif_files = sorted(set(tif_files))  # Remove duplicates and sort

        if len(tif_files) > 0:
            print(f"   Found {len(tif_files)} TIF file(s)")

            for idx, tif_path in enumerate(tif_files, 1):
                tif_basename = os.path.basename(tif_path)
                print(
                    f"   [{idx}/{len(tif_files)}] Processing {tif_basename}...", end=" "
                )

                try:
                    with rasterio.open(tif_path) as src:
                        arr_tif, _, bounds_tif = reproject_to_epsg4326(src)

                    # Use same colormap and vmin/vmax as original MBES
                    rgba_tif, _, _, _ = colorize_to_rgba_array(
                        arr_tif,
                        cmap_name=cmap_name,
                        vmin=used_vmin,
                        vmax=used_vmax,
                        pct_clip=pct_clip,
                        make_transparent_on_nan=True,
                    )

                    # Add overlay with TIF filename as layer name, hidden by default
                    add_mbes_overlay_from_array(
                        fmap,
                        rgba_tif,
                        bounds_tif,
                        name=tif_basename,
                        opacity=tif_opacity,
                        show=False,  # Hidden by default
                    )
                    print("✅")

                except Exception as e:
                    print(f"❌ Error: {e}")

            print(
                f"✅ Added {len(tif_files)} additional TIF overlay(s) (toggle via layer control)"
            )
        else:
            print(f"⚠️  No TIF files found in {tif_folder}")

    # 6) Add navigation track and start/end markers
    coords = list(zip(nav["lat"].to_numpy(), nav["lon"].to_numpy()))
    folium.PolyLine(
        coords, color="cyan", weight=3, opacity=0.9, tooltip="Navigation Track"
    ).add_to(folium.FeatureGroup(name="Navigation Track", show=True).add_to(fmap))

    start = nav.iloc[0]
    end = nav.iloc[-1]
    folium.Marker(
        [float(start["lat"]), float(start["lon"])],
        tooltip="Start",
        popup=f"START<br>{start['datetime']}<br>Depth: {start['depth']:.1f} m",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fmap)
    folium.Marker(
        [float(end["lat"]), float(end["lon"])],
        tooltip="End",
        popup=f"END<br>{end['datetime']}<br>Depth: {end['depth']:.1f} m",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fmap)

    # 7) Legend
    branca_cmap = make_branca_colormap(
        mpl_cmap, used_vmin, used_vmax, n=12, caption=legend_caption
    )
    branca_cmap.add_to(fmap)

    # 8) HSI footprint layer (outline)
    if hsi_h5_dir or hsi_h5_paths:
        add_hsi_from_ecef(
            fmap,
            hsi_h5_paths=hsi_h5_paths,
            hsi_h5_dir=hsi_h5_dir,
            recursive=hsi_recursive,
            layer_name=hsi_layer_name,
            stride_tracks=hsi_stride_tracks,
            stride_slits=hsi_stride_slits,
            add_lines=hsi_add_lines,
            add_points=hsi_add_points,
            point_cluster=hsi_point_cluster,
            color=hsi_color,
            ecef_epsg=hsi_ecef_epsg,
            geodetic_epsg=hsi_geodetic_epsg,
        )

    # 8b) HSI RGB overlay (spectral colors) — heavily thinned
    if hsi_add_rgb and (hsi_h5_dir or hsi_h5_paths):
        add_hsi_rgb_overlay(
            fmap,
            hsi_h5_paths=hsi_h5_paths,
            hsi_h5_dir=hsi_h5_dir,
            recursive=hsi_recursive,
            layer_name=hsi_rgb_layer_name,
            stride_tracks=hsi_rgb_stride_tracks,
            stride_slits=hsi_rgb_stride_slits,
            keep_fraction=hsi_rgb_keep_fraction,  # << keep only 10% after striding
            add_tooltip=hsi_rgb_add_tooltip,  # off by default to shrink HTML
            red_wl=hsi_rgb_red_wl,
            green_wl=hsi_rgb_green_wl,
            blue_wl=hsi_rgb_blue_wl,
            normalize=hsi_rgb_normalize,
            use_corrected=hsi_rgb_use_corrected,
            point_radius=hsi_rgb_point_radius,
            point_opacity=hsi_rgb_point_opacity,
            ecef_epsg=hsi_ecef_epsg,
            geodetic_epsg=hsi_geodetic_epsg,
        )

    # 9) Controls and save
    print("\n💾 Adding layer controls and saving map...")
    folium.LayerControl(collapsed=False).add_to(fmap)

    print("   Writing HTML file", end="", flush=True)

    # Simple progress indicator using threading
    import threading
    import time

    start_time = time.time()
    stop_spinner = False

    def spinner():
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        t0 = time.time()
        while not stop_spinner:
            elapsed = int(time.time() - t0)
            print(
                f"\r   Writing HTML file {chars[idx % len(chars)]} ({elapsed}s)",
                end="",
                flush=True,
            )
            idx += 1
            time.sleep(0.1)

    spinner_thread = threading.Thread(target=spinner, daemon=True)
    spinner_thread.start()

    try:
        fmap.save(output_html)
    finally:
        stop_spinner = True
        time.sleep(0.2)

    elapsed_total = int(time.time() - start_time)
    print(f"\r   Writing HTML file... Done! ({elapsed_total}s)                ")

    file_size_mb = os.path.getsize(output_html) / (1024 * 1024)
    print(f"\n✅ Map saved: {output_html}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"🎨 Legend range: vmin={used_vmin:.3f}, vmax={used_vmax:.3f}")
    return output_html


# ------------------------ Run it ------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Building interactive map with MBES + HSI RGB overlay (thinned)")
    print("=" * 70)
    print("⚡ New default: keep only 10% of strided RGB points (very small HTML)")
    print("=" * 70)

    # Input paths from config
    NAV_CSV = config.NAV_CSV
    GEOTIFF = config.MBES_GEOTIFF
    HSI_DIR = config.OUTPUT_FOLDER  # Georeferenced H5 files

    # Output in dev folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_HTML = os.path.join(script_dir, "mjosa_mbes_hsi_map.html")

    print(f"\n📍 Input paths:")
    print(f"  Navigation CSV: {NAV_CSV}")
    print(f"  MBES GeoTIFF:   {GEOTIFF}")
    print(f"  HSI H5 folder:  {HSI_DIR}")
    print(f"\n📄 Output:")
    print(f"  HTML map: {OUTPUT_HTML}\n")

    # Verify paths exist
    if not os.path.exists(NAV_CSV):
        print(f"❌ ERROR: Navigation CSV not found: {NAV_CSV}")
        sys.exit(1)
    if not os.path.exists(GEOTIFF):
        print(f"❌ ERROR: MBES GeoTIFF not found: {GEOTIFF}")
        sys.exit(1)
    if not os.path.isdir(HSI_DIR):
        print(f"❌ ERROR: HSI output folder not found: {HSI_DIR}")
        sys.exit(1)

    # Count H5 files
    h5_files = glob.glob(os.path.join(HSI_DIR, "*.h5"))
    print(f"✅ Found {len(h5_files)} H5 files in output folder")
    if len(h5_files) == 0:
        print(f"⚠️  WARNING: No H5 files found. Map will only show MBES and navigation.")

    build_map_with_mbes(
        NAV_CSV,
        GEOTIFF,
        output_html=OUTPUT_HTML,
        cmap_name="viridis",
        opacity=0.70,
        pct_clip=(2, 98),
        vmin=None,
        vmax=None,
        legend_caption="MBES (depth, m)",
        # HSI via folder (georeferenced H5 files from main.py)
        hsi_h5_dir=HSI_DIR,
        hsi_recursive=False,  # All H5 files are directly in OUTPUT_FOLDER
        hsi_layer_name="HSI footprint (outline)",
        hsi_stride_tracks=10,
        hsi_stride_slits=10,
        hsi_add_lines=True,
        hsi_add_points=False,
        hsi_point_cluster=False,
        hsi_color="#ffff00",
        # HSI RGB overlay (actual spectral data colors) — very small footprint
        hsi_add_rgb=True,
        hsi_rgb_layer_name="HSI RGB (spectral, thinned)",
        hsi_rgb_stride_tracks=10,  # keep these; thinning happens after striding
        hsi_rgb_stride_slits=10,
        hsi_rgb_keep_fraction=0.05,  # keep ~5% of strided cells
        hsi_rgb_add_tooltip=False,  # tooltips removed for size
        hsi_rgb_red_wl=654.2,
        hsi_rgb_green_wl=560,
        hsi_rgb_blue_wl=440.3,
        hsi_rgb_normalize=True,
        hsi_rgb_use_corrected=False,
        hsi_rgb_point_radius=0.3,
        hsi_rgb_point_opacity=0.8,
        # EPSG codes from config
        hsi_ecef_epsg=config.EPSG_ECEF,
        hsi_geodetic_epsg=4979,  # WGS84 geodetic (lon, lat, h)
    )

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"📂 Open the map in your browser:")
    print(f"   {OUTPUT_HTML}\n")
    print("💡 Tip: Toggle layers on/off using the layer control (top right)")
    print(
        "   - 'HSI RGB (spectral, thinned)' shows the hyperspectral colors at ~10% density"
    )
    print("   - 'HSI footprint (outline)' shows the coverage area")
    print("\n⚙️  To make it EVEN smaller:")
    print(
        "   - Increase hsi_rgb_stride_* and/or reduce hsi_rgb_keep_fraction to 0.05 or 0.02"
    )
    print("=" * 70)


def test_multiple_tifs():
    """
    Test function to verify multiple TIF overlay feature.
    Uses the additional TIF folder provided by user.
    """
    print("=" * 70)
    print("TEST: Building map with multiple TIF overlays")
    print("=" * 70)

    # Input paths from config
    NAV_CSV = config.NAV_CSV
    GEOTIFF = config.MBES_GEOTIFF
    HSI_DIR = config.OUTPUT_FOLDER

    # Additional TIF folder
    TIF_FOLDER = r"E:\mjosa_new_oct_2025\all_tifs_from_eiva\relevant_tifs_only"

    # Output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_HTML = os.path.join(script_dir, "test_multiple_tifs_map.html")

    print(f"\n📍 Input paths:")
    print(f"  Navigation CSV:    {NAV_CSV}")
    print(f"  Main MBES GeoTIFF: {GEOTIFF}")
    print(f"  Additional TIFs:   {TIF_FOLDER}")
    print(f"  HSI H5 folder:     {HSI_DIR}")
    print(f"\n📄 Output:")
    print(f"  HTML map: {OUTPUT_HTML}\n")

    # Verify paths
    if not os.path.exists(NAV_CSV):
        print(f"❌ ERROR: Navigation CSV not found: {NAV_CSV}")
        sys.exit(1)
    if not os.path.exists(GEOTIFF):
        print(f"❌ ERROR: MBES GeoTIFF not found: {GEOTIFF}")
        sys.exit(1)
    if not os.path.isdir(TIF_FOLDER):
        print(f"❌ ERROR: TIF folder not found: {TIF_FOLDER}")
        sys.exit(1)

    build_map_with_mbes(
        NAV_CSV,
        GEOTIFF,
        output_html=OUTPUT_HTML,
        cmap_name="viridis",
        opacity=0.70,
        pct_clip=(2, 98),
        vmin=None,
        vmax=None,
        legend_caption="MBES (depth, m)",
        # NEW FEATURE: Multiple TIF overlays
        tif_folder=TIF_FOLDER,
        tif_opacity=0.70,
        # HSI settings (minimal for testing)
        hsi_h5_dir=HSI_DIR,
        hsi_recursive=False,
        hsi_layer_name="HSI footprint (outline)",
        hsi_stride_tracks=10,
        hsi_stride_slits=10,
        hsi_add_lines=True,
        hsi_add_points=False,
        hsi_color="#ffff00",
        hsi_add_rgb=False,  # Disable RGB for faster testing
        hsi_ecef_epsg=config.EPSG_ECEF,
        hsi_geodetic_epsg=4979,
    )

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)
    print(f"📂 Open the test map in your browser:")
    print(f"   {OUTPUT_HTML}\n")
    print("💡 Check the layer control (top right):")
    print("   - Main 'MBES overlay' should be visible by default")
    print("   - Additional TIF layers (named by timestamp) should be hidden")
    print("   - Toggle each TIF layer to verify it appears correctly")
    print("=" * 70)
