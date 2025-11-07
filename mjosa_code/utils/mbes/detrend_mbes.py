"""
Detrend MBES bathymetry data and visualize residuals.

This module contains the ``MBESDetrender`` class along with a few helper
functions and classes for detrending multibeam echo sounder (MBES)
swath data.  It reproduces functionality from a previous implementation
and adds an optional overlay for highlighting a user‑defined footprint on
the residual panel of the triptych plot.  When a footprint is
provided, the corresponding area on the residual heatmap will be
filled with black while the surrounding data remain unchanged.

The overlay accepts either a precomputed Matplotlib ``Path`` or a
tuple ``(E, N, mask)`` where ``E`` and ``N`` are the eastings and
northings of the UHI (hyperspectral) grid and ``mask`` is a boolean
array indicating which UHI pixels belong to the footprint.  If a tuple
is supplied, the code will trace the 0.5 contour of the mask to
construct a compound path.  Passing ``uhi_footprint=None`` disables
the overlay and restores the original behaviour.

This file is designed to be imported from Jupyter notebooks using the
following pattern::

    import sys, os
    sys.path.append(os.path.abspath("../"))
    from utils.other.detrend_mbes import MBESDetrender

You can then create an instance, detrend the data, and plot the
results with an optional footprint overlay:

    mb = MBESDetrender(tif_path, coord_system="ned", ned_origin=(lon0, lat0, h0))
    mb.load().detrend(order_x=2)
    # compute (E, N, mask) for your UHI footprint
    mb.plot_triptych(uhi_footprint=(E, N, mask))

The rest of the API mirrors the original implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Union

import os
import csv
import json
import time
import warnings

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation
from pyproj import Transformer

__all__ = ["MBESDetrender", "ScoreWeights", "RunResult", "AutoTuneMBES"]


# --------------------------- UHI footprint helpers ---------------------------


def pcolormesh_pad(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad coordinate arrays for pcolormesh corner coordinates.

    Parameters
    ----------
    X, Y : np.ndarray
        2D coordinate arrays at cell centers (shape: T×S).

    Returns
    -------
    Xc, Yc : np.ndarray
        2D coordinate arrays at cell corners (shape: (T+1)×(S+1)).
    """
    Xc = np.pad(X, ((0, 1), (0, 1)), mode="edge")
    Yc = np.pad(Y, ((0, 1), (0, 1)), mode="edge")
    return Xc, Yc


def _boundary_segments_from_mask(
    mask: np.ndarray, Xc: np.ndarray, Yc: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Build boundary segments EXACTLY along cell edges from a boolean mask.

    This function extracts all edges where the mask transitions from True to
    False (or vice versa), as well as all outer edges where the mask is True.
    The result is a set of line segments that precisely trace the boundary of
    the True region in the mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean 2D array (shape: T×S) indicating valid pixels.
    Xc, Yc : np.ndarray
        2D coordinate arrays at cell corners (shape: (T+1)×(S+1)).

    Returns
    -------
    segs_xy : np.ndarray
        Array of shape (N, 2, 2) containing [[x0,y0],[x1,y1]] for each segment.
    segs_idx : list
        List of ((i0,j0),(i1,j1)) corner-index pairs for loop tracing.
    """
    T, S = mask.shape
    segs_xy = []
    segs_idx = []

    # Vertical internal edges (between columns)
    vdiff = mask[:, 1:] != mask[:, :-1]  # shape (T, S-1)
    iv, jv = np.nonzero(vdiff)
    jline = jv + 1
    for i, j in zip(iv, jline):
        a = (i, j)
        b = (i + 1, j)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # Left outer edge (j=0)
    i_left = np.nonzero(mask[:, 0])[0]
    for i in i_left:
        a = (i, 0)
        b = (i + 1, 0)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # Right outer edge (j=S)
    i_right = np.nonzero(mask[:, -1])[0]
    for i in i_right:
        a = (i, S)
        b = (i + 1, S)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # Horizontal internal edges (between rows)
    hdiff = mask[1:, :] != mask[:-1, :]  # shape (T-1, S)
    ih, jh = np.nonzero(hdiff)
    iline = ih + 1
    for i, j in zip(iline, jh):
        a = (i, j)
        b = (i, j + 1)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # Top outer edge (i=0)
    j_top = np.nonzero(mask[0, :])[0]
    for j in j_top:
        a = (0, j)
        b = (0, j + 1)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # Bottom outer edge (i=T)
    j_bot = np.nonzero(mask[-1, :])[0]
    for j in j_bot:
        a = (T, j)
        b = (T, j + 1)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    return np.asarray(segs_xy, dtype=float), segs_idx


def _trace_loops_from_segments(
    segs_idx: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    """Convert corner-index segments into closed loops of corner indices.

    This function builds an adjacency graph from the segments and traces closed
    loops by following edges. It assumes the boundary graph is composed of
    closed rings (typical for a mask boundary).

    Parameters
    ----------
    segs_idx : list
        List of ((i0,j0),(i1,j1)) corner-index pairs.

    Returns
    -------
    loops : list
        List of loops, where each loop is a list of (i,j) corner indices.
    """
    # Build adjacency (undirected)
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for a, b in segs_idx:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Visit edges
    unvisited = set(frozenset((a, b)) for a, b in segs_idx)
    loops: List[List[Tuple[int, int]]] = []

    while unvisited:
        # Start from any remaining edge
        e = unvisited.pop()
        a, b = tuple(e)
        # Build a loop starting with (a->b)
        loop = [a, b]
        prev, cur = a, b

        while True:
            nbrs = adj[cur]
            # Choose next neighbor that is not the previous vertex
            nxt_candidates = [n for n in nbrs if n != prev]
            if not nxt_candidates:
                break  # open edge (shouldn't happen)
            nxt = nxt_candidates[0]
            # mark edge cur<->nxt visited
            edge = frozenset((cur, nxt))
            if edge in unvisited:
                unvisited.remove(edge)
            loop.append(nxt)
            prev, cur = cur, nxt
            if nxt == loop[0]:
                break  # closed

        loops.append(loop)

        # Remove any already-visited edges incident to this loop
        i = 0
        while i < len(loop) - 1:
            edge = frozenset((loop[i], loop[i + 1]))
            unvisited.discard(edge)
            i += 1

    return loops


def _compound_path_from_loops(
    loops: List[List[Tuple[int, int]]], Xc: np.ndarray, Yc: np.ndarray
) -> MplPath:
    """Create a matplotlib Compound Path from corner-index loops.

    Parameters
    ----------
    loops : list
        List of loops, where each loop is a list of (i,j) corner indices.
    Xc, Yc : np.ndarray
        2D coordinate arrays at cell corners.

    Returns
    -------
    path : matplotlib.path.Path
        Compound path ready for plotting.
    """
    paths = []
    for ring in loops:
        xs = [Xc[idx] for idx in ring]
        ys = [Yc[idx] for idx in ring]
        # Ensure closed
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xs.append(xs[0])
            ys.append(ys[0])
        verts = np.column_stack([xs, ys])
        codes = np.full(len(verts), MplPath.LINETO, dtype=MplPath.code_type)
        codes[0] = MplPath.MOVETO
        codes[-1] = MplPath.CLOSEPOLY
        paths.append(MplPath(verts, codes))
    if len(paths) == 1:
        return paths[0]
    return MplPath.make_compound_path(*paths)


def _robust_z(a: np.ndarray) -> np.ndarray:
    """Compute robust z-scores using median and MAD.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    z : np.ndarray
        Robust z-scores: (a - median) / MAD, where MAD is scaled to match
        standard deviation (MAD * 1.4826). Returns NaN for all-NaN input.
    """
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.full_like(a, np.nan, dtype=float)
    m = np.nanmedian(a)
    mad = median_abs_deviation(a[finite], scale="normal")
    if not np.isfinite(mad) or mad == 0:
        mad = 1.0
    return (a - m) / mad


def _minmax_normalize(a: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range using min-max normalization.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    normalized : np.ndarray
        Normalized array with values in [0, 1] range. Returns NaN for all-NaN input.
    """
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.full_like(a, np.nan, dtype=float)
    vmin = np.nanmin(a)
    vmax = np.nanmax(a)
    if vmax == vmin:
        # All values are the same, return 0.5 (middle of range)
        return np.full_like(a, 0.5, dtype=float)
    return (a - vmin) / (vmax - vmin)


# --------------------------- plotting helpers --------------------------------


def _pad_limits(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pad_frac: float = 0.03,
    equal_aspect: bool = True,
) -> None:
    """Pad axis limits by a fraction of the range and optionally enforce equal aspect.

    This helper expands the provided limits slightly to give plots some breathing
    room around the data.  When ``equal_aspect`` is True, the aspect ratio of
    the axes will be set to equal.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axes on which to set limits and aspect ratio.
    x_min, x_max, y_min, y_max: float
        The minimum and maximum values of the data along the x and y axes.
    pad_frac: float, optional
        Fraction by which to pad the limits.  Defaults to 0.03 (3%).
    equal_aspect: bool, optional
        If True, enforce an equal aspect ratio.  Defaults to True.
    """
    xr = max(1e-12, x_max - x_min)
    yr = max(1e-12, y_max - y_min)
    ax.set_xlim(x_min - xr * pad_frac, x_max + xr * pad_frac)
    ax.set_ylim(y_min - yr * pad_frac, y_max + yr * pad_frac)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
        try:
            ax.set_box_aspect(yr / xr)  # Matplotlib ≥ 3.3 supports set_box_aspect
        except Exception:
            pass
    try:
        # avoid scientific notation for small ranges
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    except Exception:
        pass


def _robust_sym_vlim(arr: np.ndarray, q: float = 0.98) -> Tuple[float, float]:
    """Compute symmetric colour limits based on robust percentiles.

    This function computes the absolute value of the 98th percentile
    (or specified quantile ``q``) of the input array and returns
    symmetric limits ``(-vmax, vmax)``.  If the array contains no finite
    values, default limits of (-1, 1) are returned.

    Parameters
    ----------
    arr: ndarray
        Input array containing data to be visualized.
    q: float, optional
        Quantile used to estimate the colour limits.  Defaults to 0.98.
    """
    m = np.isfinite(arr)
    if not m.any():
        return (-1, 1)
    lo, hi = np.nanquantile(arr[m], [(1 - q) / 2, 1 - (1 - q) / 2])
    vmax = max(abs(lo), abs(hi))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = (np.nanstd(arr[m]) or 1.0) * 3
    return (-vmax, vmax)


# --------------------------- smoothing helpers -------------------------------


def _rows_from_meters(pix_size_y: float, meters: float, poly: int = 2) -> int:
    """Convert smoothing length in metres to window length in rows for Savitzky–Golay.

    Ensures that the window length is odd and large enough for the polynomial
    order.  If the array of valid values is shorter than the desired window,
    a fallback smoothing method will be used.

    Parameters
    ----------
    pix_size_y: float
        Pixel height in metres along the y dimension.
    meters: float
        Desired smoothing length in metres.
    poly: int, optional
        Polynomial order used in the Savitzky–Golay filter.  Defaults to 2.

    Returns
    -------
    int
        Number of rows to use as the filter window.
    """
    n = max(5, int(round((meters or 0.0) / max(abs(pix_size_y) or 1.0, 1e-9))))
    if n % 2 == 0:
        n += 1
    if n <= poly + 2:
        # ensure window length is large enough
        n = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
    return n


def _savgol_nanaware(
    arr: Sequence[float], pix_size_y: float, meters: float, poly: int = 2
) -> np.ndarray:
    """Apply a Savitzky–Golay filter along rows with NaN handling.

    This function first fills internal NaN gaps with linear interpolation,
    applies a Savitzky–Golay filter of appropriate window size, and returns
    the smoothed array.  If insufficient data points are available to
    support the chosen window length, a Gaussian smoothing fallback is
    applied instead.

    Parameters
    ----------
    arr: sequence of float
        Input array (1‑D) to smooth.
    pix_size_y: float
        Pixel height in metres along the y dimension.
    meters: float
        Desired smoothing length in metres.
    poly: int, optional
        Polynomial order used in the Savitzky–Golay filter.  Defaults to 2.

    Returns
    -------
    ndarray
        Smoothed output array with the same shape as ``arr``.
    """
    a = np.asarray(arr, float)
    valid = np.isfinite(a)
    if valid.sum() < 3:
        return a.copy()
    nwin = _rows_from_meters(pix_size_y, meters, poly=poly)
    if valid.sum() < nwin:
        # Fallback: NaN‑aware Gaussian via normalization
        tmp = a.copy()
        tmp[~valid] = 0.0
        w = valid.astype(float)
        sigma = max(0.5, nwin / 6)
        num = gaussian_filter1d(tmp, sigma=sigma, mode="reflect", truncate=3.5)
        den = gaussian_filter1d(w, sigma=sigma, mode="reflect", truncate=3.5)
        out = np.where(den > 1e-6, num / den, np.nan)
        out[~valid] = np.nan
        return out
    # Prefill internal gaps with linear interpolation, then SG filter
    filled = a.copy()
    if (~valid).any():
        idx = np.arange(filled.size)
        filled[~valid] = np.interp(idx[~valid], idx[valid], filled[valid])
    out = savgol_filter(filled, window_length=nwin, polyorder=poly, mode="interp")
    return out


# ------------------------------ MBESDetrender --------------------------------


@dataclass
class MBESDetrender:
    """Detrend multibeam echo sounder (MBES) raster data.

    This class loads a raster (e.g., a GeoTIFF containing bathymetry), fits
    and subtracts a smoothly varying trend along the cross‑track direction,
    and computes residuals representing local variations.  The trend is
    modelled with a per‑row polynomial fitted at the swath centre and
    smoothed along the along‑track direction.  Plotting functions are
    provided to visualize the original data, the estimated trend, and
    the residuals.

    The ``coord_system`` parameter controls whether the output coordinates
    (x_cols, y_rows) are returned in UTM (default) or transformed to a
    local North‑East‑Down frame (``ned``).  When using ``ned``, a
    ``ned_origin`` (lon0, lat0, h0) must be provided.
    """

    tif_path: str
    nodata_value: Optional[float] = 9999.0
    coord_system: str = "utm"
    ned_origin: Optional[Tuple[float, float, float]] = None  # (lon0, lat0, h0)
    epsg_utm: int = 32632
    epsg_geo: int = 4326
    smooth_baseline_m: float = 10.0
    smooth_tilt_m: float = 20.0
    smooth_curv_m: Optional[float] = None
    smooth_center_m: float = 6.0
    max_row_iters: int = 3
    clip_sigma: float = 6.0
    central_frac: float = 0.8
    # UHI footprint overlay parameters
    uhi_transect_folder: Optional[str] = None
    uhi_files: Optional[List[str]] = None
    uhi_track_range: Optional[Tuple[int, int]] = None
    # internal fields
    data: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    transform: Optional[rasterio.Affine] = None
    extent: Optional[Tuple[float, float, float, float]] = None
    px_x: Optional[float] = None
    px_y: Optional[float] = None
    x_cols: Optional[np.ndarray] = None
    y_rows: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    A_s: Optional[np.ndarray] = None
    B_s: Optional[np.ndarray] = None
    C_s: Optional[np.ndarray] = None
    x_center_row: Optional[np.ndarray] = None
    x_center_s: Optional[np.ndarray] = None
    trend: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    _transformer_utm_to_geo: Optional[Transformer] = None
    _ned_origin_utm: Optional[Tuple[float, float]] = None
    uhi_footprint: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = (
        None  # (E, N, valid_mask)
    )
    uhi_footprint_adjusted: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = (
        None  # Adjusted footprint: (E_adj, N_adj, valid_mask)
    )
    # UHI data cache - loaded once, reused by all methods
    _uhi_transect_cache: Optional[object] = None  # Cache the transect object
    _uhi_cube_cache: Optional[object] = None
    _uhi_cube_cache_key: Optional[Tuple] = None  # Track which files are cached
    _uhi_data_corrected_cache: Optional[np.ndarray] = None
    _uhi_mean_cache: Optional[np.ndarray] = None
    _uhi_rgb_cache: Optional[np.ndarray] = None
    _uhi_resampled_cache: Optional[np.ndarray] = None
    _uhi_cache_params: Optional[Dict] = None

    def load(self) -> "MBESDetrender":
        """Load raster data from ``tif_path`` and initialize coordinates.

        This method reads the first band of the raster file at ``tif_path``
        and constructs the coordinate grids ``x_cols`` and ``y_rows`` using
        the raster transform.  The extent of the data is stored, and
        nodata values are replaced with NaN.  If ``coord_system`` is
        ``ned``, the coordinates are transformed into a local NED frame
        relative to ``ned_origin``.  The loaded object is returned to
        facilitate method chaining.
        """
        with rasterio.open(self.tif_path) as src:
            z = src.read(1).astype(np.float64)
            tf = src.transform
            b = src.bounds
        if self.nodata_value is not None:
            z = np.where(z == self.nodata_value, np.nan, z)
        self.valid_mask = np.isfinite(z)
        self.data = z
        self.transform = tf
        self.extent = (float(b.left), float(b.right), float(b.bottom), float(b.top))
        a, _, c, _, e, f = tf.a, tf.b, tf.c, tf.d, tf.e, tf.f
        H, W = z.shape
        cols = np.arange(W)
        rows = np.arange(H)
        # coordinate arrays at pixel centres
        self.x_cols = c + a * (cols + 0.5)
        self.y_rows = f + e * (rows + 0.5)
        self.px_x = abs(a)
        self.px_y = abs(e)

        # Setup NED coordinate transformation if requested
        if self.coord_system.lower() == "ned":
            if self.ned_origin is None:
                raise ValueError(
                    "NED coordinate system requires ned_origin=(lon0, lat0, h0)"
                )
            # Create transformer from UTM to geographic
            self._transformer_utm_to_geo = Transformer.from_crs(
                f"EPSG:{self.epsg_utm}", f"EPSG:{self.epsg_geo}", always_xy=True
            )
            # Get NED origin in UTM coordinates
            lon0, lat0, h0 = self.ned_origin
            transformer_geo_to_utm = Transformer.from_crs(
                f"EPSG:{self.epsg_geo}", f"EPSG:{self.epsg_utm}", always_xy=True
            )
            x0_utm, y0_utm = transformer_geo_to_utm.transform(lon0, lat0)
            self._ned_origin_utm = (x0_utm, y0_utm)
            # NED: North = +Y_UTM - Y0, East = +X_UTM - X0
            self.x_cols = self.x_cols - x0_utm  # Eastings
            self.y_rows = self.y_rows - y0_utm  # Northings
            self.extent = (
                self.extent[0] - x0_utm,
                self.extent[1] - x0_utm,
                self.extent[2] - y0_utm,
                self.extent[3] - y0_utm,
            )

        # Load UHI footprint if parameters are provided
        self._load_uhi_footprint()

        return self

    def _load_uhi_footprint(self) -> None:
        """Load UHI footprint automatically if parameters are provided.

        This method attempts to load the UHI transect data and extract the
        footprint coordinates in the same coordinate system as the MBES data.
        If any required parameter is missing or if loading fails, it silently
        skips the footprint loading (with a warning message).

        The footprint is stored as self.uhi_footprint = (E, N, valid_mask).
        """
        import sys
        import os

        # Check if all parameters are already provided (skip config import if so)
        if (
            self.uhi_transect_folder is not None
            and self.uhi_files is not None
            and self.uhi_track_range is not None
        ):
            # All parameters provided explicitly, no need to import config
            transect_folder = self.uhi_transect_folder
            uhi_files = self.uhi_files
            track_range = self.uhi_track_range
        else:
            # Import config module to get defaults for missing parameters
            # Try multiple import paths for flexibility
            OUTPUT_FOLDER = None
            UHI_FILES = None
            UHI_TRACK_RANGE = None

            try:
                # Try mjosa_code config first (preferred)
                from utils.common import config

                OUTPUT_FOLDER = config.OUTPUT_FOLDER
                UHI_FILES = config.UHI_FILES
                UHI_TRACK_RANGE = config.UHI_TRACK_RANGE
            except ImportError:
                try:
                    # Fall back to gref_pipeline config
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    gref_pipeline_dir = os.path.join(
                        os.path.dirname(os.path.dirname(current_dir)), "gref_pipeline"
                    )
                    if gref_pipeline_dir not in sys.path:
                        sys.path.insert(0, gref_pipeline_dir)
                    from config import OUTPUT_FOLDER, UHI_FILES, UHI_TRACK_RANGE
                except ImportError:
                    warnings.warn(
                        "Could not import config module for UHI footprint defaults. Skipping UHI footprint loading."
                    )
                    return

            # Use provided parameters or fall back to config defaults
            transect_folder = self.uhi_transect_folder or OUTPUT_FOLDER
            uhi_files = self.uhi_files or UHI_FILES
            track_range = self.uhi_track_range or UHI_TRACK_RANGE

        # Check if all required parameters are available
        if not transect_folder or not uhi_files or not track_range:
            # Silently skip - no warning if user didn't provide params
            return

        # Check if NED coordinate system is being used (required for UHI footprint)
        if self.coord_system.lower() != "ned":
            warnings.warn(
                "UHI footprint loading requires coord_system='ned'. "
                "Skipping UHI footprint. Set coord_system='ned' and provide ned_origin."
            )
            return

        if self.ned_origin is None:
            warnings.warn(
                "UHI footprint loading requires ned_origin. Skipping UHI footprint."
            )
            return

        try:
            # Import georef module
            # Try to find gref4hsi/final_act/utils/gref_pipeline from mjosa_code location
            current_dir = os.path.dirname(
                os.path.abspath(__file__)
            )  # mjosa_code/utils/mbes
            mjosa_code_root = os.path.dirname(
                os.path.dirname(current_dir)
            )  # mjosa_code
            gref4hsi_root = os.path.dirname(mjosa_code_root)  # gref4hsi
            georef_dir = os.path.join(gref4hsi_root, "gref4hsi", "final_act", "utils")

            if georef_dir not in sys.path:
                sys.path.insert(0, georef_dir)
            from gref_pipeline import georef
            from gref_pipeline.georef import _ecef_to_ned_arrays
        except ImportError:
            try:
                # Alternative: try direct import (if gref_pipeline is in sys.path)
                from utils.gref_pipeline import georef
                from utils.gref_pipeline.georef import _ecef_to_ned_arrays
            except ImportError:
                warnings.warn(
                    "Could not import georef module. Skipping UHI footprint loading."
                )
                return

        try:
            # Load transect and select UHI files (use cache to avoid duplicate loading)
            if self._uhi_transect_cache is None:
                transect = georef.load_transect(transect_folder)
                self._uhi_transect_cache = transect
            else:
                transect = self._uhi_transect_cache

            # Cache the cube object to avoid calling select_files() multiple times
            cache_key = (transect_folder, tuple(uhi_files))
            if (
                self._uhi_cube_cache is None
                or not hasattr(self, "_uhi_cube_cache_key")
                or self._uhi_cube_cache_key != cache_key
            ):
                cube = transect.select_files(uhi_files)
                self._uhi_cube_cache = cube
                self._uhi_cube_cache_key = cache_key
            else:
                cube = self._uhi_cube_cache

            # Extract track range
            track_start, track_end = track_range
            X_ecef = cube.X_ecef[track_start:track_end, :]
            Y_ecef = cube.Y_ecef[track_start:track_end, :]
            Z_ecef = cube.Z_ecef[track_start:track_end, :]

            # Build valid mask (all coordinates must be finite)
            valid_mask = np.isfinite(X_ecef) & np.isfinite(Y_ecef) & np.isfinite(Z_ecef)

            # Also check RGB if available
            if hasattr(cube, "R") and hasattr(cube, "G") and hasattr(cube, "B"):
                R = cube.R[track_start:track_end, :]
                G = cube.G[track_start:track_end, :]
                B = cube.B[track_start:track_end, :]
                rgb_valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
                valid_mask = valid_mask & rgb_valid

            # Convert to NED coordinates
            lon0, lat0, h0 = self.ned_origin
            N, E, _ = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)

            # Store footprint
            self.uhi_footprint = (E, N, valid_mask)

        except Exception as e:
            warnings.warn(
                f"Failed to load UHI footprint: {e}. Continuing without footprint overlay."
            )
            self.uhi_footprint = None

    def adjust_uhi_alignment(self, dx: float = 0.0, dy: float = 0.0) -> "MBESDetrender":
        """Adjust UHI footprint alignment with translation.

        This method applies a translation offset to the UHI footprint coordinates
        without modifying the original footprint or the underlying hyperspectral
        data. The adjusted footprint is stored separately and can be used in
        plotting methods by setting `use_adjusted=True`.

        Calling this method multiple times replaces the previous adjustment.

        Parameters
        ----------
        dx : float, optional
            East offset in meters (positive = shift east). Default: 0.0
        dy : float, optional
            North offset in meters (positive = shift north). Default: 0.0

        Returns
        -------
        self : MBESDetrender
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If UHI footprint hasn't been loaded yet.

        Examples
        --------
        >>> mb_ned.adjust_uhi_alignment(dx=0.5, dy=-0.3)
        >>> fig = mb_ned.plot_uhi_mbes_comparison(use_adjusted=True)
        """
        if self.uhi_footprint is None:
            raise ValueError(
                "UHI footprint not loaded. Call load() first to load the footprint."
            )

        # Unpack original footprint
        E_orig, N_orig, mask = self.uhi_footprint

        # Apply translation
        E_adjusted = E_orig + dx
        N_adjusted = N_orig + dy

        # Store adjusted footprint
        self.uhi_footprint_adjusted = (E_adjusted, N_adjusted, mask)

        # Clear resampled cache since coordinates changed
        self._uhi_resampled_cache = None

        print(f"✅ UHI footprint adjusted: dx={dx:.3f}m E, dy={dy:.3f}m N")

        return self

    def _ensure_uhi_cube_loaded(
        self, window_size: int = 1000, strength: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and cache UHI cube data (loads once, reuses thereafter).

        This method checks if the UHI cube is already loaded with the same
        parameters. If yes, it returns the cached data. If no, it loads the
        cube, applies illumination correction, and caches everything.

        Parameters
        ----------
        window_size : int
            Window size for illumination correction.
        strength : float
            Strength parameter for illumination correction.

        Returns
        -------
        data_corrected : np.ndarray
            Illumination-corrected cube data (T, S, wavelengths).
        uhi_mean : np.ndarray
            Mean across wavelengths (T, S).
        rgb : np.ndarray
            RGB composite (T, S, 3).
        cube : object
            The loaded cube object (for access to other attributes if needed).
        """
        # Check if already cached with same parameters
        current_params = {
            "window_size": window_size,
            "strength": strength,
            "transect_folder": self.uhi_transect_folder,
            "files": tuple(self.uhi_files) if self.uhi_files else None,
            "track_range": self.uhi_track_range,
        }

        if (
            self._uhi_cube_cache is not None
            and self._uhi_cache_params == current_params
        ):
            # Return cached data
            return (
                self._uhi_data_corrected_cache,
                self._uhi_mean_cache,
                self._uhi_rgb_cache,
                self._uhi_cube_cache,
            )

        # Not cached or different parameters - need to load
        print("🔄 Loading UHI cube data (first time only)...")

        try:
            import sys
            import os

            # Try mjosa_code config first
            try:
                from utils.common import config

                OUTPUT_FOLDER = config.OUTPUT_FOLDER
                UHI_FILES = config.UHI_FILES
                UHI_TRACK_RANGE = config.UHI_TRACK_RANGE
            except ImportError:
                # Fall back to gref_pipeline config
                current_dir = os.path.dirname(os.path.abspath(__file__))
                mjosa_code_root = os.path.dirname(os.path.dirname(current_dir))
                gref4hsi_root = os.path.dirname(mjosa_code_root)
                gref_pipeline_dir = os.path.join(
                    gref4hsi_root, "gref4hsi", "final_act", "utils", "gref_pipeline"
                )
                if gref_pipeline_dir not in sys.path:
                    sys.path.insert(0, gref_pipeline_dir)
                from config import OUTPUT_FOLDER, UHI_FILES, UHI_TRACK_RANGE

            # Try importing georef
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                mjosa_code_root = os.path.dirname(os.path.dirname(current_dir))
                gref4hsi_root = os.path.dirname(mjosa_code_root)
                georef_dir = os.path.join(
                    gref4hsi_root, "gref4hsi", "final_act", "utils"
                )
                if georef_dir not in sys.path:
                    sys.path.insert(0, georef_dir)
                from gref_pipeline import georef
            except ImportError:
                from utils.gref_pipeline import georef
        except ImportError:
            raise ImportError(
                "Could not import required modules (config, georef). "
                "Check your Python path."
            )

        transect_folder = self.uhi_transect_folder or OUTPUT_FOLDER
        uhi_files = self.uhi_files or UHI_FILES
        track_range = self.uhi_track_range or UHI_TRACK_RANGE
        track_start, track_end = track_range

        # Load transect (cache it to avoid "Rebuilding grids..." message)
        if self._uhi_transect_cache is None:
            transect = georef.load_transect(transect_folder)
            self._uhi_transect_cache = transect
        else:
            transect = self._uhi_transect_cache

        # Use cached cube if available (avoids calling select_files() which rebuilds grids)
        cache_key = (transect_folder, tuple(uhi_files))
        if (
            self._uhi_cube_cache is not None
            and hasattr(self, "_uhi_cube_cache_key")
            and self._uhi_cube_cache_key == cache_key
        ):
            cube = self._uhi_cube_cache
        else:
            # Need to load cube for the first time
            cube = transect.select_files(uhi_files)
            self._uhi_cube_cache = cube
            self._uhi_cube_cache_key = cache_key

        # Apply illumination correction (this checks if already applied)
        cube.apply_illumination_correction(window_size=window_size, strength=strength)

        # Extract data
        data_corrected = cube.data_corrected[track_start:track_end, :, :]
        uhi_mean = np.nanmean(data_corrected, axis=2)

        # Get RGB from CORRECTED data (like plot_georef does with use_corrected=True)
        # Extract RGB channels from corrected cube using correct parameter names
        R, G, B = cube._extract_rgb_from_cube(
            cube.data_corrected, Rnm=654.2, Gnm=560.0, Bnm=440.3
        )
        # Extract track range
        R = R[track_start:track_end, :]
        G = G[track_start:track_end, :]
        B = B[track_start:track_end, :]

        # Normalize each channel independently (like plot_georef does)
        for C in (R, G, B):
            m, M = np.nanmin(C), np.nanmax(C)
            if np.isfinite(m) and np.isfinite(M) and M > m:
                C[:] = (C - m) / (M - m)

        rgb = np.stack([R, G, B], axis=-1)
        rgb = np.clip(rgb, 0, 1)  # Cache everything (including cube object to reuse)
        self._uhi_cube_cache = cube
        self._uhi_data_corrected_cache = data_corrected
        self._uhi_mean_cache = uhi_mean
        self._uhi_rgb_cache = rgb
        self._uhi_cache_params = current_params

        print("✅ UHI cube cached! Subsequent calls will reuse this data.")

        return data_corrected, uhi_mean, rgb, cube

    def detrend(
        self,
        order_x: int = 2,
        remove_cross_track_tilt: bool = True,
        robust: bool = True,
        smooth_baseline_m: Optional[float] = None,
        smooth_tilt_m: Optional[float] = None,
        smooth_curv_m: Optional[float] = None,
        smooth_center_m: Optional[float] = None,
        central_frac: Optional[float] = None,
    ) -> "MBESDetrender":
        """Fit and subtract a per‑row polynomial trend from the data.

        The detrending proceeds by fitting a polynomial of order 0, 1 or 2 to
        each row, centred on the row median across the central swath portion.
        The resulting polynomial coefficients are then smoothed along the
        along‑track dimension using NaN‑aware Savitzky–Golay filters (or
        Gaussian fallback).  The smoothed trend is subtracted from the
        original data to yield residuals.

        Parameters
        ----------
        order_x: int, optional
            Polynomial order to fit per row (0, 1 or 2).  Defaults to 2.
        remove_cross_track_tilt: bool, optional
            If True, subtract the cross‑track slope from the baseline trend.
        robust: bool, optional
            If True, perform a few iterations of robust fitting to mitigate
            outliers.  Defaults to True.
        smooth_baseline_m, smooth_tilt_m, smooth_curv_m, smooth_center_m: float, optional
            Override the default smoothing lengths for baseline, tilt,
            curvature and swath centre.  If ``smooth_curv_m`` is None,
            ``smooth_tilt_m`` is used.
        central_frac: float, optional
            Fraction of beams around the centre of each row used for fitting.

        Returns
        -------
        MBESDetrender
            The instance itself, for method chaining.
        """
        if self.data is None:
            self.load()
        # override smoothing parameters if provided
        if smooth_baseline_m is not None:
            self.smooth_baseline_m = float(smooth_baseline_m)
        if smooth_tilt_m is not None:
            self.smooth_tilt_m = float(smooth_tilt_m)
        if smooth_curv_m is not None:
            self.smooth_curv_m = float(smooth_curv_m)
        if smooth_center_m is not None:
            self.smooth_center_m = float(smooth_center_m)
        if central_frac is not None:
            self.central_frac = float(central_frac)
        if self.smooth_curv_m is None:
            self.smooth_curv_m = self.smooth_tilt_m

        Z = self.data
        x = self.x_cols
        H, W = Z.shape
        A = np.full(H, np.nan)
        B = np.full(H, 0.0)
        C = np.full(H, 0.0)
        x_center = np.full(H, np.nan)
        # per‑row robust polynomial around each row's centre
        for i in range(H):
            row = Z[i, :]
            m = np.isfinite(row)
            if m.sum() < 6:
                continue
            xv = x[m]
            yv = row[m]
            xc_row = np.median(xv)
            x_center[i] = xc_row
            if 0 < self.central_frac < 1:
                lo = np.quantile(xv, 0.5 - 0.5 * self.central_frac)
                hi = np.quantile(xv, 0.5 + 0.5 * self.central_frac)
                keep = (xv >= lo) & (xv <= hi)
                xv, yv = xv[keep], yv[keep]
            xi = xv - xc_row
            deg = (
                2
                if (order_x >= 2 and remove_cross_track_tilt)
                else (1 if remove_cross_track_tilt else 0)
            )
            coef = np.polyfit(xi, yv, deg)
            if deg == 0:
                a0, a1, a2 = float(coef[0]), 0.0, 0.0
            elif deg == 1:
                a1, a0 = float(coef[0]), float(coef[1])
                a2 = 0.0
            else:
                a2, a1, a0 = float(coef[0]), float(coef[1]), float(coef[2])
            if robust and deg > 0:
                # refine with iteratively reweighted least squares
                for _ in range(self.max_row_iters):
                    pred = a0 + a1 * xi + a2 * (xi**2)
                    resid = yv - pred
                    s = median_abs_deviation(resid, scale="normal")
                    if not np.isfinite(s) or s == 0:
                        break
                    w = 1.0 / (1.0 + (resid / (3 * s)) ** 2)
                    X = np.vstack([np.ones_like(xi), xi])
                    if deg == 2:
                        X = np.vstack([X, xi**2])
                    XtW = X * w
                    beta = np.linalg.lstsq(XtW.T, (yv * w), rcond=None)[0]
                    if deg == 2:
                        a0, a1, a2 = float(beta[0]), float(beta[1]), float(beta[2])
                    else:
                        a0, a1 = float(beta[0]), float(beta[1])
            A[i] = a0
            B[i] = a1 if remove_cross_track_tilt else 0.0
            C[i] = a2 if (order_x >= 2 and remove_cross_track_tilt) else 0.0

        # clip crazy rows and fill short gaps
        def _clip_and_fill(arr: np.ndarray) -> np.ndarray:
            s = arr.copy()
            m = np.isfinite(s)
            if m.sum() < 5:
                return s
            med = np.nanmedian(s[m])
            mad = median_abs_deviation(s[m], scale="normal") or np.nanstd(s[m])
            if np.isfinite(mad) and mad > 0:
                lo, hi = med - self.clip_sigma * mad, med + self.clip_sigma * mad
                s[(s < lo) | (s > hi)] = np.nan
            idx = np.arange(len(s))
            good = np.isfinite(s)
            if good.sum() >= 2:
                s[~good] = np.interp(idx[~good], idx[good], s[good])
            return s

        A = _clip_and_fill(A)
        B = _clip_and_fill(B)
        C = _clip_and_fill(C)
        x_center = _clip_and_fill(x_center)
        # along‑track smoothing
        A_s = _savgol_nanaware(A, self.px_y, self.smooth_baseline_m, poly=2)
        B_s = _savgol_nanaware(B, self.px_y, self.smooth_tilt_m, poly=2)
        C_s = _savgol_nanaware(C, self.px_y, self.smooth_curv_m, poly=2)
        Xc = _savgol_nanaware(x_center, self.px_y, self.smooth_center_m, poly=2)
        # reconstruct trend & residuals
        xi_grid = self.x_cols[None, :] - Xc[:, None]
        trend = A_s[:, None] + B_s[:, None] * xi_grid + C_s[:, None] * (xi_grid**2)
        if self.valid_mask is not None:
            trend = np.where(self.valid_mask, trend, np.nan)
        resid = self.data - trend
        resid[~np.isfinite(self.data)] = np.nan
        # store results
        self.A, self.B, self.C = A, B, C
        self.A_s, self.B_s, self.C_s = A_s, B_s, C_s
        self.x_center_row, self.x_center_s = x_center, Xc
        self.trend, self.residuals = trend, resid
        return self

    # --- plotting ---

    def _extent(self) -> Tuple[float, float, float, float]:
        return (
            self.extent[0],
            self.extent[1],
            self.extent[2],
            self.extent[3],
        )

    def _data_window(
        self, percentile: float = 99.5
    ) -> Tuple[float, float, float, float]:
        rows, cols = np.where(np.isfinite(self.data))
        if rows.size == 0:
            return self._extent()
        p_lo = (100 - percentile) / 2
        p_hi = 100 - p_lo
        c0 = int(np.percentile(cols, p_lo))
        c1 = int(np.percentile(cols, p_hi))
        r0 = int(np.percentile(rows, p_lo))
        r1 = int(np.percentile(rows, p_hi))
        x_min, x_max = self.x_cols[c0], self.x_cols[c1]
        y_min, y_max = self.y_rows[r1], self.y_rows[r0]
        return (x_min, x_max, y_min, y_max)

    def plot_triptych(
        self,
        cmap_orig: str = "terrain",
        cmap_resid: str = "RdBu_r",
        figsize: Tuple[int, int] = (9, 14),
        crop_to_data: bool = True,
        bounds_percentile: float = 99.5,
        margin_frac: float = 0.03,
        equal_aspect: bool = True,
        show: bool = True,
        *,
        footprint_style: str = "outline",
        uhi_footprint: Optional[
            Union[MplPath, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
        use_adjusted: bool = False,
    ) -> plt.Figure:
        """Plot original, trend, and residual panels with optional footprint overlay.

        This method produces a figure with three columns: the original raster,
        the estimated trend, and the residual (detrended) bathymetry.  The UHI
        footprint overlay can be rendered in different styles (outline, fill, or both).

        The footprint is automatically loaded during load() if UHI parameters are
        provided. You can also override it by passing uhi_footprint explicitly.

        Parameters
        ----------
        cmap_orig: str, optional
            Colormap used for the original and trend panels.  Defaults to
            ``"terrain"``.
        cmap_resid: str, optional
            Colormap used for the residual panel.  Defaults to ``"RdBu_r"``.
        figsize: tuple, optional
            Size of the figure in inches.  Defaults to (9, 14).
        crop_to_data: bool, optional
            If True, crop axes to the extents of the finite data (within
            ``bounds_percentile``).  Otherwise use the full raster extent.
        bounds_percentile: float, optional
            Percentile used when cropping to data.  Defaults to 99.5.
        margin_frac: float, optional
            Fraction of the data range to pad the axes by.  Defaults to 0.03.
        equal_aspect: bool, optional
            If True, enforce equal aspect ratio on all panels.  Defaults to
            True.
        show: bool, optional
            If True, display the figure immediately.  Defaults to True.
        footprint_style: str, optional
            Style for rendering the UHI footprint overlay. Options:
            - 'outline': Black outline only (no fill) - default
            - 'fill': Filled black region (no outline)
            - 'both': Black fill with black outline
        uhi_footprint: Path or (E, N, mask), optional
            Optional override for the UHI footprint. If not provided, uses
            the footprint loaded during load(). Can be a matplotlib Path or
            a tuple (E, N, mask).
        use_adjusted: bool, optional
            If True, use the adjusted UHI footprint (after alignment shifts).
            If False, use the original footprint. Defaults to False.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the triptych.
        """
        assert self.trend is not None, "Run detrend() first."
        # axis labels based on coordinate system
        if self.coord_system.lower() == "ned":
            xlabel, ylabel = "East (m)", "North (m)"
        else:
            xlabel, ylabel = "Easting (m)", "Northing (m)"
        fig, axs = plt.subplots(
            1,
            3,
            figsize=figsize,
            constrained_layout=True,
            num="MBES Triptych",
        )
        # Original panel
        im0 = axs[0].imshow(
            np.ma.masked_invalid(self.data),
            extent=self._extent(),
            origin="upper",
            cmap=cmap_orig,
        )
        axs[0].set_title("Original")
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        fig.colorbar(im0, ax=axs[0], label="Depth [m]", fraction=0.025, pad=0.02)
        # Trend panel
        im1 = axs[1].imshow(
            np.ma.masked_invalid(self.trend),
            extent=self._extent(),
            origin="upper",
            cmap=cmap_orig,
        )
        ord_label = (
            2 if (self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0.001) else 1
        )
        axs[1].set_title(
            f"Estimated trend\norder={ord_label}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m",
            fontsize=10,
        )
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        fig.colorbar(im1, ax=axs[1], label="Depth [m]", fraction=0.025, pad=0.02)
        # Residual panel
        vmin, vmax = _robust_sym_vlim(self.residuals, q=0.98)
        im2 = axs[2].imshow(
            np.ma.masked_invalid(self.residuals),
            extent=self._extent(),
            origin="upper",
            cmap=cmap_resid,
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )
        axs[2].set_title("Residuals (local variations)")
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel(ylabel)
        fig.colorbar(im2, ax=axs[2], label="m", fraction=0.025, pad=0.02)
        # Determine cropping extents
        if crop_to_data:
            x_min, x_max, y_min, y_max = self._data_window(bounds_percentile)
        else:
            x_min, x_max, y_min, y_max = self._extent()

        # Overlay the UHI footprint on the residual panel
        # Use provided footprint or fall back to stored footprint (adjusted or original)
        if uhi_footprint is not None:
            footprint_to_use = uhi_footprint
        elif use_adjusted and self.uhi_footprint_adjusted is not None:
            footprint_to_use = self.uhi_footprint_adjusted
        else:
            footprint_to_use = self.uhi_footprint

        if footprint_to_use is not None:
            # Handle both Path and (E, N, mask) tuple formats
            if isinstance(footprint_to_use, MplPath):
                fp_path = footprint_to_use
            elif (
                isinstance(footprint_to_use, tuple)
                and len(footprint_to_use) == 3
                and isinstance(footprint_to_use[0], np.ndarray)
                and isinstance(footprint_to_use[1], np.ndarray)
                and isinstance(footprint_to_use[2], np.ndarray)
            ):
                E, N, mask = footprint_to_use
                # Build pixel-perfect boundary path using helper functions
                Xc, Yc = pcolormesh_pad(E, N)
                segs_xy, segs_idx = _boundary_segments_from_mask(mask, Xc, Yc)

                if len(segs_idx) > 0:
                    loops = _trace_loops_from_segments(segs_idx)
                    fp_path = _compound_path_from_loops(loops, Xc, Yc)
                else:
                    fp_path = None
            else:
                raise TypeError(
                    "uhi_footprint must be a matplotlib.path.Path or (E, N, mask) tuple"
                )

            # Render footprint according to style
            if fp_path is not None:
                if footprint_style == "outline":
                    # Black outline only (no fill)
                    patch = PathPatch(
                        fp_path,
                        transform=axs[2].transData,
                        facecolor="none",
                        edgecolor="black",
                        linewidth=2.0,
                        zorder=10,
                        alpha=1.0,
                    )
                    axs[2].add_patch(patch)
                elif footprint_style == "fill":
                    # Filled black region (no outline)
                    patch = PathPatch(
                        fp_path,
                        transform=axs[2].transData,
                        facecolor="black",
                        edgecolor="none",
                        zorder=10,
                        alpha=1.0,
                    )
                    axs[2].add_patch(patch)
                elif footprint_style == "both":
                    # Black fill with black outline
                    patch = PathPatch(
                        fp_path,
                        transform=axs[2].transData,
                        facecolor="black",
                        edgecolor="black",
                        linewidth=2.0,
                        zorder=10,
                        alpha=1.0,
                    )
                    axs[2].add_patch(patch)
                else:
                    warnings.warn(
                        f"Unknown footprint_style='{footprint_style}'. "
                        "Valid options: 'outline', 'fill', 'both'. Using 'outline'."
                    )
                    # Default to outline
                    patch = PathPatch(
                        fp_path,
                        transform=axs[2].transData,
                        facecolor="none",
                        edgecolor="black",
                        linewidth=2.0,
                        zorder=10,
                        alpha=1.0,
                    )
                    axs[2].add_patch(patch)
        elif uhi_footprint is None and self.uhi_footprint is None:
            # No footprint available - show plots without overlay (silent, no error)
            pass
        # apply padded limits on all panels
        for ax in axs:
            _pad_limits(
                ax,
                x_min,
                x_max,
                y_min,
                y_max,
                pad_frac=margin_frac,
                equal_aspect=equal_aspect,
            )
        if show:
            plt.show(block=False)
        return fig

    def plot_residual_heatmap(
        self,
        figsize: Tuple[int, int] = (6, 14),
        cmap: str = "RdBu_r",
        vmax: Optional[float] = None,
        crop_to_data: bool = True,
        bounds_percentile: float = 99.5,
        margin_frac: float = 0.03,
        equal_aspect: bool = True,
        title_suffix: str = "",
        show: bool = True,
    ) -> plt.Figure:
        """Plot only the residual heatmap.

        This method plots the residuals computed by ``detrend`` as a single
        heatmap.  Cropping and aspect ratio options mirror those of
        ``plot_triptych``.

        Parameters and return value match the original implementation.  See
        ``plot_triptych`` for more details on arguments.
        """
        assert self.residuals is not None, "Run detrend() first."
        if self.coord_system.lower() == "ned":
            xlabel, ylabel = "East (m)", "North (m)"
        else:
            xlabel, ylabel = "Easting (m)", "Northing (m)"
        if vmax is None:
            vmin, vmax = _robust_sym_vlim(self.residuals, q=0.98)
        else:
            vmin, vmax = -abs(vmax), abs(vmax)
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            constrained_layout=True,
            num="MBES Residuals",
        )
        im = ax.imshow(
            np.ma.masked_invalid(self.residuals),
            extent=self._extent(),
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if crop_to_data:
            x_min, x_max, y_min, y_max = self._data_window(bounds_percentile)
        else:
            x_min, x_max, y_min, y_max = self._extent()
        _pad_limits(
            ax,
            x_min,
            x_max,
            y_min,
            y_max,
            pad_frac=margin_frac,
            equal_aspect=equal_aspect,
        )
        ord_label = (
            2 if (self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0.001) else 1
        )
        settings = f"order={ord_label}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m, center={self.smooth_center_m:.2f}m"
        ax.set_title(
            f"Detrended bathymetry (local variations)\n{settings}{(' — ' + title_suffix) if title_suffix else ''}",
            fontsize=10,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, label="m", fraction=0.035, pad=0.02)
        if show:
            plt.show(block=False)
        return fig

    def plot_components(
        self,
        figsize: Tuple[int, int] = (10, 3),
        bounds_percentile: float = 99.5,
        margin_frac: float = 0.03,
        show: bool = True,
    ) -> plt.Figure:
        """Plot the individual components of the trend and swath centre.

        This function plots the raw and smoothed baseline A(y), tilt B(y),
        curvature C(y) (if present), and swath centre x̄(y) across the
        along‑track dimension.  The y-axis corresponds to the y coordinate
        (northing in UTM or NED), inverted to match map convention.
        """
        assert self.A_s is not None, "Run detrend() first."
        if self.coord_system.lower() == "ned":
            xlabel_e, ylabel_n = "East (m)", "North (m)"
        else:
            xlabel_e, ylabel_n = "Easting (m)", "Northing (m)"
        y = self.y_rows
        fig, ax = plt.subplots(
            2,
            2,
            figsize=figsize,
            constrained_layout=True,
            num="MBES Components",
        )
        # baseline A(y)
        ax[0, 0].plot(self.A, y, alpha=0.25, label="row fit")
        ax[0, 0].plot(self.A_s, y, lw=2, label="smoothed")
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_title(f"A(y) baseline — σ={self.smooth_baseline_m:.2f}m")
        ax[0, 0].set_xlabel("Depth (m)")
        ax[0, 0].set_ylabel(ylabel_n)
        ax[0, 0].legend()
        # tilt B(y)
        ax[0, 1].plot(self.B, y, alpha=0.25, label="row fit")
        ax[0, 1].plot(self.B_s, y, lw=2, label="smoothed")
        ax[0, 1].invert_yaxis()
        ax[0, 1].set_title(f"B(y) tilt — σ={self.smooth_tilt_m:.2f}m")
        ax[0, 1].set_xlabel("Slope (m/m)")
        ax[0, 1].set_ylabel(ylabel_n)
        ax[0, 1].legend()
        # curvature C(y)
        if self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0:
            ax[1, 0].plot(self.C, y, alpha=0.25, label="row fit")
            ax[1, 0].plot(self.C_s, y, lw=2, label="smoothed")
            ax[1, 0].invert_yaxis()
            ax[1, 0].set_title(f"C(y) curvature — σ={self.smooth_curv_m:.2f}m")
            ax[1, 0].set_xlabel("1/m")
            ax[1, 0].set_ylabel(ylabel_n)
            ax[1, 0].legend()
        else:
            ax[1, 0].axis("off")
        # swath centre x̄(y)
        ax[1, 1].plot(self.x_center_row, y, alpha=0.25, label="row median x")
        ax[1, 1].plot(self.x_center_s, y, lw=2, label="smoothed")
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_title(
            f"Swath centre x̄(y) — σ={self.smooth_center_m:.2f}m, central={self.central_frac*100:.0f}%"
        )
        ax[1, 1].set_xlabel(xlabel_e)
        ax[1, 1].set_ylabel(ylabel_n)
        ax[1, 1].legend()
        # keep y range similar to map panels
        x_min, x_max, y_min, y_max = self._data_window(bounds_percentile)
        for a in ax.flat:
            a.set_ylim(
                y_min - (y_max - y_min) * margin_frac,
                y_max + (y_max - y_min) * margin_frac,
            )
        if show:
            plt.show(block=False)
        return fig

    # --- convenience ---
    def plot_all(
        self, *, show: bool = True, include_uhi_comparison: bool = True
    ) -> Tuple[plt.Figure, ...]:
        """Plot all available visualizations.

        By default, generates five figures:
        1. Triptych (data, trend, residuals)
        2. Components (trend parameters)
        3. Residual heatmap
        4. UHI-MBES comparison (three-panel: RGB | residuals | Δz) - if UHI data available
        5. Density scatter (UHI vs MBES correlation) - if UHI data available

        If UHI footprint is not available or include_uhi_comparison=False, returns
        only the first three figures.

        Parameters
        ----------
        show : bool, optional
            Whether to display all plots, by default True.
        include_uhi_comparison : bool, optional
            Whether to include UHI comparison plots (requires UHI footprint data),
            by default True.

        Returns
        -------
        tuple of matplotlib.figure.Figure
            Tuple of generated figures. Either (f1, f2, f3) or (f1, f2, f3, f4, f5)
            depending on UHI data availability.
        """
        f1 = self.plot_triptych(show=False)
        f2 = self.plot_components(show=False)
        f3 = self.plot_residual_heatmap(show=False)

        # Try to generate UHI comparison plots if requested and data is available
        figures = [f1, f2, f3]
        if (
            include_uhi_comparison
            and self.uhi_footprint is not None
            and self.coord_system.lower() == "ned"
        ):
            try:
                f4 = self.plot_uhi_mbes_comparison(show=False)
                f5 = self.plot_density_scatter(show=False)
                figures.extend([f4, f5])
            except Exception as e:
                warnings.warn(
                    f"Could not generate UHI comparison plots: {e}. "
                    "Returning basic plots only."
                )

        if show:
            plt.show(block=False)
        return tuple(figures)

    def plot_zoomed_residuals(
        self,
        outline_color: str = "black",
        outline_width: float = 1.2,
        outline_alpha: float = 0.9,
        cmap: str = "RdBu_r",
        figsize: Tuple[int, int] = (11, 9),  # Match plot_georef default
        vmin: Optional[float] = None,  # Manual color scale min
        vmax: Optional[float] = None,  # Manual color scale max
        use_adjusted: bool = False,  # Use alignment-adjusted UHI coordinates
        show: bool = True,
    ) -> plt.Figure:
        """Plot MBES residuals zoomed to UHI footprint with pixel-perfect outline.

        This method creates a separate figure showing only the MBES residuals
        within the UHI footprint region. The residuals are clipped to show only
        the area inside the footprint, and a pixel-perfect outline is drawn
        around the footprint boundary.

        Requirements:
        - Must use coord_system='ned' (raises error otherwise)
        - Must have loaded UHI footprint (self.uhi_footprint must exist)
        - Must have run detrend() first

        Parameters
        ----------
        outline_color : str, optional
            Color of the footprint outline. Defaults to 'black'.
        outline_width : float, optional
            Width of the outline in points. Defaults to 1.2.
        outline_alpha : float, optional
            Opacity of the outline (0=transparent, 1=opaque). Defaults to 0.9.
        cmap : str, optional
            Colormap for the residuals. Defaults to 'RdBu_r' (red-blue reversed).
        figsize : tuple, optional
            Figure size in inches (width, height). Defaults to (9, 11).
        show : bool, optional
            If True, display the figure immediately. Defaults to True.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.

        Raises
        ------
        ValueError
            If coord_system is not 'ned' or if UHI footprint is not loaded.
        AssertionError
            If detrend() has not been run yet.
        """
        # Validate requirements
        assert self.trend is not None, "Run detrend() first."

        if self.coord_system.lower() != "ned":
            raise ValueError(
                "plot_zoomed_residuals() requires coord_system='ned'. "
                f"Current coord_system='{self.coord_system}'"
            )

        if self.uhi_footprint is None:
            raise ValueError(
                "plot_zoomed_residuals() requires UHI footprint data. "
                "Ensure UHI footprint was loaded during .load() or provide manually."
            )

        if self.ned_origin is None:
            raise ValueError("plot_zoomed_residuals() requires ned_origin to be set.")

        # Extract UHI footprint (use adjusted if requested)
        if use_adjusted and self.uhi_footprint_adjusted is not None:
            E_uhi, N_uhi, mask_uhi = self.uhi_footprint_adjusted
        else:
            E_uhi, N_uhi, mask_uhi = self.uhi_footprint

        # MBES is already in NED coordinates (x_cols = East, y_rows = North)
        # No coordinate transformation needed since load() already converted to NED
        mbes_e_ned = self.x_cols
        mbes_n_ned = self.y_rows

        # Build pixel-perfect UHI footprint outline using helper functions
        Xc, Yc = pcolormesh_pad(E_uhi, N_uhi)
        segs_xy, segs_idx = _boundary_segments_from_mask(mask_uhi, Xc, Yc)

        if len(segs_idx) == 0:
            warnings.warn("No valid UHI footprint boundary found. Creating empty plot.")
            fig, ax = plt.subplots(1, 1, figsize=figsize, num="MBES Residuals (Zoomed)")
            ax.text(0.5, 0.5, "No valid UHI footprint", ha="center", va="center")
            return fig

        loops = _trace_loops_from_segments(segs_idx)
        comp_path = _compound_path_from_loops(loops, Xc, Yc)

        # Create figure
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, num="MBES Residuals within UHI Footprint (NED)"
        )

        # Determine color limits for residuals
        if vmin is None or vmax is None:
            # Auto-calculate symmetric limits
            auto_vmin, auto_vmax = _robust_sym_vlim(self.residuals, q=0.98)
            vmin = vmin if vmin is not None else auto_vmin
            vmax = vmax if vmax is not None else auto_vmax

        # Create MBES meshgrid for extent calculation
        extent_ned = [
            np.min(mbes_e_ned),
            np.max(mbes_e_ned),
            np.min(mbes_n_ned),
            np.max(mbes_n_ned),
        ]

        # Plot full-resolution MBES residuals
        im = ax.imshow(
            np.ma.masked_invalid(self.residuals),
            extent=extent_ned,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )

        # Clip MBES image to UHI footprint using PathPatch
        patch = PathPatch(comp_path, transform=ax.transData, facecolor="none")
        im.set_clip_path(patch)

        # Draw the pixel-perfect outline (jagged grid edges)
        if len(segs_xy) > 0:
            lc = LineCollection(
                segs_xy,
                colors=outline_color,
                linewidths=outline_width,
                alpha=outline_alpha,
                zorder=10,
            )
            ax.add_collection(lc)

        # Zoom to footprint bounding box WITHOUT padding (to match plot_georef behavior)
        xs_outline = comp_path.vertices[:, 0]
        ys_outline = comp_path.vertices[:, 1]
        x_min, x_max = np.nanmin(xs_outline), np.nanmax(xs_outline)
        y_min, y_max = np.nanmin(ys_outline), np.nanmax(ys_outline)

        # Set tight limits without padding (like plot_georef)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        # Labels and title
        ax.set_title(
            "MBES Residuals within UHI Footprint (NED)\n"
            "Pixel-perfect outline, full MBES resolution"
        )
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, label="Residuals [m]")

        fig.tight_layout()

        if show:
            plt.show(block=False)

        return fig

    def plot_colorbar_only(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "RdBu_r",
        label: str = "MBES Residuals [m]",
        figsize: Tuple[float, float] = (2, 6),
        show: bool = True,
    ) -> plt.Figure:
        """Plot standalone colorbar for MBES residuals.

        Creates a separate figure with just the colorbar, useful for combining
        with other plots or creating legends.

        Parameters
        ----------
        vmin : float, optional
            Minimum value for colorbar. If None, auto-calculated.
        vmax : float, optional
            Maximum value for colorbar. If None, auto-calculated.
        cmap : str, optional
            Colormap name. Defaults to 'RdBu_r'.
        label : str, optional
            Colorbar label. Defaults to 'MBES Residuals [m]'.
        figsize : tuple, optional
            Figure size (width, height). Defaults to (2, 6).
        show : bool, optional
            If True, display immediately. Defaults to True.

        Returns
        -------
        matplotlib.figure.Figure
            The colorbar figure.
        """
        assert self.trend is not None, "Run detrend() first."

        # Determine color limits
        if vmin is None or vmax is None:
            auto_vmin, auto_vmax = _robust_sym_vlim(self.residuals, q=0.98)
            vmin = vmin if vmin is not None else auto_vmin
            vmax = vmax if vmax is not None else auto_vmax

        # Create figure with just colorbar
        fig, ax = plt.subplots(figsize=figsize, num="MBES Residuals Colorbar")

        # Create a dummy mappable for the colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Create colorbar
        cbar = fig.colorbar(sm, cax=ax, label=label)

        fig.tight_layout()

        if show:
            plt.show(block=False)

        return fig

    def plot_z_normalized_comparison(
        self,
        outline_color: str = "black",
        outline_width: float = 1.2,
        outline_alpha: float = 0.9,
        figsize: Tuple[int, int] = (11, 9),
        use_adjusted: bool = False,
        flip_uhi_sign: bool = False,
        normalization_method: str = "zscore",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot z-score normalized comparison of MBES and UHI data.

        Creates a single heatmap showing the difference between MBES residuals
        and UHI depth data after normalizing both to z-scores (or minmax).
        This is equivalent to panel 4 from plot_uhi_mbes_comparison().

        Parameters
        ----------
        outline_color : str, optional
            Color of footprint outline. Defaults to 'black'.
        outline_width : float, optional
            Width of outline. Defaults to 1.2.
        outline_alpha : float, optional
            Opacity of outline. Defaults to 0.9.
        figsize : tuple, optional
            Figure size (width, height). Defaults to (11, 9) to match plot_georef.
        use_adjusted : bool, optional
            Use alignment-adjusted UHI coordinates. Defaults to False.
        flip_uhi_sign : bool, optional
            Flip sign of UHI data (depth → negative depth). Defaults to False.
        normalization_method : str, optional
            'zscore' or 'minmax'. Defaults to 'zscore'.
        vmin : float, optional
            Minimum color scale value. If None, uses robust percentile.
        vmax : float, optional
            Maximum color scale value. If None, uses robust percentile.
        show : bool, optional
            Display immediately. Defaults to True.

        Returns
        -------
        matplotlib.figure.Figure
            The comparison figure.
        """
        # Validation
        if self.coord_system.lower() != "ned":
            raise ValueError(
                "plot_z_normalized_comparison() requires coord_system='ned'. "
                f"Current coord_system='{self.coord_system}'"
            )

        if self.uhi_footprint is None:
            raise ValueError("UHI footprint required.")

        if self.residuals is None:
            raise ValueError("Residuals not computed. Call .detrend() first.")

        if self.ned_origin is None:
            raise ValueError("ned_origin required for NED coordinate system.")

        # Select footprint (original or adjusted)
        using_adjusted = use_adjusted and self.uhi_footprint_adjusted is not None
        if using_adjusted:
            E_uhi, N_uhi, mask_valid = self.uhi_footprint_adjusted
        else:
            E_uhi, N_uhi, mask_valid = self.uhi_footprint

        # Load UHI cube (cached after first call)
        _, uhi_mean, _, _ = self._ensure_uhi_cube_loaded(window_size=1000, strength=1.0)

        # Build pixel-perfect footprint path
        Xc_uhi, Yc_uhi = pcolormesh_pad(E_uhi, N_uhi)
        segs_xy, segs_idx = _boundary_segments_from_mask(mask_valid, Xc_uhi, Yc_uhi)

        if len(segs_idx) == 0:
            warnings.warn("No valid UHI footprint boundary.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No valid footprint", ha="center", va="center")
            return fig

        loops = _trace_loops_from_segments(segs_idx)
        footprint_path = _compound_path_from_loops(loops, Xc_uhi, Yc_uhi)
        xs_outline = footprint_path.vertices[:, 0]
        ys_outline = footprint_path.vertices[:, 1]
        x_min, x_max = np.nanmin(xs_outline), np.nanmax(xs_outline)
        y_min, y_max = np.nanmin(ys_outline), np.nanmax(ys_outline)

        # MBES grid in NED
        from scipy.spatial import cKDTree

        mbes_e_ned = self.x_cols
        mbes_n_ned = self.y_rows
        extent_ned = [
            np.min(mbes_e_ned),
            np.max(mbes_e_ned),
            np.min(mbes_n_ned),
            np.max(mbes_n_ned),
        ]

        # Create MBES grid for inside check
        MBES_E, MBES_N = np.meshgrid(mbes_e_ned, mbes_n_ned)

        # Determine which MBES pixels are inside UHI footprint
        pts_mbes = np.column_stack([MBES_E.ravel(), MBES_N.ravel()])
        inside = footprint_path.contains_points(pts_mbes).reshape(MBES_E.shape)

        # Resample UHI mean to MBES grid (reuse cache if available)
        if self._uhi_resampled_cache is not None and not using_adjusted:
            uhi_on_mbes = self._uhi_resampled_cache
        else:
            valid_uhi = (
                mask_valid
                & np.isfinite(uhi_mean)
                & np.isfinite(E_uhi)
                & np.isfinite(N_uhi)
            )
            uhi_on_mbes = np.full(self.residuals.shape, np.nan, dtype=float)
            if np.any(valid_uhi):
                uhi_pts = np.column_stack([E_uhi[valid_uhi], N_uhi[valid_uhi]])
                uhi_vals = uhi_mean[valid_uhi]
                tree = cKDTree(uhi_pts)
                idx_inside = np.where(inside.ravel())[0]
                d, nn = tree.query(pts_mbes[idx_inside], k=1)
                flat = uhi_on_mbes.ravel()
                flat[idx_inside] = uhi_vals[nn]
                uhi_on_mbes = flat.reshape(MBES_E.shape)
            if not using_adjusted:
                # Cache only if not adjusted
                self._uhi_resampled_cache = uhi_on_mbes

        # Compute normalized values for comparison
        mbes_in = np.where(inside, self.residuals, np.nan)
        uhi_in = np.where(inside, uhi_on_mbes, np.nan)

        # Optionally flip UHI sign
        if flip_uhi_sign:
            uhi_in = -uhi_in

        # Extract valid paired samples BEFORE normalization
        valid_pairs_mask = np.isfinite(mbes_in) & np.isfinite(uhi_in)
        mbes_valid = mbes_in[valid_pairs_mask]
        uhi_valid = uhi_in[valid_pairs_mask]

        # Normalize each dataset INDEPENDENTLY
        if normalization_method.lower() == "minmax":
            # Normalize to [0, 1] then center at 0
            mbes_min, mbes_max = np.nanmin(mbes_valid), np.nanmax(mbes_valid)
            uhi_min, uhi_max = np.nanmin(uhi_valid), np.nanmax(uhi_valid)
            mbes_norm = (mbes_valid - mbes_min) / (mbes_max - mbes_min) - 0.5
            uhi_norm = (uhi_valid - uhi_min) / (uhi_max - uhi_min) - 0.5
            norm_label = "normalized [0-1]"
        elif normalization_method.lower() == "zscore":
            # Robust z-score: (x - median) / MAD
            from scipy.stats import median_abs_deviation

            mbes_med = np.nanmedian(mbes_valid)
            uhi_med = np.nanmedian(uhi_valid)
            mbes_mad = median_abs_deviation(mbes_valid, nan_policy="omit")
            uhi_mad = median_abs_deviation(uhi_valid, nan_policy="omit")
            mbes_norm = (mbes_valid - mbes_med) / mbes_mad
            uhi_norm = (uhi_valid - uhi_med) / uhi_mad
            norm_label = "z-units"
        else:
            raise ValueError(
                f"Invalid normalization_method: {normalization_method}. "
                f"Choose 'zscore' or 'minmax'."
            )

        # Reconstruct grids with normalized values
        z_mbes = np.full_like(mbes_in, np.nan)
        z_uhi = np.full_like(uhi_in, np.nan)
        z_mbes[valid_pairs_mask] = mbes_norm
        z_uhi[valid_pairs_mask] = uhi_norm

        # Δ = MBES - UHI
        diff_z = z_mbes - z_uhi

        # Determine color limits
        finite_diff = np.isfinite(diff_z)

        if vmin is None or vmax is None:
            if normalization_method.lower() == "minmax":
                # Use symmetric limits for minmax
                if np.any(finite_diff):
                    diff_data = diff_z[finite_diff]
                    p2 = float(np.nanpercentile(diff_data, 2.0))
                    p98 = float(np.nanpercentile(diff_data, 98.0))
                    vlim = max(abs(p2), abs(p98))
                    vmin_auto, vmax_auto = -vlim, vlim
                else:
                    vmin_auto, vmax_auto = -1.0, 1.0
            else:
                # Zscore: robust symmetric
                vmax_auto = (
                    np.nanpercentile(np.abs(diff_z[finite_diff]), 98.0)
                    if np.any(finite_diff)
                    else 1.0
                )
                vmin_auto = -vmax_auto

            if vmin is None:
                vmin = vmin_auto
            if vmax is None:
                vmax = vmax_auto

        # Create figure
        fig, ax = plt.subplots(
            figsize=figsize, num="Z-Normalized Comparison: MBES − UHI"
        )

        # Plot difference heatmap
        im = ax.imshow(
            np.ma.masked_invalid(diff_z),
            extent=extent_ned,
            origin="upper",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        # Clip to footprint
        im.set_clip_path(
            PathPatch(footprint_path, transform=ax.transData, facecolor="none")
        )

        # Draw outline
        if len(segs_xy) > 0:
            lc = LineCollection(
                segs_xy,
                colors=outline_color,
                linewidths=outline_width,
                alpha=outline_alpha,
                zorder=10,
            )
            ax.add_collection(lc)

        # Set limits (tight, no padding)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        # Labels
        ax.set_title(f"Δ ({norm_label}): MBES − UHI")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")

        # Colorbar
        fig.colorbar(
            im,
            ax=ax,
            label=f"Δ ({norm_label})\n+ve=MBES shallower",
        )

        fig.tight_layout()

        if show:
            plt.show(block=False)

        return fig

    def plot_uhi_mbes_comparison(
        self,
        outline_color: str = "black",
        outline_width: float = 1.0,
        outline_alpha: float = 0.9,
        figsize: Tuple[int, int] = (26, 6.5),
        use_adjusted: bool = False,
        flip_uhi_sign: bool = False,
        normalization_method: str = "zscore",
        show: bool = True,
    ) -> plt.Figure:
        """Plot four-panel comparison: Raw UHI RGB | Corrected UHI RGB | MBES residuals | Δz heatmap.

        This method creates a four-panel figure comparing UHI hyperspectral data
        with MBES bathymetry residuals, all normalized using robust z-scores for
        comparison. The panels show:

        1. UHI RGB composite (raw, no correction)
        2. UHI RGB composite (illumination-corrected)
        3. MBES residuals (detrended bathymetry)
        4. Δz heatmap (normalized difference: z(MBES) - z(UHI))
           - Positive (red): MBES shallower than UHI predicts
           - Negative (blue): MBES deeper than UHI predicts
           - Near zero (white): Good agreement

        All four panels are clipped to the UHI footprint with pixel-perfect outline.

        Parameters
        ----------
        outline_color : str, optional
            Color of the footprint outline, by default "black".
        outline_width : float, optional
            Width of the footprint outline, by default 1.0.
        outline_alpha : float, optional
            Transparency of the footprint outline, by default 0.9.
        figsize : tuple, optional
            Figure size (width, height), by default (20, 6.5).
        use_adjusted : bool, optional
            Use adjusted footprint if available, by default False.
        flip_uhi_sign : bool, optional
            If True, invert UHI values before normalization (use if dark=shallow, bright=deep).
            If False (default), assume bright=shallow (typical for water depth imaging).
        normalization_method : str, optional
            Normalization method for comparison. Options:
            - 'zscore' (default): Robust z-score normalization (values typically -6 to +6)
              Formula: (x - median) / MAD
            - 'minmax': Min-max normalization to [0, 1] range
              Formula: (x - min) / (max - min)
        show : bool, optional
            Whether to show the plot immediately, by default True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.

        Raises
        ------
        ValueError
            If coord_system is not 'ned', or if UHI footprint data is not loaded,
            or if detrend() has not been called.
        """
        # Validation
        if self.coord_system.lower() != "ned":
            raise ValueError(
                "plot_uhi_mbes_comparison requires coord_system='ned'. "
                "Create MBESDetrender with coord_system='ned' and ned_origin=(lon0, lat0, h0)."
            )

        if self.uhi_footprint is None:
            raise ValueError(
                "UHI footprint not loaded. Ensure uhi_transect_folder, uhi_files, "
                "and uhi_track_range are set."
            )

        if self.residuals is None:
            raise ValueError("Residuals not computed. Call .detrend() before plotting.")

        if self.ned_origin is None:
            raise ValueError("ned_origin required for NED coordinate system.")

        # Select footprint (original or adjusted)
        using_adjusted = use_adjusted and self.uhi_footprint_adjusted is not None
        if using_adjusted:
            E_uhi, N_uhi, mask_valid = self.uhi_footprint_adjusted
        else:
            E_uhi, N_uhi, mask_valid = self.uhi_footprint

        # Load UHI cube (cached after first call)
        data_corr, uhi_mean, rgb_corrected, cube = self._ensure_uhi_cube_loaded(
            window_size=1000, strength=1.0
        )

        # Get track range (use config if not explicitly set)
        try:
            from utils.common import config

            UHI_TRACK_RANGE = config.UHI_TRACK_RANGE
        except ImportError:
            try:
                from gref_pipeline.config import UHI_TRACK_RANGE
            except ImportError:
                try:
                    from config import UHI_TRACK_RANGE
                except ImportError:
                    raise ImportError("Could not import UHI_TRACK_RANGE from config")

        track_range = self.uhi_track_range or UHI_TRACK_RANGE
        track_start, track_end = track_range

        # Extract RAW RGB (from cube.R, cube.G, cube.B which are built from raw data)
        # The cube's R, G, B are already extracted during build_grids_and_rgb()
        R_raw = cube.R[track_start:track_end, :].copy()
        G_raw = cube.G[track_start:track_end, :].copy()
        B_raw = cube.B[track_start:track_end, :].copy()

        # Normalize raw RGB channels
        for C in (R_raw, G_raw, B_raw):
            m, M = np.nanmin(C), np.nanmax(C)
            if np.isfinite(m) and np.isfinite(M) and M > m:
                C[:] = (C - m) / (M - m)

        rgb_raw = np.stack([R_raw, G_raw, B_raw], axis=-1)
        rgb_raw = np.clip(rgb_raw, 0, 1)

        # Build pixel-perfect footprint path
        Xc_uhi, Yc_uhi = pcolormesh_pad(E_uhi, N_uhi)
        segs_xy, segs_idx = _boundary_segments_from_mask(mask_valid, Xc_uhi, Yc_uhi)
        loops = _trace_loops_from_segments(segs_idx)
        footprint_path = _compound_path_from_loops(loops, Xc_uhi, Yc_uhi)
        xs_outline = footprint_path.vertices[:, 0]
        ys_outline = footprint_path.vertices[:, 1]
        x_min, x_max = np.nanmin(xs_outline), np.nanmax(xs_outline)
        y_min, y_max = np.nanmin(ys_outline), np.nanmax(ys_outline)

        # RGB already loaded from cache

        # MBES is already in NED coordinates (self.x_cols = East, self.y_rows = North)
        # because coord_system='ned' was set during initialization
        from scipy.spatial import cKDTree

        mbes_e_ned = self.x_cols
        mbes_n_ned = self.y_rows
        extent_ned = [
            np.min(mbes_e_ned),
            np.max(mbes_e_ned),
            np.min(mbes_n_ned),
            np.max(mbes_n_ned),
        ]

        # Create MBES grid for inside check
        MBES_E, MBES_N = np.meshgrid(mbes_e_ned, mbes_n_ned)

        # Determine which MBES pixels are inside UHI footprint
        pts_mbes = np.column_stack([MBES_E.ravel(), MBES_N.ravel()])
        inside = footprint_path.contains_points(pts_mbes).reshape(MBES_E.shape)

        # Resample UHI mean to MBES grid (cached after first time, but skip cache if using adjusted footprint)
        if self._uhi_resampled_cache is not None and not using_adjusted:
            uhi_on_mbes = self._uhi_resampled_cache
        else:
            valid_uhi = (
                mask_valid
                & np.isfinite(uhi_mean)
                & np.isfinite(E_uhi)
                & np.isfinite(N_uhi)
            )
            uhi_on_mbes = np.full(self.residuals.shape, np.nan, dtype=float)
            if np.any(valid_uhi):
                uhi_pts = np.column_stack([E_uhi[valid_uhi], N_uhi[valid_uhi]])
                uhi_vals = uhi_mean[valid_uhi]
                tree = cKDTree(uhi_pts)
                idx_inside = np.where(inside.ravel())[0]
                d, nn = tree.query(pts_mbes[idx_inside], k=1)
                flat = uhi_on_mbes.ravel()
                flat[idx_inside] = uhi_vals[nn]
                uhi_on_mbes = flat.reshape(MBES_E.shape)
            # Cache the resampled data
            self._uhi_resampled_cache = uhi_on_mbes

        # Compute normalized values for comparison
        mbes_in = np.where(inside, self.residuals, np.nan)
        uhi_in = np.where(inside, uhi_on_mbes, np.nan)

        # Optionally flip UHI sign (if dark water = shallow instead of deep)
        if flip_uhi_sign:
            uhi_in = -uhi_in

        # Extract valid paired samples BEFORE normalization
        # (normalization should be applied to paired data, not full grids)
        valid_pairs_mask = np.isfinite(mbes_in) & np.isfinite(uhi_in)
        mbes_valid = mbes_in[valid_pairs_mask]
        uhi_valid = uhi_in[valid_pairs_mask]

        # Normalize each dataset INDEPENDENTLY to standardize them (center=0, spread=1)
        # This makes them comparable: both have same scale but preserve their patterns
        if normalization_method.lower() == "minmax":
            # Normalize each to [0, 1] based on their own range
            mbes_min, mbes_max = np.nanmin(mbes_valid), np.nanmax(mbes_valid)
            uhi_min, uhi_max = np.nanmin(uhi_valid), np.nanmax(uhi_valid)
            mbes_norm = (mbes_valid - mbes_min) / (mbes_max - mbes_min)
            uhi_norm = (uhi_valid - uhi_min) / (uhi_max - uhi_min)
            # Center both around 0.5 by subtracting 0.5 so difference is centered at 0
            mbes_norm = mbes_norm - 0.5
            uhi_norm = uhi_norm - 0.5
            norm_label = "normalized [0-1]"
        elif normalization_method.lower() == "zscore":
            # Normalize each to mean=0, std=1 based on their own statistics
            mbes_med = np.nanmedian(mbes_valid)
            uhi_med = np.nanmedian(uhi_valid)
            mbes_mad = median_abs_deviation(mbes_valid, nan_policy="omit")
            uhi_mad = median_abs_deviation(uhi_valid, nan_policy="omit")
            mbes_norm = (mbes_valid - mbes_med) / mbes_mad
            uhi_norm = (uhi_valid - uhi_med) / uhi_mad
            norm_label = "z-units"
        else:
            raise ValueError(
                f"Invalid normalization_method: {normalization_method}. "
                f"Choose 'zscore' or 'minmax'."
            )

        # Reconstruct grids with normalized values
        z_mbes = np.full_like(mbes_in, np.nan)
        z_uhi = np.full_like(uhi_in, np.nan)
        z_mbes[valid_pairs_mask] = mbes_norm
        z_uhi[valid_pairs_mask] = uhi_norm

        # Heatmap shows: Δ = MBES - UHI
        # Positive Δ = MBES shallower than UHI predicts (red)
        # Negative Δ = MBES deeper than UHI predicts (blue)
        diff_z = z_mbes - z_uhi

        # Determine color limits
        vmin_mbes, vmax_mbes = _robust_sym_vlim(self.residuals, q=0.98)
        finite_diff = np.isfinite(diff_z)

        # For minmax normalization, use symmetric limits around zero
        # For zscore, use robust percentile-based limits
        if normalization_method.lower() == "minmax":
            # Minmax: differences are in [-1, 1] range, use symmetric limits
            # But compute based on actual data distribution to handle bias
            if np.any(finite_diff):
                diff_data = diff_z[finite_diff]
                # Use 2nd and 98th percentiles to handle outliers
                vmin_diff = float(np.nanpercentile(diff_data, 2.0))
                vmax_diff = float(np.nanpercentile(diff_data, 98.0))
                # Make symmetric around zero for better visualization
                vlim = max(abs(vmin_diff), abs(vmax_diff))
                vmin_diff, vmax_diff = -vlim, vlim
            else:
                vmin_diff, vmax_diff = -1.0, 1.0
        else:
            # Zscore: use robust symmetric limits based on absolute values
            vmax_diff = (
                np.nanpercentile(np.abs(diff_z[finite_diff]), 98.0)
                if np.any(finite_diff)
                else 1.0
            )
            vmin_diff = -vmax_diff

        # Create figure with 4 panels in 1x4 layout
        from matplotlib import gridspec

        fig = plt.figure(
            constrained_layout=True,
            figsize=figsize,
            num="UHI-MBES Comparison: Raw RGB | Corrected RGB | Residuals | Δz",
        )
        gs = gridspec.GridSpec(ncols=4, nrows=1, figure=fig, width_ratios=[1, 1, 1, 1])

        axR = fig.add_subplot(gs[0, 0])  # Raw UHI RGB
        axU = fig.add_subplot(gs[0, 1])  # Corrected UHI RGB
        axM = fig.add_subplot(gs[0, 2])  # MBES residuals
        axD = fig.add_subplot(gs[0, 3])  # Δz heatmap

        # Panel 1: Raw UHI RGB composite using pcolormesh (native grid coords)
        rgb_raw_plot = rgb_raw.copy()
        rgb_raw_plot[~mask_valid] = np.nan

        # Plot raw RGB using pcolormesh (NO colormap - RGB is the color!)
        im_raw = axR.pcolormesh(Xc_uhi, Yc_uhi, rgb_raw_plot, shading="flat")
        im_raw.set_clip_path(
            PathPatch(footprint_path, transform=axR.transData, facecolor="none")
        )

        # Draw outline
        if len(segs_xy) > 0:
            axR.add_collection(
                LineCollection(
                    segs_xy,
                    colors=outline_color,
                    linewidths=outline_width,
                    alpha=outline_alpha,
                )
            )
        _pad_limits(axR, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True)
        axR.set_title("UHI RGB (raw) — NED")
        axR.set_xlabel("East (m)")
        axR.set_ylabel("North (m)")

        # Panel 2: Corrected UHI RGB composite using pcolormesh (native grid coords)
        rgb_corr_plot = rgb_corrected.copy()
        rgb_corr_plot[~mask_valid] = np.nan

        # Plot corrected RGB using pcolormesh (NO colormap - RGB is the color!)
        im_uhi = axU.pcolormesh(Xc_uhi, Yc_uhi, rgb_corr_plot, shading="flat")
        im_uhi.set_clip_path(
            PathPatch(footprint_path, transform=axU.transData, facecolor="none")
        )

        # Draw outline
        if len(segs_xy) > 0:
            axU.add_collection(
                LineCollection(
                    segs_xy,
                    colors=outline_color,
                    linewidths=outline_width,
                    alpha=outline_alpha,
                )
            )
        _pad_limits(axU, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True)
        axU.set_title("UHI RGB (illum-corrected) — NED")
        axU.set_xlabel("East (m)")
        axU.set_ylabel("North (m)")

        # Panel 3: MBES residuals (clipped to footprint)
        im_mbes = axM.imshow(
            np.ma.masked_invalid(self.residuals),
            extent=extent_ned,
            origin="upper",
            cmap="RdBu_r",
            vmin=vmin_mbes,
            vmax=vmax_mbes,
        )
        im_mbes.set_clip_path(
            PathPatch(footprint_path, transform=axM.transData, facecolor="none")
        )
        if len(segs_xy) > 0:
            axM.add_collection(
                LineCollection(
                    segs_xy,
                    colors=outline_color,
                    linewidths=outline_width,
                    alpha=outline_alpha,
                )
            )
        _pad_limits(axM, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True)
        axM.set_title("MBES Residuals — NED")
        axM.set_xlabel("East (m)")
        axM.set_ylabel("North (m)")
        fig.colorbar(
            im_mbes, ax=axM, fraction=0.025, pad=0.02, label="MBES Residuals [m]"
        )

        # Panel 4: Δz heatmap (normalized difference)
        im_diff = axD.imshow(
            np.ma.masked_invalid(diff_z),
            extent=extent_ned,
            origin="upper",
            cmap="RdBu_r",
            vmin=vmin_diff,
            vmax=vmax_diff,
        )
        im_diff.set_clip_path(
            PathPatch(footprint_path, transform=axD.transData, facecolor="none")
        )
        if len(segs_xy) > 0:
            axD.add_collection(
                LineCollection(
                    segs_xy,
                    colors=outline_color,
                    linewidths=outline_width,
                    alpha=outline_alpha,
                )
            )
        _pad_limits(axD, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True)
        axD.set_title(f"Δ ({norm_label}): MBES − UHI — NED")
        axD.set_xlabel("East (m)")
        axD.set_ylabel("North (m)")
        fig.colorbar(
            im_diff,
            ax=axD,
            fraction=0.025,
            pad=0.02,
            label=f"Δ ({norm_label})\n+ve=MBES shallower",
        )

        if show:
            plt.show(block=False)

        return fig

    def plot_density_scatter(
        self,
        gridsize: int = 80,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 9),
        use_adjusted: bool = False,
        flip_uhi_sign: bool = False,
        normalization_method: str = "zscore",
        show: bool = True,
    ) -> plt.Figure:
        """Plot hexbin density scatter comparing normalized UHI vs MBES.

        This method creates a density scatter plot (hexbin) comparing normalized
        UHI hyperspectral data with MBES bathymetry residuals. Both datasets are
        normalized using robust z-scores for comparison. The plot includes:

        - Hexbin density visualization
        - 1:1 reference line (perfect agreement)
        - Linear regression fit with slope displayed
        - Correlation statistics

        Parameters
        ----------
        gridsize : int, optional
            Number of hexagons in x-direction for hexbin plot, by default 80.
        cmap : str, optional
            Colormap for hexbin density, by default "viridis".
        figsize : tuple, optional
            Figure size (width, height), by default (10, 9).
        use_adjusted : bool, optional
            Use adjusted footprint if available, by default False.
        flip_uhi_sign : bool, optional
            If True, invert UHI values before normalization (use if dark=shallow, bright=deep).
            If False (default), assume bright=shallow (typical for water depth imaging).
        normalization_method : str, optional
            Normalization method for comparison. Options:
            - 'zscore' (default): Robust z-score normalization (values typically -6 to +6)
            - 'minmax': Min-max normalization to [0, 1] range
        show : bool, optional
            Whether to show the plot immediately, by default True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.

        Raises
        ------
        ValueError
            If coord_system is not 'ned', or if UHI footprint data is not loaded,
            or if detrend() has not been called.
        """
        # Validation
        if self.coord_system.lower() != "ned":
            raise ValueError(
                "plot_density_scatter requires coord_system='ned'. "
                "Create MBESDetrender with coord_system='ned' and ned_origin=(lon0, lat0, h0)."
            )

        if self.uhi_footprint is None:
            raise ValueError(
                "UHI footprint not loaded. Ensure uhi_transect_folder, uhi_files, "
                "and uhi_track_range are set."
            )

        if self.residuals is None:
            raise ValueError("Residuals not computed. Call .detrend() before plotting.")

        if self.ned_origin is None:
            raise ValueError("ned_origin required for NED coordinate system.")

        # Select footprint (original or adjusted)
        if use_adjusted and self.uhi_footprint_adjusted is not None:
            E_uhi, N_uhi, mask_valid = self.uhi_footprint_adjusted
            # Clear cache when using adjusted footprint (coordinates changed)
            using_adjusted = True
        else:
            E_uhi, N_uhi, mask_valid = self.uhi_footprint
            using_adjusted = False

        # Load UHI cube (cached after first call)
        data_corr, uhi_mean, rgb, cube = self._ensure_uhi_cube_loaded(
            window_size=1000, strength=1.0
        )

        # Build pixel-perfect footprint path
        Xc_uhi, Yc_uhi = pcolormesh_pad(E_uhi, N_uhi)
        segs_xy, segs_idx = _boundary_segments_from_mask(mask_valid, Xc_uhi, Yc_uhi)
        loops = _trace_loops_from_segments(segs_idx)
        footprint_path = _compound_path_from_loops(loops, Xc_uhi, Yc_uhi)

        # MBES is already in NED coordinates (self.x_cols = East, self.y_rows = North)
        from scipy.spatial import cKDTree

        mbes_e_ned = self.x_cols
        mbes_n_ned = self.y_rows

        # Create MBES grid for inside check
        MBES_E, MBES_N = np.meshgrid(mbes_e_ned, mbes_n_ned)

        # Determine which MBES pixels are inside UHI footprint
        pts_mbes = np.column_stack([MBES_E.ravel(), MBES_N.ravel()])
        inside = footprint_path.contains_points(pts_mbes).reshape(MBES_E.shape)

        # Resample UHI mean to MBES grid (don't use cache if using adjusted footprint)
        if self._uhi_resampled_cache is not None and not using_adjusted:
            uhi_on_mbes = self._uhi_resampled_cache
        else:
            valid_uhi = (
                mask_valid
                & np.isfinite(uhi_mean)
                & np.isfinite(E_uhi)
                & np.isfinite(N_uhi)
            )
            uhi_on_mbes = np.full(self.residuals.shape, np.nan, dtype=float)
            if np.any(valid_uhi):
                uhi_pts = np.column_stack([E_uhi[valid_uhi], N_uhi[valid_uhi]])
                uhi_vals = uhi_mean[valid_uhi]
                tree = cKDTree(uhi_pts)
                idx_inside = np.where(inside.ravel())[0]
                d, nn = tree.query(pts_mbes[idx_inside], k=1)
                flat = uhi_on_mbes.ravel()
                flat[idx_inside] = uhi_vals[nn]
                uhi_on_mbes = flat.reshape(MBES_E.shape)
            # Cache the resampled data
            self._uhi_resampled_cache = uhi_on_mbes

        # Compute normalized values for comparison
        mbes_in = np.where(inside, self.residuals, np.nan)
        uhi_in = np.where(inside, uhi_on_mbes, np.nan)

        # Optionally flip UHI sign (if dark water = shallow instead of deep)
        if flip_uhi_sign:
            uhi_in = -uhi_in

        # DEBUG: Print diagnostics
        print(f"\n=== DENSITY SCATTER DEBUG ===")
        print(f"flip_uhi_sign: {flip_uhi_sign}")
        print(f"normalization_method: {normalization_method}")
        print(f"inside.shape: {inside.shape}, inside.sum(): {np.sum(inside)}")
        print(f"mbes_in.shape: {mbes_in.shape}, finite: {np.isfinite(mbes_in).sum()}")
        print(f"uhi_in.shape: {uhi_in.shape}, finite: {np.isfinite(uhi_in).sum()}")
        print(f"mbes_in range: [{np.nanmin(mbes_in):.4f}, {np.nanmax(mbes_in):.4f}]")
        print(f"uhi_in range: [{np.nanmin(uhi_in):.4f}, {np.nanmax(uhi_in):.4f}]")

        # FIX: Extract valid pairs BEFORE normalization to avoid creating spurious correlation
        # Normalizing each dataset separately can artificially create correlation patterns
        valid_pairs_raw = np.isfinite(mbes_in) & np.isfinite(uhi_in) & inside
        mbes_raw = mbes_in[valid_pairs_raw].ravel()
        uhi_raw = uhi_in[valid_pairs_raw].ravel()

        # Apply normalization method to the PAIRED data only (not the full grids)
        if normalization_method.lower() == "minmax":
            # Normalize using the ranges of the paired samples
            z_mbes = _minmax_normalize(mbes_raw)
            z_uhi = _minmax_normalize(uhi_raw)
            norm_label = "normalized [0-1]"
        elif normalization_method.lower() == "zscore":
            # Normalize using robust z-scores of the paired samples
            z_mbes = _robust_z(mbes_raw)
            z_uhi = _robust_z(uhi_raw)
            norm_label = "z-units"
        else:
            raise ValueError(
                f"Invalid normalization_method: {normalization_method}. "
                f"Choose 'zscore' or 'minmax'."
            )

        print(f"After {normalization_method} (on paired samples only):")
        print(f"z_mbes range: [{np.nanmin(z_mbes):.4f}, {np.nanmax(z_mbes):.4f}]")
        print(f"z_uhi range: [{np.nanmin(z_uhi):.4f}, {np.nanmax(z_uhi):.4f}]")
        print(
            f"z_mbes median: {np.nanmedian(z_mbes):.4f}, std: {np.nanstd(z_mbes):.4f}"
        )
        print(f"z_uhi median: {np.nanmedian(z_uhi):.4f}, std: {np.nanstd(z_uhi):.4f}")
        print(f"===========================\n")

        # Use the normalized paired data directly (already extracted as 1D arrays)
        zM = z_mbes
        zU = z_uhi

        if zM.size == 0:
            raise RuntimeError(
                "No overlapping valid samples to plot. Check footprints/masks."
            )

        # Compute statistics
        pearson_r = float(np.corrcoef(zM, zU)[0, 1]) if zM.size > 1 else np.nan
        from scipy.stats import spearmanr

        spearman_rho = (
            float(spearmanr(zM, zU, nan_policy="omit").correlation)
            if zM.size > 1
            else np.nan
        )
        # Linear fit: zU = a + b*zM
        b, a = np.polyfit(zM, zU, 1) if zM.size > 1 else (np.nan, np.nan)
        r2 = float(pearson_r**2) if np.isfinite(pearson_r) else np.nan

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, num="Density Scatter: UHI vs MBES")

        # Hexbin density plot
        hb = ax.hexbin(zM, zU, gridsize=gridsize, mincnt=1, cmap=cmap)

        # Determine axis limits (symmetric)
        lim = np.nanmax(np.abs([np.nanmin([zM, zU]), np.nanmax([zM, zU])]))
        lim = float(np.clip(lim, 2.0, 6.0))

        # Plot 1:1 line
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="1:1 line")

        # Plot regression line
        ax.plot(
            [-lim, lim],
            [a + b * (-lim), a + b * (lim)],
            "r-",
            lw=1.3,
            label=f"fit: y = {a:.2f} + {b:.2f}x",
        )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"MBES residuals ({norm_label})", fontsize=12)
        ax.set_ylabel(f"UHI mean intensity ({norm_label})", fontsize=12)
        ax.set_title(f"Density Scatter: UHI vs MBES ({norm_label})", fontsize=14)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = fig.colorbar(hb, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("count", fontsize=11)

        # Add statistics text box
        stats_text = (
            f"n = {zM.size:,}\n"
            f"Pearson r = {pearson_r:.3f}\n"
            f"Spearman ρ = {spearman_rho:.3f}\n"
            f"R² = {r2:.3f}\n"
            f"slope = {b:.3f}\n"
            f"intercept = {a:.3f}"
        )
        ax.text(
            0.03,
            0.97,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            fontsize=10,
        )

        ax.legend(loc="lower right", fontsize=10)

        fig.tight_layout()

        if show:
            plt.show(block=False)

        return fig


# ------------------------------ tuner helpers --------------------------------


def _finite(a: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def _nanmad(a: Sequence[float]) -> float:
    a = _finite(a)
    if a.size == 0:
        return np.nan
    m = np.median(a)
    return np.median(np.abs(a - m))


def _robust_std(a: Sequence[float]) -> float:
    mad = _nanmad(a)
    return 1.4826 * mad if np.isfinite(mad) else np.nan


def _banding_metric(resid: np.ndarray) -> float:
    col_med = np.nanmedian(resid, axis=0)
    num = np.nanstd(col_med)
    den = _robust_std(resid)
    return float(num / (den + 1e-12)) if np.isfinite(den) else 0.0


def _edge_bias_metric(resid: np.ndarray, center_frac: float = 0.6) -> float:
    rstd = _robust_std(resid)
    if not np.isfinite(rstd) or rstd == 0:
        return 0.0
    W = resid.shape[1]
    c0 = int((1.0 - center_frac) * 0.5 * W)
    c1 = W - c0
    col_med = np.nanmedian(resid, axis=0)
    center = _finite(col_med[c0:c1])
    edges = _finite(np.r_[col_med[:c0], col_med[c1:]]) if c0 > 0 else np.array([])
    if center.size == 0 or edges.size == 0:
        return 0.0
    return float(abs(np.nanmedian(center) - np.nanmedian(edges)) / (rstd + 1e-12))


def _x_slope_metric(resid: np.ndarray) -> float:
    col_med = np.nanmedian(resid, axis=0)
    x = np.arange(col_med.size)
    m = np.isfinite(col_med)
    if m.sum() < 5:
        return 0.0
    x0 = x[m] - x[m].mean()
    y0 = col_med[m] - np.nanmedian(col_med[m])
    slope = np.sum(x0 * y0) / (np.sum(x0 * x0) + 1e-12)
    return float(abs(slope) / (_robust_std(resid) + 1e-12))


def _roughness_1d(v: Sequence[float]) -> float:
    v = _finite(v)
    if v.size < 3:
        return 0.0
    return float(np.nanmedian(np.abs(np.diff(v))))


def _unique_key(d: Dict) -> str:
    """Stable string key for parameter dictionaries.

    Floats are rounded to six decimal places to avoid minor numeric
    differences (e.g., 0.6 vs 0.6000000001) producing different keys.
    """
    norm: Dict[str, Union[int, float]] = {}
    for k in sorted(d):
        v = d[k]
        if isinstance(v, float):
            norm[k] = round(v, 6)
        else:
            norm[k] = v
    return json.dumps(norm, sort_keys=True)


def _coarse_space(allowed_orders: Sequence[int] = (2, 3)) -> List[Dict]:
    space: List[Dict] = []
    for order in allowed_orders:
        for cf in [0.7, 0.8, 0.9]:
            for base in [0.4, 0.8, 1.2]:
                for tilt in [0.4, 0.8, 1.2]:
                    for center in [0.6, 1.0, 1.6]:
                        space.append(
                            dict(
                                order=order,
                                central_frac=cf,
                                baseline_m=base,
                                tilt_m=tilt,
                                center_m=center,
                            )
                        )
    return space


def _neighbors(best: Dict) -> List[Dict]:
    mults = [0.7, 0.85, 1.0, 1.2, 1.5]
    cfs = [
        max(0.6, min(0.95, best["central_frac"] + d))
        for d in (-0.1, -0.05, 0.0, 0.05, 0.1)
    ]
    out: List[Dict] = []
    for cf in cfs:
        for mb in mults:
            for mt in mults:
                for mc in mults:
                    out.append(
                        dict(
                            order=best["order"],
                            central_frac=cf,
                            baseline_m=max(0.1, best["baseline_m"] * mb),
                            tilt_m=max(0.1, best["tilt_m"] * mt),
                            center_m=max(0.1, best["center_m"] * mc),
                        )
                    )
    uniq: List[Dict] = []
    seen: set = set()
    for p in out:
        k = _unique_key(p)
        if k not in seen:
            uniq.append(p)
            seen.add(k)
    return uniq


# --------------------------------- tuner -------------------------------------


@dataclass
class ScoreWeights:
    """Weights applied to diagnostic terms in the auto tuner score."""

    w_rstd: float = 1.0
    w_banding: float = 1.0
    w_edge_bias: float = 3.0
    w_x_slope: float = 1.0
    w_rough_B: float = 0.3
    w_rough_X: float = 0.3


@dataclass
class RunResult:
    """Results of a single auto tuning trial."""

    params: Dict
    score: float
    diagnostics: Dict


class AutoTuneMBES:
    """Fast, robust auto‑tuning for ``MBESDetrender`` smoothing parameters.

    This class searches over a discrete parameter space of polynomial order,
    central fraction and smoothing lengths to minimize a weighted objective
    function of various residual diagnostics.  It logs every trial to a CSV
    so you can resume or inspect previous runs.  Typical use::

        mb = MBESDetrender(tif_path).load().detrend()
        tuner = AutoTuneMBES(mb)
        best_params = tuner.fit()
        mb_best = tuner.apply_best_and_plot()

    """

    def __init__(
        self,
        mbes: MBESDetrender,
        *,
        allowed_orders: Sequence[int] = (2, 3),
        row_stride: int = 4,
        col_stride: int = 4,
        weights: ScoreWeights = ScoreWeights(),
        log_csv: str = "mbes_tuning_log.csv",
        progress_every: int = 5,
    ) -> None:
        self.src_path = mbes.tif_path
        self.allowed_orders = tuple(
            sorted(set([o for o in allowed_orders if o in (2, 3)]))
        ) or (2, 3)
        self.row_stride = max(1, int(row_stride))
        self.col_stride = max(1, int(col_stride))
        self.weights = weights
        self.log_csv = os.path.abspath(log_csv)
        self.progress_every = max(1, int(progress_every))
        self.tried: Dict[str, Tuple[float, Dict, Dict]] = {}
        self.best_params: Optional[Dict] = None
        self.best_score: Optional[float] = None

    # ---- CSV I/O ----
    def _write_row(self, p: Dict, res: RunResult) -> None:
        path = self.log_csv
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(
                    [
                        "order",
                        "central_frac",
                        "baseline_m",
                        "tilt_m",
                        "center_m",
                        "score",
                        "rstd",
                        "banding",
                        "edge_bias",
                        "x_slope",
                        "rough_B",
                        "rough_X",
                    ]
                )
            w.writerow(
                [
                    p["order"],
                    p["central_frac"],
                    p["baseline_m"],
                    p["tilt_m"],
                    p["center_m"],
                    res.score,
                    res.diagnostics.get("rstd"),
                    res.diagnostics.get("banding"),
                    res.diagnostics.get("edge_bias"),
                    res.diagnostics.get("x_slope"),
                    res.diagnostics.get("rough_B"),
                    res.diagnostics.get("rough_X"),
                ]
            )

    def _load_tried(self) -> None:
        if not os.path.exists(self.log_csv):
            print(f"[resume] no CSV at {self.log_csv}")
            return
        count_before = len(self.tried)
        with open(self.log_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                params = dict(
                    order=int(float(row["order"])),
                    central_frac=float(row["central_frac"]),
                    baseline_m=float(row["baseline_m"]),
                    tilt_m=float(row["tilt_m"]),
                    center_m=float(row["center_m"]),
                )
                if params["order"] not in self.allowed_orders:
                    continue
                key = _unique_key(params)
                self.tried[key] = (float(row["score"]), params, {})
        loaded = len(self.tried) - count_before
        print(f"[resume] loaded {loaded} trials from {self.log_csv}")

    # ---- 1 trial ----
    def _run_once(self, p: Dict) -> RunResult:
        mb = MBESDetrender(self.src_path).load()
        mb.detrend(
            order_x=p["order"],
            smooth_baseline_m=p["baseline_m"],
            smooth_tilt_m=p["tilt_m"],
            smooth_center_m=p["center_m"],
            robust=True,
            central_frac=p["central_frac"],
        )
        r = self._score_run(mb)
        r.params = p
        return r

    def _score_run(self, mb: MBESDetrender) -> RunResult:
        resid = mb.residuals
        assert resid is not None
        rs = resid[:: max(1, self.row_stride), :: max(1, self.col_stride)]
        rstd = _robust_std(rs)
        band = _banding_metric(rs)
        edge = _edge_bias_metric(rs, center_frac=0.6)
        xs = _x_slope_metric(rs)
        rough_B = _roughness_1d(getattr(mb, "B_s", np.array([])))
        rough_X = _roughness_1d(getattr(mb, "x_center_s", np.array([])))
        score = (
            self.weights.w_rstd * (rstd if np.isfinite(rstd) else 0.0)
            + self.weights.w_banding * band
            + self.weights.w_edge_bias * edge
            + self.weights.w_x_slope * xs
            + self.weights.w_rough_B * rough_B
            + self.weights.w_rough_X * rough_X
        )
        diags = dict(
            rstd=rstd,
            banding=band,
            edge_bias=edge,
            x_slope=xs,
            rough_B=rough_B,
            rough_X=rough_X,
        )
        return RunResult(params={}, score=float(score), diagnostics=diags)

    # ---- the search ----
    def fit(
        self,
        *,
        max_rounds: int = 2,
        resume: bool = True,
        early_tol: float = 1e-4,
    ) -> Dict:
        if resume:
            self._load_tried()
        space = _coarse_space(self.allowed_orders)
        t0 = time.time()
        for i, p in enumerate(space, 1):
            key = _unique_key(p)
            if key in self.tried:
                continue
            res = self._run_once(p)
            self.tried[key] = (res.score, p, res.diagnostics)
            self._write_row(p, res)
            if i % self.progress_every == 0:
                print(f"... tried {i}/{len(space)} (last {res.score:.4f})")
        best_key, (best_score, best_params, _) = min(
            self.tried.items(), key=lambda kv: kv[1][0]
        )
        self.best_score, self.best_params = best_score, best_params
        print(f"Coarse best: {best_params} -> {best_score:.6f} ({time.time()-t0:.1f}s)")
        # Local refinement
        for r in range(1, max_rounds + 1):
            improved = False
            cand = _neighbors(self.best_params)
            print(f"Round {r}: {len(cand)} neighbors")
            for j, p in enumerate(cand, 1):
                key = _unique_key(p)
                if key in self.tried:
                    continue
                res = self._run_once(p)
                self.tried[key] = (res.score, p, res.diagnostics)
                self._write_row(p, res)
                if res.score + early_tol < self.best_score:
                    self.best_score = res.score
                    self.best_params = p
                    improved = True
                    print(f"  better: {p}  {res.score:.6f}")
                if j % self.progress_every == 0:
                    print(f"   ... {j}/{len(cand)} done (best={self.best_score:.4f})")
            if not improved:
                print(f"Stop after round {r}: no improvement.")
                break
        print(f"Best params: {self.best_params}  score={self.best_score:.6f}")
        return self.best_params  # type: ignore[return-value]

    # ---- apply & plot ----
    def apply_best_and_plot(self) -> MBESDetrender:
        assert self.best_params is not None, "Run .fit() first."
        p = self.best_params
        mb = (
            MBESDetrender(self.src_path)
            .load()
            .detrend(
                order_x=p["order"],
                smooth_baseline_m=p["baseline_m"],
                smooth_tilt_m=p["tilt_m"],
                smooth_center_m=p["center_m"],
                robust=True,
                central_frac=p["central_frac"],
            )
        )
        print("Plotting with best params...")
        mb.plot_triptych(figsize=(9, 14), equal_aspect=True)
        mb.plot_components(figsize=(10, 8))
        mb.plot_residual_heatmap(figsize=(6, 14), equal_aspect=True)
        return mb
