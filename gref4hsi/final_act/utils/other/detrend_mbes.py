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
        return self

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
        uhi_footprint: Optional[
            Union[MplPath, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
    ) -> plt.Figure:
        """Plot original, trend, and residual panels with optional footprint overlay.

        This method produces a figure with three columns: the original raster,
        the estimated trend, and the residual (detrended) bathymetry.  If
        ``uhi_footprint`` is provided, the footprint region will be filled
        with black on the residual panel while all other data remain
        unchanged.  Acceptable values for ``uhi_footprint`` are either a
        Matplotlib ``Path`` object delineating the footprint, or a tuple
        ``(E, N, mask)`` where ``E`` and ``N`` are meshgrids of eastings
        and northings (as returned by ``numpy.meshgrid``) and ``mask`` is a
        boolean array of the same shape indicating membership in the
        footprint.

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
        uhi_footprint: Path or (E, N, mask), optional
            Optional overlay footprint to fill with black on the residual panel.

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
        fig.colorbar(im0, ax=axs[0], label="Depth (m)", fraction=0.035, pad=0.02)
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
        fig.colorbar(im1, ax=axs[1], label="Depth (m)", fraction=0.035, pad=0.02)
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
        fig.colorbar(im2, ax=axs[2], label="m", fraction=0.035, pad=0.02)
        # Determine cropping extents
        if crop_to_data:
            x_min, x_max, y_min, y_max = self._data_window(bounds_percentile)
        else:
            x_min, x_max, y_min, y_max = self._extent()
        # Overlay the footprint as a black fill on the residual panel if provided
        if uhi_footprint is not None:
            # helper to build a Path from (E, N, mask)
            def _build_path(
                E: np.ndarray, N: np.ndarray, mask: np.ndarray
            ) -> Optional[MplPath]:
                # create a contour at 0.5 to trace the mask boundary
                cs = axs[2].contour(
                    E, N, mask.astype(float), levels=[0.5], linewidths=0
                )
                polys: List[np.ndarray] = []
                for coll in cs.collections:
                    for p in coll.get_paths():
                        v = p.vertices
                        if v.shape[0] >= 3:
                            polys.append(v.copy())
                # remove temporary contour
                for coll in cs.collections:
                    coll.remove()
                if not polys:
                    return None
                verts_all: List[np.ndarray] = []
                codes_all: List[np.ndarray] = []
                for poly in polys:
                    codes = np.full(
                        poly.shape[0], MplPath.LINETO, dtype=MplPath.code_type
                    )
                    codes[0] = MplPath.MOVETO
                    verts_all.append(poly)
                    codes_all.append(codes)
                return MplPath(np.vstack(verts_all), np.concatenate(codes_all))

            # determine if user supplied a Path or a tuple
            if isinstance(uhi_footprint, MplPath):
                fp_path = uhi_footprint
            elif (
                isinstance(uhi_footprint, tuple)
                and len(uhi_footprint) == 3
                and isinstance(uhi_footprint[0], np.ndarray)
                and isinstance(uhi_footprint[1], np.ndarray)
                and isinstance(uhi_footprint[2], np.ndarray)
            ):
                E, N, mask = uhi_footprint
                fp_path = _build_path(E, N, mask)
            else:
                raise TypeError(
                    "uhi_footprint must be a matplotlib.path.Path or (E, N, mask) tuple"
                )
            if fp_path is not None:
                patch = PathPatch(
                    fp_path,
                    transform=axs[2].transData,
                    facecolor="k",
                    edgecolor="none",
                    zorder=10,
                )
                axs[2].add_patch(patch)
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
        figsize: Tuple[int, int] = (10, 8),
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
        self, *, show: bool = True
    ) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        """Plot triptych, components, and residual heatmap.

        Returns the three figures created by ``plot_triptych``,
        ``plot_components``, and ``plot_residual_heatmap`` respectively.
        """
        f1 = self.plot_triptych(show=False)
        f2 = self.plot_components(show=False)
        f3 = self.plot_residual_heatmap(show=False)
        if show:
            plt.show(block=False)
        return f1, f2, f3


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
