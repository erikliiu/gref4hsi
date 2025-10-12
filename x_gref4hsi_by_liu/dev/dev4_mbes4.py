# mbes_detrend_plot_fixed.py
# Standalone: pip install rasterio numpy matplotlib scipy

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation


# ---------- helpers -----------------------------------------------------------


def _pad_limits(ax, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True):
    """Pad axes limits a bit and set a data-driven aspect so tall/skinny swaths
    fill the figure instead of being squashed into a sliver."""
    xr = max(1e-12, x_max - x_min)
    yr = max(1e-12, y_max - y_min)

    ax.set_xlim(x_min - xr * pad_frac, x_max + xr * pad_frac)
    ax.set_ylim(y_min - yr * pad_frac, y_max + yr * pad_frac)

    if equal_aspect:
        # Keep meters equal AND resize the axes box to the data ratio.
        # (Prevents the “half a figure” look with very tall/skinny rasters.)
        ax.set_aspect("equal", adjustable="datalim")
        try:
            ax.set_box_aspect(yr / xr)  # Matplotlib ≥ 3.3
        except Exception:
            pass

    # Plain numeric ticks (no +6.74e6 offsets)
    try:
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    except Exception:
        pass


def _robust_sym_vlim(arr, q=0.98):
    m = np.isfinite(arr)
    if not m.any():
        return (-1, 1)
    lo, hi = np.nanquantile(arr[m], [(1 - q) / 2, 1 - (1 - q) / 2])
    vmax = max(abs(lo), abs(hi))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = (np.nanstd(arr[m]) or 1.0) * 3
    return (-vmax, vmax)


def _rows_from_meters(pix_size_y, meters, poly=2):
    n = max(5, int(round((meters or 0.0) / max(abs(pix_size_y) or 1.0, 1e-9))))
    if n % 2 == 0:
        n += 1
    if n <= poly + 2:
        n = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
    return n


def _savgol_nanaware(arr, pix_size_y, meters, poly=2):
    """Savitzky–Golay smoothing that copes with NaNs; falls back to Gaussian if
    window > #valid samples."""
    a = np.asarray(arr, float)
    valid = np.isfinite(a)
    if valid.sum() < 3:
        return a.copy()

    nwin = _rows_from_meters(pix_size_y, meters, poly=poly)

    if valid.sum() < nwin:
        # Fallback: NaN-aware Gaussian via normalization
        tmp = a.copy()
        tmp[~valid] = 0.0
        w = valid.astype(float)
        sigma = max(0.5, nwin / 6)
        num = gaussian_filter1d(tmp, sigma=sigma, mode="reflect", truncate=3.5)
        den = gaussian_filter1d(w, sigma=sigma, mode="reflect", truncate=3.5)
        out = np.where(den > 1e-6, num / den, np.nan)
        out[~valid] = np.nan
        return out

    # Prefill internal gaps with linear interpolation, then SG with edge interp
    filled = a.copy()
    if (~valid).any():
        idx = np.arange(filled.size)
        filled[~valid] = np.interp(idx[~valid], idx[valid], filled[valid])
    out = savgol_filter(filled, window_length=nwin, polyorder=poly, mode="interp")
    return out


# ---------- main class --------------------------------------------------------


@dataclass
class MBESDetrender:
    tif_path: str
    nodata_value: Optional[float] = 9999.0

    # smoothing scales (meters, along-track)
    smooth_baseline_m: float = 10.0  # A(y)
    smooth_tilt_m: float = 20.0  # B(y)
    smooth_curv_m: Optional[float] = None  # C(y); defaults to tilt_m
    smooth_center_m: float = 6.0  # x̄(y)

    # robust fit options
    max_row_iters: int = 3
    clip_sigma: float = 6.0
    central_frac: float = 0.8  # use central 80% of beams

    # internals (filled by load/detrend)
    data: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    transform: Optional[object] = None
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

    # --- I/O ---
    def load(self):
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
        self.x_cols = c + a * (cols + 0.5)
        self.y_rows = f + e * (rows + 0.5)
        self.px_x = abs(a)
        self.px_y = abs(e)
        return self

    # --- detrend ---
    def detrend(
        self,
        order_x: int = 2,  # 0: baseline, 1: +tilt, 2: +curvature
        remove_cross_track_tilt: bool = True,
        robust: bool = True,
        smooth_baseline_m: Optional[float] = None,
        smooth_tilt_m: Optional[float] = None,
        smooth_curv_m: Optional[float] = None,
        smooth_center_m: Optional[float] = None,
        central_frac: Optional[float] = None,
    ):
        if self.data is None:
            self.load()

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

        # per-row robust polynomial around each row's center
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

        # clip crazy rows, fill short gaps
        def _clip_and_fill(arr):
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

        # along-track smoothing (edge safe)
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

        # store
        self.A, self.B, self.C = A, B, C
        self.A_s, self.B_s, self.C_s = A_s, B_s, C_s
        self.x_center_row, self.x_center_s = x_center, Xc
        self.trend, self.residuals = trend, resid
        return self

    # --- plotting ---
    def _extent(self):
        return (self.extent[0], self.extent[1], self.extent[2], self.extent[3])

    def _data_window(self, percentile=99.5):
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
        cmap_orig="terrain",
        cmap_resid="RdBu_r",
        figsize=(9, 14),  # tall default for skinny swaths
        crop_to_data=True,
        bounds_percentile=99.5,
        margin_frac=0.03,
        equal_aspect=True,
    ):
        assert self.trend is not None, "Run detrend() first."
        fig, axs = plt.subplots(
            1, 3, figsize=figsize, constrained_layout=True, num="MBES Triptych"
        )

        im0 = axs[0].imshow(
            np.ma.masked_invalid(self.data),
            extent=self._extent(),
            origin="upper",
            cmap=cmap_orig,
        )
        axs[0].set_title("Original")
        axs[0].set_xlabel("Easting (m)")
        axs[0].set_ylabel("Northing (m)")
        fig.colorbar(im0, ax=axs[0], label="Depth (m)", fraction=0.035, pad=0.02)

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
        axs[1].set_xlabel("Easting (m)")
        axs[1].set_ylabel("Northing (m)")
        fig.colorbar(im1, ax=axs[1], label="Depth (m)", fraction=0.035, pad=0.02)

        vmin, vmax = _robust_sym_vlim(self.residuals, q=0.98)
        im2 = axs[2].imshow(
            np.ma.masked_invalid(self.residuals),
            extent=self._extent(),
            origin="upper",
            cmap=cmap_resid,
            vmin=vmin,
            vmax=vmax,
        )
        axs[2].set_title("Residuals (local variations)")
        axs[2].set_xlabel("Easting (m)")
        axs[2].set_ylabel("Northing (m)")
        fig.colorbar(im2, ax=axs[2], label="m", fraction=0.035, pad=0.02)

        if crop_to_data:
            x_min, x_max, y_min, y_max = self._data_window(bounds_percentile)
        else:
            x_min, x_max, y_min, y_max = self._extent()

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

        plt.show(block=False)
        return fig

    def plot_residual_heatmap(
        self,
        figsize=(6, 14),
        cmap="RdBu_r",
        vmax=None,
        crop_to_data=True,
        bounds_percentile=99.5,
        margin_frac=0.03,
        equal_aspect=True,
        title_suffix="",
    ):
        assert self.residuals is not None, "Run detrend() first."
        if vmax is None:
            vmin, vmax = _robust_sym_vlim(self.residuals, q=0.98)
        else:
            vmin, vmax = -abs(vmax), abs(vmax)

        fig, ax = plt.subplots(
            1, 1, figsize=figsize, constrained_layout=True, num="MBES Residuals"
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
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        fig.colorbar(im, ax=ax, label="m", fraction=0.035, pad=0.02)
        plt.show(block=False)
        return fig

    def plot_components(
        self, figsize=(10, 8), bounds_percentile=99.5, margin_frac=0.03
    ):
        assert self.A_s is not None, "Run detrend() first."
        y = self.y_rows
        fig, ax = plt.subplots(
            2, 2, figsize=figsize, constrained_layout=True, num="MBES Components"
        )

        ax[0, 0].plot(self.A, y, alpha=0.25, label="row fit")
        ax[0, 0].plot(self.A_s, y, lw=2, label="smoothed")
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_title(f"A(y) baseline — σ={self.smooth_baseline_m:.2f}m")
        ax[0, 0].set_xlabel("Depth (m)")
        ax[0, 0].set_ylabel("Northing (m)")
        ax[0, 0].legend()

        ax[0, 1].plot(self.B, y, alpha=0.25, label="row fit")
        ax[0, 1].plot(self.B_s, y, lw=2, label="smoothed")
        ax[0, 1].invert_yaxis()
        ax[0, 1].set_title(f"B(y) tilt — σ={self.smooth_tilt_m:.2f}m")
        ax[0, 1].set_xlabel("Slope (m/m)")
        ax[0, 1].set_ylabel("Northing (m)")
        ax[0, 1].legend()

        if self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0:
            ax[1, 0].plot(self.C, y, alpha=0.25, label="row fit")
            ax[1, 0].plot(self.C_s, y, lw=2, label="smoothed")
            ax[1, 0].invert_yaxis()
            ax[1, 0].set_title(f"C(y) curvature — σ={self.smooth_curv_m:.2f}m")
            ax[1, 0].set_xlabel("1/m")
            ax[1, 0].set_ylabel("Northing (m)")
            ax[1, 0].legend()
        else:
            ax[1, 0].axis("off")

        ax[1, 1].plot(self.x_center_row, y, alpha=0.25, label="row median x")
        ax[1, 1].plot(self.x_center_s, y, lw=2, label="smoothed")
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_title(
            f"Swath center x̄(y) — σ={self.smooth_center_m:.2f}m, central={self.central_frac*100:.0f}%"
        )
        ax[1, 1].set_xlabel("Easting (m)")
        ax[1, 1].set_ylabel("Northing (m)")
        ax[1, 1].legend()

        # keep y range similar to map panels
        x_min, x_max, y_min, y_max = self._data_window(bounds_percentile)
        for a in ax.flat:
            a.set_ylim(
                y_min - (y_max - y_min) * margin_frac,
                y_max + (y_max - y_min) * margin_frac,
            )

        plt.show(block=False)
        return fig


# ---------- example usage -----------------------------------------------------
if __name__ == "__main__":
    plt.ion()

    tif = r"E:\mjosa_new\DTM\geotiff_2.tif"  # <-- change if needed
    mb = MBESDetrender(tif).load()

    mb.detrend(
        order_x=3,
        smooth_baseline_m=0.80,
        smooth_tilt_m=0.70,
        smooth_center_m=1.0,
        robust=True,
        central_frac=0.8,
    )

    # Tall defaults + data-driven box aspect prevent the “half figure” look
    mb.plot_triptych(figsize=(9, 14), equal_aspect=True)
    mb.plot_components(figsize=(10, 8))
    mb.plot_residual_heatmap(figsize=(6, 14), equal_aspect=True)

    plt.show(block=True)
