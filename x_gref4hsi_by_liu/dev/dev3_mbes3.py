import numpy as np
import rasterio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation


# ---------- helper: padded limits so plots aren't “edge-to-edge” ----------
def _pad_limits(ax, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True):
    xr = max(1e-12, x_max - x_min)
    yr = max(1e-12, y_max - y_min)
    ax.set_xlim(x_min - xr * pad_frac, x_max + xr * pad_frac)
    ax.set_ylim(y_min - yr * pad_frac, y_max + yr * pad_frac)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


@dataclass
class MBESDetrender:
    tif_path: str
    nodata_value: Optional[float] = 9999.0
    crop_start_m: float = 0.0
    crop_end_m: float = 0.0

    # --- smoothing scales (meters, along-track) ---
    smooth_baseline_m: float = 10.0  # for A(y)
    smooth_tilt_m: float = 20.0  # for B(y)
    smooth_curv_m: Optional[float] = None  # for C(y); default = tilt_m
    smooth_center_m: float = 6.0  # for the swath center x̄(y)

    # --- robust fit options ---
    max_row_iters: int = 3
    clip_sigma: float = 6.0
    central_frac: float = 0.8

    # internals
    data: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    transform: Optional[object] = None
    extent: Optional[Tuple[float, float, float, float]] = None
    px_x: Optional[float] = None
    px_y: Optional[float] = None
    x_cols: Optional[np.ndarray] = None
    y_rows: Optional[np.ndarray] = None

    # fitted pieces
    x_center_row: Optional[np.ndarray] = None
    x_center_s: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    A_s: Optional[np.ndarray] = None
    B_s: Optional[np.ndarray] = None
    C_s: Optional[np.ndarray] = None

    trend: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    # ---------- I/O ----------
    def load(self):
        with rasterio.open(self.tif_path) as src:
            z = src.read(1).astype(np.float64)
            tf = src.transform
            bounds = src.bounds

        if self.nodata_value is not None:
            z = np.where(z == self.nodata_value, np.nan, z)

        self.valid_mask = np.isfinite(z)

        a, b, c, d, e, f = tf.a, tf.b, tf.c, tf.d, tf.e, tf.f
        H, W = z.shape
        cols = np.arange(W)
        rows = np.arange(H)
        self.x_cols = c + a * (cols + 0.5)
        self.y_rows = f + e * (rows + 0.5)
        self.px_x = abs(a)
        self.px_y = abs(e)

        crop_applied = False
        if self.crop_start_m > 0:
            crop_start_rows = int(np.ceil(self.crop_start_m / self.px_y))
            if crop_start_rows > 0:
                self.valid_mask[-crop_start_rows:, :] = False
                crop_applied = True
        if self.crop_end_m > 0:
            crop_end_rows = int(np.ceil(self.crop_end_m / self.px_y))
            if crop_end_rows > 0:
                self.valid_mask[:crop_end_rows, :] = False
                crop_applied = True

        if crop_applied:
            z = np.where(self.valid_mask, z, np.nan)

        self.data = z
        self.transform = tf
        self.extent = (
            float(bounds.left),
            float(bounds.right),
            float(bounds.bottom),
            float(bounds.top),
        )
        return self

    def uhi_footprint_mask(self, uhi_x_coords, uhi_y_coords, buffer_m=0.0, dilate_px=0):
        from scipy.ndimage import binary_dilation

        assert self.data is not None and self.px_x and self.px_y, "Call .load() first."

        x = np.asarray(uhi_x_coords, dtype=float).ravel()
        y = np.asarray(uhi_y_coords, dtype=float).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            return np.zeros_like(self.data, dtype=bool)

        H, W = self.data.shape
        col0_left = self.x_cols[0] - self.px_x * 0.5
        cols = np.floor((x - col0_left) / self.px_x).astype(int)
        row0_top = self.y_rows[0] + self.px_y * 0.5
        rows = np.floor((row0_top - y) / self.px_y).astype(int)

        good = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows = rows[good]
        cols = cols[good]
        mask = np.zeros((H, W), dtype=bool)
        if rows.size:
            mask[rows, cols] = True

        total_dilate = dilate_px
        if buffer_m and buffer_m > 0:
            dx = int(np.ceil(buffer_m / max(self.px_x, 1e-12)))
            dy = int(np.ceil(buffer_m / max(self.px_y, 1e-12)))
            total_dilate = max(total_dilate, max(dx, dy))
        if total_dilate > 0:
            mask = binary_dilation(mask, iterations=int(total_dilate))

        if getattr(self, "valid_mask", None) is not None:
            mask &= self.valid_mask
        return mask

    # ---------- detrend ----------
    def detrend(
        self,
        order_x: int = 1,
        remove_cross_track_tilt: bool = True,
        robust: bool = True,
        smooth_baseline_m: Optional[float] = None,
        smooth_tilt_m: Optional[float] = None,
        smooth_curv_m: Optional[float] = None,
        smooth_center_m: Optional[float] = None,
        central_frac: Optional[float] = None,
    ):
        """
        Trend per row: A(y) + B(y)*(x - x̄(y)) [+ C(y)*(x - x̄(y))^2]
        Then smooth A,B,C,x̄ along-track (rows) with Savitzky–Golay (edge-safe).
        """
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

        z = self.data
        x = self.x_cols
        H, W = z.shape

        valid_rows = np.any(self.valid_mask, axis=1)
        valid_row_indices = np.where(valid_rows)[0]

        A = np.full(H, np.nan)
        B = np.full(H, 0.0)
        C = np.full(H, 0.0)
        x_center = np.full(H, np.nan)

        # --- per-row robust polynomial about the row center ---
        for i in valid_row_indices:
            row = z[i, :]
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
                a0 = float(coef[0])
                a1 = 0.0
                a2 = 0.0
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
                        a2 = 0.0

            A[i] = a0
            B[i] = a1 if remove_cross_track_tilt else 0.0
            C[i] = a2 if (order_x >= 2 and remove_cross_track_tilt) else 0.0

        # --- clip crazy rows then edge-safe along-track smoothing ---
        def _clip_and_fill(arr):
            s = arr.copy()
            m = np.isfinite(s) & valid_rows
            if m.sum() < 5:
                return s
            med = np.nanmedian(s[m])
            mad = median_abs_deviation(s[m], scale="normal") or np.nanstd(s[m])
            if np.isfinite(mad) and mad > 0:
                lo, hi = med - self.clip_sigma * mad, med + self.clip_sigma * mad
                outlier = (s < lo) | (s > hi)
                s[outlier & valid_rows] = np.nan

            # linear fill only *within* the valid-row span (no extrapolation over cropped blocks)
            good = np.where(np.isfinite(s) & valid_rows)[0]
            if good.size >= 2:
                # fill holes between first and last good row with linear interpolation;
                # keep the ends anchored at the nearest good value so Savitzky–Golay has sane edges
                full = np.arange(good[0], good[-1] + 1)
                sseg = s[full]
                mseg = np.isfinite(sseg)
                if mseg.sum() >= 2:
                    sseg[~mseg] = np.interp(
                        np.flatnonzero(~mseg), np.flatnonzero(mseg), sseg[mseg]
                    )
                s[full] = sseg
            return s

        A = _clip_and_fill(A)
        B = _clip_and_fill(B)
        C = _clip_and_fill(C)
        x_center = _clip_and_fill(x_center)

        # window sizes (rows) from meters; ensure odd and >= poly+2
        def _rows_from_m(meters):
            return max(5, int(round(meters / max(self.px_y or 1.0, 1e-9))))

        def _savgol_nanaware(arr, meters, poly=2):
            out = arr.copy()
            nwin = _rows_from_m(meters)
            if nwin % 2 == 0:
                nwin += 1
            if nwin <= poly + 2:
                nwin = poly + 3
                if nwin % 2 == 0:
                    nwin += 1
            # If too few valid samples for Savitzky–Golay, fall back to Gaussian (reflect)
            valid = np.isfinite(arr)
            if valid.sum() < nwin:
                tmp = arr.copy()
                tmp[~valid] = 0.0
                m = valid.astype(float)
                num = gaussian_filter1d(
                    tmp, sigma=max(0.5, nwin / 6), mode="reflect", truncate=3.5
                )
                den = gaussian_filter1d(
                    m, sigma=max(0.5, nwin / 6), mode="reflect", truncate=3.5
                )
                out = np.where(den > 1e-3, num / den, np.nan)
                out[~valid] = np.nan
                return out
            # prefill internal NaNs by local linear interpolation (already done in _clip_and_fill);
            # now Savitzky–Golay with edge interpolation preserves slope/curvature at ends
            filled = out.copy()
            nan_idx = ~np.isfinite(filled)
            if nan_idx.any():
                idx = np.arange(filled.size)
                filled[nan_idx] = np.interp(
                    idx[nan_idx], idx[~nan_idx], filled[~nan_idx]
                )
            out = savgol_filter(
                filled, window_length=nwin, polyorder=poly, mode="interp"
            )
            # keep gaps outside valid-rows region as NaN
            out[~valid_rows] = np.nan
            return out

        A_s = _savgol_nanaware(A, self.smooth_baseline_m, poly=2)
        B_s = _savgol_nanaware(B, self.smooth_tilt_m, poly=2)
        C_s = _savgol_nanaware(C, self.smooth_curv_m, poly=2)
        x_center_s = _savgol_nanaware(x_center, self.smooth_center_m, poly=2)

        xi_grid = self.x_cols[None, :] - x_center_s[:, None]
        trend = A_s[:, None] + B_s[:, None] * xi_grid + C_s[:, None] * (xi_grid**2)

        if getattr(self, "valid_mask", None) is not None:
            trend = np.where(self.valid_mask, trend, np.nan)

        resid = self.data - trend
        resid[~np.isfinite(self.data)] = np.nan

        self.A, self.B, self.C = A, B, C
        self.A_s, self.B_s, self.C_s = A_s, B_s, C_s
        self.x_center_row, self.x_center_s = x_center, x_center_s
        self.trend, self.residuals = trend, resid
        return self

    # ---------- plotting ----------
    def _extent_imshow(self):
        return (self.extent[0], self.extent[1], self.extent[2], self.extent[3])

    def _robust_sym_vlim(self, arr, q=0.98):
        m = np.isfinite(arr)
        if not m.any():
            return (-1, 1)
        lo, hi = np.nanquantile(arr[m], [(1 - q) / 2, 1 - (1 - q) / 2])
        vmax = max(abs(lo), abs(hi))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = (np.nanstd(arr[m]) or 1.0) * 3
        return (-vmax, vmax)

    def _get_data_bounds(self, percentile=99.5):
        if self.valid_mask is None:
            return self._extent_imshow()
        rows, cols = np.where(self.valid_mask)
        if len(rows) == 0:
            return self._extent_imshow()
        pct_low = (100 - percentile) / 2
        pct_high = 100 - pct_low
        col_min = int(np.percentile(cols, pct_low))
        col_max = int(np.percentile(cols, pct_high))
        row_min = int(np.percentile(rows, pct_low))
        row_max = int(np.percentile(rows, pct_high))
        x_min = self.x_cols[col_min]
        x_max = self.x_cols[col_max]
        y_min = self.y_rows[row_max]
        y_max = self.y_rows[row_min]
        return (x_min, x_max, y_min, y_max)

    def plot_data_coverage(self, figsize=(12, 8), margin_frac=0.03, equal_aspect=True):
        if self.valid_mask is None:
            print("No valid_mask available. Run load() first.")
            return
        fig, axes = plt.subplots(2, 2, figsize=figsize, num="Data Coverage Diagnostic")

        axes[0, 0].imshow(
            self.valid_mask,
            extent=self._extent_imshow(),
            origin="upper",
            cmap="gray",
            interpolation="nearest",
        )
        axes[0, 0].set_title("Data Coverage (white=valid)")
        axes[0, 0].set_xlabel("Easting (m)")
        axes[0, 0].set_ylabel("Northing (m)")
        x_min, x_max, y_min, y_max = self._get_data_bounds(percentile=99.5)
        _pad_limits(
            axes[0, 0],
            x_min,
            x_max,
            y_min,
            y_max,
            pad_frac=margin_frac,
            equal_aspect=equal_aspect,
        )

        row_coverage = np.sum(self.valid_mask, axis=1)
        axes[0, 1].plot(row_coverage, self.y_rows)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_title("Valid pixels per row (along-track)")
        axes[0, 1].set_xlabel("Number of valid pixels")
        axes[0, 1].set_ylabel("Northing (m)")
        axes[0, 1].grid(True, alpha=0.3)

        col_coverage = np.sum(self.valid_mask, axis=0)
        axes[1, 0].plot(self.x_cols, col_coverage)
        axes[1, 0].set_title("Valid pixels per column (across-track)")
        axes[1, 0].set_xlabel("Easting (m)")
        axes[1, 0].set_ylabel("Number of valid pixels")
        axes[1, 0].grid(True, alpha=0.3)

        rows, cols = np.where(self.valid_mask)
        if len(rows) > 0:
            stats_text = f"""Data Coverage Statistics:

Total valid pixels: {len(rows):,}
Coverage: {len(rows)/self.valid_mask.size*100:.1f}%

Easting (across-track):
  Min: {self.x_cols[cols.min()]:.2f} m
  Max: {self.x_cols[cols.max()]:.2f} m
  Range: {self.x_cols[cols.max()] - self.x_cols[cols.min()]:.2f} m

Northing (along-track):
  Min: {self.y_rows[rows.max()]:.2f} m
  Max: {self.y_rows[rows.min()]:.2f} m
  Range: {self.y_rows[rows.min()] - self.y_rows[rows.max()]:.2f} m
"""
            axes[1, 1].text(
                0.05,
                0.95,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=9,
                va="top",
                family="monospace",
            )
            axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show(block=False)
        return fig

    def plot_triptych(
        self,
        cmap_orig="terrain",
        cmap_resid="RdBu_r",
        figsize=(14, 5),
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
            extent=self._extent_imshow(),
            origin="upper",
            cmap=cmap_orig,
        )
        axs[0].set_title("Original")
        axs[0].set_xlabel("Easting (m)")
        axs[0].set_ylabel("Northing (m)")
        fig.colorbar(im0, ax=axs[0], label="Depth (m)")

        im1 = axs[1].imshow(
            np.ma.masked_invalid(self.trend),
            extent=self._extent_imshow(),
            origin="upper",
            cmap=cmap_orig,
        )
        order = (
            2 if (self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0.001) else 1
        )
        settings_str = f"order={order}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m"
        axs[1].set_title(f"Estimated trend\n{settings_str}", fontsize=10)
        axs[1].set_xlabel("Easting (m)")
        axs[1].set_ylabel("Northing (m)")
        fig.colorbar(im1, ax=axs[1], label="Depth (m)")

        vmin, vmax = self._robust_sym_vlim(self.residuals, q=0.98)
        im2 = axs[2].imshow(
            np.ma.masked_invalid(self.residuals),
            extent=self._extent_imshow(),
            origin="upper",
            cmap=cmap_resid,
            vmin=vmin,
            vmax=vmax,
        )
        axs[2].set_title("Residuals (local variations)")
        axs[2].set_xlabel("Easting (m)")
        axs[2].set_ylabel("Northing (m)")
        fig.colorbar(im2, ax=axs[2], label="m")

        if crop_to_data:
            x_min, x_max, y_min, y_max = self._get_data_bounds(
                percentile=bounds_percentile
            )
        else:
            x_min, x_max, y_min, y_max = self.extent
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

    def plot_components(
        self,
        figsize=(12, 6),
        margin_frac=0.03,
        equal_aspect=True,
        bounds_percentile=99.5,
    ):
        assert self.A_s is not None, "Run detrend() first."
        y = self.y_rows
        fig, ax = plt.subplots(
            2, 2, figsize=figsize, constrained_layout=True, num="MBES Components"
        )

        ax[0, 0].plot(self.A, y, alpha=0.25, label="row fit")
        ax[0, 0].plot(self.A_s, y, lw=2, label="smoothed")
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_title(
            f"Along-track baseline A(y)\nσ={self.smooth_baseline_m:.2f}m", fontsize=10
        )
        ax[0, 0].set_xlabel("Depth (m) at row center")
        ax[0, 0].set_ylabel("Northing (m)")
        ax[0, 0].legend()

        ax[0, 1].plot(self.B, y, alpha=0.25, label="row fit")
        ax[0, 1].plot(self.B_s, y, lw=2, label="smoothed")
        ax[0, 1].invert_yaxis()
        ax[0, 1].set_title(
            f"Cross-track tilt B(y)\nσ={self.smooth_tilt_m:.2f}m", fontsize=10
        )
        ax[0, 1].set_xlabel("Slope (m/m)")
        ax[0, 1].set_ylabel("Northing (m)")
        ax[0, 1].legend()

        if self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0:
            ax[1, 0].plot(self.C, y, alpha=0.25, label="row fit")
            ax[1, 0].plot(self.C_s, y, lw=2, label="smoothed")
            ax[1, 0].invert_yaxis()
            ax[1, 0].set_title(
                f"Cross-track curvature C(y)\nσ={self.smooth_curv_m:.2f}m", fontsize=10
            )
            ax[1, 0].set_xlabel("Quadratic coeff (m/m²)")
            ax[1, 0].set_ylabel("Northing (m)")
            ax[1, 0].legend()

        ax[1, 1].plot(self.x_center_row, y, alpha=0.25, label="row median x")
        ax[1, 1].plot(self.x_center_s, y, lw=2, label="smoothed")
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_title(
            f"Swath center x̄(y)\nσ={self.smooth_center_m:.2f}m, central={self.central_frac*100:.0f}%",
            fontsize=10,
        )
        ax[1, 1].set_xlabel("Easting (m)")
        ax[1, 1].set_ylabel("Northing (m)")
        ax[1, 1].legend()

        x_min, x_max, y_min, y_max = self._get_data_bounds(percentile=bounds_percentile)
        for axis in ax.flat:
            axis.set_ylim(
                y_min - (y_max - y_min) * margin_frac,
                y_max + (y_max - y_min) * margin_frac,
            )

        plt.show(block=False)
        return fig

    def plot_residual_heatmap(
        self,
        figsize=(6, 12),
        cmap="RdBu_r",
        vmax=None,
        crop_to_data=True,
        bounds_percentile=99.5,
        margin_frac=0.03,
        equal_aspect=True,
    ):
        assert self.residuals is not None, "Run detrend() first."
        if vmax is None:
            vmin, vmax = self._robust_sym_vlim(self.residuals, q=0.98)
        else:
            vmin = -abs(vmax)
            vmax = abs(vmax)
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, constrained_layout=True, num="MBES Residuals"
        )
        im = ax.imshow(
            np.ma.masked_invalid(self.residuals),
            extent=self._extent_imshow(),
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if crop_to_data:
            x_min, x_max, y_min, y_max = self._get_data_bounds(
                percentile=bounds_percentile
            )
        else:
            x_min, x_max, y_min, y_max = self.extent
        _pad_limits(
            ax,
            x_min,
            x_max,
            y_min,
            y_max,
            pad_frac=margin_frac,
            equal_aspect=equal_aspect,
        )

        order = (
            2 if (self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0.001) else 1
        )
        settings_str = f"order={order}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m, center={self.smooth_center_m:.2f}m"
        ax.set_title(
            f"Detrended bathymetry (local variations)\n{settings_str}", fontsize=10
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        fig.colorbar(im, ax=ax, label="m")
        plt.show(block=False)
        return fig

    def save_residual_geotiff(self, out_path: str):
        assert (
            self.residuals is not None and self.transform is not None
        ), "Run detrend() first."
        with rasterio.open(self.tif_path) as src:
            profile = src.profile
        profile.update(dtype="float32", count=1, nodata=np.nan)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(self.residuals.astype("float32"), 1)
        print(f"✅ Saved: {out_path}")

    def plot_subregion_residuals(
        self,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        figsize=(8, 10),
        cmap="RdBu_r",
        vmax=None,
        title_suffix="",
        margin_frac=0.03,
        equal_aspect=True,
    ):
        assert self.residuals is not None, "Run detrend() first."
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        col_mask = (self.x_cols >= x_min) & (self.x_cols <= x_max)
        row_mask = (self.y_rows >= y_min) & (self.y_rows <= y_max)
        col_indices = np.where(col_mask)[0]
        row_indices = np.where(row_mask)[0]
        if len(col_indices) == 0 or len(row_indices) == 0:
            print(
                f"⚠️  No MBES data found in region: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]"
            )
            return None
        col_start, col_end = col_indices[0], col_indices[-1] + 1
        row_start, row_end = row_indices[0], row_indices[-1] + 1
        subregion = self.residuals[row_start:row_end, col_start:col_end]
        sub_extent = (
            self.x_cols[col_start] - self.px_x / 2,
            self.x_cols[col_end - 1] + self.px_x / 2,
            self.y_rows[row_end - 1] - self.px_y / 2,
            self.y_rows[row_start] + self.px_y / 2,
        )
        if vmax is None:
            vmin, vmax = self._robust_sym_vlim(subregion, q=0.98)
        else:
            vmin = -abs(vmax)
            vmax = abs(vmax)
        fig, ax = plt.subplots(
            1, 1, figsize=figsize, constrained_layout=True, num="MBES Subregion"
        )
        im = ax.imshow(
            np.ma.masked_invalid(subregion),
            extent=sub_extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        order = (
            2 if (self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0.001) else 1
        )
        settings_str = f"order={order}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m"
        title = f"MBES Detrended Bathymetry - Subregion\n{settings_str}"
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        _pad_limits(
            ax,
            x_min,
            x_max,
            y_min,
            y_max,
            pad_frac=margin_frac,
            equal_aspect=equal_aspect,
        )
        fig.colorbar(im, ax=ax, label="Residual depth (m)")
        plt.show(block=False)
        return fig

    def plot_subregion_residuals_masked(
        self,
        uhi_x_coords,
        uhi_y_coords,
        x_bounds=None,
        y_bounds=None,
        figsize=(8, 10),
        cmap="RdBu_r",
        vmax=None,
        title_suffix="",
        buffer_m=0.0,
        margin_frac=0.03,
        equal_aspect=True,
    ):
        assert self.residuals is not None, "Run detrend() first."
        if x_bounds is None:
            x_min, x_max = np.nanmin(uhi_x_coords), np.nanmax(uhi_x_coords)
        else:
            x_min, x_max = x_bounds
        if y_bounds is None:
            y_min, y_max = np.nanmin(uhi_y_coords), np.nanmax(uhi_y_coords)
        else:
            y_min, y_max = y_bounds
        x_min -= buffer_m
        x_max += buffer_m
        y_min -= buffer_m
        y_max += buffer_m

        col_mask = (self.x_cols >= x_min) & (self.x_cols <= x_max)
        row_mask = (self.y_rows >= y_min) & (self.y_rows <= y_max)
        col_indices = np.where(col_mask)[0]
        row_indices = np.where(row_mask)[0]
        if len(col_indices) == 0 or len(row_indices) == 0:
            print(
                f"⚠️  No MBES data found in region: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]"
            )
            return None, None
        col_start, col_end = col_indices[0], col_indices[-1] + 1
        row_start, row_end = row_indices[0], row_indices[-1] + 1

        subregion = self.residuals[row_start:row_end, col_start:col_end].copy()
        sub_y_rows = self.y_rows[row_start:row_end]
        sub_x_cols = self.x_cols[col_start:col_end]

        uhi_x_flat = uhi_x_coords.ravel()
        uhi_y_flat = uhi_y_coords.ravel()
        valid = np.isfinite(uhi_x_flat) & np.isfinite(uhi_y_flat)
        uhi_x_valid = uhi_x_flat[valid]
        uhi_y_valid = uhi_y_flat[valid]

        coverage_mask = np.zeros(subregion.shape, dtype=bool)
        half_px_x = self.px_x / 2
        half_px_y = self.px_y / 2
        for x_uhi, y_uhi in zip(uhi_x_valid, uhi_y_valid):
            col_idx = np.argmin(np.abs(sub_x_cols - x_uhi))
            row_idx = np.argmin(np.abs(sub_y_rows - y_uhi))
            if (abs(x_uhi - sub_x_cols[col_idx]) <= half_px_x) and (
                abs(y_uhi - sub_y_rows[row_idx]) <= half_px_y
            ):
                coverage_mask[row_idx, col_idx] = True

        masked_subregion = np.ma.masked_where(~coverage_mask, subregion)
        masked_subregion = np.ma.masked_invalid(masked_subregion)

        sub_extent = (
            sub_x_cols[0] - self.px_x / 2,
            sub_x_cols[-1] + self.px_x / 2,
            sub_y_rows[-1] - self.px_y / 2,
            sub_y_rows[0] + self.px_y / 2,
        )

        if vmax is None:
            vmin, vmax = self._robust_sym_vlim(masked_subregion, q=0.98)
        else:
            vmin = -abs(vmax)
            vmax = abs(vmax)

        fig, ax = plt.subplots(
            1, 1, figsize=figsize, constrained_layout=True, num="MBES Masked"
        )
        im = ax.imshow(
            masked_subregion,
            extent=sub_extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        order = (
            2 if (self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0.001) else 1
        )
        settings_str = f"order={order}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m"
        title = f"MBES Detrended Bathymetry - Masked to UHI Coverage\n{settings_str}"
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        _pad_limits(
            ax,
            x_min,
            x_max,
            y_min,
            y_max,
            pad_frac=margin_frac,
            equal_aspect=equal_aspect,
        )
        fig.colorbar(im, ax=ax, label="Residual depth (m)")
        plt.show(block=False)
        return fig, coverage_mask

    def compare_with_uhi_footprint(
        self,
        uhi_x_coords,
        uhi_y_coords,
        x_bounds=None,
        y_bounds=None,
        buffer_m=0.0,
        figsize=(16, 10),
        cmap="RdBu_r",
        vmax=None,
        title_suffix="",
        margin_frac=0.03,
        equal_aspect=True,
    ):
        assert self.residuals is not None, "Run detrend() first."
        mask_full = self.uhi_footprint_mask(uhi_x_coords, uhi_y_coords, buffer_m=0.0)

        if x_bounds is None:
            x_min, x_max = np.nanmin(uhi_x_coords), np.nanmax(uhi_x_coords)
        else:
            x_min, x_max = x_bounds
        if y_bounds is None:
            y_min, y_max = np.nanmin(uhi_y_coords), np.nanmax(uhi_y_coords)
        else:
            y_min, y_max = y_bounds
        x_min -= buffer_m
        x_max += buffer_m
        y_min -= buffer_m
        y_max += buffer_m

        col_mask = (self.x_cols >= x_min) & (self.x_cols <= x_max)
        row_mask = (self.y_rows >= y_min) & (self.y_rows <= y_max)
        col_indices = np.where(col_mask)[0]
        row_indices = np.where(row_mask)[0]
        if len(col_indices) == 0 or len(row_indices) == 0:
            print("⚠️  No MBES data in bounds")
            return None, mask_full

        col_start, col_end = col_indices[0], col_indices[-1] + 1
        row_start, row_end = row_indices[0], row_indices[-1] + 1
        subregion_rect = self.residuals[row_start:row_end, col_start:col_end].copy()
        mask_sub = mask_full[row_start:row_end, col_start:col_end]

        sub_extent = (
            self.x_cols[col_start] - self.px_x / 2,
            self.x_cols[col_end - 1] + self.px_x / 2,
            self.y_rows[row_end - 1] - self.px_y / 2,
            self.y_rows[row_start] + self.px_y / 2,
        )

        if vmax is None:
            vmin, vmax = self._robust_sym_vlim(subregion_rect, q=0.98)
        else:
            vmin, vmax = -abs(vmax), abs(vmax)

        fig_comparison, axes = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True, num="UHI-MBES Comparison"
        )

        im1 = axes[0].imshow(
            np.ma.masked_invalid(subregion_rect),
            extent=sub_extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].set_title("MBES - Full Subregion", fontsize=12)
        axes[0].set_xlabel("Easting (m)")
        axes[0].set_ylabel("Northing (m)")
        fig_comparison.colorbar(im1, ax=axes[0], label="Residual depth (m)")

        masked_subregion = np.ma.masked_where(~mask_sub, subregion_rect)
        masked_subregion = np.ma.masked_invalid(masked_subregion)
        im2 = axes[1].imshow(
            masked_subregion,
            extent=sub_extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title("MBES - Masked to UHI Footprint (exact pixels)", fontsize=12)
        axes[1].set_xlabel("Easting (m)")
        axes[1].set_ylabel("Northing (m)")
        fig_comparison.colorbar(im2, ax=axes[1], label="Residual depth (m)")

        x0, x1, y0, y1 = sub_extent
        _pad_limits(
            axes[0], x0, x1, y0, y1, pad_frac=margin_frac, equal_aspect=equal_aspect
        )
        _pad_limits(
            axes[1], x0, x1, y0, y1, pad_frac=margin_frac, equal_aspect=equal_aspect
        )

        order = (
            2 if (self.C_s is not None and np.nanmax(np.abs(self.C_s)) > 0.001) else 1
        )
        settings_str = f"order={order}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m"
        fig_comparison.suptitle(
            f"MBES vs UHI Footprint Comparison\n{settings_str}", fontsize=14
        )

        plt.show(block=False)
        return fig_comparison, mask_full


if __name__ == "__main__":
    plt.ion()
    print("\n📂 Loading MBES GeoTIFF...")
    mb = MBESDetrender(
        r"E:\mjosa_new\DTM\geotiff_2.tif",
        crop_start_m=0.0,
        crop_end_m=0.0,
    ).load()

    print(f"   Shape: {mb.data.shape}")
    print(f"   Resolution: {mb.px_x:.4f} × {mb.px_y:.4f} m")

    print("\n🔍 Data coverage...")
    mb.plot_data_coverage(margin_frac=0.03, equal_aspect=True)

    print("\n⚙️  Detrending settings:")
    params = {
        "order_x": 2,  # linear + (optional) quadratic across-track
        "smooth_baseline_m": 0.01,  # adjust as needed
        "smooth_tilt_m": 0.01,
        "smooth_center_m": 10.0,
        "robust": True,
        "central_frac": 0.9,
    }
    for k, v in params.items():
        print(f"   {k:20s} = {v}")

    mb.detrend(**params)

    mb.plot_triptych(margin_frac=0.03, equal_aspect=True, bounds_percentile=99.5)
    mb.plot_components(margin_frac=0.03, bounds_percentile=99.5)
    mb.plot_residual_heatmap(
        figsize=(5, 12), margin_frac=0.03, equal_aspect=True, bounds_percentile=99.5
    )

    plt.show(block=True)
