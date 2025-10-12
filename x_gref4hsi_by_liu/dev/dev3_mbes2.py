import numpy as np
import rasterio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.stats import median_abs_deviation


@dataclass
class MBESDetrender:
    tif_path: str
    nodata_value: Optional[float] = 9999.0
    crop_start_m: float = (
        0.0  # crop this many meters from transect start (bottom in image)
    )
    crop_end_m: float = 0.0  # crop this many meters from transect end (top in image)

    # --- smoothing scales (meters, along-track) ---
    smooth_baseline_m: float = 10.0  # for A(y)
    smooth_tilt_m: float = 20.0  # for B(y)
    smooth_curv_m: Optional[float] = None  # for C(y); default = tilt_m
    smooth_center_m: float = 6.0  # for the swath center x̄(y)

    # --- robust fit options ---
    max_row_iters: int = 3
    clip_sigma: float = 6.0  # clip crazy A/B/C rows before smoothing
    central_frac: float = 0.8  # fit only central 80% of valid beams to avoid edges

    # internals
    data: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None  # True where we have valid data
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

        # Keep the footprint of valid data (True where we have data)
        self.valid_mask = np.isfinite(z)

        # 1D coords (pixel centers) – assumes north-up (no rotation)
        a, b, c, d, e, f = tf.a, tf.b, tf.c, tf.d, tf.e, tf.f
        H, W = z.shape
        cols = np.arange(W)
        rows = np.arange(H)
        self.x_cols = c + a * (cols + 0.5)
        self.y_rows = f + e * (rows + 0.5)
        self.px_x = abs(a)
        self.px_y = abs(e)

        # Crop noisy start/end: mask out specified distances from transect start/end (along-track)
        crop_applied = False
        if self.crop_start_m > 0:
            # Convert meters to number of rows (along-track direction)
            # crop_start = beginning of transect = BOTTOM of image (last rows)
            crop_start_rows = int(np.ceil(self.crop_start_m / self.px_y))
            if crop_start_rows > 0:
                self.valid_mask[-crop_start_rows:, :] = False  # Bottom rows
                crop_applied = True

        if self.crop_end_m > 0:
            # Convert meters to number of rows (along-track direction)
            # crop_end = end of transect = TOP of image (first rows)
            crop_end_rows = int(np.ceil(self.crop_end_m / self.px_y))
            if crop_end_rows > 0:
                self.valid_mask[:crop_end_rows, :] = False  # Top rows
                crop_applied = True

        # Apply the cropped mask to the data
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
        """
        Build a boolean mask (same shape as MBES) of MBES pixels covered by the UHI.
        Assumes north-up (no rotation), which your code already assumes.

        Parameters:
        -----------
        uhi_x_coords, uhi_y_coords : arrays (T,S) in the SAME CRS as MBES (UTM!)
        buffer_m  : expand footprint by this many meters (converted to pixel dilation)
        dilate_px : extra integer dilation in pixels (after buffer)

        Returns:
        --------
        mask : ndarray (H, W), dtype=bool
            Boolean mask where True = MBES pixel covered by UHI data
        """
        from scipy.ndimage import binary_dilation

        assert self.data is not None and self.px_x and self.px_y, "Call .load() first."

        # 1) stack valid UHI points
        x = np.asarray(uhi_x_coords, dtype=float).ravel()
        y = np.asarray(uhi_y_coords, dtype=float).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            return np.zeros_like(self.data, dtype=bool)

        H, W = self.data.shape

        # 2) convert world coords -> pixel indices (constant scale, north-up)
        # column 0 spans [x_cols[0]-px_x/2, x_cols[0]+px_x/2)
        col0_left = self.x_cols[0] - self.px_x * 0.5
        cols = np.floor((x - col0_left) / self.px_x).astype(int)

        # row 0 spans [y_rows[0]+px_y/2 (top), y_rows[0]-px_y/2 (bottom)] since y decreases with row
        row0_top = self.y_rows[0] + self.px_y * 0.5
        rows = np.floor((row0_top - y) / self.px_y).astype(int)

        # 3) clip to raster bounds
        valid_indices = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows = rows[valid_indices]
        cols = cols[valid_indices]

        if rows.size == 0:
            return np.zeros_like(self.data, dtype=bool)

        # 4) paint mask
        mask = np.zeros((H, W), dtype=bool)
        mask[rows, cols] = True

        # 5) optional expansion: buffer in meters -> pixels -> dilation
        total_dilate = dilate_px
        if buffer_m and buffer_m > 0:
            # convert meters to pixels (round up)
            dx = int(np.ceil(buffer_m / max(self.px_x, 1e-12)))
            dy = int(np.ceil(buffer_m / max(self.px_y, 1e-12)))
            total_dilate = max(total_dilate, max(dx, dy))
        if total_dilate > 0:
            mask = binary_dilation(mask, iterations=int(total_dilate))

        # 6) keep only inside original MBES footprint
        if getattr(self, "valid_mask", None) is not None:
            mask &= self.valid_mask

        return mask

    # ---------- detrend ----------
    def detrend(
        self,
        order_x: int = 1,  # 1: linear tilt, 2: add quadratic
        remove_cross_track_tilt: bool = True,
        robust: bool = True,
        smooth_baseline_m: Optional[float] = None,
        smooth_tilt_m: Optional[float] = None,
        smooth_curv_m: Optional[float] = None,
        smooth_center_m: Optional[float] = None,
        central_frac: Optional[float] = None,
    ):
        """
        Build trend: A_s(y) + B_s(y)*(x - x̄_s(y)) [+ C_s(y)*(x - x̄_s(y))^2]
        where x̄_s(y) is the smoothed per-row swath center.
        """
        if self.data is None:
            self.load()

        # overrides
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

        # Identify which rows have ANY valid data after cropping
        valid_rows = np.any(
            self.valid_mask, axis=1
        )  # True for rows with at least one valid pixel
        valid_row_indices = np.where(valid_rows)[0]  # Actual row indices to process

        A = np.full(H, np.nan)
        B = np.full(H, 0.0)
        C = np.full(H, 0.0)
        x_center = np.full(H, np.nan)

        # --- per-row robust polynomial fit around the *row's* center ---
        # Only process rows that have valid data (skip cropped rows entirely)
        for i in valid_row_indices:
            row = z[i, :]
            m = np.isfinite(row)
            if m.sum() < 6:
                continue

            xv = x[m]
            yv = row[m]

            # row center = median of valid x -> avoids extrapolation when swath shifts
            xc_row = np.median(xv)
            x_center[i] = xc_row

            # use only central fraction to avoid noisy edges
            if 0 < self.central_frac < 1:
                lo = np.quantile(xv, 0.5 - 0.5 * self.central_frac)
                hi = np.quantile(xv, 0.5 + 0.5 * self.central_frac)
                keep = (xv >= lo) & (xv <= hi)
                xv, yv = xv[keep], yv[keep]

            xi = xv - xc_row

            # initial LS fit
            deg = (
                2
                if (order_x >= 2 and remove_cross_track_tilt)
                else (1 if remove_cross_track_tilt else 0)
            )
            coef = np.polyfit(xi, yv, deg)  # highest power first
            # convert to a0 + a1*xi + a2*xi^2 with a0 = value at xi=0
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
                    # weighted LS in polynomial basis
                    X = np.vstack([np.ones_like(xi), xi])
                    if deg == 2:
                        X = np.vstack([X, xi**2])
                    W = w
                    # normal equations with weights
                    XtW = X * W
                    beta = np.linalg.lstsq(XtW.T, (yv * W), rcond=None)[0]
                    if deg == 2:
                        a0, a1, a2 = float(beta[0]), float(beta[1]), float(beta[2])
                    else:
                        a0, a1 = float(beta[0]), float(beta[1])
                        a2 = 0.0

            A[i] = a0
            B[i] = a1 if remove_cross_track_tilt else 0.0
            C[i] = a2 if (order_x >= 2 and remove_cross_track_tilt) else 0.0

        # --- clip crazy rows then smooth along-track (in rows) ---
        def _clip_and_fill(arr):
            s = arr.copy()
            # Only work on valid rows (don't interpolate over cropped regions)
            m = np.isfinite(s) & valid_rows  # Must be finite AND in a valid row
            if m.sum() < 5:
                return s
            med = np.nanmedian(s[m])
            mad = median_abs_deviation(s[m], scale="normal") or np.nanstd(s[m])
            if not np.isfinite(mad) or mad == 0:
                return s
            lo, hi = med - self.clip_sigma * mad, med + self.clip_sigma * mad
            # Clip outliers only in valid rows
            outlier = (s < lo) | (s > hi)
            s[outlier & valid_rows] = np.nan
            # Interpolate only within the valid row region (don't fill cropped rows)
            good = np.where(np.isfinite(s) & valid_rows)[0]
            if good.size >= 2:
                # Only interpolate within the range of valid rows
                interp_range = valid_rows.copy()
                s[interp_range] = np.interp(
                    np.arange(len(s))[interp_range],
                    good,
                    s[good],
                    left=np.nan,
                    right=np.nan,
                )
            return s

        A = _clip_and_fill(A)
        B = _clip_and_fill(B)
        C = _clip_and_fill(C)
        x_center = _clip_and_fill(x_center)

        # pixel -> meters to get Gaussian sigma in rows
        sigmaA = max(0.5, self.smooth_baseline_m / (self.px_y or 1.0))
        sigmaB = max(0.5, self.smooth_tilt_m / (self.px_y or 1.0))
        sigmaC = max(
            0.5, (self.smooth_curv_m or self.smooth_tilt_m) / (self.px_y or 1.0)
        )
        sigmaX = max(0.5, self.smooth_center_m / (self.px_y or 1.0))

        # NaN-aware Gaussian smoothing: only smooth over valid rows
        def _smooth_valid(arr, sigma):
            """Gaussian smooth only over valid (non-NaN) entries"""
            result = arr.copy()
            mask = np.isfinite(arr)
            if mask.sum() < 3:
                return result
            # Smooth the values and the mask separately
            arr_filled = arr.copy()
            arr_filled[~mask] = 0  # Fill NaN with 0 for convolution
            smoothed = gaussian_filter1d(
                arr_filled, sigma=sigma, mode="nearest", truncate=3.5
            )
            mask_smoothed = gaussian_filter1d(
                mask.astype(float), sigma=sigma, mode="nearest", truncate=3.5
            )
            # Normalize: divide by the smoothed mask to get proper weighted average
            result = np.where(mask_smoothed > 0.01, smoothed / mask_smoothed, np.nan)
            # Keep NaN where original was NaN (don't extrapolate into cropped regions)
            result[~mask] = np.nan
            return result

        A_s = _smooth_valid(A, sigmaA)
        B_s = _smooth_valid(B, sigmaB)
        C_s = _smooth_valid(C, sigmaC)
        x_center_s = _smooth_valid(x_center, sigmaX)

        # --- reconstruct trend on the full grid (broadcasting, no huge arrays) ---
        xi_grid = self.x_cols[None, :] - x_center_s[:, None]
        trend = A_s[:, None] + B_s[:, None] * xi_grid + C_s[:, None] * (xi_grid**2)

        # Mask trend outside the original data footprint
        if hasattr(self, "valid_mask") and self.valid_mask is not None:
            trend = np.where(self.valid_mask, trend, np.nan)

        resid = self.data - trend
        resid[~np.isfinite(self.data)] = np.nan

        # store
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
        """
        Get robust data bounds by using percentiles to ignore sparse outliers.

        Parameters:
        -----------
        percentile : float
            Use this percentile of valid pixels (default 99.5 = ignore outermost 0.5%)

        Returns:
        --------
        tuple : (x_min, x_max, y_min, y_max)
        """
        if self.valid_mask is None:
            return self._extent_imshow()

        # Get row and column indices of all valid pixels
        rows, cols = np.where(self.valid_mask)

        if len(rows) == 0:
            return self._extent_imshow()

        # Use percentiles to ignore outliers
        pct_low = (100 - percentile) / 2
        pct_high = 100 - pct_low

        col_min = int(np.percentile(cols, pct_low))
        col_max = int(np.percentile(cols, pct_high))
        row_min = int(np.percentile(rows, pct_low))
        row_max = int(np.percentile(rows, pct_high))

        # Convert to coordinates
        x_min = self.x_cols[col_min]
        x_max = self.x_cols[col_max]
        y_min = self.y_rows[row_max]  # inverted because origin='upper'
        y_max = self.y_rows[row_min]

        return (x_min, x_max, y_min, y_max)

    def plot_data_coverage(self, figsize=(12, 8)):
        """
        Diagnostic plot showing data coverage and histograms to identify outliers.
        """
        if self.valid_mask is None:
            print("No valid_mask available. Run load() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize, num="Data Coverage Diagnostic")

        # 1. Coverage map
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

        # 2. Row coverage histogram (along-track)
        row_coverage = np.sum(self.valid_mask, axis=1)
        axes[0, 1].plot(row_coverage, self.y_rows)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_title("Valid pixels per row (along-track)")
        axes[0, 1].set_xlabel("Number of valid pixels")
        axes[0, 1].set_ylabel("Northing (m)")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Column coverage histogram (across-track)
        col_coverage = np.sum(self.valid_mask, axis=0)
        axes[1, 0].plot(self.x_cols, col_coverage)
        axes[1, 0].set_title("Valid pixels per column (across-track)")
        axes[1, 0].set_xlabel("Easting (m)")
        axes[1, 0].set_ylabel("Number of valid pixels")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Statistics
        rows, cols = np.where(self.valid_mask)
        if len(rows) > 0:
            stats_text = f"""Data Coverage Statistics:
            
Total valid pixels: {len(rows):,}
Coverage: {len(rows)/self.valid_mask.size*100:.1f}%

Easting (across-track):
  Min: {self.x_cols[cols.min()]:.2f} m
  Max: {self.x_cols[cols.max()]:.2f} m
  Range: {self.x_cols[cols.max()] - self.x_cols[cols.min()]:.2f} m
  
  Percentile 0.5%: {self.x_cols[int(np.percentile(cols, 0.5))]:.2f} m
  Percentile 99.5%: {self.x_cols[int(np.percentile(cols, 99.5))]:.2f} m
  Robust range: {self.x_cols[int(np.percentile(cols, 99.5))] - self.x_cols[int(np.percentile(cols, 0.5))]:.2f} m

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
                verticalalignment="top",
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
        # Display settings in title
        settings_str = f"order={2 if np.nanmax(np.abs(self.C_s)) > 0.001 else 1}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m"
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

        # Optionally crop all panels to data extent (using robust percentiles)
        if crop_to_data:
            x_min, x_max, y_min, y_max = self._get_data_bounds(percentile=99.5)
            for ax in axs:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

        plt.show(block=False)
        return fig

    def plot_components(self, figsize=(12, 6)):
        assert self.A_s is not None, "Run detrend() first."
        y = self.y_rows
        fig, ax = plt.subplots(
            2, 2, figsize=figsize, constrained_layout=True, num="MBES Components"
        )

        ax[0, 0].plot(self.A, y, alpha=0.25, label="row fit")
        ax[0, 0].plot(self.A_s, y, lw=2, label="smoothed")
        ax[0, 0].invert_yaxis()
        settings_str = f"σ={self.smooth_baseline_m:.2f}m"
        ax[0, 0].set_title(f"Along-track baseline A(y)\n{settings_str}", fontsize=10)
        ax[0, 0].set_xlabel("Depth (m) at row center")
        ax[0, 0].set_ylabel("Northing (m)")
        ax[0, 0].legend()

        ax[0, 1].plot(self.B, y, alpha=0.25, label="row fit")
        ax[0, 1].plot(self.B_s, y, lw=2, label="smoothed")
        ax[0, 1].invert_yaxis()
        settings_str = f"σ={self.smooth_tilt_m:.2f}m"
        ax[0, 1].set_title(f"Cross-track tilt B(y)\n{settings_str}", fontsize=10)
        ax[0, 1].set_xlabel("Slope (m/m)")
        ax[0, 1].set_ylabel("Northing (m)")
        ax[0, 1].legend()

        if np.nanmax(np.abs(self.C_s)) > 0:
            ax[1, 0].plot(self.C, y, alpha=0.25, label="row fit")
            ax[1, 0].plot(self.C_s, y, lw=2, label="smoothed")
            ax[1, 0].invert_yaxis()
            settings_str = f"σ={self.smooth_curv_m:.2f}m"
            ax[1, 0].set_title(
                f"Cross-track curvature C(y)\n{settings_str}", fontsize=10
            )
            ax[1, 0].set_xlabel("Quadratic coeff (m/m²)")
            ax[1, 0].set_ylabel("Northing (m)")
            ax[1, 0].legend()

        ax[1, 1].plot(self.x_center_row, y, alpha=0.25, label="row median x")
        ax[1, 1].plot(self.x_center_s, y, lw=2, label="smoothed")
        ax[1, 1].invert_yaxis()
        settings_str = (
            f"σ={self.smooth_center_m:.2f}m, central={self.central_frac*100:.0f}%"
        )
        ax[1, 1].set_title(f"Swath center x̄(y)\n{settings_str}", fontsize=10)
        ax[1, 1].set_xlabel("Easting (m)")
        ax[1, 1].set_ylabel("Northing (m)")
        ax[1, 1].legend()

        # Crop all subplots to valid data extent (only show non-cropped region)
        x_min, x_max, y_min, y_max = self._get_data_bounds(percentile=99.5)
        for axis in ax.flat:
            axis.set_ylim(y_max, y_min)  # Inverted because origin is at top

        plt.show(block=False)
        return fig

    def plot_residual_heatmap(
        self, figsize=(6, 12), cmap="RdBu_r", vmax=None, crop_to_data=True
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

        # Optionally crop view to data extent (using robust percentiles)
        if crop_to_data:
            x_min, x_max, y_min, y_max = self._get_data_bounds(percentile=99.5)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Display settings in title
        order = 2 if np.nanmax(np.abs(self.C_s)) > 0.001 else 1
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
    ):
        """
        Plot detrended bathymetry for a specific geographic subregion.

        Parameters:
        -----------
        x_bounds : tuple of (x_min, x_max)
            Easting bounds in meters (same coordinate system as MBES)
        y_bounds : tuple of (y_min, y_max)
            Northing bounds in meters (same coordinate system as MBES)
        figsize : tuple, default=(8, 10)
            Figure size
        cmap : str, default="RdBu_r"
            Colormap for residuals
        vmax : float, optional
            Symmetric color limits. If None, uses robust percentiles
        title_suffix : str, optional
            Additional text to append to title

        Returns:
        --------
        fig : matplotlib figure
        """
        assert self.residuals is not None, "Run detrend() first."

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        # Find pixel indices corresponding to these bounds
        # x_cols: easting coordinates (column centers)
        # y_rows: northing coordinates (row centers)
        col_mask = (self.x_cols >= x_min) & (self.x_cols <= x_max)
        row_mask = (self.y_rows >= y_min) & (self.y_rows <= y_max)

        col_indices = np.where(col_mask)[0]
        row_indices = np.where(row_mask)[0]

        if len(col_indices) == 0 or len(row_indices) == 0:
            print(
                f"⚠️  No MBES data found in region: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]"
            )
            return None

        # Extract subregion
        col_start, col_end = col_indices[0], col_indices[-1] + 1
        row_start, row_end = row_indices[0], row_indices[-1] + 1

        subregion = self.residuals[row_start:row_end, col_start:col_end]

        # Compute extent for this subregion
        sub_extent = (
            self.x_cols[col_start] - self.px_x / 2,  # left
            self.x_cols[col_end - 1] + self.px_x / 2,  # right
            self.y_rows[row_end - 1] - self.px_y / 2,  # bottom
            self.y_rows[row_start] + self.px_y / 2,  # top
        )

        # Color limits
        if vmax is None:
            vmin, vmax = self._robust_sym_vlim(subregion, q=0.98)
        else:
            vmin = -abs(vmax)
            vmax = abs(vmax)

        # Create figure
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

        # Settings info
        order = 2 if np.nanmax(np.abs(self.C_s)) > 0.001 else 1
        settings_str = f"order={order}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m"

        title = f"MBES Detrended Bathymetry - Subregion\n{settings_str}"
        if title_suffix:
            title += f"\n{title_suffix}"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        cbar = fig.colorbar(im, ax=ax, label="Residual depth (m)")

        # Print info
        valid_count = np.sum(np.isfinite(subregion))
        total_count = subregion.size
        coverage_pct = 100 * valid_count / total_count if total_count > 0 else 0

        print(f"\n📊 MBES Subregion extracted:")
        print(f"   X range: [{x_min:.1f}, {x_max:.1f}] m")
        print(f"   Y range: [{y_min:.1f}, {y_max:.1f}] m")
        print(f"   Subregion shape: {subregion.shape}")
        print(f"   Coverage: {coverage_pct:.1f}% ({valid_count:,}/{total_count:,})")

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
    ):
        """
        Plot MBES residuals masked to match UHI coverage exactly.

        This creates a plot where only MBES pixels that have corresponding UHI
        measurements are shown, creating the same "raggedy" contour as the UHI plot.

        Parameters:
        -----------
        uhi_x_coords : ndarray (T, S)
            UHI X coordinates in same coordinate system as MBES
        uhi_y_coords : ndarray (T, S)
            UHI Y coordinates in same coordinate system as MBES
        x_bounds : tuple, optional
            (x_min, x_max) to limit the region. If None, uses full UHI range
        y_bounds : tuple, optional
            (y_min, y_max) to limit the region. If None, uses full UHI range
        figsize : tuple, default=(8, 10)
            Figure size
        cmap : str, default="RdBu_r"
            Colormap for residuals
        vmax : float, optional
            Symmetric color limits
        title_suffix : str, optional
            Additional text for title
        buffer_m : float, default=0.0
            Buffer distance in meters to expand UHI footprint

        Returns:
        --------
        fig : matplotlib figure
        mask : ndarray
            Boolean mask showing which MBES pixels have UHI coverage
        """
        assert self.residuals is not None, "Run detrend() first."

        print(f"🎭 Creating masked MBES plot matching UHI footprint...")

        # Determine bounds
        if x_bounds is None:
            x_min, x_max = np.nanmin(uhi_x_coords), np.nanmax(uhi_x_coords)
        else:
            x_min, x_max = x_bounds

        if y_bounds is None:
            y_min, y_max = np.nanmin(uhi_y_coords), np.nanmax(uhi_y_coords)
        else:
            y_min, y_max = y_bounds

        # Add buffer
        x_min -= buffer_m
        x_max += buffer_m
        y_min -= buffer_m
        y_max += buffer_m

        # Find MBES pixel indices for this region
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

        # Extract subregion
        subregion = self.residuals[row_start:row_end, col_start:col_end].copy()
        sub_y_rows = self.y_rows[row_start:row_end]
        sub_x_cols = self.x_cols[col_start:col_end]

        print(f"   MBES subregion shape: {subregion.shape}")
        print(f"   MBES pixel size: {self.px_x:.4f} × {self.px_y:.4f} m")

        # Step 1: Get ALL UHI pixel positions (x, y)
        uhi_x_flat = uhi_x_coords.ravel()
        uhi_y_flat = uhi_y_coords.ravel()

        # Keep only valid UHI pixels (non-NaN)
        valid = np.isfinite(uhi_x_flat) & np.isfinite(uhi_y_flat)
        uhi_x_valid = uhi_x_flat[valid]
        uhi_y_valid = uhi_y_flat[valid]

        print(f"   UHI pixels to process: {len(uhi_x_valid):,}")

        # Step 2: Create mask - for each MBES pixel, check if ANY UHI pixel is there
        coverage_mask = np.zeros(subregion.shape, dtype=bool)
        half_px_x = self.px_x / 2
        half_px_y = self.px_y / 2

        # Step 3: For EACH UHI pixel, find which MBES pixel(s) it covers
        print(f"   Marking MBES pixels covered by UHI...")
        for x_uhi, y_uhi in zip(uhi_x_valid, uhi_y_valid):
            # Find nearest MBES pixel indices
            col_idx = np.argmin(np.abs(sub_x_cols - x_uhi))
            row_idx = np.argmin(np.abs(sub_y_rows - y_uhi))

            # Check if UHI pixel actually overlaps this MBES pixel
            x_center = sub_x_cols[col_idx]
            y_center = sub_y_rows[row_idx]

            if (
                abs(x_uhi - x_center) <= half_px_x
                and abs(y_uhi - y_center) <= half_px_y
            ):
                # YES - this MBES pixel has UHI coverage
                coverage_mask[row_idx, col_idx] = True

        covered = np.sum(coverage_mask)
        total = coverage_mask.size
        print(
            f"   Result: {covered:,}/{total:,} MBES pixels covered ({100*covered/total:.1f}%)"
        )

        # Apply mask to subregion
        masked_subregion = np.ma.masked_where(~coverage_mask, subregion)
        masked_subregion = np.ma.masked_invalid(masked_subregion)

        # Compute extent
        sub_extent = (
            sub_x_cols[0] - self.px_x / 2,
            sub_x_cols[-1] + self.px_x / 2,
            sub_y_rows[-1] - self.px_y / 2,
            sub_y_rows[0] + self.px_y / 2,
        )

        # Color limits
        if vmax is None:
            vmin, vmax = self._robust_sym_vlim(masked_subregion, q=0.98)
        else:
            vmin = -abs(vmax)
            vmax = abs(vmax)

        # Create figure
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

        # Settings info
        order = 2 if np.nanmax(np.abs(self.C_s)) > 0.001 else 1
        settings_str = f"order={order}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m"

        title = f"MBES Detrended Bathymetry - Masked to UHI Coverage\n{settings_str}"
        if title_suffix:
            title += f"\n{title_suffix}"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        cbar = fig.colorbar(im, ax=ax, label="Residual depth (m)")

        # Print statistics
        total_pixels = coverage_mask.size
        covered_pixels = np.sum(coverage_mask)
        valid_data = np.sum(np.isfinite(subregion) & coverage_mask)
        coverage_pct = 100 * covered_pixels / total_pixels if total_pixels > 0 else 0

        print(f"\n📊 Masked MBES Subregion:")
        print(f"   X range: [{x_min:.1f}, {x_max:.1f}] m")
        print(f"   Y range: [{y_min:.1f}, {y_max:.1f}] m")
        print(
            f"   UHI coverage: {coverage_pct:.1f}% ({covered_pixels:,}/{total_pixels:,} pixels)"
        )
        print(
            f"   Valid MBES data: {valid_data:,} pixels ({100*valid_data/covered_pixels:.1f}% of covered area)"
        )

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
    ):
        """
        Side-by-side comparison: full MBES residuals vs. masked to UHI footprint.
        Uses vectorized pixel mapping (no bounding box artifacts).

        Parameters:
        -----------
        uhi_x_coords : ndarray (T, S)
            UHI X coordinates in same CRS as MBES (UTM!)
        uhi_y_coords : ndarray (T, S)
            UHI Y coordinates in same CRS as MBES (UTM!)
        x_bounds : tuple, optional
            (x_min, x_max) to zoom into. If None, uses full UHI range + buffer
        y_bounds : tuple, optional
            (y_min, y_max) to zoom into. If None, uses full UHI range + buffer
        buffer_m : float, default=0.0
            Buffer distance in meters to expand view (not the mask)
        figsize : tuple, default=(16, 10)
            Figure size
        cmap : str, default="RdBu_r"
            Colormap for residuals
        vmax : float, optional
            Symmetric color limits
        title_suffix : str, optional
            Additional text for title

        Returns:
        --------
        fig : matplotlib figure
        mask_full : ndarray
            Boolean mask (full MBES grid) showing UHI coverage
        """
        assert self.residuals is not None, "Run detrend() first."

        print(f"\n🎭 Creating UHI footprint comparison...")

        # Build mask on FULL MBES grid using vectorized method
        print(f"   Building vectorized UHI footprint mask...")
        mask_full = self.uhi_footprint_mask(uhi_x_coords, uhi_y_coords, buffer_m=0.0)

        covered = np.sum(mask_full)
        total = mask_full.size
        print(
            f"   ✅ Mask created: {covered:,}/{total:,} MBES pixels covered ({100*covered/total:.2f}%)"
        )

        # Determine bounds for display
        if x_bounds is None:
            x_min, x_max = np.nanmin(uhi_x_coords), np.nanmax(uhi_x_coords)
        else:
            x_min, x_max = x_bounds

        if y_bounds is None:
            y_min, y_max = np.nanmin(uhi_y_coords), np.nanmax(uhi_y_coords)
        else:
            y_min, y_max = y_bounds

        # Add display buffer (not to mask)
        x_min -= buffer_m
        x_max += buffer_m
        y_min -= buffer_m
        y_max += buffer_m

        # Find MBES subregion for display
        col_mask = (self.x_cols >= x_min) & (self.x_cols <= x_max)
        row_mask = (self.y_rows >= y_min) & (self.y_rows <= y_max)
        col_indices = np.where(col_mask)[0]
        row_indices = np.where(row_mask)[0]

        if len(col_indices) == 0 or len(row_indices) == 0:
            print(f"⚠️  No MBES data in bounds")
            return None, mask_full

        col_start, col_end = col_indices[0], col_indices[-1] + 1
        row_start, row_end = row_indices[0], row_indices[-1] + 1

        # Extract subregions
        subregion_rect = self.residuals[row_start:row_end, col_start:col_end].copy()
        mask_sub = mask_full[row_start:row_end, col_start:col_end]

        # Compute extent
        sub_extent = (
            self.x_cols[col_start] - self.px_x / 2,
            self.x_cols[col_end - 1] + self.px_x / 2,
            self.y_rows[row_end - 1] - self.px_y / 2,
            self.y_rows[row_start] + self.px_y / 2,
        )

        # Color limits
        if vmax is None:
            vmin, vmax = self._robust_sym_vlim(subregion_rect, q=0.98)
        else:
            vmin, vmax = -abs(vmax), abs(vmax)

        # Create figure
        fig_comparison, axes = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True, num="UHI-MBES Comparison"
        )

        # LEFT: Full MBES subregion
        masked_full = np.ma.masked_invalid(subregion_rect)
        im1 = axes[0].imshow(
            masked_full,
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

        # RIGHT: Masked to UHI footprint (exact pixels)
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

        # Overall title
        order = 2 if np.nanmax(np.abs(self.C_s)) > 0.001 else 1
        settings_str = f"order={order}, base={self.smooth_baseline_m:.2f}m, tilt={self.smooth_tilt_m:.2f}m"
        title = f"MBES vs UHI Footprint Comparison\n{settings_str}"
        if title_suffix:
            title += f" | {title_suffix}"
        fig_comparison.suptitle(title, fontsize=14)

        # Statistics
        valid_mbes_in_mask = np.sum(np.isfinite(subregion_rect) & mask_sub)
        covered_sub = np.sum(mask_sub)
        print(f"\n📊 Comparison Stats:")
        print(
            f"   Display region: [{x_min:.1f}, {x_max:.1f}] × [{y_min:.1f}, {y_max:.1f}] m"
        )
        print(f"   UHI coverage in view: {covered_sub:,} pixels")
        print(f"   Valid MBES data in UHI footprint: {valid_mbes_in_mask:,} pixels")

        plt.show(block=False)
        return fig_comparison, mask_full


def extract_uhi_coordinates_from_cube(
    cube, track_start, track_end, coordinate_system="UTM", epsg_utm=None
):
    """
    Extract all UHI pixel coordinates (not just bounds) for masking MBES data.

    Parameters:
    -----------
    cube : CombinedTransectCube
        The UHI data cube from georef.py
    track_start : int
        Starting track index
    track_end : int
        Ending track index (exclusive)
    coordinate_system : str, default="UTM"
        Coordinate system ("UTM", "NED", or "ECEF")
    epsg_utm : int, optional
        EPSG code for UTM zone. If None, uses config.EPSG_MBES

    Returns:
    --------
    X_coords : ndarray (T, S)
        X coordinates for all UHI pixels in specified track range
    Y_coords : ndarray (T, S)
        Y coordinates for all UHI pixels in specified track range
    """
    import sys
    from pathlib import Path
    from pyproj import Transformer

    sys.path.append(str(Path(__file__).parent.parent))
    import config

    # Slice the coordinates
    if track_end is None:
        track_end = len(cube.X_ecef)

    X_ecef_slice = cube.X_ecef[track_start:track_end]
    Y_ecef_slice = cube.Y_ecef[track_start:track_end]
    Z_ecef_slice = cube.Z_ecef[track_start:track_end]

    if coordinate_system == "UTM":
        if epsg_utm is None:
            epsg_utm = config.EPSG_MBES

        print(
            f"   Converting {X_ecef_slice.size:,} UHI pixels ECEF → UTM (EPSG:{epsg_utm})..."
        )

        transformer = Transformer.from_crs(
            f"EPSG:{config.EPSG_ECEF}", f"EPSG:{epsg_utm}", always_xy=True
        )

        X_flat = X_ecef_slice.ravel()
        Y_flat = Y_ecef_slice.ravel()
        Z_flat = Z_ecef_slice.ravel()

        X_utm, Y_utm, Z_utm = transformer.transform(X_flat, Y_flat, Z_flat)

        X_coords = X_utm.reshape(X_ecef_slice.shape)
        Y_coords = Y_utm.reshape(Y_ecef_slice.shape)

    elif coordinate_system == "NED":
        from utils.georef import _ecef_to_ned_arrays

        print("   Converting ECEF to NED...")
        lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
        north, east, down = _ecef_to_ned_arrays(
            X_ecef_slice.ravel(),
            Y_ecef_slice.ravel(),
            Z_ecef_slice.ravel(),
            lat0,
            lon0,
            h0,
        )

        X_coords = east.reshape(X_ecef_slice.shape)
        Y_coords = north.reshape(Y_ecef_slice.shape)

    elif coordinate_system == "ECEF":
        print("   Using ECEF coordinates...")
        X_coords = X_ecef_slice
        Y_coords = Y_ecef_slice

    else:
        raise ValueError(f"Unknown coordinate_system: {coordinate_system}")

    return X_coords, Y_coords


def extract_uhi_bounds_from_cube(
    cube, track_start, track_end, coordinate_system="UTM", epsg_utm=None
):
    """
    Extract X, Y bounds from a UHI CombinedTransectCube for a specific track range.

    Parameters:
    -----------
    cube : CombinedTransectCube
        The UHI data cube from georef.py
    track_start : int
        Starting track index
    track_end : int
        Ending track index (exclusive)
    coordinate_system : str, default="UTM"
        Coordinate system ("UTM", "NED", or "ECEF")
    epsg_utm : int, optional
        EPSG code for UTM zone (e.g., 32632 for UTM Zone 32N). If None, uses config.EPSG_MBES

    Returns:
    --------
    x_bounds : tuple of (x_min, x_max)
        Easting bounds in the specified coordinate system
    y_bounds : tuple of (y_min, y_max)
        Northing bounds in the specified coordinate system
    """
    # Import here to avoid circular dependency
    import sys
    from pathlib import Path
    from pyproj import Transformer

    sys.path.append(str(Path(__file__).parent.parent))
    import config

    # Slice the coordinates
    if track_end is None:
        track_end = len(cube.X_ecef)

    X_ecef_slice = cube.X_ecef[track_start:track_end]
    Y_ecef_slice = cube.Y_ecef[track_start:track_end]
    Z_ecef_slice = cube.Z_ecef[track_start:track_end]

    if coordinate_system == "UTM":
        # Convert ECEF to UTM (same coordinate system as MBES)
        if epsg_utm is None:
            epsg_utm = config.EPSG_MBES

        print(f"   Converting ECEF to UTM (EPSG:{epsg_utm})...")

        # ECEF (EPSG:4978) -> UTM
        transformer = Transformer.from_crs(
            f"EPSG:{config.EPSG_ECEF}", f"EPSG:{epsg_utm}", always_xy=True
        )

        # Transform all points
        X_flat = X_ecef_slice.ravel()
        Y_flat = Y_ecef_slice.ravel()
        Z_flat = Z_ecef_slice.ravel()

        X_utm, Y_utm, Z_utm = transformer.transform(X_flat, Y_flat, Z_flat)

        X = X_utm.reshape(X_ecef_slice.shape)
        Y = Y_utm.reshape(Y_ecef_slice.shape)

    elif coordinate_system == "NED":
        from utils.georef import _ecef_to_ned_arrays

        # Convert ECEF to NED (local relative coordinates)
        print("   Converting ECEF to NED...")
        lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
        north, east, down = _ecef_to_ned_arrays(
            X_ecef_slice.ravel(),
            Y_ecef_slice.ravel(),
            Z_ecef_slice.ravel(),
            lat0,
            lon0,
            h0,
        )
        X = east.reshape(X_ecef_slice.shape)  # Easting
        Y = north.reshape(Y_ecef_slice.shape)  # Northing

    elif coordinate_system == "ECEF":
        X = X_ecef_slice
        Y = Y_ecef_slice

    else:
        raise ValueError(f"Unsupported coordinate_system: {coordinate_system}")

    # Get bounds
    x_min = np.nanmin(X)
    x_max = np.nanmax(X)
    y_min = np.nanmin(Y)
    y_max = np.nanmax(Y)

    return (x_min, x_max), (y_min, y_max)


if __name__ == "__main__":
    # Enable interactive mode to keep figures open
    plt.ion()

    # Load MBES data
    print("\n📂 Loading MBES GeoTIFF...")
    mb = MBESDetrender(
        r"E:\mjosa_new\DTM\geotiff_2.tif",
        crop_start_m=1.0,  # Crop 1m from start (bottom in image)
        crop_end_m=10.0,  # Crop 10m from end (top in image)
    ).load()

    print(f"   Shape: {mb.data.shape}")
    print(f"   Resolution: {mb.px_x:.4f} × {mb.px_y:.4f} m")
    if mb.crop_start_m > 0 or mb.crop_end_m > 0:
        print(
            f"   Along-track crop: {mb.crop_start_m:.1f}m from start, {mb.crop_end_m:.1f}m from end"
        )

    # Show data coverage diagnostic
    print("\n🔍 Generating data coverage diagnostic...")
    mb.plot_data_coverage()

    # Configure detrending parameters
    print("\n⚙️  Detrending settings:")
    params = {
        "order_x": 4,  # allow gentle cross-track curvature
        "smooth_baseline_m": 0.1,  # along-track smoothing of A(y)
        "smooth_tilt_m": 0.1,  # along-track smoothing of B(y)
        "smooth_center_m": 1,  # smooth the drifting swath center
        "central_frac": 0.8,  # fit central 80% of beams
        "robust": True,
    }
    for key, value in params.items():
        print(f"   {key:20s} = {value}")

    # Run detrending
    mb.detrend(**params)

    # Generate plots
    fig1 = mb.plot_triptych()

    fig2 = mb.plot_components()

    fig3 = mb.plot_residual_heatmap(figsize=(5, 12))

    # ========== OPTIONAL: Plot MBES subregion matching UHI data ==========
    # Uncomment this section to plot MBES data for the same area as your UHI plot
    """
    print("\n🗺️  Extracting MBES subregion to match UHI data...")
    
    # First, load your UHI cube (adjust paths as needed)
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import georef
    import config
    
    # Load UHI transect
    transect = georef.load_transect(config.OUTPUT_FOLDER)
    cube = transect.select_files([
        "rad_uhi_20241029_115057_4",
        "rad_uhi_20241029_115057_5",
    ])
    
    # Define the track range you're interested in (same as in your notebook)
    track_start = 3039
    track_end = 4029
    
    # Extract bounds from UHI data (using UTM coordinates - same as MBES)
    x_bounds, y_bounds = extract_uhi_bounds_from_cube(
        cube, 
        track_start, 
        track_end,
        coordinate_system="UTM",  # Use UTM to match MBES coordinate system
        epsg_utm=config.EPSG_MBES  # Usually 32632 (UTM Zone 32N)
    )
    
    # Plot MBES subregion matching this area
    fig4 = mb.plot_subregion_residuals(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        figsize=(8, 10),
        title_suffix=f"UHI tracks {track_start}-{track_end}"
    )
    """

    # Keep figures open
    plt.show(block=True)
