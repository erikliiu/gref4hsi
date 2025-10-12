"""
UHI Georeference RGB + UHI Mask + MBES Residuals
Four figures + a fifth: MBES cropped to UHI footprint. All shown together.
"""

import importlib
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import rasterio
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.stats import median_abs_deviation
from scipy.spatial import cKDTree  # used to crop MBES to UHI footprint

# Add parent directory to path if needed
sys.path.append(os.path.abspath("../"))

from utils import georef
import config

importlib.reload(georef)
from utils.georef import _ecef_to_ned_arrays


# ----------------------------
# Helpers
# ----------------------------
def slice_tracks(arr, start_idx, end_idx):
    if start_idx is None and end_idx is None:
        return arr
    s = start_idx if start_idx is not None else 0
    e = end_idx if end_idx is not None else arr.shape[0]
    return arr[s:e, :]


def build_full_valid_mask(X_ecef, Y_ecef, Z_ecef, R, G, B):
    coords_valid = np.isfinite(X_ecef) & np.isfinite(Y_ecef) & np.isfinite(Z_ecef)
    rgb_valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
    return coords_valid & rgb_valid


def pcolormesh_pad(X, Y):
    Xc = np.pad(X, ((0, 1), (0, 1)), mode="edge")
    Yc = np.pad(Y, ((0, 1), (0, 1)), mode="edge")
    return Xc, Yc


@contextmanager
def suppress_matplotlib_show():
    """Temporarily make plt.show() a no-op so earlier plots do not block."""
    _orig_show = plt.show
    try:
        plt.show = lambda *a, **k: None
        yield
    finally:
        plt.show = _orig_show


# ----------------------------
# Minimal MBES Detrending Class (with CRS capture)
# ----------------------------
def _pad_limits(ax, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True):
    xr = max(1e-12, x_max - x_min)
    yr = max(1e-12, y_max - y_min)
    ax.set_xlim(x_min - xr * pad_frac, x_max + xr * pad_frac)
    ax.set_ylim(y_min - yr * pad_frac)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


@dataclass
class MBESDetrender:
    """Minimal MBES detrending for residual plot only."""

    tif_path: str
    nodata_value: Optional[float] = 9999.0
    smooth_baseline_m: float = 0.5
    smooth_tilt_m: float = 0.5
    smooth_center_m: float = 1.0

    data: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    transform: Optional[object] = None
    extent: Optional[Tuple[float, float, float, float]] = None
    px_x: Optional[float] = None
    px_y: Optional[float] = None
    x_cols: Optional[np.ndarray] = None
    y_rows: Optional[np.ndarray] = None
    crs: Optional[object] = None  # store the actual CRS

    A_s: Optional[np.ndarray] = None
    B_s: Optional[np.ndarray] = None
    C_s: Optional[np.ndarray] = None
    trend: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    def load(self):
        with rasterio.open(self.tif_path) as src:
            z = src.read(1).astype(np.float64)
            tf = src.transform
            bounds = src.bounds
            self.crs = src.crs

        if self.nodata_value is not None:
            z = np.where(z == self.nodata_value, np.nan, z)

        self.valid_mask = np.isfinite(z)
        a, b, c, d, e, f = tf.a, tf.b, tf.c, tf.d, tf.e, tf.f
        H, W = z.shape
        cols = np.arange(W)
        rows = np.arange(H)
        self.x_cols = c + a * (cols + 0.5)  # 1D Eastings along columns
        self.y_rows = f + e * (rows + 0.5)  # 1D Northings along rows
        self.px_x = abs(a)
        self.px_y = abs(e)
        self.data = z
        self.transform = tf
        self.extent = (
            float(bounds.left),
            float(bounds.right),
            float(bounds.bottom),
            float(bounds.top),
        )
        return self

    def detrend(self, order_x=2, robust=True, central_frac=0.8, max_row_iters=3):
        """Simple row-wise polynomial detrend + along-track smoothing."""
        from scipy.signal import savgol_filter

        if self.data is None:
            self.load()

        z = self.data
        x = self.x_cols
        H, W = z.shape
        valid_rows = np.any(self.valid_mask, axis=1)
        valid_row_indices = np.where(valid_rows)[0]

        A = np.full(H, np.nan)
        B = np.full(H, 0.0)
        C = np.full(H, 0.0)

        # Per-row polynomial fit
        for i in valid_row_indices:
            row = z[i, :]
            m = np.isfinite(row)
            if m.sum() < 6:
                continue
            xv = x[m]
            yv = row[m]
            xc_row = np.median(xv)

            if 0 < central_frac < 1:
                lo = np.quantile(xv, 0.5 - 0.5 * central_frac)
                hi = np.quantile(xv, 0.5 + 0.5 * central_frac)
                keep = (xv >= lo) & (xv <= hi)
                xv, yv = xv[keep], yv[keep]

            xi = xv - xc_row
            deg = min(order_x, 2)
            coef = np.polyfit(xi, yv, deg)

            if deg == 0:
                a0, a1, a2 = float(coef[0]), 0.0, 0.0
            elif deg == 1:
                a1, a0 = float(coef[0]), float(coef[1])
                a2 = 0.0
            else:
                a2, a1, a0 = float(coef[0]), float(coef[1]), float(coef[2])

            if robust and deg > 0:
                for _ in range(max_row_iters):
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
            B[i] = a1
            C[i] = a2

        # Smooth along-track
        def smooth_1d(arr, scale_m):
            if scale_m <= 0 or self.px_y <= 0:
                return arr
            m = np.isfinite(arr)
            if m.sum() < 5:
                return arr
            w = max(3, int(np.round(scale_m / self.px_y)))
            if w % 2 == 0:
                w += 1
            w = min(w, m.sum() - 1 if m.sum() > 1 else 1)
            if w < 3:
                return arr
            result = arr.copy()
            result[m] = savgol_filter(arr[m], window_length=w, polyorder=2)
            return result

        self.A_s = smooth_1d(A, self.smooth_baseline_m)
        self.B_s = smooth_1d(B, self.smooth_tilt_m)
        self.C_s = smooth_1d(C, self.smooth_tilt_m)

        # Reconstruct trend
        trend = np.full_like(z, np.nan)
        for i in valid_row_indices:
            if not np.isfinite(self.A_s[i]):
                continue
            xc = np.median(x[np.isfinite(z[i, :])])
            xi = x - xc
            trend[i, :] = self.A_s[i] + self.B_s[i] * xi + self.C_s[i] * xi**2

        self.trend = trend
        self.residuals = z - trend
        return self

    def _extent_imshow(self):
        return (
            self.x_cols[0] - self.px_x / 2,
            self.x_cols[-1] + self.px_x / 2,
            self.y_rows[-1] - self.px_y / 2,
            self.y_rows[0] + self.px_y / 2,
        )

    def _robust_sym_vlim(self, arr, q=0.98):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return -1, 1
        vmax = np.quantile(np.abs(finite), q)
        return -vmax, vmax

    def _get_data_bounds(self, percentile=99.5):
        rows, cols = np.where(self.valid_mask)
        if len(rows) == 0:
            return self.extent
        p_low = (100 - percentile) / 2
        p_high = 100 - p_low
        x_min = np.percentile(self.x_cols[cols], p_low)
        x_max = np.percentile(self.x_cols[cols], p_high)
        y_min = np.percentile(self.y_rows[rows], p_low)
        y_max = np.percentile(self.y_rows[rows], p_high)
        return x_min, x_max, y_min, y_max

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
        """Plot residuals heatmap (optional standalone view)."""
        assert self.residuals is not None, "Run detrend() first."
        if vmax is None:
            vmin, vmax = self._robust_sym_vlim(self.residuals, q=0.98)
        else:
            vmin = -abs(vmax)
            vmax = abs(vmax)

        fig, ax = plt.subplots(1, 1, figsize=figsize, num="MBES Residuals")
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
        cbar = fig.colorbar(im, ax=ax, label="m")
        fig.tight_layout()
        return fig


# ----------------------------
# UHI-on-MBES overlay (both in NED)
# ----------------------------
def plot_uhi_mask_on_mbes(
    mbes_detrender,
    uhi_x_ned,
    uhi_y_ned,
    uhi_mask,
    origin_lat,
    origin_lon,
    origin_h,
    figsize=(20, 10),
    cmap_mbes="RdBu_r",
    mask_alpha=0.5,
    vmax_mbes=None,
    decimate=2,
):
    """
    Plot UHI validity points over detrended MBES residuals, both in NED.
    Returns NED grids (full resolution) for later cropping.
    """
    from pyproj import Transformer

    mb = mbes_detrender
    assert mb.residuals is not None, "Run detrend() on MBES first."
    if getattr(mb, "crs", None) is None:
        raise ValueError("MBESDetrender.crs is None. Did you call load()?")

    # MBES residual color range
    if vmax_mbes is None:
        vmin_mbes, vmax_mbes = mb._robust_sym_vlim(mb.residuals, q=0.98)
    else:
        vmin_mbes = -abs(vmax_mbes)
        vmax_mbes = abs(vmax_mbes)

    # Valid UHI points (NED)
    x_uhi_ned = uhi_x_ned[uhi_mask]
    y_uhi_ned = uhi_y_ned[uhi_mask]
    if len(x_uhi_ned) == 0:
        print("⚠️  No valid UHI points to overlay!")
        return None

    # Convert MBES grid from source CRS -> WGS84 -> ECEF -> NED
    H, W = mb.data.shape
    MBES_X_SRC, MBES_Y_SRC = np.meshgrid(mb.x_cols, mb.y_rows)

    tf_src_to_geo = Transformer.from_crs(mb.crs, "EPSG:4326", always_xy=True)
    mbes_lon, mbes_lat = tf_src_to_geo.transform(MBES_X_SRC, MBES_Y_SRC)

    tf_geo_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    mbes_x_ecef, mbes_y_ecef, mbes_z_ecef = tf_geo_to_ecef.transform(
        mbes_lon, mbes_lat, np.zeros_like(mbes_lon)
    )

    mbes_n_ned, mbes_e_ned, _ = _ecef_to_ned_arrays(
        mbes_x_ecef, mbes_y_ecef, mbes_z_ecef, origin_lat, origin_lon, origin_h
    )

    # Build corners grids for pcolormesh from center grids
    def corners_from_centers(Xc, Yc):
        Xp = np.pad(Xc, ((0, 1), (0, 1)), mode="edge")
        Yp = np.pad(Yc, ((0, 1), (0, 1)), mode="edge")
        dx = np.nanmedian(np.diff(Xc, axis=1)) if Xc.shape[1] > 1 else 0.0
        dy = np.nanmedian(np.diff(Yc, axis=0)) if Yc.shape[0] > 1 else 0.0
        if np.isfinite(dx):
            Xp[:, 1:-1] -= dx * 0.5
        if np.isfinite(dy):
            Yp[1:-1, :] -= dy * 0.5
        if np.isfinite(dx):
            Xp[:, 0] = Xp[:, 1] - dx
            Xp[:, -1] = Xp[:, -2] + dx
        if np.isfinite(dy):
            Yp[0, :] = Yp[1, :] - dy
            Yp[-1, :] = Yp[-2, :] + dy
        return Xp, Yp

    # Optional decimation for speed (for display only)
    dec = max(1, int(decimate))
    R_show = mb.residuals[::dec, ::dec]
    E_show = mbes_e_ned[::dec, ::dec]
    N_show = mbes_n_ned[::dec, ::dec]
    Ex, Nx = corners_from_centers(E_show, N_show)

    # Create figure with two panels
    fig, (ax_both, ax_separate) = plt.subplots(
        1, 2, figsize=figsize, num="UHI and MBES Comparison (Both in NED)"
    )

    # MBES residuals as mesh in NED
    im_mbes = ax_both.pcolormesh(
        Ex,
        Nx,
        np.ma.masked_invalid(R_show),
        shading="flat",
        cmap=cmap_mbes,
        vmin=vmin_mbes,
        vmax=vmax_mbes,
        alpha=0.7,
        zorder=-5,
    )

    # Overlay UHI (NED)
    ax_both.scatter(
        x_uhi_ned,
        y_uhi_ned,
        c="lime",
        s=0.5,
        alpha=0.6,
        edgecolors="none",
        label=f"UHI valid ({len(x_uhi_ned):,} pts)",
        zorder=10,
    )

    # Titles/labels
    ax_both.set_title("MBES Residuals + UHI Footprint (NED)")
    ax_both.set_xlabel("East (m)")
    ax_both.set_ylabel("North (m)")
    ax_both.legend(loc="upper right", fontsize=9)
    ax_both.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax_both.set_aspect("equal")

    # Colorbar
    fig.colorbar(im_mbes, ax=ax_both, label="MBES Residuals (m)", pad=0.02)

    # Right: UHI-only pane (same limits)
    ax_separate.scatter(
        x_uhi_ned,
        y_uhi_ned,
        c="green",
        s=1,
        alpha=0.7,
        label=f"UHI valid ({len(x_uhi_ned):,} pts)",
    )
    ax_separate.set_title("UHI Footprint Only (NED)")
    ax_separate.set_xlabel("East (m)")
    ax_separate.set_ylabel("North (m)")
    ax_separate.legend(loc="upper right")
    ax_separate.grid(True, alpha=0.3)
    ax_separate.set_aspect("equal")

    # Match limits
    ax_separate.set_xlim(ax_both.get_xlim())
    ax_separate.set_ylim(ax_both.get_ylim())

    fig.tight_layout()

    # Return full-resolution NED grids and residuals for cropping step
    return fig, mbes_e_ned, mbes_n_ned, mb.residuals


# ----------------------------
# NEW: Crop MBES to UHI footprint (KD-tree overlap)
# ----------------------------
def crop_mbes_to_uhi_footprint(
    mbes_e_ned,
    mbes_n_ned,
    mbes_residuals,
    uhi_e_ned,
    uhi_n_ned,
    uhi_mask,
    pixel_radius=None,
    decimate=1,
):
    """
    Keep MBES residuals only where they overlap the UHI footprint.

    We flag an MBES pixel as "covered by UHI" if at least one valid UHI point
    lies within 'pixel_radius' meters of the MBES pixel center.

    pixel_radius:
        If None, uses max(mbes pixel size) estimated from the NED grids.

    Returns:
        masked_residuals (H,W), E_grid (H,W), N_grid (H,W)
    """
    # Valid UHI XY
    uhi_x = uhi_e_ned[uhi_mask].ravel()
    uhi_y = uhi_n_ned[uhi_mask].ravel()
    if uhi_x.size == 0:
        raise ValueError("No valid UHI points — cannot crop MBES to footprint.")

    # Estimate pixel size from MBES NED grids
    def robust_pixel_size(G):
        # median abs diff across rows/cols
        diffs = []
        if G.shape[1] > 1:
            diffs.append(np.nanmedian(np.abs(np.diff(G, axis=1))))
        if G.shape[0] > 1:
            diffs.append(np.nanmedian(np.abs(np.diff(G, axis=0))))
        return float(np.nanmedian(diffs)) if diffs else 1.0

    if pixel_radius is None:
        px_e = robust_pixel_size(mbes_e_ned)
        px_n = robust_pixel_size(mbes_n_ned)
        pixel_radius = max(px_e, px_n)  # one pixel radius
    pixel_radius = float(pixel_radius)

    # KD-tree on UHI points
    tree = cKDTree(np.column_stack([uhi_x, uhi_y]))

    # Build mask for MBES pixels
    E = mbes_e_ned
    N = mbes_n_ned
    H, W = E.shape
    mn_mask = np.zeros((H, W), dtype=bool)

    # Optional decimation for speed in the search pass
    dec = max(1, int(decimate))
    for r in range(0, H, dec):
        # chunk columns in vectorized way
        pts = np.column_stack([E[r, ::dec], N[r, ::dec]])
        if pts.size == 0:
            continue
        # query: any UHI point within pixel_radius?
        idxs = tree.query_ball_point(pts, r=pixel_radius)
        m_row = np.array([len(ix) > 0 for ix in idxs], dtype=bool)
        mn_mask[r, ::dec] = m_row

    # Optionally dilate a bit so we don't miss bordering pixels (one ring)
    # (Comment out if you don't want it.)
    try:
        from scipy.ndimage import binary_dilation

        mn_mask = binary_dilation(mn_mask, iterations=1)
    except Exception:
        pass

    masked_residuals = np.where(mn_mask, mbes_residuals, np.nan)
    return masked_residuals, E, N


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("\n================  UHI + MBES → Four Figures + Cropped  ================\n")

    print(f"📁 Loading transect from: {config.OUTPUT_FOLDER}")
    transect = georef.load_transect(config.OUTPUT_FOLDER)
    transect.list_files()

    print("\n🔍 Selecting files...")
    cube = transect.select_files(
        [
            "rad_uhi_20241029_115057_4",
            "rad_uhi_20241029_115057_5",
        ]
    )
    cube.describe()

    # choose your desired slice
    track_start = 3039
    track_end = 4029
    print(f"\n📊 Track range: {track_start} to {track_end - 1}")

    # 1) Georeferenced RGB figure (NED)
    print("\n🎨 Preparing Plot 1: Georeferenced UHI RGB (NED)")
    with suppress_matplotlib_show():
        cube.plot_georef_rgb(
            coordinate_system="NED",
            origin=(config.LAT0, config.LON0, config.H0),
            show_file_boundaries=True,
            normalize=True,
            track_start=track_start,
            track_end=track_end,
        )

    # 2) Validity mask figure (NED)
    print("\n🎨 Preparing Plot 2: UHI validity mask (NED)")
    X = slice_tracks(cube.X_ecef, track_start, track_end)
    Y = slice_tracks(cube.Y_ecef, track_start, track_end)
    Z = slice_tracks(cube.Z_ecef, track_start, track_end)
    R = slice_tracks(cube.R, track_start, track_end)
    G = slice_tracks(cube.G, track_start, track_end)
    B = slice_tracks(cube.B, track_start, track_end)

    full_valid = build_full_valid_mask(X, Y, Z, R, G, B)

    lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
    N_arr, E_arr, D_arr = _ecef_to_ned_arrays(X, Y, Z, lat0, lon0, h0)

    Xc_mask, Yc_mask = pcolormesh_pad(E_arr, N_arr)
    disp = full_valid.astype(float)
    disp[~full_valid] = np.nan

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    im = ax2.pcolormesh(
        Xc_mask, Yc_mask, disp, shading="flat", cmap="Greens", vmin=0, vmax=1
    )
    ax2.set_title("UHI Validity Mask (NED)")
    ax2.set_xlabel("East (m)")
    ax2.set_ylabel("North (m)")
    ax2.set_aspect("equal", adjustable="box")
    plt.colorbar(im, ax=ax2, label="Valid = 1")

    # 3) MBES Residuals figure
    print("\n🎨 Preparing Plot 3: MBES Residuals (detrended bathymetry)")
    print(f"📂 Loading MBES GeoTIFF: {config.MBES_GEOTIFF}")

    mb = MBESDetrender(
        config.MBES_GEOTIFF,
        smooth_baseline_m=0.5,
        smooth_tilt_m=0.5,
        smooth_center_m=1.0,
    ).load()

    print(f"   MBES shape: {mb.data.shape}")
    print(f"   MBES resolution: {mb.px_x:.4f} × {mb.px_y:.4f} m")
    print(f"   MBES CRS: {mb.crs}")

    print("⚙️  Detrending MBES data...")
    mb.detrend(order_x=2, robust=True, central_frac=0.8)

    print("🎨 Creating MBES residuals plot...")
    with suppress_matplotlib_show():
        mb.plot_residual_heatmap(
            figsize=(5, 12), margin_frac=0.03, equal_aspect=True, bounds_percentile=99.5
        )

    # 4) UHI Mask overlaid on MBES Residuals (and get full-res NED grids back)
    print("\n🎨 Preparing Plot 4: UHI Mask on MBES Residuals (combined in NED)")
    with suppress_matplotlib_show():
        fig4, mbes_e_ned, mbes_n_ned, mbes_residuals = plot_uhi_mask_on_mbes(
            mbes_detrender=mb,
            uhi_x_ned=E_arr,
            uhi_y_ned=N_arr,
            uhi_mask=full_valid,
            origin_lat=lat0,
            origin_lon=lon0,
            origin_h=h0,
            figsize=(20, 10),
            mask_alpha=0.5,
            decimate=2,
        )

    # 5) NEW: Crop MBES to UHI footprint and plot in a new figure
    print("\n✂️  Cropping MBES residuals to UHI footprint...")
    mbes_residuals_uhi, E_grid, N_grid = crop_mbes_to_uhi_footprint(
        mbes_e_ned,
        mbes_n_ned,
        mbes_residuals,
        E_arr,
        N_arr,
        full_valid,
        pixel_radius=None,  # auto: 1× MBES pixel
        decimate=2,
    )

    # Build pcolormesh corners
    def corners_from_centers(Xc, Yc):
        Xp = np.pad(Xc, ((0, 1), (0, 1)), mode="edge")
        Yp = np.pad(Yc, ((0, 1), (0, 1)), mode="edge")
        dx = np.nanmedian(np.diff(Xc, axis=1)) if Xc.shape[1] > 1 else 0.0
        dy = np.nanmedian(np.diff(Yc, axis=0)) if Yc.shape[0] > 1 else 0.0
        if np.isfinite(dx):
            Xp[:, 1:-1] -= dx * 0.5
        if np.isfinite(dy):
            Yp[1:-1, :] -= dy * 0.5
        if np.isfinite(dx):
            Xp[:, 0] = Xp[:, 1] - dx
            Xp[:, -1] = Xp[:, -2] + dx
        if np.isfinite(dy):
            Yp[0, :] = Yp[1, :] - dy
            Yp[-1, :] = Yp[-2, :] + dy
        return Xp, Yp

    Ex_c, Nx_c = corners_from_centers(E_grid, N_grid)
    vmin_c, vmax_c = mb._robust_sym_vlim(mbes_residuals_uhi, q=0.98)

    fig5, ax5 = plt.subplots(
        1, 1, figsize=(10, 8), num="MBES Residuals within UHI Footprint (NED)"
    )
    im5 = ax5.pcolormesh(
        Ex_c,
        Nx_c,
        np.ma.masked_invalid(mbes_residuals_uhi),
        shading="flat",
        cmap="RdBu_r",
        vmin=vmin_c,
        vmax=vmax_c,
    )
    ax5.set_title("MBES Residuals within UHI Footprint (NED)")
    ax5.set_xlabel("East (m)")
    ax5.set_ylabel("North (m)")
    ax5.set_aspect("equal")
    plt.colorbar(im5, ax=ax5, label="MBES Residuals (m)")
    fig5.tight_layout()

    # Layout where possible
    try:
        fig2.tight_layout()
    except Exception:
        pass

    # Show everything
    plt.show()

    print(
        "\n================  ✅ All figures (incl. cropped MBES) shown  ================\n"
    )
