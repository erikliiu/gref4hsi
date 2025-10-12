"""
UHI Georeference RGB + UHI Mask + MBES Residuals
Three separate figure windows, opened together.
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

# Add parent directory to path
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
# Minimal MBES Detrending Class (extracted from dev3_mbes3.py)
# ----------------------------
def _pad_limits(ax, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True):
    xr = max(1e-12, x_max - x_min)
    yr = max(1e-12, y_max - y_min)
    ax.set_xlim(x_min - xr * pad_frac, x_max + xr * pad_frac)
    ax.set_ylim(y_min - yr * pad_frac, y_max + yr * pad_frac)
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
        """Plot residuals heatmap (from dev3_mbes3.py)."""
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


def plot_uhi_mask_on_mbes(
    mbes_detrender,
    uhi_x_ned,
    uhi_y_ned,
    uhi_mask,
    origin_lat,
    origin_lon,
    origin_h,
    figsize=(10, 14),
    cmap_mbes="RdBu_r",
    cmap_mask="Greens",
    mask_alpha=0.5,
    vmax_mbes=None,
    crop_to_data=True,
    bounds_percentile=99.5,
    margin_frac=0.03,
    equal_aspect=True,
):
    """
    Plot UHI validity mask and detrended MBES data side by side or overlaid.

    Parameters:
    -----------
    mbes_detrender : MBESDetrender
        Detrended MBES data object
    uhi_x_ned : np.ndarray
        UHI X coordinates in NED (East), shape (T, S)
    uhi_y_ned : np.ndarray
        UHI Y coordinates in NED (North), shape (T, S)
    uhi_mask : np.ndarray
        UHI validity mask, shape (T, S)
    origin_lat, origin_lon, origin_h : float
        Origin of NED coordinate system
    """
    from scipy.ndimage import binary_dilation
    from pyproj import Transformer

    assert mbes_detrender.residuals is not None, "Run detrend() on MBES first."

    mb = mbes_detrender

    # Get MBES residuals color range
    if vmax_mbes is None:
        vmin_mbes, vmax_mbes = mb._robust_sym_vlim(mb.residuals, q=0.98)
    else:
        vmin_mbes = -abs(vmax_mbes)
        vmax_mbes = abs(vmax_mbes)

    # Get valid UHI points
    x_uhi_ned = uhi_x_ned[uhi_mask]  # NED East
    y_uhi_ned = uhi_y_ned[uhi_mask]  # NED North

    if len(x_uhi_ned) == 0:
        print("⚠️  No valid UHI points to overlay!")
        return None

    print(f"   UHI valid points: {len(x_uhi_ned):,}")
    print(
        f"   UHI X (NED East) range: [{np.min(x_uhi_ned):.2f}, {np.max(x_uhi_ned):.2f}] m"
    )
    print(
        f"   UHI Y (NED North) range: [{np.min(y_uhi_ned):.2f}, {np.max(y_uhi_ned):.2f}] m"
    )

    H, W = mb.data.shape
    print(f"   MBES X range: [{mb.x_cols[0]:.2f}, {mb.x_cols[-1]:.2f}]")
    print(f"   MBES Y range: [{mb.y_rows[-1]:.2f}, {mb.y_rows[0]:.2f}]")

    # Convert UHI from NED to UTM
    print("   🔄 Converting UHI from NED to UTM...")

    # Convert origin to ECEF
    tf_geo_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0_ecef, y0_ecef, z0_ecef = tf_geo_to_ecef.transform(
        origin_lon, origin_lat, origin_h
    )

    # NED to ECEF transformation
    lat_rad = np.radians(origin_lat)
    lon_rad = np.radians(origin_lon)
    sL, cL = np.sin(lat_rad), np.cos(lat_rad)
    sO, cO = np.sin(lon_rad), np.cos(lon_rad)

    # NED components
    N = y_uhi_ned  # North
    E = x_uhi_ned  # East
    D = np.zeros_like(N)  # Down = 0 (assume surface)

    # NED to delta-ECEF
    dx = -sL * cO * N - sO * E - cL * cO * D
    dy = -sL * sO * N + cO * E - cL * sO * D
    dz = cL * N - sL * D

    uhi_x_ecef = x0_ecef + dx
    uhi_y_ecef = y0_ecef + dy
    uhi_z_ecef = z0_ecef + dz

    # ECEF to Geographic
    tf_ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    uhi_lon, uhi_lat, uhi_h = tf_ecef_to_geo.transform(
        uhi_x_ecef, uhi_y_ecef, uhi_z_ecef
    )

    # Geographic to UTM
    tf_geo_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    uhi_x_utm, uhi_y_utm = tf_geo_to_utm.transform(uhi_lon, uhi_lat)

    print(f"   UHI X (UTM) range: [{np.min(uhi_x_utm):.2f}, {np.max(uhi_x_utm):.2f}]")
    print(f"   UHI Y (UTM) range: [{np.min(uhi_y_utm):.2f}, {np.max(uhi_y_utm):.2f}]")

    # ======== BETTER APPROACH: Convert MBES from UTM to NED (to match UHI) ========
    print("   🔄 Converting MBES grid from UTM to NED...")

    # Get MBES grid coordinates in UTM
    H, W = mb.data.shape
    mbes_x_utm = mb.x_cols  # Easting (1D array)
    mbes_y_utm = mb.y_rows  # Northing (1D array)

    # Create 2D grid
    MBES_X_UTM, MBES_Y_UTM = np.meshgrid(mbes_x_utm, mbes_y_utm)

    # UTM to Geographic (lat/lon)
    tf_utm_to_geo = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    mbes_lon, mbes_lat = tf_utm_to_geo.transform(MBES_X_UTM, MBES_Y_UTM)

    # Geographic to ECEF
    tf_geo_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    mbes_x_ecef, mbes_y_ecef, mbes_z_ecef = tf_geo_to_ecef.transform(
        mbes_lon, mbes_lat, np.zeros_like(mbes_lon)
    )

    # ECEF to NED using the helper function from georef
    from utils.georef import _ecef_to_ned_arrays

    mbes_n_ned, mbes_e_ned, mbes_d_ned = _ecef_to_ned_arrays(
        mbes_x_ecef, mbes_y_ecef, mbes_z_ecef, origin_lat, origin_lon, origin_h
    )

    print(
        f"   MBES X (NED East) range: [{np.min(mbes_e_ned):.2f}, {np.max(mbes_e_ned):.2f}] m"
    )
    print(
        f"   MBES Y (NED North) range: [{np.min(mbes_n_ned):.2f}, {np.max(mbes_n_ned):.2f}] m"
    )

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(
        1, 2, figsize=(20, 10), num="UHI and MBES Comparison (Both in NED)"
    )
    ax_both, ax_separate = axes

    # ========== LEFT: MBES + UHI overlay in NED ==========
    # Create extent for MBES in NED coordinates
    extent_ned = [
        np.min(mbes_e_ned),
        np.max(mbes_e_ned),  # East min, max
        np.min(mbes_n_ned),
        np.max(mbes_n_ned),  # North min, max
    ]

    im_mbes = ax_both.imshow(
        np.ma.masked_invalid(mb.residuals),
        extent=extent_ned,
        origin="upper",
        cmap=cmap_mbes,
        vmin=vmin_mbes,
        vmax=vmax_mbes,
        alpha=0.7,
    )

    # Overlay UHI points (already in NED)
    ax_both.scatter(
        x_uhi_ned,
        y_uhi_ned,
        c="lime",
        s=0.5,
        alpha=0.6,
        label=f"UHI valid ({len(x_uhi_ned):,} pts)",
        edgecolors="none",
    )

    # Title and labels
    order = 2 if (mb.C_s is not None and np.nanmax(np.abs(mb.C_s)) > 0.001) else 1
    settings_str = (
        f"order={order}, base={mb.smooth_baseline_m:.2f}m, tilt={mb.smooth_tilt_m:.2f}m"
    )
    ax_both.set_title(
        f"MBES Residuals + UHI Footprint\n{settings_str}\nCoordinate System: NED (local meters)",
        fontsize=11,
    )
    ax_both.set_xlabel("East (m)")
    ax_both.set_ylabel("North (m)")
    ax_both.legend(loc="upper right", fontsize=9)
    ax_both.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax_both.set_aspect("equal")

    # Colorbar for MBES
    cbar_mbes = fig.colorbar(im_mbes, ax=ax_both, label="MBES Residuals (m)", pad=0.02)

    # Stats text
    n_mbes_total = np.isfinite(mb.residuals).sum()
    stats_both = (
        f"MBES pixels: {n_mbes_total:,}\n"
        f"UHI points: {len(x_uhi_ned):,}\n"
        f"Coordinate system: NED"
    )
    ax_both.text(
        0.02,
        0.98,
        stats_both,
        transform=ax_both.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ========== RIGHT: UHI only (for comparison) ==========
    ax_separate.scatter(
        x_uhi_ned,
        y_uhi_ned,
        c="green",
        s=1,
        alpha=0.7,
        label=f"UHI valid ({len(x_uhi_ned):,} pts)",
    )

    ax_separate.set_title("UHI Footprint Only\nNED (local meters)", fontsize=11)
    ax_separate.set_xlabel("East (m)")
    ax_separate.set_ylabel("North (m)")
    ax_separate.legend(loc="upper right")
    ax_separate.grid(True, alpha=0.3)
    ax_separate.set_aspect("equal")

    # Match axes limits
    ax_separate.set_xlim(ax_both.get_xlim())
    ax_separate.set_ylim(ax_both.get_ylim())

    fig.tight_layout()
    return fig


# ----------------------------
# Main
# ----------------------------
print("\n================  UHI + MBES → Four Figures, One Show  ================\n")

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

track_start = 3039
track_end = 4029
print(f"\n📊 Track range: {track_start} to {track_end - 1}")

# 1) Georeferenced RGB figure
print("\n🎨 Preparing Plot 1: Georeferenced UHI RGB (NED)")
# If cube.plot_georef_rgb would normally call plt.show(), suppress it here.
with suppress_matplotlib_show():
    # If your function supports 'show=False' or 'ax=...', use that instead.
    cube.plot_georef_rgb(
        coordinate_system="NED",
        origin=(config.LAT0, config.LON0, config.H0),
        show_file_boundaries=True,
        normalize=True,
        track_start=track_start,
        track_end=track_end,
    )

# 2) Validity mask figure
print("\n🎨 Preparing Plot 2: UHI validity mask (NED)")
X = slice_tracks(cube.X_ecef, track_start, track_end)
Y = slice_tracks(cube.Y_ecef, track_start, track_end)
Z = slice_tracks(cube.Z_ecef, track_start, track_end)
R = slice_tracks(cube.R, track_start, track_end)
G = slice_tracks(cube.G, track_start, track_end)
B = slice_tracks(cube.B, track_start, track_end)

full_valid = build_full_valid_mask(X, Y, Z, R, G, B)

lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
N, E, D = _ecef_to_ned_arrays(X, Y, Z, lat0, lon0, h0)

Xc, Yc = pcolormesh_pad(E, N)
disp = full_valid.astype(float)
disp[~full_valid] = np.nan

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
im = ax2.pcolormesh(Xc, Yc, disp, shading="flat", cmap="Greens", vmin=0, vmax=1)
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

print("⚙️  Detrending MBES data...")
mb.detrend(order_x=2, robust=True, central_frac=0.8)

print("🎨 Creating MBES residuals plot...")
with suppress_matplotlib_show():
    mb.plot_residual_heatmap(
        figsize=(5, 12), margin_frac=0.03, equal_aspect=True, bounds_percentile=99.5
    )

# 4) UHI Mask overlaid on MBES Residuals
print("\n🎨 Preparing Plot 4: UHI Mask on MBES Residuals (combined)")
with suppress_matplotlib_show():
    plot_uhi_mask_on_mbes(
        mbes_detrender=mb,
        uhi_x_ned=E,
        uhi_y_ned=N,
        uhi_mask=full_valid,
        origin_lat=lat0,
        origin_lon=lon0,
        origin_h=h0,
        figsize=(8, 12),
        mask_alpha=0.5,
        crop_to_data=True,
        bounds_percentile=99.5,
        margin_frac=0.03,
    )

# Apply tight_layout only to figures that don't use constrained_layout
# Figure 2 (UHI mask) can use tight_layout
try:
    fig2.tight_layout()
except:
    pass

# Show all open figures together
plt.show()

print("\n================  ✅ All four figures opened together  ================\n")
