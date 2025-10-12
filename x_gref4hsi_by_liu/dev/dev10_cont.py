"""
UHI Georeference RGB + UHI Mask + MBES Residuals
Original four figures unchanged + NEW full-res MBES cropped by jagged UHI footprint.
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
# Helpers (UNCHANGED for the first four figures)
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
# Minimal MBES Detrending Class (UNCHANGED)
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
        """Plot residuals heatmap (unchanged)."""
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
# (UNCHANGED) Simple overlay that you already had
# ----------------------------
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
    NOTE: This function is unchanged so Figures 1–4 look exactly as before.
    """
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

    # ----- MBES grid UTM -> NED (for plotting extent only) -----
    tf_utm_to_geo = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    MBES_X_UTM, MBES_Y_UTM = np.meshgrid(mb.x_cols, mb.y_rows)
    mbes_lon, mbes_lat = tf_utm_to_geo.transform(MBES_X_UTM, MBES_Y_UTM)

    tf_geo_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    mbes_x_ecef, mbes_y_ecef, mbes_z_ecef = tf_geo_to_ecef.transform(
        mbes_lon, mbes_lat, np.zeros_like(mbes_lon)
    )
    mbes_n_ned, mbes_e_ned, _ = _ecef_to_ned_arrays(
        mbes_x_ecef, mbes_y_ecef, mbes_z_ecef, origin_lat, origin_lon, origin_h
    )

    # ----- Figure with overlay (unchanged) -----
    fig, axes = plt.subplots(
        1, 2, figsize=(20, 10), num="UHI and MBES Comparison (Both in NED)"
    )
    ax_both, ax_separate = axes

    extent_ned = [
        np.min(mbes_e_ned),
        np.max(mbes_e_ned),
        np.min(mbes_n_ned),
        np.max(mbes_n_ned),
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
    ax_both.scatter(
        x_uhi_ned,
        y_uhi_ned,
        c="lime",
        s=0.5,
        alpha=0.6,
        label=f"UHI valid ({len(x_uhi_ned):,} pts)",
        edgecolors="none",
    )
    ax_both.set_title("MBES Residuals + UHI Footprint (NED)")
    ax_both.set_xlabel("East (m)")
    ax_both.set_ylabel("North (m)")
    ax_both.legend(loc="upper right", fontsize=9)
    ax_both.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax_both.set_aspect("equal")

    fig.colorbar(im_mbes, ax=ax_both, label="MBES Residuals (m)", pad=0.02)

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
    ax_separate.set_xlim(ax_both.get_xlim())
    ax_separate.set_ylim(ax_both.get_ylim())

    fig.tight_layout()
    # return NED grids for optional use (we won't rely on this to avoid changing earlier behavior)
    return fig, (mbes_e_ned, mbes_n_ned)


# =====================================================================
# NEW STUFF **ONLY FOR FIGURE 5**  —  Pixel-accurate jagged UHI clip
# =====================================================================


def _points_in_quad_mask(Xm_sub, Ym_sub, qx, qy):
    """Return boolean mask for points in convex quad (qx,qy length 4) using sign-of-area."""

    def _in_tri(ax, ay, bx, by, cx, cy, px, py):
        s1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        s2 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
        s3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
        neg = (s1 < 0) | (s2 < 0) | (s3 < 0)
        pos = (s1 > 0) | (s2 > 0) | (s3 > 0)
        return ~(neg & pos)  # all same sign or zero

    in1 = _in_tri(qx[0], qy[0], qx[1], qy[1], qx[2], qy[2], Xm_sub, Ym_sub)
    in2 = _in_tri(qx[0], qy[0], qx[2], qy[2], qx[3], qy[3], Xm_sub, Ym_sub)
    return in1 | in2


def figure5_fullres_mbes_cropped_by_jagged_uhi(
    mb: MBESDetrender,
    origin_lat: float,
    origin_lon: float,
    origin_h: float,
    uhi_E: np.ndarray,
    uhi_N: np.ndarray,
    uhi_valid: np.ndarray,
    fig_name="MBES Full-Res within UHI Footprint (Pixel-Accurate)",
):
    """
    Build a full-res mask on the MBES grid by rasterizing each UHI pcolormesh cell
    (true quad in NED) — preserves jagged/ratchety outline exactly.
    """
    from pyproj import Transformer

    assert mb.residuals is not None, "Run detrend() first."

    # MBES grid (UTM -> NED) for coordinates of each MBES pixel center
    tf_utm_to_geo = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    MBES_X_UTM, MBES_Y_UTM = np.meshgrid(mb.x_cols, mb.y_rows)
    mbes_lon, mbes_lat = tf_utm_to_geo.transform(MBES_X_UTM, MBES_Y_UTM)

    tf_geo_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    mbes_x_ecef, mbes_y_ecef, mbes_z_ecef = tf_geo_to_ecef.transform(
        mbes_lon, mbes_lat, np.zeros_like(mbes_lon)
    )
    mbes_n_ned, mbes_e_ned, _ = _ecef_to_ned_arrays(
        mbes_x_ecef, mbes_y_ecef, mbes_z_ecef, origin_lat, origin_lon, origin_h
    )

    # UHI cell corners (quads) from pcolormesh edges
    Ec, Nc = pcolormesh_pad(uhi_E, uhi_N)  # (T+1, S+1)
    T, S = uhi_E.shape

    H, W = mb.data.shape
    mask_mbes = np.zeros((H, W), dtype=bool)

    cols_E = mbes_e_ned[0, :]  # eastings across columns
    rows_N = mbes_n_ned[:, 0]  # northings across rows

    def _range_idx(arr1d, vmin, vmax):
        lo, hi = (vmin, vmax) if vmin <= vmax else (vmax, vmin)
        if arr1d[0] <= arr1d[-1]:
            idx = np.where((arr1d >= lo) & (arr1d <= hi))[0]
        else:
            idx = np.where((arr1d <= hi) & (arr1d >= lo))[0]
        return idx

    valid_cells = np.argwhere(uhi_valid)
    if valid_cells.size == 0:
        print("⚠️  No valid UHI cells to rasterize. Skipping Figure 5.")
        return None

    print(f"Figure 5: Rasterizing {len(valid_cells):,} UHI cells to MBES grid...")

    # Loop through valid UHI cells: bbox prefilter then exact point-in-quad
    for i, j in valid_cells:
        qx = np.array(
            [Ec[i, j], Ec[i, j + 1], Ec[i + 1, j + 1], Ec[i + 1, j]], dtype=float
        )
        qy = np.array(
            [Nc[i, j], Nc[i, j + 1], Nc[i + 1, j + 1], Nc[i + 1, j]], dtype=float
        )

        emin, emax = np.min(qx), np.max(qx)
        nmin, nmax = np.min(qy), np.max(qy)

        cols = _range_idx(cols_E, emin, emax)
        rows = _range_idx(rows_N, nmin, nmax)
        if cols.size == 0 or rows.size == 0:
            continue

        Xm_sub = mbes_e_ned[np.ix_(rows, cols)]
        Ym_sub = mbes_n_ned[np.ix_(rows, cols)]
        inside = _points_in_quad_mask(Xm_sub, Ym_sub, qx, qy)
        if inside.any():
            mask_mbes[np.ix_(rows, cols)] |= inside

    cropped = np.where(mask_mbes, mb.residuals, np.nan)

    vmin, vmax = mb._robust_sym_vlim(mb.residuals, q=0.98)
    extent_ned = [
        float(np.min(mbes_e_ned)),
        float(np.max(mbes_e_ned)),
        float(np.min(mbes_n_ned)),
        float(np.max(mbes_n_ned)),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(7, 12), num=fig_name)
    im = ax.imshow(
        np.ma.masked_invalid(cropped),
        extent=extent_ned,
        origin="upper",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        "MBES residuals (FULL resolution)\nclipped by jagged UHI footprint (NED)"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, label="MBES Residuals (m)")
    fig.tight_layout()
    print("✓ Figure 5 ready (full-res MBES with exact jagged UHI outline).")
    return fig


# ----------------------------
# Main (FIRST FOUR FIGURES UNCHANGED)
# ----------------------------
print(
    "\n================  UHI + MBES → Four Figures + NEW Jagged Full-Res  ================\n"
)

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

# 1) Georeferenced RGB figure (unchanged)
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

# 2) Validity mask figure (unchanged)
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

# 3) MBES Residuals figure (unchanged)
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

# 4) UHI Mask overlaid on MBES Residuals (unchanged)
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

# 5) NEW: full-resolution MBES clipped by the exact jagged UHI footprint
print("\n🧩 Preparing Plot 5: Full-resolution MBES within jagged UHI footprint")
with suppress_matplotlib_show():
    figure5_fullres_mbes_cropped_by_jagged_uhi(
        mb=mb,
        origin_lat=lat0,
        origin_lon=lon0,
        origin_h=h0,
        uhi_E=E,
        uhi_N=N,
        uhi_valid=full_valid,
        fig_name="MBES Full-Res within UHI Footprint (Pixel-Accurate)",
    )

# Tighten only the simple mask fig (unchanged behavior)
try:
    fig2.tight_layout()
except Exception:
    pass

# Show all open figures together
plt.show()

print(
    "\n================  ✅ All figures opened (first four unchanged; Figure 5 added)  ================\n"
)
