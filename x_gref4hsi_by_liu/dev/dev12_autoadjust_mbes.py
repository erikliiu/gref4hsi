# alignment_module.py
# UHI ↔ MBES: loading, jagged footprint, metrics, and alignment search (coarse→fine)

from __future__ import annotations
import os, json, hashlib, csv, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation
from pyproj import Transformer
import rasterio

# ---- Add parent directory to path for imports ----
sys.path.append(os.path.abspath("../"))

# ---- repo modules (assumed importable in your env) ----
from utils import georef
import config
from utils.georef import _ecef_to_ned_arrays


# -----------------------------
# Helpers (robust stats & plots)
# -----------------------------
def robust_z(a: np.ndarray) -> np.ndarray:
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.full_like(a, np.nan, float)
    m = np.nanmedian(a)
    mad = median_abs_deviation(a[finite], scale="normal")
    if not np.isfinite(mad) or mad == 0:
        mad = 1.0
    return (a - m) / mad


def robust_sym_vlim(arr: np.ndarray, q: float = 0.98) -> Tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = np.quantile(np.abs(finite), q)
    return -float(vmax), float(vmax)


def pcolormesh_pad(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xc = np.pad(X, ((0, 1), (0, 1)), mode="edge")
    Yc = np.pad(Y, ((0, 1), (0, 1)), mode="edge")
    return Xc, Yc


def pad_limits(ax, x_min, x_max, y_min, y_max, pad_frac=0.03, equal_aspect=True):
    xr = max(1e-12, x_max - x_min)
    yr = max(1e-12, y_max - y_min)
    ax.set_xlim(x_min - xr * pad_frac, x_max + xr * pad_frac)
    ax.set_ylim(y_min - yr * pad_frac, y_max + yr * pad_frac)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def build_full_valid(X, Y, Z, R, G, B) -> np.ndarray:
    return (
        np.isfinite(X)
        & np.isfinite(Y)
        & np.isfinite(Z)
        & np.isfinite(R)
        & np.isfinite(G)
        & np.isfinite(B)
    )


def boundary_segments_from_mask(mask: np.ndarray, Xc: np.ndarray, Yc: np.ndarray):
    """
    EXACT jagged outline (no smoothing) including outer edges.
    Returns (segs_xy, segs_idx).
    """
    T, S = mask.shape
    segs_xy, segs_idx = [], []

    # internal vertical edges
    vdiff = mask[:, 1:] != mask[:, :-1]
    iv, jv = np.nonzero(vdiff)
    jline = jv + 1
    for i, j in zip(iv, jline):
        a = (i, j)
        b = (i + 1, j)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # outer left (j=0) and right (j=S)
    i_left = np.nonzero(mask[:, 0])[0]
    for i in i_left:
        a = (i, 0)
        b = (i + 1, 0)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    i_right = np.nonzero(mask[:, -1])[0]
    for i in i_right:
        a = (i, S)
        b = (i + 1, S)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # internal horizontal edges
    hdiff = mask[1:, :] != mask[:-1, :]
    ih, jh = np.nonzero(hdiff)
    iline = ih + 1
    for i, j in zip(iline, jh):
        a = (i, j)
        b = (i, j + 1)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # outer top/bottom
    j_top = np.nonzero(mask[0, :])[0]
    for j in j_top:
        a = (0, j)
        b = (0, j + 1)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    j_bot = np.nonzero(mask[-1, :])[0]
    for j in j_bot:
        a = (T, j)
        b = (T, j + 1)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    return np.asarray(segs_xy, float), segs_idx


def trace_loops_from_segments(segs_idx):
    adj = {}
    for a, b in segs_idx:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    unvisited = set(frozenset((a, b)) for a, b in segs_idx)
    loops = []
    while unvisited:
        e = unvisited.pop()
        a, b = tuple(e)
        loop = [a, b]
        prev, cur = a, b
        while True:
            nbrs = adj.get(cur, [])
            nxt = next((n for n in nbrs if n != prev), None)
            if nxt is None:
                break
            edge = frozenset((cur, nxt))
            if edge in unvisited:
                unvisited.remove(edge)
            loop.append(nxt)
            prev, cur = cur, nxt
            if nxt == loop[0]:
                break
        loops.append(loop)
        for i in range(len(loop) - 1):
            unvisited.discard(frozenset((loop[i], loop[i + 1])))
    return loops


def compound_path_from_loops(loops, Xc, Yc) -> MplPath:
    paths = []
    for ring in loops:
        xs = [Xc[idx] for idx in ring]
        ys = [Yc[idx] for idx in ring]
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xs.append(xs[0])
            ys.append(ys[0])
        verts = np.column_stack([xs, ys])
        codes = np.full(len(verts), MplPath.LINETO, dtype=MplPath.code_type)
        codes[0] = MplPath.MOVETO
        codes[-1] = MplPath.CLOSEPOLY
        paths.append(MplPath(verts, codes))
    return paths[0] if len(paths) == 1 else MplPath.make_compound_path(*paths)


# -----------------------------
# Minimal MBES detrending
# -----------------------------
@dataclass
class MBESDetrender:
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
    epsg: Optional[int] = None
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
            self.epsg = src.crs.to_epsg() if src.crs else None
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
        if self.data is None:
            self.load()
        z = self.data
        x = self.x_cols
        H, W = z.shape
        valid_rows = np.any(self.valid_mask, axis=1)
        idxs = np.where(valid_rows)[0]
        A = np.full(H, np.nan)
        B = np.full(H, 0.0)
        C = np.full(H, 0.0)

        for i in idxs:
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
                    beta = np.linalg.lstsq((X * w).T, (yv * w), rcond=None)[0]
                    if deg == 2:
                        a0, a1, a2 = float(beta[0]), float(beta[1]), float(beta[2])
                    else:
                        a0, a1 = float(beta[0]), float(beta[1])
                        a2 = 0.0
            A[i], B[i], C[i] = a0, a1, a2

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
            out = arr.copy()
            out[m] = savgol_filter(arr[m], window_length=w, polyorder=2)
            return out

        self.A_s = smooth_1d(A, self.smooth_baseline_m)
        self.B_s = smooth_1d(B, self.smooth_tilt_m)
        self.C_s = smooth_1d(C, self.smooth_tilt_m)

        # reconstruct trend and residuals
        trend = np.full_like(z, np.nan)
        for i in idxs:
            if not np.isfinite(self.A_s[i]):
                continue
            xc = np.median(x[np.isfinite(z[i, :])])
            xi = x - xc
            trend[i, :] = self.A_s[i] + self.B_s[i] * xi + self.C_s[i] * (xi**2)
        self.trend = trend
        self.residuals = z - trend
        return self

    def robust_sym_vlim(self, arr=None, q=0.98):
        a = self.residuals if arr is None else arr
        return robust_sym_vlim(a, q=q)


# -----------------------------
# Alignment search (dx,dy,θ)
# -----------------------------
@dataclass
class LevelSpec:
    theta_range: Tuple[float, float]  # degrees (min,max)
    theta_step: float  # degrees
    dx_range: Tuple[float, float]  # meters
    dy_range: Tuple[float, float]  # meters
    step_xy: float  # meters
    stride: int = 1  # sample every Nth MBES pixel
    k: int = 1  # NN sample from UHI
    sigma: float = 0.0  # (kept for future blur; not used here)


@dataclass
class GateSpec:
    min_pairs: int = 1000  # require at least this many matches to score
    min_score: float = -0.25  # if level-best < this → stop (no hope)
    stop_if_score_ge: float = 0.35  # early stop if good enough


class AlignmentSearch:
    """
    Consistent pipeline:
      - footprint & KDTree built from the exact sliced arrays used for sampling
      - Pearson r on raw values (scale-invariant); z-scaling available if needed
    """

    def __init__(
        self,
        E: np.ndarray,
        N: np.ndarray,
        mask_valid: np.ndarray,
        uhi_corr: np.ndarray,
        mb: MBESDetrender,
        lat0: float,
        lon0: float,
        h0: float,
    ):
        # store UHI
        self.E = E
        self.N = N
        self.mask_valid = mask_valid.astype(bool)
        self.uhi_corr = uhi_corr

        # jagged footprint path
        Xc, Yc = pcolormesh_pad(E, N)
        segs_xy, segs_idx = boundary_segments_from_mask(self.mask_valid, Xc, Yc)
        loops = trace_loops_from_segments(segs_idx)
        self.footprint_path0 = compound_path_from_loops(loops, Xc, Yc)
        self.outline_lc = LineCollection(segs_xy, colors="k", linewidths=1.1, alpha=0.9)

        # UHI KDTree
        valid_uhi = (
            self.mask_valid & np.isfinite(uhi_corr) & np.isfinite(E) & np.isfinite(N)
        )
        self.uhi_pts = np.column_stack([E[valid_uhi], N[valid_uhi]])
        self.uhi_vals = uhi_corr[valid_uhi]
        if self.uhi_pts.size == 0:
            raise RuntimeError("No valid UHI points for KDTree (check masks/slice).")
        self.tree_uhi = cKDTree(self.uhi_pts)

        # pivot: geometric center of valid UHI footprint
        self.cx = float(np.nanmean(E[valid_uhi]))
        self.cy = float(np.nanmean(N[valid_uhi]))

        # MBES → NED (once)
        MBES_X, MBES_Y = np.meshgrid(mb.x_cols, mb.y_rows)
        epsg_src = getattr(mb, "epsg", 32633) if getattr(mb, "epsg", None) else 32633
        tf_to_ll = Transformer.from_crs(f"EPSG:{epsg_src}", "EPSG:4326", always_xy=True)
        mbes_lon, mbes_lat = tf_to_ll.transform(MBES_X, MBES_Y)
        tf_geo_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
        mbes_x_ecef, mbes_y_ecef, mbes_z_ecef = tf_geo_to_ecef.transform(
            mbes_lon, mbes_lat, np.zeros_like(mbes_lon)
        )
        mbes_n_ned, mbes_e_ned, _ = _ecef_to_ned_arrays(
            mbes_x_ecef, mbes_y_ecef, mbes_z_ecef, lat0, lon0, h0
        )
        self.mbes_e = mbes_e_ned
        self.mbes_n = mbes_n_ned
        self.mbes_res = mb.residuals
        self.P_mbes = np.column_stack([self.mbes_e.ravel(), self.mbes_n.ravel()])

        # base sanity: check overlap at (0,0,0)
        inside0 = self.footprint_path0.contains_points(self.P_mbes)
        self.base_pairs = int(np.sum(inside0))
        if self.base_pairs == 0:
            raise RuntimeError(
                "No MBES pixels inside the UHI footprint at (θ=0, dx=0, dy=0). "
                "Check coordinate transforms and slices."
            )

    # Pearson correlation (scale-invariant)
    @staticmethod
    def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            return np.nan
        x = x[m]
        y = y[m]
        x = (x - x.mean()) / (x.std(ddof=1) if x.std(ddof=1) != 0 else 1.0)
        y = (y - y.mean()) / (y.std(ddof=1) if y.std(ddof=1) != 0 else 1.0)
        return float(np.clip(np.corrcoef(x, y)[0, 1], -1, 1))

    def score_one(
        self,
        theta_deg: float,
        dx: float,
        dy: float,
        stride: int,
        k: int,
        min_pairs: int,
    ) -> Tuple[float, int]:
        # inverse-transform MBES centers back to base UHI frame
        rad = np.deg2rad(theta_deg)
        c, s = np.cos(rad), np.sin(rad)
        Rinv = np.array([[c, s], [-s, c]], float)

        P = self.P_mbes[::stride]
        Pprime = (P - np.array([dx, dy])) - np.array([self.cx, self.cy])
        Pprime = Pprime @ Rinv.T + np.array([self.cx, self.cy])

        inside = self.footprint_path0.contains_points(Pprime)
        if not inside.any():
            return np.nan, 0

        idx_inside = np.where(inside)[0]
        if idx_inside.size < min_pairs:
            return np.nan, int(idx_inside.size)

        # NN sample UHI at Pprime[inside]
        d, nn = self.tree_uhi.query(Pprime[idx_inside], k=k)
        u = self.uhi_vals[nn].astype(float).ravel()

        # MBES residuals at the corresponding (downsampled) pixels
        z = self.mbes_res.ravel()[::stride][idx_inside].astype(float).ravel()

        r = self.pearson_r(u, z)
        return r, int(idx_inside.size)

    def run(
        self,
        levels: List[LevelSpec],
        gates: GateSpec,
        log_csv: Optional[str] = None,
        viz_every: int = 0,
        show_progress: bool = True,
        show_final: bool = True,
    ) -> Dict:
        # CSV logging setup
        writer = None
        f_csv = None
        if log_csv:
            exists = os.path.exists(log_csv)
            f_csv = open(log_csv, "a", newline="")
            writer = csv.writer(f_csv)
            if not exists:
                writer.writerow(
                    [
                        "ts",
                        "level",
                        "theta_deg",
                        "dx_m",
                        "dy_m",
                        "stride",
                        "k",
                        "pairs",
                        "score",
                    ]
                )

        best = {
            "score": -np.inf,
            "theta": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "n": 0,
            "level": -1,
        }
        try:
            # Optional progress bar
            try:
                from tqdm import tqdm

                use_pbar = True
            except Exception:
                use_pbar = False

            for li, lvl in enumerate(levels, 1):
                th_vals = np.arange(
                    lvl.theta_range[0], lvl.theta_range[1] + 1e-9, lvl.theta_step
                )
                dx_vals = np.arange(
                    lvl.dx_range[0], lvl.dx_range[1] + 1e-9, lvl.step_xy
                )
                dy_vals = np.arange(
                    lvl.dy_range[0], lvl.dy_range[1] + 1e-9, lvl.step_xy
                )

                total = len(th_vals) * len(dx_vals) * len(dy_vals)
                pbar = (
                    tqdm(total=total, desc=f"Level {li}", unit="try")
                    if use_pbar and show_progress
                    else None
                )

                level_best = -np.inf
                level_any = False

                tries = 0
                for th in th_vals:
                    for dy in dy_vals:
                        for dx in dx_vals:
                            tries += 1
                            score, pairs = self.score_one(
                                th, dx, dy, lvl.stride, lvl.k, gates.min_pairs
                            )
                            level_any |= np.isfinite(score)

                            if writer:
                                writer.writerow(
                                    [
                                        time.time(),
                                        li,
                                        th,
                                        dx,
                                        dy,
                                        lvl.stride,
                                        lvl.k,
                                        pairs,
                                        score,
                                    ]
                                )

                            if np.isfinite(score) and score > level_best:
                                level_best = score

                            # keep global best
                            if np.isfinite(score) and score > best["score"]:
                                best.update(
                                    {
                                        "score": float(score),
                                        "theta": float(th),
                                        "dx": float(dx),
                                        "dy": float(dy),
                                        "n": int(pairs),
                                        "level": li,
                                    }
                                )
                                # quick viz of "best so far"
                                if viz_every and (tries % max(1, int(viz_every)) == 0):
                                    self._quick_viz(
                                        best,
                                        title=f"Level {li} — best so far (score={score:.3f})",
                                    )

                            if pbar:
                                pbar.update(1)

                if pbar:
                    pbar.close()

                # if we never got a finite score, bail early
                if not level_any:
                    print(
                        f"• Early stop after Level {li}: pairs<{gates.min_pairs} everywhere."
                    )
                    break

                # if clearly poor, stop
                if level_best < gates.min_score:
                    print(
                        f"• Early stop after Level {li}: best r<{gates.min_score:.2f}"
                    )
                    break

                # if already good enough, stop
                if best["score"] >= gates.stop_if_score_ge:
                    print(
                        f"• Early stop: r≥{gates.stop_if_score_ge:.2f} reached at Level {li}."
                    )
                    break

            # final overlay
            if show_final:
                self._quick_viz(best, title="Best alignment")
        finally:
            if f_csv:
                f_csv.close()

        return best

    def _quick_viz(self, best: Dict, title: str = ""):
        # transform the footprint for display at "best"
        rad = np.deg2rad(best["theta"])
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], float)
        dx0, dy0 = best["dx"], best["dy"]
        verts = self.footprint_path0.vertices.copy()
        mask_close = np.isfinite(verts[:, 0]) & np.isfinite(verts[:, 1])
        V = verts[mask_close]
        V2 = (
            (V - np.array([self.cx, self.cy])) @ R.T
            + np.array([self.cx, self.cy])
            + np.array([dx0, dy0])
        )
        verts[mask_close] = V2
        footprint_path_best = MplPath(verts, self.footprint_path0.codes)

        # plot
        vmin_mbes, vmax_mbes = robust_sym_vlim(self.mbes_res, q=0.98)
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 10))
        extent_ned = [
            float(np.nanmin(self.mbes_e)),
            float(np.nanmax(self.mbes_e)),
            float(np.nanmin(self.mbes_n)),
            float(np.nanmax(self.mbes_n)),
        ]
        im = ax.imshow(
            np.ma.masked_invalid(self.mbes_res),
            extent=extent_ned,
            origin="upper",
            cmap="RdBu_r",
            vmin=vmin_mbes,
            vmax=vmax_mbes,
        )
        ax.add_patch(
            PathPatch(
                footprint_path_best,
                transform=ax.transData,
                facecolor="none",
                edgecolor="y",
                lw=1.3,
                alpha=0.95,
            )
        )
        ax.set_title(
            f"{title}\nθ={best['theta']:+.2f}°, dx={best['dx']:+.2f} m, dy={best['dy']:+.2f} m, r={best['score']:.3f}"
        )
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        pad_limits(
            ax,
            footprint_path_best.vertices[:, 0].min(),
            footprint_path_best.vertices[:, 0].max(),
            footprint_path_best.vertices[:, 1].min(),
            footprint_path_best.vertices[:, 1].max(),
            pad_frac=0.03,
            equal_aspect=True,
        )
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="MBES Residuals (m)")
        plt.show()


# -----------------------------
# Defaults & high-level wrapper
# -----------------------------
def default_levels_small() -> List[LevelSpec]:
    """
    Start small (your last request): ±3° in 1° steps, ±0.5 m in 0.5 m steps.
    You can pass your own levels list to auto_align_from_config() to change this.
    """
    return [
        LevelSpec(
            theta_range=(-3.0, 3.0),
            theta_step=1.0,
            dx_range=(-0.5, 0.5),
            dy_range=(-0.5, 0.5),
            step_xy=0.5,
            stride=2,
            k=1,
            sigma=0.0,
        ),
        # you can add a second, finer level later if you like
    ]


def default_gates_small() -> GateSpec:
    return GateSpec(min_pairs=1000, min_score=-0.25, stop_if_score_ge=0.35)


def auto_align_from_config(
    file_ids: List[str],
    track_start: int,
    track_end: int,
    log_csv: Optional[str],
    levels: List[LevelSpec],
    gates: GateSpec,
    metric: str = "pearson",
    viz_every: int = 10,
    show_progress: bool = True,
    show_final: bool = True,
) -> Tuple[Dict, AlignmentSearch]:
    """
    One-call runner:
      - load transect, slice tracks
      - illumination correction
      - jagged footprint from that *same* slice
      - detrend MBES
      - run alignment search
    """
    if metric.lower() != "pearson":
        print("Only 'pearson' is implemented for scoring (scale-invariant).")

    # --- UHI: load & slice ---
    transect = georef.load_transect(config.OUTPUT_FOLDER)
    cube = transect.select_files(file_ids)

    # slice arrays
    X = cube.X_ecef[track_start:track_end, :]
    Y = cube.Y_ecef[track_start:track_end, :]
    Z = cube.Z_ecef[track_start:track_end, :]
    R = cube.R[track_start:track_end, :]
    G = cube.G[track_start:track_end, :]
    B = cube.B[track_start:track_end, :]

    full_valid = build_full_valid(X, Y, Z, R, G, B)

    lat0, lon0, h0 = config.LAT0, config.LON0, config.H0
    N, E, D = _ecef_to_ned_arrays(X, Y, Z, lat0, lon0, h0)

    # illumination correction (persistent)
    cube.apply_illumination_correction(window_size=1000, strength=1.0)
    wavelengths = cube.wavelengths
    idx_green = int(np.argmin(np.abs(wavelengths - 560.0)))
    uhi_corr = cube.data_corrected[track_start:track_end, :, idx_green]

    # --- MBES ---
    mb = (
        MBESDetrender(
            config.MBES_GEOTIFF,
            smooth_baseline_m=0.5,
            smooth_tilt_m=0.5,
            smooth_center_m=1.0,
        )
        .load()
        .detrend(order_x=2, robust=True, central_frac=0.8)
    )

    # --- build search object (this verifies overlap in base position) ---
    aligner = AlignmentSearch(
        E=E,
        N=N,
        mask_valid=full_valid,
        uhi_corr=uhi_corr,
        mb=mb,
        lat0=lat0,
        lon0=lon0,
        h0=h0,
    )

    # --- quick visual of the starting position (0,0,0) ---
    print(f"Start — base pairs inside footprint: {aligner.base_pairs:,}")
    aligner._quick_viz(
        {
            "theta": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "score": float("nan"),
            "n": aligner.base_pairs,
        },
        title="Start — best score=nan",
    )

    # --- run the grid search (coarse→fine) ---
    best = aligner.run(
        levels=levels,
        gates=gates,
        log_csv=log_csv,
        viz_every=viz_every,
        show_progress=show_progress,
        show_final=show_final,
    )

    # summary
    print(
        f"\nFinal verdict: r={best['score']:.3f}  θ={best['theta']:+.2f}°, "
        f"dx={best['dx']:+.2f} m, dy={best['dy']:+.2f} m, pairs={best['n']}\n"
    )

    return best, aligner


# run_alignment.py (not a real file; just paste this below File 1 in the same cell)

best, aligner = auto_align_from_config(
    file_ids=["rad_uhi_20241029_115057_4", "rad_uhi_20241029_115057_5"],
    track_start=3039,
    track_end=4029,
    log_csv="./alignment_search_log.csv",  # all tries appended here
    levels=default_levels_small(),  # your requested ±0.5 m, ±3° (one level)
    gates=default_gates_small(),  # sane gates & early-stop
    metric="pearson",
    viz_every=10,  # quick overlay every N accepted updates
    show_progress=True,
    show_final=True,
)

print("BEST:", best)
