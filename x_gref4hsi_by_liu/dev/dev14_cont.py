# alignment_module.py
# UHI ↔ MBES alignment with live progress & per-level summaries.

from __future__ import annotations
import os, csv, time, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

# Add paths for imports
sys.path.insert(0, os.path.abspath(".."))  # for x_gref4hsi_by_liu (config, utils)
sys.path.insert(
    0, os.path.abspath("../../gref4hsi/final_act")
)  # for full-featured modules

# repo modules (your project provides these)
from utils.gref_pipeline import georef
from utils.gref_pipeline.georef import _ecef_to_ned_arrays
from utils.other.detrend_mbes import MBESDetrender  # Import full-featured version
import config


# ---------- small helpers ----------
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
    """EXACT jagged outline including outer edges."""
    T, S = mask.shape
    segs_xy, segs_idx = [], []

    # internal vertical
    vdiff = mask[:, 1:] != mask[:, :-1]
    iv, jv = np.nonzero(vdiff)
    jline = jv + 1
    for i, j in zip(iv, jline):
        a = (i, j)
        b = (i + 1, j)
        segs_idx.append((a, b))
        segs_xy.append([[Xc[a], Yc[a]], [Xc[b], Yc[b]]])

    # outer left/right
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

    # internal horizontal
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


# ---------- search specs ----------
@dataclass
class LevelSpec:
    theta_range: Tuple[float, float]
    theta_step: float
    dx_range: Tuple[float, float]
    dy_range: Tuple[float, float]
    step_xy: float
    stride: int = 1
    k: int = 1
    sigma: float = 0.0
    sample_frac: float = 1.0  # random subsample of grid (0< x ≤1)
    max_tries: Optional[int] = None  # hard cap on grid tries


@dataclass
class GateSpec:
    min_pairs: int = 1000
    min_score: float = -0.25
    stop_if_score_ge: float = 0.35


# ---------- alignment search ----------
class AlignmentSearch:
    def __init__(self, E, N, mask_valid, uhi_corr, mb: MBESDetrender, lat0, lon0, h0):
        self.E = E
        self.N = N
        self.mask_valid = mask_valid.astype(bool)
        self.uhi_corr = uhi_corr
        self.mb = mb  # Store MBES detrender for plotting

        # footprint path
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
            raise RuntimeError("No valid UHI points for KDTree.")
        self.tree_uhi = cKDTree(self.uhi_pts)

        # pivot
        self.cx = float(np.nanmean(E[valid_uhi]))
        self.cy = float(np.nanmean(N[valid_uhi]))

        # MBES coordinates - check if already in NED or needs transformation
        MBES_X, MBES_Y = np.meshgrid(mb.x_cols, mb.y_rows)

        # Check if MBES is already in NED coordinate system
        if hasattr(mb, "coord_system") and mb.coord_system.lower() == "ned":
            # MBES is already in NED - use directly
            self.mbes_e = MBES_X  # East
            self.mbes_n = MBES_Y  # North
        else:
            # MBES is in UTM/geographic - need to transform to NED
            epsg_src = (
                getattr(mb, "epsg", 32633) if getattr(mb, "epsg", None) else 32633
            )
            tf_to_ll = Transformer.from_crs(
                f"EPSG:{epsg_src}", "EPSG:4326", always_xy=True
            )
            mbes_lon, mbes_lat = tf_to_ll.transform(MBES_X, MBES_Y)
            tf_geo_to_ecef = Transformer.from_crs(
                "EPSG:4979", "EPSG:4978", always_xy=True
            )
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

        # base overlap
        base_inside = self.footprint_path0.contains_points(self.P_mbes)
        self.base_pairs = int(np.sum(base_inside))
        if self.base_pairs == 0:
            raise RuntimeError("No MBES pixels inside UHI footprint at base.")

        # keep last-level results for summaries
        self.last_level = None

    @staticmethod
    def _pearson_r(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            return np.nan
        x = x[m]
        y = y[m]
        x = (x - x.mean()) / (x.std(ddof=1) or 1.0)
        y = (y - y.mean()) / (y.std(ddof=1) or 1.0)
        return float(np.clip(np.corrcoef(x, y)[0, 1], -1, 1))

    def score_one(self, theta_deg, dx, dy, stride, k, min_pairs) -> Tuple[float, int]:
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

        d, nn = self.tree_uhi.query(Pprime[idx_inside], k=k)
        u = self.uhi_vals[nn].astype(float).ravel()
        z = self.mbes_res.ravel()[::stride][idx_inside].astype(float).ravel()
        return self._pearson_r(u, z), int(idx_inside.size)

    def run(
        self,
        levels: List[LevelSpec],
        gates: GateSpec,
        log_csv: Optional[str] = None,
        viz_every: int = 0,
        show_progress: bool = True,
        show_final: bool = True,
        plot_each_level: bool = True,
        rng_seed: int = 0,
    ) -> Dict:
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
            from tqdm import tqdm

            have_tqdm = True
        except Exception:
            have_tqdm = False

        rng = np.random.default_rng(rng_seed)

        try:
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
                nth, ndx, ndy = len(th_vals), len(dx_vals), len(dy_vals)

                # storage for level summary plots
                r_cube = np.full((nth, ndy, ndx), np.nan, dtype=float)

                # build grid and optionally subsample/limit
                grid = np.array(np.meshgrid(th_vals, dy_vals, dx_vals, indexing="ij"))
                combos = grid.reshape(3, -1).T  # (n,3): [th, dy, dx]
                n_total = combos.shape[0]
                sel_idx = np.arange(n_total)

                if 0 < lvl.sample_frac < 1.0:
                    n_pick = max(1, int(np.floor(n_total * lvl.sample_frac)))
                    sel_idx = rng.choice(sel_idx, size=n_pick, replace=False)
                if lvl.max_tries is not None:
                    sel_idx = sel_idx[: int(lvl.max_tries)]
                combos = combos[sel_idx]
                n_eval = combos.shape[0]

                pbar = (
                    tqdm(total=n_eval, desc=f"Level {li}", unit="try")
                    if (have_tqdm and show_progress)
                    else None
                )

                level_best = -np.inf
                accepted = 0

                # fast indexers from values → integer indices
                def idx_th(val):
                    return int(np.round((val - th_vals[0]) / lvl.theta_step))

                def idx_dx(val):
                    return int(np.round((val - dx_vals[0]) / lvl.step_xy))

                def idx_dy(val):
                    return int(np.round((val - dy_vals[0]) / lvl.step_xy))

                for j, (th, dy, dx) in enumerate(combos, 1):
                    score, pairs = self.score_one(
                        float(th),
                        float(dx),
                        float(dy),
                        lvl.stride,
                        lvl.k,
                        gates.min_pairs,
                    )

                    # fill cube if this point maps onto the regular grid
                    try:
                        r_cube[idx_th(th), idx_dy(dy), idx_dx(dx)] = score
                    except Exception:
                        pass  # in case of float roundoff outside grid

                    if writer:
                        writer.writerow(
                            [
                                time.time(),
                                li,
                                float(th),
                                float(dx),
                                float(dy),
                                lvl.stride,
                                lvl.k,
                                pairs,
                                score,
                            ]
                        )

                    if np.isfinite(score):
                        accepted += 1
                        if score > level_best:
                            level_best = score
                        if score > best["score"]:
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
                            if viz_every and (accepted % max(1, int(viz_every)) == 0):
                                self._quick_viz(best, title=f"Level {li} — best so far")

                    if pbar:
                        acc_rate = 100.0 * accepted / j
                        pbar.set_postfix(
                            best=f"{best['score']:.3f}",
                            level=f"{level_best:.3f}",
                            acc=f"{acc_rate:.1f}%",
                            pairs=pairs,
                        )
                        pbar.update(1)

                if pbar:
                    pbar.close()

                # store last-level info for plotting summaries
                self.last_level = {
                    "r_cube": r_cube,
                    "th_vals": th_vals,
                    "dx_vals": dx_vals,
                    "dy_vals": dy_vals,
                    "level_best": float(level_best),
                }

                # per-level summary plots
                if plot_each_level:
                    self._plot_level_summary()

                # early-stop logic
                if accepted == 0:
                    print(
                        f"• Early stop after Level {li}: no positions reached min_pairs={gates.min_pairs}."
                    )
                    break
                if level_best < gates.min_score:
                    print(
                        f"• Early stop after Level {li}: level-best r<{gates.min_score:.2f}."
                    )
                    break
                if best["score"] >= gates.stop_if_score_ge:
                    print(
                        f"• Early stop: r≥{gates.stop_if_score_ge:.2f} at Level {li}."
                    )
                    break

            if show_final:
                self._quick_viz(best, title="Best alignment")

        finally:
            if f_csv:
                f_csv.close()

        return best

    # ---------- small displays ----------
    def _quick_viz(self, best: Dict, title: str = ""):
        rad = np.deg2rad(best["theta"])
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], float)
        dx0, dy0 = best["dx"], best["dy"]

        verts = self.footprint_path0.vertices.copy()
        mask = np.isfinite(verts[:, 0]) & np.isfinite(verts[:, 1])
        V = verts[mask]
        V2 = (
            (V - np.array([self.cx, self.cy])) @ R.T
            + np.array([self.cx, self.cy])
            + np.array([dx0, dy0])
        )
        verts[mask] = V2
        footprint_path_best = MplPath(verts, self.footprint_path0.codes)

        vmin_mbes, vmax_mbes = robust_sym_vlim(self.mbes_res, q=0.98)
        fig, ax = plt.subplots(1, 1, figsize=(5.6, 10))
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
        plt.show(block=False)
        plt.pause(0.1)

    def _plot_level_summary(self):
        """Three panels: r(dx,dy) at best θ, r vs θ, and overlay at best."""
        L = self.last_level
        if L is None:
            return
        r_cube = L["r_cube"]
        th_vals = L["th_vals"]
        dx_vals = L["dx_vals"]
        dy_vals = L["dy_vals"]

        # (1) best over dx,dy per theta
        r_vs_theta = np.nanmax(r_cube, axis=(1, 2))
        it_best = int(np.nanargmax(r_vs_theta)) if np.isfinite(r_vs_theta).any() else 0
        th_best = th_vals[it_best]
        R_slice = r_cube[it_best, :, :]

        # argmax within that slice for dx,dy
        j_best, i_best = np.unravel_index(int(np.nanargmax(R_slice)), R_slice.shape)
        dx_best = dx_vals[i_best]
        dy_best = dy_vals[j_best]
        r_best = float(R_slice[j_best, i_best])

        # (2) build overlay path for that transform
        rad = np.deg2rad(th_best)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], float)

        verts = self.footprint_path0.vertices.copy()
        mask = np.isfinite(verts[:, 0]) & np.isfinite(verts[:, 1])
        V = verts[mask]
        V2 = (
            (V - np.array([self.cx, self.cy])) @ R.T
            + np.array([self.cx, self.cy])
            + np.array([dx_best, dy_best])
        )
        verts[mask] = V2
        path_best = MplPath(verts, self.footprint_path0.codes)

        # (3) draw
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
        axA, axB, axC = axes

        imA = axA.imshow(
            R_slice,
            origin="lower",
            extent=[
                dx_vals[0] - 0.5 * (dx_vals[1] - dx_vals[0]),
                dx_vals[-1] + 0.5 * (dx_vals[1] - dx_vals[0]),
                dy_vals[0] - 0.5 * (dy_vals[1] - dy_vals[0]),
                dy_vals[-1] + 0.5 * (dy_vals[1] - dy_vals[0]),
            ],
            cmap="viridis",
            vmin=-1,
            vmax=1,
            aspect="equal",
        )
        axA.scatter([dx_best], [dy_best], c="r", s=50, marker="x", zorder=5)
        axA.set_title(f"r(dx,dy) at θ={th_best:+.1f}°")
        axA.set_xlabel("dx (m)")
        axA.set_ylabel("dy (m)")
        fig.colorbar(imA, ax=axA, fraction=0.045, pad=0.02, label="Pearson r")

        axB.plot(th_vals, r_vs_theta, lw=2)
        axB.axvline(th_best, color="r", ls="--", lw=1)
        axB.set_title("max r over (dx,dy) vs θ")
        axB.set_xlabel("θ (deg)")
        axB.set_ylabel("max r")

        vmin_mbes, vmax_mbes = robust_sym_vlim(self.mbes_res, q=0.98)
        extent_ned = [
            float(np.nanmin(self.mbes_e)),
            float(np.nanmax(self.mbes_e)),
            float(np.nanmin(self.mbes_n)),
            float(np.nanmax(self.mbes_n)),
        ]
        imC = axC.imshow(
            np.ma.masked_invalid(self.mbes_res),
            extent=extent_ned,
            origin="upper",
            cmap="RdBu_r",
            vmin=vmin_mbes,
            vmax=vmax_mbes,
        )
        axC.add_patch(
            PathPatch(
                path_best,
                transform=axC.transData,
                facecolor="none",
                edgecolor="k",
                lw=1.2,
                alpha=0.95,
            )
        )
        axC.set_title(
            f"Overlay @ best: θ={th_best:+.1f}°, dx={dx_best:+.2f} m, dy={dy_best:+.2f} m\nr={r_best:.3f}"
        )
        axC.set_xlabel("East (m)")
        axC.set_ylabel("North (m)")
        pad_limits(
            axC,
            path_best.vertices[:, 0].min(),
            path_best.vertices[:, 0].max(),
            path_best.vertices[:, 1].min(),
            path_best.vertices[:, 1].max(),
            pad_frac=0.03,
            equal_aspect=True,
        )
        fig.colorbar(imC, ax=axC, fraction=0.045, pad=0.02, label="MBES Residuals (m)")

        plt.show(block=False)
        plt.pause(0.1)

    def plot_density_scatter(self, best: Dict, gridsize: int = 80):
        """Density scatter plot: MBES residuals vs UHI intensity at best alignment."""
        theta_deg = best["theta"]
        dx_m = best["dx"]
        dy_m = best["dy"]

        # Transform MBES points
        rad = np.deg2rad(theta_deg)
        c, s = np.cos(rad), np.sin(rad)
        Rinv = np.array([[c, s], [-s, c]], float)
        Pp = (self.P_mbes - np.array([dx_m, dy_m])) - np.array([self.cx, self.cy])
        Pp = Pp @ Rinv.T + np.array([self.cx, self.cy])

        # Get points inside footprint
        inside = self.footprint_path0.contains_points(Pp)
        idx = np.where(inside)[0]
        if idx.size == 0:
            print("No overlap for scatter plot")
            return

        # Query UHI values
        _, nn = self.tree_uhi.query(Pp[idx], k=1)
        u = self.uhi_vals[nn].astype(float)
        z = self.mbes_res.ravel()[idx].astype(float)

        # Robust z-scores
        zU = robust_z(u)
        zM = robust_z(z)
        m = np.isfinite(zU) & np.isfinite(zM)
        zU = zU[m]
        zM = zM[m]

        if zU.size < 50:
            print("Too few valid pairs for scatter plot")
            return

        # Compute metrics
        r = self._pearson_r(zM, zU)
        b, a = np.polyfit(zM, zU, 1) if zU.size > 1 else (np.nan, np.nan)

        # Plot
        fig = plt.figure(figsize=(7.2, 6.0))
        ax = fig.add_subplot(111)
        hb = ax.hexbin(zM, zU, gridsize=gridsize, mincnt=1, cmap="viridis")
        lim = float(
            np.clip(
                np.nanmax(np.abs([zM.min(), zM.max(), zU.min(), zU.max()])), 2.0, 6.0
            )
        )

        # 1:1 line
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.1, label="1:1")
        # Fit line
        ax.plot(
            [-lim, lim],
            [a + b * (-lim), a + b * lim],
            "r-",
            lw=1.2,
            label=f"fit  b={b:.2f}",
        )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("z(MBES residuals)")
        ax.set_ylabel("z(UHI intensity)")
        ax.set_title(
            f"Density scatter: θ={theta_deg:+.1f}°, dx={dx_m:+.2f} m, dy={dy_m:+.2f} m  (r={r:.3f}, n={zU.size})"
        )
        cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("count")
        ax.legend(loc="lower right")
        plt.show(block=False)
        plt.pause(0.1)

        print(f"\n📊 Scatter plot metrics:")
        print(f"   Correlation (r): {r:.4f}")
        print(f"   Fit slope (b): {b:.4f}")
        print(f"   Fit intercept (a): {a:.4f}")
        print(f"   Pairs: {zU.size:,}")

    def plot_mbes_with_uhi_footprint(self, mb: MBESDetrender, title_suffix: str = ""):
        """Plot MBES residuals with UHI footprint overlay - BOTH full and zoomed views."""
        if mb.residuals is None:
            print("⚠️  MBES not detrended yet!")
            return

        # Create figure with 2 subplots: full view and zoomed view
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot MBES residuals
        vmin, vmax = robust_sym_vlim(mb.residuals, q=0.98)

        # Labels
        if hasattr(mb, "coord_system") and mb.coord_system.lower() == "ned":
            xlabel, ylabel = "East (m)", "North (m)"
        else:
            xlabel, ylabel = "Easting (m)", "Northing (m)"

        # Build boundary from mask (shared for both plots)
        Xc, Yc = pcolormesh_pad(self.E, self.N)
        segs_xy, segs_idx = boundary_segments_from_mask(self.mask_valid, Xc, Yc)

        # LEFT PLOT: FULL MBES VIEW
        im1 = ax1.imshow(
            np.ma.masked_invalid(mb.residuals),
            extent=mb.extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )
        lc1 = LineCollection(
            segs_xy, colors="black", linewidths=2.5, alpha=1.0, zorder=10
        )
        ax1.add_collection(lc1)
        ax1.set_xlabel(xlabel, fontsize=11)
        ax1.set_ylabel(ylabel, fontsize=11)
        ax1.set_title(f"Full MBES View{title_suffix}", fontsize=12)
        ax1.set_aspect("equal", adjustable="box")
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02)
        cbar1.set_label("Residuals (m)", fontsize=10)

        # RIGHT PLOT: ZOOMED TO UHI FOOTPRINT
        im2 = ax2.imshow(
            np.ma.masked_invalid(mb.residuals),
            extent=mb.extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )
        lc2 = LineCollection(
            segs_xy, colors="black", linewidths=2.5, alpha=1.0, zorder=10
        )
        ax2.add_collection(lc2)
        ax2.set_xlabel(xlabel, fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title(f"Zoomed to UHI Footprint{title_suffix}", fontsize=12)

        # Zoom to UHI footprint region
        if np.any(self.mask_valid):
            valid_e = self.E[self.mask_valid]
            valid_n = self.N[self.mask_valid]
            e_min, e_max = valid_e.min(), valid_e.max()
            n_min, n_max = valid_n.min(), valid_n.max()

            # Add margin
            e_range = e_max - e_min
            n_range = n_max - n_min
            margin = 0.15  # 15% margin

            ax2.set_xlim(e_min - margin * e_range, e_max + margin * e_range)
            ax2.set_ylim(n_min - margin * n_range, n_max + margin * n_range)

        ax2.set_aspect("equal", adjustable="box")
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
        cbar2.set_label("Residuals (m)", fontsize=10)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        return fig


# ---------- sensible defaults + wrapper ----------
def default_levels_tiny() -> List[LevelSpec]:
    """Tiny grid for smoke tests (fast)."""
    return [
        LevelSpec(
            theta_range=(-1.0, 1.0),
            theta_step=1.0,
            dx_range=(-0.5, 0.5),
            dy_range=(-0.5, 0.5),
            step_xy=0.5,
            stride=4,
            k=1,
            sample_frac=1.0,
            max_tries=None,
        )
    ]


def default_levels_small() -> List[LevelSpec]:
    """≈63 tries: ±3° in 1° steps, ±0.5 m in 0.5 m steps."""
    return [
        LevelSpec(
            theta_range=(-3.0, 3.0),
            theta_step=1.0,
            dx_range=(-0.5, 0.5),
            dy_range=(-0.5, 0.5),
            step_xy=0.5,
            stride=2,
            k=1,
            sample_frac=1.0,
            max_tries=None,
        )
    ]


def default_levels_medium() -> List[LevelSpec]:
    """Coarse→fine, still quick if stride>1 or sample_frac<1."""
    return [
        LevelSpec(
            theta_range=(-5.0, 5.0),
            theta_step=2.0,
            dx_range=(-1.0, 1.0),
            dy_range=(-1.0, 1.0),
            step_xy=1.0,
            stride=3,
            k=1,
            sample_frac=0.5,
            max_tries=150,
        ),
        LevelSpec(
            theta_range=(-2.0, 2.0),
            theta_step=1.0,
            dx_range=(-0.5, 0.5),
            dy_range=(-0.5, 0.5),
            step_xy=0.25,
            stride=2,
            k=1,
            sample_frac=1.0,
            max_tries=None,
        ),
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
    plot_each_level: bool = True,
    rng_seed: int = 0,
) -> Tuple[Dict, AlignmentSearch]:
    """One-call pipeline: load → slice → correct → detrend → search."""
    if metric.lower() != "pearson":
        print("Only 'pearson' is implemented (scale-invariant).")

    # UHI
    transect = georef.load_transect(config.OUTPUT_FOLDER)
    cube = transect.select_files(file_ids)

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

    # MBES - USE NED COORDINATES TO MATCH UHI FOOTPRINT
    mb = (
        MBESDetrender(
            config.MBES_GEOTIFF,
            coord_system="ned",  # NED coordinates to match UHI footprint!
            ned_origin=(lon0, lat0, h0),  # Same origin as UHI conversion
            epsg_utm=config.EPSG_UTM,
            epsg_geo=config.EPSG_GEOGRAPHIC,
            smooth_baseline_m=0.5,
            smooth_tilt_m=0.5,
            smooth_center_m=1.0,
        )
        .load()
        .detrend(order_x=2, robust=True, central_frac=0.8)
    )

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

    print(f"Start — base pairs inside footprint: {aligner.base_pairs:,}")
    aligner._quick_viz(
        {
            "theta": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "score": float("nan"),
            "n": aligner.base_pairs,
        },
        title="Start — base overlay",
    )

    # Show MBES + UHI footprint at START (before alignment)
    print("\n" + "=" * 70)
    print("INITIAL: MBES residuals with UHI footprint overlay (BEFORE alignment)")
    print("=" * 70)
    aligner.plot_mbes_with_uhi_footprint(
        aligner.mb, title_suffix=" - INITIAL (θ=0°, dx=0m, dy=0m)"
    )
    print("✅ Check if UHI footprint (BLACK OUTLINE) is roughly over MBES data.")
    print("   This shows the alignment BEFORE any corrections.\n")

    best = aligner.run(
        levels=levels,
        gates=gates,
        log_csv=log_csv,
        viz_every=viz_every,
        show_progress=show_progress,
        show_final=show_final,
        plot_each_level=plot_each_level,
        rng_seed=rng_seed,
    )

    print(
        "\nFinal verdict:"
        f" r={best['score']:.3f}  θ={best['theta']:+.2f}°,"
        f" dx={best['dx']:+.2f} m, dy={best['dy']:+.2f} m, pairs={best['n']}\n"
    )

    return best, aligner


# ============================================================================
# MAIN RUN - Edit parameters below and run this file
# ============================================================================

if __name__ == "__main__":
    # Search parameters
    dx_range = (-0.5, 0.5)  # meters
    dy_range = (-0.5, 0.5)  # meters
    theta_range = (-3.0, 3.0)  # degrees
    step_dx = 0.5
    step_dy = 0.5
    step_theta = 3
    stride = 4  # higher = faster (try 4 for quick tests)

    # Data selection
    file_ids = ["rad_uhi_20241029_115057_4", "rad_uhi_20241029_115057_5"]
    track_start = 3039
    track_end = 4029

    # Create level spec
    levels = [
        LevelSpec(
            theta_range=theta_range,
            theta_step=step_theta,
            dx_range=dx_range,
            dy_range=dy_range,
            step_xy=step_dx,
            stride=stride,
            k=1,
            sample_frac=1.0,
            max_tries=None,
        )
    ]

    # Run
    best, aligner = auto_align_from_config(
        file_ids=file_ids,
        track_start=track_start,
        track_end=track_end,
        log_csv="./alignment_search_log.csv",
        levels=levels,
        gates=default_gates_small(),
        metric="pearson",
        viz_every=1,
        show_progress=True,
        show_final=True,
        plot_each_level=True,
        rng_seed=0,
    )

    print("\nBEST:", best)

    # Generate scatter plot
    print("\n" + "=" * 70)
    print("Generating density scatter plot...")
    print("=" * 70)
    aligner.plot_density_scatter(best, gridsize=80)

    # Plot MBES residuals with UHI footprint overlay - FINAL
    print("\n" + "=" * 70)
    print("FINAL: MBES residuals with UHI footprint overlay (AFTER alignment)")
    print("=" * 70)
    aligner.plot_mbes_with_uhi_footprint(
        aligner.mb,
        title_suffix=f" - FINAL (θ={best['theta']:+.2f}°, dx={best['dx']:+.2f}m, dy={best['dy']:+.2f}m)",
    )
    print("✅ MBES + UHI footprint plot generated!")
    print("\n💡 Compare this FINAL plot with the INITIAL plot:")
    print("   • INITIAL: Shows alignment before any corrections")
    print("   • FINAL: Shows alignment after optimization")
    print(
        "   • The BLACK OUTLINE (UHI footprint) should now overlap correctly with MBES features"
    )
    print(
        "   • If the footprint is still way off, there's a coordinate transformation issue."
    )

    # Keep plots open
    input("\nPress Enter to close all plots and exit...")
