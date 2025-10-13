#!/usr/bin/env python3
"""
Georeferenced HSI Plotting with UTC Timestamp Display

This module provides complete functionality for:
- Reading HDF5 files with georeferenced hyperspectral data
- Displaying RGB composites in LATLON, NED, or ECEF coordinates
- Showing UTC timestamp ranges for plotted data
- Interactive coordinate inspection on click

Key classes:
- GeoFile: Wraps single HDF5 file
- TransectDataSet: Discovers and manages multiple HDF5 files
- CombinedTransectCube: Combines multiple files into logical cube

Usage:
    import dev3_timestamp_complete as geo

    # Load data from folder
    transect = geo.load_transect("/path/to/h5/files")
    transect.list_files()

    # Select files and create combined cube
    cube = transect.select_all_files()

    # Plot with timestamps
    cube.plot_georef(
        coordinate_system="NED",
        track_start=100,
        track_end=200,
        interactive=True
    )
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io
import contextlib
import warnings

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pyproj import Transformer

# Import config from the correct location
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gref_pipeline"))
    import config
except Exception:
    config = None

# ============================================================================
# // Timestamp alignment: positive value means "show later" than raw UHI time.
UHI_TIMESTAMP_OFFSET_S = 508.0  # apply +508 s shift to UHI timestamps
# ============================================================================

# ============================================================================
# CRS Conversion Helpers
# ============================================================================


def _ecef_of_geodetic(
    lat_deg: float, lon_deg: float, h_m: float
) -> Tuple[float, float, float]:
    """Convert geodetic coordinates (lat, lon, height) to ECEF (X, Y, Z)."""
    tf = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = tf.transform(lon_deg, lat_deg, h_m)
    return float(x0), float(y0), float(z0)


def _ecef_to_ned_arrays(x, y, z, lat0, lon0, h0):
    """
    Vectorized ECEF -> NED (North-East-Down) conversion.

    Args:
        x, y, z: ECEF coordinates (arrays)
        lat0, lon0, h0: Reference point for local NED frame

    Returns:
        n, e, d: North, East, Down coordinates (arrays)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    x0, y0, z0 = _ecef_of_geodetic(lat0, lon0, h0)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat = np.radians(lat0)
    lon = np.radians(lon0)

    sL = np.sin(lat)
    cL = np.cos(lat)
    sO = np.sin(lon)
    cO = np.cos(lon)

    n = (-sL * cO) * dx + (-sL * sO) * dy + (cL) * dz
    e = (-sO) * dx + (cO) * dy
    d = (-cL * cO) * dx + (-cL * sO) * dy + (-sL) * dz
    return n, e, d


def unix_to_utc(timestamp):
    """Convert Unix timestamp (seconds) to UTC string."""
    from datetime import datetime

    try:
        return datetime.utcfromtimestamp(float(timestamp)).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except (ValueError, OSError, TypeError):
        return "N/A"


# ============================================================================
# HDF5 File Wrapper
# ============================================================================


class GeoFile:
    """
    Wraps a single HDF5 file containing georeferenced hyperspectral data.

    Expected datasets:
        - processed/radiance/dataCube: (T, S, B) radiance values
        - processed/radiance/dataCube_corrected: (T, S, B) corrected radiance (optional)
        - processed/radiance/calibration/spectral/band2Wavelength: (B,) wavelengths
        - processed/georef/points_ecef_crs: (T, S, 3) ECEF coordinates
        - processed/radiance/timestamps: (T,) Unix timestamps
    """

    # Dataset paths
    DSET_RGB_MAIN = "processed/radiance/dataCube"
    DSET_RGB_CORR = "processed/radiance/dataCube_corrected"
    DSET_WAVELEN = "processed/radiance/calibration/spectral/band2Wavelength"

    DSET_G_POINTS = "processed/georef/points_ecef_crs"  # preferred
    DSET_G_POINTS_ALT = "processed/georef/points_ecef"  # legacy fallback

    # Legacy flattened format
    DSET_G_FRAMES = "processed/georef/frame_indices"
    DSET_G_PIXELS = "processed/georef/ray_indices"
    DSET_G_FRAMES_GRID = "processed/georef/frame_nr_grid"
    DSET_G_PIXELS_GRID = "processed/georef/pixel_nr_grid"

    # Timestamp datasets (preferred first)
    DSET_TIMESTAMPS = [
        "processed/radiance/timestamp",  # singular
        "processed/radiance/timestamps",  # plural, fallback
        "processed/timestamp",
        "timestamp",
        "timestamps",
    ]

    def __init__(self, path: str, use_corrected: bool = False):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.use_corrected = use_corrected

        self.shape: Optional[Tuple[int, int, int]] = None  # (T, S, B)
        self.wavelengths: Optional[np.ndarray] = None
        self.has_georef: bool = False
        self.timestamps: Optional[np.ndarray] = None  # Unix timestamps per track

        self._check()

    def _check(self):
        """Read metadata from HDF5 file."""
        try:
            with h5py.File(self.path, "r") as f:
                # Radiance cube
                dset_cube = (
                    self.DSET_RGB_CORR if self.use_corrected else self.DSET_RGB_MAIN
                )
                if dset_cube not in f and self.DSET_RGB_MAIN in f:
                    dset_cube = self.DSET_RGB_MAIN
                    self.use_corrected = False
                if dset_cube in f:
                    self.shape = tuple(f[dset_cube].shape)  # (T,S,B)
                else:
                    self.shape = None

                # Wavelengths
                self.wavelengths = (
                    f[self.DSET_WAVELEN][()] if self.DSET_WAVELEN in f else None
                )

                # Georef presence
                pts_ok = (self.DSET_G_POINTS in f) or (self.DSET_G_POINTS_ALT in f)
                self.has_georef = bool(pts_ok)

                # Timestamps
                for ts_path in self.DSET_TIMESTAMPS:
                    if ts_path in f:
                        self.timestamps = np.asarray(f[ts_path][()]).ravel()
                        break

        except Exception as e:
            print(f"⚠️  Could not read metadata from {self.name}: {e}")
            self.shape = None
            self.wavelengths = None
            self.has_georef = False
            self.timestamps = None

    def _band_index(self, wl_nm: float) -> Optional[int]:
        """Find band index closest to requested wavelength."""
        if self.wavelengths is None:
            return None
        return int(np.argmin(np.abs(self.wavelengths - wl_nm)))

    def build_grids_and_rgb(
        self, red_wl=654.2, green_wl=560.0, blue_wl=440.3
    ) -> Dict[str, np.ndarray]:
        """
        Build georeferenced grids and extract RGB channels.

        Returns:
            dict with keys: X_ecef, Y_ecef, Z_ecef, R, G, B, T, S
            All arrays have shape (T, S) where T=tracks, S=spatial pixels
        """
        if self.shape is None:
            raise RuntimeError(f"{self.name}: radiance cube not found")
        T, S, B = self.shape

        dset_cube = self.DSET_RGB_CORR if self.use_corrected else self.DSET_RGB_MAIN

        X = np.full((T, S), np.nan, dtype=np.float64)
        Y = np.full((T, S), np.nan, dtype=np.float64)
        Z = np.full((T, S), np.nan, dtype=np.float64)
        R = np.full((T, S), np.nan, dtype=np.float64)
        G = np.full((T, S), np.nan, dtype=np.float64)
        Bc = np.full((T, S), np.nan, dtype=np.float64)

        with h5py.File(self.path, "r") as f:
            # Georef points
            if self.DSET_G_POINTS in f:
                P = f[self.DSET_G_POINTS][()]
            elif self.DSET_G_POINTS_ALT in f:
                P = f[self.DSET_G_POINTS_ALT][()]
            else:
                raise RuntimeError(f"{self.name}: no processed/georef points dataset")

            if P.ndim == 3 and P.shape[-1] == 3:
                # GRIDDED FORMAT: (T, S, 3)
                X[:, :], Y[:, :], Z[:, :] = P[..., 0], P[..., 1], P[..., 2]

                # Band indices
                iR = self._band_index(red_wl)
                iG = self._band_index(green_wl)
                iB = self._band_index(blue_wl)
                if iR is None or iG is None or iB is None:
                    raise RuntimeError(f"{self.name}: wavelengths not available")

                data = f[dset_cube][()]  # (T,S,B)
                R[:, :], G[:, :], Bc[:, :] = data[..., iR], data[..., iG], data[..., iB]

                # Mask invalid geometry
                invalid = ~(np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))
                R[invalid] = np.nan
                G[invalid] = np.nan
                Bc[invalid] = np.nan

            elif P.ndim == 2 and P.shape[1] == 3:
                # LEGACY FLAT FORMAT: (N,3) + indices
                if (self.DSET_G_FRAMES in f) and (self.DSET_G_PIXELS in f):
                    frames = f[self.DSET_G_FRAMES][()].astype(int)
                    slits = f[self.DSET_G_PIXELS][()].astype(int)
                elif (self.DSET_G_FRAMES_GRID in f) and (self.DSET_G_PIXELS_GRID in f):
                    frames_grid = f[self.DSET_G_FRAMES_GRID][()]
                    slits_grid = f[self.DSET_G_PIXELS_GRID][()]
                    if frames_grid.ndim == 2:
                        valid = np.isfinite(P[:, 0])
                        frames = frames_grid.ravel()[valid].astype(int)
                        slits = slits_grid.ravel()[valid].astype(int)
                    else:
                        frames = frames_grid.astype(int)
                        slits = slits_grid.astype(int)
                else:
                    raise RuntimeError(
                        f"{self.name}: flat points require frame_indices & pixel_indices"
                    )
                if len(frames) != len(P) or len(slits) != len(P):
                    raise RuntimeError(f"{self.name}: georef arrays length mismatch")

                iR = self._band_index(red_wl)
                iG = self._band_index(green_wl)
                iB = self._band_index(blue_wl)
                if iR is None or iG is None or iB is None:
                    raise RuntimeError(f"{self.name}: wavelengths not available")

                data = f[dset_cube]  # lazy

                for i in range(len(P)):
                    t = int(frames[i])
                    s = int(slits[i])
                    if 0 <= t < T and 0 <= s < S:
                        X[t, s] = P[i, 0]
                        Y[t, s] = P[i, 1]
                        Z[t, s] = P[i, 2]
                        R[t, s] = data[t, s, iR]
                        G[t, s] = data[t, s, iG]
                        Bc[t, s] = data[t, s, iB]
            else:
                raise RuntimeError(f"{self.name}: unexpected georef shape {P.shape}")

        return dict(X_ecef=X, Y_ecef=Y, Z_ecef=Z, R=R, G=G, B=Bc, T=T, S=S)


# ============================================================================
# Transect Management
# ============================================================================


class TransectDataSet:
    """Discovers and manages HDF5 files in a folder."""

    def __init__(self, folder: str, use_corrected: bool = False):
        self.folder = folder
        self.use_corrected = use_corrected
        self.files: Dict[str, GeoFile] = {}
        self._discover()

    def _discover(self):
        """Scan folder for valid HDF5 files."""
        if not os.path.isdir(self.folder):
            print(f"❌ Folder not found: {self.folder}")
            return
        for fn in sorted(os.listdir(self.folder)):
            if not fn.lower().endswith(".h5"):
                continue
            path = os.path.join(self.folder, fn)
            gf = GeoFile(path, use_corrected=self.use_corrected)
            if gf.shape is not None and gf.has_georef:
                self.files[gf.name] = gf
            else:
                if gf.shape is not None:
                    print(f"⚠️  {fn}: radiance found but no georef dataset – skipped")
                else:
                    print(f"⚠️  {fn}: no radiance cube – skipped")

    def list_files(self):
        """Print available files."""
        print(
            f"\n📋 Files in {os.path.basename(self.folder)} (corrected={self.use_corrected}):"
        )
        for i, (name, gf) in enumerate(self.files.items(), 1):
            print(f"{i:2d}. {name}  | shape={gf.shape}  | has_georef={gf.has_georef}")
        if not self.files:
            print("  (none)")

    def select_files(self, names: List[str]) -> "CombinedTransectCube":
        """Select specific files by name."""
        missing = [n for n in names if n not in self.files]
        if missing:
            print(f"⚠️  Missing files: {missing}")
        chosen = [self.files[n] for n in names if n in self.files]
        if not chosen:
            raise ValueError("No valid files selected")
        return CombinedTransectCube(chosen, self.folder, self.use_corrected)

    def select_all_files(self) -> "CombinedTransectCube":
        """Select all available files."""
        return self.select_files(list(self.files.keys()))

    def select_files_by_pattern(self, pattern: str) -> "CombinedTransectCube":
        """Select files matching glob pattern."""
        import fnmatch

        names = [n for n in self.files if fnmatch.fnmatch(n, pattern)]
        if not names:
            raise ValueError(f"No files match pattern: {pattern}")
        return self.select_files(names)


class CombinedTransectCube:
    """Combines multiple GeoFiles into a single logical cube along track."""

    def __init__(self, geofiles: List[GeoFile], folder: str, use_corrected: bool):
        self.geofiles = geofiles
        self.folder = folder
        self.use_corrected = use_corrected
        self.name = f"Combined_{os.path.basename(folder)}"
        self.file_boundaries = []  # list of dict with start_track etc.

        # Combined grids (filled in on demand)
        self.X_ecef: Optional[np.ndarray] = None
        self.Y_ecef: Optional[np.ndarray] = None
        self.Z_ecef: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.G: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.wavelengths = geofiles[0].wavelengths if geofiles else None
        self.timestamps: Optional[np.ndarray] = None  # Combined timestamps

        self._build_combined()

    def _build_combined(self):
        """Build combined grids from all selected files."""
        print("🔄 Rebuilding grids from georef hits for selected files...")
        xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
        timestamps_list = []
        T_accum = 0
        for gf in self.geofiles:
            print(f"   • {gf.name}")
            d = gf.build_grids_and_rgb()
            T, S = d["T"], d["S"]
            self.file_boundaries.append(
                {
                    "file": gf.name,
                    "start_track": T_accum,
                    "end_track": T_accum + T - 1,
                    "n_tracks": T,
                }
            )
            xs.append(d["X_ecef"])
            ys.append(d["Y_ecef"])
            zs.append(d["Z_ecef"])
            rs.append(d["R"])
            gs.append(d["G"])
            bs.append(d["B"])

            # Collect timestamps; prefer exact length T, else pad/truncate
            if gf.timestamps is not None:
                ts = np.asarray(gf.timestamps).ravel().astype(float)
                # === Apply global UHI time shift (+508 s) ===
                ts = ts + UHI_TIMESTAMP_OFFSET_S
                if len(ts) == T:
                    timestamps_list.append(ts)
                elif len(ts) > T:
                    timestamps_list.append(ts[:T])
                else:
                    pad = np.full(T, np.nan, dtype=float)
                    pad[: len(ts)] = ts
                    timestamps_list.append(pad)
            else:
                timestamps_list.append(np.full(T, np.nan, dtype=float))

            T_accum += T

        # Combine along track axis
        self.X_ecef = np.concatenate(xs, axis=0)
        self.Y_ecef = np.concatenate(ys, axis=0)
        self.Z_ecef = np.concatenate(zs, axis=0)
        self.R = np.concatenate(rs, axis=0)
        self.G = np.concatenate(gs, axis=0)
        self.B = np.concatenate(bs, axis=0)
        self.timestamps = (
            np.concatenate(timestamps_list, axis=0) if timestamps_list else None
        )

        print(f"✅ Combined shapes: X/Y/Z {self.X_ecef.shape}, RGB {self.R.shape}")

    def _rgb_from_wavelengths(self, Rnm, Gnm, Bnm, normalize):
        """Recompute R/G/B from requested wavelengths."""
        xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
        for gf in self.geofiles:
            d = gf.build_grids_and_rgb(red_wl=Rnm, green_wl=Gnm, blue_wl=Bnm)
            xs.append(d["X_ecef"])
            ys.append(d["Y_ecef"])
            zs.append(d["Z_ecef"])
            rs.append(d["R"])
            gs.append(d["G"])
            bs.append(d["B"])
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        Z = np.concatenate(zs, axis=0)
        R = np.concatenate(rs, axis=0)
        G = np.concatenate(gs, axis=0)
        B = np.concatenate(bs, axis=0)

        if normalize:
            for C in (R, G, B):
                m, M = np.nanmin(C), np.nanmax(C)
                if np.isfinite(m) and np.isfinite(M) and M > m:
                    C[:] = (C - m) / (M - m)
        return X, Y, Z, R, G, B

    def _extract_rgb_from_cube(self, data_cube, Rnm, Gnm, Bnm):
        """Extract RGB channels from a full data cube (T, S, B) by wavelength."""
        gf = self.geofiles[0]
        iR = gf._band_index(Rnm)
        iG = gf._band_index(Gnm)
        iB = gf._band_index(Bnm)
        if iR is None or iG is None or iB is None:
            raise RuntimeError("Requested wavelengths not available")
        R = data_cube[:, :, iR]
        G = data_cube[:, :, iG]
        B = data_cube[:, :, iB]
        return R, G, B

    def plot_georef(
        self,
        red_wl=654.2,
        green_wl=560.0,
        blue_wl=440.3,
        normalize=True,
        figsize=(11, 9),
        coordinate_system=None,  # "ECEF", "NED", "LATLON", or None (auto = LATLON)
        use_local_origin=True,  # only used when coordinate_system == "ECEF"
        origin=None,  # (lat, lon, h); required for NED/ECEF, ignored for LATLON
        show_file_boundaries=True,
        alpha_for_nodata=0.0,
        interactive=True,  # click to show coordinates
        track_start=None,  # inclusive
        track_end=None,  # exclusive
        use_corrected=False,  # use illumination corrected data if available
        # Trajectory options (kept for compatibility)
        show_trajectory=False,
        nav_csv_path=None,
        trajectory_color="red",
        trajectory_linewidth=2.0,
        trajectory_alpha=0.85,
        trajectory_decimate=1,
        return_fig=False,
        quiet=True,  # suppress non interactive prints and warnings
        **pcolor_kwargs,
    ):
        """
        Plot RGB georeferenced composite with UTC timestamp display.

        Args:
            red_wl, green_wl, blue_wl: Wavelengths (nm) for RGB channels
            normalize: Normalize each channel to [0,1]
            figsize: Figure size (width, height) in inches
            coordinate_system: "LATLON", "NED", or "ECEF"
            use_local_origin: If True (ECEF only), plot relative to origin
            origin: (lat, lon, h) reference point for NED/ECEF
            show_file_boundaries: Draw yellow dotted lines at file boundaries
            alpha_for_nodata: Alpha value for pixels with no data
            interactive: Enable click-to-inspect functionality
            track_start, track_end: Track range to plot (None = all)
            use_corrected: Use illumination-corrected data if available
            quiet: Suppress warnings
            **pcolor_kwargs: Additional arguments for pcolormesh

        Returns:
            (fig, ax) if return_fig=True, else None
        """

        def _silence():
            return contextlib.ExitStack()

        # Default display CRS
        if coordinate_system is None:
            coordinate_system = "LATLON"
        coord_sys = coordinate_system.upper()

        # Default origin for NED/ECEF
        if coord_sys in ["NED", "ECEF"]:
            if origin is None:
                if (
                    (config is not None)
                    and hasattr(config, "LAT0")
                    and hasattr(config, "LON0")
                ):
                    origin = (
                        float(config.LAT0),
                        float(config.LON0),
                        float(getattr(config, "H0", 0.0)),
                    )
                else:
                    origin = (60.8011575, 10.7122345, 0.0)
        else:
            origin = None

        with _silence() as stack:
            if quiet:
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("ignore")
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

            # Prepare grids & RGB
            if use_corrected:
                if not hasattr(self, "data_corrected") or self.data_corrected is None:
                    raise ValueError(
                        "Corrected data not available. Run apply_illumination_correction() first."
                    )
                X_ecef, Y_ecef, Z_ecef = self.X_ecef, self.Y_ecef, self.Z_ecef
                R, G, B = self._extract_rgb_from_cube(
                    self.data_corrected, red_wl, green_wl, blue_wl
                )
                if normalize:
                    for C in (R, G, B):
                        m, M = np.nanmin(C), np.nanmax(C)
                        if np.isfinite(m) and np.isfinite(M) and M > m:
                            C[:] = (C - m) / (M - m)
            else:
                X_ecef, Y_ecef, Z_ecef, R, G, B = self._rgb_from_wavelengths(
                    red_wl, green_wl, blue_wl, normalize
                )

        T_total, S = X_ecef.shape

        # Slice by track_start/end
        start_idx = track_start if track_start is not None else 0
        end_idx = track_end if track_end is not None else T_total
        if start_idx < 0 or start_idx >= T_total:
            raise ValueError(f"track_start={start_idx} out of range [0, {T_total})")
        if end_idx <= start_idx or end_idx > T_total:
            raise ValueError(f"track_end={end_idx} must be in ({start_idx}, {T_total}]")

        X_ecef = X_ecef[start_idx:end_idx, :]
        Y_ecef = Y_ecef[start_idx:end_idx, :]
        Z_ecef = Z_ecef[start_idx:end_idx, :]
        R = R[start_idx:end_idx, :]
        G = G[start_idx:end_idx, :]
        B = B[start_idx:end_idx, :]
        T, S = X_ecef.shape

        # Build display coordinates
        if coord_sys == "LATLON":
            tf_ecef_to_geo = Transformer.from_crs(
                "EPSG:4978", "EPSG:4979", always_xy=True
            )
            lon, lat, height = tf_ecef_to_geo.transform(X_ecef, Y_ecef, Z_ecef)
            Xp, Yp = lon, lat
            xlabel, ylabel = "Longitude (°)", "Latitude (°)"
            valid_mask = np.isfinite(lon) & np.isfinite(lat)
            if valid_mask.any():
                lat0 = float(np.nanmean(lat[valid_mask]))
                lon0 = float(np.nanmean(lon[valid_mask]))
                title_origin = (lat0, lon0, 0.0)
            else:
                title_origin = (0.0, 0.0, 0.0)
        elif coord_sys == "NED":
            lat0, lon0, h0 = origin
            N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)
            Xp, Yp = E, N
            xlabel, ylabel = f"East (m) from {lat0}°, {lon0}°", "North (m)"
            title_origin = origin
        elif coord_sys == "ECEF":
            lat0, lon0, h0 = origin
            if use_local_origin:
                x0, y0, z0 = _ecef_of_geodetic(lat0, lon0, h0)
                Xp, Yp = X_ecef - x0, Y_ecef - y0
                xlabel, ylabel = f"ΔECEF X (m) from {lat0}°, {lon0}°", "ΔECEF Y (m)"
            else:
                Xp, Yp = X_ecef, Y_ecef
                xlabel, ylabel = "ECEF X (m)", "ECEF Y (m)"
            title_origin = origin
        else:
            raise ValueError("coordinate_system must be 'LATLON', 'NED', or 'ECEF'")

        # Compose RGB stack for plotting
        RGB = np.dstack([R, G, B]).astype(np.float64)
        alpha = np.ones((T, S), dtype=np.float64)
        alpha[~(np.isfinite(R) & np.isfinite(G) & np.isfinite(B))] = alpha_for_nodata
        RGB[alpha == 0] = np.nan

        # pcolormesh wants corners
        Xc = np.pad(Xp, ((0, 1), (0, 1)), mode="edge")
        Yc = np.pad(Yp, ((0, 1), (0, 1)), mode="edge")

        # Fill any invalid corners via nearest neighbor (robustness)
        mask_valid = np.isfinite(Xc) & np.isfinite(Yc)
        if not mask_valid.all():
            from scipy.ndimage import distance_transform_edt

            invalid_mask = ~mask_valid
            if invalid_mask.any():
                idx = distance_transform_edt(
                    invalid_mask, return_distances=False, return_indices=True
                )
                Xc[invalid_mask] = Xc[tuple(idx[:, invalid_mask])]
                Yc[invalid_mask] = Yc[tuple(idx[:, invalid_mask])]

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.pcolormesh(Xc, Yc, RGB, shading="flat", **pcolor_kwargs)

        # File boundaries (only those inside the shown slice)
        if show_file_boundaries and len(self.file_boundaries) > 1:
            for b in self.file_boundaries[1:]:
                j_abs = b["start_track"]
                if start_idx <= j_abs < end_idx:
                    j = j_abs - start_idx
                    if 0 <= j < T:
                        ax.plot(
                            Xp[j, :],
                            Yp[j, :],
                            color="yellow",
                            linestyle=":",
                            linewidth=2,
                            alpha=0.9,
                        )

        # Aspect
        if coord_sys == "LATLON":
            if title_origin and title_origin[0] != 0:
                lat_center = title_origin[0]
                ax.set_aspect(1.0 / np.cos(np.radians(lat_center)), adjustable="box")
            else:
                ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("equal", adjustable="box")

        # Title (with timestamp range if available)
        title = f"RGB georeferenced composite ({self.name}) [{coord_sys}]"
        if coord_sys == "LATLON":
            if title_origin:
                title += f"\ncenter ~ {title_origin[0]:.4f}°, {title_origin[1]:.4f}°"
        elif coord_sys == "NED" or (coord_sys == "ECEF" and use_local_origin):
            title += f"\norigin @ {title_origin[0]:.6f}°, {title_origin[1]:.6f}°"

        # Add UTC time range based on the plotted slice [start_idx:end_idx)
        if self.timestamps is not None and len(self.timestamps) > 0:
            ts = self.timestamps[max(0, start_idx) : max(0, end_idx)]
            if ts.size > 0:
                start_time = next((unix_to_utc(v) for v in ts if np.isfinite(v)), "N/A")
                end_time = next(
                    (unix_to_utc(v) for v in ts[::-1] if np.isfinite(v)), "N/A"
                )
                title += f"\nTime: {start_time} → {end_time}"

        if interactive:
            title += "\n(Click to show coordinates)"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Interactivity: click to print coordinates
        if interactive:
            tf_ecef_to_geo = Transformer.from_crs(
                "EPSG:4978", "EPSG:4979", always_xy=True
            )
            click_data = {
                "Xp": Xp,
                "Yp": Yp,
                "X_ecef": X_ecef,
                "Y_ecef": Y_ecef,
                "Z_ecef": Z_ecef,
                "R": R,
                "G": G,
                "B": B,
                "origin": origin,
                "coord_system": coord_sys,
                "use_local_origin": use_local_origin,
                "track_start_abs": start_idx,
            }

            def on_click(event):
                if event.inaxes is None:
                    return
                x_click, y_click = event.xdata, event.ydata
                dist = (click_data["Xp"] - x_click) ** 2 + (
                    click_data["Yp"] - y_click
                ) ** 2
                try:
                    min_idx = np.nanargmin(dist)
                except Exception:
                    return
                ti, si = np.unravel_index(min_idx, click_data["Xp"].shape)
                x_plot = click_data["Xp"][ti, si]
                y_plot = click_data["Yp"][ti, si]
                x_e = click_data["X_ecef"][ti, si]
                y_e = click_data["Y_ecef"][ti, si]
                z_e = click_data["Z_ecef"][ti, si]
                r_val = click_data["R"][ti, si]
                g_val = click_data["G"][ti, si]
                b_val = click_data["B"][ti, si]
                if not (np.isfinite(x_e) and np.isfinite(y_e) and np.isfinite(z_e)):
                    return
                lon_deg, lat_deg, height = tf_ecef_to_geo.transform(x_e, y_e, z_e)
                print("\n" + "=" * 70)
                print(f"📍 track={click_data['track_start_abs'] + ti}, slit={si}")
                if click_data["coord_system"] == "LATLON":
                    print(f"   Lat/Lon: {y_plot:.6f}°, {x_plot:.6f}°")
                elif click_data["coord_system"] == "NED":
                    print(f"   NED: E={x_plot:.2f} m, N={y_plot:.2f} m")
                else:
                    print(f"   Plot coords: ({x_plot:.2f}, {y_plot:.2f})")
                print(f"   ECEF: X={x_e:.2f} m, Y={y_e:.2f} m, Z={z_e:.2f} m")
                print(
                    f"   WGS84: Lat={lat_deg:.6f}°, Lon={lon_deg:.6f}°, h={height:.2f} m"
                )
                if np.isfinite(r_val):
                    print(f"   RGB: R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")
                else:
                    print("   RGB: No data")
                print("=" * 70)

            fig.canvas.mpl_connect("button_press_event", on_click)
            print("💡 Interactive mode: Click on the plot to display coordinates")

        fig.tight_layout()
        plt.show()
        return (fig, ax) if return_fig else None


# ============================================================================
# Convenience Functions
# ============================================================================


def load_transect(
    folder_path: Optional[str] = None, use_corrected: bool = False
) -> TransectDataSet:
    """
    Creates a TransectDataSet scanning the folder for H5s with georef outputs.

    Args:
        folder_path: Path to folder containing HDF5 files (if None, uses config.OUTPUT_FOLDER)
        use_corrected: Use corrected radiance data if available

    Returns:
        TransectDataSet instance
    """
    if folder_path is None and config is not None and hasattr(config, "OUTPUT_FOLDER"):
        folder_path = config.OUTPUT_FOLDER
    if folder_path is None:
        raise ValueError("Provide folder_path or set config.OUTPUT_FOLDER")
    return TransectDataSet(folder_path, use_corrected=use_corrected)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Georeferenced HSI Plotting with Timestamps")
    print("=" * 70)

    # Use config.OUTPUT_FOLDER
    if config is not None and hasattr(config, "OUTPUT_FOLDER"):
        FOLDER_PATH = config.OUTPUT_FOLDER
        print(f"\n✅ Using config.OUTPUT_FOLDER:\n   {FOLDER_PATH}\n")
    else:
        print("\n❌ Could not load config.OUTPUT_FOLDER")
        print("Make sure config.py is in ../gref_pipeline/")
        sys.exit(1)

    try:
        transect = load_transect(folder_path=FOLDER_PATH)
        transect.list_files()

        if not transect.files:
            print("\n❌ No valid HDF5 files found!")
            print(f"Check folder: {FOLDER_PATH}")
        else:
            # Select all files
            cube = transect.select_files(
                [
                    "rad_uhi_20241029_115057_1",
                    "rad_uhi_20241029_115057_2",
                    "rad_uhi_20241029_115057_3",
                    "rad_uhi_20241029_115057_4",
                    "rad_uhi_20241029_115057_5",
                ]
            )

            # Plot with timestamps
            print("\n📊 Plotting georeferenced composite with UTC timestamps...")
            cube.plot_georef(
                coordinate_system="NED",
                track_start=8637,
                track_end=9709,
                interactive=True,
            )
            print("\n✅ Done! Click on pixels to inspect coordinates.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
