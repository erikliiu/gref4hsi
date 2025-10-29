import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, List, Optional, Tuple
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pyproj import Transformer
import json
from datetime import datetime
from types import SimpleNamespace

try:
    from gref_pipeline import config
except Exception:
    config = None

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ------------------------- CRS helpers -------------------------


def _ecef_of_geodetic(
    lat_deg: float, lon_deg: float, h_m: float
) -> Tuple[float, float, float]:
    tf = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = tf.transform(lon_deg, lat_deg, h_m)
    return float(x0), float(y0), float(z0)


def _ecef_to_ned_arrays(x, y, z, lat0, lon0, h0):
    """Vectorized ECEF -> NED (returns north, east, down) relative to (lat0, lon0, h0)."""
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
    """Convert unix timestamp to UTC string format, applying TIME_OFFSET_SEC from config."""
    from datetime import datetime

    try:
        # Apply time offset from config if available
        offset = getattr(config, "TIME_OFFSET_SEC", 0) if config is not None else 0
        corrected_timestamp = float(timestamp) + float(offset)
        return datetime.utcfromtimestamp(corrected_timestamp).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except (ValueError, OSError):
        return "N/A"


# ------------------------- Data wrappers -------------------------


class GeoFile:
    """
    Wraps a single H5 file containing:
      - processed/radiance/dataCube(_corrected): (T, S, B)
      - processed/radiance/calibration/spectral/band2Wavelength
      - processed/georef/points_ecef_crs: (N, 3) or (T, S, 3)
      - processed/georef/frame_indices: (N,)
      - processed/georef/ray_indices: (N,)  [slit index]
    """

    # dataset names
    DSET_RGB_MAIN = "processed/radiance/dataCube"
    DSET_RGB_CORR = "processed/radiance/dataCube_corrected"
    DSET_WAVELEN = "processed/radiance/calibration/spectral/band2Wavelength"
    DSET_G_POINTS = "processed/georef/points_ecef_crs"  # preferred
    DSET_G_POINTS_ALT = "processed/georef/points_ecef"  # fallback
    # Support both old flattened and new gridded formats
    DSET_G_FRAMES = "processed/georef/frame_indices"  # old flattened format
    DSET_G_PIXELS = "processed/georef/ray_indices"  # old flattened format (slit)
    DSET_G_FRAMES_GRID = "processed/georef/frame_nr_grid"  # new gridded format
    DSET_G_PIXELS_GRID = "processed/georef/pixel_nr_grid"  # new gridded format (slit)
    # Timestamp datasets (common locations in H5 files)
    DSET_TIMESTAMPS = [
        "processed/radiance/timestamps",
        "processed/timestamp",
        "timestamp",
        "timestamps",
    ]

    def __init__(self, path: str, use_corrected: bool = False):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.use_corrected = use_corrected

        self.shape = None  # (T, S, B)
        self.wavelengths = None
        self.has_georef = False
        self.timestamps = None  # Unix timestamps for each track
        self._check()

    def _check(self):
        try:
            with h5py.File(self.path, "r") as f:
                # radiance cube
                dset_cube = (
                    self.DSET_RGB_CORR if self.use_corrected else self.DSET_RGB_MAIN
                )
                if dset_cube not in f and self.DSET_RGB_MAIN in f:
                    # fallback to main if corrected missing
                    dset_cube = self.DSET_RGB_MAIN
                    self.use_corrected = False
                if dset_cube in f:
                    self.shape = tuple(f[dset_cube].shape)
                else:
                    self.shape = None

                # wavelengths
                if self.DSET_WAVELEN in f:
                    self.wavelengths = f[self.DSET_WAVELEN][()]
                else:
                    self.wavelengths = None

                # georef presence
                pts_ok = (self.DSET_G_POINTS in f) or (self.DSET_G_POINTS_ALT in f)
                idx_ok = (self.DSET_G_FRAMES in f) and (self.DSET_G_PIXELS in f)
                idx_grid_ok = (self.DSET_G_FRAMES_GRID in f) and (
                    self.DSET_G_PIXELS_GRID in f
                )
                # If points are a (T,S,3) grid, indices aren't required, but we still accept them
                self.has_georef = pts_ok

                # Load timestamps if available
                for ts_path in self.DSET_TIMESTAMPS:
                    if ts_path in f:
                        self.timestamps = f[ts_path][:]
                        break

        except Exception as e:
            print(f"⚠️  Could not read metadata from {self.name}: {e}")
            self.shape = None
            self.wavelengths = None
            self.has_georef = False
            self.timestamps = None

    def _band_index(self, wl_nm: float) -> Optional[int]:
        if self.wavelengths is None:
            return None
        return int(np.argmin(np.abs(self.wavelengths - wl_nm)))

    def build_grids_and_rgb(
        self,
        red_wl=654.2,
        green_wl=560.0,
        blue_wl=440.3,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with:
            X_ecef, Y_ecef, Z_ecef : (T, S)
            R, G, B                 : (T, S) floats (NaN where no hit)
            T, S                    : ints
        All grids have matching (T, S) derived from the radiance cube.
        """
        if self.shape is None:
            raise RuntimeError(f"{self.name}: radiance cube not found")

        T, S, B = self.shape

        # ------- read radiance cube lazily for sampling -------
        dset_cube = self.DSET_RGB_CORR if self.use_corrected else self.DSET_RGB_MAIN

        # ------- prepare empty grids -------
        X = np.full((T, S), np.nan, dtype=np.float64)
        Y = np.full((T, S), np.nan, dtype=np.float64)
        Z = np.full((T, S), np.nan, dtype=np.float64)
        R = np.full((T, S), np.nan, dtype=np.float64)
        G = np.full((T, S), np.nan, dtype=np.float64)
        Bc = np.full((T, S), np.nan, dtype=np.float64)

        # ------- read georef -------
        with h5py.File(self.path, "r") as f:
            # points
            if self.DSET_G_POINTS in f:
                P = f[self.DSET_G_POINTS][()]
            elif self.DSET_G_POINTS_ALT in f:
                P = f[self.DSET_G_POINTS_ALT][()]
            else:
                raise RuntimeError(f"{self.name}: no processed/georef points dataset")

            # two possible shapes
            if P.ndim == 3 and P.shape[-1] == 3:
                # GRIDDED FORMAT: (T, S, 3) - NEW main.py output
                print(
                    f"  {self.name}: Using GRIDDED format (T={P.shape[0]}, S={P.shape[1]})"
                )

                # Extract coordinates directly from grid
                X[:, :], Y[:, :], Z[:, :] = P[..., 0], P[..., 1], P[..., 2]

                # Get band indices for RGB
                idxR = self._band_index(red_wl)
                idxG = self._band_index(green_wl)
                idxB = self._band_index(blue_wl)
                if idxR is None or idxG is None or idxB is None:
                    raise RuntimeError(f"{self.name}: wavelengths not available")

                # Sample radiance cube for RGB values
                data = f[dset_cube][()]  # Load full cube

                # Extract RGB bands
                R[:, :] = data[..., idxR]
                G[:, :] = data[..., idxG]
                Bc[:, :] = data[..., idxB]

                # Set NaN for invalid geometry (where ECEF points are NaN)
                invalid_mask = ~(np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))
                R[invalid_mask] = np.nan
                G[invalid_mask] = np.nan
                Bc[invalid_mask] = np.nan

            elif P.ndim == 2 and P.shape[1] == 3:
                # FLATTENED FORMAT: (N, 3) + indices -> rebuild grids (OLD format)
                print(f"  {self.name}: Using FLATTENED format (N={P.shape[0]} points)")

                # Try old naming first, then new gridded naming
                if (self.DSET_G_FRAMES in f) and (self.DSET_G_PIXELS in f):
                    frames = f[self.DSET_G_FRAMES][()].astype(int)
                    slits = f[self.DSET_G_PIXELS][()].astype(int)
                elif (self.DSET_G_FRAMES_GRID in f) and (self.DSET_G_PIXELS_GRID in f):
                    # Gridded indices might be (T, S) - flatten them
                    frames_grid = f[self.DSET_G_FRAMES_GRID][()]
                    slits_grid = f[self.DSET_G_PIXELS_GRID][()]
                    if frames_grid.ndim == 2:
                        # Grid format - extract valid indices
                        valid_mask = np.isfinite(
                            P[:, 0]
                        )  # Valid points have finite coords
                        # This shouldn't happen for flattened points, but handle it anyway
                        frames = frames_grid.ravel()[valid_mask].astype(int)
                        slits = slits_grid.ravel()[valid_mask].astype(int)
                    else:
                        frames = frames_grid.astype(int)
                        slits = slits_grid.astype(int)
                else:
                    raise RuntimeError(
                        f"{self.name}: flat points require frame_indices & pixel_indices"
                    )
                if len(frames) != len(P) or len(slits) != len(P):
                    raise RuntimeError(f"{self.name}: georef arrays length mismatch")

                # band indices
                idxR = self._band_index(red_wl)
                idxG = self._band_index(green_wl)
                idxB = self._band_index(blue_wl)
                if idxR is None or idxG is None or idxB is None:
                    raise RuntimeError(f"{self.name}: wavelengths not available")

                # sample radiance lazily row-by-row (memory-safe)
                data = f[dset_cube]

                # If multiple hits happened for same (frame, slit), keep the last (or avg – here last wins)
                for i in range(len(P)):
                    t = int(frames[i])
                    s = int(slits[i])
                    if 0 <= t < T and 0 <= s < S:
                        X[t, s] = P[i, 0]
                        Y[t, s] = P[i, 1]
                        Z[t, s] = P[i, 2]
                        R[t, s] = data[t, s, idxR]
                        G[t, s] = data[t, s, idxG]
                        Bc[t, s] = data[t, s, idxB]
                # done
            else:
                raise RuntimeError(f"{self.name}: unexpected georef shape {P.shape}")

        return dict(X_ecef=X, Y_ecef=Y, Z_ecef=Z, R=R, G=G, B=Bc, T=T, S=S)


# ------------------------- Transect containers -------------------------


class TransectDataSet:
    """Discovers H5s in a folder and exposes selection like your previous helper."""

    def __init__(self, folder: str, use_corrected: bool = False):
        self.folder = folder
        self.use_corrected = use_corrected
        self.files: Dict[str, GeoFile] = {}
        self._discover()

    def _discover(self):
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
                # still allow if radiance exists but georef missing?
                if gf.shape is not None:
                    print(f"⚠️  {fn}: radiance found but no georef dataset – skipped")
                else:
                    print(f"⚠️  {fn}: no radiance cube – skipped")

    def list_files(self):
        print(
            f"\n📋 Files in {os.path.basename(self.folder)} (corrected={self.use_corrected}):"
        )
        for i, (name, gf) in enumerate(self.files.items(), 1):
            print(f"{i:2d}. {name}  | shape={gf.shape}  | has_georef={gf.has_georef}")
        if not self.files:
            print("  (none)")

    def select_files(
        self, names: List[str], normalize_per_file: bool = False
    ) -> "CombinedTransectCube":
        """
        Select files and combine them into a CombinedTransectCube.

        Parameters:
        -----------
        names : List[str]
            List of file names (without .h5 extension) to select
        normalize_per_file : bool, default=True
            If True, normalize each file's RGB to [0,1] before combining to avoid
            color discontinuities from different acquisition conditions
        """
        missing = [n for n in names if n not in self.files]
        if missing:
            print(f"⚠️  Missing files: {missing}")
        chosen = [self.files[n] for n in names if n in self.files]
        if not chosen:
            raise ValueError("No valid files selected")
        return CombinedTransectCube(
            chosen, self.folder, self.use_corrected, normalize_per_file
        )

    def select_all_files(
        self, normalize_per_file: bool = False
    ) -> "CombinedTransectCube":
        return self.select_files(
            list(self.files.keys()), normalize_per_file=normalize_per_file
        )

    def select_files_by_pattern(
        self, pattern: str, normalize_per_file: bool = False
    ) -> "CombinedTransectCube":
        import fnmatch

        names = [n for n in self.files if fnmatch.fnmatch(n, pattern)]
        if not names:
            raise ValueError(f"No files match pattern: {pattern}")
        return self.select_files(names, normalize_per_file=normalize_per_file)


class CombinedTransectCube:
    """Combines multiple GeoFiles into a single logical cube along track."""

    def __init__(
        self,
        geofiles: List[GeoFile],
        folder: str,
        use_corrected: bool,
        normalize_per_file: bool = False,
    ):
        self.geofiles = geofiles
        self.folder = folder
        self.use_corrected = use_corrected
        self.normalize_per_file = (
            normalize_per_file  # NEW: normalize each file's RGB independently
        )
        self.name = f"Combined_{os.path.basename(folder)}"
        self.file_boundaries = []  # list of dict with start_track etc.

        # combined grids (filled in on demand)
        self.X_ecef = None
        self.Y_ecef = None
        self.Z_ecef = None
        self.R = None
        self.G = None
        self.B = None
        self.wavelengths = geofiles[0].wavelengths if geofiles else None
        self.timestamps = None  # Combined timestamps from all files

        # Full hyperspectral cube (loaded on demand for methods that need it)
        self.data = None  # Shape: (T_combined, S, B)
        self.data_corrected = (
            None  # Corrected version if illumination correction applied
        )

        # NEW: Persistent color mapping for ROIs (shared across plot functions)
        self.roi_color_map = {}  # Will auto-populate when plotting ROIs

        self._build_combined()
        self._load_full_cube()  # Load the full hyperspectral data

    def _build_combined(self):
        print("🔄 Rebuilding grids from georef hits for selected files...")
        xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
        timestamps_list = []
        T_accum = 0
        for i, gf in enumerate(self.geofiles):
            print(f"   • {gf.name}")
            d = (
                gf.build_grids_and_rgb()
            )  # default RGB wavelengths; you can re-run with kwargs in plot
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

            # NEW: Optionally normalize each file's RGB independently
            R_file, G_file, B_file = d["R"], d["G"], d["B"]
            if self.normalize_per_file:
                for C in (R_file, G_file, B_file):
                    m, M = np.nanmin(C), np.nanmax(C)
                    if np.isfinite(m) and np.isfinite(M) and M > m:
                        C[:] = (C - m) / (M - m)
                print(f"      ✓ Normalized RGB to [0,1]")

            rs.append(R_file)
            gs.append(G_file)
            bs.append(B_file)

            # Collect timestamps if available
            if gf.timestamps is not None and len(gf.timestamps) == T:
                timestamps_list.append(gf.timestamps)
            elif gf.timestamps is not None:
                # Timestamps exist but don't match track count, pad or truncate
                if len(gf.timestamps) > T:
                    timestamps_list.append(gf.timestamps[:T])
                else:
                    # Pad with NaN if too few
                    padded = np.full(T, np.nan)
                    padded[: len(gf.timestamps)] = gf.timestamps
                    timestamps_list.append(padded)
            else:
                # No timestamps for this file, use NaN
                timestamps_list.append(np.full(T, np.nan))

            T_accum += T

        # combine along track axis
        self.X_ecef = np.concatenate(xs, axis=0)
        self.Y_ecef = np.concatenate(ys, axis=0)
        self.Z_ecef = np.concatenate(zs, axis=0)
        self.R = np.concatenate(rs, axis=0)
        self.G = np.concatenate(gs, axis=0)
        self.B = np.concatenate(bs, axis=0)

        # Combine timestamps
        if timestamps_list:
            self.timestamps = np.concatenate(timestamps_list, axis=0)
        else:
            self.timestamps = None

        print(f"✅ Combined shapes: X/Y/Z {self.X_ecef.shape}, RGB {self.R.shape}")

    def _load_full_cube(self):
        """
        Load the full hyperspectral cube (T, S, B) from all H5 files.
        This enables methods like plot_rgb() with custom wavelengths, plot_spectrum(), etc.
        """
        print("🔄 Loading full hyperspectral cube...")
        cubes = []

        dset_name = (
            "processed/radiance/dataCube_corrected"
            if self.use_corrected
            else "processed/radiance/dataCube"
        )

        for gf in self.geofiles:
            print(f"   • Loading {gf.name}...")
            try:
                with h5py.File(gf.path, "r") as f:
                    # Try corrected first, fallback to raw
                    if dset_name in f:
                        cube = f[dset_name][()]
                    elif "processed/radiance/dataCube" in f:
                        cube = f["processed/radiance/dataCube"][()]
                    else:
                        raise KeyError(f"No radiance cube found in {gf.name}")

                    cubes.append(cube)
            except Exception as e:
                print(f"      ⚠️  Failed to load: {e}")
                return  # Exit if we can't load all cubes

        # Concatenate along track axis
        self.data = np.concatenate(cubes, axis=0)
        print(f"✅ Loaded full cube: {self.data.shape} (T × S × B)")

    def describe(self):
        T, S = self.X_ecef.shape
        print(f"\n=== {self.name} ===")
        print(f"Tracks × Slits: {T} × {S}")
        print(
            f"RGB coverage: R/G/B finite ratios "
            f"{np.isfinite(self.R).mean():.2f}/{np.isfinite(self.G).mean():.2f}/{np.isfinite(self.B).mean():.2f}"
        )
        print("\n📋 File boundaries:")
        for b in self.file_boundaries:
            print(
                f"  {b['file']}: tracks {b['start_track']}–{b['end_track']} ({b['n_tracks']})"
            )

    def adjust_uhi_alignment(self, dx=0.0, dy=0.0):
        """
        Adjust UHI alignment offset by setting config.UHI_ALIGNMENT_DX/DY.

        This shifts the UHI data in NED coordinates when using plot_georef with
        apply_alignment_shift=True (default). Useful for fine-tuning alignment
        with MBES or other reference data.

        Parameters:
        -----------
        dx : float
            East shift in meters (added to E coordinate in NED)
        dy : float
            North shift in meters (added to N coordinate in NED)

        Example:
        --------
        # Shift UHI data 0.05m west and 3m south
        cube.adjust_uhi_alignment(dx=-0.05, dy=-3.0)

        # Then plot with alignment applied (default behavior)
        cube.plot_georef(use_corrected=True, coordinate_system='NED')
        """
        try:
            from gref_pipeline import config

            config.UHI_ALIGNMENT_DX = dx
            config.UHI_ALIGNMENT_DY = dy

            print(f"✅ UHI alignment adjusted:")
            print(f"   dx (East):  {dx:+.3f} m")
            print(f"   dy (North): {dy:+.3f} m")
            print(
                f"\n💡 This will be applied in plot_georef() with coordinate_system='NED'"
            )
        except ImportError:
            print("❌ Could not import gref_pipeline.config")
        except Exception as e:
            print(f"❌ Error setting alignment: {e}")

    @staticmethod
    def _read_mbes_geotiff_as_epsg(geotiff_path: str, target_epsg: int):
        """
        Read MBES GeoTIFF band-1 and reproject to target_epsg if needed.
        Returns (arr[nan where nodata], affine_transform, used_epsg).
        """
        with rasterio.open(geotiff_path) as src:
            src_epsg = src.crs.to_epsg() if src.crs else None
            if src_epsg == target_epsg:
                arr = src.read(1, masked=True).filled(np.nan)
                return arr, src.transform, src_epsg
            # reproject
            dst_crs = rasterio.crs.CRS.from_epsg(target_epsg)
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            dst = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
            return dst, transform, target_epsg

    @staticmethod
    def _xy_from_affine_grid(transform, h, w):
        """
        Build pixel-center X,Y grids from an affine transform (UTM/Projected meters).
        Affine: x = c + a*col + b*row, y = f + d*col + e*row
        """
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        X = transform.c + cc * transform.a + rr * transform.b + transform.a * 0.5
        Y = transform.f + cc * transform.d + rr * transform.e + transform.e * 0.5
        return X, Y

    # ------------------- plotting -------------------

    def _rgb_from_wavelengths(self, Rnm, Gnm, Bnm, normalize):
        """Recompute R/G/B from requested wavelengths (without rebuilding georef)."""
        # We have to resample the radiance cube again per file with new band indices.
        # For speed, keep the original intial RGB if wavelengths didn’t change.
        # Here for simplicity we rebuild per-file and re-concatenate.
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
        # Get wavelengths from first file (assume all files have same wavelengths)
        gf = self.geofiles[0]

        # Find band indices
        idxR = gf._band_index(Rnm)
        idxG = gf._band_index(Gnm)
        idxB = gf._band_index(Bnm)

        if idxR is None or idxG is None or idxB is None:
            raise RuntimeError(f"Requested wavelengths not available")

        # Extract channels
        R = data_cube[:, :, idxR]
        G = data_cube[:, :, idxG]
        B = data_cube[:, :, idxB]

        return R, G, B

    def _get_roi_color(self, roi_name, default_colors, custom_color_map=None):
        """
        Get consistent color for an ROI across all plots.
        Uses hardcoded colors for specific ROIs, then custom_color_map, then persistent color map, then assigns new color.

        Parameters:
        -----------
        roi_name : str
            Name of the ROI
        default_colors : list
            List of default colors to cycle through
        custom_color_map : dict, optional
            Custom color mapping for this specific plot

        Returns:
        --------
        str : Color code for this ROI
        """
        # Hardcoded colors for specific ROIs (always used unless overridden by custom_color_map)
        HARDCODED_ROI_COLORS = {
            "single bomb ring": "#1E90FF",  # dodger blue
            "single bomb inside": "#00FFFF",  # neon cyan blue
            "double bomb 1": "#FF8C00",  # dark orange
            "double bomb 2": "#FFD700",  # gold
            "tripple bomb 1": "#800080",  # purple
            "tripple bomb 2": "#8A2BE2",  # violet
            "tripple bomb 3": "#FF00FF",  # bright magenta
            "all bombs": "#1E90FF",  # blue (group class)
            "dark bomb": "#000000",  # black
            "dark spots": "#000000",  # pure black
            "dark sediment": "#555555",  # medium-dark gray
            "sediment": "#8B4513",  # brown
            "brown leaf": "#8B4513",  # brown (same as sediment)
            "yellow leaf": "#FFD700",  # gold/yellow (same as double bomb 2)
        }

        # Priority 1: Custom color map for this specific plot (allows override)
        if custom_color_map and roi_name in custom_color_map:
            return custom_color_map[roi_name]

        # Priority 2: Hardcoded colors for known ROIs
        if roi_name in HARDCODED_ROI_COLORS:
            # Save to persistent map for consistency
            self.roi_color_map[roi_name] = HARDCODED_ROI_COLORS[roi_name]
            return HARDCODED_ROI_COLORS[roi_name]

        # Priority 3: Persistent color map (already assigned)
        if roi_name in self.roi_color_map:
            return self.roi_color_map[roi_name]

        # Priority 4: Assign new color and save it
        # Find next available color that's not already used
        used_colors = set(self.roi_color_map.values())
        for color in default_colors:
            if color not in used_colors:
                self.roi_color_map[roi_name] = color
                return color

        # If all colors used, cycle through again
        idx = len(self.roi_color_map) % len(default_colors)
        color = default_colors[idx]
        self.roi_color_map[roi_name] = color
        return color

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
        apply_alignment_shift=False,  # apply UHI alignment shift from config (NED only)
        # trajectory options
        show_trajectory=False,
        nav_csv_path=None,  # CSV must have lon/lat
        trajectory_color="red",
        trajectory_linewidth=2.0,
        trajectory_alpha=0.85,
        trajectory_decimate=1,  # >=1
        # perimeter line options (NEW - from plot_rgb)
        perimeter_line=None,  # Lines in (slit, track) coords
        line_colors=["red", "blue", "orange", "magenta"],
        line_width=2,
        line_style="-",
        line_labels=None,
        # ROI options (NEW - from plot_rgb)
        roi_collection=None,  # Dict of named ROIs or 'all' to use self.roi_collection
        roi_colors=["yellow", "cyan", "magenta", "orange", "lime", "red", "blue"],
        roi_color_map=None,  # Dict mapping ROI names to specific colors, e.g., {"Sediment": "brown", "dark1": "black"}
        roi_marker_size=100,
        roi_marker_shape="s",  # Marker shape: 's'=square, 'o'=circle, '^'=triangle, 'D'=diamond, 'v'=triangle_down, '<'=triangle_left, '>'=triangle_right, 'p'=pentagon, '*'=star, 'h'=hexagon, '+'=plus, 'x'=x
        roi_marker_edgewidth=2,  # Edge width for ROI markers (0 = no edge)
        roi_show_numbers=False,
        roi_legend_loc="best",  # Legend location: 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center', 'outside', or None to hide
        roi_legend_markersize=10,  # Size of color markers in legend (default=10)
        roi_legend_marker_border=True,  # Whether to show black border on legend markers
        return_fig=False,
        quiet=True,  # suppress non interactive prints and warnings
        **pcolor_kwargs,
    ):
        """
        RGB georeferenced composite in LATLON, NED, or ECEF with optional trajectory.
        Only interactive click readouts are printed when interactive=True.

        Parameters
        ----------
        apply_alignment_shift : bool, optional
            If True and coordinate_system=='NED', applies the alignment shift from
            config (UHI_ALIGNMENT_DX, UHI_ALIGNMENT_DY) to match MBES data.
            Default: False
        """
        import io, contextlib, warnings
        import numpy as np
        import matplotlib.pyplot as plt
        from pyproj import Transformer

        # ---------- helpers for robust timestamp handling ----------
        def _normalize_epoch_scalar(v):
            """Handle sec / ms / µs / ns -> seconds (float)."""
            try:
                v = float(v)
            except Exception:
                return np.nan
            # thresholds chosen so 2020+ in various units land correctly
            if v > 1e15:  # micro/nano
                return v / 1e9
            if v > 1e12:  # ms
                return v / 1e3
            return v  # seconds

        def _read_ts_from_file(h5_path):
            """Read timestamps array from a single H5 file with robust key search."""
            import h5py

            KEYS = [
                "processed/radiance/timestamps",  # preferred
                "processed/radiance/timestamp",
                "processed/timestamps",
                "processed/timestamp",
                "timestamps",
                "timestamp",
            ]
            with h5py.File(h5_path, "r") as f:
                # exact matches first
                for k in KEYS:
                    if k in f:
                        return np.asarray(f[k][()]).ravel()
                # case-insensitive search inside processed/radiance
                try:
                    grp = f["processed"]["radiance"]
                    for name, obj in grp.items():
                        if hasattr(obj, "shape") and "timestamp" in name.lower():
                            return np.asarray(obj[()]).ravel()
                except Exception:
                    pass

                # global fall-back: any dataset containing 'timestamp'
                def _walk(g):
                    for k, v in g.items():
                        if hasattr(v, "shape"):
                            if "timestamp" in k.lower():
                                return np.asarray(v[()]).ravel()
                        if isinstance(v, type(g)):
                            out = _walk(v)
                            if out is not None:
                                return out
                    return None

                out = _walk(f)
                return out if out is not None else np.array([], dtype=float)

        def _utc_range_for_slice(start_idx, end_idx):
            """Return (start_txt, end_txt) for plotted slice using self.timestamps or per-file H5 read."""
            # 1) try self.timestamps
            ts = getattr(self, "timestamps", None)
            if ts is not None and len(ts) >= end_idx:
                sl = np.asarray(ts[start_idx:end_idx]).ravel()
                if sl.size:
                    sln = np.array(
                        [_normalize_epoch_scalar(v) for v in sl], dtype=float
                    )
                    finite = np.isfinite(sln)
                    if finite.any():
                        first = sln[np.where(finite)[0][0]]
                        last = sln[np.where(finite)[0][-1]]
                        return unix_to_utc(first), unix_to_utc(last)

            # 2) robust per-file read just for the overlapped range
            parts = []
            name_to_gf = {gf.name: gf for gf in self.geofiles}
            for b in self.file_boundaries:
                f_start = b["start_track"]
                f_end = b["end_track"] + 1  # exclusive
                # overlap with [start_idx, end_idx)
                a = max(start_idx, f_start)
                z = min(end_idx, f_end)
                if a >= z:
                    continue
                gf = name_to_gf[b["file"]]
                full_ts = _read_ts_from_file(gf.path)
                if full_ts.size == 0:
                    parts.append(np.full(z - a, np.nan))
                    continue
                # cut relative window within this file
                rel0 = a - f_start
                rel1 = rel0 + (z - a)
                rel0 = max(0, rel0)
                rel1 = min(len(full_ts), rel1)
                seg = full_ts[rel0:rel1]
                if len(seg) < (z - a):
                    pad = np.full((z - a,), np.nan)
                    pad[: len(seg)] = seg
                    seg = pad
                parts.append(np.asarray(seg))
            if parts:
                sl = np.concatenate(parts).ravel()
                sln = np.array([_normalize_epoch_scalar(v) for v in sl], dtype=float)
                finite = np.isfinite(sln)
                if finite.any():
                    first = sln[np.where(finite)[0][0]]
                    last = sln[np.where(finite)[0][-1]]
                    return unix_to_utc(first), unix_to_utc(last)
            return "N/A", "N/A"

        def _silence():
            return contextlib.ExitStack()

        if coordinate_system is None:
            coordinate_system = "LATLON"

        if coordinate_system.upper() in ["NED", "ECEF"]:
            if origin is None:
                if (
                    "config" in globals()
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

        T, S = X_ecef.shape

        # Always define start_idx for perimeter/ROI coordinate conversion
        start_idx = track_start if track_start is not None else 0
        end_idx = track_end if track_end is not None else T

        if track_start is not None or track_end is not None:
            if start_idx < 0 or start_idx >= T:
                raise ValueError(f"track_start={start_idx} out of range [0, {T})")
            if end_idx <= start_idx or end_idx > T:
                raise ValueError(f"track_end={end_idx} must be in ({start_idx}, {T}]")
            X_ecef = X_ecef[start_idx:end_idx, :]
            Y_ecef = Y_ecef[start_idx:end_idx, :]
            Z_ecef = Z_ecef[start_idx:end_idx, :]
            R = R[start_idx:end_idx, :]
            G = G[start_idx:end_idx, :]
            B = B[start_idx:end_idx, :]
            T, S = X_ecef.shape

        RGB = np.dstack([R, G, B]).astype(np.float64)
        alpha = np.ones((T, S), dtype=np.float64)
        alpha[~(np.isfinite(R) & np.isfinite(G) & np.isfinite(B))] = 0.0

        if coordinate_system.upper() == "LATLON":
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
                origin = (lat0, lon0, 0.0)
            else:
                origin = (0.0, 0.0, 0.0)
        elif coordinate_system.upper() == "NED":
            lat0, lon0, h0 = origin
            N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)

            # Apply alignment shift if requested (to match MBES coordinates)
            if apply_alignment_shift:
                try:
                    if "config" in globals():
                        dx = getattr(config, "UHI_ALIGNMENT_DX", 0.0)
                        dy = getattr(config, "UHI_ALIGNMENT_DY", 0.0)
                    else:
                        # Try importing config
                        try:
                            from gref_pipeline import config as cfg

                            dx = getattr(cfg, "UHI_ALIGNMENT_DX", 0.0)
                            dy = getattr(cfg, "UHI_ALIGNMENT_DY", 0.0)
                        except ImportError:
                            dx, dy = 0.0, 0.0
                    E = E + dx
                    N = N + dy
                    if not quiet:
                        print(
                            f"Applied UHI alignment shift: dx={dx:.3f}m E, dy={dy:.3f}m N"
                        )
                except Exception:
                    if not quiet:
                        print("⚠️  Could not apply alignment shift")

            Xp, Yp = E, N
            xlabel, ylabel = f"East (m) from {lat0}°, {lon0}°", "North (m)"
        elif coordinate_system.upper() == "ECEF":
            lat0, lon0, h0 = origin
            if use_local_origin:
                x0, y0, z0 = _ecef_of_geodetic(lat0, lon0, h0)
                Xp, Yp = X_ecef - x0, Y_ecef - y0
                xlabel, ylabel = f"ΔECEF X (m) from {lat0}°, {lon0}°", "ΔECEF Y (m)"
            else:
                Xp, Yp = X_ecef, Y_ecef
                xlabel, ylabel = "ECEF X (m)", "ECEF Y (m)"
        else:
            raise ValueError("coordinate_system must be 'LATLON', 'NED', or 'ECEF'")

        alpha[~(np.isfinite(Xp) & np.isfinite(Yp))] = 0.0
        Xc = np.pad(Xp, ((0, 1), (0, 1)), mode="edge")
        Yc = np.pad(Yp, ((0, 1), (0, 1)), mode="edge")
        RGB[alpha == 0] = np.nan

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

        fig, ax = plt.subplots(figsize=figsize)
        ax.pcolormesh(Xc, Yc, RGB, shading="flat", **pcolor_kwargs)

        # Store coordinate arrays for interactive tools (e.g., plot_georef_interactive_lines)
        ax._gref_coord_arrays = (Xp, Yp, start_idx)

        if show_file_boundaries and len(self.file_boundaries) > 1:
            for b in self.file_boundaries[1:]:
                j = b["start_track"]
                if j < T:
                    ax.plot(
                        Xp[j, :],
                        Yp[j, :],
                        color="yellow",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.9,
                    )

        # ========== NEW: Perimeter lines (from plot_rgb) ==========
        if perimeter_line is not None and len(perimeter_line) > 0:
            # Handle single line or list of lines
            if isinstance(perimeter_line[0], (int, float)):
                lines_to_plot = [perimeter_line]
            else:
                lines_to_plot = perimeter_line

            for line_idx, line in enumerate(lines_to_plot):
                (slit1, track1), (slit2, track2) = line

                # Convert (slit, track) indices to georeferenced coordinates
                # Tracks are relative to the slice, so adjust by start_idx
                t1 = track1 - start_idx
                t2 = track2 - start_idx

                # Bounds check
                if 0 <= t1 < T and 0 <= t2 < T and 0 <= slit1 < S and 0 <= slit2 < S:
                    x1, y1 = Xp[t1, slit1], Yp[t1, slit1]
                    x2, y2 = Xp[t2, slit2], Yp[t2, slit2]

                    color = line_colors[line_idx % len(line_colors)]

                    if line_labels and line_idx < len(line_labels):
                        label = line_labels[line_idx]
                    elif len(lines_to_plot) > 1:
                        label = f"Line {line_idx + 1}"
                    else:
                        label = "Perimeter line"

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=line_width,
                        linestyle=line_style,
                        label=label,
                        zorder=10,
                    )

        # ========== NEW: ROI collection (from plot_rgb) ==========
        if roi_collection is not None:
            # Helper function to sort ROIs in display order
            def sort_rois_by_category(roi_dict):
                """Sort ROIs: single bombs → double → triple → dark features → sediment"""
                order_keywords = [
                    ["single bomb"],
                    ["double bomb"],
                    ["tripple bomb"],
                    ["dark"],
                    ["sediment"],
                ]

                def get_sort_key(roi_name):
                    roi_lower = roi_name.lower()
                    for idx, keywords in enumerate(order_keywords):
                        if any(kw in roi_lower for kw in keywords):
                            return (idx, roi_name)
                    return (len(order_keywords), roi_name)

                sorted_items = sorted(
                    roi_dict.items(), key=lambda x: get_sort_key(x[0])
                )
                return dict(sorted_items)

            # Determine which ROIs to plot
            if roi_collection == "all":
                if hasattr(self, "roi_collection") and self.roi_collection:
                    rois_to_plot = self.roi_collection
                else:
                    if not quiet:
                        print(
                            "⚠️  No ROI collection found. Use plot_interactive_rgb() first."
                        )
                    rois_to_plot = {}
            elif isinstance(roi_collection, list):
                # NEW: Support list of ROI names (like plot_spectrum)
                if hasattr(self, "roi_collection") and self.roi_collection:
                    rois_to_plot = {
                        name: self.roi_collection[name]
                        for name in roi_collection
                        if name in self.roi_collection
                    }
                else:
                    if not quiet:
                        print("⚠️  No ROI collection found.")
                    rois_to_plot = {}
            elif isinstance(roi_collection, dict):
                rois_to_plot = roi_collection
            else:
                if not quiet:
                    print(
                        "❌ roi_collection must be 'all', a list of ROI names, or a dictionary"
                    )
                rois_to_plot = {}

            # Sort ROIs for consistent legend order
            rois_to_plot = sort_rois_by_category(rois_to_plot)

            # Plot each ROI with different color
            for roi_idx, (roi_name, roi_pixels_list) in enumerate(rois_to_plot.items()):
                if roi_pixels_list:
                    # Convert (slit, track) to georeferenced coordinates
                    roi_x_coords = []
                    roi_y_coords = []

                    for slit, track in roi_pixels_list:
                        # Adjust track by start_idx
                        t = track - start_idx

                        # Bounds check
                        if 0 <= t < T and 0 <= slit < S:
                            roi_x_coords.append(Xp[t, slit])
                            roi_y_coords.append(Yp[t, slit])

                    if roi_x_coords:
                        # NEW: Get consistent color across all plots
                        color = self._get_roi_color(roi_name, roi_colors, roi_color_map)

                        # Plot ROI
                        ax.scatter(
                            roi_x_coords,
                            roi_y_coords,
                            c=color,
                            s=roi_marker_size,
                            marker=roi_marker_shape,
                            edgecolors="black" if roi_marker_edgewidth > 0 else "none",
                            linewidths=roi_marker_edgewidth,
                            alpha=0.8,
                            label=f"{roi_name} ({len(roi_x_coords)})",
                            zorder=11,
                        )

                        # Optional: Add numbers
                        if roi_show_numbers:
                            for i, (x, y) in enumerate(
                                zip(roi_x_coords, roi_y_coords), 1
                            ):
                                ax.text(
                                    x,
                                    y,
                                    str(i),
                                    ha="center",
                                    va="center",
                                    fontsize=8,
                                    fontweight="bold",
                                    color="white",
                                    bbox=dict(
                                        boxstyle="round,pad=0.3",
                                        facecolor=color,
                                        edgecolor="black",
                                        linewidth=1,
                                    ),
                                    zorder=12,
                                )

        if coordinate_system.upper() == "LATLON":
            if origin and origin[0] != 0:
                lat_center = origin[0]
                ax.set_aspect(1.0 / np.cos(np.radians(lat_center)), adjustable="box")
            else:
                ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("equal", adjustable="box")

        title = (
            f"RGB georeferenced composite ({self.name}) [{coordinate_system.upper()}]"
        )
        if coordinate_system.upper() == "LATLON":
            if origin:
                title += f"\ncenter ~ {origin[0]:.4f}°, {origin[1]:.4f}°"
        elif coordinate_system.upper() == "NED" or (
            coordinate_system.upper() == "ECEF" and use_local_origin
        ):
            title += f"\norigin @ {origin[0]:.6f}°, {origin[1]:.6f}°"

        # Add UTC time information using robust timestamp reading
        t0_txt, t1_txt = _utc_range_for_slice(start_idx, end_idx)
        title += f"\nTime (displayed): {t0_txt} → {t1_txt}"

        if interactive:
            title += "\n(Click to show coordinates)"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if show_trajectory:
            try:
                if nav_csv_path is None:
                    raise ValueError(
                        "nav_csv_path is required when show_trajectory=True"
                    )

                try:
                    import pandas as pd

                    nav = pd.read_csv(nav_csv_path)
                    cols = {c.lower(): c for c in nav.columns}

                    def pick(*names):
                        for n in names:
                            if n in cols:
                                return nav[cols[n]].to_numpy()
                        raise KeyError(f"Missing columns: {names}")

                    lon = pick("lon", "longitude")
                    lat = pick("lat", "latitude")
                    if any(k in cols for k in ("depth", "z", "height", "h")):
                        depth = None
                        for n in ("depth", "z", "height", "h"):
                            if n in cols:
                                depth = nav[cols[n]].to_numpy()
                                break
                    else:
                        depth = None
                except Exception:
                    import numpy as np

                    nav = np.genfromtxt(
                        nav_csv_path,
                        delimiter=",",
                        names=True,
                        dtype=None,
                        encoding="utf-8",
                    )
                    names = {n.lower(): n for n in nav.dtype.names}
                    lon_key = names.get("lon") or names.get("longitude")
                    lat_key = names.get("lat") or names.get("latitude")
                    if lon_key is None or lat_key is None:
                        raise KeyError(
                            "CSV must have 'lon/longitude' and 'lat/latitude' columns"
                        )
                    lon = nav[lon_key]
                    lat = nav[lat_key]
                    depth_key = (
                        names.get("depth")
                        or names.get("z")
                        or names.get("height")
                        or names.get("h")
                    )
                    depth = nav[depth_key] if depth_key else None

                disp = coordinate_system.upper()
                if disp == "LATLON":
                    x_traj, y_traj = lon, lat
                else:
                    tf_geo_to_ecef = Transformer.from_crs(
                        "EPSG:4979", "EPSG:4978", always_xy=True
                    )
                    h_in = np.zeros_like(lon) if depth is None else -np.array(depth)
                    x_e, y_e, z_e = tf_geo_to_ecef.transform(lon, lat, h_in)

                    if disp == "NED":
                        lat0, lon0, h0 = origin
                        N_t, E_t, D_t = _ecef_to_ned_arrays(
                            x_e, y_e, z_e, lat0, lon0, h0
                        )
                        x_traj, y_traj = E_t, N_t
                    elif disp == "ECEF":
                        if use_local_origin:
                            x0, y0, z0 = _ecef_of_geodetic(
                                origin[0], origin[1], origin[2]
                            )
                            x_traj, y_traj = x_e - x0, y_e - y0
                        else:
                            x_traj, y_traj = x_e, y_e
                    else:
                        raise ValueError(
                            f"Unknown coordinate system: {coordinate_system}"
                        )

                step = max(1, int(trajectory_decimate))
                if step > 1:
                    x_traj = x_traj[::step]
                    y_traj = y_traj[::step]

                ax.plot(
                    x_traj,
                    y_traj,
                    color=trajectory_color,
                    linewidth=trajectory_linewidth,
                    alpha=trajectory_alpha,
                    zorder=10,
                )
            except Exception:
                if not quiet:
                    import traceback

                    print("Trajectory overlay failed:\n" + traceback.format_exc())

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
                "coord_system": coordinate_system.upper(),
                "use_local_origin": use_local_origin,
            }

            def on_click(event):
                import numpy as np

                if event.inaxes is None:
                    return
                x_click, y_click = event.xdata, event.ydata
                dist = (click_data["Xp"] - x_click) ** 2 + (
                    click_data["Yp"] - y_click
                ) ** 2
                min_idx = np.nanargmin(dist)
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
                print(f"📍 track={ti}, slit={si}")
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
                if click_data["origin"] and click_data["coord_system"] != "NED":
                    lat0, lon0, h0 = click_data["origin"]
                    Np, Ep, Dp = _ecef_to_ned_arrays(
                        np.array([[x_e]]),
                        np.array([[y_e]]),
                        np.array([[z_e]]),
                        lat0,
                        lon0,
                        h0,
                    )
                    print(
                        f"   NED (from center): E={Ep[0,0]:.2f} m, N={Np[0,0]:.2f} m, D={Dp[0,0]:.2f} m"
                    )
                if np.isfinite(r_val):
                    print(f"   RGB: R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")
                else:
                    print("   RGB: No data")
                print("=" * 70)

            fig.canvas.mpl_connect("button_press_event", on_click)
            print("💡 Interactive mode: Click on the plot to display coordinates")

        # Add legend if perimeter lines or ROIs are shown
        if (
            perimeter_line is not None or roi_collection is not None
        ) and roi_legend_loc is not None:
            if roi_legend_loc == "outside":
                # Place legend outside the plot area on the right
                legend = ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=9,
                    framealpha=0.9,
                    markerscale=roi_legend_markersize / 6,
                )
            else:
                legend = ax.legend(
                    loc=roi_legend_loc,
                    fontsize=9,
                    framealpha=0.9,
                    markerscale=roi_legend_markersize / 6,
                )

            # Configure legend marker borders
            if not roi_legend_marker_border and legend:
                for handle in legend.legend_handles:
                    if hasattr(handle, "set_edgecolor"):
                        handle.set_edgecolor("none")
                        handle.set_linewidth(0)

        fig.tight_layout()
        plt.show()
        return (fig, ax) if return_fig else None

    def plot_georef_interactive_lines(
        self,
        red_wl=654.2,
        green_wl=560.0,
        blue_wl=440.3,
        normalize=True,
        figsize=(15, 10),
        coordinate_system=None,
        use_local_origin=True,
        origin=None,
        use_corrected=False,
        track_start=None,
        track_end=None,
        line_width=2,
        line_color="red",
    ):
        """
        Interactive tool for defining lines by clicking on georeferenced plot.

        Click 2 points to define each line (start point → end point).
        Close the window when done to return the list of lines.

        Parameters
        ----------
        red_wl, green_wl, blue_wl : float
            Wavelengths for RGB composite
        normalize : bool
            Normalize RGB channels
        figsize : tuple
            Figure size
        coordinate_system : str
            'LATLON', 'NED', or 'ECEF'
        use_local_origin : bool
            For NED, use local origin
        origin : tuple
            Custom origin (lat, lon, alt_m)
        use_corrected : bool
            Use illumination-corrected data
        track_start, track_end : int
            Track range to display
        line_width : float
            Width of drawn lines
        line_color : str
            Color of completed lines

        Returns
        -------
        list
            List of lines in format [((slit1, track1), (slit2, track2)), ...]
            Compatible with plot_georef(perimeter_line=...) parameter
        """
        import matplotlib.pyplot as plt

        # First generate the georef plot to get coordinate arrays
        fig, ax = self.plot_georef(
            red_wl=red_wl,
            green_wl=green_wl,
            blue_wl=blue_wl,
            normalize=normalize,
            figsize=figsize,
            coordinate_system=coordinate_system,
            use_local_origin=use_local_origin,
            origin=origin,
            use_corrected=use_corrected,
            track_start=track_start,
            track_end=track_end,
            return_fig=True,
            interactive=False,
        )

        # Extract coordinate arrays that plot_georef stored for us
        if hasattr(ax, "_gref_coord_arrays"):
            Xp, Yp, start_idx = ax._gref_coord_arrays
            print(
                f"✓ Coordinate arrays extracted: Xp shape={Xp.shape}, Yp shape={Yp.shape}"
            )
            print(f"✓ Track offset: start_idx={start_idx}")
            print(
                f"✓ Coordinate range: X=[{np.nanmin(Xp):.2f}, {np.nanmax(Xp):.2f}], Y=[{np.nanmin(Yp):.2f}, {np.nanmax(Yp):.2f}]"
            )
        else:
            raise RuntimeError(
                "plot_georef did not store coordinate arrays. "
                "This may be due to an older version of the code."
            )

        # State for click tracking
        click_state = {
            "points": [],  # List of (x, y, slit_idx, track_idx)
            "lines": [],  # List of completed lines
            "circles": [],  # Visual markers for clicked points
            "line_objs": [],  # Visual line objects
        }

        def onclick(event):
            print(
                f"🖱️ Click detected: button={event.button}, inaxes={event.inaxes is not None}"
            )
            if event.inaxes != ax:
                print("   → Click outside axis, ignoring")
                return

            # Get click coordinates
            x_click, y_click = event.xdata, event.ydata
            print(f"   → Click coords: x={x_click:.2f}, y={y_click:.2f}")

            # Find nearest point in the georef grid
            distances = np.sqrt((Xp - x_click) ** 2 + (Yp - y_click) ** 2)
            min_idx = np.nanargmin(distances)
            track_offset, slit_idx = np.unravel_index(min_idx, Xp.shape)

            # Convert to absolute track index
            track_idx = start_idx + track_offset

            # Get actual coordinates at this point
            x_actual = Xp[track_offset, slit_idx]
            y_actual = Yp[track_offset, slit_idx]

            # Add point
            click_state["points"].append((x_actual, y_actual, slit_idx, track_idx))

            # Draw circle marker
            circle = plt.Circle(
                (x_actual, y_actual),
                radius=5,
                color="cyan",
                fill=True,
                alpha=0.8,
                zorder=10,
            )
            ax.add_patch(circle)
            click_state["circles"].append(circle)

            print(
                f"Point {len(click_state['points'])}: slit={slit_idx}, track={track_idx}"
            )

            # If we have 2 points, create a line
            if len(click_state["points"]) == 2:
                p1 = click_state["points"][0]
                p2 = click_state["points"][1]

                # Draw line between points
                (line,) = ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color=line_color,
                    linewidth=line_width,
                    zorder=9,
                )
                click_state["line_objs"].append(line)

                # Store line in (slit, track) format
                line_coords = ((p1[2], p1[3]), (p2[2], p2[3]))
                click_state["lines"].append(line_coords)

                print(f"✓ Line created: {line_coords}")
                print(f"  Total lines: {len(click_state['lines'])}")

                # Reset for next line
                click_state["points"] = []

            fig.canvas.draw()

        # Connect event handler
        cid = fig.canvas.mpl_connect("button_press_event", onclick)

        # Display instructions
        ax.set_title(
            ax.get_title()
            + "\n\n🖱️ Click 2 points for each line | Close window when done",
            fontsize=10,
        )

        plt.show()

        # Disconnect handler
        fig.canvas.mpl_disconnect(cid)

        # Print summary
        print("\n" + "=" * 60)
        print(f"Interactive line definition complete!")
        if len(click_state["lines"]) == 0:
            print("⚠️  No lines were created!")
            print("   To create lines, click 2 points on the plot (start → end)")
            print("   You can create multiple lines before closing the window")
        else:
            print(f"Created {len(click_state['lines'])} lines:")
            for i, line in enumerate(click_state["lines"], 1):
                print(f"  Line {i}: {line}")
        print("=" * 60)

        return click_state["lines"]

    def plot_georef_with_trajectory(
        self,
        nav_csv_path=None,
        red_wl=654.2,
        green_wl=560.0,
        blue_wl=440.3,
        normalize=True,
        figsize=(11, 9),
        coordinate_system=None,
        use_local_origin=True,
        origin=None,
        show_file_boundaries=True,
        show_trajectory=True,
        trajectory_color="red",
        trajectory_linewidth=2.0,
        trajectory_alpha=0.8,
        trajectory_label="Navigation trajectory",
        interactive=True,
        # --- MBES args ---
        mbes_geotiff_path=None,  # e.g. config.MBES_GEOTIFF
        mbes_epsg=None,  # e.g. config.EPSG_MBES (int). If None, tries config or 4326 for LATLON
        mbes_cmap="Greys",
        mbes_alpha=0.6,
        mbes_step=4,  # decimation for speed (>=1)
        **pcolor_kwargs,
    ):
        """
        Render RGB composite (UHI) + optional MBES underlay + navigation trajectory.
        MBES is reprojected to the display CRS:
        - LATLON: MBES -> EPSG:4326, plotted as lon/lat
        - NED:    MBES -> lon/lat -> ECEF -> NED(E, N) about 'origin'
        - ECEF:   MBES -> lon/lat -> ECEF (Δ if use_local_origin)
        """
        import sys
        from pathlib import Path

        # Add parent directory to path
        sys.path.append(str(Path(__file__).parent.parent.parent))

        from gref_pipeline import config
        from utils.gref_pipeline import utils
        from pyproj import Transformer

        # -------- Display CRS defaults --------
        if coordinate_system is None:
            coordinate_system = "LATLON"

        # -------- Draw UHI base figure first (no show yet) --------
        original_interactive = plt.isinteractive()
        plt.ioff()

        self.plot_georef(
            red_wl=red_wl,
            green_wl=green_wl,
            blue_wl=blue_wl,
            normalize=normalize,
            figsize=figsize,
            coordinate_system=coordinate_system,
            use_local_origin=use_local_origin,
            origin=origin,
            show_file_boundaries=show_file_boundaries,
            alpha_for_nodata=0.0,
            interactive=False,
            **pcolor_kwargs,
        )

        fig = plt.gcf()
        ax = plt.gca()

        # Capture/ensure origin for NED/ECEF math
        if origin is None and coordinate_system.upper() in ["NED", "ECEF"]:
            if hasattr(config, "LAT0") and hasattr(config, "LON0"):
                origin = (
                    float(config.LAT0),
                    float(config.LON0),
                    float(getattr(config, "H0", 0.0)),
                )
            else:
                origin = (60.8011575, 10.7122345, 0.0)

        # -------- MBES underlay (optional) --------
        if mbes_geotiff_path is not None:
            try:
                # choose EPSG to read MBES
                if mbes_epsg is None:
                    if hasattr(config, "EPSG_MBES"):
                        mbes_epsg = int(config.EPSG_MBES)
                    else:
                        mbes_epsg = (
                            4326 if coordinate_system.upper() == "LATLON" else None
                        )
                if mbes_epsg is None:
                    raise ValueError("Please provide mbes_epsg or set config.EPSG_MBES")

                # NOTE: use self. to call the helpers inside the class
                arr, tf, used_epsg = self._read_mbes_geotiff_as_epsg(
                    mbes_geotiff_path, mbes_epsg
                )
                h, w = arr.shape
                Xp_raw, Yp_raw = self._xy_from_affine_grid(
                    tf, h, w
                )  # in EPSG=used_epsg

                # Decimate for speed
                step = max(1, int(mbes_step))
                arr_d = arr[::step, ::step]
                Xr = Xp_raw[::step, ::step]
                Yr = Yp_raw[::step, ::step]

                # Convert MBES grid to display coordinates
                disp = coordinate_system.upper()
                if disp == "LATLON":
                    tf_to_ll = Transformer.from_crs(
                        f"EPSG:{used_epsg}", "EPSG:4326", always_xy=True
                    )
                    lon_m, lat_m = tf_to_ll.transform(Xr, Yr)
                    Xm, Ym = lon_m, lat_m

                elif disp == "NED":
                    tf_to_ll = Transformer.from_crs(
                        f"EPSG:{used_epsg}", "EPSG:4326", always_xy=True
                    )
                    lon_m, lat_m = tf_to_ll.transform(Xr, Yr)
                    tf_geo_to_ecef = Transformer.from_crs(
                        "EPSG:4979", "EPSG:4978", always_xy=True
                    )
                    x_ecef, y_ecef, z_ecef = tf_geo_to_ecef.transform(
                        lon_m, lat_m, np.zeros_like(lon_m)
                    )
                    N_m, E_m, D_m = _ecef_to_ned_arrays(
                        x_ecef, y_ecef, z_ecef, origin[0], origin[1], origin[2]
                    )
                    Xm, Ym = E_m, N_m

                elif disp == "ECEF":
                    tf_to_ll = Transformer.from_crs(
                        f"EPSG:{used_epsg}", "EPSG:4326", always_xy=True
                    )
                    lon_m, lat_m = tf_to_ll.transform(Xr, Yr)
                    tf_geo_to_ecef = Transformer.from_crs(
                        "EPSG:4979", "EPSG:4978", always_xy=True
                    )
                    x_ecef, y_ecef, z_ecef = tf_geo_to_ecef.transform(
                        lon_m, lat_m, np.zeros_like(lon_m)
                    )
                    if use_local_origin:
                        x0, y0, z0 = _ecef_of_geodetic(origin[0], origin[1], origin[2])
                        Xm, Ym = x_ecef - x0, y_ecef - y0
                    else:
                        Xm, Ym = x_ecef, y_ecef
                else:
                    raise ValueError(f"Unknown coordinate system: {coordinate_system}")

                # Robust color limits
                finite = np.isfinite(arr_d)
                if finite.any():
                    vmin, vmax = np.nanpercentile(arr_d[finite], [2, 98])
                else:
                    vmin, vmax = 0.0, 1.0

                # pcolormesh wants corners
                Xc = np.pad(Xm, ((0, 1), (0, 1)), mode="edge")
                Yc = np.pad(Ym, ((0, 1), (0, 1)), mode="edge")

                # Draw MBES below UHI
                im_mbes = ax.pcolormesh(
                    Xc,
                    Yc,
                    arr_d,
                    shading="flat",
                    cmap=mbes_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=mbes_alpha,
                    zorder=-5,
                )
                cbar = plt.colorbar(im_mbes, ax=ax, fraction=0.035, pad=0.02)
                cbar.set_label("MBES depth/elevation")

                print(
                    f"✓ MBES underlay added from '{mbes_geotiff_path}' (EPSG:{used_epsg}), step={step}"
                )

            except Exception as e:
                print(f"⚠️  MBES overlay failed: {e}")

        # -------- Trajectory overlay --------
        if show_trajectory:
            try:
                if nav_csv_path is None:
                    nav_csv_path = config.NAV_CSV
                nav_df = utils.load_csv_navigation(nav_csv_path, config.CSV_COLUMNS)

                lon = nav_df["longitude"].to_numpy()
                lat = nav_df["latitude"].to_numpy()
                depth = nav_df["depth"].to_numpy()

                x_ecef, y_ecef, z_ecef = utils.geographic_to_ecef(
                    lon,
                    lat,
                    -depth,
                    epsg_geo=config.EPSG_GEOGRAPHIC,
                    epsg_ecef=config.EPSG_ECEF,
                )

                if coordinate_system.upper() == "LATLON":
                    x_traj, y_traj = lon, lat
                elif coordinate_system.upper() == "NED":
                    lat0, lon0, h0 = origin
                    N, E, D = _ecef_to_ned_arrays(
                        x_ecef, y_ecef, z_ecef, lat0, lon0, h0
                    )
                    x_traj, y_traj = E, N
                elif coordinate_system.upper() == "ECEF":
                    if use_local_origin:
                        x0, y0, z0 = _ecef_of_geodetic(origin[0], origin[1], origin[2])
                        x_traj, y_traj = x_ecef - x0, y_ecef - y0
                    else:
                        x_traj, y_traj = x_ecef, y_ecef
                else:
                    raise ValueError(f"Unknown coordinate system: {coordinate_system}")

                ax.plot(
                    x_traj,
                    y_traj,
                    color=trajectory_color,
                    linewidth=trajectory_linewidth,
                    alpha=trajectory_alpha,
                    label=trajectory_label,
                    zorder=10,
                )
                # pin legend to avoid slow "best"
                plt.legend(loc="upper right", fontsize=8)
                print(f"✓ Added trajectory with {len(lon)} points")
            except Exception as e:
                print(f"⚠️  Could not add trajectory: {e}")

        # -------- Interactivity --------
        if interactive:
            # (reuse your click handler from plot_georef if you want it here)
            pass

        if original_interactive:
            plt.ion()
        plt.tight_layout()
        plt.show()
        return fig, ax

    # Backward compatibility alias
    def plot_georef_rgb(self, *args, **kwargs):
        """Deprecated: Use plot_georef() instead."""
        return self.plot_georef(*args, **kwargs)

    # def plot_georef_with_trajectory(
    #     self,
    #     nav_csv_path=None,
    #     red_wl=654.2,
    #     green_wl=560.0,
    #     blue_wl=440.3,
    #     normalize=True,
    #     figsize=(11, 9),
    #     coordinate_system=None,
    #     use_local_origin=True,
    #     origin=None,
    #     show_file_boundaries=True,
    #     show_trajectory=True,
    #     trajectory_color="red",
    #     trajectory_linewidth=2.0,
    #     trajectory_alpha=0.8,
    #     trajectory_label="Navigation trajectory",
    #     interactive=True,
    #     **pcolor_kwargs,
    # ):
    #     """
    #     Render an RGB composite with navigation trajectory overlay.

    #     This function extends the standard plot_georef() by adding the full
    #     navigation trajectory from the merged CSV file.

    #     Parameters:
    #     -----------
    #     nav_csv_path : str, optional
    #         Path to navigation CSV file. If None, uses config.NAV_CSV
    #     red_wl, green_wl, blue_wl : float
    #         Wavelengths for RGB composite (default: 654.2, 560.0, 440.3 nm)
    #     normalize : bool
    #         Whether to normalize RGB values (default: True)
    #     figsize : tuple
    #         Figure size (default: (11, 9))
    #     coordinate_system : str or None
    #         "LATLON" (default), "NED", or "ECEF"
    #     use_local_origin : bool
    #         For ECEF mode, use local origin offset (default: True)
    #     origin : tuple or None
    #         (lat, lon, h) for NED/ECEF modes. Auto-computed for LATLON
    #     show_file_boundaries : bool
    #         Show H5 file boundaries (default: True)
    #     show_trajectory : bool
    #         Show navigation trajectory overlay (default: True)
    #     trajectory_color : str
    #         Color for trajectory line (default: "red")
    #     trajectory_linewidth : float
    #         Width of trajectory line (default: 2.0)
    #     trajectory_alpha : float
    #         Alpha transparency for trajectory (default: 0.8)
    #     trajectory_label : str
    #         Label for trajectory in legend (default: "Navigation trajectory")
    #     interactive : bool
    #         Enable click-to-show coordinates (default: True)
    #     **pcolor_kwargs : dict
    #         Additional arguments passed to pcolormesh

    #     Returns:
    #     --------
    #     fig, ax : matplotlib Figure and Axes objects
    #     """
    #     import config
    #     from utils import utils

    #     # Default to LATLON if not specified
    #     if coordinate_system is None:
    #         coordinate_system = "LATLON"

    #     # Get navigation CSV path
    #     if nav_csv_path is None:
    #         nav_csv_path = config.NAV_CSV

    #     # First, call the standard plot_georef but capture the figure
    #     # We need to temporarily disable interactive mode to modify the plot
    #     original_interactive = plt.isinteractive()
    #     plt.ioff()  # Turn off interactive mode temporarily

    #     # Call the original plot_georef method
    #     self.plot_georef(
    #         red_wl=red_wl,
    #         green_wl=green_wl,
    #         blue_wl=blue_wl,
    #         normalize=normalize,
    #         figsize=figsize,
    #         coordinate_system=coordinate_system,
    #         use_local_origin=use_local_origin,
    #         origin=origin,
    #         show_file_boundaries=show_file_boundaries,
    #         alpha_for_nodata=0.0,
    #         interactive=False,  # We'll add interactivity after adding trajectory
    #         **pcolor_kwargs,
    #     )

    #     # Get current figure and axes
    #     fig = plt.gcf()
    #     ax = plt.gca()

    #     # Add trajectory if requested
    #     if show_trajectory:
    #         try:
    #             # Load navigation data
    #             nav_df = utils.load_csv_navigation(nav_csv_path, config.CSV_COLUMNS)

    #             # Extract coordinates
    #             lon = nav_df["longitude"].to_numpy()
    #             lat = nav_df["latitude"].to_numpy()
    #             depth = nav_df["depth"].to_numpy()

    #             # Convert to ECEF first
    #             x_ecef, y_ecef, z_ecef = utils.geographic_to_ecef(
    #                 lon,
    #                 lat,
    #                 -depth,  # depth is negative for underwater
    #                 epsg_geo=config.EPSG_GEOGRAPHIC,
    #                 epsg_ecef=config.EPSG_ECEF,
    #             )

    #             # Transform to the same coordinate system as the plot
    #             from pyproj import Transformer

    #             if coordinate_system.upper() == "LATLON":
    #                 # Already have lat/lon
    #                 x_traj, y_traj = lon, lat

    #             elif coordinate_system.upper() == "NED":
    #                 # Convert ECEF to NED
    #                 if origin is None:
    #                     # Use config origin or compute from data
    #                     if hasattr(config, "LAT0") and hasattr(config, "LON0"):
    #                         origin = (
    #                             float(config.LAT0),
    #                             float(config.LON0),
    #                             float(getattr(config, "H0", 0.0)),
    #                         )
    #                     else:
    #                         origin = (60.8011575, 10.7122345, 0.0)

    #                 lat0, lon0, h0 = origin
    #                 N, E, D = _ecef_to_ned_arrays(
    #                     x_ecef, y_ecef, z_ecef, lat0, lon0, h0
    #                 )
    #                 x_traj, y_traj = E, N

    #             elif coordinate_system.upper() == "ECEF":
    #                 # ECEF or ECEF-local
    #                 if origin is None:
    #                     if hasattr(config, "LAT0") and hasattr(config, "LON0"):
    #                         origin = (
    #                             float(config.LAT0),
    #                             float(config.LON0),
    #                             float(getattr(config, "H0", 0.0)),
    #                         )
    #                     else:
    #                         origin = (60.8011575, 10.7122345, 0.0)

    #                 if use_local_origin:
    #                     lat0, lon0, h0 = origin
    #                     x0, y0, z0 = _ecef_of_geodetic(lat0, lon0, h0)
    #                     x_traj, y_traj = x_ecef - x0, y_ecef - y0
    #                 else:
    #                     x_traj, y_traj = x_ecef, y_ecef
    #             else:
    #                 raise ValueError(f"Unknown coordinate system: {coordinate_system}")

    #             # Plot trajectory
    #             ax.plot(
    #                 x_traj,
    #                 y_traj,
    #                 color=trajectory_color,
    #                 linewidth=trajectory_linewidth,
    #                 alpha=trajectory_alpha,
    #                 label=trajectory_label,
    #                 zorder=10,  # Draw on top
    #             )

    #             # Update legend
    #             ax.legend()

    #             print(f"✅ Added trajectory with {len(lon)} navigation points")

    #         except Exception as e:
    #             print(f"⚠️  Could not add trajectory: {e}")

    #     # Add interactive click handler if requested
    #     if interactive:
    #         from pyproj import Transformer

    #         # Get the coordinate data from cube
    #         X_ecef, Y_ecef, Z_ecef = self.X_ecef, self.Y_ecef, self.Z_ecef
    #         R, G, B = self.R, self.G, self.B

    #         # Recompute display coordinates (same logic as in plot_georef)
    #         if coordinate_system.upper() == "LATLON":
    #             tf_ecef_to_geo = Transformer.from_crs(
    #                 "EPSG:4978", "EPSG:4979", always_xy=True
    #             )
    #             lon_grid, lat_grid, height = tf_ecef_to_geo.transform(
    #                 X_ecef, Y_ecef, Z_ecef
    #             )
    #             Xp, Yp = lon_grid, lat_grid
    #         elif coordinate_system.upper() == "NED":
    #             lat0, lon0, h0 = origin
    #             N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)
    #             Xp, Yp = E, N
    #         elif coordinate_system.upper() == "ECEF":
    #             if use_local_origin:
    #                 lat0, lon0, h0 = origin
    #                 x0, y0, z0 = _ecef_of_geodetic(lat0, lon0, h0)
    #                 Xp, Yp = X_ecef - x0, Y_ecef - y0
    #             else:
    #                 Xp, Yp = X_ecef, Y_ecef

    #         # Store data for click handler
    #         click_data = {
    #             "Xp": Xp,
    #             "Yp": Yp,
    #             "X_ecef": X_ecef,
    #             "Y_ecef": Y_ecef,
    #             "Z_ecef": Z_ecef,
    #             "R": R,
    #             "G": G,
    #             "B": B,
    #             "origin": origin,
    #             "coord_system": coordinate_system.upper(),
    #             "use_local_origin": use_local_origin,
    #         }

    #         # Create transformer for ECEF to lat/lon
    #         tf_ecef_to_geo = Transformer.from_crs(
    #             "EPSG:4978", "EPSG:4979", always_xy=True
    #         )

    #         def on_click(event):
    #             if event.inaxes is None:
    #                 return

    #             # Get click position
    #             x_click, y_click = event.xdata, event.ydata

    #             # Find nearest grid point
    #             dist = (click_data["Xp"] - x_click) ** 2 + (
    #                 click_data["Yp"] - y_click
    #             ) ** 2
    #             min_idx = np.nanargmin(dist)
    #             track_idx, slit_idx = np.unravel_index(min_idx, click_data["Xp"].shape)

    #             # Get coordinates at this point
    #             x_plot = click_data["Xp"][track_idx, slit_idx]
    #             y_plot = click_data["Yp"][track_idx, slit_idx]
    #             x_ecef = click_data["X_ecef"][track_idx, slit_idx]
    #             y_ecef = click_data["Y_ecef"][track_idx, slit_idx]
    #             z_ecef = click_data["Z_ecef"][track_idx, slit_idx]

    #             # Get RGB values
    #             r_val = click_data["R"][track_idx, slit_idx]
    #             g_val = click_data["G"][track_idx, slit_idx]
    #             b_val = click_data["B"][track_idx, slit_idx]

    #             # Check if valid point
    #             if not (
    #                 np.isfinite(x_ecef) and np.isfinite(y_ecef) and np.isfinite(z_ecef)
    #             ):
    #                 print(f"⚠️  Invalid point at track={track_idx}, slit={slit_idx}")
    #                 return

    #             # Convert ECEF to lat/lon/height
    #             lon_deg, lat_deg, height = tf_ecef_to_geo.transform(
    #                 x_ecef, y_ecef, z_ecef
    #             )

    #             # Print info
    #             print("\n" + "=" * 70)
    #             print(f"📍 Clicked at track={track_idx}, slit={slit_idx}")

    #             # Show coordinates based on current display mode
    #             if click_data["coord_system"] == "LATLON":
    #                 print(f"   Lat/Lon: {y_plot:.6f}°, {x_plot:.6f}°")
    #             elif click_data["coord_system"] == "NED":
    #                 print(f"   NED: E={x_plot:.2f}m, N={y_plot:.2f}m")
    #             else:
    #                 print(f"   Plot coords: ({x_plot:.2f}, {y_plot:.2f})")

    #             # Always show ECEF and WGS84 for reference
    #             print(f"   ECEF: X={x_ecef:.2f}m, Y={y_ecef:.2f}m, Z={z_ecef:.2f}m")
    #             print(
    #                 f"   WGS84: Lat={lat_deg:.6f}°, Lon={lon_deg:.6f}°, h={height:.2f}m"
    #             )

    #             if np.isfinite(r_val):
    #                 print(f"   RGB: R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")
    #             else:
    #                 print(f"   RGB: No data")
    #             print("=" * 70)

    #         # Connect click event
    #         fig.canvas.mpl_connect("button_press_event", on_click)
    #         print("💡 Interactive mode: Click on the plot to display coordinates")

    #     # Restore interactive mode and show
    #     if original_interactive:
    #         plt.ion()

    #     plt.tight_layout()
    #     plt.show()

    #     return fig, ax

    def apply_illumination_correction(
        self, window_size=1000, strength=1.0, force_recompute=False
    ):
        """
        Per-slit illumination normalization with automatic persistence.

        Parameters:
        -----------
        window_size : int or None
            - None (or <=1 or >=T_total): global correction (single median per slit-band).
            - int: rolling median across tracks of length window_size.
        strength : float
            Correction strength in [0,1]: 0=no change, 1=full correction.
        force_recompute : bool
            If True, recompute even if saved correction exists.

        The corrected data is automatically saved to the HDF5 files and loaded
        on subsequent runs if the parameters match.
        """
        import h5py
        from scipy.ndimage import median_filter
        import numpy as np

        # --- Check if already computed and saved ---
        if not force_recompute and self.has_illumination_correction(
            window_size, strength
        ):
            print(
                f"✅ Illumination correction already applied with window={window_size}, strength={strength}"
            )
            print(f"   Loading from disk...")
            return self.load_illumination_correction(window_size, strength)

        # --- sizes ---
        T_total = 0
        S = B = None
        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                dset_name = gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                if dset_name not in f:
                    dset_name = gf.DSET_RGB_MAIN
                t, s, b = f[dset_name].shape
                T_total += t
                if S is None:
                    S, B = s, b

        # decide mode
        use_global = (
            (window_size is None) or (window_size <= 1) or (window_size >= T_total)
        )
        mode_txt = "global" if use_global else f"rolling (window={int(window_size)})"
        print(f"🔄 Computing illumination correction: {mode_txt}, strength={strength}")

        self.data_corrected = np.zeros((T_total, S, B), dtype=np.float32)

        try:
            from tqdm import tqdm

            pbar = tqdm(total=S, desc="   Processing slits", unit="slit")
            use_tqdm = True
        except Exception:
            use_tqdm = False
            pbar = None

        # --- process one slit at a time ---
        for s in range(S):
            slit_data_list = []
            for gf in self.geofiles:
                with h5py.File(gf.path, "r") as f:
                    dset_name = (
                        gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                    )
                    if dset_name not in f:
                        dset_name = gf.DSET_RGB_MAIN
                    slit_slice = f[dset_name][:, s, :].astype(np.float32)  # (T_file, B)
                    slit_data_list.append(slit_slice)
            slit_data = np.concatenate(slit_data_list, axis=0)  # (T_total, B)
            del slit_data_list

            for b in range(B):
                ts = slit_data[:, b]  # (T_total,)

                if use_global:
                    ref = np.nanmedian(ts)
                    if not np.isfinite(ref) or ref == 0:
                        ref = 1.0
                    corrected = ts / ref
                else:
                    ref_vec = median_filter(ts, size=int(window_size), mode="nearest")
                    ref_vec[ref_vec == 0] = 1.0
                    corrected = ts / ref_vec

                self.data_corrected[:, s, b] = (
                    1 - strength
                ) * ts + strength * corrected

            del slit_data
            if use_tqdm:
                pbar.update(1)

        if use_tqdm:
            pbar.close()

        print(f"✅ data_corrected ready ({mode_txt})")

        # --- Save to disk for future use ---
        try:
            self.save_illumination_correction(
                window_size=window_size, strength=strength
            )
        except Exception as e:
            print(f"⚠️  Could not save illumination correction: {e}")

        return self.data_corrected

    def apply_illumination_correction_v2(
        self, window_size=1000, strength=1.0, force_recompute=False
    ):
        """
        V2: Fixed version using pandas rolling median instead of scipy median_filter.

        Per-slit illumination normalization with automatic persistence.

        Parameters:
        -----------
        window_size : int or None
            - None (or <=1 or >=T_total): global correction (single median per slit-band).
            - int: rolling median across tracks of length window_size.
        strength : float
            Correction strength in [0,1]: 0=no change, 1=full correction.
        force_recompute : bool
            If True, recompute even if saved correction exists.
        """
        import h5py
        import numpy as np
        import pandas as pd

        print("🔄 Using V2 algorithm (pandas rolling median)")

        # --- Check if already computed and saved ---
        if not force_recompute and self.has_illumination_correction(
            window_size, strength
        ):
            print(
                f"✅ Illumination correction already applied with window={window_size}, strength={strength}"
            )
            print(f"   Loading from disk...")
            return self.load_illumination_correction(window_size, strength)

        # --- sizes ---
        T_total = 0
        S = B = None
        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                dset_name = gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                if dset_name not in f:
                    dset_name = gf.DSET_RGB_MAIN
                t, s, b = f[dset_name].shape
                T_total += t
                if S is None:
                    S, B = s, b

        # decide mode
        use_global = (
            (window_size is None) or (window_size <= 1) or (window_size >= T_total)
        )
        mode_txt = "global" if use_global else f"rolling (window={int(window_size)})"
        print(f"🔄 Computing illumination correction: {mode_txt}, strength={strength}")

        self.data_corrected = np.zeros((T_total, S, B), dtype=np.float32)

        try:
            from tqdm import tqdm

            pbar = tqdm(total=S, desc="   Processing slits", unit="slit")
            use_tqdm = True
        except Exception:
            use_tqdm = False
            pbar = None

        # --- process one slit at a time ---
        for s in range(S):
            slit_data_list = []
            for gf in self.geofiles:
                with h5py.File(gf.path, "r") as f:
                    dset_name = (
                        gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                    )
                    if dset_name not in f:
                        dset_name = gf.DSET_RGB_MAIN
                    slit_slice = f[dset_name][:, s, :].astype(np.float32)  # (T_file, B)
                    slit_data_list.append(slit_slice)
            slit_data = np.concatenate(slit_data_list, axis=0)  # (T_total, B)
            del slit_data_list

            for b in range(B):
                ts = slit_data[:, b]  # (T_total,)

                if use_global:
                    ref = np.nanmedian(ts)
                    if not np.isfinite(ref) or ref == 0:
                        ref = 1.0
                    corrected = ts / ref
                else:
                    # V2: Use pandas rolling median for proper 1D processing
                    ref_vec = (
                        pd.Series(ts)
                        .rolling(window=int(window_size), center=True, min_periods=1)
                        .median()
                        .values
                    )
                    ref_vec[ref_vec == 0] = 1.0
                    ref_vec[~np.isfinite(ref_vec)] = 1.0
                    corrected = ts / ref_vec

                self.data_corrected[:, s, b] = (
                    1 - strength
                ) * ts + strength * corrected

            del slit_data
            if use_tqdm:
                pbar.update(1)

        if use_tqdm:
            pbar.close()

        print(f"✅ data_corrected ready ({mode_txt}) using V2 algorithm")

        # --- Save to disk ---
        print("💾 Saving illumination correction to HDF5 files...")
        self.save_illumination_correction(window_size, strength)

        return self.data_corrected

    def _get_correction_dataset_name(self, window_size=1000, strength=1.0):
        """
        Generate dataset name for illumination correction with specific parameters.
        Allows multiple cached versions with different parameters.
        """
        # Convert window_size to string (None -> 'global')
        w_str = "global" if window_size is None else str(int(window_size))
        s_str = f"{strength:.2f}".replace(".", "p")  # 1.0 -> "1p00"
        return f"processed/radiance/dataCube_illum_corrected_w{w_str}_s{s_str}"

    def has_illumination_correction(self, window_size=1000, strength=1.0):
        """
        Check if illumination correction with these parameters has already been applied.
        Returns True if all files have the corrected dataset with matching metadata.
        """
        import h5py

        dset_name = self._get_correction_dataset_name(window_size, strength)

        for gf in self.geofiles:
            try:
                with h5py.File(gf.path, "r") as f:
                    # Check if corrected dataset exists
                    if dset_name not in f:
                        return False

                    # Verify metadata matches
                    dset = f[dset_name]
                    if "window_size" not in dset.attrs or "strength" not in dset.attrs:
                        return False

                    saved_window = dset.attrs["window_size"]
                    saved_strength = dset.attrs["strength"]

                    # Compare parameters
                    if (
                        saved_window != window_size
                        or abs(saved_strength - strength) > 1e-6
                    ):
                        return False
            except Exception as e:
                print(f"⚠️  Error checking {gf.name}: {e}")
                return False

        return True

    def load_illumination_correction(self, window_size=1000, strength=1.0):
        """
        Load previously saved illumination-corrected data from HDF5 files.
        Populates self.data_corrected.
        """
        import h5py

        dset_name = self._get_correction_dataset_name(window_size, strength)

        # Calculate total size
        T_total = sum(gf.shape[0] for gf in self.geofiles)
        S, B = self.geofiles[0].shape[1], self.geofiles[0].shape[2]

        print(
            f"📂 Loading saved illumination correction from {len(self.geofiles)} files..."
        )

        self.data_corrected = np.zeros((T_total, S, B), dtype=np.float32)

        t_offset = 0
        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                if dset_name not in f:
                    raise RuntimeError(
                        f"{gf.name}: corrected data not found at {dset_name}"
                    )

                dset = f[dset_name]
                T_file = dset.shape[0]
                self.data_corrected[t_offset : t_offset + T_file, :, :] = dset[()]

                # Print metadata
                if "window_size" in dset.attrs and "strength" in dset.attrs:
                    print(
                        f"   {gf.name}: window={dset.attrs['window_size']}, strength={dset.attrs['strength']}"
                    )

                t_offset += T_file

        print(f"✅ Loaded illumination correction from disk")
        return self.data_corrected

    def save_illumination_correction(self, window_size=1000, strength=1.0):
        """
        Save the illumination-corrected data to HDF5 files.
        Splits self.data_corrected back into individual files and saves with
        unique names based on parameters, allowing multiple cached versions.
        """
        import h5py
        import time

        if self.data_corrected is None:
            raise RuntimeError(
                "No corrected data to save. Run apply_illumination_correction first."
            )

        dset_path = self._get_correction_dataset_name(window_size, strength)

        print(f"💾 Saving illumination correction to {len(self.geofiles)} files...")
        print(f"   Dataset: {dset_path}")

        t_offset = 0
        for gf in self.geofiles:
            T_file = gf.shape[0]
            corrected_chunk = self.data_corrected[t_offset : t_offset + T_file, :, :]

            # Retry mechanism in case file is temporarily locked
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with h5py.File(
                        gf.path, "a"
                    ) as f:  # 'a' = read/write, create if not exists
                        # Remove old dataset if it exists (overwrite same parameters)
                        if dset_path in f:
                            del f[dset_path]

                        # Create new dataset with compression
                        dset = f.create_dataset(
                            dset_path,
                            data=corrected_chunk,
                            compression="gzip",
                            compression_opts=4,
                            dtype=np.float32,
                        )

                        # Save metadata
                        dset.attrs["window_size"] = (
                            window_size if window_size is not None else -1
                        )
                        dset.attrs["strength"] = strength
                        dset.attrs["description"] = (
                            "Illumination-corrected radiance data"
                        )

                        print(f"   ✓ {gf.name}: saved {corrected_chunk.shape}")
                    break  # Success, exit retry loop

                except BlockingIOError as e:
                    if attempt < max_retries - 1:
                        print(
                            f"   ⚠️  File locked, retrying in 1 second... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(1)
                    else:
                        print(
                            f"   ❌ Failed to save to {gf.name} after {max_retries} attempts"
                        )
                        print(
                            f"      Try closing any programs that might have the file open"
                        )
                        raise

            t_offset += T_file

        print(f"✅ Illumination correction saved to disk")

    def list_saved_corrections(self):
        """
        Display all cached illumination correction versions across all files.
        Shows parameters, dataset paths, and sizes for each cached version.
        """
        import h5py

        print("\n📋 Cached illumination corrections:")
        print("=" * 80)

        corrections_found = False
        for gf in self.geofiles:
            file_corrections = []

            with h5py.File(gf.path, "r") as f:
                # Look for all datasets matching the pattern
                if "processed/radiance" in f:
                    for key in f["processed/radiance"].keys():
                        if key.startswith("dataCube_illum_corrected_"):
                            dset = f[f"processed/radiance/{key}"]

                            # Extract metadata
                            window = dset.attrs.get("window_size", "unknown")
                            if window == -1:
                                window = "global"
                            strength = dset.attrs.get("strength", "unknown")

                            # Get size info
                            shape = dset.shape
                            size_mb = dset.nbytes / (1024 * 1024)

                            file_corrections.append(
                                {
                                    "dataset": key,
                                    "window": window,
                                    "strength": strength,
                                    "shape": shape,
                                    "size_mb": size_mb,
                                }
                            )

            if file_corrections:
                corrections_found = True
                print(f"\n📁 File: {gf.name}")
                for corr in file_corrections:
                    print(
                        f"   ✓ window={corr['window']}, strength={corr['strength']:.1f}"
                    )
                    print(f"     Dataset: {corr['dataset']}")
                    print(
                        f"     Shape: {corr['shape']}, Size: {corr['size_mb']:.1f} MB"
                    )

        if not corrections_found:
            print("\n   No cached corrections found in any files.")

        print("=" * 80)

    def delete_correction(self, window_size=1000, strength=1.0, confirm=True):
        """
        Delete a specific cached illumination correction from all files.

        Parameters:
        -----------
        window_size : int or None
            The window size of the correction to delete. None for global correction.
        strength : float
            The strength parameter of the correction to delete.
        confirm : bool, default=True
            If True, asks for confirmation before deleting.

        Returns:
        --------
        bool : True if deletion was successful, False if cancelled or not found.
        """
        import h5py

        dset_path = self._get_correction_dataset_name(window_size, strength)

        # Check if it exists in any files
        found_in_files = []
        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                if dset_path in f:
                    found_in_files.append(gf.name)

        if not found_in_files:
            print(
                f"❌ No cached correction found with window={window_size}, strength={strength}"
            )
            return False

        # Ask for confirmation if requested
        if confirm:
            print(
                f"\n⚠️  About to delete correction from {len(found_in_files)} file(s):"
            )
            print(f"   Parameters: window={window_size}, strength={strength}")
            print(f"   Dataset: {dset_path}")
            for fname in found_in_files:
                print(f"     - {fname}")

            response = input("\n   Delete these? (yes/no): ").strip().lower()
            if response not in ["yes", "y"]:
                print("   Deletion cancelled.")
                return False

        # Delete from all files
        print(f"\n🗑️  Deleting correction from {len(found_in_files)} file(s)...")
        for gf in self.geofiles:
            if gf.name in found_in_files:
                with h5py.File(gf.path, "a") as f:
                    if dset_path in f:
                        del f[dset_path]
                        print(f"   ✓ Deleted from {gf.name}")

        print(f"✅ Correction deleted successfully")
        return True

    def apply_illumination_correction_method2(self, window_size=1000):
        """
        METHOD 2: Intensity-based correction (preserves color ratios).
        MEMORY-EFFICIENT: Processes one slit at a time.

        This method computes a single intensity correction factor based on the mean
        across all bands, then applies the SAME correction to all bands. This preserves
        the RGB color ratios while still removing illumination variations.

        Parameters:
        -----------
        window_size : int, default=1000
            Number of tracks to use in the rolling window.

        Returns:
        --------
        self.data_corrected : np.ndarray
            Corrected data cube with shape (T, S, B)
        """
        from scipy.ndimage import median_filter

        print(f"🔄 METHOD 2: Intensity-based correction (window={window_size})...")
        print("   This method preserves color ratios by normalizing intensity only.")

        # Get dimensions by reading all file shapes
        T_total = 0
        S = None
        B = None

        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                dset_name = gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                if dset_name not in f:
                    dset_name = gf.DSET_RGB_MAIN
                shape = f[dset_name].shape
                T_total += shape[0]
                if S is None:
                    S = shape[1]
                    B = shape[2]

        print(f"   Total shape: ({T_total}, {S}, {B})")

        # Pre-allocate output
        self.data_corrected = np.zeros((T_total, S, B), dtype=np.float32)

        # Import tqdm
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False

        # Process each slit
        if use_tqdm:
            pbar = tqdm(total=S, desc="   Processing slits", unit="slit")

        for s in range(S):
            # Load this slit from all files: shape (T_total, B)
            slit_data_list = []
            for gf in self.geofiles:
                with h5py.File(gf.path, "r") as f:
                    dset_name = (
                        gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                    )
                    if dset_name not in f:
                        dset_name = gf.DSET_RGB_MAIN
                    slit_slice = f[dset_name][:, s, :].astype(np.float32)
                    slit_data_list.append(slit_slice)

            slit_data = np.concatenate(slit_data_list, axis=0)  # (T_total, B)
            del slit_data_list

            # Compute intensity (mean across all bands): shape (T_total,)
            intensity_series = np.mean(slit_data, axis=1)

            # Compute rolling median of intensity
            intensity_ref = median_filter(
                intensity_series, size=window_size, mode="nearest"
            )
            intensity_ref[intensity_ref == 0] = 1.0

            # Compute correction factor (same for all bands)
            correction_factor = intensity_series / intensity_ref  # (T_total,)

            # Apply same correction to all bands
            for b in range(B):
                self.data_corrected[:, s, b] = slit_data[:, b] / correction_factor

            del slit_data

            if use_tqdm:
                pbar.update(1)

        if use_tqdm:
            pbar.close()

        print("✅ data_corrected ready (METHOD 2: intensity-based, colors preserved)")
        return self.data_corrected

    def apply_illumination_correction_method3(
        self, window_size=1000, reference_band_index=None
    ):
        """
        METHOD 3: Reference-band correction (preserves color ratios).
        MEMORY-EFFICIENT: Processes one slit at a time.

        This method uses a single reference band (e.g., a stable near-infrared band)
        to compute the correction factor, then applies it to ALL bands. This preserves
        RGB color ratios while normalizing based on a stable reference.

        Parameters:
        -----------
        window_size : int, default=1000
            Number of tracks to use in the rolling window.

        reference_band_index : int, optional
            Index of the band to use as reference (0-209 for 210 bands).
            If None, uses the middle band (band 105).

        Returns:
        --------
        self.data_corrected : np.ndarray
            Corrected data cube with shape (T, S, B)
        """
        from scipy.ndimage import median_filter

        print(f"🔄 METHOD 3: Reference-band correction (window={window_size})...")
        print(
            "   This method preserves color ratios by normalizing based on a reference band."
        )

        # Get dimensions by reading all file shapes
        T_total = 0
        S = None
        B = None

        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                dset_name = gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                if dset_name not in f:
                    dset_name = gf.DSET_RGB_MAIN
                shape = f[dset_name].shape
                T_total += shape[0]
                if S is None:
                    S = shape[1]
                    B = shape[2]

        # Choose reference band (default: middle band)
        if reference_band_index is None:
            reference_band_index = B // 2

        if reference_band_index < 0 or reference_band_index >= B:
            raise ValueError(
                f"reference_band_index={reference_band_index} out of range [0, {B})"
            )

        print(f"   Using reference band: {reference_band_index}/{B}")
        print(f"   Total shape: ({T_total}, {S}, {B})")

        # Pre-allocate output
        self.data_corrected = np.zeros((T_total, S, B), dtype=np.float32)

        # Import tqdm
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False

        # Process each slit
        if use_tqdm:
            pbar = tqdm(total=S, desc="   Processing slits", unit="slit")

        for s in range(S):
            # Load this slit from all files: shape (T_total, B)
            slit_data_list = []
            for gf in self.geofiles:
                with h5py.File(gf.path, "r") as f:
                    dset_name = (
                        gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                    )
                    if dset_name not in f:
                        dset_name = gf.DSET_RGB_MAIN
                    slit_slice = f[dset_name][:, s, :].astype(np.float32)
                    slit_data_list.append(slit_slice)

            slit_data = np.concatenate(slit_data_list, axis=0)  # (T_total, B)
            del slit_data_list

            # Extract reference band time series: shape (T_total,)
            ref_series = slit_data[:, reference_band_index]

            # Compute rolling median of reference band
            ref_median = median_filter(ref_series, size=window_size, mode="nearest")
            ref_median[ref_median == 0] = 1.0

            # Compute correction factor (same for all bands)
            correction_factor = ref_series / ref_median  # (T_total,)

            # Apply same correction to all bands
            for b in range(B):
                self.data_corrected[:, s, b] = slit_data[:, b] / correction_factor

            del slit_data

            if use_tqdm:
                pbar.update(1)

        if use_tqdm:
            pbar.close()

        print(
            f"✅ data_corrected ready (METHOD 3: reference-band {reference_band_index}, colors preserved)"
        )
        return self.data_corrected

    def slice_tracks(self, start_track=0, end_track=None):
        """
        Create a new CombinedTransectCube with a subset of tracks.
        Tracks are relative to self.track_offset.
        """
        if end_track is None:
            end_track = self.data.shape[0]

        # Validate
        if not (0 <= start_track < self.data.shape[0]):
            raise ValueError(f"start_track {start_track} out of range")
        if not (start_track < end_track <= self.data.shape[0]):
            raise ValueError(f"end_track {end_track} invalid")

        new_cube = SimpleNamespace()
        new_cube.data = self.data[start_track:end_track, :, :]
        new_cube.wavelengths = self.wavelengths
        new_cube.track_offset = self.track_offset + start_track
        new_cube.name = f"{self.name}_tracks{start_track}-{end_track}"
        print(
            f"✅ Sliced cube: tracks {start_track}-{end_track}, shape {new_cube.data.shape}"
        )
        return new_cube

    def get_file_info(self, track_index):
        """Returns file boundary info for a given track index."""
        for i, (start, end) in enumerate(self.file_boundaries):
            if start <= track_index < end:
                return {"file_number": i + 1, "start_track": start, "end_track": end}
        return None

    def plot_rgb(
        self,
        red_wl=654.2,
        green_wl=560,
        blue_wl=440.3,
        spacing=4,
        track_index=None,
        slit_index=None,
        normalize=True,
        show_file_boundaries=True,
        figsize=None,
        use_corrected=False,
        perimeter_line=None,
        roi_pixels=None,  # Single ROI (backward compatibility)
        roi_collection=None,  # NEW: Multiple ROIs with names
        roi_colors=["yellow", "cyan", "magenta", "orange", "lime", "red", "blue"],
        roi_color_map=None,  # Dict mapping ROI names to specific colors, e.g., {"Sediment": "brown", "dark1": "black"}
        roi_marker_size=100,
        roi_marker_shape="s",  # Marker shape: 's'=square, 'o'=circle, '^'=triangle, 'D'=diamond, etc.
        roi_marker_edgewidth=2,  # Edge width for ROI markers (0 = no edge)
        roi_show_numbers=False,
        roi_legend_loc="best",  # Legend location: 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center', 'outside', or None to hide
        roi_legend_markersize=10,  # Size of color markers in legend (default=10)
        roi_legend_marker_border=True,  # Whether to show black border on legend markers
        line_colors=["red", "blue", "orange", "magenta"],
        line_width=2,
        line_style="-",
        line_labels=None,
    ):
        """
        Enhanced RGB plot supporting multiple named ROIs with different colors.

        NEW MULTI-ROI FEATURES:
        - roi_collection: Dict of named ROIs or 'all' to use cube.roi_collection
        - Automatic color assignment per ROI
        - Legend shows ROI names and pixel counts
        """

        # Original RGB setup (unchanged)
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )

        red_idx = np.argmin(np.abs(self.wavelengths - red_wl))
        green_idx = np.argmin(np.abs(self.wavelengths - green_wl))
        blue_idx = np.argmin(np.abs(self.wavelengths - blue_wl))

        R = cube_data[:, :, red_idx].T.copy()
        G = cube_data[:, :, green_idx].T.copy()
        B = cube_data[:, :, blue_idx].T.copy()

        if normalize:
            for C in (R, G, B):
                if C.max() != C.min():
                    C[:] = (C - C.min()) / (C.max() - C.min())

        rgb_image = np.stack([R, G, B], axis=-1)
        n_tracks, n_slits = cube_data.shape[0], cube_data.shape[1]

        if figsize is None:
            figsize = (12 * spacing, 6)
        plt.figure(figsize=figsize)

        plt.imshow(
            rgb_image, aspect="auto", origin="lower", extent=[0, n_tracks, 0, n_slits]
        )

        # File boundaries (unchanged)
        if show_file_boundaries:
            for b in self.file_boundaries[1:]:
                plt.axvline(
                    b["start_track"],
                    color="yellow",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.8,
                    label="File boundary" if b == self.file_boundaries[1] else "",
                )

        # Cross-hairs (unchanged)
        if track_index is not None:
            plt.axvline(track_index, color="cyan", linestyle="--", linewidth=2)
        if slit_index is not None:
            plt.axhline(slit_index, color="lime", linestyle="--", linewidth=2)

        # Perimeter lines (unchanged)
        if perimeter_line is not None:
            if isinstance(perimeter_line[0], (int, float)):
                lines_to_plot = [perimeter_line]
            else:
                lines_to_plot = perimeter_line

            for line_idx, line in enumerate(lines_to_plot):
                (slit1, track1), (slit2, track2) = line
                color = line_colors[line_idx % len(line_colors)]

                if line_labels and line_idx < len(line_labels):
                    label = line_labels[line_idx]
                elif len(lines_to_plot) > 1:
                    label = f"Line {line_idx + 1}"
                else:
                    label = "Perimeter line"

                plt.plot(
                    [track1, track2],
                    [slit1, slit2],
                    color=color,
                    linewidth=line_width,
                    linestyle=line_style,
                    label=label,
                )

        # NEW: Handle multiple ROIs
        if roi_collection is not None:
            # Helper function to sort ROIs in display order
            def sort_rois_by_category(roi_dict):
                """Sort ROIs: single bombs → double → triple → dark features → sediment"""
                order_keywords = [
                    ["single bomb"],
                    ["double bomb"],
                    ["tripple bomb"],
                    ["dark"],
                    ["sediment"],
                ]

                def get_sort_key(roi_name):
                    roi_lower = roi_name.lower()
                    for idx, keywords in enumerate(order_keywords):
                        if any(kw in roi_lower for kw in keywords):
                            return (idx, roi_name)
                    return (len(order_keywords), roi_name)

                sorted_items = sorted(
                    roi_dict.items(), key=lambda x: get_sort_key(x[0])
                )
                return dict(sorted_items)

            # Determine which ROIs to plot
            if roi_collection == "all":
                if hasattr(self, "roi_collection") and self.roi_collection:
                    rois_to_plot = self.roi_collection
                else:
                    print(
                        "⚠️  No ROI collection found. Use plot_interactive_rgb() first."
                    )
                    rois_to_plot = {}
            elif isinstance(roi_collection, list):
                # NEW: Support list of ROI names (like plot_spectrum)
                if hasattr(self, "roi_collection") and self.roi_collection:
                    rois_to_plot = {
                        name: self.roi_collection[name]
                        for name in roi_collection
                        if name in self.roi_collection
                    }
                else:
                    print("⚠️  No ROI collection found.")
                    rois_to_plot = {}
            elif isinstance(roi_collection, dict):
                rois_to_plot = roi_collection
            else:
                print(
                    "❌ roi_collection must be 'all', a list of ROI names, or a dictionary"
                )
                rois_to_plot = {}

            # Sort ROIs for consistent legend order
            rois_to_plot = sort_rois_by_category(rois_to_plot)

            # Plot each ROI with different color
            for roi_idx, (roi_name, roi_pixels_list) in enumerate(rois_to_plot.items()):
                if roi_pixels_list:
                    # Validate coordinates
                    valid_rois = [
                        (slit, track)
                        for slit, track in roi_pixels_list
                        if 0 <= slit < n_slits and 0 <= track < n_tracks
                    ]

                    if valid_rois:
                        roi_tracks = [track for slit, track in valid_rois]
                        roi_slits = [slit for slit, track in valid_rois]

                        # NEW: Get consistent color across all plots
                        color = self._get_roi_color(roi_name, roi_colors, roi_color_map)

                        # Plot ROI with unique color
                        plt.scatter(
                            roi_tracks,
                            roi_slits,
                            c=color,
                            s=roi_marker_size,
                            marker=roi_marker_shape,
                            edgecolors="black" if roi_marker_edgewidth > 0 else "none",
                            linewidths=roi_marker_edgewidth,
                            alpha=0.8,
                            label=f"{roi_name} ({len(valid_rois)})",
                        )

                        # Optional: Add numbers for each ROI
                        if roi_show_numbers:
                            for i, (slit, track) in enumerate(valid_rois, 1):
                                plt.text(
                                    track,
                                    slit + 3,
                                    str(i),
                                    ha="center",
                                    va="bottom",
                                    fontsize=8,
                                    fontweight="bold",
                                    color="black",
                                    bbox=dict(
                                        boxstyle="round,pad=0.2",
                                        facecolor="white",
                                        alpha=0.8,
                                    ),
                                )

        # Backward compatibility: single ROI support
        elif roi_pixels is not None and len(roi_pixels) > 0:
            valid_rois = [
                (slit, track)
                for slit, track in roi_pixels
                if 0 <= slit < n_slits and 0 <= track < n_tracks
            ]

            if valid_rois:
                roi_tracks = [track for slit, track in valid_rois]
                roi_slits = [slit for slit, track in valid_rois]

                plt.scatter(
                    roi_tracks,
                    roi_slits,
                    c=roi_colors[0],
                    s=roi_marker_size,
                    marker=roi_marker_shape,
                    edgecolors="black" if roi_marker_edgewidth > 0 else "none",
                    linewidths=roi_marker_edgewidth,
                    alpha=0.8,
                    label=f"ROI Pixels ({len(valid_rois)})",
                )

        # Grid (unchanged)
        for x in np.arange(0, n_tracks + 1, 50):
            plt.axvline(x, color="black", linewidth=0.5, alpha=0.3)
        for y in np.arange(0, n_slits + 1, 50):
            plt.axhline(y, color="black", linewidth=0.5, alpha=0.3)

        plt.xlabel("Track Index")
        plt.ylabel("Slit Pixel Index")
        title = f"RGB Composite - {self.name}\n(R={red_wl}nm, G={green_wl}nm, B={blue_wl}nm)"
        if use_corrected:
            title = "Corrected " + title
        plt.title(title)

        # Smart legend display
        has_overlays = (
            (show_file_boundaries and len(self.file_boundaries) > 1)
            or perimeter_line is not None
            or roi_collection is not None
            or (roi_pixels is not None and len(roi_pixels) > 0)
        )
        if has_overlays and roi_legend_loc is not None:
            if roi_legend_loc == "outside":
                # Place legend outside the plot area on the right
                legend = plt.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    markerscale=roi_legend_markersize / 6,
                )
            else:
                legend = plt.legend(
                    loc=roi_legend_loc,
                    markerscale=roi_legend_markersize / 6,
                )

            # Configure legend marker borders
            if not roi_legend_marker_border and legend:
                for handle in legend.legend_handles:
                    if hasattr(handle, "set_edgecolor"):
                        handle.set_edgecolor("none")
                        handle.set_linewidth(0)

        plt.tight_layout()
        plt.show()

    def plot_line_intensity_profile(
        self,
        perimeter_line,
        wavelength=None,
        wavelength_range=None,  # NEW: Wavelength range for averaging ('red', 'green', 'blue', 'all', or tuple)
        use_average=False,
        use_corrected=False,
        figsize=(12, 6),
        line_colors=["red", "blue", "purple", "magenta", "pink"],
        line_width=2,
        moving_average_window=1,
        moving_average_percent=None,  # NEW: Auto-calculate window as % of line length (overrides moving_average_window)
        show_markers=True,
        marker_size=4,
        line_labels=None,
        normalize_intensities=False,  # NEW: Normalize intensity values
        normalization_method="minmax",  # NEW: 'minmax', 'zscore', or 'mean'
    ):
        """
        Plot intensity profiles with optional intensity normalization for better comparison.

        WAVELENGTH OPTIONS:
        - wavelength=654.2: Single wavelength (creates 1 plot)
        - use_average=True: Average across wavelengths
          - wavelength_range='red': Average red bands (640-680nm)
          - wavelength_range='green': Average green bands (520-580nm)
          - wavelength_range='blue': Average blue bands (420-480nm)
          - wavelength_range='all' or None: Average all bands
          - wavelength_range=(500, 600): Custom range in nm

        NORMALIZATION OPTIONS:
        - normalize_intensities=True: Enable intensity normalization
        - normalization_method='minmax': Scale to [0,1] range
        - normalization_method='zscore': Z-score normalization (mean=0, std=1)
        - normalization_method='mean': Divide by mean (relative to average)
        """

        # Handle single vs multiple lines
        if isinstance(perimeter_line[0], (int, float)):
            lines_to_plot = [perimeter_line]
        else:
            lines_to_plot = perimeter_line

        # Validate lines
        for i, line in enumerate(lines_to_plot, 1):
            if not (isinstance(line, list) and len(line) == 2):
                raise ValueError(f"Line {i} must be [(slit1,track1), (slit2,track2)]")

        # Calculate consistent sampling length
        line_lengths = []
        for line in lines_to_plot:
            (slit1, track1), (slit2, track2) = line
            length = max(abs(track2 - track1), abs(slit2 - slit1)) + 1
            line_lengths.append(length)

        max_length = max(line_lengths)

        # Auto-calculate moving average window from percentage
        if moving_average_percent is not None:
            moving_average_window = max(
                1, int(max_length * moving_average_percent / 100)
            )
            print(
                f"📊 Auto-calculated moving average window: {moving_average_window} samples ({moving_average_percent}% of {max_length} samples)"
            )

        # Get data and wavelength setup
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )
        n_tracks, n_slits, n_wavelengths = cube_data.shape

        # Determine which wavelength bands to use
        if use_average:
            # Define wavelength ranges for averaging
            if wavelength_range is None or wavelength_range == "all":
                wl_mask = np.ones(len(self.wavelengths), dtype=bool)
                range_label = "all wavelengths"
            elif wavelength_range == "red":
                wl_mask = (self.wavelengths >= 640) & (self.wavelengths <= 680)
                range_label = "red (640-680nm)"
            elif wavelength_range == "green":
                wl_mask = (self.wavelengths >= 520) & (self.wavelengths <= 580)
                range_label = "green (520-580nm)"
            elif wavelength_range == "blue":
                wl_mask = (self.wavelengths >= 420) & (self.wavelengths <= 480)
                range_label = "blue (420-480nm)"
            elif (
                isinstance(wavelength_range, (tuple, list))
                and len(wavelength_range) == 2
            ):
                wl_min, wl_max = wavelength_range
                wl_mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
                range_label = f"{wl_min}-{wl_max}nm"
            else:
                raise ValueError(f"Invalid wavelength_range: {wavelength_range}")

            if not np.any(wl_mask):
                raise ValueError(f"No wavelengths found in range: {wavelength_range}")

            method = "average"
            base_ylabel = f"Average Intensity ({range_label})"
            wl_indices = np.where(wl_mask)[0]

        elif wavelength is not None:
            wl_idx = np.argmin(np.abs(self.wavelengths - wavelength))
            method = "specific"
            base_ylabel = f"Intensity at {self.wavelengths[wl_idx]:.1f} nm"
            wl_indices = [wl_idx]
        else:
            median_wl = np.median(self.wavelengths)
            wl_idx = np.argmin(np.abs(self.wavelengths - median_wl))
            method = "median"
            base_ylabel = f"Intensity at {self.wavelengths[wl_idx]:.1f} nm"
            wl_indices = [wl_idx]

        # NEW: Adjust ylabel based on normalization
        if normalize_intensities:
            if normalization_method == "minmax":
                ylabel = f"Normalized {base_ylabel} [0-1]"
            elif normalization_method == "zscore":
                ylabel = f"Z-Score {base_ylabel}"
            elif normalization_method == "mean":
                ylabel = f"Relative {base_ylabel}"
            else:
                ylabel = f"Normalized {base_ylabel}"
        else:
            ylabel = base_ylabel

        # Get wavelength info for display
        if method == "average":
            actual_wl_info = range_label
        else:
            actual_wl = self.wavelengths[wl_indices[0]]
            actual_wl_info = f"{actual_wl:.1f} nm"

        plt.figure(figsize=figsize)
        all_results = []

        for line_idx, line in enumerate(lines_to_plot):
            (slit1, track1), (slit2, track2) = line

            # Validate coordinates
            for i, (s, t) in enumerate([(slit1, track1), (slit2, track2)], 1):
                if not (0 <= s < n_slits):
                    raise ValueError(
                        f"Line {line_idx+1}, Point {i}: slit {s} out of range [0, {n_slits-1}]"
                    )
                if not (0 <= t < n_tracks):
                    raise ValueError(
                        f"Line {line_idx+1}, Point {i}: track {t} out of range [0, {n_tracks-1}]"
                    )

            # Generate consistent sampling points
            track_indices = np.linspace(track1, track2, max_length, dtype=int)
            slit_indices = np.linspace(slit1, slit2, max_length, dtype=int)

            # Extract raw intensities
            intensities = []
            for track_idx, slit_idx in zip(track_indices, slit_indices):
                if 0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits:
                    if len(wl_indices) == 1:
                        # Single wavelength
                        intensity = cube_data[track_idx, slit_idx, wl_indices[0]]
                    else:
                        # Average across multiple wavelengths
                        intensity = np.mean(cube_data[track_idx, slit_idx, wl_indices])
                    intensities.append(intensity)

            intensities = np.array(intensities)

            # NEW: Apply intensity normalization
            if normalize_intensities and len(intensities) > 0:
                if normalization_method == "minmax":
                    # Scale to [0, 1]
                    if intensities.max() != intensities.min():
                        intensities = (intensities - intensities.min()) / (
                            intensities.max() - intensities.min()
                        )
                elif normalization_method == "zscore":
                    # Z-score normalization (mean=0, std=1)
                    if intensities.std() != 0:
                        intensities = (
                            intensities - intensities.mean()
                        ) / intensities.std()
                elif normalization_method == "mean":
                    # Divide by mean (relative values)
                    if intensities.mean() != 0:
                        intensities = intensities / intensities.mean()

            sample_indices = np.arange(len(intensities))

            # Apply smoothing
            if moving_average_window > 1:
                if moving_average_window > len(intensities):
                    moving_average_window = len(intensities)

                window = np.ones(moving_average_window) / moving_average_window
                smoothed = np.convolve(intensities, window, mode="valid")

                std_devs = []
                for i in range(len(smoothed)):
                    window_data = intensities[i : i + moving_average_window]
                    std_devs.append(np.std(window_data))
                std_devs = np.array(std_devs)

                half_window = moving_average_window // 2
                plot_intensities = smoothed
                plot_sample_indices = np.arange(len(smoothed))
                has_std = True
            else:
                plot_intensities = intensities
                plot_sample_indices = sample_indices
                std_devs = None
                has_std = False

            # Styling and labels
            color = line_colors[line_idx % len(line_colors)]

            if line_labels and line_idx < len(line_labels):
                label = line_labels[line_idx]
            elif len(lines_to_plot) > 1:
                label = f"Line {line_idx + 1}"
            else:
                label = "Smoothed" if has_std else "Raw intensity"

            # Plot the line
            marker_style = "o" if show_markers else None
            marker_size_actual = marker_size if show_markers else 0

            plt.plot(
                plot_sample_indices,
                plot_intensities,
                color=color,
                linewidth=line_width,
                marker=marker_style,
                markersize=marker_size_actual,
                label=label,
            )

            # Add std dev bands
            if has_std:
                plt.fill_between(
                    plot_sample_indices,
                    plot_intensities - std_devs,
                    plot_intensities + std_devs,
                    color=color,
                    alpha=0.1,
                )

            all_results.append(
                {
                    "line_index": line_idx + 1,
                    "start_point": (slit1, track1),
                    "end_point": (slit2, track2),
                    "sample_indices": plot_sample_indices,
                    "intensities": plot_intensities,
                    "std_deviation": std_devs,
                    "raw_intensities": intensities,
                    "normalized": normalize_intensities,
                    "normalization_method": (
                        normalization_method if normalize_intensities else None
                    ),
                    "color": color,
                }
            )

        # Plot formatting
        plt.xlabel("Sample Index")
        plt.ylabel(ylabel)

        # Smart title with normalization info
        if len(lines_to_plot) == 1:
            title = f"Intensity Profile ({method} wavelength)"
        else:
            title = f"Multi-Line Comparison ({len(lines_to_plot)} lines, {method} wavelength)"

        if normalize_intensities:
            title += f" - {normalization_method.upper()} normalized"

        if moving_average_window > 1:
            title += f"\nSmoothed with {moving_average_window}-point moving average"

        plt.title(title)
        plt.grid(True, alpha=0.3)

        if len(lines_to_plot) > 1 or has_std:
            plt.legend()

        # Updated stats
        norm_info = (
            f" ({normalization_method} normalized)" if normalize_intensities else ""
        )
        stats_text = f"Wavelength: {actual_wl_info}{norm_info} | Sample points: {max_length} | Lines: {len(lines_to_plot)}"
        plt.figtext(0.02, 0.01, stats_text, fontsize=9, ha="left")

        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.show()

        return {
            "wavelength": actual_wl_info,
            "wavelength_method": method,
            "smoothing_window": moving_average_window,
            "number_of_lines": len(lines_to_plot),
            "sample_points": max_length,
            "normalized": normalize_intensities,
            "normalization_method": (
                normalization_method if normalize_intensities else None
            ),
            "lines": all_results,
        }

    def plot_wavelength_comparison(
        self,
        line,
        wavelength_ranges=None,
        use_corrected=False,
        figsize=(12, 8),
        moving_average_window=1,
        moving_average_percent=None,  # NEW: Auto-calculate window as % of line length
        show_markers=False,
        marker_size=4,
        normalize_intensities=False,
        normalization_method="minmax",
    ):
        """
        Compare different wavelength ranges for a single line.

        Parameters:
        -----------
        line : tuple
            Single line definition: ((slit1, track1), (slit2, track2))
        wavelength_ranges : list, optional
            List of wavelength ranges to compare. Default: ['red', 'green', 'blue', 'all']
            Can use: 'red', 'green', 'blue', 'all', or tuples like (500, 600)

        Example:
        --------
        cube.plot_wavelength_comparison(
            line=lines[0],  # First line
            wavelength_ranges=['red', 'green', 'blue', 'all'],
            moving_average_window=50,
            use_corrected=True
        )
        """

        if wavelength_ranges is None:
            wavelength_ranges = ["red", "green", "blue", "all"]

        # Validate line format
        if not (isinstance(line, (list, tuple)) and len(line) == 2):
            raise ValueError("Line must be [(slit1,track1), (slit2,track2)]")

        (slit1, track1), (slit2, track2) = line

        # Get data
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )
        n_tracks, n_slits, n_wavelengths = cube_data.shape

        # Validate coordinates
        if not (0 <= slit1 < n_slits and 0 <= slit2 < n_slits):
            raise ValueError(f"Slit indices out of range [0, {n_slits-1}]")
        if not (0 <= track1 < n_tracks and 0 <= track2 < n_tracks):
            raise ValueError(f"Track indices out of range [0, {n_tracks-1}]")

        # Calculate sampling length
        length = max(abs(track2 - track1), abs(slit2 - slit1)) + 1

        # Auto-calculate moving average window from percentage
        if moving_average_percent is not None:
            moving_average_window = max(1, int(length * moving_average_percent / 100))
            print(
                f"📊 Auto-calculated moving average window: {moving_average_window} samples ({moving_average_percent}% of {length} samples)"
            )

        # Generate sampling points
        track_indices = np.linspace(track1, track2, length, dtype=int)
        slit_indices = np.linspace(slit1, slit2, length, dtype=int)

        plt.figure(figsize=figsize)

        # Color scheme for different ranges
        range_colors = {
            "red": "#e74c3c",
            "green": "#27ae60",
            "blue": "#3498db",
            "all": "#95a5a6",
        }

        for wl_range in wavelength_ranges:
            # Define wavelength mask
            if wl_range == "all":
                wl_mask = np.ones(len(self.wavelengths), dtype=bool)
                range_label = "All wavelengths"
            elif wl_range == "red":
                wl_mask = (self.wavelengths >= 640) & (self.wavelengths <= 680)
                range_label = "Red (640-680nm)"
            elif wl_range == "green":
                wl_mask = (self.wavelengths >= 520) & (self.wavelengths <= 580)
                range_label = "Green (520-580nm)"
            elif wl_range == "blue":
                wl_mask = (self.wavelengths >= 420) & (self.wavelengths <= 480)
                range_label = "Blue (420-480nm)"
            elif isinstance(wl_range, (tuple, list)) and len(wl_range) == 2:
                wl_min, wl_max = wl_range
                wl_mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
                range_label = f"{wl_min}-{wl_max}nm"
            else:
                raise ValueError(f"Invalid wavelength_range: {wl_range}")

            if not np.any(wl_mask):
                print(f"⚠️ Warning: No wavelengths found in range: {wl_range}")
                continue

            wl_indices = np.where(wl_mask)[0]

            # Extract intensities
            intensities = []
            for track_idx, slit_idx in zip(track_indices, slit_indices):
                if 0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits:
                    # Average across wavelength range
                    intensity = np.mean(cube_data[track_idx, slit_idx, wl_indices])
                    intensities.append(intensity)

            intensities = np.array(intensities)

            # Apply normalization
            if normalize_intensities and len(intensities) > 0:
                if normalization_method == "minmax":
                    if intensities.max() != intensities.min():
                        intensities = (intensities - intensities.min()) / (
                            intensities.max() - intensities.min()
                        )
                elif normalization_method == "zscore":
                    if intensities.std() != 0:
                        intensities = (
                            intensities - intensities.mean()
                        ) / intensities.std()
                elif normalization_method == "mean":
                    if intensities.mean() != 0:
                        intensities = intensities / intensities.mean()

            sample_indices = np.arange(len(intensities))

            # Apply smoothing
            if moving_average_window > 1:
                if moving_average_window > len(intensities):
                    moving_average_window = len(intensities)

                window = np.ones(moving_average_window) / moving_average_window
                smoothed = np.convolve(intensities, window, mode="valid")

                std_devs = []
                for i in range(len(smoothed)):
                    window_data = intensities[i : i + moving_average_window]
                    std_devs.append(np.std(window_data))
                std_devs = np.array(std_devs)

                half_window = moving_average_window // 2
                plot_intensities = smoothed
                plot_sample_indices = np.arange(len(smoothed))
                has_std = True
            else:
                plot_intensities = intensities
                plot_sample_indices = sample_indices
                std_devs = None
                has_std = False

            # Get color
            color = range_colors.get(wl_range, "#34495e")
            if isinstance(wl_range, (tuple, list)):
                color = "#9b59b6"  # Purple for custom ranges

            # Plot
            marker_style = "o" if show_markers else None
            marker_size_actual = marker_size if show_markers else 0

            plt.plot(
                plot_sample_indices,
                plot_intensities,
                color=color,
                linewidth=2.5,
                marker=marker_style,
                markersize=marker_size_actual,
                label=range_label,
                alpha=0.8,
            )

            # Add std dev bands
            if has_std:
                plt.fill_between(
                    plot_sample_indices,
                    plot_intensities - std_devs,
                    plot_intensities + std_devs,
                    color=color,
                    alpha=0.15,
                )

        # Formatting
        plt.xlabel("Sample Index", fontsize=12)

        if normalize_intensities:
            if normalization_method == "minmax":
                ylabel = "Normalized Intensity [0-1]"
            elif normalization_method == "zscore":
                ylabel = "Z-Score Intensity"
            elif normalization_method == "mean":
                ylabel = "Relative Intensity"
            else:
                ylabel = "Normalized Intensity"
        else:
            ylabel = "Average Intensity"

        plt.ylabel(ylabel, fontsize=12)

        title = f"Wavelength Range Comparison"
        if normalize_intensities:
            title += f" ({normalization_method.upper()} normalized)"
        if moving_average_window > 1:
            title += f"\nSmoothed with {moving_average_window}-point moving average"

        plt.title(title, fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc="best")

        # Stats
        stats_text = (
            f"Line: ({slit1},{track1}) → ({slit2},{track2}) | Sample points: {length}"
        )
        if normalize_intensities:
            stats_text += f" | {normalization_method} normalized"
        plt.figtext(0.02, 0.01, stats_text, fontsize=9, ha="left")

        plt.tight_layout()
        plt.show()

    def list_rois(self):
        """Display all saved ROIs"""
        if not hasattr(self, "roi_collection") or not self.roi_collection:
            print("❌ No ROIs saved yet")
            return

        print(f"\n📂 ROI Collection ({len(self.roi_collection)} ROIs):")
        print("=" * 50)
        for i, (name, pixels) in enumerate(self.roi_collection.items(), 1):
            slits = [s for s, t in pixels]
            tracks = [t for s, t in pixels]
            print(f"{i:2d}. '{name}': {len(pixels)} pixels")
            print(f"     Slit range:  {min(slits):3d} - {max(slits):3d}")
            print(f"     Track range: {min(tracks):3d} - {max(tracks):3d}")

    def delete_roi(self, roi_name):
        """Delete a specific ROI"""
        if not hasattr(self, "roi_collection") or roi_name not in self.roi_collection:
            print(f"❌ ROI '{roi_name}' not found")
            return

        pixel_count = len(self.roi_collection[roi_name])
        del self.roi_collection[roi_name]
        print(f"🗑️  Deleted ROI '{roi_name}' ({pixel_count} pixels)")

    def _ensure_roi_tuples(self):
        """
        Internal helper: Ensure all ROI pixels are stored as tuples, not lists.
        This fixes issues from importing JSON where lists get loaded instead of tuples.
        """
        for roi_name, pixels in self.roi_collection.items():
            # Convert any lists to tuples
            self.roi_collection[roi_name] = set(
                tuple(pixel) if isinstance(pixel, list) else pixel for pixel in pixels
            )

    def export_rois(self, filename=None):
        """Export ROIs to JSON file"""
        if not hasattr(self, "roi_collection") or not self.roi_collection:
            print("❌ No ROIs to export")
            return

        if filename is None:
            filename = f"roi_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Ensure all pixels are tuples (fixes any lingering list objects)
        self._ensure_roi_tuples()

        # Convert sets to lists for JSON serialization
        roi_collection_serializable = {
            roi_name: list(pixels) for roi_name, pixels in self.roi_collection.items()
        }

        export_data = {
            "transect_name": self.name,
            "export_date": datetime.now().isoformat(),
            "roi_collection": roi_collection_serializable,
        }

        try:
            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"💾 Exported {len(self.roi_collection)} ROIs to {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
            print(f"💡 This usually means ROI data is corrupted. Try reimporting ROIs.")

    def import_rois(self, filename):
        """Import ROIs from JSON file"""
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            if not hasattr(self, "roi_collection"):
                self.roi_collection = {}

            imported_rois = data["roi_collection"]

            # Convert lists to tuples (JSON stores lists, we need tuples for sets)
            for roi_name, pixels in imported_rois.items():
                self.roi_collection[roi_name] = set(
                    tuple(pixel) if isinstance(pixel, list) else pixel
                    for pixel in pixels
                )

            print(f"📂 Imported {len(imported_rois)} ROIs from {filename}")

        except Exception as e:
            print(f"❌ Import failed: {e}")

    def combine_rois(self, roi_names_to_combine, new_roi_name):
        """
        Combine multiple ROIs into a single ROI with a new name.

        Parameters:
        -----------
        roi_names_to_combine : list of str
            List of ROI names to combine
        new_roi_name : str
            Name for the combined ROI

        Returns:
        --------
        int : Number of pixels in the combined ROI

        Example:
        --------
        cube.combine_rois(roi_names_to_combine=["sed1", "sed2", "sed3"], new_roi_name="all_sediment")
        """
        # Ensure all ROIs have tuples (not lists)
        self._ensure_roi_tuples()

        combined_pixels = set()

        for roi_name in roi_names_to_combine:
            if roi_name in self.roi_collection:
                roi_pixels = self.roi_collection[roi_name]
                combined_pixels.update(roi_pixels)
                print(f"  Added {len(roi_pixels)} pixels from '{roi_name}'")
            else:
                print(f"  Warning: ROI '{roi_name}' not found, skipping")

        if combined_pixels:
            self.roi_collection[new_roi_name] = combined_pixels
            print(f"✅ Created ROI '{new_roi_name}' with {len(combined_pixels)} pixels")
            return len(combined_pixels)
        else:
            print(f"❌ No pixels found, ROI '{new_roi_name}' not created")
            return 0

    def rename_roi(self, old_roi_name, new_roi_name):
        """
        Rename an existing ROI.

        Parameters:
        -----------
        old_roi_name : str
            Current name of the ROI
        new_roi_name : str
            New name for the ROI

        Returns:
        --------
        bool : True if successful, False otherwise

        Example:
        --------
        cube.rename_roi(old_roi_name="sed1", new_roi_name="sediment_1")
        """
        if old_roi_name not in self.roi_collection:
            print(f"❌ ROI '{old_roi_name}' not found")
            return False

        if new_roi_name in self.roi_collection:
            print(f"❌ ROI '{new_roi_name}' already exists")
            return False

        # Copy pixels to new name and delete old name
        self.roi_collection[new_roi_name] = self.roi_collection[old_roi_name]
        del self.roi_collection[old_roi_name]

        print(
            f"✅ Renamed ROI '{old_roi_name}' → '{new_roi_name}' ({len(self.roi_collection[new_roi_name])} pixels)"
        )
        return True

    def plot_interactive_rgb(
        self,
        red_wl=654.2,
        green_wl=560,
        blue_wl=440.3,
        normalize=True,
        use_corrected=False,
        figsize=(50, 10),
        roi_name=None,
        load_existing=True,
        highlight_color=[1, 0, 0, 0.8],  # Bright red with transparency
        marker_size=3,  # Size of marker in pixels (e.g., 3 = 3x3 square)
    ):
        """
        Interactive RGB with pixel highlighting and right-click deletion.

        CONTROLS:
        - Left click: Add pixel (prevents duplicates)
        - Right click: Remove pixel
        - Close window: Save ROI
        """

        # ROI management setup
        if not hasattr(self, "roi_collection"):
            self.roi_collection = {}

        if roi_name is None:
            roi_name = input("Enter ROI name: ").strip()
            if not roi_name:
                roi_name = f"ROI_{len(self.roi_collection) + 1}"

        # Load existing or start fresh
        if load_existing and roi_name in self.roi_collection:
            # Convert lists to tuples for hashable set items
            roi_data = self.roi_collection[roi_name]
            current_roi = set(
                tuple(pixel) if isinstance(pixel, list) else pixel for pixel in roi_data
            )
            print(f"📂 Loaded '{roi_name}' with {len(current_roi)} pixels")
        else:
            current_roi = set()
            print(f"✨ Creating new ROI '{roi_name}'")

        # RGB data preparation
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )

        red_idx = np.argmin(np.abs(self.wavelengths - red_wl))
        green_idx = np.argmin(np.abs(self.wavelengths - green_wl))
        blue_idx = np.argmin(np.abs(self.wavelengths - blue_wl))

        R = cube_data[:, :, red_idx].T.copy()
        G = cube_data[:, :, green_idx].T.copy()
        B = cube_data[:, :, blue_idx].T.copy()

        if normalize:
            for C in (R, G, B):
                if C.max() != C.min():
                    C[:] = (C - C.min()) / (C.max() - C.min())

        rgb_image = np.stack([R, G, B], axis=-1)
        n_tracks, n_slits = cube_data.shape[0], cube_data.shape[1]

        # Create plot with overlay
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            rgb_image, aspect="auto", origin="lower", extent=[0, n_tracks, 0, n_slits]
        )

        # Highlight overlay layer
        highlight_overlay = np.zeros((n_slits, n_tracks, 4), dtype=np.float32)
        overlay_im = ax.imshow(
            highlight_overlay,
            aspect="auto",
            origin="lower",
            extent=[0, n_tracks, 0, n_slits],
            interpolation="nearest",
        )

        def update_highlights():
            """Refresh pixel highlights"""
            highlight_overlay[:, :, :] = 0

            half_size = marker_size // 2
            for slit_idx, track_idx in current_roi:
                # Draw a marker_size x marker_size square centered on the pixel
                for dy in range(-half_size, half_size + 1):
                    for dx in range(-half_size, half_size + 1):
                        y = slit_idx + dy
                        x = track_idx + dx
                        if 0 <= y < n_slits and 0 <= x < n_tracks:
                            highlight_overlay[y, x, :] = highlight_color

            overlay_im.set_array(highlight_overlay)
            ax.set_title(f"{roi_name}: {len(current_roi)} pixels selected")
            fig.canvas.draw_idle()

        # Initialize highlights
        update_highlights()

        def on_click(event):
            """Handle both adding and removing pixels"""
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return

            # Get pixel coordinates
            track_idx = int(np.round(event.xdata))
            slit_idx = int(np.round(event.ydata))

            # Validate bounds
            if not (0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits):
                print("⚠️ Click outside valid area")
                return

            pixel = (slit_idx, track_idx)

            if event.button == 1:  # Left click - ADD pixel
                if pixel in current_roi:
                    print(
                        f"📌 Pixel already selected: (slit={slit_idx}, track={track_idx})"
                    )
                else:
                    current_roi.add(pixel)
                    print(
                        f"✅ Added pixel: (slit={slit_idx}, track={track_idx}) - Total: {len(current_roi)}"
                    )

            elif event.button == 3:  # Right click - REMOVE pixel
                if pixel in current_roi:
                    current_roi.remove(pixel)
                    print(
                        f"❌ Removed pixel: (slit={slit_idx}, track={track_idx}) - Total: {len(current_roi)}"
                    )
                else:
                    print(f"⚠️ Pixel not selected: (slit={slit_idx}, track={track_idx})")

            # Update display
            update_highlights()

        def on_close(event):
            """Save ROI collection"""
            if current_roi:
                self.roi_collection[roi_name] = list(current_roi)
                print(f"💾 Saved ROI '{roi_name}' with {len(current_roi)} pixels")
            else:
                print(f"🗑️ No pixels in '{roi_name}'")

        # Event handlers
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("close_event", on_close)

        # Grid for better pixel visibility
        for x in range(0, n_tracks + 1, 50):
            ax.axvline(x, color="black", linewidth=0.5, alpha=0.2)
        for y in range(0, n_slits + 1, 50):
            ax.axhline(y, color="black", linewidth=0.5, alpha=0.2)

        ax.set_xlabel("Track Index")
        ax.set_ylabel("Slit Pixel Index")

        # Updated instructions
        instructions = f"""
        🖱️ PIXEL EDITOR: '{roi_name}'
        • Left click: Add pixel (red highlight)
        • Right click: Remove pixel
        • Close window: Save ROI
        """
        fig.text(
            0.02,
            0.98,
            instructions,
            fontsize=10,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

        return list(current_roi)

    def plot_spectrum(
        self,
        track_index=None,
        slit_index=None,
        roi_pixels=None,
        roi_name=None,
        roi_names=None,
        ylabel="Intensity",  # Will update dynamically
        use_corrected=False,
        wavelength_range=None,
        wavelength_smoothing=1,
        figsize=(12, 6),
        colors=[
            "#FF0000",
            "#0000FF",
            "#00FFFF",
            "#FF00FF",
            "#800080",
            "#A52A2A",
            "#000000",
            "#FFFF00",
            "#00FF00",
            "#B8860B",
        ],
        color_map=None,  # Dict mapping ROI names to specific colors, e.g., {"Sediment": "brown", "dark1": "black"}
        use_inline_labels=True,
        normalize=False,  # DEPRECATED: Use normalize_method instead (kept for backward compatibility)
        normalize_method=None,  # NEW: Advanced normalization options
        show_std=True,  # NEW: Control standard deviation bands
        interpolate_wavelengths=None,  # NEW: List of wavelength indices to interpolate (e.g., [104] for 560nm dip)
        legend_loc="best",  # NEW: Legend location ('best', 'upper right', 'outside', etc., or None to hide)
    ):
        """Enhanced spectrum plotting with optional normalization, inline labels, std control, wavelength interpolation, and legend placement.

        Parameters:
        -----------
        normalize : bool, optional (DEPRECATED)
            If True, divides by mean (kept for backward compatibility). Use normalize_method instead.

        normalize_method : str or None, optional
            Advanced normalization method for spectral offset removal:
            - None or False: No normalization (default, shows raw/corrected data)
            - "mean": Divide by mean (simple scaling, preserves amplitude differences)
            - "mean_center": Subtract mean (removes DC offset, preserves shape) ⭐ Recommended for offset removal
            - "snv": Standard Normal Variate (removes offset + scale, preserves shape) ⭐ Recommended for spectral comparison
            - "msc": Multiplicative Scatter Correction (physically meaningful, uses mean spectrum as reference)
            - "minmax": Min-max scaling to [0,1] (NOT recommended for spectral shape comparison)
            - "l2": L2 vector normalization (useful for clustering/PCA)

        interpolate_wavelengths : list of int, optional
            List of wavelength indices to interpolate (replace with linear interpolation from neighbors).
            Example: [104] will interpolate wavelength 104 (~560nm) from wavelengths 103 and 105.
            The function will print which wavelengths (nm) are being interpolated.
        """

        cube = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )
        data_label = "Corrected" if use_corrected else "Raw"

        if cube is None:
            print("❌ No data loaded.")
            return

        # Helper function to sort ROIs in display order
        def sort_rois_by_category(roi_dict):
            """Sort ROIs: single bombs → double → triple → dark features → sediment"""
            order_keywords = [
                # Single bombs first
                ["single bomb"],
                # Double bombs second
                ["double bomb"],
                # Triple bombs third
                ["tripple bomb"],
                # Dark features fourth
                ["dark"],
                # Sediment last
                ["sediment"],
            ]

            def get_sort_key(roi_name):
                """Return (category_index, roi_name) for sorting"""
                roi_lower = roi_name.lower()
                for idx, keywords in enumerate(order_keywords):
                    if any(kw in roi_lower for kw in keywords):
                        return (idx, roi_name)
                return (len(order_keywords), roi_name)  # Unknown ROIs go last

            sorted_items = sorted(roi_dict.items(), key=lambda x: get_sort_key(x[0]))
            return dict(sorted_items)

        # ROI handling logic (unchanged)
        is_roi_analysis = False
        rois_to_plot = {}

        if roi_names is not None:
            if not hasattr(self, "roi_collection"):
                print("❌ No ROI collection found")
                return

            if roi_names == "all":
                rois_to_plot = self.roi_collection.copy()
            elif isinstance(roi_names, list):
                for name in roi_names:
                    if name in self.roi_collection:
                        rois_to_plot[name] = self.roi_collection[name]
            # Sort ROIs for consistent legend order
            rois_to_plot = sort_rois_by_category(rois_to_plot)
            is_roi_analysis = True

        elif roi_name is not None:
            if hasattr(self, "roi_collection") and roi_name in self.roi_collection:
                rois_to_plot[roi_name] = self.roi_collection[roi_name]
                is_roi_analysis = True
            else:
                print(f"❌ ROI '{roi_name}' not found")
                return

        elif roi_pixels is not None:
            rois_to_plot["ROI"] = roi_pixels
            is_roi_analysis = True

        elif track_index is not None and slit_index is not None:
            pass
        else:
            print("❌ Must specify pixel coordinates or ROI")
            return

        # Wavelength filtering
        wavelengths = self.wavelengths
        if wavelength_range is not None:
            wl_start, wl_end = wavelength_range
            wl_mask = (wavelengths >= wl_start) & (wavelengths <= wl_end)
            wavelengths = wavelengths[wl_mask]
            range_info = f" ({wl_start}-{wl_end}nm)"
        else:
            wl_mask = slice(None)
            range_info = ""

        # NEW: Wavelength interpolation setup
        # Print which wavelengths will be interpolated
        if interpolate_wavelengths is not None and len(interpolate_wavelengths) > 0:
            print(f"\n🔧 Interpolating {len(interpolate_wavelengths)} wavelength(s):")
            for wl_idx in interpolate_wavelengths:
                if 0 <= wl_idx < len(self.wavelengths):
                    wl_nm = self.wavelengths[wl_idx]
                    print(f"   • Index {wl_idx}: {wl_nm:.2f} nm")
                else:
                    print(
                        f"   ⚠️  Index {wl_idx} out of range (0-{len(self.wavelengths)-1})"
                    )
            print()

        def interpolate_bad_wavelengths(spectrum_2d, bad_indices):
            """
            Interpolate specific wavelengths using linear interpolation from neighbors.

            Parameters:
            -----------
            spectrum_2d : np.ndarray
                2D array of shape (n_pixels, n_wavelengths) or 1D array (n_wavelengths,)
            bad_indices : list of int
                Wavelength indices to interpolate

            Returns:
            --------
            np.ndarray : Spectrum with interpolated values
            """
            if bad_indices is None or len(bad_indices) == 0:
                return spectrum_2d

            spectrum_2d = np.array(spectrum_2d)
            is_1d = spectrum_2d.ndim == 1
            if is_1d:
                spectrum_2d = spectrum_2d[
                    np.newaxis, :
                ]  # Make 2D for uniform processing

            result = spectrum_2d.copy()

            for idx in bad_indices:
                if idx <= 0 or idx >= spectrum_2d.shape[1] - 1:
                    continue  # Can't interpolate edge wavelengths

                # Linear interpolation from neighbors
                result[:, idx] = (result[:, idx - 1] + result[:, idx + 1]) / 2.0

            return result[0] if is_1d else result

        def smooth_spectrum(spectrum, window_size):
            if window_size <= 1:
                return spectrum
            if window_size > len(spectrum):
                window_size = len(spectrum)
            # Use pandas rolling for proper alignment - same as illumination correction V2
            import pandas as pd

            series = pd.Series(spectrum)
            smoothed = series.rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
            return smoothed.values

        # Determine which normalization method to use
        # Handle backward compatibility: normalize=True maps to "mean"
        if normalize and normalize_method is None:
            active_normalization = "mean"
        elif normalize_method:
            active_normalization = normalize_method
        else:
            active_normalization = None

        # Normalization function with multiple methods
        def normalize_spectrum(spectrum, method=None):
            """
            Apply normalization to spectrum using specified method.

            Parameters:
            -----------
            spectrum : np.ndarray
                1D spectrum array
            method : str or None
                Normalization method to apply

            Returns:
            --------
            np.ndarray : Normalized spectrum
            """
            if method is None or len(spectrum) == 0:
                return spectrum

            spectrum = np.array(spectrum, dtype=float)

            if method == "mean":
                # Simple mean normalization (divide by mean)
                spectrum_mean = np.mean(spectrum)
                if spectrum_mean != 0:
                    return spectrum / spectrum_mean
                return spectrum

            elif method == "mean_center":
                # Mean centering (subtract mean) - removes DC offset
                return spectrum - np.mean(spectrum)

            elif method == "snv":
                # Standard Normal Variate - removes offset and scale
                spectrum_mean = np.mean(spectrum)
                spectrum_std = np.std(spectrum)
                if spectrum_std > 0:
                    return (spectrum - spectrum_mean) / spectrum_std
                return spectrum - spectrum_mean

            elif method == "minmax":
                # Min-max scaling to [0, 1]
                spectrum_min = np.min(spectrum)
                spectrum_max = np.max(spectrum)
                if spectrum_max > spectrum_min:
                    return (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
                return spectrum

            elif method == "l2":
                # L2 normalization (vector length = 1)
                norm = np.linalg.norm(spectrum)
                if norm > 0:
                    return spectrum / norm
                return spectrum

            else:
                print(f"⚠️  Unknown normalization method '{method}', using raw spectrum")
                return spectrum

        # MSC-specific normalization function (requires reference spectrum)
        def normalize_spectrum_msc(spectrum, reference):
            """
            Multiplicative Scatter Correction.
            Models each spectrum as: spectrum = a + b * reference
            Then corrects as: corrected = (spectrum - a) / b
            """
            if reference is None or len(spectrum) != len(reference):
                return spectrum

            # Fit linear model: spectrum = a + b * reference
            coeffs = np.polyfit(reference, spectrum, 1)
            b, a = coeffs[0], coeffs[1]  # slope and intercept

            if abs(b) > 1e-10:  # Avoid division by zero
                return (spectrum - a) / b
            return spectrum

        # MSC requires reference spectrum, compute it once if needed
        reference_spectrum_msc = None
        if active_normalization == "msc" and is_roi_analysis:
            # Compute mean spectrum across all ROIs as reference
            print("📊 Computing reference spectrum for MSC normalization...")
            all_spectra_for_ref = []
            for roi_name_key, roi_pixels_list in rois_to_plot.items():
                for slit_idx, track_idx in roi_pixels_list:
                    track_offset = getattr(self, "track_offset", 0)
                    rel_track = track_idx - track_offset
                    if 0 <= rel_track < cube.shape[0] and 0 <= slit_idx < cube.shape[1]:
                        spectrum = cube[rel_track, slit_idx, :]
                        # Apply interpolation if specified
                        if interpolate_wavelengths:
                            spectrum = interpolate_bad_wavelengths(
                                spectrum[np.newaxis, :], interpolate_wavelengths
                            )[0]
                        # Apply wavelength filtering
                        spectrum = spectrum[wl_mask]
                        all_spectra_for_ref.append(spectrum)

            if all_spectra_for_ref:
                reference_spectrum_msc = np.mean(all_spectra_for_ref, axis=0)
                print(
                    f"   ✓ Reference computed from {len(all_spectra_for_ref)} spectra"
                )

        # Dynamic y-label based on normalization
        if ylabel == "Intensity":  # Only change default label
            if active_normalization:
                label_map = {
                    "mean": "Mean-Normalized Intensity",
                    "mean_center": "Mean-Centered Intensity",
                    "snv": "SNV-Normalized Intensity",
                    "msc": "MSC-Corrected Intensity",
                    "minmax": "Min-Max Scaled Intensity",
                    "l2": "L2-Normalized Intensity",
                }
                ylabel = label_map.get(active_normalization, "Normalized Intensity")
            else:
                ylabel = "Intensity"

        plt.figure(figsize=figsize)

        if is_roi_analysis:
            for i, (roi_name_key, roi_pixels_list) in enumerate(rois_to_plot.items()):
                valid_spectra = []

                for slit_idx, track_idx in roi_pixels_list:
                    # Use track_offset if available, otherwise assume 0 for combined cubes
                    track_offset = getattr(self, "track_offset", 0)
                    rel_track = track_idx - track_offset
                    if 0 <= rel_track < cube.shape[0] and 0 <= slit_idx < cube.shape[1]:
                        # Get FULL spectrum first (before wavelength filtering)
                        spectrum = cube[rel_track, slit_idx, :]
                        valid_spectra.append(spectrum)

                if valid_spectra:
                    spectra_array = np.array(valid_spectra)

                    # NEW: Apply interpolation to remove bad wavelengths (BEFORE wavelength filtering!)
                    spectra_array = interpolate_bad_wavelengths(
                        spectra_array, interpolate_wavelengths
                    )

                    # NOW apply wavelength filtering
                    spectra_array = spectra_array[:, wl_mask]
                    avg_spectrum = np.mean(spectra_array, axis=0)
                    std_spectrum = np.std(spectra_array, axis=0)

                    # Apply smoothing first
                    if wavelength_smoothing > 1:
                        smoothed_avg = smooth_spectrum(
                            avg_spectrum, wavelength_smoothing
                        )
                        smoothed_std = smooth_spectrum(
                            std_spectrum, wavelength_smoothing
                        )
                        # pandas rolling with center=True keeps the same length, no trimming needed
                        plot_wavelengths = wavelengths
                        plot_avg = smoothed_avg
                        plot_std = smoothed_std
                    else:
                        plot_wavelengths = wavelengths
                        plot_avg = avg_spectrum
                        plot_std = std_spectrum

                    # Apply normalization
                    if (
                        active_normalization == "msc"
                        and reference_spectrum_msc is not None
                    ):
                        plot_avg = normalize_spectrum_msc(
                            plot_avg, reference_spectrum_msc
                        )
                        if show_std:
                            plot_std = normalize_spectrum_msc(
                                plot_std, reference_spectrum_msc
                            )
                    else:
                        plot_avg = normalize_spectrum(plot_avg, active_normalization)
                        if show_std:
                            plot_std = normalize_spectrum(
                                plot_std, active_normalization
                            )

                    # NEW: Get consistent color across all plots
                    color = self._get_roi_color(roi_name_key, colors, color_map)

                    # Always plot the main line (with label for legend)
                    plt.plot(
                        plot_wavelengths,
                        plot_avg,
                        color=color,
                        linewidth=2,
                        label=roi_name_key,
                    )

                    # NEW: Only show std bands if show_std=True
                    if show_std:
                        plt.fill_between(
                            plot_wavelengths,
                            plot_avg - plot_std,
                            plot_avg + plot_std,
                            color=color,
                            alpha=0.15,
                        )

                    if use_inline_labels:
                        label_x = plot_wavelengths[-1] * 0.95
                        label_y = plot_avg[-10:].mean()

                        plt.text(
                            label_x,
                            label_y,
                            roi_name_key,
                            color=color,
                            fontweight="bold",
                            fontsize=10,
                            ha="right",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                alpha=0.8,
                                edgecolor=color,
                            ),
                        )

            title = f"{data_label} ROI Spectra{range_info}"
            if active_normalization:
                title += f" ({active_normalization.upper()})"
            if wavelength_smoothing > 1:
                title += f" (λ-smooth: {wavelength_smoothing})"
            title += f"\n({self.name})"

        else:
            # Single pixel mode
            track_offset = getattr(self, "track_offset", 0)
            rel_track = track_index - track_offset
            # Get FULL spectrum first (before wavelength filtering)
            spectrum = cube[rel_track, slit_index, :]

            # NEW: Apply interpolation to remove bad wavelengths (BEFORE wavelength filtering!)
            spectrum = interpolate_bad_wavelengths(spectrum, interpolate_wavelengths)

            # NOW apply wavelength filtering
            spectrum = spectrum[wl_mask]

            if wavelength_smoothing > 1:
                smoothed_spectrum = smooth_spectrum(spectrum, wavelength_smoothing)
                # pandas rolling with center=True keeps the same length, no trimming needed
                plot_wavelengths = wavelengths
                plot_spectrum = smoothed_spectrum
            else:
                plot_wavelengths = wavelengths
                plot_spectrum = spectrum

            # Apply normalization (MSC not supported in single-pixel mode)
            if active_normalization == "msc":
                print(
                    "⚠️  MSC normalization not available in single-pixel mode, using SNV instead"
                )
                plot_spectrum = normalize_spectrum(plot_spectrum, "snv")
            else:
                plot_spectrum = normalize_spectrum(plot_spectrum, active_normalization)

            plt.plot(plot_wavelengths, plot_spectrum, color=colors[0], linewidth=2)

            title = f"{data_label} Spectrum"
            if active_normalization:
                title += f" ({active_normalization.upper()})"
            if wavelength_smoothing > 1:
                title += f" (λ-smooth: {wavelength_smoothing})"

        plt.xlabel("Wavelength (nm)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Force exact wavelength range with no padding
        if wavelength_range is not None:
            ax = plt.gca()
            ax.set_xlim(wavelength_range[0], wavelength_range[1])
            ax.margins(x=0)
            ax.autoscale(enable=False, axis="x")

        # NEW: Handle legend placement (including "outside" option)
        if not use_inline_labels or not is_roi_analysis:
            if legend_loc is not None:
                ax = plt.gca()
                if legend_loc == "outside":
                    # Place legend outside the plot area on the right
                    ax.legend(
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=9,
                        framealpha=0.9,
                    )
                else:
                    ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9)

        plt.tight_layout()
        plt.show()


# ------------------------- convenience -------------------------


def load_transect(
    folder_path: Optional[str] = None, use_corrected: bool = False
) -> TransectDataSet:
    """
    Creates a TransectDataSet scanning the folder for H5s with georef outputs.
    If folder_path is None, uses config.OUTPUT_FOLDER when available.
    """
    if folder_path is None and config is not None and hasattr(config, "OUTPUT_FOLDER"):
        folder_path = config.OUTPUT_FOLDER
    if folder_path is None:
        raise ValueError("Provide folder_path or set config.OUTPUT_FOLDER")
    return TransectDataSet(folder_path, use_corrected=use_corrected)


def print_processing_statistics(stats_file_path=None, track_start=None, track_end=None):
    """
    Load and print processing statistics from JSON file.

    Parameters:
    -----------
    stats_file_path : str or Path, optional
        Path to the statistics JSON file. If None, uses config.OUTPUT_FOLDER/processing_statistics.json
    track_start : int, optional
        Starting track index for segment-specific statistics (inclusive)
    track_end : int, optional
        Ending track index for segment-specific statistics (exclusive)
    """
    import json
    from pathlib import Path
    from datetime import datetime

    # Determine stats file path
    if stats_file_path is None:
        stats_file = Path(config.OUTPUT_FOLDER) / "processing_statistics.json"
    else:
        stats_file = Path(stats_file_path)

    if not stats_file.exists():
        print(f"❌ Statistics file not found: {stats_file}")
        print("Run main.py first to generate the statistics file.")
        return None

    # Load statistics
    with open(stats_file, "r") as f:
        stats = json.load(f)

    # Helper function to safely get nested dict values
    def get_nested(data, *keys, default="N/A"):
        """Safely get nested dictionary values."""
        try:
            for key in keys:
                data = data[key]
            return data
        except (KeyError, TypeError):
            return default

    # Note: unix_to_utc is now defined at module level (line ~53)

    print("=" * 80)
    print("PROCESSING STATISTICS")
    print("=" * 80)

    # Session info
    print("\n📅 SESSION INFO")
    print(f"  Started: {get_nested(stats, 'processing_session', 'start_time')}")
    print(f"  Finished: {get_nested(stats, 'processing_session', 'end_time')}")
    duration_sec = get_nested(stats, "processing_session", "duration_sec", default=0)
    duration_human = get_nested(
        stats, "processing_session", "duration_human", default="N/A"
    )
    if duration_sec != "N/A" and duration_sec != 0:
        print(f"  Processing duration: {duration_human}")

    # Configuration
    print("\n⚙️  CONFIGURATION")
    print(
        f"  Navigation CSV: {Path(get_nested(stats, 'configuration', 'nav_csv', default='')).name}"
    )
    print(
        f"  MBES GeoTIFF: {Path(get_nested(stats, 'configuration', 'mbes_geotiff', default='')).name}"
    )
    print(
        f"  Output folder: {Path(get_nested(stats, 'configuration', 'output_folder', default='')).name}"
    )
    print(
        f"  Max ray length: {get_nested(stats, 'configuration', 'max_ray_length_m', default=0):.1f} m"
    )
    print(
        f"  Failure threshold: {get_nested(stats, 'configuration', 'early_failure_threshold_pct', default=0):.1f}%"
    )

    # Navigation data
    print("\n🧭 NAVIGATION DATA")
    nav_records = get_nested(stats, "navigation_data", "total_records", default=0)
    print(f"  Total records: {nav_records:,} (navigation data points in CSV)")
    time_start = get_nested(stats, "navigation_data", "time_range", "start", default=0)
    time_end = get_nested(stats, "navigation_data", "time_range", "end", default=0)
    time_dur = get_nested(
        stats, "navigation_data", "time_range", "duration_sec", default=0
    )
    print(f"  Start time: {time_start:.2f} ({unix_to_utc(time_start)})")
    print(f"  End time: {time_end:.2f} ({unix_to_utc(time_end)})")
    print(f"  Duration: {time_dur:.1f} s ({time_dur/60:.1f} min)")

    # MBES mesh
    print("\n🗺️  MBES MESH")
    print(f"  Vertices: {get_nested(stats, 'mbes_mesh', 'n_vertices', default=0):,}")
    print(f"  Faces: {get_nested(stats, 'mbes_mesh', 'n_faces', default=0):,}")
    x_min = get_nested(stats, "mbes_mesh", "bounds", "x_min", default=0)
    x_max = get_nested(stats, "mbes_mesh", "bounds", "x_max", default=0)
    y_min = get_nested(stats, "mbes_mesh", "bounds", "y_min", default=0)
    y_max = get_nested(stats, "mbes_mesh", "bounds", "y_max", default=0)
    z_min = get_nested(stats, "mbes_mesh", "bounds", "z_min", default=0)
    z_max = get_nested(stats, "mbes_mesh", "bounds", "z_max", default=0)
    print(f"  Bounds (X): [{x_min:.2f}, {x_max:.2f}] m")
    print(f"  Bounds (Y): [{y_min:.2f}, {y_max:.2f}] m")
    print(f"  Bounds (Z): [{z_min:.2f}, {z_max:.2f}] m")

    # Camera
    print("\n📷 CAMERA CALIBRATION")
    focal = get_nested(stats, "camera_calibration", "focal_length", default=0)
    cx = get_nested(stats, "camera_calibration", "principal_point", default=0)
    width = get_nested(stats, "camera_calibration", "width_pixels", default=0)
    print(f"  Focal length: {focal:.2f} px")
    print(f"  Principal point: {cx:.2f} px")
    print(f"  Image width: {width:.0f} px")

    # Combined transect statistics (MAIN SECTION)
    if "aggregated_statistics" in stats:
        print("\n" + "=" * 80)
        print("📊 COMBINED TRANSECT STATISTICS (All H5 files together)")
        print("=" * 80)
        agg = stats["aggregated_statistics"]

        # File processing summary
        print(f"\n  Files processed: {get_nested(agg, 'total_files', default=0)} total")
        print(f"    ✅ Successful: {get_nested(agg, 'successful_files', default=0)}")
        print(f"    ❌ Failed: {get_nested(agg, 'failed_files', default=0)}")

        # HSI and ray tracing
        print(f"\n  HSI Data:")
        print(f"    Total frames: {get_nested(agg, 'total_frames', default=0):,}")
        print(f"    Total rays: {get_nested(agg, 'total_rays', default=0):,}")
        print(f"    Successful hits: {get_nested(agg, 'total_hits', default=0):,}")
        print(
            f"    Overall success rate: {get_nested(agg, 'overall_success_rate_pct', default=0):.2f}%"
        )

        # Mission metrics (combined)
        if "mission_metrics" in agg:
            print(f"\n  📏 Mission Metrics (entire transect):")
            dist_m = get_nested(agg, "mission_metrics", "total_distance_m", default=0)
            dur_sec = get_nested(
                agg, "mission_metrics", "total_duration_sec", default=0
            )
            speed_ms = get_nested(agg, "mission_metrics", "avg_speed_ms", default=0)
            speed_kmh = get_nested(agg, "mission_metrics", "avg_speed_kmh", default=0)
            print(f"    Total distance: {dist_m:.1f} m ({dist_m/1000:.3f} km)")
            print(f"    Total survey duration: {dur_sec:.1f} s ({dur_sec/60:.1f} min)")
            print(f"    Average speed: {speed_ms:.2f} m/s ({speed_kmh:.2f} km/h)")

        # Coverage metrics (combined)
        if "coverage_metrics" in agg:
            print(f"\n  📐 Coverage Metrics (entire transect):")
            area_m2 = get_nested(
                agg, "coverage_metrics", "total_coverage_area_m2", default=0
            )
            mean_swath = get_nested(
                agg, "coverage_metrics", "mean_swath_width_m", default=0
            )
            min_swath = get_nested(
                agg, "coverage_metrics", "min_swath_width_m", default=0
            )
            max_swath = get_nested(
                agg, "coverage_metrics", "max_swath_width_m", default=0
            )
            eff_pct = get_nested(
                agg, "coverage_metrics", "mean_coverage_efficiency_pct", default=0
            )
            print(
                f"    Total covered area: {area_m2:.1f} m² ({area_m2/10000:.4f} hectares)"
            )
            print(f"    Mean swath width: {mean_swath:.2f} m")
            print(f"    Swath width range: [{min_swath:.2f}, {max_swath:.2f}] m")
            print(f"    Coverage efficiency: {eff_pct:.1f}%")
            print(f"      (= covered area / survey bounding box area)")

    # Segment-specific statistics (if track range provided)
    if track_start is not None and track_end is not None:
        print("\n" + "=" * 80)
        print(f"🎯 SEGMENT STATISTICS (tracks {track_start} to {track_end})")
        print("=" * 80)

        # Validate track range
        total_frames_agg = get_nested(
            stats, "aggregated_statistics", "total_frames", default=0
        )
        if track_start < 0 or track_end > total_frames_agg or track_start >= track_end:
            print(
                f"  ⚠️  Invalid track range: [{track_start}, {track_end}) for total {total_frames_agg} tracks"
            )
        else:
            segment_tracks = track_end - track_start
            print(f"\n  📏 Segment Info:")
            print(
                f"    Track range: [{track_start}, {track_end}) (length: {segment_tracks} tracks)"
            )
            print(
                f"    Percentage of transect: {100 * segment_tracks / max(1, total_frames_agg):.1f}%"
            )

            # Calculate segment statistics from individual files
            segment_hits = 0
            segment_rays = 0
            segment_files = []
            segment_start_time = None
            segment_end_time = None

            if "individual_files" in stats:
                cumulative_track = 0
                for file_stat in stats["individual_files"]:
                    if file_stat.get("status") == "failed":
                        continue

                    file_frames = get_nested(
                        file_stat, "hsi_data", "n_frames", default=0
                    )
                    file_slits = get_nested(file_stat, "hsi_data", "n_slits", default=0)
                    file_end_track = cumulative_track + file_frames

                    # Check if this file overlaps with segment
                    if file_end_track > track_start and cumulative_track < track_end:
                        segment_files.append(file_stat.get("filename", "Unknown"))

                        # Calculate overlap
                        overlap_start = max(track_start, cumulative_track)
                        overlap_end = min(track_end, file_end_track)
                        overlap_tracks = overlap_end - overlap_start

                        # Get file-level statistics
                        file_hits = get_nested(
                            file_stat, "ray_tracing", "successful_hits", default=0
                        )
                        file_rays = get_nested(
                            file_stat, "ray_tracing", "total_rays", default=0
                        )

                        # Estimate segment contribution (proportional to overlap)
                        if file_frames > 0:
                            overlap_ratio = overlap_tracks / file_frames
                            segment_hits += int(file_hits * overlap_ratio)
                            segment_rays += int(file_rays * overlap_ratio)

                        # Time range
                        if (
                            "hsi_data" in file_stat
                            and "time_range" in file_stat["hsi_data"]
                        ):
                            file_time_start = get_nested(
                                file_stat, "hsi_data", "time_range", "start", default=0
                            )
                            file_time_end = get_nested(
                                file_stat, "hsi_data", "time_range", "end", default=0
                            )
                            file_duration = get_nested(
                                file_stat,
                                "hsi_data",
                                "time_range",
                                "duration_sec",
                                default=0,
                            )

                            if file_duration > 0 and file_frames > 0:
                                # Estimate time for segment within this file
                                time_per_track = file_duration / file_frames
                                local_start_track = overlap_start - cumulative_track
                                local_end_track = overlap_end - cumulative_track

                                est_start_time = (
                                    file_time_start + local_start_track * time_per_track
                                )
                                est_end_time = (
                                    file_time_start + local_end_track * time_per_track
                                )

                                if (
                                    segment_start_time is None
                                    or est_start_time < segment_start_time
                                ):
                                    segment_start_time = est_start_time
                                if (
                                    segment_end_time is None
                                    or est_end_time > segment_end_time
                                ):
                                    segment_end_time = est_end_time

                    cumulative_track = file_end_track

            # Now extract detailed segment data from individual file statistics
            print(f"\n  💡 Extracting detailed segment metrics from file statistics...")

            # Collect detailed segment data from overlapping files
            segment_file_stats = []
            segment_nav_depth = []
            segment_nav_roll = []
            segment_nav_pitch = []
            segment_mission_distances = []
            segment_mission_durations = []
            segment_coverage_areas = []
            segment_coverage_swaths = []

            if "individual_files" in stats:
                cumulative_track = 0

                for file_stat in stats["individual_files"]:
                    if file_stat.get("status") == "failed":
                        continue

                    file_frames = get_nested(
                        file_stat, "hsi_data", "n_frames", default=0
                    )
                    file_end_track = cumulative_track + file_frames

                    # Check if this file overlaps with segment
                    if file_end_track > track_start and cumulative_track < track_end:
                        filename = file_stat.get("filename", "Unknown")

                        # Calculate overlap
                        overlap_start = max(track_start, cumulative_track)
                        overlap_end = min(track_end, file_end_track)
                        overlap_tracks = overlap_end - overlap_start
                        overlap_ratio = (
                            overlap_tracks / file_frames if file_frames > 0 else 0
                        )

                        # Store file info
                        segment_file_stats.append(
                            {
                                "filename": filename,
                                "overlap_tracks": overlap_tracks,
                                "overlap_ratio": overlap_ratio,
                                "file_stat": file_stat,
                            }
                        )

                        # Extract navigation data from file statistics
                        if "navigation" in file_stat:
                            nav = file_stat["navigation"]

                            # Depth
                            depth_min = get_nested(
                                nav, "depth_range", "min", default=None
                            )
                            depth_max = get_nested(
                                nav, "depth_range", "max", default=None
                            )
                            depth_mean = get_nested(
                                nav, "depth_range", "mean", default=None
                            )
                            if depth_min is not None:
                                segment_nav_depth.append(
                                    {
                                        "min": depth_min,
                                        "max": depth_max,
                                        "mean": depth_mean,
                                        "weight": overlap_tracks,
                                    }
                                )

                            # Roll
                            roll_min = get_nested(
                                nav, "attitude_ranges", "roll", "min", default=None
                            )
                            roll_max = get_nested(
                                nav, "attitude_ranges", "roll", "max", default=None
                            )
                            roll_std = get_nested(
                                nav, "attitude_ranges", "roll", "std", default=None
                            )
                            if roll_min is not None:
                                segment_nav_roll.append(
                                    {
                                        "min": roll_min,
                                        "max": roll_max,
                                        "std": roll_std,
                                        "weight": overlap_tracks,
                                    }
                                )

                            # Pitch
                            pitch_min = get_nested(
                                nav, "attitude_ranges", "pitch", "min", default=None
                            )
                            pitch_max = get_nested(
                                nav, "attitude_ranges", "pitch", "max", default=None
                            )
                            pitch_std = get_nested(
                                nav, "attitude_ranges", "pitch", "std", default=None
                            )
                            if pitch_min is not None:
                                segment_nav_pitch.append(
                                    {
                                        "min": pitch_min,
                                        "max": pitch_max,
                                        "std": pitch_std,
                                        "weight": overlap_tracks,
                                    }
                                )

                        # Extract mission metrics
                        if "mission_metrics" in file_stat:
                            mission = file_stat["mission_metrics"]
                            dist = get_nested(mission, "total_distance_m", default=0)
                            speed_ms = get_nested(mission, "avg_speed_ms", default=0)

                            # Proportional distance for this segment
                            segment_distance = dist * overlap_ratio
                            segment_mission_distances.append(segment_distance)
                            segment_mission_durations.append(
                                segment_distance / speed_ms if speed_ms > 0 else 0
                            )

                        # Extract coverage metrics
                        if "coverage_metrics" in file_stat:
                            coverage = file_stat["coverage_metrics"]
                            area = get_nested(
                                coverage, "coverage_area", "total_covered_m2", default=0
                            )
                            swath_mean = get_nested(
                                coverage, "swath_width", "mean_m", default=0
                            )
                            swath_std = get_nested(
                                coverage, "swath_width", "std_m", default=0
                            )

                            # Proportional coverage for this segment
                            segment_coverage_areas.append(area * overlap_ratio)
                            if swath_mean > 0:
                                segment_coverage_swaths.append(
                                    {
                                        "mean": swath_mean,
                                        "std": swath_std,
                                        "weight": overlap_tracks,
                                    }
                                )

                    cumulative_track = file_end_track

            # Print segment statistics
            if segment_rays > 0:
                segment_success_rate = 100 * segment_hits / segment_rays
                print(f"\n  🎯 Ray Tracing (segment):")
                print(f"    Total rays: {segment_rays:,}")
                print(f"    Successful hits: {segment_hits:,}")
                print(f"    Success rate: {segment_success_rate:.2f}%")

            if segment_start_time is not None and segment_end_time is not None:
                segment_duration = segment_end_time - segment_start_time
                print(f"\n  ⏱️  Time Range (segment):")
                print(f"    Start: {unix_to_utc(segment_start_time)}")
                print(f"    End: {unix_to_utc(segment_end_time)}")
                print(
                    f"    Duration: {segment_duration:.1f} s ({segment_duration/60:.1f} min)"
                )

            # Navigation statistics from file stats
            if segment_nav_depth or segment_nav_roll or segment_nav_pitch:
                print(f"\n  🧭 Navigation (segment):")

                # Depth statistics
                if segment_nav_depth:
                    depth_min = min(d["min"] for d in segment_nav_depth)
                    depth_max = max(d["max"] for d in segment_nav_depth)
                    # Weighted mean
                    total_weight = sum(d["weight"] for d in segment_nav_depth)
                    depth_mean = (
                        sum(d["mean"] * d["weight"] for d in segment_nav_depth)
                        / total_weight
                        if total_weight > 0
                        else 0
                    )
                    print(
                        f"    Depth: {depth_min:.2f} to {depth_max:.2f} m (mean: {depth_mean:.2f} m)"
                    )

                # Roll statistics
                if segment_nav_roll:
                    roll_min = min(d["min"] for d in segment_nav_roll)
                    roll_max = max(d["max"] for d in segment_nav_roll)
                    # Weighted average of std devs
                    total_weight = sum(d["weight"] for d in segment_nav_roll)
                    roll_std_avg = (
                        sum(d["std"] * d["weight"] for d in segment_nav_roll)
                        / total_weight
                        if total_weight > 0
                        else 0
                    )
                    print(
                        f"    Roll: {roll_min:.2f}° to {roll_max:.2f}° (σ={roll_std_avg:.2f}°)"
                    )

                # Pitch statistics
                if segment_nav_pitch:
                    pitch_min = min(d["min"] for d in segment_nav_pitch)
                    pitch_max = max(d["max"] for d in segment_nav_pitch)
                    total_weight = sum(d["weight"] for d in segment_nav_pitch)
                    pitch_std_avg = (
                        sum(d["std"] * d["weight"] for d in segment_nav_pitch)
                        / total_weight
                        if total_weight > 0
                        else 0
                    )
                    print(
                        f"    Pitch: {pitch_min:.2f}° to {pitch_max:.2f}° (σ={pitch_std_avg:.2f}°)"
                    )

            # Mission metrics from file stats
            if segment_mission_distances:
                total_distance = sum(segment_mission_distances)
                print(f"\n  📏 Mission Metrics (segment):")
                print(f"    Distance: {total_distance:.1f} m")

                if segment_duration > 0:
                    avg_speed_ms = total_distance / segment_duration
                    avg_speed_kmh = avg_speed_ms * 3.6
                    print(
                        f"    Speed: {avg_speed_ms:.2f} m/s ({avg_speed_kmh:.2f} km/h)"
                    )

            # Coverage metrics from file stats
            if segment_coverage_areas or segment_coverage_swaths:
                print(f"\n  📐 Coverage (segment):")

                if segment_coverage_areas:
                    total_area = sum(segment_coverage_areas)
                    print(f"    Area: {total_area:.1f} m²")

                if segment_coverage_swaths:
                    # Weighted average swath
                    total_weight = sum(d["weight"] for d in segment_coverage_swaths)
                    swath_mean = (
                        sum(d["mean"] * d["weight"] for d in segment_coverage_swaths)
                        / total_weight
                        if total_weight > 0
                        else 0
                    )
                    swath_std = (
                        sum(d["std"] * d["weight"] for d in segment_coverage_swaths)
                        / total_weight
                        if total_weight > 0
                        else 0
                    )
                    print(f"    Swath width: {swath_mean:.2f} ± {swath_std:.2f} m")

            # List segment files with details
            if segment_file_stats:
                print(f"\n  📄 Files in Segment:")
                for file_info in segment_file_stats:
                    fname = file_info["filename"]
                    overlap = file_info["overlap_tracks"]
                    ratio = file_info["overlap_ratio"] * 100
                    print(f"    • {fname}: {overlap} tracks ({ratio:.1f}% of file)")

            # HSI data summary for segment
            if segment_file_stats:
                # Get slits/bands from first file
                first_file = segment_file_stats[0]["file_stat"]
                segment_slits = get_nested(first_file, "hsi_data", "n_slits", default=0)
                segment_bands = get_nested(first_file, "hsi_data", "n_bands", default=0)
                if segment_slits > 0 and segment_bands > 0:
                    print(f"\n  📸 HSI Data (segment):")
                    print(f"    Frames: {segment_tracks}")
                    print(f"    Slits per frame: {segment_slits}")
                    print(f"    Bands: {segment_bands}")
                    print(f"    Total pixels: {segment_tracks * segment_slits:,}")

    # Individual files
    if "individual_files" in stats and len(stats["individual_files"]) > 0:
        print("\n" + "=" * 80)
        print("📄 INDIVIDUAL FILE STATISTICS")
        print("=" * 80)

        for i, file_stat in enumerate(stats["individual_files"], 1):
            if file_stat.get("status") == "failed":
                print(
                    f"\n[{i}] {get_nested(file_stat, 'filename', default='Unknown')} ❌ FAILED"
                )
                continue

            print(f"\n[{i}] {get_nested(file_stat, 'filename', default='Unknown')}")
            print(
                f"  ⏱️  Processing time: {get_nested(file_stat, 'processing_time', default='N/A')}"
            )

            # HSI data
            if "hsi_data" in file_stat:
                hsi = file_stat["hsi_data"]
                n_frames = get_nested(hsi, "n_frames", default=0)
                n_slits = get_nested(hsi, "n_slits", default=0)
                n_bands = get_nested(hsi, "n_bands", default=0)
                duration = get_nested(hsi, "time_range", "duration_sec", default=0)
                time_start_file = get_nested(hsi, "time_range", "start", default=0)
                print(
                    f"  📸 HSI: {n_frames} frames × {n_slits} slits × {n_bands} bands"
                )
                print(f"      Duration: {duration:.1f} s")
                print(f"      Start: {unix_to_utc(time_start_file)}")

            # Navigation
            if "navigation" in file_stat:
                nav = file_stat["navigation"]
                print(f"  🧭 Navigation:")
                depth_min = get_nested(nav, "depth_range", "min", default=0)
                depth_max = get_nested(nav, "depth_range", "max", default=0)
                depth_mean = get_nested(nav, "depth_range", "mean", default=0)
                print(
                    f"      Depth: {depth_min:.2f} to {depth_max:.2f} m (mean: {depth_mean:.2f} m)"
                )

                roll_min = get_nested(nav, "attitude_ranges", "roll", "min", default=0)
                roll_max = get_nested(nav, "attitude_ranges", "roll", "max", default=0)
                roll_std = get_nested(nav, "attitude_ranges", "roll", "std", default=0)
                pitch_min = get_nested(
                    nav, "attitude_ranges", "pitch", "min", default=0
                )
                pitch_max = get_nested(
                    nav, "attitude_ranges", "pitch", "max", default=0
                )
                pitch_std = get_nested(
                    nav, "attitude_ranges", "pitch", "std", default=0
                )
                print(
                    f"      Roll: {roll_min:.2f}° to {roll_max:.2f}° (σ={roll_std:.2f}°)"
                )
                print(
                    f"      Pitch: {pitch_min:.2f}° to {pitch_max:.2f}° (σ={pitch_std:.2f}°)"
                )

            # Mission metrics
            if "mission_metrics" in file_stat:
                mission = file_stat["mission_metrics"]
                dist = get_nested(mission, "total_distance_m", default=0)
                speed_ms = get_nested(mission, "avg_speed_ms", default=0)
                speed_kmh = get_nested(mission, "avg_speed_kmh", default=0)
                print(f"  📏 Mission:")
                print(f"      Distance: {dist:.1f} m")
                print(f"      Speed: {speed_ms:.2f} m/s ({speed_kmh:.2f} km/h)")

            # Coverage
            if "coverage_metrics" in file_stat:
                coverage = file_stat["coverage_metrics"]
                area = get_nested(
                    coverage, "coverage_area", "total_covered_m2", default=0
                )
                swath_mean = get_nested(coverage, "swath_width", "mean_m", default=0)
                swath_std = get_nested(coverage, "swath_width", "std_m", default=0)
                eff = get_nested(coverage, "coverage_area", "efficiency_pct", default=0)
                print(f"  📐 Coverage:")
                print(f"      Area: {area:.1f} m²")
                print(f"      Swath width: {swath_mean:.2f} ± {swath_std:.2f} m")
                print(f"      Efficiency: {eff:.1f}%")

            # Ray tracing
            if "ray_tracing" in file_stat:
                rt = file_stat["ray_tracing"]
                success_pct = get_nested(rt, "success_rate_pct", default=0)
                hits = get_nested(rt, "successful_hits", default=0)
                total = get_nested(rt, "total_rays", default=1)
                trimesh_hits = get_nested(rt, "trimesh_hits", default=0)
                pv_retries = get_nested(rt, "pyvista_retries", default=0)
                pv_recoveries = get_nested(rt, "pyvista_recoveries", default=0)
                print(f"  🎯 Ray tracing:")
                print(f"      Success rate: {success_pct:.2f}% ({hits:,}/{total:,})")
                print(f"      Trimesh hits: {trimesh_hits:,}")
                print(
                    f"      PyVista retries: {pv_retries:,} (recovered: {pv_recoveries:,})"
                )

    print("\n" + "=" * 80)
    print(f"Statistics loaded from: {stats_file}")
    print("=" * 80)

    return stats
