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
    from utils.common import config
except Exception:
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
            if gf.shape is not None:
                # Load files with radiance data, even if georef is missing
                self.files[gf.name] = gf
                if not gf.has_georef:
                    print(
                        f"⚠️  {fn}: radiance found but no georef dataset – loaded anyway"
                    )
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
        self,
        names: List[str],
        normalize_per_file: bool = False,
        load_datacube: str = None,
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
        load_datacube : str, optional
            Name of saved datacube to load instead of default radiance data.
            Examples: "dataCube_pseudo_reflectance", "dataCube_normalized_pseudo_reflectance"
            If None (default), loads standard radiance data (raw or corrected based on use_corrected).
        """
        missing = [n for n in names if n not in self.files]
        if missing:
            print(f"⚠️  Missing files: {missing}")
        chosen = [self.files[n] for n in names if n in self.files]
        if not chosen:
            raise ValueError("No valid files selected")
        return CombinedTransectCube(
            chosen, self.folder, self.use_corrected, normalize_per_file, load_datacube
        )

    def select_all_files(
        self, normalize_per_file: bool = False, load_datacube: str = None
    ) -> "CombinedTransectCube":
        return self.select_files(
            list(self.files.keys()),
            normalize_per_file=normalize_per_file,
            load_datacube=load_datacube,
        )

    def select_files_by_pattern(
        self, pattern: str, normalize_per_file: bool = False, load_datacube: str = None
    ) -> "CombinedTransectCube":
        import fnmatch

        names = [n for n in self.files if fnmatch.fnmatch(n, pattern)]
        if not names:
            raise ValueError(f"No files match pattern: {pattern}")
        return self.select_files(
            names, normalize_per_file=normalize_per_file, load_datacube=load_datacube
        )


class CombinedTransectCube:
    """Combines multiple GeoFiles into a single logical cube along track."""

    def __init__(
        self,
        geofiles: List[GeoFile],
        folder: str,
        use_corrected: bool,
        normalize_per_file: bool = False,
        load_datacube: str = None,
    ):
        self.geofiles = geofiles
        self.folder = folder
        self.use_corrected = use_corrected
        self.normalize_per_file = (
            normalize_per_file  # NEW: normalize each file's RGB independently
        )
        self.load_datacube = load_datacube  # NEW: custom datacube to load
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

        # Check if all files have georef data
        all_have_georef = all(gf.has_georef for gf in self.geofiles)
        some_have_georef = any(gf.has_georef for gf in self.geofiles)

        # Error if mixing files with and without georef
        if some_have_georef and not all_have_georef:
            raise ValueError(
                "Cannot mix files with and without georef data. "
                "All selected files must be consistent."
            )

        # Set flag to control which methods are available
        self.has_georef = all_have_georef

        # Build spatial grids ONLY if georef exists
        if self.has_georef:
            self._build_combined()
        else:
            print("⚠️  Files have no georef data - skipping spatial grid building")
            print(
                "✅ Spectral analysis functions available (plot_spectrum, illumination correction, etc.)"
            )

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

        # Determine which dataset to load
        if self.load_datacube:
            # Load custom saved datacube (e.g., "dataCube_pseudo_reflectance")
            dset_name = f"processed/radiance/{self.load_datacube}"
            print(f"   📦 Loading custom datacube: {self.load_datacube}")
        else:
            # Load standard radiance data
            dset_name = (
                "processed/radiance/dataCube_corrected"
                if self.use_corrected
                else "processed/radiance/dataCube"
            )

        for gf in self.geofiles:
            print(f"   • Loading {gf.name}...")
            try:
                with h5py.File(gf.path, "r") as f:
                    if dset_name in f:
                        cube = f[dset_name][()]
                    elif not self.load_datacube and "processed/radiance/dataCube" in f:
                        # Fallback to raw if corrected not found (only for default loading)
                        cube = f["processed/radiance/dataCube"][()]
                    else:
                        raise KeyError(f"Dataset '{dset_name}' not found in {gf.name}")

                    cubes.append(cube)
            except Exception as e:
                print(f"      ⚠️  Failed to load: {e}")
                raise  # Raise error so user knows what went wrong

        # Concatenate along track axis
        self.data = np.concatenate(cubes, axis=0)

        if self.load_datacube:
            print(
                f"✅ Loaded custom datacube '{self.load_datacube}': {self.data.shape} (T × S × B)"
            )
        else:
            print(f"✅ Loaded full cube: {self.data.shape} (T × S × B)")

    def describe(self):
        print(f"\n=== {self.name} ===")

        if self.has_georef:
            T, S = self.X_ecef.shape
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
        else:
            # No georef data - show datacube info only
            if self.data is not None:
                T, S, B = self.data.shape
                print(f"Datacube shape: {T} tracks × {S} slits × {B} bands")
                print(
                    f"Wavelength range: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm"
                )
            print("\n⚠️  No georef data available")
            print(
                "✅ Spectral analysis functions available: plot_spectrum, illumination correction, etc."
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
        # Use the global config that was imported at the top of this file
        config.UHI_ALIGNMENT_DX = dx
        config.UHI_ALIGNMENT_DY = dy

        print(f"✅ UHI alignment adjusted:")
        print(f"   dx (East):  {dx:+.3f} m")
        print(f"   dy (North): {dy:+.3f} m")
        print(
            f"\n💡 This will be applied in plot_georef() with coordinate_system='NED'"
        )

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
        # Use self.wavelengths (updated by apply_wavelength_filter) instead of geofile
        # This ensures RGB extraction works after wavelength cropping
        wl = self.wavelengths

        # Find closest wavelength indices
        idxR = np.argmin(np.abs(wl - Rnm)) if wl is not None and len(wl) > 0 else None
        idxG = np.argmin(np.abs(wl - Gnm)) if wl is not None and len(wl) > 0 else None
        idxB = np.argmin(np.abs(wl - Bnm)) if wl is not None and len(wl) > 0 else None

        if idxR is None or idxG is None or idxB is None:
            raise RuntimeError(f"Requested wavelengths not available")

        # Extract channels (MUST COPY to avoid modifying source data during normalization)
        R = data_cube[:, :, idxR].copy()
        G = data_cube[:, :, idxG].copy()
        B = data_cube[:, :, idxB].copy()

        return R, G, B

    def _get_roi_color(self, roi_name, default_colors, custom_color_map=None):
        """
        Get consistent color for an ROI across all plots.
        Uses hardcoded colors for specific ROIs, then custom_color_map, then persistent color map, then assigns new color.

        NEW: Smart pattern matching - ROIs with keywords (bomb, dark, halo, rust, etc.) get consistent color hues.

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
        # 🎨 COLOR PALETTE - Define once, reuse everywhere
        COLOR_BOMBS = "#1E90FF"  # dodger blue (main bomb color)
        COLOR_DARK = "#000000"  # black (main dark feature color)
        COLOR_SEDIMENT = "#8b4513"  # saddle brown (main sediment color)
        COLOR_UNKNOWN = "#8A2BE2"  # violet purple (for unknown/rejected)
        COLOR_FILTERED_DARK = "#FFFF00"  # yellow (for filtered dark pixels)
        COLOR_FILTERED_DARK = "#FFFF00"  # yellow (for filtered dark pixels)
        COLOR_DARK_SEDIMENT = "#3F3F3F"  # medium-dark gray (distinct from pure black)
        COLOR_HALO = "#e5e57f"  # medium-dark gray (distinct from pure black)

        HARDCODED_ROI_COLORS = {
            # Individual bomb features
            "single bomb ring": COLOR_BOMBS,
            "single bomb inside": "#00FFFF",  # neon cyan blue (lighter variant)
            "double bomb 1": "#FF8C00",  # dark orange
            "double bomb 2": "#FFD700",  # gold
            "tripple bomb 1": "#800080",  # purple
            "tripple bomb 2": "#8A2BE2",  # violet
            "tripple bomb 3": "#FF00FF",  # bright magenta
            "all bombs": COLOR_BOMBS,
            "dark bomb": COLOR_DARK,
            # Dark features
            "dark spots": COLOR_DARK,
            "dark sediment": COLOR_DARK_SEDIMENT,  # medium-dark gray (distinct from pure black)
            # Sediment variants
            "sediment": COLOR_SEDIMENT,
            "brown leaf": COLOR_SEDIMENT,
            "yellow leaf": "#FFD700",  # gold/yellow
            # 🔥 VALIDATION ROIs (inherit colors from training)
            "validation_bombs": COLOR_BOMBS,
            "validation_dark": COLOR_DARK,
            "validation_sediment": COLOR_SEDIMENT,
            # 🔥 TRAINING ROIs (base colors)
            "training_bombs": COLOR_BOMBS,
            "training_dark": COLOR_DARK,
            "training_sediment": COLOR_SEDIMENT,
            # 🔥 CLASSIFICATION output (same colors as training)
            "classified_unknown": COLOR_UNKNOWN,
            "unclassified": COLOR_UNKNOWN,
            "classified_bombs": COLOR_BOMBS,
            "classified_dark": COLOR_DARK,
            "classified_sediment": COLOR_SEDIMENT,
            # 🔥 FILTERED classes (filtered_dark is distinct for visibility)
            "filtered_bombs": COLOR_BOMBS,
            "filtered_dark": COLOR_FILTERED_DARK,  # yellow (stands out)
            "filtered_sediment": "#D2B48C",  # tan (lighter brown)
            # 🔥 MERGED class for validation
            "new_sediment": COLOR_SEDIMENT,
            # 🔥 DISPLAY names (for plot_classification_map after name mapping)
            "bombs": COLOR_BOMBS,
            "dark": COLOR_DARK,
            "dark areas": COLOR_DARK,
            "sediment": COLOR_SEDIMENT,
            "dark_bomb": COLOR_DARK,
            "dark ": COLOR_DARK,
            "dark_pit": COLOR_DARK_SEDIMENT,
            "dark area near bomb": COLOR_DARK,
            "isolated dark area": COLOR_DARK_SEDIMENT,
            "halo": COLOR_HALO,
            "radial features a": "#FF00FF",
            "rust": "#b12222",
        }

        # NEW: Smart pattern matching for consistent color hues by feature type
        # Maps keyword patterns to color groups (hue families)
        PATTERN_COLOR_GROUPS = {
            # Bombs: Blue/Cyan family (variations in saturation/lightness)
            "bomb": [
                "#0000FF",
                "#1E90FF",
                "#4169E1",
                "#00BFFF",
                "#87CEEB",
            ],  # dark blue, dodger blue, royal blue, deep sky blue, light sky blue
            # Dark features: Black/Gray family
            "dark": [
                "#000000",
                "#2F2F2F",
                "#4F4F4F",
                "#696969",
                "#808080",
            ],  # pure black, dark gray, dim gray, medium gray, gray
            # Halos: Yellow/Orange family
            "halo": [
                "#FFD700",
                "#FFA500",
                "#FF8C00",
                "#FFAA00",
                "#FFB733",
            ],  # gold, orange, dark orange, bright orange, yellow-orange
            # Arms: Magenta/Pink family
            "arms": [
                "#FF00FF",
                "#FF69B4",
                "#FF1493",
                "#C71585",
                "#DB7093",
            ],  # magenta, hot pink, deep pink, medium violet red, pale violet red
            # Rust: Red/Brown family
            "rust": [
                "#B22222",
                "#CD5C5C",
                "#A52A2A",
                "#8B4513",
                "#D2691E",
            ],  # firebrick, indian red, brown, saddle brown, chocolate
            # Pits: Dark gray/black tones
            "pit": [
                "#1C1C1C",
                "#363636",
                "#4A4A4A",
                "#5E5E5E",
                "#727272",
            ],  # very dark gray variations
            # Sediment: Brown/Tan family (kept as-is)
            "sediment": [
                "#8B4513",
                "#A0522D",
                "#CD853F",
                "#DEB887",
                "#D2B48C",
            ],  # saddle brown, sienna, peru, burlywood, tan
        }

        # Priority 1: Custom color map for this specific plot (allows override)
        if custom_color_map and roi_name in custom_color_map:
            return custom_color_map[roi_name]

        # Priority 2: Hardcoded colors for known ROIs
        # Strip pixel count suffix if present (e.g., "bombs (2369 px)" → "bombs")
        base_roi_name = roi_name
        if " px)" in roi_name:
            # Extract base name before pixel count
            base_roi_name = roi_name.rsplit(" (", 1)[0]

        # Check both the full name and the base name
        if roi_name in HARDCODED_ROI_COLORS:
            # Save to persistent map for consistency
            self.roi_color_map[roi_name] = HARDCODED_ROI_COLORS[roi_name]
            return HARDCODED_ROI_COLORS[roi_name]
        elif base_roi_name in HARDCODED_ROI_COLORS:
            # Found base name, use that color
            self.roi_color_map[roi_name] = HARDCODED_ROI_COLORS[base_roi_name]
            return HARDCODED_ROI_COLORS[base_roi_name]

        # Priority 3: Persistent color map (already assigned)
        if roi_name in self.roi_color_map:
            return self.roi_color_map[roi_name]

        # Priority 4: NEW - Smart pattern-based color assignment
        # Check if ROI name contains any pattern keywords
        roi_name_lower = roi_name.lower()
        matched_pattern = None
        for pattern in PATTERN_COLOR_GROUPS:
            if pattern in roi_name_lower:
                matched_pattern = pattern
                break

        if matched_pattern:
            # Count how many ROIs with this pattern already exist
            pattern_count = sum(
                1
                for name in self.roi_color_map.keys()
                if matched_pattern in name.lower()
            )
            color_group = PATTERN_COLOR_GROUPS[matched_pattern]
            # Cycle through colors in this group
            color = color_group[pattern_count % len(color_group)]
            self.roi_color_map[roi_name] = color
            return color

        # Priority 5: Fallback - Assign from default colors (for ROIs without pattern match)
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
        # red_wl=654.2,
        # green_wl=560.0,
        # blue_wl=440.3,
        red_wl=620.0,
        green_wl=565.0,
        blue_wl=490.0,
        normalize=True,
        contrast_stretch=None,  # NEW: Contrast enhancement (e.g., 2.0 for 2% linear stretch)
        figsize=(11, 9),
        coordinate_system="NED",  # "ECEF", "NED", "LATLON", or None (auto = LATLON)
        use_local_origin=True,  # only used when coordinate_system == "ECEF"
        origin=None,  # (lat, lon, h); required for NED/ECEF, ignored for LATLON
        show_file_boundaries=True,
        alpha_for_nodata=0.0,
        interactive=True,  # click to show coordinates
        track_start=None,  # inclusive
        track_end=None,  # exclusive
        use_corrected=False,  # use illumination corrected data if available
        depth_overlay=False,  # NEW: Display Beer-Lambert depth heatmap instead of RGB
        depth_cmap="RdBu_r",  # Colormap for depth (blue=elevated/shallow, red=deep/pits)
        depth_vmin=None,  # Manual min depth (meters), auto if None
        depth_vmax=None,  # Manual max depth (meters), auto if None
        depth_cbar_fraction=0.046,  # Colorbar width as fraction of axis width
        depth_cbar_pad=0.04,  # Colorbar padding from axis
        depth_cbar_shrink=1.0,  # Colorbar height as fraction of axis height (0.0-1.0)
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
        roi_overlay_mode="markers",  # "markers" (scatter plot) or "solid" (overlay colored pixels on RGB)
        roi_solid_pixel_size=1,  # Size of each pixel in solid overlay mode (must be odd: 1, 3, 5, 7, etc.)
        roi_marker_size=100,
        roi_marker_shape="s",  # Marker shape: 's'=square, 'o'=circle, '^'=triangle, 'D'=diamond, 'v'=triangle_down, '<'=triangle_left, '>'=triangle_right, 'p'=pentagon, '*'=star, 'h'=hexagon, '+'=plus, 'x'=x
        roi_marker_edgewidth=2,  # Edge width for ROI markers (0 = no edge)
        roi_show_numbers=False,
        roi_legend_loc="best",  # Legend location: 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center', 'outside', or None to hide
        roi_legend_markersize=10,  # Size of color markers in legend (default=10)
        roi_legend_marker_border=True,  # Whether to show black border on legend markers
        roi_legend_show_counts=True,  # Whether to show pixel counts in legend labels (e.g., "bombs (2369)")
        # Cartographic options (NEW - map-style features)
        rotation_deg=0,  # Rotate plot by yaw angle in degrees (counter-clockwise positive)
        add_scale_bar=False,  # Add scale bar to plot
        scale_bar_length_m=None,  # Manual scale bar length in meters (auto-calculate if None)
        scale_bar_position="lower left",  # Position: 'lower left', 'lower right', 'upper left', 'upper right'
        scale_bar_color="black",  # Color of scale bar
        scale_bar_fontsize=10,  # Font size for scale bar text
        scale_bar_linewidth=4,  # Thickness of scale bar line
        add_north_arrow=False,  # Add north arrow to plot
        north_arrow_position="upper right",  # Position: 'lower left', 'lower right', 'upper left', 'upper right'
        north_arrow_size=0.08,  # Size of north arrow relative to plot (0.0-1.0)
        north_arrow_color="black",  # Color of north arrow
        north_arrow_linewidth=4,  # Thickness of north arrow line
        north_arrow_head_size=40,  # Size of arrow head (mutation_scale)
        # Single-wavelength colormap options (NEW - from plot_rgb)
        use_wavelength_colormap=False,  # If True, plot single wavelength with natural color
        wavelength_colormap_target=None,  # Target wavelength (nm) for single-wavelength mode
        derivative_order=0,  # Derivative order: 0=raw, 1=first derivative, 2=second derivative
        derivative_window=2,  # Window size for derivative computation
        vmin=(
            0.53,
            0.58,
            0.2,
        ),  # Manual min value for colormap normalization (single value or per-channel tuple)
        vmax=(
            1.35,
            1.37,
            1.73,
        ),  # Manual max value for colormap normalization (single value or per-channel tuple)
        colorbar_fraction=0.046,  # Size of colorbar relative to main plot (width for vertical, height for horizontal)
        colorbar_pad=0.04,  # Padding between plot and colorbar
        colorbar_shrink=1.0,  # Shrink factor for colorbar length (1.0 = full length)
        colorbar_orientation="vertical",  # Orientation: "vertical" or "horizontal"
        return_fig=False,
        quiet=True,  # suppress non interactive prints and warnings
        **pcolor_kwargs,
    ):
        """
        RGB georeferenced composite in LATLON, NED, or ECEF with optional trajectory.
        Only interactive click readouts are printed when interactive=True.

        Parameters
        ----------
        contrast_stretch : float, optional
            Apply linear contrast stretch by clipping the darkest and brightest percentiles.
            Value represents the percentage to clip on each end (e.g., 2.0 for 2% stretch).
            Common values: 2.0 (standard), 1.0 (subtle), 5.0 (aggressive).
            If None, no contrast enhancement is applied. Default: None

        apply_alignment_shift : bool, optional
            If True and coordinate_system=='NED', applies the alignment shift from
            config (UHI_ALIGNMENT_DX, UHI_ALIGNMENT_DY) to match MBES data.
            Default: False

        roi_overlay_mode : str, optional
            Mode for displaying ROI pixels: "markers" (scatter plot) or "solid" (overlay colored
            pixels directly on RGB image). Default: "markers"

        roi_solid_pixel_size : int, optional
            Size of each pixel block in solid overlay mode (only used when roi_overlay_mode="solid").
            Must be an odd number (1, 3, 5, 7, ...). Size 1 = single pixel (default), size 3 = 3×3 block
            centered on original pixel, size 5 = 5×5 block, etc. Default: 1

        rotation_deg : float, optional
            Rotate the entire plot by this angle in degrees (counter-clockwise positive).
            Useful for creating map-style visualizations with custom orientations.
            Default: 0 (no rotation)

        add_scale_bar : bool, optional
            Add a scale bar to the plot showing distance. Default: False

        scale_bar_length_m : float, optional
            Manual scale bar length in meters. If None, automatically calculates
            a "nice" round number (e.g., 100m, 500m, 1km) based on plot size.
            Default: None (auto)

        scale_bar_position : str, optional
            Position of scale bar: 'lower left', 'lower right', 'upper left', 'upper right'.
            Default: 'lower left'

        scale_bar_color : str, optional
            Color of scale bar and text. Default: 'black'

        scale_bar_fontsize : int, optional
            Font size for scale bar label. Default: 10

        scale_bar_linewidth : float, optional
            Thickness of the scale bar line and ticks. Default: 4

        add_north_arrow : bool, optional
            Add a north arrow to the plot. Arrow automatically adjusts for rotation_deg.
            Default: False

        north_arrow_position : str, optional
            Position of north arrow: 'lower left', 'lower right', 'upper left', 'upper right'.
            Default: 'upper right'

        north_arrow_size : float, optional
            Size of north arrow relative to plot height (0.0-1.0). Default: 0.08

        north_arrow_color : str, optional
            Color of north arrow and label. Default: 'black'

        north_arrow_linewidth : float, optional
            Thickness of the north arrow line. Default: 4

        north_arrow_head_size : float, optional
            Size of the arrow head (mutation_scale parameter). Larger values create
            bigger arrow heads. Default: 40
        """
        # Check if georef data is available
        if not self.has_georef:
            raise RuntimeError(
                "Cannot use plot_georef() - selected files have no georef data. "
                "Only spectral analysis functions are available (plot_spectrum, illumination correction, etc.)."
            )

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

        # Validate roi_solid_pixel_size parameter
        if roi_overlay_mode == "solid" and roi_solid_pixel_size != 1:
            if not isinstance(roi_solid_pixel_size, int):
                raise ValueError(
                    f"roi_solid_pixel_size must be an integer, got {type(roi_solid_pixel_size).__name__}: {roi_solid_pixel_size}"
                )
            if roi_solid_pixel_size < 1:
                raise ValueError(
                    f"roi_solid_pixel_size must be >= 1, got {roi_solid_pixel_size}"
                )
            if roi_solid_pixel_size % 2 == 0:
                raise ValueError(
                    f"roi_solid_pixel_size must be an odd number (1, 3, 5, 7, ...), got {roi_solid_pixel_size}. "
                    f"Use {roi_solid_pixel_size - 1} or {roi_solid_pixel_size + 1} instead."
                )

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

        # Initialize wavelength colormap variables (will be populated if use_wavelength_colormap=True)
        mesh = None
        wavelength_colormap_obj = None
        wavelength_intensity_raw = None
        wavelength_intensity_normalized = None
        actual_wl = None

        print(
            f"🔧 DEBUG: plot_georef called with use_wavelength_colormap={use_wavelength_colormap}, wavelength_colormap_target={wavelength_colormap_target}"
        )

        with _silence() as stack:
            if quiet:
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("ignore")
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

            # NEW: Single-wavelength colormap mode
            if use_wavelength_colormap:
                print(f"🌈 Single-wavelength colormap mode activated!")
                print(f"   Target wavelength: {wavelength_colormap_target} nm")
                if wavelength_colormap_target is None:
                    raise ValueError(
                        "wavelength_colormap_target must be specified when use_wavelength_colormap=True"
                    )

                # Get data cube
                if use_corrected:
                    if (
                        not hasattr(self, "data_corrected")
                        or self.data_corrected is None
                    ):
                        raise ValueError(
                            "Corrected data not available. Run apply_illumination_correction() first."
                        )
                    cube_data = self.data_corrected
                else:
                    cube_data = self.data

                X_ecef, Y_ecef, Z_ecef = self.X_ecef, self.Y_ecef, self.Z_ecef

                # Import the colormap creation function
                from ..ndi_analysis_utils import create_wavelength_colormap

                # Find closest wavelength index
                wl_idx = np.argmin(
                    np.abs(self.wavelengths - wavelength_colormap_target)
                )
                actual_wl = self.wavelengths[wl_idx]

                if not quiet:
                    print(f"🎨 Single-wavelength georef mode: {actual_wl:.1f} nm")

                # Compute intensity based on derivative order
                if derivative_order == 0:
                    # Raw intensity
                    intensity_map = cube_data[:, :, wl_idx].copy()
                    derivative_label = "Raw Intensity"
                elif derivative_order == 1:
                    # First derivative: dI/dλ
                    if (
                        wl_idx < derivative_window
                        or wl_idx >= len(self.wavelengths) - derivative_window
                    ):
                        raise ValueError(
                            f"Wavelength {wavelength_colormap_target} nm too close to edge for derivative with window={derivative_window}"
                        )

                    intensity_forward = cube_data[:, :, wl_idx + derivative_window]
                    intensity_backward = cube_data[:, :, wl_idx - derivative_window]
                    wl_forward = self.wavelengths[wl_idx + derivative_window]
                    wl_backward = self.wavelengths[wl_idx - derivative_window]

                    intensity_map = (intensity_forward - intensity_backward) / (
                        wl_forward - wl_backward
                    )
                    derivative_label = "1st Derivative (dI/dλ)"
                elif derivative_order == 2:
                    # Second derivative: d²I/dλ²
                    if (
                        wl_idx < derivative_window
                        or wl_idx >= len(self.wavelengths) - derivative_window
                    ):
                        raise ValueError(
                            f"Wavelength {wavelength_colormap_target} nm too close to edge for derivative with window={derivative_window}"
                        )

                    intensity_center = cube_data[:, :, wl_idx]
                    intensity_forward = cube_data[:, :, wl_idx + derivative_window]
                    intensity_backward = cube_data[:, :, wl_idx - derivative_window]
                    h = (
                        self.wavelengths[wl_idx + derivative_window]
                        - self.wavelengths[wl_idx]
                    )

                    intensity_map = (
                        intensity_forward - 2 * intensity_center + intensity_backward
                    ) / (h**2)
                    derivative_label = "2nd Derivative (d²I/dλ²)"
                else:
                    raise ValueError(
                        f"derivative_order must be 0, 1, or 2, got {derivative_order}"
                    )

                # IMPORTANT: In wavelength mode, vmin/vmax might be RGB tuples from default parameters
                # We need scalar values for single-wavelength normalization
                vmin_for_wavelength = vmin
                vmax_for_wavelength = vmax

                if isinstance(vmin, (tuple, list)) and len(vmin) > 1:
                    # Convert RGB tuple to scalar (use mean of the three channels)
                    vmin_for_wavelength = np.mean(vmin)
                    if not quiet:
                        print(
                            f"⚠️  Converted vmin tuple {vmin} to scalar {vmin_for_wavelength:.4f}"
                        )

                if isinstance(vmax, (tuple, list)) and len(vmax) > 1:
                    # Convert RGB tuple to scalar (use mean of the three channels)
                    vmax_for_wavelength = np.mean(vmax)
                    if not quiet:
                        print(
                            f"⚠️  Converted vmax tuple {vmax} to scalar {vmax_for_wavelength:.4f}"
                        )

                # Normalize to [0, 1] range
                if vmin_for_wavelength is not None and vmax_for_wavelength is not None:
                    # User-specified range
                    intensity_normalized = np.clip(
                        (intensity_map - vmin_for_wavelength)
                        / (vmax_for_wavelength - vmin_for_wavelength),
                        0,
                        1,
                    )
                else:
                    # Auto range
                    intensity_min = np.nanmin(intensity_map)
                    intensity_max = np.nanmax(intensity_map)
                    if intensity_max > intensity_min:
                        intensity_normalized = (intensity_map - intensity_min) / (
                            intensity_max - intensity_min
                        )
                    else:
                        intensity_normalized = np.zeros_like(intensity_map)

                # Store normalized intensity for pcolormesh (don't convert to RGB yet!)
                # pcolormesh will apply the colormap automatically, allowing colorbar to work
                wavelength_intensity_normalized = intensity_normalized

                # Create wavelength colormap
                wavelength_cmap = create_wavelength_colormap(actual_wl)

                # Store colormap info for later use
                wavelength_colormap_obj = wavelength_cmap
                wavelength_intensity_raw = intensity_map

                # Print raw intensity range (same as plot_rgb)
                print(f"   Derivative: {derivative_order}")
                print(
                    f"   Raw intensity range (before normalization): [{np.nanmin(intensity_map):.4f}, {np.nanmax(intensity_map):.4f}]"
                )

                # Create dummy R,G,B for compatibility with rest of code
                # These won't be used in wavelength mode - we'll use intensity_normalized directly
                R = intensity_normalized
                G = intensity_normalized
                B = intensity_normalized

            else:
                # Standard RGB mode
                if use_corrected:
                    if (
                        not hasattr(self, "data_corrected")
                        or self.data_corrected is None
                    ):
                        raise ValueError(
                            "Corrected data not available. Run apply_illumination_correction() first."
                        )
                    X_ecef, Y_ecef, Z_ecef = self.X_ecef, self.Y_ecef, self.Z_ecef
                    R, G, B = self._extract_rgb_from_cube(
                        self.data_corrected, red_wl, green_wl, blue_wl
                    )
                    if normalize:
                        # Check if vmin/vmax are provided for manual normalization
                        if vmin is not None and vmax is not None:
                            # Support both single values and per-channel tuples
                            if isinstance(vmin, (tuple, list)) and isinstance(
                                vmax, (tuple, list)
                            ):
                                # Per-channel normalization
                                if len(vmin) != 3 or len(vmax) != 3:
                                    raise ValueError(
                                        f"vmin and vmax must have 3 values for per-channel normalization, got {len(vmin)} and {len(vmax)}"
                                    )
                                for i, (C, ch_name) in enumerate(
                                    [(R, "Red"), (G, "Green"), (B, "Blue")]
                                ):
                                    ch_min, ch_max = vmin[i], vmax[i]
                                    if ch_max > ch_min:
                                        C[:] = np.clip(
                                            (C - ch_min) / (ch_max - ch_min), 0, 1
                                        )
                                    else:
                                        raise ValueError(
                                            f"{ch_name} channel: vmax ({ch_max}) must be > vmin ({ch_min})"
                                        )
                                if not quiet:
                                    print(
                                        f"✨ Manual per-channel normalization applied:"
                                    )
                                    print(f"   Red:   [{vmin[0]:.4f}, {vmax[0]:.4f}]")
                                    print(f"   Green: [{vmin[1]:.4f}, {vmax[1]:.4f}]")
                                    print(f"   Blue:  [{vmin[2]:.4f}, {vmax[2]:.4f}]")
                            else:
                                # Single value normalization (same range for all channels)
                                if vmax > vmin:
                                    for C in (R, G, B):
                                        C[:] = np.clip((C - vmin) / (vmax - vmin), 0, 1)
                                    if not quiet:
                                        print(
                                            f"✨ Manual normalization applied to all channels: [{vmin:.4f}, {vmax:.4f}]"
                                        )
                                else:
                                    raise ValueError(
                                        f"vmax ({vmax}) must be > vmin ({vmin})"
                                    )
                        else:
                            # Auto normalization: use global min/max per channel
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
            # Also slice wavelength intensity data if in wavelength mode
            if use_wavelength_colormap and wavelength_intensity_normalized is not None:
                wavelength_intensity_normalized = wavelength_intensity_normalized[
                    start_idx:end_idx, :
                ]
                # Also slice raw intensity for colorbar range calculation
                if wavelength_intensity_raw is not None:
                    wavelength_intensity_raw = wavelength_intensity_raw[
                        start_idx:end_idx, :
                    ]
            T, S = X_ecef.shape

        # ============================================================================
        # NEW: Beer-Lambert Depth Overlay
        # ============================================================================
        if depth_overlay:
            # Check if depth data has been computed
            if not hasattr(self, "depth_map") or self.depth_map is None:
                raise RuntimeError(
                    "❌ ERROR: Depth map not computed!\\n"
                    "   Please run: cube.compute_beer_lambert_depth()\\n"
                    "   before using depth_overlay=True"
                )

            if not quiet:
                print("🌊 Using Beer-Lambert depth overlay instead of RGB")
                print(
                    f"   Depth range requested: {self.depth_track_start} to {self.depth_track_end}"
                )
                print(f"   Plot range requested: {start_idx} to {end_idx}")

            # Verify track ranges match
            if start_idx < self.depth_track_start or end_idx > self.depth_track_end:
                raise RuntimeError(
                    f"❌ ERROR: Track range mismatch!\\n"
                    f"   Depth computed for tracks [{self.depth_track_start}, {self.depth_track_end}]\\n"
                    f"   But plot requests tracks [{start_idx}, {end_idx}]\\n"
                    f"   Please recompute depth with matching track range."
                )

            # Extract depth data for requested track range (relative to depth_track_start)
            depth_start_offset = start_idx - self.depth_track_start
            depth_end_offset = end_idx - self.depth_track_start
            depth_data = self.depth_map[
                depth_start_offset:depth_end_offset, :
            ]  # (T, S)

            if not quiet:
                print(f"   Depth data shape: {depth_data.shape}")
                print(
                    f"   Depth range: {depth_data.min():.3f} to {depth_data.max():.3f} m"
                )

            # Determine colormap range
            if depth_vmin is None:
                depth_vmin = np.percentile(depth_data[np.isfinite(depth_data)], 2)
            if depth_vmax is None:
                depth_vmax = np.percentile(depth_data[np.isfinite(depth_data)], 98)

            # Always print colorbar limits for user reference
            print(
                f"🎨 UHI plot_georef depth colorbar range: depth_vmin={depth_vmin:.4f}, depth_vmax={depth_vmax:.4f}"
            )

            if not quiet:
                print(f"   Colormap range: [{depth_vmin:.3f}, {depth_vmax:.3f}] m")

            # Normalize depth to [0, 1] for colormap
            depth_normalized = np.clip(
                (depth_data - depth_vmin) / (depth_vmax - depth_vmin), 0, 1
            )

            # Apply colormap
            import matplotlib.cm as cm

            cmap = cm.get_cmap(depth_cmap)
            depth_rgba = cmap(depth_normalized)  # (T, S, 4) RGBA

            # Extract RGB channels (discard alpha)
            R = depth_rgba[:, :, 0]
            G = depth_rgba[:, :, 1]
            B = depth_rgba[:, :, 2]

            if not quiet:
                print(f"✅ Depth colormap applied ({depth_cmap})")

            # Override wavelength colormap mode if it was set
            use_wavelength_colormap = False

        RGB = np.dstack([R, G, B]).astype(np.float64)
        alpha = np.ones((T, S), dtype=np.float64)
        alpha[~(np.isfinite(R) & np.isfinite(G) & np.isfinite(B))] = 0.0

        # Apply contrast stretch if requested (skip for depth overlay as it has its own normalization)
        if contrast_stretch is not None and contrast_stretch > 0 and not depth_overlay:
            for i in range(3):  # Apply to R, G, B channels separately
                channel = RGB[:, :, i]
                valid_data = channel[np.isfinite(channel)]
                if len(valid_data) > 0:
                    # Calculate percentile values
                    p_low = np.percentile(valid_data, contrast_stretch)
                    p_high = np.percentile(valid_data, 100 - contrast_stretch)

                    # Clip and stretch to [0, 1]
                    if p_high > p_low:
                        channel_stretched = np.clip(channel, p_low, p_high)
                        channel_stretched = (channel_stretched - p_low) / (
                            p_high - p_low
                        )
                        RGB[:, :, i] = channel_stretched

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
            xlabel, ylabel = f"East (m) from {lat0}°, {lon0}°", "North [m]"
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

        # Apply rotation if specified
        if rotation_deg != 0:
            # Convert to radians (counter-clockwise positive)
            theta = np.deg2rad(rotation_deg)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Rotation matrix: [x', y'] = R * [x, y]
            # R = [[cos, -sin], [sin, cos]]
            Xp_rot = Xp * cos_theta - Yp * sin_theta
            Yp_rot = Xp * sin_theta + Yp * cos_theta
            Xp = Xp_rot
            Yp = Yp_rot

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

        # ========== SOLID ROI OVERLAY (applied to RGB before pcolormesh) ==========
        if (
            roi_overlay_mode == "solid"
            and roi_collection is not None
            and not use_wavelength_colormap
        ):
            # Convert RGB to uint8 for pixel manipulation
            rgb_overlay = np.stack([R, G, B], axis=-1)
            rgb_overlay = np.clip(rgb_overlay * 255, 0, 255).astype(np.uint8)

            # Prepare ROI collection
            if roi_collection == "all":
                rois_to_plot = self.roi_collection
            elif isinstance(roi_collection, list):
                rois_to_plot = {
                    name: self.roi_collection[name]
                    for name in roi_collection
                    if name in self.roi_collection
                }
            elif isinstance(roi_collection, dict):
                rois_to_plot = roi_collection
            else:
                rois_to_plot = {}

            # Apply solid colors to ROI pixels
            roi_pixel_counts = {}  # Track counts for legend

            # Calculate pixel radius for block painting (e.g., size=3 → radius=1, size=5 → radius=2)
            pixel_radius = roi_solid_pixel_size // 2

            for roi_idx, (roi_name, roi_pixels_list) in enumerate(rois_to_plot.items()):
                if roi_pixels_list:
                    # Get color for this ROI
                    color = self._get_roi_color(roi_name, roi_colors, roi_color_map)

                    # Convert matplotlib color to RGB tuple
                    from matplotlib.colors import to_rgb

                    rgb_color = np.array(to_rgb(color)) * 255
                    rgb_color = rgb_color.astype(np.uint8)

                    pixel_count = 0
                    # Apply color to each ROI pixel (and surrounding pixels if size > 1)
                    for slit, track in roi_pixels_list:
                        # Adjust track by start_idx
                        t = track - start_idx

                        # Bounds check for center pixel
                        if 0 <= t < T and 0 <= slit < S:
                            # Paint center pixel and surrounding block
                            for dt in range(-pixel_radius, pixel_radius + 1):
                                for ds in range(-pixel_radius, pixel_radius + 1):
                                    t_paint = t + dt
                                    s_paint = slit + ds

                                    # Bounds check for each pixel in the block
                                    if 0 <= t_paint < T and 0 <= s_paint < S:
                                        rgb_overlay[t_paint, s_paint] = rgb_color

                            pixel_count += 1

                    roi_pixel_counts[roi_name] = pixel_count

            # Convert back to [0, 1] range for pcolormesh
            R = rgb_overlay[:, :, 0] / 255.0
            G = rgb_overlay[:, :, 1] / 255.0
            B = rgb_overlay[:, :, 2] / 255.0
            RGB = np.dstack([R, G, B])

        # Remove roi_overlay_mode from pcolor_kwargs if present (it's not a valid pcolormesh parameter)
        pcolor_kwargs.pop("roi_overlay_mode", None)

        # Create mesh object - use different approach for wavelength mode vs RGB mode
        if use_wavelength_colormap:
            # Wavelength mode: pass raw intensity data with actual vmin/vmax
            # This provides proper dynamic range control for visualization

            # Use raw intensity values directly (not normalized)
            display_data = wavelength_intensity_raw

            # DEBUG: Show what type vmin/vmax are
            print(f"🔍 DEBUG vmin type: {type(vmin)}, value: {vmin}")
            print(f"🔍 DEBUG vmax type: {type(vmax)}, value: {vmax}")

            # For wavelength mode, check if vmin/vmax are scalars (user-specified)
            # If they're tuples, they're the RGB defaults - ignore them and use data range
            if isinstance(vmin, (tuple, list)) or isinstance(vmax, (tuple, list)):
                # RGB tuple defaults - not applicable for single wavelength
                # Use actual data range instead
                display_vmin = np.nanmin(display_data)
                display_vmax = np.nanmax(display_data)
                print(f"✨ Auto-computed vmin/vmax from wavelength data:")
                print(f"   Intensity range: [{display_vmin:.6f}, {display_vmax:.6f}]")
            elif vmin is not None and vmax is not None:
                # User specified scalar vmin/vmax - use them
                display_vmin = vmin
                display_vmax = vmax
                print(f"✨ Using user-specified vmin/vmax:")
                print(f"   Intensity range: [{display_vmin:.6f}, {display_vmax:.6f}]")
            else:
                # No vmin/vmax specified at all - use data range
                display_vmin = np.nanmin(display_data)
                display_vmax = np.nanmax(display_data)
                print(f"✨ Auto-computed vmin/vmax from wavelength data:")
                print(f"   Intensity range: [{display_vmin:.6f}, {display_vmax:.6f}]")

            mesh = ax.pcolormesh(
                Xc,
                Yc,
                display_data,
                shading="flat",
                cmap=wavelength_colormap_obj,
                vmin=display_vmin,
                vmax=display_vmax,
                **pcolor_kwargs,
            )
        else:
            # RGB mode: pass 3D RGB array
            mesh = ax.pcolormesh(Xc, Yc, RGB, shading="flat", **pcolor_kwargs)

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

            # Check overlay mode: solid or markers
            if roi_overlay_mode == "solid":
                # Solid overlay mode: ROI pixels already colored in RGB above
                # Just add legend entries without scatter plots
                from matplotlib.patches import Patch

                legend_handles = []
                for roi_name, roi_pixels_list in rois_to_plot.items():
                    if roi_name in roi_pixel_counts and roi_pixel_counts[roi_name] > 0:
                        color = self._get_roi_color(roi_name, roi_colors, roi_color_map)
                        # Check if roi_name already contains pixel count (e.g., "bombs (2369 px)" or "bombs (2369)")
                        # If so, don't add it again to avoid duplication
                        if " px)" in roi_name or (
                            roi_legend_show_counts
                            and ")" in roi_name
                            and "(" in roi_name
                        ):
                            # Already has pixel count, use as-is
                            label = roi_name
                        elif roi_legend_show_counts:
                            # Add pixel count if requested
                            label = f"{roi_name} ({roi_pixel_counts[roi_name]})"
                        else:
                            # Don't add pixel count
                            label = roi_name
                        legend_handles.append(
                            Patch(
                                facecolor=color,
                                label=label,
                            )
                        )

                # Add legend if we have entries
                if legend_handles and roi_legend_loc is not None:
                    if roi_legend_loc == "outside":
                        ax.legend(
                            handles=legend_handles,
                            loc="center left",
                            bbox_to_anchor=(1.02, 0.5),
                            fontsize=9,
                            framealpha=0.9,
                        )
                    else:
                        ax.legend(
                            handles=legend_handles,
                            loc=roi_legend_loc,
                            fontsize=9,
                            framealpha=0.9,
                        )
            else:
                # Marker mode: Plot each ROI with scatter (original behavior)
                # Plot each ROI with different color
                for roi_idx, (roi_name, roi_pixels_list) in enumerate(
                    rois_to_plot.items()
                ):
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
                            color = self._get_roi_color(
                                roi_name, roi_colors, roi_color_map
                            )

                            # Plot ROI with solid colors - FAST METHOD
                            ax.scatter(
                                roi_x_coords,
                                roi_y_coords,
                                c=color,
                                s=roi_marker_size**2 * 50,
                                marker="s",
                                edgecolors=(
                                    "black" if roi_marker_edgewidth > 0 else "none"
                                ),
                                linewidths=roi_marker_edgewidth,
                                alpha=1.0,
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

        # Add legend if perimeter lines or ROIs are shown (only for marker mode)
        # Note: In solid overlay mode, legend is already created in the ROI plotting section above
        if (
            perimeter_line is not None
            or (roi_collection is not None and roi_overlay_mode != "solid")
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

        # ========== Add scale bar if requested ==========
        if add_scale_bar:
            from matplotlib.patches import Rectangle
            from matplotlib.lines import Line2D

            # Get plot coordinate range
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]

            # Determine scale bar length in plot coordinates
            if scale_bar_length_m is None:
                # Auto-calculate nice scale bar length (typically 10-20% of plot width)
                # For NED/ECEF, coordinates are already in meters
                # For LATLON, need to estimate meters from degrees
                if coordinate_system.upper() == "LATLON":
                    # Rough approximation: 1 degree latitude ≈ 111 km
                    # Use mean latitude for better estimate
                    if origin is not None:
                        mean_lat = origin[0]
                    else:
                        mean_lat = np.nanmean(Yp)  # Use mean of plot latitude

                    # Meters per degree longitude varies with latitude
                    m_per_deg_lon = 111320 * np.cos(np.deg2rad(mean_lat))
                    m_per_deg_lat = 111320

                    # Estimate plot width in meters
                    plot_width_m = x_range * m_per_deg_lon
                    target_length = plot_width_m * 0.15
                else:
                    # NED or ECEF - already in meters
                    plot_width_m = x_range
                    target_length = plot_width_m * 0.15

                # Debug output
                if not quiet:
                    print(f"[Scale Bar Debug]")
                    print(f"  Coordinate system: {coordinate_system}")
                    print(f"  Plot x_range: {x_range:.2f}")
                    print(f"  Plot width in meters: {plot_width_m:.2f} m")
                    print(f"  Target scale bar length (15%): {target_length:.2f} m")

                # Choose nice round number (powers of 10, 2, or 5)
                if target_length > 0:
                    magnitude = 10 ** np.floor(np.log10(target_length))
                    normalized = target_length / magnitude

                    if normalized < 2:
                        nice_length = magnitude
                    elif normalized < 5:
                        nice_length = 2 * magnitude
                    else:
                        nice_length = 5 * magnitude

                    scale_bar_length_m = nice_length
                else:
                    # Fallback if something is wrong
                    scale_bar_length_m = 100.0  # Default 100m

                if not quiet:
                    print(f"  Final scale bar length: {scale_bar_length_m:.2f} m")
            else:
                # User specified scale bar length - use it directly
                if not quiet:
                    print(
                        f"[Scale Bar] Using user-specified length: {scale_bar_length_m:.2f} m"
                    )

            # Convert scale bar length from meters to plot coordinates
            if coordinate_system.upper() == "LATLON":
                if origin is not None:
                    mean_lat = origin[0]
                else:
                    mean_lat = np.nanmean(Yp)
                m_per_deg_lon = 111320 * np.cos(np.deg2rad(mean_lat))
                scale_bar_length_plot = scale_bar_length_m / m_per_deg_lon
            else:
                # NED or ECEF - already in meters
                scale_bar_length_plot = scale_bar_length_m

            # Position scale bar
            margin_x = 0.05 * x_range
            margin_y = 0.05 * y_range

            if "lower" in scale_bar_position:
                y_pos = ylim[0] + margin_y
            else:  # upper
                y_pos = ylim[1] - margin_y - 0.02 * y_range

            if "left" in scale_bar_position:
                x_pos = xlim[0] + margin_x
            else:  # right
                x_pos = xlim[1] - margin_x - scale_bar_length_plot

            # Draw scale bar with user-specified thickness
            ax.plot(
                [x_pos, x_pos + scale_bar_length_plot],
                [y_pos, y_pos],
                color=scale_bar_color,
                linewidth=scale_bar_linewidth,
                solid_capstyle="butt",
                zorder=1000,
            )

            # Add smaller ticks at ends
            tick_height = 0.008 * y_range  # Smaller ticks
            ax.plot(
                [x_pos, x_pos],
                [y_pos - tick_height, y_pos + tick_height],
                color=scale_bar_color,
                linewidth=scale_bar_linewidth,
                zorder=1000,
            )
            ax.plot(
                [x_pos + scale_bar_length_plot, x_pos + scale_bar_length_plot],
                [y_pos - tick_height, y_pos + tick_height],
                color=scale_bar_color,
                linewidth=scale_bar_linewidth,
                zorder=1000,
            )

            # Add text label (no background)
            if scale_bar_length_m >= 1000:
                label_text = f"{scale_bar_length_m/1000:.1f} km"
            elif scale_bar_length_m >= 1:
                label_text = f"{int(scale_bar_length_m)} m"
            else:
                # For sub-meter lengths, show decimal
                label_text = f"{scale_bar_length_m:.1f} m"

            ax.text(
                x_pos + scale_bar_length_plot / 2,
                y_pos + 0.020 * y_range,
                label_text,
                ha="center",
                va="bottom",
                fontsize=scale_bar_fontsize,
                color=scale_bar_color,
                weight="bold",
                zorder=1000,
            )

        # ========== Add north arrow if requested ==========
        if add_north_arrow:
            from matplotlib.patches import FancyArrow, FancyArrowPatch, Polygon
            import matplotlib.patches as mpatches

            # Position north arrow
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]

            margin_x = 0.05 * x_range
            margin_y = 0.05 * y_range

            if "lower" in north_arrow_position:
                y_center = ylim[0] + margin_y + north_arrow_size * y_range / 2
            else:  # upper
                y_center = ylim[1] - margin_y - north_arrow_size * y_range / 2

            if "left" in north_arrow_position:
                x_center = xlim[0] + margin_x + north_arrow_size * x_range / 2
            else:  # right
                x_center = xlim[1] - margin_x - north_arrow_size * x_range / 2

            # Calculate arrow dimensions - shorter and thicker
            arrow_length = north_arrow_size * y_range * 0.5  # Reduced from 0.8 to 0.5

            # North direction in rotated coordinates
            # If plot is rotated by rotation_deg, north arrow needs to point in opposite direction
            north_angle_rad = np.deg2rad(
                -rotation_deg
            )  # Negative because we rotated the data

            # Arrow start and end points
            x_start = x_center
            y_start = y_center - arrow_length / 2
            dx = arrow_length * np.sin(north_angle_rad)
            dy = arrow_length * np.cos(north_angle_rad)

            # Draw arrow using FancyArrowPatch with user-specified head size and line thickness
            arrow = FancyArrowPatch(
                (x_start, y_start),
                (x_start + dx, y_start + dy),
                arrowstyle="->",
                mutation_scale=north_arrow_head_size,
                linewidth=north_arrow_linewidth,
                color=north_arrow_color,
                zorder=1000,
            )
            ax.add_patch(arrow)

            # Add "N" label at arrow tip (no background box)
            text_offset = arrow_length * 0.20
            label_x = x_start + dx + text_offset * np.sin(north_angle_rad)
            label_y = y_start + dy + text_offset * np.cos(north_angle_rad)

            ax.text(
                label_x,
                label_y,
                "N",
                ha="center",
                va="center",
                fontsize=scale_bar_fontsize + 6,
                color=north_arrow_color,
                weight="bold",
                zorder=1000,
            )

        # -------- Colorbar for wavelength mode --------
        if use_wavelength_colormap:
            # Add colorbar with wavelength information (with size controls)
            cbar = plt.colorbar(
                mesh,
                ax=ax,
                fraction=colorbar_fraction,
                pad=colorbar_pad,
                shrink=colorbar_shrink,
                orientation=colorbar_orientation,
            )

            # Compute actual range - use the raw intensity values, not normalized [0,1]
            if vmin is not None and vmax is not None:
                # If vmin/vmax were tuples, they were already converted to scalars earlier
                # Use the original vmin/vmax (not the normalized 0-1 range)
                if isinstance(vmin, (tuple, list)):
                    range_min = np.mean(vmin)
                else:
                    range_min = vmin
                if isinstance(vmax, (tuple, list)):
                    range_max = np.mean(vmax)
                else:
                    range_max = vmax
            else:
                range_min = np.nanmin(wavelength_intensity_raw)
                range_max = np.nanmax(wavelength_intensity_raw)

            # Set colorbar label
            if derivative_order == 0:
                cbar.set_label(
                    f"Intensity at {actual_wl:.1f} nm", rotation=270, labelpad=15
                )
            elif derivative_order == 1:
                cbar.set_label(
                    f"dI/dλ at {actual_wl:.1f} nm", rotation=270, labelpad=15
                )
            elif derivative_order == 2:
                cbar.set_label(
                    f"d²I/dλ² at {actual_wl:.1f} nm", rotation=270, labelpad=15
                )

            # Update tick labels to show actual intensity values
            cbar_ticks = cbar.get_ticks()
            cbar.set_ticklabels(
                [f"{range_min + t * (range_max - range_min):.3f}" for t in cbar_ticks]
            )

        # -------- Colorbar for depth overlay mode --------
        if depth_overlay:
            # Create a scalar mappable for the colorbar
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=depth_vmin, vmax=depth_vmax)
            sm = cm.ScalarMappable(cmap=depth_cmap, norm=norm)
            sm.set_array([])

            # Add colorbar with adjustable size (width, height, padding)
            cbar = plt.colorbar(
                sm,
                ax=ax,
                fraction=depth_cbar_fraction,
                pad=depth_cbar_pad,
                shrink=depth_cbar_shrink,
            )
            cbar.set_label(
                "Relative Depth Δz (m)",
                rotation=270,
                labelpad=20,
                fontsize=12,
                fontweight="bold",
            )

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

        # -------- Colorbar for wavelength mode --------
        if use_wavelength_colormap:
            # Add colorbar with wavelength information (with size controls)
            cbar = plt.colorbar(
                mesh,
                ax=ax,
                fraction=colorbar_fraction,
                pad=colorbar_pad,
                shrink=colorbar_shrink,
                orientation=colorbar_orientation,
            )

            # Compute actual range - use the raw intensity values, not normalized [0,1]
            if vmin is not None and vmax is not None:
                # If vmin/vmax were tuples, they were already converted to scalars earlier
                # Use the original vmin/vmax (not the normalized 0-1 range)
                if isinstance(vmin, (tuple, list)):
                    range_min = np.mean(vmin)
                else:
                    range_min = vmin
                if isinstance(vmax, (tuple, list)):
                    range_max = np.mean(vmax)
                else:
                    range_max = vmax
            else:
                range_min = np.nanmin(wavelength_intensity_raw)
                range_max = np.nanmax(wavelength_intensity_raw)

            # Set colorbar label
            if derivative_order == 0:
                cbar.set_label(
                    f"Intensity at {actual_wl:.1f} nm", rotation=270, labelpad=15
                )
            elif derivative_order == 1:
                cbar.set_label(
                    f"dI/dλ at {actual_wl:.1f} nm", rotation=270, labelpad=15
                )
            elif derivative_order == 2:
                cbar.set_label(
                    f"d²I/dλ² at {actual_wl:.1f} nm", rotation=270, labelpad=15
                )

            # Update tick labels to show actual intensity values
            cbar_ticks = cbar.get_ticks()
            cbar.set_ticklabels(
                [f"{range_min + t * (range_max - range_min):.3f}" for t in cbar_ticks]
            )

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

    @staticmethod
    def _compute_ref_for_slit_band(args):
        """
        Helper function for parallel computation of reference values.
        Computes reference for a single (slit, band) pair.

        Args:
            args: Tuple of (s, b, file_paths, dset_name, use_global, window_size)

        Returns:
            Tuple of (s, b, ref_value) where ref_value is either a scalar or array
        """
        import h5py
        import numpy as np
        import pandas as pd

        s, b, file_paths, dset_name, use_global, window_size = args

        # Collect values across all files for this slit-band
        values = []
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                if dset_name not in f:
                    raise ValueError(f"Dataset {dset_name} not found in {file_path}")
                values.append(f[dset_name][:, s, b])

        ts = np.concatenate(values)

        if use_global:
            # Global: compute single median
            ref = np.nanmedian(ts)
            if not np.isfinite(ref) or ref == 0:
                ref = 1.0
            return (s, b, ref)
        else:
            # Rolling: compute rolling median
            ref_vec = (
                pd.Series(ts)
                .rolling(window=int(window_size), center=True, min_periods=1)
                .median()
                .values
            )
            ref_vec[ref_vec == 0] = 1.0
            ref_vec[~np.isfinite(ref_vec)] = 1.0
            return (s, b, ref_vec.astype(np.float32))

    def apply_illumination_correction_v2(
        self,
        window_size=500,
        strength=1.0,
        force_recompute=False,
        use_smoothed_input=False,  # NEW: Use smoothed raw data as input
        smooth_method="gaussian",  # NEW: Smoothing method used
        smooth_sigma=2,  # NEW: For gaussian
        smooth_window=10,  # NEW: For moving_average/savgol
        smooth_polyorder=2,  # NEW: For savgol
        smooth_kernel=5,  # NEW: For median
        smooth_lambda=1e4,  # NEW: For whittaker
        smooth_sigma_spatial=2,  # NEW: For bilateral
        smooth_sigma_intensity=0.1,  # NEW: For bilateral
    ):
        """
        V2: Memory-efficient version using file-by-file processing and direct disk writes.

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
        use_smoothed_input : bool, default=False
            NEW: If True, uses smoothed raw data as input instead of raw data.
            Requires that apply_spectral_smoothing(apply_to_raw=True) was run first.
        smooth_method : str
            NEW: Smoothing method to look for ('gaussian', 'moving_average', etc.)
        smooth_sigma, smooth_window, etc. : various
            NEW: Parameters for the smoothing method used.

        Example:
        --------
        # Standard workflow (no smoothing):
        >>> cube.apply_illumination_correction_v2()

        # NEW: With pre-smoothed raw data:
        >>> cube.apply_spectral_smoothing(method="gaussian", gaussian_sigma=5, apply_to_raw=True)
        >>> cube.apply_illumination_correction_v2(use_smoothed_input=True, smooth_method="gaussian", smooth_sigma=5)
        """
        import h5py
        import numpy as np
        import pandas as pd

        # Build smooth_params dictionary for naming
        smooth_params = None
        if use_smoothed_input:
            smooth_params = {
                "gaussian_sigma": smooth_sigma,
                "wavelength_smoothing": smooth_window,
                "savgol_polyorder": smooth_polyorder,
                "median_kernel_size": smooth_kernel,
                "whittaker_lambda": smooth_lambda,
                "bilateral_sigma_spatial": smooth_sigma_spatial,
                "bilateral_sigma_intensity": smooth_sigma_intensity,
            }

        print("🔄 Using V2 algorithm (pandas rolling median, memory-efficient)")
        if use_smoothed_input:
            print(f"   📊 Using smoothed raw data as input (method={smooth_method})")

        # --- Check if already computed and saved ---
        if not force_recompute and self.has_illumination_correction(
            window_size, strength, use_smoothed_input, smooth_method, smooth_params
        ):
            print(
                f"✅ Illumination correction already applied with window={window_size}, strength={strength}"
            )
            if use_smoothed_input:
                print(f"   (with smoothed input: {smooth_method})")
            print(f"   Loading from disk...")
            return self.load_illumination_correction(
                window_size, strength, use_smoothed_input, smooth_method, smooth_params
            )

        # NEW: Check if smoothed raw data exists when use_smoothed_input=True
        if use_smoothed_input:
            # Check if smoothed raw data cache exists
            if not self.has_spectral_smoothing(
                method=smooth_method,
                wavelength_smoothing=smooth_window,
                gaussian_sigma=smooth_sigma,
                savgol_polyorder=smooth_polyorder,
                median_kernel_size=smooth_kernel,
                whittaker_lambda=smooth_lambda,
                bilateral_sigma_spatial=smooth_sigma_spatial,
                bilateral_sigma_intensity=smooth_sigma_intensity,
                apply_to_raw=True,  # Check for RAW smoothing cache
            ):
                raise ValueError(
                    f"❌ Smoothed raw data not found (method={smooth_method}).\n"
                    f"   You must run apply_spectral_smoothing(method='{smooth_method}', ..., apply_to_raw=True) first!\n"
                    f"   Example:\n"
                    f"   >>> cube.apply_spectral_smoothing(method='{smooth_method}', gaussian_sigma={smooth_sigma}, apply_to_raw=True)\n"
                    f"   >>> cube.apply_illumination_correction_v2(use_smoothed_input=True, smooth_method='{smooth_method}', smooth_sigma={smooth_sigma})"
                )

            print(f"✅ Found smoothed raw data cache (method={smooth_method})")

            # Determine smoothed raw data dataset name
            smoothed_raw_dset = self._get_smoothing_dataset_name(
                method=smooth_method,
                wavelength_smoothing=smooth_window,
                gaussian_sigma=smooth_sigma,
                savgol_polyorder=smooth_polyorder,
                median_kernel_size=smooth_kernel,
                whittaker_lambda=smooth_lambda,
                bilateral_sigma_spatial=smooth_sigma_spatial,
                bilateral_sigma_intensity=smooth_sigma_intensity,
                apply_to_raw=True,
            )
        else:
            smoothed_raw_dset = None

        # --- sizes ---
        T_total = 0
        S = B = None
        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                # NEW: Use smoothed raw data if requested
                if use_smoothed_input and smoothed_raw_dset:
                    if smoothed_raw_dset in f:
                        dset_name = smoothed_raw_dset
                    else:
                        raise ValueError(
                            f"❌ Smoothed raw data not found in {gf.name}: {smoothed_raw_dset}"
                        )
                else:
                    # Original logic: use raw or corrected data
                    dset_name = (
                        gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                    )
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
        print(f"💾 Memory-efficient mode: Processing file-by-file, direct disk write")

        # PHASE 1: Compute reference statistics across all files (memory-efficient)
        print("📊 Phase 1/2: Computing reference statistics...")

        # Determine number of parallel workers
        # Use ThreadPool instead of multiprocessing.Pool to avoid file locking issues on Windows
        from multiprocessing.pool import ThreadPool
        import multiprocessing as mp

        n_workers = min(mp.cpu_count() - 1, 8)  # Use all CPUs minus 1, max 8
        print(f"   Using {n_workers} parallel workers (thread-based)")

        try:
            from tqdm import tqdm

            pbar = tqdm(total=S * B, desc="   Computing refs", unit="slit-band")
            use_tqdm = True
        except Exception:
            use_tqdm = False
            pbar = None

        # Determine dataset name to use
        file_paths = [gf.path for gf in self.geofiles]

        # Determine which dataset to load from
        if use_smoothed_input and smoothed_raw_dset:
            dset_name_to_use = smoothed_raw_dset
        else:
            # Check first file for dataset name
            with h5py.File(file_paths[0], "r") as f:
                gf0 = self.geofiles[0]
                dset_name_to_use = (
                    gf0.DSET_RGB_CORR if gf0.use_corrected else gf0.DSET_RGB_MAIN
                )
                if dset_name_to_use not in f:
                    dset_name_to_use = gf0.DSET_RGB_MAIN

        # Store reference values (small memory footprint: S × B × 4 bytes)
        if use_global:
            # Global: one value per (slit, band)
            ref_values = np.ones((S, B), dtype=np.float32)

            # Prepare arguments for parallel processing
            args_list = [
                (s, b, file_paths, dset_name_to_use, True, None)
                for s in range(S)
                for b in range(B)
            ]

            # Process in parallel (using ThreadPool to avoid Windows file locking)
            with ThreadPool(processes=n_workers) as pool:
                if use_tqdm:
                    results = []
                    for result in pool.imap_unordered(
                        self._compute_ref_for_slit_band, args_list
                    ):
                        results.append(result)
                        pbar.update(1)
                else:
                    results = pool.map(self._compute_ref_for_slit_band, args_list)

            # Fill in results
            for s, b, ref in results:
                ref_values[s, b] = ref
        else:
            # Rolling: one vector per (slit, band)
            ref_values = {}

            # Prepare arguments for parallel processing
            args_list = [
                (s, b, file_paths, dset_name_to_use, False, window_size)
                for s in range(S)
                for b in range(B)
            ]

            # Process in parallel (using ThreadPool to avoid Windows file locking)
            with ThreadPool(processes=n_workers) as pool:
                if use_tqdm:
                    for result in pool.imap_unordered(
                        self._compute_ref_for_slit_band, args_list
                    ):
                        s, b, ref_vec = result
                        ref_values[(s, b)] = ref_vec
                        pbar.update(1)
                else:
                    results = pool.map(self._compute_ref_for_slit_band, args_list)
                    for s, b, ref_vec in results:
                        ref_values[(s, b)] = ref_vec

        if use_tqdm:
            pbar.close()

        # Force cleanup: ensure all threads have released file handles
        # Windows can be slow to release HDF5 file locks even after closing
        import time
        import gc

        gc.collect()  # Force garbage collection to close any lingering file handles
        time.sleep(1.0)  # Give Windows time to release file locks (increased from 0.5s)
        print("   ✓ Phase 1 complete, waiting for file locks to release...")

        # PHASE 2: Apply correction file-by-file and save directly to disk
        print("💾 Phase 2/2: Applying correction and saving to disk...")

        dset_name_out = self._get_correction_dataset_name(window_size, strength)

        for gf in self.geofiles:
            print(f"   Processing {gf.name}...")

            # First, determine input dataset name and get dimensions
            with h5py.File(gf.path, "r") as f_in:
                # NEW: Use smoothed raw data if requested
                if use_smoothed_input and smoothed_raw_dset:
                    if smoothed_raw_dset in f_in:
                        dset_name_in = smoothed_raw_dset
                    else:
                        raise ValueError(
                            f"❌ Smoothed raw data not found in {gf.name}: {smoothed_raw_dset}"
                        )
                else:
                    # Original logic
                    dset_name_in = (
                        gf.DSET_RGB_CORR if gf.use_corrected else gf.DSET_RGB_MAIN
                    )
                    if dset_name_in not in f_in:
                        dset_name_in = gf.DSET_RGB_MAIN

                T_file, S_file, B_file = f_in[dset_name_in].shape

            # File is now closed - safe to open for writing
            # Process in chunks to limit memory
            chunk_size = 500  # Process 500 tracks at a time

            # Create output dataset with retry logic (Windows file locking can be sticky)
            max_retries = 3
            retry_delay = 1.0

            for attempt in range(max_retries):
                try:
                    f_out = h5py.File(gf.path, "a")
                    break  # Success!
                except BlockingIOError as e:
                    if attempt < max_retries - 1:
                        print(
                            f"      ⏳ File locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        import time

                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise RuntimeError(
                            f"❌ Could not open {gf.name} after {max_retries} attempts. "
                            "File may be open in another notebook or process. "
                            "Close other notebooks using this file and try again."
                        ) from e

            # Now we have the file open
            with f_out:
                if dset_name_out in f_out:
                    del f_out[dset_name_out]

                dset_out = f_out.create_dataset(
                    dset_name_out,
                    shape=(T_file, S_file, B_file),
                    dtype=np.float32,
                    chunks=(min(100, T_file), S_file, B_file),
                    compression="gzip",
                    compression_opts=1,
                )

                # Add metadata
                dset_out.attrs["window_size"] = (
                    window_size if window_size is not None else -1
                )
                dset_out.attrs["strength"] = strength
                dset_out.attrs["correction_method"] = (
                    "rolling_v2" if not use_global else "global_v2"
                )

                # Find track offset for this file
                t_offset = 0
                for gf_prev in self.geofiles:
                    if gf_prev == gf:
                        break
                    with h5py.File(gf_prev.path, "r") as f_tmp:
                        # NEW: Use smoothed raw data if requested
                        if use_smoothed_input and smoothed_raw_dset:
                            if smoothed_raw_dset in f_tmp:
                                dset_tmp = smoothed_raw_dset
                            else:
                                raise ValueError(
                                    f"❌ Smoothed raw data not found in {gf_prev.name}: {smoothed_raw_dset}"
                                )
                        else:
                            # Original logic
                            dset_tmp = (
                                gf_prev.DSET_RGB_CORR
                                if gf_prev.use_corrected
                                else gf_prev.DSET_RGB_MAIN
                            )
                            if dset_tmp not in f_tmp:
                                dset_tmp = gf_prev.DSET_RGB_MAIN
                        t_offset += f_tmp[dset_tmp].shape[0]

                # Reopen file for reading to get input data
                with h5py.File(gf.path, "r") as f_in:
                    # Process in chunks
                    for chunk_start in range(0, T_file, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, T_file)
                        chunk_data = f_in[dset_name_in][
                            chunk_start:chunk_end, :, :
                        ].astype(np.float32)

                        # Apply correction
                        for s in range(S_file):
                            for b in range(B_file):
                                ts = chunk_data[:, s, b]

                                if use_global:
                                    ref = ref_values[s, b]
                                    corrected = ts / ref
                                else:
                                    # Extract relevant portion of ref_vec
                                    ref_vec = ref_values[(s, b)][
                                        t_offset + chunk_start : t_offset + chunk_end
                                    ]
                                    corrected = ts / ref_vec

                                chunk_data[:, s, b] = (
                                    1 - strength
                                ) * ts + strength * corrected

                        # Write chunk to disk
                        dset_out[chunk_start:chunk_end, :, :] = chunk_data
                        del chunk_data

        # Clear reference values
        del ref_values

        print(f"✅ Illumination correction saved to disk ({mode_txt})")
        print(f"   Loading corrected data into memory...")

        # Load the corrected data (uses existing load function)
        return self.load_illumination_correction(window_size, strength)

    def _get_correction_dataset_name(
        self,
        window_size=1000,
        strength=1.0,
        use_smoothed_input=False,
        smooth_method=None,
        smooth_params=None,
    ):
        """
        Generate dataset name for illumination correction with specific parameters.
        Allows multiple cached versions with different parameters.

        NEW: Includes smoothing parameters in name if use_smoothed_input=True.
        """
        # Convert window_size to string (None -> 'global')
        w_str = "global" if window_size is None else str(int(window_size))

        # Base name (removed strength since always 1.0, removed "illum" as requested)
        base_name = f"processed/radiance/dataCube_corrected_w{w_str}"

        # Add smoothing suffix if applicable
        if use_smoothed_input and smooth_method and smooth_params:
            if smooth_method == "gaussian":
                sigma = int(smooth_params.get("gaussian_sigma", 2))
                return f"{base_name}_smooth_gaussian_sig{sigma}"

            elif smooth_method == "moving_average":
                window = int(smooth_params.get("wavelength_smoothing", 10))
                return f"{base_name}_smooth_movavg_w{window}"

            elif smooth_method == "savgol":
                window = int(smooth_params.get("wavelength_smoothing", 10))
                poly = int(smooth_params.get("savgol_polyorder", 2))
                return f"{base_name}_smooth_savgol_w{window}_p{poly}"

            elif smooth_method == "median":
                kernel = int(smooth_params.get("median_kernel_size", 5))
                return f"{base_name}_smooth_median_k{kernel}"

            elif smooth_method == "whittaker":
                lam = smooth_params.get("whittaker_lambda", 1e4)
                lam_str = f"{lam:.0e}".replace("+", "")
                return f"{base_name}_smooth_whittaker_lam{lam_str}"

            elif smooth_method == "bilateral":
                ss = int(smooth_params.get("bilateral_sigma_spatial", 2))
                si_str = f"{smooth_params.get('bilateral_sigma_intensity', 0.1):.2f}".replace(
                    ".", "p"
                )
                return f"{base_name}_smooth_bilateral_ss{ss}_si{si_str}"

        return base_name

    def has_illumination_correction(
        self,
        window_size=1000,
        strength=1.0,
        use_smoothed_input=False,
        smooth_method=None,
        smooth_params=None,
    ):
        """
        Check if illumination correction with these parameters has already been applied.
        Returns True if all files have the corrected dataset with matching metadata.
        """
        import h5py

        dset_name = self._get_correction_dataset_name(
            window_size, strength, use_smoothed_input, smooth_method, smooth_params
        )

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

    def load_illumination_correction(
        self,
        window_size=1000,
        strength=1.0,
        use_smoothed_input=False,
        smooth_method=None,
        smooth_params=None,
    ):
        """
        Load previously saved illumination-corrected data from HDF5 files.
        Populates self.data_corrected.
        """
        import h5py

        dset_name = self._get_correction_dataset_name(
            window_size, strength, use_smoothed_input, smooth_method, smooth_params
        )

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
        legacy_found = False

        for gf in self.geofiles:
            file_corrections = []
            file_legacy = []

            with h5py.File(gf.path, "r") as f:
                # Look for all datasets matching the pattern
                if "processed/radiance" in f:
                    for key in f["processed/radiance"].keys():
                        # Check for NEW format
                        if key.startswith("dataCube_corrected_"):
                            dset = f[f"processed/radiance/{key}"]

                            # Extract metadata
                            window = dset.attrs.get("window_size", "unknown")
                            if window == -1:
                                window = "global"

                            # Get size info
                            shape = dset.shape
                            size_mb = dset.nbytes / (1024 * 1024)

                            file_corrections.append(
                                {
                                    "dataset": key,
                                    "window": window,
                                    "shape": shape,
                                    "size_mb": size_mb,
                                }
                            )

                        # Check for OLD/LEGACY format
                        elif key.startswith("dataCube_illum_corrected_"):
                            dset = f[f"processed/radiance/{key}"]

                            # Extract metadata
                            window = dset.attrs.get("window_size", "unknown")
                            if window == -1:
                                window = "global"
                            strength = dset.attrs.get("strength", "unknown")

                            # Get size info
                            shape = dset.shape
                            size_mb = dset.nbytes / (1024 * 1024)

                            file_legacy.append(
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
                    print(f"   ✓ window={corr['window']}")
                    print(f"     Dataset: {corr['dataset']}")
                    print(
                        f"     Shape: {corr['shape']}, Size: {corr['size_mb']:.1f} MB"
                    )

            if file_legacy:
                legacy_found = True
                if not file_corrections:  # Only print filename if not already printed
                    print(f"\n📁 File: {gf.name}")
                print(
                    f"   ⚠️  LEGACY FORMAT (use cube.rename_legacy_corrections() to migrate):"
                )
                for corr in file_legacy:
                    print(
                        f"      window={corr['window']}, strength={corr['strength']:.1f}"
                    )
                    print(f"      Dataset: {corr['dataset']}")
                    print(
                        f"      Shape: {corr['shape']}, Size: {corr['size_mb']:.1f} MB"
                    )

        if not corrections_found and not legacy_found:
            print("\n   No cached corrections found in any files.")
        elif legacy_found:
            print(
                "\n💡 Tip: Run cube.rename_legacy_corrections() to convert old format to new format"
            )

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

    def rename_legacy_corrections(self, confirm=True):
        """
        Rename old 'dataCube_illum_corrected_*' datasets to new 'dataCube_corrected_*' format.

        This is a one-time utility to migrate old cached corrections to the new naming scheme.
        Old format: dataCube_illum_corrected_w{window}_s{strength}
        New format: dataCube_corrected_w{window} (strength removed since always 1.0)

        Parameters:
        -----------
        confirm : bool, default=True
            If True, asks for confirmation before renaming.

        Returns:
        --------
        int : Number of datasets renamed
        """
        import h5py

        print("\n🔄 Searching for legacy correction datasets...")
        print("=" * 80)

        rename_plan = []  # List of (file, old_path, new_path) tuples

        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                if "processed/radiance" in f:
                    for key in f["processed/radiance"].keys():
                        if key.startswith("dataCube_illum_corrected_"):
                            # Parse old name to extract parameters
                            # Format: dataCube_illum_corrected_w{window}_s{strength}
                            parts = key.split("_")

                            # Find window size
                            window_str = None
                            for part in parts:
                                if part.startswith("w"):
                                    window_str = part[1:]  # Remove 'w' prefix
                                    break

                            if window_str:
                                # Generate new name (no strength, no "illum")
                                new_key = f"dataCube_corrected_w{window_str}"
                                old_path = f"processed/radiance/{key}"
                                new_path = f"processed/radiance/{new_key}"

                                # Check if new name already exists
                                if new_key in f["processed/radiance"]:
                                    print(f"⚠️  File: {gf.name}")
                                    print(f"   Old: {key}")
                                    print(
                                        f"   New: {new_key} (ALREADY EXISTS - skipping)"
                                    )
                                else:
                                    rename_plan.append(
                                        (
                                            gf.path,
                                            gf.name,
                                            old_path,
                                            new_path,
                                            key,
                                            new_key,
                                        )
                                    )

        if not rename_plan:
            print("\n✅ No legacy corrections found (or all already renamed)")
            print("=" * 80)
            return 0

        # Show rename plan
        print(f"\n📋 Found {len(rename_plan)} dataset(s) to rename:\n")
        for _, fname, _, _, old_key, new_key in rename_plan:
            print(f"📁 File: {fname}")
            print(f"   Old: {old_key}")
            print(f"   New: {new_key}")
            print()

        if confirm:
            response = input("Proceed with renaming? (yes/no): ").strip().lower()
            if response not in ["yes", "y"]:
                print("❌ Renaming cancelled.")
                return 0

        # Perform renames
        print("\n🔄 Renaming datasets...")
        renamed_count = 0

        for fpath, fname, old_path, new_path, old_key, new_key in rename_plan:
            try:
                with h5py.File(fpath, "a") as f:
                    # Copy dataset to new location (HDF5 doesn't support direct rename)
                    f.copy(old_path, new_path)

                    # Delete old dataset
                    del f[old_path]

                    print(f"   ✓ Renamed in {fname}")
                    renamed_count += 1
            except Exception as e:
                print(f"   ❌ Error renaming in {fname}: {e}")

        print(f"\n✅ Successfully renamed {renamed_count} dataset(s)")
        print("=" * 80)
        return renamed_count

    # ============================================================================
    # 🌊 Beer-Lambert Depth Estimation
    # ============================================================================

    def compute_beer_lambert_depth(
        self,
        track_start=None,
        track_end=None,
        window_size=500,
        wavelength_range=(500, 650),
        quiet=False,
    ):
        """
        Compute relative seafloor depth (Δz) using Beer-Lambert attenuation law.

        This method estimates local height variations from pseudo-reflectance by modeling
        underwater light attenuation. Bright pixels indicate shallow/elevated areas (bombs/mounds),
        dark pixels indicate deeper areas (pits).

        Physics:
        --------
        Double-pass Beer-Lambert attenuation: L(λ) = L0(λ) * exp(-2 * c(λ) * z)
        Relative depth: Δz(x,y,λ) = - (1 / (2 * c(λ))) * ln(L(x,y,λ) / L_ref(x,y,λ))

        where:
          - L(x,y,λ): pseudo-reflectance per pixel (illumination-corrected data)
          - L_ref(x,y,λ): local reference intensity (rolling mean baseline)
          - c(λ): spectral attenuation coefficient [m^-1] (Holbach 2025; Løvås 2023)
          - Δz: relative depth in meters

        Parameters
        ----------
        track_start : int, optional
            Start track index (inclusive). If None, uses beginning of data.
        track_end : int, optional
            End track index (exclusive). If None, uses end of data.
        window_size : int, default=500
            Rolling mean window size (pixels) for computing local reference intensity.
            Default matches illumination correction window.
        wavelength_range : tuple, default=(500, 650)
            Wavelength range (nm) for averaging depth maps. This "optical window"
            has stable attenuation and good signal. Default: (500, 650) nm.
        quiet : bool, default=False
            If True, suppress progress messages.

        Returns
        -------
        None
            Stores results in:
            - self.depth_map : 2D array (tracks, pixels) of mean relative depth [meters]
            - self.depth_spectral : 3D array (tracks, pixels, bands) of per-wavelength depth [meters]
            - self.depth_wavelengths : 1D array of wavelengths used for depth computation
            - self.depth_attenuation : 1D array of c(λ) values used

        Raises
        ------
        RuntimeError
            If illumination-corrected data is not available (must run apply_illumination_correction_v2 first)
            If invalid/NaN pixels are found in the data

        Notes
        -----
        - Must run after apply_illumination_correction_v2()
        - Typical Δz range: 0 to 0.5 meters
        - Positive Δz = elevated (shallow), negative Δz = depressed (deep)
        - Use plot_georef(depth_overlay=True) to visualize

        References
        ----------
        - Holbach, A. et al. (2025). Coastal Case 2 water optics
        - Løvås, H. (2023). UHI attenuation measurements, Mjøsa fjord

        Example
        -------
        >>> cube.apply_illumination_correction_v2(window_size=500)
        >>> cube.compute_beer_lambert_depth(
        ...     track_start=config.UHI_TRACK_RANGE[0],
        ...     track_end=config.UHI_TRACK_RANGE[1],
        ...     window_size=500
        ... )
        >>> cube.plot_georef(depth_overlay=True, coordinate_system='NED')
        """
        from scipy.ndimage import uniform_filter1d

        if not quiet:
            print("🌊 Computing Beer-Lambert relative depth estimation...")
            print("=" * 80)

        # Check that illumination correction has been applied
        if self.data_corrected is None:
            raise RuntimeError(
                "❌ ERROR: Illumination-corrected data not found!\\n"
                "   Beer-Lambert depth estimation requires illumination correction.\\n"
                "   Please run: cube.apply_illumination_correction_v2()\\n"
                "   before computing depth."
            )

        # Determine track range
        T_total = self.data_corrected.shape[0]
        if track_start is None:
            track_start = 0
        if track_end is None:
            track_end = T_total

        # Extract region of interest
        data_region = self.data_corrected[track_start:track_end, :, :]  # (T, S, B)
        T, S, B = data_region.shape

        if not quiet:
            print(f"Track range: {track_start} to {track_end} ({T} tracks)")
            print(f"Spatial pixels: {S}")
            print(f"Spectral bands: {B}")

        # Define attenuation coefficients (Coastal Case 2 water, Holbach 2025; Løvås 2023)
        attenuation_wavelengths = np.array([400, 450, 500, 520, 550, 600, 650, 700])
        attenuation_coefficients = np.array(
            [0.58, 0.50, 0.38, 0.34, 0.32, 0.35, 0.42, 0.55]
        )

        # Interpolate attenuation coefficients to match data wavelengths
        c_lambda = np.interp(
            self.wavelengths, attenuation_wavelengths, attenuation_coefficients
        )

        if not quiet:
            print(f"\\nAttenuation coefficients c(λ):")
            print(
                f"   Interpolated from {len(attenuation_wavelengths)} reference points"
            )
            print(f"   Range: {c_lambda.min():.3f} to {c_lambda.max():.3f} m⁻¹")
            print(
                f"   Mean (500-650 nm): {c_lambda[(self.wavelengths >= 500) & (self.wavelengths <= 650)].mean():.3f} ± {c_lambda[(self.wavelengths >= 500) & (self.wavelengths <= 650)].std():.3f} m⁻¹"
            )

        # Plot attenuation curve
        if not quiet:
            import matplotlib.pyplot as plt

            fig_atten, ax_atten = plt.subplots(1, 1, figsize=(8, 4))
            ax_atten.plot(
                attenuation_wavelengths,
                attenuation_coefficients,
                "ro",
                markersize=10,
                label="Reference data (Holbach 2025; Løvås 2023)",
            )
            ax_atten.plot(
                self.wavelengths, c_lambda, "b-", linewidth=2, label="Interpolated c(λ)"
            )
            ax_atten.axvspan(
                wavelength_range[0],
                wavelength_range[1],
                alpha=0.2,
                color="green",
                label="Analysis range",
            )
            ax_atten.set_xlabel("Wavelength [nm]", fontsize=12, fontweight="bold")
            ax_atten.set_ylabel(
                "Attenuation coefficient c(λ) [m⁻¹]", fontsize=12, fontweight="bold"
            )
            ax_atten.set_title(
                "Beer-Lambert Attenuation Coefficients (Coastal Case 2 Water)",
                fontsize=13,
                fontweight="bold",
            )
            ax_atten.legend(loc="best", fontsize=10)
            ax_atten.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            print(f"✅ Attenuation curve plotted")

        # Check for invalid pixels
        if np.any(np.isnan(data_region)) or np.any(np.isinf(data_region)):
            n_invalid = np.sum(np.isnan(data_region) | np.isinf(data_region))
            raise RuntimeError(
                f"❌ ERROR: Found {n_invalid} invalid (NaN/Inf) pixels in illumination-corrected data!\\n"
                f"   All pixels must be valid for Beer-Lambert depth estimation.\\n"
                f"   Please check your illumination correction step."
            )

        # Check for non-positive values (would cause ln() to fail)
        if np.any(data_region <= 0):
            n_nonpositive = np.sum(data_region <= 0)
            print(
                f"⚠️  WARNING: Found {n_nonpositive} non-positive pixels ({100*n_nonpositive/data_region.size:.3f}%)"
            )
            print(f"   Setting to small positive value (1e-10) to avoid ln() errors")
            data_region = np.maximum(data_region, 1e-10)

        if not quiet:
            print(
                f"\\n🔄 Computing local reference intensity (rolling mean, window={window_size} px)..."
            )

        # Compute local reference intensity using rolling mean along track (per slit-band)
        # Shape: (T, S, B)
        L_ref = np.zeros_like(data_region)

        for s in range(S):
            for b in range(B):
                # Rolling mean along track direction
                L_ref[:, s, b] = uniform_filter1d(
                    data_region[:, s, b], size=window_size, mode="nearest"
                )

        if not quiet:
            print(f"✅ Local reference computed")

        # Avoid division by zero
        L_ref = np.maximum(L_ref, 1e-10)

        if not quiet:
            print(f"\\n🧮 Computing per-wavelength relative depth Δz(λ)...")

        # Compute relative depth per wavelength
        # Beer-Lambert: L = L0 * exp(-2*c*z)
        # Solving: Δz = -ln(L/L_ref) / (2*c)
        # BUT we want POSITIVE Δz for elevated (brighter), so flip sign:
        # Δz = +ln(L/L_ref) / (2*c)
        # Result: Brighter → positive Δz (elevated), Darker → negative Δz (deep)
        ratio = data_region / L_ref
        ratio = np.clip(ratio, 1e-10, None)  # Avoid log(0)

        ln_ratio = np.log(ratio)

        # Broadcast c_lambda to match shape (T, S, B)
        c_lambda_broadcast = c_lambda[np.newaxis, np.newaxis, :]  # (1, 1, B)

        # Compute depth with FLIPPED sign (so brighter = positive = elevated)
        depth_spectral = (1.0 / (2.0 * c_lambda_broadcast)) * ln_ratio  # (T, S, B)

        if not quiet:
            print(f"✅ Per-wavelength depth computed")
            print(f"   Shape: {depth_spectral.shape}")
            print(
                f"   Range: {depth_spectral.min():.3f} to {depth_spectral.max():.3f} m"
            )

        # Select wavelengths in the analysis range
        wl_mask = (self.wavelengths >= wavelength_range[0]) & (
            self.wavelengths <= wavelength_range[1]
        )
        n_bands_used = np.sum(wl_mask)

        if n_bands_used == 0:
            raise RuntimeError(
                f"❌ ERROR: No wavelengths found in range {wavelength_range} nm!\\n"
                f"   Available range: {self.wavelengths.min():.1f} to {self.wavelengths.max():.1f} nm"
            )

        if not quiet:
            print(
                f"\\n📊 Averaging depth across wavelength range {wavelength_range[0]}-{wavelength_range[1]} nm..."
            )
            print(f"   Using {n_bands_used} bands")

        # Compute mean depth across selected wavelengths
        depth_mean = np.mean(depth_spectral[:, :, wl_mask], axis=2)  # (T, S)

        if not quiet:
            print(f"✅ Mean depth computed")
            print(f"   Shape: {depth_mean.shape}")
            print(f"   Mean: {depth_mean.mean():.3f} m")
            print(f"   Std:  {depth_mean.std():.3f} m")
            print(f"   Min:  {depth_mean.min():.3f} m (deepest/darkest)")
            print(f"   Max:  {depth_mean.max():.3f} m (shallowest/brightest)")

        # Store results
        self.depth_map = depth_mean
        self.depth_spectral = depth_spectral
        self.depth_wavelengths = self.wavelengths[wl_mask]
        self.depth_attenuation = c_lambda
        self.depth_track_start = track_start
        self.depth_track_end = track_end

        if not quiet:
            print(f"\\n✅ Beer-Lambert depth estimation complete!")
            print(f"   Results stored in:")
            print(f"   - cube.depth_map (2D array, mean depth)")
            print(f"   - cube.depth_spectral (3D array, per-wavelength depth)")
            print(f"   - cube.depth_wavelengths (wavelengths used)")
            print(f"   - cube.depth_attenuation (c(λ) coefficients)")
            print(f"\\n💡 Visualize with: cube.plot_georef(depth_overlay=True)")
            print("=" * 80)

    # ============================================================================
    # � Spectral Smoothing Cache Functions
    # ============================================================================

    def _get_smoothing_dataset_name(
        self,
        method="gaussian",
        wavelength_smoothing=10,
        gaussian_sigma=2.0,
        savgol_polyorder=2,
        median_kernel_size=5,
        whittaker_lambda=1e4,
        bilateral_sigma_spatial=2.0,
        bilateral_sigma_intensity=0.1,
        apply_to_raw=False,  # NEW: Distinguish raw vs corrected data smoothing
    ):
        """
        Generate dataset name for spectral smoothing with specific parameters.
        Allows multiple cached versions with different parameters.
        """
        # Build parameter string based on method
        if method == "gaussian":
            params = f"sig{gaussian_sigma:.1f}".replace(".", "p")
        elif method == "savgol":
            params = f"w{wavelength_smoothing}_p{savgol_polyorder}"
        elif method == "moving_average":
            params = f"w{wavelength_smoothing}"
        elif method == "median":
            params = f"k{median_kernel_size}"
        elif method == "whittaker":
            # Use scientific notation for lambda (e.g., 1e4 -> "1e4")
            lam_str = f"{whittaker_lambda:.0e}".replace("+", "")
            params = f"lam{lam_str}"
        elif method == "bilateral":
            ss_str = f"{bilateral_sigma_spatial:.1f}".replace(".", "p")
            si_str = f"{bilateral_sigma_intensity:.2f}".replace(".", "p")
            params = f"ss{ss_str}_si{si_str}"
        else:
            params = "unknown"

        # NEW: Different cache names for raw vs corrected data
        data_type = "raw" if apply_to_raw else "corrected"
        return f"processed/radiance/dataCube_{data_type}_smoothed_{method}_{params}"

    def has_spectral_smoothing(
        self,
        method="gaussian",
        wavelength_smoothing=10,
        gaussian_sigma=2.0,
        savgol_polyorder=2,
        median_kernel_size=5,
        whittaker_lambda=1e4,
        bilateral_sigma_spatial=2.0,
        bilateral_sigma_intensity=0.1,
        apply_to_raw=False,  # NEW: Check for raw or corrected data cache
    ):
        """
        Check if spectral smoothing with these parameters has already been applied.
        Returns True if all files have the smoothed dataset with matching metadata.
        """
        import h5py

        dset_name = self._get_smoothing_dataset_name(
            method=method,
            wavelength_smoothing=wavelength_smoothing,
            gaussian_sigma=gaussian_sigma,
            savgol_polyorder=savgol_polyorder,
            median_kernel_size=median_kernel_size,
            whittaker_lambda=whittaker_lambda,
            bilateral_sigma_spatial=bilateral_sigma_spatial,
            bilateral_sigma_intensity=bilateral_sigma_intensity,
            apply_to_raw=apply_to_raw,  # NEW: Pass through
        )

        for gf in self.geofiles:
            try:
                with h5py.File(gf.path, "r") as f:
                    # Check if smoothed dataset exists
                    if dset_name not in f:
                        return False

                    # Verify metadata matches
                    dset = f[dset_name]
                    if "method" not in dset.attrs:
                        return False

                    saved_method = dset.attrs["method"]
                    if saved_method != method:
                        return False

                    # Check method-specific parameters
                    if method == "gaussian" and "gaussian_sigma" in dset.attrs:
                        if abs(dset.attrs["gaussian_sigma"] - gaussian_sigma) > 1e-6:
                            return False
                    elif method == "savgol":
                        if (
                            dset.attrs.get("wavelength_smoothing", -1)
                            != wavelength_smoothing
                            or dset.attrs.get("savgol_polyorder", -1)
                            != savgol_polyorder
                        ):
                            return False
                    elif method == "moving_average":
                        if (
                            dset.attrs.get("wavelength_smoothing", -1)
                            != wavelength_smoothing
                        ):
                            return False
                    elif method == "median":
                        if (
                            dset.attrs.get("median_kernel_size", -1)
                            != median_kernel_size
                        ):
                            return False
                    elif method == "whittaker":
                        if (
                            abs(
                                dset.attrs.get("whittaker_lambda", 0) - whittaker_lambda
                            )
                            > 1e-3
                        ):
                            return False
                    elif method == "bilateral":
                        if (
                            abs(
                                dset.attrs.get("bilateral_sigma_spatial", 0)
                                - bilateral_sigma_spatial
                            )
                            > 1e-6
                            or abs(
                                dset.attrs.get("bilateral_sigma_intensity", 0)
                                - bilateral_sigma_intensity
                            )
                            > 1e-6
                        ):
                            return False

            except Exception as e:
                print(f"⚠️  Error checking smoothing cache in {gf.name}: {e}")
                return False

        return True

    def load_spectral_smoothing(
        self,
        method="gaussian",
        wavelength_smoothing=10,
        gaussian_sigma=2.0,
        savgol_polyorder=2,
        median_kernel_size=5,
        whittaker_lambda=1e4,
        bilateral_sigma_spatial=2.0,
        bilateral_sigma_intensity=0.1,
        apply_to_raw=False,  # NEW: Load raw or corrected data cache
    ):
        """
        Load previously saved spectral smoothing from HDF5 files.
        Populates self.data_corrected (or self.data if apply_to_raw=True).
        """
        import h5py

        dset_name = self._get_smoothing_dataset_name(
            method=method,
            wavelength_smoothing=wavelength_smoothing,
            gaussian_sigma=gaussian_sigma,
            savgol_polyorder=savgol_polyorder,
            median_kernel_size=median_kernel_size,
            whittaker_lambda=whittaker_lambda,
            bilateral_sigma_spatial=bilateral_sigma_spatial,
            bilateral_sigma_intensity=bilateral_sigma_intensity,
            apply_to_raw=apply_to_raw,  # NEW: Pass through
        )

        # Calculate total size
        T_total = sum(gf.shape[0] for gf in self.geofiles)
        S, B = self.geofiles[0].shape[1], self.geofiles[0].shape[2]

        data_type = "raw" if apply_to_raw else "corrected"
        print(
            f"📂 Loading saved spectral smoothing ({method}, {data_type} data) from {len(self.geofiles)} files..."
        )

        smoothed_data = np.zeros((T_total, S, B), dtype=np.float32)

        t_offset = 0
        for gf in self.geofiles:
            with h5py.File(gf.path, "r") as f:
                if dset_name not in f:
                    raise RuntimeError(
                        f"{gf.name}: smoothed data not found at {dset_name}"
                    )

                dset = f[dset_name]
                T_file = dset.shape[0]
                smoothed_data[t_offset : t_offset + T_file, :, :] = dset[()]

                # Print metadata
                if "method" in dset.attrs:
                    print(f"   {gf.name}: method={dset.attrs['method']}")

                t_offset += T_file

        # Assign to correct attribute
        if apply_to_raw:
            self.data = smoothed_data
        else:
            self.data_corrected = smoothed_data

        print(f"✅ Loaded spectral smoothing from disk (saved computation time!)")
        return smoothed_data

    def save_spectral_smoothing(
        self,
        method="gaussian",
        wavelength_smoothing=10,
        gaussian_sigma=2.0,
        savgol_polyorder=2,
        median_kernel_size=5,
        whittaker_lambda=1e4,
        bilateral_sigma_spatial=2.0,
        bilateral_sigma_intensity=0.1,
        apply_to_raw=False,  # NEW: Save raw or corrected data
    ):
        """
        Save the spectrally smoothed data to HDF5 files.
        Splits self.data_corrected (or self.data if apply_to_raw=True) back into individual files
        and saves with unique names based on method and parameters.
        """
        import h5py
        import time

        # Determine which data to save
        data_to_save = self.data if apply_to_raw else self.data_corrected
        data_type = "raw" if apply_to_raw else "corrected"

        if data_to_save is None:
            raise RuntimeError(
                f"No {data_type} data to save. Run apply_spectral_smoothing first."
            )

        dset_path = self._get_smoothing_dataset_name(
            method=method,
            wavelength_smoothing=wavelength_smoothing,
            gaussian_sigma=gaussian_sigma,
            savgol_polyorder=savgol_polyorder,
            median_kernel_size=median_kernel_size,
            whittaker_lambda=whittaker_lambda,
            bilateral_sigma_spatial=bilateral_sigma_spatial,
            bilateral_sigma_intensity=bilateral_sigma_intensity,
            apply_to_raw=apply_to_raw,  # NEW: Pass through
        )

        print(
            f"💾 Saving spectral smoothing ({data_type} data) to {len(self.geofiles)} files..."
        )
        print(f"   Dataset: {dset_path}")

        t_offset = 0
        for gf in self.geofiles:
            T_file = gf.shape[0]
            smoothed_chunk = data_to_save[t_offset : t_offset + T_file, :, :]

            # Retry mechanism in case file is temporarily locked
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with h5py.File(gf.path, "a") as f:
                        # Remove old dataset if it exists (overwrite same parameters)
                        if dset_path in f:
                            del f[dset_path]

                        # Create new dataset with compression
                        dset = f.create_dataset(
                            dset_path,
                            data=smoothed_chunk,
                            compression="gzip",
                            compression_opts=4,
                            dtype=np.float32,
                        )

                        # Save metadata (method-specific)
                        dset.attrs["method"] = method
                        dset.attrs["description"] = (
                            f"Spectrally smoothed radiance ({method})"
                        )

                        if method == "gaussian":
                            dset.attrs["gaussian_sigma"] = gaussian_sigma
                        elif method == "savgol":
                            dset.attrs["wavelength_smoothing"] = wavelength_smoothing
                            dset.attrs["savgol_polyorder"] = savgol_polyorder
                        elif method == "moving_average":
                            dset.attrs["wavelength_smoothing"] = wavelength_smoothing
                        elif method == "median":
                            dset.attrs["median_kernel_size"] = median_kernel_size
                        elif method == "whittaker":
                            dset.attrs["whittaker_lambda"] = whittaker_lambda
                        elif method == "bilateral":
                            dset.attrs["bilateral_sigma_spatial"] = (
                                bilateral_sigma_spatial
                            )
                            dset.attrs["bilateral_sigma_intensity"] = (
                                bilateral_sigma_intensity
                            )

                        print(f"   ✓ {gf.name}: saved {smoothed_chunk.shape}")
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

        print(f"✅ Spectral smoothing saved to disk")

    # ============================================================================
    # �🔥 NEW: Spectral Pre-processing Functions (In-Memory Only)
    # ============================================================================

    def apply_wavelength_interpolation(self, interpolate_wavelengths=None, quiet=False):
        """
        Interpolate bad wavelengths by replacing them with linear interpolation from neighbors.

        This permanently modifies data_corrected (in memory only, not saved to disk).
        Useful for fixing known bad wavelengths (e.g., sensor artifacts, atmospheric absorption).

        Parameters:
        -----------
        interpolate_wavelengths : list of int, optional
            List of wavelength INDICES to interpolate (e.g., [64, 107, 138]).
            Each index will be replaced with the average of its neighboring wavelengths.
            If None or empty, no interpolation is performed.
        quiet : bool, default=False
            If True, suppresses progress messages

        Returns:
        --------
        np.ndarray : The interpolated datacube (T, S, B)

        Example:
        --------
        >>> cube.apply_illumination_correction_v2()
        >>> cube.apply_wavelength_interpolation(interpolate_wavelengths=[64, 107, 138])
        >>> # Wavelengths at indices 64, 107, 138 are now interpolated from neighbors
        """
        if self.data_corrected is None:
            raise ValueError(
                "No corrected data available. Run apply_illumination_correction_v2() first."
            )

        if interpolate_wavelengths is None or len(interpolate_wavelengths) == 0:
            if not quiet:
                print("⚠️  No wavelengths to interpolate (list is empty)")
            return self.data_corrected

        if not quiet:
            print("=" * 60)
            print("🔧 WAVELENGTH INTERPOLATION")
            print("=" * 60)
            print(f"📊 Datacube shape: {self.data_corrected.shape}")
            print(f"🎯 Interpolating {len(interpolate_wavelengths)} wavelength(s)")

        T, S, B = self.data_corrected.shape

        # Validate indices
        valid_indices = []
        for idx in interpolate_wavelengths:
            if idx < 1 or idx >= B - 1:
                if not quiet:
                    print(f"   ⚠️  Skipping index {idx} (out of valid range [1, {B-2}])")
            else:
                valid_indices.append(idx)
                wl_value = (
                    self.wavelengths[idx] if hasattr(self, "wavelengths") else idx
                )
                if not quiet:
                    print(
                        f"   🔧 Index {idx} ({wl_value:.1f} nm) → interpolated from neighbors"
                    )

        if len(valid_indices) == 0:
            if not quiet:
                print("   ⚠️  No valid indices to interpolate")
            return self.data_corrected

        # Apply linear interpolation for each bad wavelength
        for idx in valid_indices:
            # Replace with average of left and right neighbors
            self.data_corrected[:, :, idx] = (
                self.data_corrected[:, :, idx - 1] + self.data_corrected[:, :, idx + 1]
            ) / 2.0

        if not quiet:
            print(f"\n✅ Interpolation complete!")
            print(f"   {len(valid_indices)} wavelength(s) interpolated")
            print("=" * 60)

        return self.data_corrected

    def apply_percentile_clipping(
        self, lower_percentile=5, upper_percentile=95, quiet=False
    ):
        """
        Clip extreme spectral values per-pixel using percentiles to remove spikes.

        This removes outlier spikes in each pixel's spectrum by clipping values outside
        the percentile range. Useful for removing sensor artifacts and high-frequency noise
        before smoothing, especially at wavelength edges where SNR is poor.

        **Use case:** You have huge spikes at <450nm and >700nm that contaminate smoothing
        in the good 450-700nm range. Run this BEFORE smoothing to clip those spikes.

        Parameters:
        -----------
        lower_percentile : float, default=5
            Lower percentile threshold (0-100). Values below this are clipped.
        upper_percentile : float, default=95
            Upper percentile threshold (0-100). Values above this are clipped.
        quiet : bool, default=False
            If True, suppresses progress messages

        Returns:
        --------
        np.ndarray : The clipped datacube (T, S, B)

        Example:
        --------
        >>> cube.apply_illumination_correction_v2()
        >>> cube.apply_percentile_clipping(lower_percentile=5, upper_percentile=95)
        >>> cube.apply_spectral_smoothing(wavelength_smoothing=20)
        >>> cube.apply_wavelength_filter(wavelength_range=(450, 700))
        """
        if self.data_corrected is None:
            raise ValueError(
                "No corrected data available. Run apply_illumination_correction_v2() first."
            )

        if not quiet:
            print("=" * 60)
            print("✂️ PERCENTILE CLIPPING (Per-Spectrum)")
            print("=" * 60)
            print(f"📊 Datacube shape: {self.data_corrected.shape}")
            print(
                f"📉 Clipping range: {lower_percentile}th - {upper_percentile}th percentile"
            )

        T, S, B = self.data_corrected.shape
        original_min = self.data_corrected.min()
        original_max = self.data_corrected.max()

        # Progress bar
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False

        # Process each spectrum (pixel) independently
        total_pixels = T * S
        if use_tqdm:
            pbar = tqdm(total=total_pixels, desc="   Clipping pixels", unit="pixel")

        for t in range(T):
            for s in range(S):
                spectrum = self.data_corrected[t, s, :]

                # Compute percentiles for this spectrum
                p_low = np.percentile(spectrum, lower_percentile)
                p_high = np.percentile(spectrum, upper_percentile)

                # Clip values outside percentile range
                self.data_corrected[t, s, :] = np.clip(spectrum, p_low, p_high)

                if use_tqdm:
                    pbar.update(1)

        if use_tqdm:
            pbar.close()

        clipped_min = self.data_corrected.min()
        clipped_max = self.data_corrected.max()

        if not quiet:
            print(
                f"\n📈 Value range before clipping: [{original_min:.4f}, {original_max:.4f}]"
            )
            print(
                f"📉 Value range after clipping:  [{clipped_min:.4f}, {clipped_max:.4f}]"
            )
            print(f"✅ Clipping complete!")
            print("=" * 60)

        return self.data_corrected

    def apply_edge_padding(
        self, good_wavelength_range=(450, 700), padding_method="constant", quiet=False
    ):
        """
        Pad wavelength bands outside the 'good' range with edge values to prevent
        smoothing contamination.

        **The Problem:** When you smooth with a large window (e.g., 20 bands), bad bands
        outside your good range (450-700nm) contaminate the edges inside the good range.

        **The Solution:** Replace bad bands (<450nm and >700nm) with the edge values from
        the good range BEFORE smoothing. This way smoothing doesn't see the volatile spikes.

        Parameters:
        -----------
        good_wavelength_range : tuple of (min_wl, max_wl)
            Wavelength range in nanometers that contains good data (e.g., (450, 700))
        padding_method : str, default="constant"
            How to pad the bad bands:
            - "constant": Replace with edge value (simplest, recommended)
            - "linear": Linear extrapolation from first/last N bands
        quiet : bool, default=False
            If True, suppresses progress messages

        Returns:
        --------
        np.ndarray : The padded datacube (T, S, B) with same shape

        Example:
        --------
        >>> cube.apply_illumination_correction_v2()
        >>> cube.apply_percentile_clipping(lower_percentile=5, upper_percentile=95)
        >>> cube.apply_edge_padding(good_wavelength_range=(450, 700))  # Pad bad edges
        >>> cube.apply_spectral_smoothing(wavelength_smoothing=20)     # Now safe to smooth
        >>> cube.apply_wavelength_filter(wavelength_range=(450, 700))  # Crop to good range
        """
        if self.data_corrected is None:
            raise ValueError(
                "No corrected data available. Run apply_illumination_correction_v2() first."
            )

        wl_min, wl_max = good_wavelength_range

        # Find indices of good wavelength range
        good_mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
        good_indices = np.where(good_mask)[0]

        if len(good_indices) == 0:
            raise ValueError(f"No wavelengths found in good range {wl_min}-{wl_max} nm")

        first_good_idx = good_indices[0]
        last_good_idx = good_indices[-1]

        if not quiet:
            print("=" * 60)
            print(f"🛡️ EDGE PADDING: {padding_method.upper()}")
            print("=" * 60)
            print(f"📊 Datacube shape: {self.data_corrected.shape}")
            print(
                f"🌊 Wavelength range: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm"
            )
            print(
                f"✅ Good range: {wl_min}-{wl_max} nm (indices {first_good_idx}-{last_good_idx})"
            )
            print(
                f"⚠️  Bad bands: {first_good_idx} before, {len(self.wavelengths) - last_good_idx - 1} after"
            )

        T, S, B = self.data_corrected.shape

        if padding_method == "constant":
            # Replace bad bands with edge values (simplest)
            for t in range(T):
                for s in range(S):
                    # Pad bands before good range with first good value
                    if first_good_idx > 0:
                        edge_value_low = self.data_corrected[t, s, first_good_idx]
                        self.data_corrected[t, s, :first_good_idx] = edge_value_low

                    # Pad bands after good range with last good value
                    if last_good_idx < B - 1:
                        edge_value_high = self.data_corrected[t, s, last_good_idx]
                        self.data_corrected[t, s, last_good_idx + 1 :] = edge_value_high

        elif padding_method == "linear":
            # Linear extrapolation from first/last 5 good bands
            n_extrap_bands = min(5, len(good_indices) // 4)  # Use first/last 5 bands

            for t in range(T):
                for s in range(S):
                    spectrum = self.data_corrected[t, s, :]

                    # Extrapolate lower edge
                    if first_good_idx > 0 and n_extrap_bands >= 2:
                        x_fit = self.wavelengths[
                            first_good_idx : first_good_idx + n_extrap_bands
                        ]
                        y_fit = spectrum[
                            first_good_idx : first_good_idx + n_extrap_bands
                        ]
                        coeffs = np.polyfit(x_fit, y_fit, 1)  # Linear fit

                        # Extrapolate to lower wavelengths
                        x_extrap = self.wavelengths[:first_good_idx]
                        y_extrap = np.polyval(coeffs, x_extrap)
                        self.data_corrected[t, s, :first_good_idx] = y_extrap

                    # Extrapolate upper edge
                    if last_good_idx < B - 1 and n_extrap_bands >= 2:
                        x_fit = self.wavelengths[
                            last_good_idx - n_extrap_bands + 1 : last_good_idx + 1
                        ]
                        y_fit = spectrum[
                            last_good_idx - n_extrap_bands + 1 : last_good_idx + 1
                        ]
                        coeffs = np.polyfit(x_fit, y_fit, 1)  # Linear fit

                        # Extrapolate to upper wavelengths
                        x_extrap = self.wavelengths[last_good_idx + 1 :]
                        y_extrap = np.polyval(coeffs, x_extrap)
                        self.data_corrected[t, s, last_good_idx + 1 :] = y_extrap

        else:
            raise ValueError(
                f"Unknown padding_method: {padding_method}. Choose 'constant' or 'linear'."
            )

        if not quiet:
            print(f"\n✅ Edge padding complete ({padding_method} method)")
            print("=" * 60)

        return self.data_corrected

    def apply_wavelength_filter(self, wavelength_range=(490, 680), quiet=False):
        """
        Filter datacube to only include specified wavelength range.

        This permanently modifies data_corrected (in memory only, not saved to disk)
        by cropping the wavelength dimension. All subsequent operations will only
        use the filtered wavelength range.

        Parameters:
        -----------
        wavelength_range : tuple of (min_wl, max_wl)
            Wavelength range in nanometers to keep (e.g., (490, 680))
        quiet : bool, default=False
            If True, suppresses progress messages

        Returns:
        --------
        np.ndarray : The filtered datacube (T, S, filtered_B)

        Example:
        --------
        >>> cube.apply_illumination_correction_v2()
        >>> cube.apply_wavelength_filter(wavelength_range=(490, 680))
        >>> # Now data_corrected only contains wavelengths 490-680 nm
        >>> cube.train_svm_with_cv(...)  # No wavelength_range parameter needed!
        """
        if self.data_corrected is None:
            raise ValueError(
                "No corrected data available. Run apply_illumination_correction_v2() first."
            )

        wl_min, wl_max = wavelength_range
        wl_mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
        wl_indices = np.where(wl_mask)[0]

        if len(wl_indices) == 0:
            raise ValueError(f"No wavelengths found in range {wl_min}-{wl_max} nm")

        if not quiet:
            print("=" * 60)
            print("🔪 WAVELENGTH FILTERING")
            print("=" * 60)
            print(f"📊 Original wavelengths: {len(self.wavelengths)}")
            print(f"🎯 Target range: {wl_min}-{wl_max} nm")
            print(f"✂️  Filtered wavelengths: {len(wl_indices)}")
            print(
                f"📉 Range: {self.wavelengths[wl_indices[0]]:.1f} - {self.wavelengths[wl_indices[-1]]:.1f} nm"
            )

        # Crop datacube to wavelength range
        self.data_corrected = self.data_corrected[:, :, wl_indices].copy()

        # Update wavelengths array
        self.wavelengths = self.wavelengths[wl_indices].copy()

        if not quiet:
            print(f"\n✅ Datacube shape after filtering: {self.data_corrected.shape}")
            print(f"   Memory: {self.data_corrected.nbytes / (1024**3):.2f} GB")
            print("=" * 60)

        return self.data_corrected

    def apply_spectral_normalization(self, method="l2", quiet=False):
        """
        Normalize each pixel's spectrum using specified method.

        This permanently modifies data_corrected (in memory only, not saved to disk).
        Normalization is applied to ALL wavelengths currently in the datacube.
        Use apply_wavelength_filter() first if you want to normalize only a specific range.

        Parameters:
        -----------
        method : str
            Normalization method:
            - "l2": L2 vector normalization (unit length) ⭐ Recommended for SVM
            - "mean": Divide by mean (simple scaling)
            - "mean_center": Subtract mean (removes DC offset)
            - "snv": Standard Normal Variate (removes offset + scale)
            - "msc": Multiplicative Scatter Correction (uses global mean as reference)
            - "minmax": Min-max scaling to [0,1]
        quiet : bool, default=False
            If True, suppresses progress messages

        Returns:
        --------
        np.ndarray : The normalized datacube (T, S, B)

        Example:
        --------
        >>> cube.apply_illumination_correction_v2()
        >>> cube.apply_wavelength_filter(wavelength_range=(490, 680))  # Optional: filter first
        >>> cube.apply_spectral_normalization(method="l2")  # Normalize
        >>> cube.train_svm_with_cv(...)  # Train on normalized spectra
        """
        if self.data_corrected is None:
            raise ValueError(
                "No corrected data available. Run apply_illumination_correction_v2() first."
            )

        if not quiet:
            print("=" * 60)
            print(f"📐 SPECTRAL NORMALIZATION: {method.upper()}")
            print("=" * 60)
            print(f"📊 Datacube shape: {self.data_corrected.shape}")
            print(
                f"🌊 Wavelengths: {len(self.wavelengths)} ({self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm)"
            )

        T, S, B = self.data_corrected.shape

        # Reshape to (n_pixels, n_wavelengths) for easier processing
        spectra = self.data_corrected.reshape(-1, B)

        if method == "l2":
            # L2 normalization: scale each spectrum to unit length
            norms = np.linalg.norm(spectra, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            spectra_normalized = spectra / norms

        elif method == "mean":
            # Divide by mean
            means = np.mean(spectra, axis=1, keepdims=True)
            means[means == 0] = 1
            spectra_normalized = spectra / means

        elif method == "mean_center":
            # Subtract mean (removes DC offset)
            means = np.mean(spectra, axis=1, keepdims=True)
            spectra_normalized = spectra - means

        elif method == "snv":
            # Standard Normal Variate: (x - mean) / std
            means = np.mean(spectra, axis=1, keepdims=True)
            stds = np.std(spectra, axis=1, keepdims=True)
            stds[stds == 0] = 1  # Avoid division by zero
            spectra_normalized = (spectra - means) / stds

        elif method == "msc":
            # Multiplicative Scatter Correction: use global mean as reference
            reference = np.mean(spectra, axis=0, keepdims=True)  # Mean spectrum
            means = np.mean(spectra, axis=1, keepdims=True)
            stds = np.std(spectra, axis=1, keepdims=True)
            stds[stds == 0] = 1
            ref_mean = np.mean(reference)
            ref_std = np.std(reference)
            spectra_normalized = (spectra - means) * (ref_std / stds) + ref_mean

        elif method == "minmax":
            # Min-max scaling to [0, 1]
            mins = np.min(spectra, axis=1, keepdims=True)
            maxs = np.max(spectra, axis=1, keepdims=True)
            ranges = maxs - mins
            ranges[ranges == 0] = 1
            spectra_normalized = (spectra - mins) / ranges

        else:
            raise ValueError(
                f"Unknown normalization method: {method}. "
                f"Choose from: l2, mean, mean_center, snv, msc, minmax"
            )

        # Reshape back to (T, S, B)
        self.data_corrected = spectra_normalized.reshape(T, S, B).astype(np.float32)

        if not quiet:
            print(f"\n✅ Normalization complete!")
            print(f"   Method: {method}")
            print(
                f"   Value range: [{self.data_corrected.min():.4f}, {self.data_corrected.max():.4f}]"
            )
            print("=" * 60)

        return self.data_corrected

    def apply_spectral_smoothing(
        self,
        wavelength_smoothing=10,
        method="gaussian",
        savgol_polyorder=2,
        gaussian_sigma=2.0,
        median_kernel_size=5,
        whittaker_lambda=1e4,
        bilateral_sigma_spatial=2.0,
        bilateral_sigma_intensity=0.1,
        force_recompute=False,
        quiet=False,
        apply_to_raw=False,  # NEW: Apply to raw data instead of corrected data
    ):
        """
        Smooth each pixel's spectrum along the wavelength axis.

        This permanently modifies data_corrected (or self.data if apply_to_raw=True) in memory.
        Smoothing is applied to ALL wavelengths currently in the datacube.

        Results are automatically cached to disk. Use force_recompute=True to recalculate.

        Parameters:
        -----------
        wavelength_smoothing : int
            Window size for smoothing (used for moving_average and savgol methods)
            Larger values = more smoothing
        method : str, default="gaussian"
            Smoothing method:
            - "gaussian": Gaussian filter (sigma-based smoothing, good for noise reduction)
            - "savgol": Savitzky-Golay filter (preserves peaks/valleys)
            - "moving_average": Simple rolling mean
            - "median": Median filter (robust to outliers)
            - "whittaker": Whittaker smoother (excellent feature preservation)
            - "bilateral": Bilateral filter (edge-preserving)
        savgol_polyorder : int, default=2
            Polynomial order for Savitzky-Golay filter (2 or 3)
            Only used if method="savgol"
        gaussian_sigma : float, default=2.0
            Standard deviation for Gaussian filter
            Only used if method="gaussian". Larger values = more smoothing
        median_kernel_size : int, default=5
            Kernel size for median filter (must be odd)
            Only used if method="median". Larger values = more smoothing
        whittaker_lambda : float, default=1e4
            Smoothing parameter for Whittaker smoother (range: 1e2 to 1e6)
            Only used if method="whittaker". Larger values = more smoothing
        bilateral_sigma_spatial : float, default=2.0
            Spatial sigma for bilateral filter
            Only used if method="bilateral". Larger values = more smoothing
        bilateral_sigma_intensity : float, default=0.1
            Intensity sigma for bilateral filter
            Only used if method="bilateral". Smaller values = stronger edge preservation
        force_recompute : bool, default=False
            If True, recalculate smoothing even if cached version exists
            If False, load from cache if available (much faster!)
        quiet : bool, default=False
            If True, suppresses progress messages
        apply_to_raw : bool, default=False
            NEW: If True, applies smoothing to raw data (self.data) instead of corrected data
            This is useful for smoothing BEFORE illumination correction.
            When True, saves to 'data_raw_smooth_...' cache.
            When False (default), works on data_corrected as before.

        Returns:
        --------
        np.ndarray : The smoothed datacube (T, S, B)

        Example:
        --------
        # Standard workflow (smooth AFTER illumination):
        >>> cube.apply_illumination_correction_v2()
        >>> cube.apply_spectral_smoothing(method="gaussian", gaussian_sigma=2.0)

        # NEW: Smooth BEFORE illumination:
        >>> cube.apply_spectral_smoothing(method="gaussian", gaussian_sigma=5.0, apply_to_raw=True)
        >>> cube.apply_illumination_correction_v2(use_smoothed_raw=True, smooth_method="gaussian", smooth_sigma=5.0)
        """
        # Determine which data to work on
        if apply_to_raw:
            if self.data is None:
                raise ValueError("No raw data available. Load data first.")
            target_data = self.data
            data_type = "raw"
        else:
            if self.data_corrected is None:
                raise ValueError(
                    "No corrected data available. Run apply_illumination_correction_v2() first."
                )
            target_data = self.data_corrected
            data_type = "corrected"

        # Check if smoothing has already been computed and saved
        if not force_recompute and self.has_spectral_smoothing(
            method=method,
            wavelength_smoothing=wavelength_smoothing,
            gaussian_sigma=gaussian_sigma,
            savgol_polyorder=savgol_polyorder,
            median_kernel_size=median_kernel_size,
            whittaker_lambda=whittaker_lambda,
            bilateral_sigma_spatial=bilateral_sigma_spatial,
            bilateral_sigma_intensity=bilateral_sigma_intensity,
            apply_to_raw=apply_to_raw,  # NEW: Check correct cache
        ):
            if not quiet:
                cache_type = "raw" if apply_to_raw else "corrected"
                print(
                    f"💾 Found cached smoothing ({method}, {cache_type} data) - loading from disk (much faster!)"
                )
            return self.load_spectral_smoothing(
                method=method,
                wavelength_smoothing=wavelength_smoothing,
                gaussian_sigma=gaussian_sigma,
                savgol_polyorder=savgol_polyorder,
                median_kernel_size=median_kernel_size,
                whittaker_lambda=whittaker_lambda,
                bilateral_sigma_spatial=bilateral_sigma_spatial,
                bilateral_sigma_intensity=bilateral_sigma_intensity,
                apply_to_raw=apply_to_raw,  # NEW: Load from correct cache
            )

        # Only check wavelength_smoothing for methods that use it
        if method in ["moving_average", "savgol"] and wavelength_smoothing <= 1:
            if not quiet:
                print("⚠️  wavelength_smoothing <= 1, no smoothing applied")
            return self.data_corrected

        if not quiet:
            print("=" * 60)
            print(
                f"🌊 SPECTRAL SMOOTHING: {method.upper()} ({'RAW' if apply_to_raw else 'CORRECTED'} data)"
            )
            print("=" * 60)
            print(f"📊 Datacube shape: {target_data.shape}")
            if method in ["moving_average", "savgol"]:
                print(f"🪟 Window size: {wavelength_smoothing}")
            if method == "savgol":
                print(f"📐 Polynomial order: {savgol_polyorder}")
            elif method == "gaussian":
                print(f"📐 Gaussian sigma: {gaussian_sigma}")
            elif method == "median":
                print(f"📐 Median kernel size: {median_kernel_size}")
            elif method == "whittaker":
                print(f"📐 Whittaker lambda: {whittaker_lambda}")
            elif method == "bilateral":
                print(
                    f"📐 Bilateral sigma (spatial/intensity): {bilateral_sigma_spatial}/{bilateral_sigma_intensity}"
                )

        T, S, B = target_data.shape

        if method == "savgol":
            from scipy.signal import savgol_filter

            # Ensure window size is odd
            if wavelength_smoothing % 2 == 0:
                wavelength_smoothing += 1
                if not quiet:
                    print(
                        f"   ⚠️  Adjusted window to odd number: {wavelength_smoothing}"
                    )

            # Ensure polyorder < window_size
            if savgol_polyorder >= wavelength_smoothing:
                savgol_polyorder = wavelength_smoothing - 1
                if not quiet:
                    print(f"   ⚠️  Adjusted polyorder to: {savgol_polyorder}")

            # Apply savgol filter along wavelength axis
            smoothed_data = savgol_filter(
                target_data,
                window_length=wavelength_smoothing,
                polyorder=savgol_polyorder,
                axis=2,  # Wavelength axis
            ).astype(np.float32)

        elif method == "moving_average":
            import pandas as pd
            from tqdm import tqdm

            # Apply rolling mean along wavelength axis for each pixel
            smoothed_data = np.zeros_like(target_data)

            # Progress bar over temporal dimension
            for t in tqdm(
                range(T), desc="Smoothing spectra (moving average)", disable=quiet
            ):
                for s in range(S):
                    spectrum = pd.Series(target_data[t, s, :])
                    smoothed_spectrum = spectrum.rolling(
                        window=wavelength_smoothing, center=True, min_periods=1
                    ).mean()
                    smoothed_data[t, s, :] = smoothed_spectrum.values

            smoothed_data = smoothed_data.astype(np.float32)

        elif method == "gaussian":
            from scipy.ndimage import gaussian_filter1d
            from tqdm import tqdm

            # Apply Gaussian filter along wavelength axis
            smoothed_data = np.zeros_like(target_data)

            # Progress bar over temporal dimension
            for t in tqdm(range(T), desc="Smoothing spectra (gaussian)", disable=quiet):
                for s in range(S):
                    spectrum = target_data[t, s, :]
                    smoothed_data[t, s, :] = gaussian_filter1d(
                        spectrum, sigma=gaussian_sigma, mode="nearest"
                    )

            smoothed_data = smoothed_data.astype(np.float32)

        elif method == "median":
            from scipy.ndimage import median_filter
            from tqdm import tqdm

            # Ensure kernel size is odd
            kernel_size = median_kernel_size
            if kernel_size % 2 == 0:
                kernel_size += 1
                if not quiet:
                    print(f"   ⚠️  Adjusted kernel size to odd number: {kernel_size}")

            # Apply median filter along wavelength axis
            smoothed_data = np.zeros_like(target_data)

            for t in tqdm(range(T), desc="Smoothing spectra (median)", disable=quiet):
                for s in range(S):
                    spectrum = target_data[t, s, :]
                    smoothed_data[t, s, :] = median_filter(
                        spectrum, size=kernel_size, mode="nearest"
                    )

            smoothed_data = smoothed_data.astype(np.float32)

        elif method == "whittaker":
            from scipy.sparse import diags, eye
            from scipy.sparse.linalg import spsolve
            from tqdm import tqdm

            # Apply Whittaker smoother along wavelength axis
            smoothed_data = np.zeros_like(target_data)

            # Pre-compute the Whittaker matrix once (same for all spectra)
            m = B
            E = eye(m, format="csc")
            D = diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m))
            W = E + whittaker_lambda * (D.T @ D)

            for t in tqdm(
                range(T), desc="Smoothing spectra (whittaker)", disable=quiet
            ):
                for s in range(S):
                    spectrum = target_data[t, s, :]
                    try:
                        smoothed_data[t, s, :] = spsolve(W, spectrum)
                    except Exception:
                        # If Whittaker fails, keep original spectrum
                        smoothed_data[t, s, :] = spectrum

            smoothed_data = smoothed_data.astype(np.float32)

        elif method == "bilateral":
            from tqdm import tqdm

            # Apply bilateral filter along wavelength axis
            smoothed_data = np.zeros_like(target_data)

            for t in tqdm(
                range(T), desc="Smoothing spectra (bilateral)", disable=quiet
            ):
                for s in range(S):
                    spectrum = target_data[t, s, :]

                    # Bilateral filter
                    for i in range(B):
                        half_window = int(3 * bilateral_sigma_spatial)
                        start = max(0, i - half_window)
                        end = min(B, i + half_window + 1)

                        spatial_dist = np.arange(start, end) - i
                        spatial_weight = np.exp(
                            -(spatial_dist**2) / (2 * bilateral_sigma_spatial**2)
                        )

                        intensity_diff = spectrum[start:end] - spectrum[i]
                        intensity_weight = np.exp(
                            -(intensity_diff**2) / (2 * bilateral_sigma_intensity**2)
                        )

                        combined_weight = spatial_weight * intensity_weight
                        combined_weight /= np.sum(combined_weight)

                        smoothed_data[t, s, i] = np.sum(
                            spectrum[start:end] * combined_weight
                        )

            smoothed_data = smoothed_data.astype(np.float32)

        else:
            raise ValueError(
                f"Unknown smoothing method: {method}. "
                f"Choose from: gaussian, savgol, moving_average, median, whittaker, bilateral"
            )

        # Assign smoothed data to the correct attribute
        if apply_to_raw:
            self.data = smoothed_data
            if not quiet:
                print(f"\n✅ Raw data smoothing complete! (self.data updated)")
        else:
            self.data_corrected = smoothed_data
            if not quiet:
                print(
                    f"\n✅ Corrected data smoothing complete! (self.data_corrected updated)"
                )

        if not quiet:
            print(f"   Method: {method}")
            if method == "savgol":
                print(
                    f"   Window: {wavelength_smoothing}, Polyorder: {savgol_polyorder}"
                )
            elif method == "moving_average":
                print(f"   Window: {wavelength_smoothing}")
            elif method == "gaussian":
                print(f"   Sigma: {gaussian_sigma}")
            elif method == "median":
                print(
                    f"   Kernel size: {median_kernel_size if median_kernel_size % 2 == 1 else median_kernel_size + 1}"
                )
            elif method == "whittaker":
                print(f"   Lambda: {whittaker_lambda}")
            elif method == "bilateral":
                print(
                    f"   Sigma (spatial/intensity): {bilateral_sigma_spatial}/{bilateral_sigma_intensity}"
                )
            print("=" * 60)

        # Save smoothed data to disk for future use
        if not quiet:
            cache_type = "raw" if apply_to_raw else "corrected"
            print(f"\n💾 Caching smoothed {cache_type} data to disk for future use...")
        self.save_spectral_smoothing(
            method=method,
            wavelength_smoothing=wavelength_smoothing,
            gaussian_sigma=gaussian_sigma,
            savgol_polyorder=savgol_polyorder,
            median_kernel_size=median_kernel_size,
            whittaker_lambda=whittaker_lambda,
            bilateral_sigma_spatial=bilateral_sigma_spatial,
            bilateral_sigma_intensity=bilateral_sigma_intensity,
            apply_to_raw=apply_to_raw,  # NEW: Save to correct cache
        )

        # Return the smoothed data (either raw or corrected)
        return smoothed_data

    def save_as_pseudo_reflectance(
        self,
        interpolate_wavelengths=None,
        wavelength_smoothing=1,
        smoothing_method="moving_average",  # NEW: Smoothing method ('moving_average' or 'savgol')
        savgol_polyorder=2,  # NEW: Polynomial order for Savitzky-Golay filter
        normalize_method=None,  # NEW: Normalization method (None, 'mean', 'mean_center', 'snv', 'msc', 'minmax', 'l2')
        track_start=None,  # NEW: Start track index for segment extraction
        track_end=None,  # NEW: End track index for segment extraction
        overwrite=False,
    ):
        """
        Save pseudo-reflectance datacube to HDF5 files.

        Pseudo-reflectance = illumination corrected + bad wavelength interpolation + smoothing

        Processing pipeline:
        1. Start with illumination-corrected data (self.data_corrected)
        2. Extract segment (if track_start/track_end specified)
        3. Interpolate bad wavelengths (linear interpolation from neighbors)
        4. Apply wavelength smoothing (rolling window)
        5. Apply normalization (if specified, per-pixel)
        6. Save to HDF5 as 'dataCube_pseudo_reflectance' or 'dataCube_normalized_pseudo_reflectance'

        Parameters:
        -----------
        interpolate_wavelengths : list of int, optional
            List of wavelength indices to interpolate (replace with linear interpolation).
            Example: [64, 107, 138] will interpolate these band indices.

        wavelength_smoothing : int, default=1
            Window size for wavelength smoothing.
            1 = no smoothing, larger values = more smoothing.
            Applied along wavelength axis for each pixel independently.

        smoothing_method : str, default='moving_average'
            Smoothing method to use:
            - "moving_average": Rolling mean (pandas-based, consistent with plot_spectrum)
            - "savgol": Savitzky-Golay filter (preserves spectral features better)

        savgol_polyorder : int, default=2
            Polynomial order for Savitzky-Golay filter (only used if smoothing_method='savgol').
            Typical values: 2 (quadratic) or 3 (cubic). Must be less than wavelength_smoothing.

        normalize_method : str or None, default=None
            Normalization method to apply to each pixel's spectrum:
            - None: No normalization (saves as pseudo-reflectance)
            - "mean": Divide by mean (simple scaling)
            - "mean_center": Subtract mean (removes DC offset) ⭐ Good for comparison
            - "snv": Standard Normal Variate (removes offset + scale) ⭐ Good for ML
            - "msc": Multiplicative Scatter Correction (uses global mean as reference)
            - "minmax": Min-max scaling to [0,1]
            - "l2": L2 vector normalization
            When set, saves to 'dataCube_normalized_pseudo_reflectance' instead.

        track_start : int, optional
            Starting track index for segment extraction (inclusive).
            If None, starts from beginning (index 0).

        track_end : int, optional
            Ending track index for segment extraction (exclusive).
            If None, goes to end of data.

        overwrite : bool, default=False
            If True, overwrites existing pseudo-reflectance data.
            If False and data exists, skips saving.

        Returns:
        --------
        np.ndarray : The pseudo-reflectance datacube (T, S, B)

        Example:
        --------
        >>> cube.apply_illumination_correction_v2()
        >>> # Save entire datacube
        >>> cube.save_as_pseudo_reflectance(
        ...     interpolate_wavelengths=[64, 107, 138],  # Bad bands
        ...     wavelength_smoothing=5,  # Smooth over 5 wavelengths
        ...     overwrite=False
        ... )
        >>> # Save only a segment
        >>> cube.save_as_pseudo_reflectance(
        ...     interpolate_wavelengths=[64, 107, 138],
        ...     wavelength_smoothing=10,
        ...     track_start=100,  # Start at track 100
        ...     track_end=500,    # End at track 500
        ...     overwrite=True
        ... )
        >>> # Save normalized datacube for ML/classification
        >>> cube.save_as_pseudo_reflectance(
        ...     interpolate_wavelengths=[64, 107, 138],
        ...     wavelength_smoothing=10,
        ...     normalize_method="mean_center",  # or "snv" for ML
        ...     track_start=100,
        ...     track_end=500,
        ...     overwrite=True
        ... )
        """
        import h5py
        import pandas as pd

        # Check if corrected data exists
        if not hasattr(self, "data_corrected") or self.data_corrected is None:
            raise RuntimeError(
                "❌ No illumination-corrected data found. "
                "Run cube.apply_illumination_correction_v2() first."
            )

        # Determine dataset path based on normalization
        if normalize_method:
            dset_path = "processed/radiance/dataCube_normalized_pseudo_reflectance"
        else:
            dset_path = "processed/radiance/dataCube_pseudo_reflectance"

        # Check if already exists
        if not overwrite:
            already_exists = True
            for gf in self.geofiles:
                try:
                    with h5py.File(gf.path, "r") as f:
                        if dset_path not in f:
                            already_exists = False
                            break
                except Exception:
                    already_exists = False
                    break

            if already_exists:
                print(f"✅ Pseudo-reflectance already exists in all files")
                print(f"   Use overwrite=True to recalculate")
                return self.data_corrected  # Return existing corrected data

        print("=" * 90)
        print("🔄 CREATING PSEUDO-REFLECTANCE DATACUBE")
        print("=" * 90)

        # Handle segment extraction
        T_total, S, B = self.data_corrected.shape
        if track_start is None:
            track_start = 0
        if track_end is None:
            track_end = T_total

        # Validate track range
        if track_start < 0 or track_end > T_total or track_start >= track_end:
            raise ValueError(
                f"Invalid track range: track_start={track_start}, track_end={track_end}. "
                f"Valid range is [0, {T_total})"
            )

        # Print pipeline info
        pipeline_steps = []
        if track_start > 0 or track_end < T_total:
            pipeline_steps.append("Segment Extraction")
        else:
            pipeline_steps.append("Illumination Correction")
        pipeline_steps.extend(["Interpolation", "Smoothing"])
        if normalize_method:
            pipeline_steps.append("Normalization")
        pipeline_steps.append("Save")

        print(f"   Pipeline: {' → '.join(pipeline_steps)}")
        if track_start > 0 or track_end < T_total:
            print(
                f"   📏 Segment: tracks {track_start} to {track_end} (length={track_end - track_start})"
            )
        print()

        # Print parameters
        if interpolate_wavelengths and len(interpolate_wavelengths) > 0:
            print(f"   🔧 Interpolating {len(interpolate_wavelengths)} wavelength(s):")
            for wl_idx in interpolate_wavelengths:
                if 0 <= wl_idx < len(self.wavelengths):
                    wl_nm = self.wavelengths[wl_idx]
                    print(f"      • Index {wl_idx}: {wl_nm:.2f} nm")
                else:
                    print(
                        f"      ⚠️  Index {wl_idx} out of range (0-{len(self.wavelengths)-1})"
                    )
        else:
            print(f"   • No wavelength interpolation")

        if wavelength_smoothing > 1:
            if smoothing_method == "savgol":
                print(
                    f"   🔧 Wavelength smoothing: {smoothing_method} (window={wavelength_smoothing}, polyorder={savgol_polyorder})"
                )
            else:
                print(
                    f"   🔧 Wavelength smoothing: {smoothing_method} (window={wavelength_smoothing})"
                )
        else:
            print(f"   • No wavelength smoothing")

        if normalize_method:
            print(f"   🔧 Normalization: {normalize_method}")
        else:
            print(f"   • No normalization")

        print()

        # Start with corrected data and extract segment
        pseudo_reflectance = self.data_corrected[track_start:track_end, :, :].copy()
        T, S, B = pseudo_reflectance.shape

        # Define normalization function (per-pixel)
        def normalize_spectrum_pixel(spectrum, method):
            """Apply normalization to a single pixel's spectrum."""
            if method is None or len(spectrum) == 0:
                return spectrum

            spectrum = np.array(spectrum, dtype=float)

            if method == "mean":
                spectrum_mean = np.mean(spectrum)
                if spectrum_mean != 0:
                    return spectrum / spectrum_mean
                return spectrum

            elif method == "mean_center":
                return spectrum - np.mean(spectrum)

            elif method == "snv":
                spectrum_mean = np.mean(spectrum)
                spectrum_std = np.std(spectrum)
                if spectrum_std > 0:
                    return (spectrum - spectrum_mean) / spectrum_std
                return spectrum - spectrum_mean

            elif method == "minmax":
                spectrum_min = np.min(spectrum)
                spectrum_max = np.max(spectrum)
                if spectrum_max > spectrum_min:
                    return (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
                return spectrum

            elif method == "l2":
                norm = np.linalg.norm(spectrum)
                if norm > 0:
                    return spectrum / norm
                return spectrum

            else:
                return spectrum

        # Step 1: Interpolate bad wavelengths
        if interpolate_wavelengths and len(interpolate_wavelengths) > 0:
            print(f"   Step 1/2: Interpolating bad wavelengths...")

            valid_indices = [
                idx for idx in interpolate_wavelengths if 0 < idx < B - 1
            ]  # Can't interpolate edges

            if len(valid_indices) < len(interpolate_wavelengths):
                skipped = len(interpolate_wavelengths) - len(valid_indices)
                print(f"      ⚠️  Skipping {skipped} edge wavelength(s)")

            for idx in valid_indices:
                # Linear interpolation from neighbors
                pseudo_reflectance[:, :, idx] = (
                    pseudo_reflectance[:, :, idx - 1]
                    + pseudo_reflectance[:, :, idx + 1]
                ) / 2.0

            print(f"      ✓ Interpolated {len(valid_indices)} wavelength(s)")
        else:
            print(f"   Step 1/2: Skipped (no interpolation)")

        # Step 2: Wavelength smoothing
        if wavelength_smoothing > 1:
            method_label = "SG" if smoothing_method == "savgol" else "MA"
            if smoothing_method == "savgol":
                print(
                    f"   Step 2/2: Smoothing wavelengths ({method_label}, window={wavelength_smoothing}, polyorder={savgol_polyorder})..."
                )
            else:
                print(
                    f"   Step 2/2: Smoothing wavelengths ({method_label}, window={wavelength_smoothing})..."
                )

            # Smooth along wavelength axis for each pixel
            smoothed = np.zeros_like(pseudo_reflectance)

            if smoothing_method == "savgol":
                # Savitzky-Golay filter
                from scipy.signal import savgol_filter

                # Ensure window_size is odd (required for savgol)
                window = wavelength_smoothing
                if window % 2 == 0:
                    window += 1

                # Ensure polyorder < window_size
                polyorder = savgol_polyorder
                if polyorder >= window:
                    polyorder = window - 1

                for t in range(T):
                    for s in range(S):
                        spectrum = pseudo_reflectance[t, s, :]
                        # Apply Savitzky-Golay with polynomial extrapolation at edges
                        smoothed[t, s, :] = savgol_filter(
                            spectrum, window, polyorder, mode="interp"
                        )

            else:
                # Moving average (default)
                for t in range(T):
                    for s in range(S):
                        spectrum = pseudo_reflectance[t, s, :]

                        # Use pandas rolling for consistent behavior with plot_spectrum
                        series = pd.Series(spectrum)
                        smoothed_series = series.rolling(
                            window=wavelength_smoothing, center=True, min_periods=1
                        ).mean()

                        smoothed[t, s, :] = smoothed_series.values

            pseudo_reflectance = smoothed
            print(f"      ✓ Smoothing complete")
        else:
            print(f"   Step 2/2: Skipped (no smoothing)")

        # Step 3: Normalization (per-pixel)
        if normalize_method:
            print(f"   Step 3/3: Normalizing spectra ({normalize_method})...")

            # MSC requires reference spectrum (global mean)
            if normalize_method == "msc":
                print(f"      Computing reference spectrum...")
                reference_spectrum = np.nanmean(pseudo_reflectance, axis=(0, 1))

                normalized = np.zeros_like(pseudo_reflectance)
                for t in range(T):
                    for s in range(S):
                        spectrum = pseudo_reflectance[t, s, :]
                        # MSC: fit linear model and correct
                        coeffs = np.polyfit(reference_spectrum, spectrum, 1)
                        b, a = coeffs[0], coeffs[1]
                        if abs(b) > 1e-10:
                            normalized[t, s, :] = (spectrum - a) / b
                        else:
                            normalized[t, s, :] = spectrum
                pseudo_reflectance = normalized
            else:
                # Other normalization methods (per-pixel)
                normalized = np.zeros_like(pseudo_reflectance)
                for t in range(T):
                    for s in range(S):
                        spectrum = pseudo_reflectance[t, s, :]
                        normalized[t, s, :] = normalize_spectrum_pixel(
                            spectrum, normalize_method
                        )
                pseudo_reflectance = normalized

            print(f"      ✓ Normalization complete")
        else:
            print(f"   Step 3/3: Skipped (no normalization)")

        print()
        print(f"   💾 Saving to HDF5 files...")

        # Map segment indices to files
        # Build cumulative track offsets for each file
        file_offsets = [0]
        for gf in self.geofiles:
            file_offsets.append(file_offsets[-1] + gf.shape[0])

        # Save to relevant file(s)
        saved_count = 0
        segment_offset = 0  # Offset within the pseudo_reflectance array

        for i, gf in enumerate(self.geofiles):
            file_start = file_offsets[i]
            file_end = file_offsets[i + 1]

            # Check if this file overlaps with the segment
            if file_end <= track_start or file_start >= track_end:
                # No overlap, skip this file
                continue

            # Calculate the overlap region
            overlap_start = (
                max(track_start, file_start) - file_start
            )  # Relative to file
            overlap_end = min(track_end, file_end) - file_start  # Relative to file
            overlap_length = overlap_end - overlap_start

            # Extract the corresponding chunk from pseudo_reflectance
            data_chunk = pseudo_reflectance[
                segment_offset : segment_offset + overlap_length, :, :
            ]
            segment_offset += overlap_length

            try:
                with h5py.File(gf.path, "a") as f:
                    # Remove old dataset if overwriting
                    if dset_path in f:
                        del f[dset_path]

                    # Create new dataset
                    dset = f.create_dataset(
                        dset_path,
                        data=data_chunk,
                        compression="gzip",
                        compression_opts=4,
                        dtype=np.float32,
                    )

                    # Save metadata
                    if normalize_method:
                        dset.attrs["description"] = (
                            "Normalized pseudo-reflectance (illumination corrected + interpolated + smoothed + normalized)"
                        )
                        dset.attrs["normalization_method"] = normalize_method
                    else:
                        dset.attrs["description"] = (
                            "Pseudo-reflectance (illumination corrected + interpolated + smoothed)"
                        )
                    dset.attrs["interpolate_wavelengths"] = (
                        str(interpolate_wavelengths)
                        if interpolate_wavelengths
                        else "None"
                    )
                    dset.attrs["wavelength_smoothing"] = wavelength_smoothing
                    dset.attrs["smoothing_method"] = smoothing_method
                    dset.attrs["track_start"] = track_start
                    dset.attrs["track_end"] = track_end
                    dset.attrs["segment_length"] = track_end - track_start
                    if smoothing_method == "savgol":
                        dset.attrs["savgol_polyorder"] = savgol_polyorder

                    print(
                        f"      ✓ {gf.name}: {data_chunk.shape} (tracks {overlap_start} to {overlap_end})"
                    )
                    saved_count += 1

            except Exception as e:
                print(f"      ❌ Failed to save {gf.name}: {e}")
                raise

        print()
        print("=" * 90)
        print(f"✅ PSEUDO-REFLECTANCE SAVED")
        print(f"   Dataset: {dset_path}")
        print(f"   Shape: {pseudo_reflectance.shape}")
        print(f"   Files saved: {saved_count}/{len(self.geofiles)}")
        if track_start > 0 or track_end < T_total:
            print(f"   Segment: tracks {track_start} to {track_end}")
        print("=" * 90)

        return pseudo_reflectance

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
        # red_wl=654.2,
        # green_wl=560,
        # blue_wl=440.3,
        red_wl=620.0,
        green_wl=565.0,
        blue_wl=490.0,
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
        roi_overlay_mode="markers",  # "markers" (scatter plot) or "solid" (overlay colored pixels on RGB)
        roi_solid_pixel_size=1,  # Size of each pixel in solid overlay mode (must be odd: 1, 3, 5, 7, etc.)
        roi_marker_size=100,
        roi_marker_shape="s",  # Marker shape: 's'=square, 'o'=circle, '^'=triangle, 'D'=diamond, etc.
        roi_marker_edgewidth=0,  # Edge width for ROI markers (0 = no edge)
        roi_show_numbers=False,
        roi_legend_loc="best",  # Legend location: 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center', 'outside', or None to hide
        roi_legend_markersize=10,  # Size of color markers in legend (default=10)
        roi_legend_marker_border=True,  # Whether to show black border on legend markers
        roi_legend_show_counts=True,  # Whether to show pixel counts in legend labels (e.g., "bombs (2369)")
        line_colors=["red", "blue", "orange", "magenta"],
        line_width=2,
        line_style="-",
        line_labels=None,
        flip_axes=False,  # NEW: If True, swap tracks/slits (tracks become vertical)
        flip_horizontal=False,  # NEW: If True, invert horizontal axis direction
        flip_vertical=False,  # NEW: If True, invert vertical axis direction
        crop_center_track=None,  # NEW: Track index for crop center (requires crop_center_slit and crop_width)
        crop_center_slit=None,  # NEW: Slit index for crop center (requires crop_center_track and crop_width)
        crop_width=None,  # NEW: Width in slit pixels for square physical crop (track width = crop_width/3.61)
        crop_aspect_ratio=3.61,  # NEW: Ratio of track:slit pixel physical size for CROP DIMENSIONS (default 3.61 for UHI)
        display_aspect_ratio=1.0,  # NEW: Display stretch ratio (1.0=square pixels, 3.61=physical spacing, only for crops)
        derivative_order=0,  # NEW: 0 (raw), 1 (first derivative), 2 (second derivative)
        derivative_window=2,  # NEW: Window size for derivative computation
        use_wavelength_colormap=False,  # NEW: If True, plot single wavelength with natural color
        wavelength_colormap_target=None,  # NEW: Target wavelength (nm) for single-wavelength mode (required if use_wavelength_colormap=True)
        # vmin=None,  # NEW: Minimum value for normalization (if None, uses 2nd percentile)
        # vmax=None,  # NEW: Maximum value for normalization (if None, uses 98th percentile)
        vmin=(0.53, 0.58, 0.2),
        vmax=(1.35, 1.37, 1.73),
        colorbar_fraction=0.046,  # NEW: Fraction of axes width for colorbar (default 0.046, use 0.15 for larger, None for auto full-height)
        colorbar_orientation="vertical",  # NEW: Orientation: "vertical" or "horizontal"
        colorbar_pad=0.04,  # Padding between plot and colorbar
        colorbar_shrink=1.0,  # Shrink factor for colorbar length (1.0 = full length)
    ):
        """
        Enhanced RGB plot supporting multiple named ROIs with different colors.

        NEW SINGLE-WAVELENGTH COLORMAP MODE:
        - use_wavelength_colormap: Enable single-wavelength visualization with natural colors
        - wavelength_colormap_target: Target wavelength (nm) to visualize
        - When enabled, plots single wavelength heatmap (white → wavelength color)
        - All other features (derivatives, cropping, flipping, ROIs) still work
        - RGB parameters (red_wl, green_wl, blue_wl) are ignored in this mode

        NEW MULTI-ROI FEATURES:
        - roi_collection: Dict of named ROIs or 'all' to use cube.roi_collection
        - Automatic color assignment per ROI
        - Legend shows ROI names and pixel counts

        NEW AXIS CONTROL FEATURES:
        - flip_axes: Swap tracks/slits axes (default: tracks=horizontal, slits=vertical)
        - flip_horizontal: Invert horizontal axis direction (right-to-left instead of left-to-right)
        - flip_vertical: Invert vertical axis direction (top-to-bottom instead of bottom-to-top)

        NEW CROPPING FEATURES:
        - crop_center_track: GLOBAL track index at center of crop (for CombinedTransectCube, same indices as shown in full transect plots)
        - crop_center_slit: GLOBAL slit index at center of crop (0 to n_slits-1, typically 0-1023 for UHI data)
        - crop_width: Width in slit pixels (track width adjusted by crop_aspect_ratio for square physical area)
        - crop_aspect_ratio: Ratio of track:slit pixel physical size for CROP DIMENSIONS (default 3.61 for UHI data)
          * Controls how many track vs slit pixels are extracted in the crop
          * track_pixels = crop_width / crop_aspect_ratio
        - display_aspect_ratio: Display stretch ratio for VISUALIZATION only (default 1.0 = square pixels)
          * Controls how stretched the image appears on screen
          * 1.0 = square pixels (equal spacing in both directions)
          * 3.61 = physical spacing (matches sensor's actual track:slit pixel spacing)
          * Independent of crop dimensions - only affects visual appearance
          * Only applies to crops, not full transects
        - All three crop parameters must be provided together
        - Cropping creates a square PHYSICAL area (not square in pixels due to aspect ratio)
        - Cropping is applied after all flip transformations
        - Axis labels show original coordinates of cropped region
        - Overlays (ROIs, boundaries, lines) are disabled when cropping is applied
        - Crop boundaries are clipped to image edges if they extend beyond

        NEW DERIVATIVE FEATURES:
        - derivative_order: 0 (raw intensity), 1 (first derivative), 2 (second derivative)
        - derivative_window: Window size for derivative computation (default=2)
        - Derivatives are computed along wavelength axis for each RGB channel
        - First derivative: dI/dλ - rate of spectral change
        - Second derivative: d²I/dλ² - curvature of spectral features
        - Derivative values are normalized to [0,1] per channel using min-max scaling
        """

        # Original RGB setup (unchanged)
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )

        # NEW: Single-wavelength colormap mode
        if use_wavelength_colormap:
            if wavelength_colormap_target is None:
                raise ValueError(
                    "wavelength_colormap_target must be specified when use_wavelength_colormap=True"
                )

            # Import the colormap creation function from ndi_analysis_utils
            from ..ndi_analysis_utils import create_wavelength_colormap

            # Find closest wavelength index
            wl_idx = np.argmin(np.abs(self.wavelengths - wavelength_colormap_target))
            actual_wl = self.wavelengths[wl_idx]

            print(f"🎨 Single-wavelength colormap mode: {actual_wl:.1f} nm")

            # Compute intensity based on derivative order
            if derivative_order == 0:
                # Raw intensity
                intensity_map = cube_data[:, :, wl_idx].T.copy()
                derivative_label = "Raw Intensity"
            elif derivative_order == 1:
                # First derivative: dI/dλ
                if (
                    wl_idx < derivative_window
                    or wl_idx >= len(self.wavelengths) - derivative_window
                ):
                    raise ValueError(
                        f"Wavelength {wavelength_colormap_target} nm too close to edge for derivative with window={derivative_window}"
                    )

                intensity_forward = cube_data[:, :, wl_idx + derivative_window]
                intensity_backward = cube_data[:, :, wl_idx - derivative_window]
                wl_forward = self.wavelengths[wl_idx + derivative_window]
                wl_backward = self.wavelengths[wl_idx - derivative_window]

                intensity_map = (
                    (intensity_forward - intensity_backward)
                    / (wl_forward - wl_backward)
                ).T.copy()
                derivative_label = "1st Derivative (dI/dλ)"
            elif derivative_order == 2:
                # Second derivative: d²I/dλ²
                if (
                    wl_idx < derivative_window
                    or wl_idx >= len(self.wavelengths) - derivative_window
                ):
                    raise ValueError(
                        f"Wavelength {wavelength_colormap_target} nm too close to edge for 2nd derivative with window={derivative_window}"
                    )

                intensity_center = cube_data[:, :, wl_idx]
                intensity_forward = cube_data[:, :, wl_idx + derivative_window]
                intensity_backward = cube_data[:, :, wl_idx - derivative_window]

                h = (
                    self.wavelengths[wl_idx + derivative_window]
                    - self.wavelengths[wl_idx]
                )
                intensity_map = (
                    (intensity_forward - 2 * intensity_center + intensity_backward)
                    / (h**2)
                ).T.copy()
                derivative_label = "2nd Derivative (d²I/dλ²)"
            else:
                raise ValueError(
                    f"Invalid derivative_order: {derivative_order}. Must be 0, 1, or 2."
                )

            # DO NOT normalize yet - wait until after cropping for better contrast!
            # Normalization will happen after cropping (if crop is applied)

            # Store as single-channel "image" for compatibility with rest of plotting code
            # We'll convert this to use a wavelength-specific colormap later
            wavelength_image = intensity_map
            wavelength_cmap = create_wavelength_colormap(actual_wl)
            n_tracks, n_slits = cube_data.shape[0], cube_data.shape[1]

            print(f"   Derivative: {derivative_label}")
            print(
                f"   Raw intensity range (before normalization): [{np.nanmin(intensity_map):.4f}, {np.nanmax(intensity_map):.4f}]"
            )
            print(
                f"   ⚠️ Normalization will be applied AFTER cropping for optimal contrast"
            )

        else:
            # Standard RGB mode
            red_idx = np.argmin(np.abs(self.wavelengths - red_wl))
            green_idx = np.argmin(np.abs(self.wavelengths - green_wl))
            blue_idx = np.argmin(np.abs(self.wavelengths - blue_wl))

        # NEW: Apply derivative if requested (RGB mode only)
        if not use_wavelength_colormap and derivative_order > 0:
            # Compute derivatives for each RGB channel
            def compute_derivative_2d(data_2d, wl_idx, order, window):
                """Compute derivative at a specific wavelength for 2D spatial data"""
                if order == 1:
                    # First derivative: dI/dλ
                    if wl_idx < window or wl_idx >= len(self.wavelengths) - window:
                        raise ValueError(
                            f"Wavelength index {wl_idx} too close to edge for derivative with window={window}"
                        )

                    intensity_forward = data_2d[:, :, wl_idx + window]
                    intensity_backward = data_2d[:, :, wl_idx - window]
                    wl_forward = self.wavelengths[wl_idx + window]
                    wl_backward = self.wavelengths[wl_idx - window]

                    derivative = (intensity_forward - intensity_backward) / (
                        wl_forward - wl_backward
                    )

                elif order == 2:
                    # Second derivative: d²I/dλ²
                    if wl_idx < window or wl_idx >= len(self.wavelengths) - window:
                        raise ValueError(
                            f"Wavelength index {wl_idx} too close to edge for 2nd derivative with window={window}"
                        )

                    intensity_center = data_2d[:, :, wl_idx]
                    intensity_forward = data_2d[:, :, wl_idx + window]
                    intensity_backward = data_2d[:, :, wl_idx - window]

                    h = self.wavelengths[wl_idx + window] - self.wavelengths[wl_idx]
                    derivative = (
                        intensity_forward - 2 * intensity_center + intensity_backward
                    ) / (h**2)
                else:
                    raise ValueError(
                        f"Invalid derivative_order: {order}. Must be 0, 1, or 2."
                    )

                return derivative

            # Compute derivatives for R, G, B channels
            R = compute_derivative_2d(
                cube_data, red_idx, derivative_order, derivative_window
            ).T.copy()
            G = compute_derivative_2d(
                cube_data, green_idx, derivative_order, derivative_window
            ).T.copy()
            B = compute_derivative_2d(
                cube_data, blue_idx, derivative_order, derivative_window
            ).T.copy()

            print(
                f"📊 Derivative order {derivative_order} computed at R={red_wl}nm, G={green_wl}nm, B={blue_wl}nm"
            )
        elif not use_wavelength_colormap:
            # Extract RGB channels - default: (tracks, slits) -> transpose to (slits, tracks) for imshow
            R = cube_data[:, :, red_idx].T.copy()
            G = cube_data[:, :, green_idx].T.copy()
            B = cube_data[:, :, blue_idx].T.copy()

        # Normalize each channel to [0, 1] range (RGB mode only, wavelength mode has own normalization)
        if not use_wavelength_colormap and normalize:
            # Check if vmin/vmax are provided for manual normalization
            if vmin is not None and vmax is not None:
                # Support both single values and per-channel tuples
                # Single value: vmin=0.5, vmax=1.5 → apply to all channels
                # Per-channel: vmin=(0.4, 0.5, 0.3), vmax=(1.5, 1.6, 1.4) → R, G, B
                if isinstance(vmin, (tuple, list)) and isinstance(vmax, (tuple, list)):
                    # Per-channel normalization
                    if len(vmin) != 3 or len(vmax) != 3:
                        raise ValueError(
                            f"vmin and vmax must have 3 values for per-channel normalization, got {len(vmin)} and {len(vmax)}"
                        )
                    for i, (C, ch_name) in enumerate(
                        [(R, "Red"), (G, "Green"), (B, "Blue")]
                    ):
                        ch_min, ch_max = vmin[i], vmax[i]
                        if ch_max > ch_min:
                            C[:] = np.clip((C - ch_min) / (ch_max - ch_min), 0, 1)
                        else:
                            raise ValueError(
                                f"{ch_name} channel: vmax ({ch_max}) must be > vmin ({ch_min})"
                            )
                    print(f"✨ Manual per-channel normalization applied:")
                    print(f"   Red:   [{vmin[0]:.4f}, {vmax[0]:.4f}]")
                    print(f"   Green: [{vmin[1]:.4f}, {vmax[1]:.4f}]")
                    print(f"   Blue:  [{vmin[2]:.4f}, {vmax[2]:.4f}]")
                else:
                    # Single value normalization (same range for all channels)
                    if vmax > vmin:
                        for C in (R, G, B):
                            C[:] = np.clip((C - vmin) / (vmax - vmin), 0, 1)
                        print(
                            f"✨ Manual normalization applied to all channels: [{vmin:.4f}, {vmax:.4f}]"
                        )
                    else:
                        raise ValueError(f"vmax ({vmax}) must be > vmin ({vmin})")
            else:
                # Auto normalization: use global min/max per channel
                for C in (R, G, B):
                    if C.max() != C.min():
                        C[:] = (C - C.min()) / (C.max() - C.min())

        # Create image for display
        if use_wavelength_colormap:
            # Single-channel wavelength image (will use colormap)
            display_image = wavelength_image
        else:
            # RGB composite
            rgb_image = np.stack([R, G, B], axis=-1)
            display_image = rgb_image
            n_tracks, n_slits = cube_data.shape[0], cube_data.shape[1]

        # Initialize roi_pixel_counts at function scope (needed for legend in solid mode)
        roi_pixel_counts = {}

        # ========== SOLID ROI OVERLAY (applied to RGB before transformations) ==========
        if (
            roi_overlay_mode == "solid"
            and roi_collection is not None
            and not use_wavelength_colormap
        ):
            # Convert RGB to uint8 for pixel manipulation
            rgb_overlay = np.clip(display_image * 255, 0, 255).astype(np.uint8)

            # Prepare ROI collection
            if roi_collection == "all":
                if hasattr(self, "roi_collection") and self.roi_collection:
                    rois_to_plot = self.roi_collection
                else:
                    rois_to_plot = {}
            elif isinstance(roi_collection, list):
                if hasattr(self, "roi_collection") and self.roi_collection:
                    rois_to_plot = {
                        name: self.roi_collection[name]
                        for name in roi_collection
                        if name in self.roi_collection
                    }
                else:
                    rois_to_plot = {}
            elif isinstance(roi_collection, dict):
                rois_to_plot = roi_collection
            else:
                rois_to_plot = {}

            # Apply solid colors to ROI pixels (roi_pixel_counts already initialized above)

            # Calculate pixel radius for block painting (e.g., size=3 → radius=1, size=5 → radius=2)
            pixel_radius = roi_solid_pixel_size // 2

            for roi_idx, (roi_name, roi_pixels_list) in enumerate(rois_to_plot.items()):
                if roi_pixels_list:
                    # Get color for this ROI
                    color = self._get_roi_color(roi_name, roi_colors, roi_color_map)

                    # Convert matplotlib color to RGB tuple
                    from matplotlib.colors import to_rgb

                    rgb_color = np.array(to_rgb(color)) * 255
                    rgb_color = rgb_color.astype(np.uint8)

                    pixel_count = 0
                    # Apply color to each ROI pixel (and surrounding pixels if size > 1)
                    for slit, track in roi_pixels_list:
                        # Bounds check for center pixel
                        if 0 <= track < n_tracks and 0 <= slit < n_slits:
                            # Paint center pixel and surrounding block
                            for dt in range(-pixel_radius, pixel_radius + 1):
                                for ds in range(-pixel_radius, pixel_radius + 1):
                                    t_paint = track + dt
                                    s_paint = slit + ds

                                    # Bounds check for each pixel in the block
                                    if (
                                        0 <= t_paint < n_tracks
                                        and 0 <= s_paint < n_slits
                                    ):
                                        # Note: display_image is (slits, tracks, 3) but we're in (track, slit) coords
                                        # Need to swap indices: display[slit, track, :]
                                        rgb_overlay[s_paint, t_paint] = rgb_color

                            pixel_count += 1

                    roi_pixel_counts[roi_name] = pixel_count

            # Convert back to [0, 1] range
            display_image = rgb_overlay.astype(np.float64) / 255.0

        # Apply axis transformations
        if flip_axes:
            # Swap dimensions: tracks become vertical, slits become horizontal
            if use_wavelength_colormap:
                display_image = np.transpose(
                    display_image, (1, 0)
                )  # Swap dimensions for 2D image
            else:
                display_image = np.transpose(
                    display_image, (1, 0, 2)
                )  # Swap first two dimensions for RGB

        # Flip the image data itself (not just the axis labels)
        if flip_horizontal:
            # Flip left-right: reverse along axis 1 (horizontal/width)
            display_image = np.flip(display_image, axis=1)

        if flip_vertical:
            # Flip up-down: reverse along axis 0 (vertical/height)
            display_image = np.flip(display_image, axis=0)

        # Handle cropping - extract square region centered at specified coordinates
        # NOTE: Crop coordinates are specified in ORIGINAL (track, slit) space before any flips
        crop_applied = False
        crop_track_min = crop_track_max = crop_slit_min = crop_slit_max = None

        if (
            crop_center_track is not None
            or crop_center_slit is not None
            or crop_width is not None
        ):
            # Validate that all three crop parameters are provided
            if (
                crop_center_track is None
                or crop_center_slit is None
                or crop_width is None
            ):
                raise ValueError(
                    "All three crop parameters (crop_center_track, crop_center_slit, crop_width) must be provided together"
                )

            # Calculate crop bounds in original coordinate space
            # crop_width is in slit pixels; adjust track pixels by aspect ratio to get square physical area
            half_width_slit = crop_width // 2
            half_width_track = int(crop_width / crop_aspect_ratio / 2)

            # Convert global track indices to local indices (relative to this cube's data)
            track_offset = getattr(self, "track_offset", 0)
            crop_center_track_local = crop_center_track - track_offset

            crop_track_min = max(0, crop_center_track_local - half_width_track)
            crop_track_max = min(n_tracks, crop_center_track_local + half_width_track)
            crop_slit_min = max(0, crop_center_slit - half_width_slit)
            crop_slit_max = min(n_slits, crop_center_slit + half_width_slit)

            print(
                f"📐 Crop center: global track {crop_center_track} → local track {crop_center_track_local}, slit {crop_center_slit}"
            )
            print(f"   Aspect ratio correction: {crop_aspect_ratio:.2f}:1 (track:slit)")
            print(
                f"   Slit pixels: {crop_slit_max - crop_slit_min}, Track pixels: {crop_track_max - crop_track_min}"
            )

            # Extract cropped region from transformed display_image
            # display_image shape depends on mode:
            # - RGB mode: (n_slits, n_tracks, 3) initially, then transformations applied
            # - Wavelength mode: (n_slits, n_tracks) initially, then transformations applied
            # After flip_axes: RGB becomes (n_tracks, n_slits, 3), wavelength becomes (n_tracks, n_slits)
            # After flip_horizontal/vertical: data is flipped but shape stays same

            # Determine current shape and extract crop
            current_shape = display_image.shape

            if flip_axes:
                # Current shape: (n_tracks, n_slits) or (n_tracks, n_slits, 3)
                # Axis 0 = tracks, Axis 1 = slits
                # Need to account for flips that were applied to the data
                if flip_vertical:
                    # Tracks were flipped, so indices are reversed
                    row_slice = slice(
                        n_tracks - crop_track_max, n_tracks - crop_track_min
                    )
                else:
                    row_slice = slice(crop_track_min, crop_track_max)

                if flip_horizontal:
                    # Slits were flipped
                    col_slice = slice(n_slits - crop_slit_max, n_slits - crop_slit_min)
                else:
                    col_slice = slice(crop_slit_min, crop_slit_max)
            else:
                # Current shape: (n_slits, n_tracks, 3)
                # Axis 0 = slits, Axis 1 = tracks
                if flip_vertical:
                    # Slits were flipped
                    row_slice = slice(n_slits - crop_slit_max, n_slits - crop_slit_min)
                else:
                    row_slice = slice(crop_slit_min, crop_slit_max)

                if flip_horizontal:
                    # Tracks were flipped
                    col_slice = slice(
                        n_tracks - crop_track_max, n_tracks - crop_track_min
                    )
                else:
                    col_slice = slice(crop_track_min, crop_track_max)

            # Extract the crop
            if use_wavelength_colormap:
                # 2D image
                display_image = display_image[row_slice, col_slice]
            else:
                # RGB image
                display_image = display_image[row_slice, col_slice, :]
            crop_applied = True

            print(
                f"🔍 Crop applied: track [{crop_track_min}:{crop_track_max}], slit [{crop_slit_min}:{crop_slit_max}]"
            )
            print(f"   Resulting shape: {display_image.shape}")

            # Print raw intensity range AFTER cropping (before normalization)
            if use_wavelength_colormap:
                crop_min = np.nanmin(display_image)
                crop_max = np.nanmax(display_image)
                print(
                    f"   📊 Raw intensity range of CROPPED region (before normalization): [{crop_min:.6f}, {crop_max:.6f}]"
                )

        # CRITICAL: Apply normalization AFTER cropping for wavelength mode
        # This ensures percentiles are computed on the displayed region, not the full transect
        if use_wavelength_colormap and normalize:
            # Use fixed vmin/vmax if provided, otherwise compute percentiles
            if vmin is not None and vmax is not None:
                intensity_min = vmin
                intensity_max = vmax
                print(f"✨ Normalization applied AFTER crop:")
                print(
                    f"   🔒 FIXED normalization range (user-specified): [{intensity_min:.6f}, {intensity_max:.6f}]"
                )
            else:
                # Normalize the cropped region using percentile-based scaling
                intensity_min = np.nanpercentile(display_image, 2)
                intensity_max = np.nanpercentile(display_image, 98)
                print(f"✨ Normalization applied AFTER crop:")
                print(
                    f"   📊 AUTO percentile range (2nd-98th): [{intensity_min:.6f}, {intensity_max:.6f}]"
                )

            if intensity_max > intensity_min:
                display_image = (display_image - intensity_min) / (
                    intensity_max - intensity_min
                )
                # Clip to [0, 1] to handle values outside percentile range
                display_image = np.clip(display_image, 0, 1)
            else:
                display_image = np.zeros_like(display_image)

            print(
                f"   Normalized output range: [{np.nanmin(display_image):.4f}, {np.nanmax(display_image):.4f}]"
            )

        # Set up extent (defines coordinate range for axes) - keep axes consistent with actual data
        if crop_applied:
            # Calculate REQUESTED global bounds (not clipped) for axis labels
            # This ensures extent shows what user requested, even if crop was clipped at boundaries
            requested_track_min_global = crop_center_track - half_width_track
            requested_track_max_global = crop_center_track + half_width_track
            requested_slit_min = crop_center_slit - half_width_slit
            requested_slit_max = crop_center_slit + half_width_slit

            if flip_axes:
                # After flip: horizontal=slits, vertical=tracks
                x_min, x_max = requested_slit_min, requested_slit_max
                y_min, y_max = requested_track_min_global, requested_track_max_global
                xlabel_text = "Slit Pixel Index"
                ylabel_text = "Track Index"
            else:
                # Default: horizontal=tracks, vertical=slits
                x_min, x_max = requested_track_min_global, requested_track_max_global
                y_min, y_max = requested_slit_min, requested_slit_max
                xlabel_text = "Track Index"
                ylabel_text = "Slit Pixel Index"
        else:
            # Use full image ranges
            if flip_axes:
                # After flip: horizontal=slits, vertical=tracks
                x_min, x_max = 0, n_slits
                y_min, y_max = 0, n_tracks
                xlabel_text = "Slit Pixel Index"
                ylabel_text = "Track Index"
            else:
                # Default: horizontal=tracks, vertical=slits
                x_min, x_max = 0, n_tracks
                y_min, y_max = 0, n_slits
                xlabel_text = "Track Index"
                ylabel_text = "Slit Pixel Index"

        # Origin is always 'lower' now since we flip the actual data
        origin = "lower"

        # Helper function to transform coordinates based on flipping and cropping
        def transform_coords(track, slit):
            """Transform track/slit coordinates based on flip settings and crop offset.

            INPUT: track, slit in ORIGINAL full-image coordinates
            OUTPUT: x_coord, y_coord in display coordinates (accounting for crop + flips)
            """
            # The extent is set in original coordinates, and imshow with extent uses
            # data coordinates directly. When we flip the DATA, the coordinate system
            # in the extent needs to be adjusted to match.

            # Transform based on actual data flipping
            if flip_axes:
                # After axis swap: x=slit, y=track (in original coordinate space)
                x_coord = slit
                y_coord = track
                # When we flip the data horizontally/vertically, we need to mirror
                # the coordinates within the extent range (not the full image range)
                if flip_horizontal:
                    # Mirror x within its extent range
                    if crop_applied:
                        x_coord = crop_slit_min + crop_slit_max - x_coord
                    else:
                        x_coord = n_slits - 1 - x_coord
                if flip_vertical:
                    # Mirror y within its extent range
                    if crop_applied:
                        y_coord = crop_track_min + crop_track_max - y_coord
                    else:
                        y_coord = n_tracks - 1 - y_coord
            else:
                # Default: x=track, y=slit (in original coordinate space)
                x_coord = track
                y_coord = slit
                # Apply flips within the appropriate range
                if flip_horizontal:
                    if crop_applied:
                        x_coord = crop_track_min + crop_track_max - x_coord
                    else:
                        x_coord = n_tracks - 1 - x_coord
                if flip_vertical:
                    if crop_applied:
                        y_coord = crop_slit_min + crop_slit_max - y_coord
                    else:
                        y_coord = n_slits - 1 - y_coord
            return x_coord, y_coord  # Set up figure size

        if figsize is None:
            if crop_applied:
                # For cropped images, make it square with reasonable size
                figsize = (8, 8)
            else:
                figsize = (12 * spacing, 6)

        fig, ax = plt.subplots(figsize=figsize)

        # For cropped images, use aspect='auto' but with square figsize
        # aspect='equal' causes matplotlib to adjust limits beyond the extent
        if use_wavelength_colormap:
            # Single-wavelength colormap mode
            im = ax.imshow(
                display_image,
                aspect="auto",
                origin=origin,
                extent=[x_min, x_max, y_min, y_max],
                cmap=wavelength_cmap,
                vmin=0,
                vmax=1,
            )
            # Add colorbar for wavelength mode
            if colorbar_fraction is None:
                # Auto full-height colorbar
                cbar = plt.colorbar(
                    im,
                    ax=ax,
                    pad=colorbar_pad,
                    shrink=colorbar_shrink,
                    orientation=colorbar_orientation,
                )
            else:
                # User-specified fraction
                cbar = plt.colorbar(
                    im,
                    ax=ax,
                    fraction=colorbar_fraction,
                    pad=colorbar_pad,
                    shrink=colorbar_shrink,
                    orientation=colorbar_orientation,
                )
            cbar.set_label(
                f"{derivative_label if derivative_order > 0 else 'Intensity'}\n(white = min, color = max)",
                fontsize=11,
            )
        else:
            # RGB composite mode
            ax.imshow(
                display_image,
                aspect="auto",
                origin=origin,
                extent=[x_min, x_max, y_min, y_max],
            )

        # Force the axis limits to match the extent exactly when cropping
        if crop_applied:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        print(f"📊 Extent: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        print(f"📐 Image shape after transformations: {display_image.shape}")

        # Note: Overlays are disabled when cropping is applied, as they would reference
        # coordinates outside the cropped region which could be misleading

        # File boundaries (adjusted for axis flipping)
        if show_file_boundaries and not crop_applied:
            for i, b in enumerate(self.file_boundaries):
                # Draw boundary line (skip first one at track 0)
                if i > 0:
                    track_boundary = b["start_track"]
                    # Transform the boundary coordinate
                    if flip_axes:
                        # Tracks are now vertical
                        if flip_vertical:
                            track_boundary = n_tracks - 1 - track_boundary
                        plt.axhline(
                            track_boundary,
                            color="yellow",
                            linestyle=":",
                            linewidth=2,
                            alpha=0.8,
                            label="File boundary" if i == 1 else "",
                        )
                    else:
                        # Default: tracks are horizontal
                        if flip_horizontal:
                            track_boundary = n_tracks - 1 - track_boundary
                        plt.axvline(
                            track_boundary,
                            color="yellow",
                            linestyle=":",
                            linewidth=2,
                            alpha=0.8,
                            label="File boundary" if i == 1 else "",
                        )

        # Cross-hairs (adjusted for axis flipping)
        if track_index is not None and not crop_applied:
            x_coord, _ = transform_coords(track_index, 0)
            if flip_axes:
                # Tracks are now vertical (y-axis)
                _, y_coord = transform_coords(track_index, 0)
                plt.axhline(y_coord, color="cyan", linestyle="--", linewidth=2)
            else:
                # Default: tracks are horizontal (x-axis)
                plt.axvline(x_coord, color="cyan", linestyle="--", linewidth=2)
        if slit_index is not None and not crop_applied:
            if flip_axes:
                # Slits are now horizontal (x-axis)
                x_coord, _ = transform_coords(0, slit_index)
                plt.axvline(x_coord, color="lime", linestyle="--", linewidth=2)
            else:
                # Default: slits are vertical (y-axis)
                _, y_coord = transform_coords(0, slit_index)
                plt.axhline(y_coord, color="lime", linestyle="--", linewidth=2)

        # Perimeter lines (unchanged)
        if perimeter_line is not None and not crop_applied:
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

                # Transform coordinates based on flip settings
                x1, y1 = transform_coords(track1, slit1)
                x2, y2 = transform_coords(track2, slit2)

                plt.plot(
                    [x1, x2],
                    [y1, y2],
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

            # Check overlay mode: solid or markers
            if roi_overlay_mode == "solid":
                # Solid overlay mode: ROI pixels already colored in RGB above
                # Just add legend entries without scatter plots
                from matplotlib.patches import Patch

                legend_handles = []
                for roi_name, roi_pixels_list in rois_to_plot.items():
                    if roi_name in roi_pixel_counts and roi_pixel_counts[roi_name] > 0:
                        color = self._get_roi_color(roi_name, roi_colors, roi_color_map)
                        # Check if roi_name already contains pixel count (e.g., "bombs (2369 px)" or "bombs (2369)")
                        # If so, don't add it again to avoid duplication
                        if " px)" in roi_name or (
                            roi_legend_show_counts
                            and ")" in roi_name
                            and "(" in roi_name
                        ):
                            # Already has pixel count, use as-is
                            label = roi_name
                        elif roi_legend_show_counts:
                            # Add pixel count if requested
                            label = f"{roi_name} ({roi_pixel_counts[roi_name]})"
                        else:
                            # Don't add pixel count
                            label = roi_name
                        legend_handles.append(
                            Patch(
                                facecolor=color,
                                label=label,
                            )
                        )

                # Add legend if we have entries
                if legend_handles and roi_legend_loc is not None:
                    if roi_legend_loc == "outside":
                        ax.legend(
                            handles=legend_handles,
                            loc="center left",
                            bbox_to_anchor=(1.02, 0.5),
                            fontsize=9,
                            framealpha=0.9,
                        )
                    else:
                        ax.legend(
                            handles=legend_handles,
                            loc=roi_legend_loc,
                            fontsize=9,
                            framealpha=0.9,
                        )
            else:
                # Marker mode: Plot each ROI with scatter (original behavior)
                for roi_idx, (roi_name, roi_pixels_list) in enumerate(
                    rois_to_plot.items()
                ):
                    if roi_pixels_list:
                        # NEW: Filter ROIs based on crop boundaries (if cropping is applied)
                        if crop_applied:
                            # Filter ROIs that fall within the crop region (using ORIGINAL coordinates)
                            valid_rois = [
                                (slit, track)
                                for slit, track in roi_pixels_list
                                if crop_slit_min <= slit < crop_slit_max
                                and crop_track_min <= track < crop_track_max
                            ]
                        else:
                            # No cropping: validate against full image bounds
                            valid_rois = [
                                (slit, track)
                                for slit, track in roi_pixels_list
                                if 0 <= slit < n_slits and 0 <= track < n_tracks
                            ]

                        if valid_rois:
                            roi_tracks = [track for slit, track in valid_rois]
                            roi_slits = [slit for slit, track in valid_rois]

                            # NEW: Get consistent color across all plots
                            color = self._get_roi_color(
                                roi_name, roi_colors, roi_color_map
                            )

                            # Transform ROI coordinates based on flip settings
                            roi_x_coords = []
                            roi_y_coords = []
                            for track, slit in zip(roi_tracks, roi_slits):
                                x, y = transform_coords(track, slit)
                                roi_x_coords.append(x)
                                roi_y_coords.append(y)

                            # DEBUG: Print first few transformed coordinates
                            if crop_applied and len(valid_rois) > 0:
                                print(
                                    f"🔍 DEBUG ROI '{roi_name}': {len(valid_rois)} pixels"
                                )
                                print(
                                    f"   First 3 original: slit={roi_slits[:3]}, track={roi_tracks[:3]}"
                                )
                                print(
                                    f"   First 3 transformed: x={roi_x_coords[:3]}, y={roi_y_coords[:3]}"
                                )
                                print(
                                    f"   Extent: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
                                )

                            # Plot ROI with solid colors - FAST METHOD using scatter with square markers
                            plt.scatter(
                                roi_x_coords,
                                roi_y_coords,
                                c=color,
                                s=roi_marker_size**2 * 50,  # Square marker size
                                marker="s",  # Square marker for filled pixels
                                edgecolors=(
                                    "black" if roi_marker_edgewidth > 0 else "none"
                                ),
                                linewidths=roi_marker_edgewidth,
                                alpha=1.0,  # Solid colors, no transparency
                                label=f"{roi_name} ({len(valid_rois)})",
                            )

                            # Optional: Add numbers for each ROI (using transformed coordinates)
                            if roi_show_numbers:
                                for i, (slit, track) in enumerate(valid_rois, 1):
                                    x, y = transform_coords(track, slit)
                                    plt.text(
                                        x,
                                        y + 3,
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
            # Filter ROIs based on crop boundaries (if cropping is applied)
            if crop_applied:
                valid_rois = [
                    (slit, track)
                    for slit, track in roi_pixels
                    if crop_slit_min <= slit < crop_slit_max
                    and crop_track_min <= track < crop_track_max
                ]
            else:
                valid_rois = [
                    (slit, track)
                    for slit, track in roi_pixels
                    if 0 <= slit < n_slits and 0 <= track < n_tracks
                ]

            if valid_rois:
                roi_tracks = [track for slit, track in valid_rois]
                roi_slits = [slit for slit, track in valid_rois]

                # Transform ROI coordinates based on flip settings
                roi_x_coords = []
                roi_y_coords = []
                for track, slit in zip(roi_tracks, roi_slits):
                    x, y = transform_coords(track, slit)
                    roi_x_coords.append(x)
                    roi_y_coords.append(y)

                # Plot ROI with solid colors - FAST METHOD
                plt.scatter(
                    roi_x_coords,
                    roi_y_coords,
                    c=roi_colors[0],
                    s=roi_marker_size**2 * 50,
                    marker="s",
                    edgecolors="black" if roi_marker_edgewidth > 0 else "none",
                    linewidths=roi_marker_edgewidth,
                    alpha=1.0,
                    label=f"ROI Pixels ({len(valid_rois)})",
                )

        # Grid (adjusted for axis flipping) - disabled for cropped images
        if not crop_applied:
            if flip_axes:
                # Grid lines based on actual axis assignment
                for x in np.arange(0, n_slits + 1, 50):
                    plt.axvline(x, color="black", linewidth=0.5, alpha=0.3)
                for y in np.arange(0, n_tracks + 1, 50):
                    plt.axhline(y, color="black", linewidth=0.5, alpha=0.3)
            else:
                for x in np.arange(0, n_tracks + 1, 50):
                    plt.axvline(x, color="black", linewidth=0.5, alpha=0.3)
                for y in np.arange(0, n_slits + 1, 50):
                    plt.axhline(y, color="black", linewidth=0.5, alpha=0.3)

        plt.xlabel(xlabel_text)
        plt.ylabel(ylabel_text)

        # Build title with file names
        file_names = [b["file"] for b in self.file_boundaries]
        if len(file_names) == 1:
            files_str = file_names[0]
        elif len(file_names) <= 3:
            files_str = ", ".join(file_names)
        else:
            files_str = f"{file_names[0]}, {file_names[1]}, ... {file_names[-1]} ({len(file_names)} files)"

        # Add derivative info to title
        if use_wavelength_colormap:
            # Single-wavelength mode: show wavelength, not RGB channels
            if derivative_order == 0:
                title_prefix = f"{wavelength_colormap_target:.1f} nm Single Wavelength"
            elif derivative_order == 1:
                title_prefix = f"{wavelength_colormap_target:.1f} nm (1st Derivative)"
            else:
                title_prefix = f"{wavelength_colormap_target:.1f} nm (2nd Derivative)"
            title = f"{title_prefix} - {files_str}"
        else:
            # RGB composite mode: show RGB wavelengths
            derivative_labels = {
                0: "RGB Composite",
                1: "RGB 1st Derivative (dI/dλ)",
                2: "RGB 2nd Derivative (d²I/dλ²)",
            }
            title_prefix = derivative_labels.get(derivative_order, "RGB Composite")
            title = f"{title_prefix} - {files_str}\n(R={red_wl}nm, G={green_wl}nm, B={blue_wl}nm)"

        if use_corrected:
            title = "Corrected " + title
        plt.title(title, fontsize=12, fontweight="bold")

        # Also set the figure window title (for Qt backend)
        fig = plt.gcf()
        fig.canvas.manager.set_window_title(f"RGB: {files_str}")

        # Smart legend display
        has_overlays = (
            (show_file_boundaries and len(self.file_boundaries) > 1)
            or perimeter_line is not None
            or roi_collection is not None
            or (roi_pixels is not None and len(roi_pixels) > 0)
        )
        legend_outside = False
        # BUGFIX: Skip this legend creation if using solid overlay mode
        # (legend already created above in solid mode block with proper handles)
        print(
            f"🐛 DEBUG legend: has_overlays={has_overlays}, roi_legend_loc={roi_legend_loc}, roi_overlay_mode={roi_overlay_mode}"
        )
        if has_overlays and roi_legend_loc is not None and roi_overlay_mode != "solid":
            print(f"🐛 DEBUG: Creating legend (should NOT happen in solid mode!)")
            if roi_legend_loc == "outside":
                # Place legend outside the plot area on the right
                legend = plt.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    markerscale=roi_legend_markersize / 6,
                )
                legend_outside = True
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

        # Handle layout adjustment based on legend position
        if crop_applied:
            # Set aspect ratio for all crops (with or without legend)
            ax.set_aspect(display_aspect_ratio, adjustable="box")

            if legend_outside:
                # When the legend is outside, expand the figure width so the axes keep their aspect ratio
                fig_width, fig_height = fig.get_size_inches()
                legend_padding = 1.75  # Extra width (inches) devoted to legend area
                fig.set_size_inches(
                    fig_width + legend_padding, fig_height, forward=True
                )
        elif not legend_outside:
            # Only use tight_layout for full transects without outside legend
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
        smoothing_method="moving_average",  # NEW: 'moving_average' or 'gaussian'
        gaussian_sigma=2.0,  # NEW: Sigma for Gaussian smoothing
        show_markers=True,
        marker_size=4,
        line_labels=None,
        normalize_intensities=False,  # NEW: Normalize intensity values
        normalization_method="minmax",  # NEW: 'minmax', 'zscore', 'mean', or 'l2'
    ):
        """
        Plot intensity profiles with optional smoothing and normalization.

        WAVELENGTH OPTIONS:
        - wavelength=654.2: Single wavelength (creates 1 plot)
        - use_average=True: Average across wavelengths
          - wavelength_range='red': Average red bands (640-680nm)
          - wavelength_range='green': Average green bands (520-580nm)
          - wavelength_range='blue': Average blue bands (420-480nm)
          - wavelength_range='all' or None: Average all bands
          - wavelength_range=(500, 600): Custom range in nm

        SMOOTHING OPTIONS:
        - smoothing_method='moving_average': Uniform moving average (box filter)
        - smoothing_method='gaussian': Gaussian smoothing with sigma parameter
        - gaussian_sigma=2.0: Sigma value for Gaussian smoothing

        NORMALIZATION OPTIONS:
        - normalize_intensities=True: Enable intensity normalization
        - normalization_method='minmax': Scale to [0,1] range
        - normalization_method='zscore': Z-score normalization (mean=0, std=1)
        - normalization_method='mean': Divide by mean (relative to average)
        - normalization_method='l2': L2 normalization (unit vector, Euclidean norm=1)
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
                range_label = ""
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
            elif normalization_method == "l2":
                ylabel = f"L2 Normalized {base_ylabel}"
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
                elif normalization_method == "l2":
                    # L2 normalization (unit vector, Euclidean norm = 1)
                    l2_norm = np.linalg.norm(intensities)
                    if l2_norm != 0:
                        intensities = intensities / l2_norm

            sample_indices = np.arange(len(intensities))

            # Apply smoothing
            if smoothing_method == "gaussian" and gaussian_sigma > 0:
                # Gaussian smoothing
                from scipy.ndimage import gaussian_filter1d

                smoothed = gaussian_filter1d(intensities, sigma=gaussian_sigma)

                # Calculate std devs in windows for visualization
                window_size = max(1, int(gaussian_sigma * 3))  # 3-sigma window
                std_devs = []
                for i in range(len(smoothed)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(intensities), i + window_size // 2 + 1)
                    window_data = intensities[start_idx:end_idx]
                    std_devs.append(np.std(window_data))
                std_devs = np.array(std_devs)

                plot_intensities = smoothed
                plot_sample_indices = sample_indices
                has_std = True

            elif moving_average_window > 1:
                # Moving average smoothing (original method)
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
        plt.ylabel("Average pseudo-reflectance intensity")

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
        smoothing_method="moving_average",  # NEW: 'moving_average' or 'gaussian'
        gaussian_sigma=2.0,  # NEW: Sigma for Gaussian smoothing
        show_markers=False,
        marker_size=4,
        normalize_intensities=False,
        normalization_method="minmax",  # NEW: 'minmax', 'zscore', 'mean', or 'l2'
    ):
        """
        Compare different wavelength ranges for a single line.

        Parameters:
        -----------
        line : tuple
            Single line definition: ((slit1, track1), (slit2, track2))
        wavelength_ranges : list, optional
            List of wavelength ranges to compare. Default: ['red', 'green', 'blue', 'all']
            Can use: 'red', 'green', 'blue', 'all', integers (e.g., 677), or tuples like (500, 600)
        smoothing_method : str
            'moving_average' for uniform box filter or 'gaussian' for Gaussian smoothing
        gaussian_sigma : float
            Sigma value for Gaussian smoothing (ignored if smoothing_method != 'gaussian')
        normalization_method : str
            'minmax' (scale to [0,1]), 'zscore' (z-score), 'mean' (divide by mean), or 'l2' (unit vector)

        Example:
        --------
        cube.plot_wavelength_comparison(
            line=lines[0],  # First line
            wavelength_ranges=['red', 'green', 'blue', 'all'],
            smoothing_method='gaussian',
            gaussian_sigma=2.0,
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

        # Helper function to convert wavelength to physical RGB color
        def wavelength_to_rgb(wavelength):
            """Convert wavelength (nm) to RGB color"""
            gamma = 0.8
            intensity_max = 255

            if wavelength < 380:
                R, G, B = 0, 0, 0
            elif 380 <= wavelength < 440:
                R = -(wavelength - 440) / (440 - 380)
                G = 0.0
                B = 1.0
            elif 440 <= wavelength < 490:
                R = 0.0
                G = (wavelength - 440) / (490 - 440)
                B = 1.0
            elif 490 <= wavelength < 510:
                R = 0.0
                G = 1.0
                B = -(wavelength - 510) / (510 - 490)
            elif 510 <= wavelength < 580:
                R = (wavelength - 510) / (580 - 510)
                G = 1.0
                B = 0.0
            elif 580 <= wavelength < 645:
                R = 1.0
                G = -(wavelength - 645) / (645 - 580)
                B = 0.0
            elif 645 <= wavelength < 781:
                R = 1.0
                G = 0.0
                B = 0.0
            else:
                R, G, B = 0, 0, 0

            # Let the intensity fall off near the vision limits
            if wavelength < 380:
                factor = 0.0
            elif 380 <= wavelength < 420:
                factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
            elif 420 <= wavelength < 701:
                factor = 1.0
            elif 701 <= wavelength < 781:
                factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
            else:
                factor = 0.0

            R = int(intensity_max * ((R * factor) ** gamma))
            G = int(intensity_max * ((G * factor) ** gamma))
            B = int(intensity_max * ((B * factor) ** gamma))

            return f"#{R:02x}{G:02x}{B:02x}"

        # Color scheme for different ranges
        range_colors = {
            "red": "#e74c3c",
            "green": "#27ae60",
            "blue": "#3498db",
            "all": "#95a5a6",
        }

        for wl_range in wavelength_ranges:
            # Define wavelength mask and color
            use_physical_color = False

            # NEW: Support single wavelength integer (find closest and use physical color)
            if isinstance(wl_range, (int, float)) and not isinstance(wl_range, bool):
                # Single wavelength - find closest match
                target_wl = wl_range  # Store target for legend
                wl_idx = np.argmin(np.abs(self.wavelengths - wl_range))
                actual_wl = self.wavelengths[wl_idx]
                wl_mask = np.zeros(len(self.wavelengths), dtype=bool)
                wl_mask[wl_idx] = True
                range_label = f"{target_wl}nm"  # Use target wavelength in legend
                color = wavelength_to_rgb(actual_wl)
                use_physical_color = True
            elif wl_range == "all":
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
                elif normalization_method == "l2":
                    # L2 normalization (unit vector, Euclidean norm = 1)
                    l2_norm = np.linalg.norm(intensities)
                    if l2_norm != 0:
                        intensities = intensities / l2_norm

            sample_indices = np.arange(len(intensities))

            # Apply smoothing
            if smoothing_method == "gaussian" and gaussian_sigma > 0:
                # Gaussian smoothing
                from scipy.ndimage import gaussian_filter1d

                smoothed = gaussian_filter1d(intensities, sigma=gaussian_sigma)

                # Calculate std devs in windows for visualization
                window_size = max(1, int(gaussian_sigma * 3))  # 3-sigma window
                std_devs = []
                for i in range(len(smoothed)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(intensities), i + window_size // 2 + 1)
                    window_data = intensities[start_idx:end_idx]
                    std_devs.append(np.std(window_data))
                std_devs = np.array(std_devs)

                plot_intensities = smoothed
                plot_sample_indices = sample_indices
                has_std = True

            elif moving_average_window > 1:
                # Moving average smoothing (original method)
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

            # Get color (unless already set for physical wavelength)
            if not use_physical_color:
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
            elif normalization_method == "l2":
                ylabel = "L2 Normalized Intensity"
            else:
                ylabel = "Normalized Intensity"
        else:
            ylabel = "Pseudo-reflectance intensity"

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
        red_wl=620.0,
        green_wl=565.0,
        blue_wl=490.0,
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

            elif (
                event.button == 3
            ):  # Right click - REMOVE nearest pixel (within 10 pixels)
                if pixel in current_roi:
                    # Exact match - delete it
                    current_roi.remove(pixel)
                    print(
                        f"❌ Removed pixel: (slit={slit_idx}, track={track_idx}) - Total: {len(current_roi)}"
                    )
                else:
                    # Find nearest pixel within 10-pixel radius
                    min_dist = float("inf")
                    nearest_pixel = None
                    max_radius = 10  # Don't delete across the map

                    for roi_pixel in current_roi:
                        roi_slit, roi_track = roi_pixel
                        dist = np.sqrt(
                            (roi_slit - slit_idx) ** 2 + (roi_track - track_idx) ** 2
                        )
                        if dist < min_dist and dist <= max_radius:
                            min_dist = dist
                            nearest_pixel = roi_pixel

                    if nearest_pixel is not None:
                        current_roi.remove(nearest_pixel)
                        print(
                            f"❌ Removed nearest pixel: (slit={nearest_pixel[0]}, track={nearest_pixel[1]}) "
                            f"[distance: {min_dist:.1f} pixels] - Total: {len(current_roi)}"
                        )
                    else:
                        print(
                            f"⚠️ No pixel within 10 pixels of click: (slit={slit_idx}, track={track_idx})"
                        )

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
        smooth_before_filter=False,  # NEW: Apply smoothing before wavelength filtering
        smoothing_method="moving_average",  # NEW: Smoothing method ('moving_average', 'savgol', 'gaussian', 'median', 'whittaker', 'bilateral')
        savgol_polyorder=2,  # NEW: Polynomial order for Savitzky-Golay filter (typically 2 or 3)
        gaussian_sigma=2.0,  # NEW: Sigma for Gaussian smoothing
        median_kernel_size=5,  # NEW: Kernel size for median filter (must be odd)
        whittaker_lambda=1e4,  # NEW: Smoothing parameter for Whittaker smoother (larger = smoother)
        bilateral_sigma_spatial=2.0,  # NEW: Spatial sigma for bilateral filter
        bilateral_sigma_intensity=0.1,  # NEW: Intensity sigma for bilateral filter
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
        use_inline_labels=False,
        normalize=False,  # DEPRECATED: Use normalize_method instead (kept for backward compatibility)
        normalize_method=None,  # NEW: Advanced normalization options
        show_std=True,  # NEW: Control standard deviation bands
        interpolate_wavelengths=None,  # NEW: List of wavelength indices to interpolate (e.g., [104] for 560nm dip)
        legend_loc="outside",  # NEW: Legend location ('best', 'upper right', 'outside', etc., or None to hide)
        ylim=None,  # NEW: Y-axis limits - tuple (ymin, ymax) or float for ±range around mean
        derivative_order=0,  # NEW: Derivative order (0=raw, 1=first derivative, 2=second derivative)
        derivative_window=2,  # NEW: Window size for derivative computation (default=2)
    ):
        """Enhanced spectrum plotting with optional normalization, inline labels, std control, wavelength interpolation, derivative analysis, and legend placement.

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

        smooth_before_filter : bool, optional (default=False)
            Control whether smoothing happens before or after wavelength range filtering:
            - False (default): Filter wavelength range first, then smooth → no influence from outside range
            - True: Smooth full spectrum first, then filter → smoother at edges but includes outside influence
            Note: Interpolation always happens before filtering (to fix bad wavelengths across full spectrum).

        smoothing_method : str, optional (default='moving_average')
            Smoothing method to use:
            - "moving_average": Rolling mean (pandas-based, same as illumination correction V2)
            - "savgol": Savitzky-Golay filter (polynomial fitting, preserves peaks/valleys better)
            - "gaussian": Gaussian filter (sigma-based smoothing, good for noise reduction)
            - "median": Median filter (robust to outliers, preserves edges)
            - "whittaker": Whittaker smoother (penalized least squares, excellent feature preservation)
            - "bilateral": Bilateral filter (edge-preserving, intensity-aware smoothing)

        savgol_polyorder : int, optional (default=2)
            Polynomial order for Savitzky-Golay filter (only used if smoothing_method='savgol').
            Typical values: 2 (quadratic) or 3 (cubic). Must be less than wavelength_smoothing.

        gaussian_sigma : float, optional (default=2.0)
            Standard deviation for Gaussian filter (only used if smoothing_method='gaussian').
            Larger values produce smoother results.

        median_kernel_size : int, optional (default=5)
            Kernel size for median filter (only used if smoothing_method='median').
            Must be odd. Larger values produce smoother results.

        whittaker_lambda : float, optional (default=1e4)
            Smoothing parameter for Whittaker smoother (only used if smoothing_method='whittaker').
            Larger values produce smoother results. Typical range: 1e2 to 1e6.

        bilateral_sigma_spatial : float, optional (default=2.0)
            Spatial sigma for bilateral filter (only used if smoothing_method='bilateral').
            Controls how far neighbors are considered. Larger values = more smoothing.

        bilateral_sigma_intensity : float, optional (default=0.1)
            Intensity sigma for bilateral filter (only used if smoothing_method='bilateral').
            Controls how similar intensities must be. Smaller values = more edge preservation.

        ylim : float, tuple, or None, optional (default=None)
            Y-axis (intensity) limits for the plot:
            - None: Auto-scale (matplotlib default)
            - float: e.g., 0.2 → sets limits to mean ± 0.2 (reduces apparent volatility)
            - tuple: (ymin, ymax) → absolute limits
            Example: ylim=0.2 with mean=1.04 → plot range [0.84, 1.24]

        derivative_order : int, optional (default=0)
            Spectral derivative order:
            - 0: Raw spectrum (default, no derivative)
            - 1: First derivative dI/dλ (rate of change, highlights slopes)
            - 2: Second derivative d²I/dλ² (curvature, highlights absorption/emission features)
            Derivatives are computed AFTER all preprocessing (smoothing, normalization).
            Use derivatives to enhance spectral features and remove baseline effects.

        derivative_window : int, optional (default=2)
            Window size for derivative computation (number of wavelength steps).
            Larger windows = smoother derivatives but less resolution.
            Default of 2 works well for most hyperspectral data.
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
                # Only sort when "all" is used (automatic sorting)
                rois_to_plot = sort_rois_by_category(rois_to_plot)
            elif isinstance(roi_names, list):
                # PRESERVE ORDER when list is provided - respect user's explicit ordering
                for name in roi_names:
                    if name in self.roi_collection:
                        rois_to_plot[name] = self.roi_collection[name]
                # DON'T sort - user provided explicit order!
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
            # When wavelength_range=None, use slice(None) to select all wavelengths
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

        def smooth_spectrum(
            spectrum,
            window_size,
            method="moving_average",
            polyorder=2,
            gaussian_sigma_val=2.0,
            median_kernel_val=5,
            whittaker_lambda_val=1e4,
            bilateral_sigma_spatial_val=2.0,
            bilateral_sigma_intensity_val=0.1,
        ):
            """
            Smooth spectrum using specified method.

            Parameters:
            -----------
            spectrum : np.ndarray
                1D spectrum array
            window_size : int
                Window size for smoothing (used for moving_average and savgol)
            method : str
                'moving_average', 'savgol', 'gaussian', 'median', 'whittaker', 'bilateral'
            polyorder : int
                Polynomial order for Savitzky-Golay (only used if method='savgol')
            gaussian_sigma_val : float
                Sigma for Gaussian smoothing
            median_kernel_val : int
                Kernel size for median filter
            whittaker_lambda_val : float
                Smoothing parameter for Whittaker smoother
            bilateral_sigma_spatial_val : float
                Spatial sigma for bilateral filter
            bilateral_sigma_intensity_val : float
                Intensity sigma for bilateral filter

            Returns:
            --------
            np.ndarray : Smoothed spectrum
            """
            # Early return only for methods that use window_size
            if method in ["moving_average", "savgol"]:
                if window_size <= 1:
                    return spectrum
                if window_size > len(spectrum):
                    window_size = len(spectrum)

            if method == "savgol":
                # Savitzky-Golay filter
                from scipy.signal import savgol_filter

                # Ensure window_size is odd (required for savgol)
                if window_size % 2 == 0:
                    window_size += 1

                # Ensure polyorder < window_size
                if polyorder >= window_size:
                    polyorder = window_size - 1

                # Apply Savitzky-Golay with polynomial extrapolation at edges
                smoothed = savgol_filter(
                    spectrum, window_size, polyorder, mode="interp"
                )
                return smoothed

            elif method == "gaussian":
                # Gaussian filter smoothing
                from scipy.ndimage import gaussian_filter1d

                smoothed = gaussian_filter1d(
                    spectrum, sigma=gaussian_sigma_val, mode="nearest"
                )
                return smoothed

            elif method == "median":
                # Median filter smoothing
                from scipy.ndimage import median_filter

                # Ensure kernel size is odd
                kernel_size = median_kernel_val
                if kernel_size % 2 == 0:
                    kernel_size += 1

                smoothed = median_filter(spectrum, size=kernel_size, mode="nearest")
                return smoothed

            elif method == "whittaker":
                # Whittaker smoother (penalized least squares)
                # Based on Eilers (2003) "A perfect smoother"
                try:
                    from scipy.sparse import diags, eye
                    from scipy.sparse.linalg import spsolve

                    m = len(spectrum)
                    E = eye(m, format="csc")
                    D = diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m))

                    # Solve (E + lambda * D^T * D) * z = spectrum
                    # where z is the smoothed spectrum
                    W = E + whittaker_lambda_val * (D.T @ D)
                    smoothed = spsolve(W, spectrum)

                    return smoothed
                except Exception as e:
                    print(f"⚠️  Whittaker smoother failed: {e}, using original spectrum")
                    return spectrum

            elif method == "bilateral":
                # Bilateral filter (edge-preserving smoothing)
                # Weights based on both spatial distance and intensity similarity
                smoothed = np.zeros_like(spectrum)

                for i in range(len(spectrum)):
                    # Define spatial window
                    half_window = int(3 * bilateral_sigma_spatial_val)
                    start = max(0, i - half_window)
                    end = min(len(spectrum), i + half_window + 1)

                    # Compute weights
                    spatial_dist = np.arange(start, end) - i
                    spatial_weight = np.exp(
                        -(spatial_dist**2) / (2 * bilateral_sigma_spatial_val**2)
                    )

                    intensity_diff = spectrum[start:end] - spectrum[i]
                    intensity_weight = np.exp(
                        -(intensity_diff**2) / (2 * bilateral_sigma_intensity_val**2)
                    )

                    # Combined weight
                    combined_weight = spatial_weight * intensity_weight
                    combined_weight /= np.sum(combined_weight)

                    # Weighted average
                    smoothed[i] = np.sum(spectrum[start:end] * combined_weight)

                return smoothed

            else:
                # Moving average (default) - Use pandas rolling for proper alignment
                import pandas as pd

                series = pd.Series(spectrum)
                smoothed = series.rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                return smoothed.values

        def compute_spectrum_derivative(spectrum, wavelengths_array, order=0, window=2):
            """
            Compute spectral derivatives.

            Parameters:
            -----------
            spectrum : np.ndarray
                1D spectrum array
            wavelengths_array : np.ndarray
                Corresponding wavelengths for the spectrum
            order : int
                Derivative order (0=raw, 1=first, 2=second)
            window : int
                Window size for derivative computation

            Returns:
            --------
            np.ndarray : Derivative spectrum (same length as input)
            """
            if order == 0:
                # No derivative, return as-is
                return spectrum

            if len(spectrum) < 2 * window + 1:
                print(
                    f"⚠️  Warning: Spectrum too short for derivative with window={window}"
                )
                return spectrum

            derivative = np.zeros_like(spectrum)

            if order == 1:
                # First derivative: dI/dλ using central difference
                for i in range(len(spectrum)):
                    if i < window or i >= len(spectrum) - window:
                        # At edges, use forward/backward difference
                        if i < window:
                            # Forward difference at start
                            if i + 1 < len(spectrum):
                                derivative[i] = (spectrum[i + 1] - spectrum[i]) / (
                                    wavelengths_array[i + 1] - wavelengths_array[i]
                                )
                        else:
                            # Backward difference at end
                            derivative[i] = (spectrum[i] - spectrum[i - 1]) / (
                                wavelengths_array[i] - wavelengths_array[i - 1]
                            )
                    else:
                        # Central difference in the middle
                        derivative[i] = (
                            spectrum[i + window] - spectrum[i - window]
                        ) / (
                            wavelengths_array[i + window]
                            - wavelengths_array[i - window]
                        )

            elif order == 2:
                # Second derivative: d²I/dλ² using three-point formula
                for i in range(len(spectrum)):
                    if i < window or i >= len(spectrum) - window:
                        # At edges, set to 0 or use lower-order approximation
                        derivative[i] = 0
                    else:
                        # Three-point formula: (I[i+w] - 2*I[i] + I[i-w]) / h²
                        h = wavelengths_array[i + window] - wavelengths_array[i]
                        derivative[i] = (
                            spectrum[i + window]
                            - 2 * spectrum[i]
                            + spectrum[i - window]
                        ) / (h**2)

            else:
                print(
                    f"⚠️  Warning: Derivative order {order} not supported, returning raw spectrum"
                )
                return spectrum

            return derivative

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

        # Dynamic y-label based on normalization and derivative order
        if ylabel == "Intensity":  # Only change default label
            if derivative_order == 1:
                ylabel = "First Derivative (dI/dλ)"
            elif derivative_order == 2:
                ylabel = "Second Derivative (d²I/dλ²)"
            elif active_normalization:
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

                    # Control smoothing order based on smooth_before_filter parameter
                    if smooth_before_filter and wavelength_smoothing > 1:
                        # OPTION 1: Smooth BEFORE filtering (smooth full spectrum)
                        for i in range(spectra_array.shape[0]):
                            spectra_array[i, :] = smooth_spectrum(
                                spectra_array[i, :],
                                wavelength_smoothing,
                                smoothing_method,
                                savgol_polyorder,
                                gaussian_sigma,
                                median_kernel_size,
                                whittaker_lambda,
                                bilateral_sigma_spatial,
                                bilateral_sigma_intensity,
                            )
                        # Then filter wavelength range
                        spectra_array = spectra_array[:, wl_mask]
                        avg_spectrum = np.mean(spectra_array, axis=0)
                        std_spectrum = np.std(spectra_array, axis=0)
                        plot_wavelengths = wavelengths
                        plot_avg = avg_spectrum
                        plot_std = std_spectrum
                    else:
                        # OPTION 2 (DEFAULT): Filter BEFORE smoothing
                        # Apply wavelength filtering first
                        spectra_array = spectra_array[:, wl_mask]
                        avg_spectrum = np.mean(spectra_array, axis=0)
                        std_spectrum = np.std(spectra_array, axis=0)

                        # Then apply smoothing
                        if wavelength_smoothing > 1:
                            smoothed_avg = smooth_spectrum(
                                avg_spectrum,
                                wavelength_smoothing,
                                smoothing_method,
                                savgol_polyorder,
                                gaussian_sigma,
                                median_kernel_size,
                                whittaker_lambda,
                                bilateral_sigma_spatial,
                                bilateral_sigma_intensity,
                            )
                            smoothed_std = smooth_spectrum(
                                std_spectrum,
                                wavelength_smoothing,
                                smoothing_method,
                                savgol_polyorder,
                                gaussian_sigma,
                                median_kernel_size,
                                whittaker_lambda,
                                bilateral_sigma_spatial,
                                bilateral_sigma_intensity,
                            )
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

                    # NEW: Compute derivative if requested
                    if derivative_order > 0:
                        plot_avg = compute_spectrum_derivative(
                            plot_avg,
                            plot_wavelengths,
                            derivative_order,
                            derivative_window,
                        )
                        if show_std:
                            plot_std = compute_spectrum_derivative(
                                plot_std,
                                plot_wavelengths,
                                derivative_order,
                                derivative_window,
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
            if derivative_order == 1:
                title += " [1st Derivative]"
            elif derivative_order == 2:
                title += " [2nd Derivative]"
            if active_normalization:
                title += f" ({active_normalization.upper()})"
            if wavelength_smoothing > 1:
                smooth_label = "SG" if smoothing_method == "savgol" else "MA"
                if smoothing_method == "savgol":
                    title += f" ({smooth_label}: w={wavelength_smoothing}, p={savgol_polyorder})"
                else:
                    title += f" ({smooth_label}: {wavelength_smoothing})"
            title += f"\n({self.name})"

        else:
            # Single pixel mode
            track_offset = getattr(self, "track_offset", 0)
            rel_track = track_index - track_offset
            # Get FULL spectrum first (before wavelength filtering)
            spectrum = cube[rel_track, slit_index, :]

            # NEW: Apply interpolation to remove bad wavelengths (BEFORE wavelength filtering!)
            spectrum = interpolate_bad_wavelengths(spectrum, interpolate_wavelengths)

            # Control smoothing order based on smooth_before_filter parameter
            if smooth_before_filter and wavelength_smoothing > 1:
                # OPTION 1: Smooth BEFORE filtering (smooth full spectrum)
                smoothed_spectrum = smooth_spectrum(
                    spectrum,
                    wavelength_smoothing,
                    smoothing_method,
                    savgol_polyorder,
                    gaussian_sigma,
                    median_kernel_size,
                    whittaker_lambda,
                    bilateral_sigma_spatial,
                    bilateral_sigma_intensity,
                )
                # Then filter wavelength range
                plot_spectrum = smoothed_spectrum[wl_mask]
                plot_wavelengths = wavelengths
            else:
                # OPTION 2 (DEFAULT): Filter BEFORE smoothing
                # Apply wavelength filtering first
                spectrum = spectrum[wl_mask]

                if wavelength_smoothing > 1:
                    smoothed_spectrum = smooth_spectrum(
                        spectrum,
                        wavelength_smoothing,
                        smoothing_method,
                        savgol_polyorder,
                        gaussian_sigma,
                        median_kernel_size,
                        whittaker_lambda,
                        bilateral_sigma_spatial,
                        bilateral_sigma_intensity,
                    )
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

            # Compute derivative if requested
            if derivative_order > 0:
                plot_spectrum = compute_spectrum_derivative(
                    plot_spectrum, plot_wavelengths, derivative_order, derivative_window
                )

            plt.plot(plot_wavelengths, plot_spectrum, color=colors[0], linewidth=2)

            title = f"{data_label} Spectrum"
            if derivative_order == 1:
                title += " [1st Derivative]"
            elif derivative_order == 2:
                title += " [2nd Derivative]"
            if active_normalization:
                title += f" ({active_normalization.upper()})"
            if wavelength_smoothing > 1:
                smooth_label = "SG" if smoothing_method == "savgol" else "MA"
                if smoothing_method == "savgol":
                    title += f" ({smooth_label}: w={wavelength_smoothing}, p={savgol_polyorder})"
                else:
                    title += f" ({smooth_label}: {wavelength_smoothing})"

        plt.xlabel("Wavelength [nm]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Force exact wavelength range with no padding
        if wavelength_range is not None:
            ax = plt.gca()
            ax.set_xlim(wavelength_range[0], wavelength_range[1])
            ax.margins(x=0)
            ax.autoscale(enable=False, axis="x")

        # NEW: Handle y-axis limits
        if ylim is not None:
            ax = plt.gca()
            if isinstance(ylim, (int, float)):
                # Single value: set range to mean ± ylim
                # Get current y data to calculate mean
                lines = ax.get_lines()
                if lines:
                    all_ydata = []
                    for line in lines:
                        all_ydata.extend(line.get_ydata())
                    mean_y = np.mean(all_ydata)
                    ax.set_ylim(mean_y - ylim, mean_y + ylim)
            elif isinstance(ylim, (tuple, list)) and len(ylim) == 2:
                # Tuple: absolute limits
                ax.set_ylim(ylim[0], ylim[1])

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

    # ==================== SVM CLASSIFICATION METHODS ====================

    def train_svm(
        self,
        roi_collection,
        class_names=None,
        svm_kernel="rbf",
        svm_C=None,
        svm_gamma=None,
        optimize_params=True,
        cv_folds=5,
        use_corrected=True,
        quiet=False,
    ):
        """
        Train an SVM classifier using ROI pixels as training data.

        Args:
            roi_collection: Can be:
                           - List of ROI names (strings) → looks up in cube.roi_collection
                           - Dict of {roi_name: [(slit, track), ...]} ROI pixels
                           Each ROI name becomes a class label
            class_names: Optional list to map ROI names to class labels. If None, uses ROI names
            svm_kernel: SVM kernel type ('rbf', 'linear', 'poly')
            svm_C: Regularization parameter. If None and optimize_params=True, will be optimized
            svm_gamma: Kernel coefficient. If None and optimize_params=True, will be optimized
            optimize_params: If True, uses GridSearchCV to find best C and gamma
            cv_folds: Number of cross-validation folds for optimization
            use_corrected: Use corrected data (data_corrected) or raw data
            quiet: Suppress output messages

        Returns:
            dict with training results including trained model and accuracy scores

        Example:
            # After loading ROIs: cube.import_rois("my_rois.json")

            # Option 1: Pass list of ROI names (like plot_georef!)
            training_rois = ["sediment", "water", "vegetation"]
            results = cube.train_svm(
                roi_collection=training_rois,
                optimize_params=True
            )

            # Option 2: Pass entire roi_collection dict
            results = cube.train_svm(
                roi_collection=cube.roi_collection,
                optimize_params=True
            )
        """
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, confusion_matrix
        import time

        if not quiet:
            print("=" * 60)
            print("🤖 SVM TRAINING")
            print("=" * 60)

        # Handle roi_collection input (list of names OR dict)
        if isinstance(roi_collection, list):
            # User passed a list of ROI names → look them up in cube.roi_collection
            if not hasattr(self, "roi_collection") or not self.roi_collection:
                raise ValueError(
                    "No ROI collection found. Load ROIs first with cube.import_rois() "
                    "or use plot_interactive_rgb()"
                )

            # Build dict from list of names
            roi_dict = {}
            for roi_name in roi_collection:
                if roi_name in self.roi_collection:
                    roi_dict[roi_name] = self.roi_collection[roi_name]
                else:
                    print(
                        f"⚠️  Warning: ROI '{roi_name}' not found in collection, skipping"
                    )

            if not roi_dict:
                raise ValueError(
                    f"None of the specified ROIs {roi_collection} were found in cube.roi_collection"
                )

            roi_collection = roi_dict
        elif not isinstance(roi_collection, dict):
            raise TypeError(
                "roi_collection must be either a list of ROI names or a dict of "
                "{roi_name: [(slit, track), ...]}}"
            )

        # Get data
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )

        if cube_data is None:
            raise ValueError("No data loaded. Load a datacube first.")

        n_tracks, n_slits, n_wavelengths = cube_data.shape

        # Prepare training data
        X_train = []
        y_train = []
        class_pixel_counts = {}

        for roi_name, roi_pixels in roi_collection.items():
            if not quiet:
                print(f"\n📍 Processing ROI: '{roi_name}'")

            valid_count = 0
            for slit_idx, track_idx in roi_pixels:
                # Convert to relative track index
                rel_track = track_idx - self.track_offset

                if 0 <= rel_track < n_tracks and 0 <= slit_idx < n_slits:
                    spectrum = cube_data[rel_track, slit_idx, :]
                    X_train.append(spectrum)
                    y_train.append(roi_name)
                    valid_count += 1

            class_pixel_counts[roi_name] = valid_count
            if not quiet:
                print(f"   ✅ {valid_count} valid pixels")

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if not quiet:
            print(f"\n📊 Training Data Summary:")
            print(f"   Total samples: {len(X_train)}")
            print(f"   Number of classes: {len(class_pixel_counts)}")
            print(f"   Features (wavelengths): {X_train.shape[1]}")
            for class_name, count in class_pixel_counts.items():
                print(f"   - {class_name}: {count} pixels")

        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)

        # Parameter optimization
        start_time = time.time()

        if optimize_params and (svm_C is None or svm_gamma is None):
            if not quiet:
                print(f"\n🔍 Optimizing SVM parameters (kernel={svm_kernel})...")
                print(f"   Testing C and gamma combinations...")

            param_grid = {
                "C": [0.1, 1, 10, 100, 1000],
                "gamma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, "scale", "auto"],
            }

            grid_search = GridSearchCV(
                SVC(kernel=svm_kernel),
                param_grid,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0 if quiet else 1,
            )

            grid_search.fit(X_train, y_train_encoded)

            best_C = grid_search.best_params_["C"]
            best_gamma = grid_search.best_params_["gamma"]

            if not quiet:
                print(f"   ✅ Best C: {best_C}")
                print(f"   ✅ Best gamma: {best_gamma}")
                print(f"   ✅ Best CV accuracy: {grid_search.best_score_:.3f}")

            svm_model = grid_search.best_estimator_
        else:
            # Use provided or default parameters
            if svm_C is None:
                svm_C = 1000
            if svm_gamma is None:
                svm_gamma = 1e-6

            if not quiet:
                print(f"\n🔧 Training SVM with:")
                print(f"   Kernel: {svm_kernel}")
                print(f"   C: {svm_C}")
                print(f"   Gamma: {svm_gamma}")

            svm_model = SVC(kernel=svm_kernel, C=svm_C, gamma=svm_gamma)
            svm_model.fit(X_train, y_train_encoded)

            # Cross-validation score
            cv_scores = cross_val_score(
                svm_model, X_train, y_train_encoded, cv=cv_folds
            )
            if not quiet:
                print(
                    f"   ✅ CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})"
                )

        training_time = time.time() - start_time

        # Training accuracy
        y_train_pred = svm_model.predict(X_train)
        train_accuracy = np.mean(y_train_pred == y_train_encoded)

        if not quiet:
            print(f"\n⏱️  Training time: {training_time:.2f} seconds")
            print(f"🎯 Training accuracy: {train_accuracy:.3f}")

            # Confusion matrix
            print("\n📈 Confusion Matrix:")
            cm = confusion_matrix(y_train_encoded, y_train_pred)
            print(cm)

            print("\n📊 Classification Report:")
            print(
                classification_report(
                    y_train_encoded, y_train_pred, target_names=le.classes_
                )
            )

        # Store model and metadata
        self.svm_model = svm_model
        self.svm_label_encoder = le
        self.svm_class_names = (
            class_names if class_names else list(roi_collection.keys())
        )
        self.svm_training_data = {
            "X_train": X_train,
            "y_train": y_train,
            "y_train_encoded": y_train_encoded,
            "class_pixel_counts": class_pixel_counts,
        }

        results = {
            "model": svm_model,
            "label_encoder": le,
            "training_accuracy": train_accuracy,
            "training_time": training_time,
            "class_names": self.svm_class_names,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "class_pixel_counts": class_pixel_counts,
            "parameters": {
                "C": svm_model.C,
                "gamma": svm_model.gamma,
                "kernel": svm_kernel,
            },
        }

        if not quiet:
            print("\n✅ SVM training complete!")
            print("=" * 60)

        return results

    def classify_svm(
        self,
        track_start=None,
        track_end=None,
        use_corrected=True,
        save_to_h5=True,
        dataset_name="svm_classification",
        quiet=False,
    ):
        """
        Classify a segment of the datacube using the trained SVM model.

        Args:
            track_start: Start track index for classification (None = start of data)
            track_end: End track index for classification (None = end of data)
            use_corrected: Use corrected data or raw data
            save_to_h5: Save classification results to H5 file
            dataset_name: Name of dataset to save in H5 file
            quiet: Suppress output messages

        Returns:
            dict with classification results including predicted labels and probabilities

        Example:
            # After training with train_svm()
            results = cube.classify_svm(
                track_start=config.UHI_TRACK_RANGE_5[0],
                track_end=config.UHI_TRACK_RANGE_5[1]
            )
        """
        import time

        if not hasattr(self, "svm_model"):
            raise ValueError("No trained SVM model found. Run train_svm() first.")

        if not quiet:
            print("=" * 60)
            print("🎯 SVM CLASSIFICATION")
            print("=" * 60)

        # Get data
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )

        if cube_data is None:
            raise ValueError("No data loaded.")

        n_tracks, n_slits, n_wavelengths = cube_data.shape

        # Handle track range
        if track_start is None:
            track_start = self.track_offset
        if track_end is None:
            track_end = self.track_offset + n_tracks

        rel_start = track_start - self.track_offset
        rel_end = track_end - self.track_offset

        rel_start = max(0, min(rel_start, n_tracks))
        rel_end = max(0, min(rel_end, n_tracks))

        if not quiet:
            print(f"\n📍 Classification segment:")
            print(f"   Track range: {track_start} to {track_end}")
            print(f"   Relative indices: {rel_start} to {rel_end}")
            print(f"   Segment size: {rel_end - rel_start} tracks × {n_slits} slits")

        # Extract segment
        segment_data = cube_data[rel_start:rel_end, :, :]
        segment_shape = segment_data.shape

        if not quiet:
            print(f"\n🔄 Classifying {segment_shape[0] * segment_shape[1]} pixels...")

        start_time = time.time()

        # Reshape for classification
        X_classify = segment_data.reshape(-1, n_wavelengths)

        # Classify
        y_pred_encoded = self.svm_model.predict(X_classify)
        y_pred = self.svm_label_encoder.inverse_transform(y_pred_encoded)

        # Reshape back to image
        classification_map = y_pred.reshape(segment_shape[0], segment_shape[1])
        classification_map_encoded = y_pred_encoded.reshape(
            segment_shape[0], segment_shape[1]
        )

        classification_time = time.time() - start_time

        if not quiet:
            print(f"   ✅ Classification complete in {classification_time:.2f} seconds")

            # Class distribution
            print(f"\n📊 Classification Results:")
            unique, counts = np.unique(y_pred, return_counts=True)
            for class_name, count in zip(unique, counts):
                percentage = (count / len(y_pred)) * 100
                print(f"   - {class_name}: {count} pixels ({percentage:.1f}%)")

        # Save to H5
        if save_to_h5:
            for file_data in self.file_list:
                file_track_start = file_data["start_track"]
                file_track_end = file_data["end_track"]

                if track_end <= file_track_start or track_start >= file_track_end:
                    continue

                seg_start_in_file = max(0, track_start - file_track_start)
                seg_end_in_file = min(
                    file_track_end - file_track_start, track_end - file_track_start
                )

                map_start = max(0, file_track_start - track_start)
                map_end = map_start + (seg_end_in_file - seg_start_in_file)

                file_classification = classification_map_encoded[map_start:map_end, :]

                try:
                    with h5py.File(file_data["filepath"], "a") as f:
                        ds_path = f"processed/{dataset_name}"

                        if ds_path in f:
                            del f[ds_path]

                        f.create_dataset(
                            ds_path,
                            data=file_classification,
                            compression="gzip",
                            compression_opts=4,
                        )

                        # Save class names as attributes
                        f[ds_path].attrs["class_names"] = self.svm_class_names
                        f[ds_path].attrs["track_start"] = track_start
                        f[ds_path].attrs["track_end"] = track_end

                    if not quiet:
                        print(f"   💾 Saved to: {file_data['name']} → {ds_path}")

                except Exception as e:
                    print(f"   ⚠️  Could not save to {file_data['name']}: {e}")

        # Store results
        self.svm_classification_map = classification_map
        self.svm_classification_map_encoded = classification_map_encoded
        self.svm_classification_range = (track_start, track_end)

        results = {
            "classification_map": classification_map,
            "classification_map_encoded": classification_map_encoded,
            "class_names": self.svm_class_names,
            "track_range": (track_start, track_end),
            "classification_time": classification_time,
            "n_pixels": len(y_pred),
            "class_distribution": dict(zip(unique, counts)),
        }

        if not quiet:
            print("\n✅ Classification complete!")
            print("=" * 60)

        return results

    def _create_grid_groups_for_sediment(
        self,
        roi_pixels,
        n_tracks,
        n_slits,
        tile_size_slit,
        tile_size_track,
        min_pixels,
        quiet=False,
    ):
        """
        Create grid-based groups for sediment ROIs.

        Divides the datacube into regular grid tiles and assigns each pixel to its tile.
        Tiles with <min_pixels are merged to neighboring tiles.

        Args:
            roi_pixels: List of (slit, track) tuples
            n_tracks: Total tracks in datacube
            n_slits: Total slits in datacube
            tile_size_slit: Grid tile size in slit direction
            tile_size_track: Grid tile size in track direction
            min_pixels: Minimum pixels per tile
            quiet: Suppress output

        Returns:
            pixel_groups: Array of group IDs for each pixel
        """
        from scipy.spatial.distance import cdist

        # Assign each pixel to a grid tile
        pixel_groups = np.zeros(len(roi_pixels), dtype=int)
        tile_to_group_id = {}
        group_id_counter = 0

        # First pass: assign pixels to grid tiles
        for i, (slit_idx, track_idx) in enumerate(roi_pixels):
            tile_row = track_idx // tile_size_track
            tile_col = slit_idx // tile_size_slit
            tile_key = (tile_row, tile_col)

            if tile_key not in tile_to_group_id:
                tile_to_group_id[tile_key] = group_id_counter
                group_id_counter += 1

            pixel_groups[i] = tile_to_group_id[tile_key]

        # Calculate group sizes
        unique_groups = np.unique(pixel_groups)
        group_sizes = {gid: np.sum(pixel_groups == gid) for gid in unique_groups}

        # Find small groups that need merging
        small_groups = [gid for gid, size in group_sizes.items() if size < min_pixels]
        large_groups = [gid for gid, size in group_sizes.items() if size >= min_pixels]

        if small_groups and large_groups:
            # Get tile coordinates for each group
            group_to_tile = {v: k for k, v in tile_to_group_id.items()}

            for small_gid in small_groups:
                small_tile = group_to_tile[small_gid]
                small_row, small_col = small_tile

                # Find neighboring tiles (8-connectivity)
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        neighbor_tile = (small_row + dr, small_col + dc)
                        if neighbor_tile in tile_to_group_id:
                            neighbor_gid = tile_to_group_id[neighbor_tile]
                            if neighbor_gid in large_groups:
                                neighbors.append(neighbor_gid)

                if neighbors:
                    # Merge to first valid neighbor
                    merge_target = neighbors[0]
                    pixel_groups[pixel_groups == small_gid] = merge_target
                    if not quiet:
                        print(
                            f"      Merged small tile {small_tile} ({group_sizes[small_gid]} px) → neighbor tile"
                        )
                else:
                    # No neighbors - try to find closest large group
                    if large_groups:
                        # Calculate distance to all large group centroids
                        small_pixels = np.array(roi_pixels)[pixel_groups == small_gid]
                        small_centroid = np.mean(small_pixels, axis=0).reshape(1, -1)

                        large_centroids = []
                        for large_gid in large_groups:
                            large_pixels = np.array(roi_pixels)[
                                pixel_groups == large_gid
                            ]
                            large_centroid = np.mean(large_pixels, axis=0)
                            large_centroids.append(large_centroid)

                        large_centroids = np.array(large_centroids)
                        distances = cdist(small_centroid, large_centroids)[0]
                        nearest_idx = np.argmin(distances)
                        merge_target = large_groups[nearest_idx]

                        pixel_groups[pixel_groups == small_gid] = merge_target
                        if not quiet:
                            print(
                                f"      Merged isolated tile {small_tile} ({group_sizes[small_gid]} px) → nearest large tile"
                            )

        return pixel_groups

    def _create_spatial_groups(
        self,
        roi_pixels_per_class,
        datacube_shape,
        closing_radius=3,
        dbscan_eps=10,
        dbscan_min_samples=20,
        min_group_size=20,
        use_grid_for_sediment=True,  # 🔥 NEW: Use grid for sediment
        sediment_grid_tile_slit=100,  # 🔥 NEW: Grid tile size in slit direction
        sediment_grid_tile_track=200,  # 🔥 NEW: Grid tile size in track direction
        sediment_min_group_size=10,  # 🔥 NEW: Min pixels for sediment grid tiles
        quiet=False,
    ):
        """
        Create spatial groups from ROI pixels to prevent data leakage in CV.

        For each class, spatially cluster pixels into groups so that CV can split by group
        (keeping all pixels from same spatial region together).

        Special handling for sediment: Uses grid-based grouping instead of connected components
        to avoid creating one giant group.

        Args:
            roi_pixels_per_class: Dict {class_name: [(slit, track), ...]}
            datacube_shape: Tuple (n_tracks, n_slits, n_wavelengths)
            closing_radius: Morphological closing radius for bombs/dark (default: 3)
            dbscan_eps: DBSCAN epsilon parameter (default: 10)
            dbscan_min_samples: DBSCAN min_samples (default: 20)
            min_group_size: Minimum pixels per group for bombs/dark (default: 20)
            use_grid_for_sediment: Use grid grouping for training_sediment (default: True)
            sediment_grid_tile_slit: Grid tile size in slit direction (default: 100)
            sediment_grid_tile_track: Grid tile size in track direction (default: 200)
            sediment_min_group_size: Min pixels for sediment tiles (default: 10)
            quiet: Suppress output

        Returns:
            groups_list: List of group IDs aligned with flattened pixel list
            group_names: List of unique group names
        """
        from scipy.ndimage import binary_closing, label
        from scipy.spatial.distance import cdist

        n_tracks, n_slits = datacube_shape[0], datacube_shape[1]

        all_groups = []
        group_counter = {}

        if not quiet:
            print(f"\n🔬 Creating spatial groups for CV...")
            print(
                f"   Settings: closing_radius={closing_radius}, min_group_size={min_group_size}"
            )
            if use_grid_for_sediment:
                print(
                    f"   Sediment grid: {sediment_grid_tile_slit}(slit) × {sediment_grid_tile_track}(track) px, min={sediment_min_group_size}"
                )

        for class_name, roi_pixels in roi_pixels_per_class.items():
            if len(roi_pixels) == 0:
                continue

            # 🔥 SPECIAL CASE: Grid-based grouping for training_sediment
            if use_grid_for_sediment and class_name == "training_sediment":
                pixel_groups = self._create_grid_groups_for_sediment(
                    roi_pixels,
                    n_tracks,
                    n_slits,
                    sediment_grid_tile_slit,
                    sediment_grid_tile_track,
                    sediment_min_group_size,
                    quiet,
                )
                method = "grid"

                # Count groups and assign names
                # Note: _create_grid_groups_for_sediment already handles min size merging
                unique_groups = np.unique(pixel_groups[pixel_groups >= 0])
                group_info = {}
                for group_id in unique_groups:
                    group_mask = pixel_groups == group_id
                    group_size = np.sum(group_mask)
                    # Don't recheck min_group_size - grid function already merged small tiles
                    group_info[group_id] = {"size": group_size, "pixels": group_mask}

                # Assign sequential group names
                if class_name not in group_counter:
                    group_counter[class_name] = 0

                final_group_names = []
                group_mapping = {}
                for group_id in sorted(group_info.keys()):
                    group_counter[class_name] += 1
                    group_name = f"sed#{group_counter[class_name]:02d}"
                    group_mapping[group_id] = group_name
                    final_group_names.append(group_name)

                # Assign group names to pixels
                for i, group_id in enumerate(pixel_groups):
                    if group_id in group_mapping:
                        all_groups.append(group_mapping[group_id])
                    else:
                        all_groups.append(f"{class_name}_dropped")

                if not quiet:
                    print(
                        f"   {class_name}: {len(roi_pixels)} pixels → {len(group_info)} groups via {method}"
                    )
                    for gname in final_group_names[:5]:  # Show first 5
                        gid = [k for k, v in group_mapping.items() if v == gname][0]
                        print(f"      {gname}: {group_info[gid]['size']} pixels")
                    if len(final_group_names) > 5:
                        print(f"      ... and {len(final_group_names) - 5} more groups")

                continue  # Skip normal processing for sediment

            # Create binary mask (for bombs/dark - normal processing)
            mask = np.zeros((n_tracks, n_slits), dtype=bool)
            for slit_idx, track_idx in roi_pixels:
                if 0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits:
                    mask[track_idx, slit_idx] = True

            # Try connected components with morphological closing
            try:
                from scipy.ndimage import generate_binary_structure

                struct = generate_binary_structure(2, 2)  # 8-connectivity

                # Morphological closing to merge nearby pixels
                if closing_radius > 0:
                    from scipy.ndimage import binary_dilation

                    for _ in range(closing_radius):
                        mask = binary_dilation(mask, structure=struct)
                    for _ in range(closing_radius):
                        mask = binary_closing(mask, structure=struct)

                # Connected components
                labeled_mask, n_components = label(mask, structure=struct)

                # Extract group labels for each ROI pixel
                pixel_groups = []
                for slit_idx, track_idx in roi_pixels:
                    if 0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits:
                        group_id = labeled_mask[track_idx, slit_idx]
                        pixel_groups.append(group_id)
                    else:
                        pixel_groups.append(-1)  # Invalid

                pixel_groups = np.array(pixel_groups)
                method = "connected_components"

            except Exception as e:
                # Fallback to DBSCAN
                if not quiet:
                    print(
                        f"   ⚠️  Connected components failed for '{class_name}', using DBSCAN"
                    )

                try:
                    from sklearn.cluster import DBSCAN

                    # Get pixel coordinates
                    coords = np.array(
                        roi_pixels
                    )  # Shape: (n_pixels, 2) = (slit, track)

                    # DBSCAN clustering
                    clustering = DBSCAN(
                        eps=dbscan_eps,
                        min_samples=dbscan_min_samples,
                        metric="euclidean",
                    )
                    pixel_groups = clustering.fit_predict(coords)

                    method = "DBSCAN"

                except Exception as e2:
                    # Final fallback: treat entire ROI as one group
                    if not quiet:
                        print(
                            f"   ⚠️  DBSCAN also failed for '{class_name}', using single group"
                        )
                    pixel_groups = np.zeros(len(roi_pixels), dtype=int)
                    method = "single_group"

            # Filter and rename groups
            unique_groups = np.unique(pixel_groups[pixel_groups >= 0])

            # Calculate group sizes and centroids
            group_info = {}
            for group_id in unique_groups:
                group_mask = pixel_groups == group_id
                group_size = np.sum(group_mask)

                if group_size >= min_group_size:
                    # Calculate centroid
                    group_pixels = np.array(roi_pixels)[group_mask]
                    centroid = np.mean(group_pixels, axis=0)
                    group_info[group_id] = {
                        "size": group_size,
                        "centroid": centroid,
                        "pixels": group_mask,
                    }

            # Handle small groups: merge to nearest large group of same class
            small_groups = [gid for gid in unique_groups if gid not in group_info]
            if small_groups and group_info:
                large_group_centroids = np.array(
                    [info["centroid"] for info in group_info.values()]
                )
                large_group_ids = list(group_info.keys())

                for small_gid in small_groups:
                    small_mask = pixel_groups == small_gid
                    small_pixels = np.array(roi_pixels)[small_mask]
                    small_centroid = np.mean(small_pixels, axis=0).reshape(1, -1)

                    # Find nearest large group
                    distances = cdist(small_centroid, large_group_centroids)[0]
                    nearest_idx = np.argmin(distances)
                    nearest_gid = large_group_ids[nearest_idx]

                    # Merge: assign small group pixels to nearest group
                    pixel_groups[small_mask] = nearest_gid
                    group_info[nearest_gid]["size"] += np.sum(small_mask)

            # Assign group names
            if class_name not in group_counter:
                group_counter[class_name] = 0

            final_group_names = []
            group_mapping = {}

            for group_id in sorted(group_info.keys()):
                group_counter[class_name] += 1

                # Create readable group names
                if "bomb" in class_name.lower():
                    group_name = f"bomb#{group_counter[class_name]}"
                elif "dark" in class_name.lower():
                    group_name = (
                        f"dark#{chr(64 + group_counter[class_name])}"  # A, B, C, ...
                    )
                elif "sediment" in class_name.lower() or "sed" in class_name.lower():
                    group_name = f"sed#{group_counter[class_name]:02d}"
                else:
                    group_name = f"{class_name}_grp{group_counter[class_name]}"

                group_mapping[group_id] = group_name
                final_group_names.append(group_name)

            # Assign group names to pixels
            for i, group_id in enumerate(pixel_groups):
                if group_id in group_mapping:
                    all_groups.append(group_mapping[group_id])
                else:
                    # Dropped pixel (too small group, no merge possible)
                    all_groups.append(f"{class_name}_dropped")

            if not quiet:
                print(
                    f"   {class_name}: {len(roi_pixels)} pixels → {len(group_info)} groups via {method}"
                )
                for gname in final_group_names:
                    gid = [k for k, v in group_mapping.items() if v == gname][0]
                    print(f"      {gname}: {group_info[gid]['size']} pixels")

        return all_groups, list(set(all_groups))

    def train_svm_with_cv(
        self,
        training_rois,
        segment_start,
        segment_end,
        wavelength_range=None,
        cv_folds=5,
        use_corrected=True,
        svm_kernel="rbf",
        optimize_params=True,
        svm_C=1.0,
        svm_gamma="scale",
        use_spatial_groups=True,  # 🔥 NEW: Use spatial clustering for groups
        closing_radius=3,  # 🔥 NEW: Morphological closing radius (bombs/dark)
        min_group_size=20,  # 🔥 NEW: Minimum pixels per group (bombs/dark)
        use_grid_for_sediment=True,  # 🔥 NEW: Use grid for sediment instead of clustering
        sediment_grid_tile_slit=100,  # 🔥 NEW: Grid tile size in slit direction
        sediment_grid_tile_track=200,  # 🔥 NEW: Grid tile size in track direction
        sediment_min_group_size=10,  # 🔥 NEW: Min pixels for sediment tiles
        subsample_per_group=None,  # 🔥 NEW: Max pixels per group (None = no limit)
        use_sample_weights=False,  # 🔥 NEW: Weight pixels by 1/group_size
        class_weight_dict=None,  # 🔥 NEW: Custom class weights (e.g., {"training_bombs": 2.0})
        add_brightness_feature=False,  # 🔥 NEW: Add brightness (mean intensity) as feature
        use_intensity_only=False,  # 🔥 NEW: Use only mean intensity instead of full spectrum
        remap_labels=None,  # 🔥 NEW: Dict mapping ROI names to final class names (e.g., {"training_bombs": "bombs", "validation_bombs": "bombs"})
        quiet=False,
    ):
        """
        Train SVM classifier with K-fold cross-validation on ROI pixels OUTSIDE the segment.

        This method:
        1. Extracts training pixels from ROIs that are OUTSIDE [segment_start, segment_end]
           OR ALL pixels if segment_start/segment_end are None
        2. Performs K-fold cross-validation to evaluate model performance
        3. Trains a final SVM model on ALL outside pixels (or all pixels if no segment defined)
        4. Returns the trained model + CV metrics + filtered ROI coordinates

        Args:
            training_rois: List of ROI names to use for training (e.g., ["training_dark", "training_sediment"])
            segment_start: Start track index of segment (pixels outside this will be used for training)
                          Set to None to use ALL pixels (no segment filtering)
            segment_end: End track index of segment (pixels outside this will be used for training)
                        Set to None to use ALL pixels (no segment filtering)
            wavelength_range: Tuple (min_wl, max_wl) to restrict wavelengths used (e.g., (500, 650))
                             If None, uses all wavelengths
            cv_folds: Number of folds for cross-validation (default: 5)
            use_corrected: Use corrected data (True) or raw data (False)
            svm_kernel: SVM kernel ('rbf', 'linear', 'poly', 'sigmoid')
            optimize_params: If True, use GridSearchCV to find best C and gamma
            svm_C: SVM penalty parameter (used if optimize_params=False)
            svm_gamma: SVM kernel coefficient (used if optimize_params=False)
            add_brightness_feature: Add mean intensity as additional feature (default: False)
            use_intensity_only: Use ONLY mean intensity (ignore spectrum). Cannot be True if add_brightness_feature=True (default: False)
            remap_labels: Optional dict mapping ROI names to final class names (e.g., {"training_bombs": "bombs", "validation_bombs": "bombs"})
                         This allows combining training and validation ROIs into single classes (default: None = use ROI names as-is)
            quiet: Suppress progress messages

        Returns:
            dict with keys:
                - 'model': Trained SVM model (SVC object)
                - 'label_encoder': LabelEncoder for class names
                - 'class_names': List of class names
                - 'cv_results': Dict with per-fold metrics
                    - 'accuracy': List of accuracy per fold
                    - 'precision': List of precision per fold (macro avg)
                    - 'recall': List of recall per fold (macro avg)
                    - 'f1': List of F1 per fold (macro avg)
                    - 'confusion_matrices': List of confusion matrices per fold
                - 'cv_mean_metrics': Dict with mean metrics across folds
                    - 'accuracy_mean', 'accuracy_std'
                    - 'precision_mean', 'precision_std'
                    - 'recall_mean', 'recall_std'
                    - 'f1_mean', 'f1_std'
                - 'best_params': Dict with best C and gamma (if optimize_params=True)
                - 'training_pixels_per_class': Dict with pixel counts per class
                - 'filtered_training_rois': Dict with ROI coordinates actually used (for visualization)
                - 'filtered_pixels_outside': Number of pixels kept (outside segment)
                - 'filtered_pixels_inside': Number of pixels rejected (inside segment)
        """
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import (
            GridSearchCV,
            GroupKFold,
        )  # 🔥 CHANGED: Removed StratifiedKFold, added GroupKFold
        from sklearn.metrics import (
            confusion_matrix,
            classification_report,
            accuracy_score,
            precision_recall_fscore_support,
        )
        import time

        # 🔥 Validation: Cannot use both intensity_only and add_brightness_feature
        if use_intensity_only and add_brightness_feature:
            raise ValueError(
                "Cannot use both use_intensity_only=True and add_brightness_feature=True. "
                "When using intensity-only mode, the feature IS the mean intensity."
            )

        if not quiet:
            print("=" * 60)
            print("🤖 SVM TRAINING WITH CROSS-VALIDATION")
            print("=" * 60)
            if use_intensity_only:
                print(
                    "💡 MODE: Intensity-only classification (mean intensity as single feature)"
                )
            elif add_brightness_feature:
                print("💡 MODE: Spectral + brightness feature")
            else:
                print("💡 MODE: Standard spectral classification")

        # Convert list of ROI names to dict
        if isinstance(training_rois, list):
            roi_dict = {}
            for roi_name in training_rois:
                if roi_name in self.roi_collection:
                    roi_dict[roi_name] = self.roi_collection[roi_name]
                else:
                    print(f"⚠️  Warning: ROI '{roi_name}' not found in roi_collection")
            training_rois = roi_dict

        if not training_rois:
            raise ValueError("No valid training ROIs provided")

        # Get datacube - ⚠️ CRITICAL: DO NOT create a copy here, just get reference
        # (Copying the full cube would waste huge amounts of memory)
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )

        if cube_data is None:
            raise ValueError(
                "No data loaded. Call cube.load_data() or cube.apply_illumination_correction_v2() first."
            )

        n_tracks, n_slits, n_wavelengths = cube_data.shape

        # Filter wavelengths if range specified
        if wavelength_range is not None:
            wl_min, wl_max = wavelength_range
            wl_mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
            wl_indices = np.where(wl_mask)[0]
            cube_data = cube_data[:, :, wl_indices]
            wavelengths_used = self.wavelengths[wl_indices]
            n_wavelengths = len(wl_indices)

            if not quiet:
                print(f"\n� Wavelength filtering:")
                print(f"   Range: {wl_min} - {wl_max} nm")
                print(
                    f"   Wavelengths used: {n_wavelengths} (from {len(self.wavelengths)})"
                )
        else:
            wl_indices = np.arange(len(self.wavelengths))
            wavelengths_used = (
                self.wavelengths.copy()
            )  # ✅ FIX: Create copy to avoid reference issues!

        if not quiet:
            print(f"\n�📦 Datacube shape: {cube_data.shape}")
            print(f"📏 Using data: {'corrected' if use_corrected else 'raw'}")
            if segment_start is None or segment_end is None:
                print(f"🎯 Segment range: NONE (using ALL pixels from ROIs)")
            else:
                print(f"🎯 Segment range: tracks {segment_start} to {segment_end}")
            print(f"📍 Training ROIs: {list(training_rois.keys())}")
            if add_brightness_feature:
                print(
                    f"💡 Brightness feature: ENABLED (mean of wavelength-filtered spectrum)"
                )

        # Store wavelength info for classification
        self.svm_wavelength_indices = wl_indices
        self.svm_wavelengths = wavelengths_used
        self.svm_add_brightness_feature = (
            add_brightness_feature  # Store for classification
        )
        self.svm_use_intensity_only = (
            use_intensity_only  # 🔥 NEW: Store for classification
        )

        # Extract training pixels OUTSIDE segment
        X_train = []
        y_train = []
        training_pixel_counts = {}
        filtered_training_rois = {}
        filtered_training_rois_for_grouping = (
            {}
        )  # Track which pixels are kept per class
        pixels_kept_outside = 0
        pixels_rejected_inside = 0

        if not quiet:
            print(f"\n🔍 Filtering ROI pixels...")

        for class_name, roi_pixels in training_rois.items():
            class_pixels = []
            filtered_pixels = []

            for slit_idx, track_idx in roi_pixels:  # ROIs stored as (slit, track)
                # Check if pixel is valid for training
                # If segment_start/end are None, use ALL pixels
                # Otherwise, use pixels OUTSIDE segment
                if segment_start is None or segment_end is None:
                    # No segment filtering - use ALL pixels
                    is_valid_pixel = True
                else:
                    # Use pixels OUTSIDE segment
                    is_valid_pixel = (
                        track_idx < segment_start or track_idx > segment_end
                    )

                if is_valid_pixel:
                    # Valid training pixel
                    if 0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits:
                        spectrum = cube_data[
                            track_idx, slit_idx, :
                        ].copy()  # ✅ FIX: Create copy to prevent modifying original datacube!
                        class_pixels.append(spectrum)
                        filtered_pixels.append(
                            (slit_idx, track_idx)
                        )  # Store as (slit, track)
                        pixels_kept_outside += 1
                else:
                    # Pixel is inside segment - reject it
                    pixels_rejected_inside += 1

            if len(class_pixels) > 0:
                # 🔥 NEW: Apply label remapping if provided
                final_label = (
                    remap_labels[class_name]
                    if remap_labels and class_name in remap_labels
                    else class_name
                )

                X_train.extend(class_pixels)
                y_train.extend([final_label] * len(class_pixels))

                # Track pixel counts by FINAL label (after remapping)
                if final_label not in training_pixel_counts:
                    training_pixel_counts[final_label] = 0
                training_pixel_counts[final_label] += len(class_pixels)

                # Keep filtered ROIs by ORIGINAL ROI name (for visualization)
                filtered_training_rois[class_name] = filtered_pixels

                # For spatial grouping, use FINAL label
                if final_label not in filtered_training_rois_for_grouping:
                    filtered_training_rois_for_grouping[final_label] = []
                filtered_training_rois_for_grouping[final_label].extend(filtered_pixels)

                if not quiet:
                    remap_msg = (
                        f" → {final_label}"
                        if remap_labels and class_name in remap_labels
                        else ""
                    )
                    if segment_start is None or segment_end is None:
                        print(f"   {class_name}{remap_msg}: {len(class_pixels)} pixels")
                    else:
                        print(
                            f"   {class_name}{remap_msg}: {len(class_pixels)} pixels (rejected {len(roi_pixels) - len(class_pixels)} inside segment)"
                        )
            else:
                if segment_start is None or segment_end is None:
                    print(
                        f"   ⚠️  Warning: No valid training pixels for class '{class_name}'!"
                    )
                else:
                    print(
                        f"   ⚠️  Warning: No valid training pixels for class '{class_name}' outside segment!"
                    )

        if not X_train:
            if segment_start is None or segment_end is None:
                raise ValueError("No training pixels found in ROIs!")
            else:
                raise ValueError("No training pixels found outside segment!")

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 🔥 NEW: Intensity-only mode - replace full spectrum with mean intensity
        if use_intensity_only:
            intensity = np.mean(X_train, axis=1, keepdims=True)  # Shape: (n_pixels, 1)
            X_train = intensity  # Replace spectrum with single intensity value
            if not quiet:
                print(f"\n💡 Intensity-only mode activated:")
                print(
                    f"   Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]"
                )
                print(
                    f"   Intensity mean: {intensity.mean():.4f} ± {intensity.std():.4f}"
                )
                print(
                    f"   Feature dimensionality: {X_train.shape[1]} (mean intensity only)"
                )

        # 🔥 NEW: Add brightness feature (mean intensity across wavelength-filtered spectrum)
        elif add_brightness_feature:
            brightness = np.mean(X_train, axis=1, keepdims=True)  # Shape: (n_pixels, 1)
            X_train = np.hstack([X_train, brightness])  # Append as last column
            if not quiet:
                print(f"\n💡 Brightness feature added:")
                print(
                    f"   Brightness range: [{brightness.min():.4f}, {brightness.max():.4f}]"
                )
                print(
                    f"   Brightness mean: {brightness.mean():.4f} ± {brightness.std():.4f}"
                )

        if not quiet:
            print(
                f"\n✅ Training data: {X_train.shape[0]} pixels, {X_train.shape[1]} features"
            )
            if use_intensity_only:
                print(f"   Features: 1 mean intensity value")
            elif add_brightness_feature:
                print(f"   Features: {X_train.shape[1]-1} wavelengths + 1 brightness")
            else:
                print(f"   Features: {X_train.shape[1]} wavelengths")
            print(f"   Kept (outside segment): {pixels_kept_outside}")
            print(f"   Rejected (inside segment): {pixels_rejected_inside}")

        # Create spatial groups for CV
        if use_spatial_groups:
            groups, unique_group_names = self._create_spatial_groups(
                filtered_training_rois_for_grouping,
                cube_data.shape,
                closing_radius=closing_radius,
                min_group_size=min_group_size,
                use_grid_for_sediment=use_grid_for_sediment,  # 🔥 NEW
                sediment_grid_tile_slit=sediment_grid_tile_slit,  # 🔥 NEW
                sediment_grid_tile_track=sediment_grid_tile_track,  # 🔥 NEW
                sediment_min_group_size=sediment_min_group_size,  # 🔥 NEW
                quiet=quiet,
            )
            groups = np.array(groups)
        else:
            # Simple grouping by class name (old behavior - not recommended)
            groups = y_train.copy()
            if not quiet:
                print(f"\n⚠️  Using simple class-based grouping (spatial_groups=False)")

        # Subsample per group if requested (prevent large groups from dominating)
        if subsample_per_group is not None:
            if not quiet:
                print(
                    f"\n✂️  Subsampling: max {subsample_per_group} pixels per group..."
                )

            keep_indices = []
            for group_name in np.unique(groups):
                group_mask = groups == group_name
                group_indices = np.where(group_mask)[0]

                if len(group_indices) > subsample_per_group:
                    # Randomly sample subsample_per_group pixels
                    np.random.seed(42)
                    sampled_indices = np.random.choice(
                        group_indices, subsample_per_group, replace=False
                    )
                    keep_indices.extend(sampled_indices)
                    if not quiet:
                        print(
                            f"   {group_name}: {len(group_indices)} → {subsample_per_group} pixels"
                        )
                else:
                    keep_indices.extend(group_indices)

            keep_indices = np.array(keep_indices)
            X_train = X_train[keep_indices]
            y_train = y_train[keep_indices]
            groups = groups[keep_indices]

            if not quiet:
                print(f"   Total: {len(keep_indices)} pixels after subsampling")

        # Calculate sample weights if requested (weight by 1/group_size)
        sample_weights = None
        if use_sample_weights:
            if not quiet:
                print(f"\n⚖️  Using sample weights: weight = 1 / group_size")

            sample_weights = np.zeros(len(groups))
            for group_name in np.unique(groups):
                group_mask = groups == group_name
                group_size = np.sum(group_mask)
                sample_weights[group_mask] = 1.0 / group_size

            # Normalize so weights sum to number of samples
            sample_weights = (
                sample_weights * len(sample_weights) / np.sum(sample_weights)
            )

            if not quiet:
                print(
                    f"   Weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}"
                )

        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        class_names = le.classes_.tolist()

        if not quiet:
            print(f"\n📊 Final training data:")
            print(f"   Classes: {class_names}")
            print(f"   Total pixels: {len(X_train)}")
            print(f"   Unique spatial groups: {len(np.unique(groups))}")
            print(f"   Group names: {sorted(np.unique(groups).tolist())}")

        # Adjust cv_folds to number of unique groups (can't have more folds than groups)
        n_groups = len(np.unique(groups))
        if cv_folds > n_groups:
            if not quiet:
                print(
                    f"\n⚠️  WARNING: Requested {cv_folds} folds but only {n_groups} ROI groups available"
                )
                print(f"   Reducing to {n_groups}-fold CV (Leave-One-Group-Out)")
            cv_folds = n_groups

        # Cross-validation with GROUPED folds (prevent data leakage)
        if not quiet:
            print(f"\n🔄 Performing {cv_folds}-fold GROUPED cross-validation...")

        # 🔥 NEW: Custom fold generation to ensure all classes in validation
        def generate_balanced_folds(X, y, groups, class_names, n_splits, quiet=False):
            """
            Generate CV folds ensuring each fold has ALL classes in validation.
            Uses Leave-One-Bomb-Out strategy: each fold gets exactly one bomb group
            plus a mix of dark and sediment groups to ensure class balance.
            """
            from collections import Counter, defaultdict
            import random

            # Group pixels by group name AND class
            group_to_class = {}
            group_to_indices = defaultdict(list)

            for idx, (group_name, class_name) in enumerate(zip(groups, y)):
                group_to_indices[group_name].append(idx)
                if group_name not in group_to_class:
                    group_to_class[group_name] = class_name

            # Separate groups by class
            class_to_groups = defaultdict(list)
            for group_name, class_name in group_to_class.items():
                class_to_groups[class_name].append(group_name)

            if not quiet:
                print(f"\n   📊 Groups per class:")
                for class_name in sorted(class_to_groups.keys()):
                    print(
                        f"      {class_name}: {len(class_to_groups[class_name])} groups"
                    )

            # Find the minority class (typically bombs with 2 groups)
            minority_class = min(
                class_to_groups.keys(), key=lambda c: len(class_to_groups[c])
            )
            minority_groups = class_to_groups[minority_class]
            n_minority_groups = len(minority_groups)

            if not quiet:
                print(
                    f"   🎯 Minority class: {minority_class} ({n_minority_groups} groups)"
                )
                print(
                    f"   🔄 Using Leave-One-{minority_class.split('_')[-1].title()}-Out strategy"
                )

            # Adjust n_splits to number of minority groups (can't have more folds)
            actual_splits = min(n_splits, n_minority_groups)
            if actual_splits < n_splits:
                if not quiet:
                    print(
                        f"   ⚠️  Reducing folds: {n_splits} → {actual_splits} (limited by {minority_class})"
                    )

            # Generate folds
            folds = []
            random.seed(42)

            # Each fold gets exactly ONE minority group in validation
            for fold_idx in range(actual_splits):
                val_groups = [minority_groups[fold_idx]]  # One minority group

                # Add groups from other classes to validation (spread evenly)
                for class_name, groups_list in class_to_groups.items():
                    if class_name == minority_class:
                        continue  # Already handled

                    # Assign groups to val in round-robin fashion
                    for i, group_name in enumerate(groups_list):
                        if i % actual_splits == fold_idx:
                            val_groups.append(group_name)

                # Get indices for train/val
                val_indices = []
                train_indices = []

                for group_name, indices in group_to_indices.items():
                    if group_name in val_groups:
                        val_indices.extend(indices)
                    else:
                        train_indices.extend(indices)

                # Convert to integer arrays explicitly
                folds.append(
                    (
                        np.array(train_indices, dtype=np.int64),
                        np.array(val_indices, dtype=np.int64),
                    )
                )

            return folds, actual_splits

        # Generate custom folds
        custom_folds, actual_cv_folds = generate_balanced_folds(
            X_train, y_train, groups, class_names, cv_folds, quiet=quiet
        )
        cv_folds = actual_cv_folds

        cv_accuracy = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []
        cv_confusion_matrices = []

        for fold_idx, (train_idx, val_idx) in enumerate(custom_folds):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = (
                y_train_encoded[train_idx],
                y_train_encoded[val_idx],
            )
            groups_fold_train = groups[train_idx]
            groups_fold_val = groups[val_idx]

            # 🔥 CHECK: Skip fold if training or validation set is empty
            if len(X_fold_train) == 0:
                if not quiet:
                    print(
                        f"\n   ⚠️  Fold {fold_idx + 1}/{cv_folds}: SKIPPED (empty training set)"
                    )
                continue
            if len(X_fold_val) == 0:
                if not quiet:
                    print(
                        f"\n   ⚠️  Fold {fold_idx + 1}/{cv_folds}: SKIPPED (empty validation set)"
                    )
                continue

            # 🔥 NEW: Verify no ROI overlap between train/val (data leakage check)
            train_roi_set = set(groups_fold_train)
            val_roi_set = set(groups_fold_val)
            overlap = train_roi_set & val_roi_set

            # Class distribution
            from collections import Counter

            train_counts = Counter(y_train[train_idx])
            val_counts = Counter(y_train[val_idx])

            # Check if all classes present in validation
            val_classes = set(val_counts.keys())
            all_classes = set(class_names)
            missing_classes = all_classes - val_classes

            if not quiet:
                print(f"\n   📂 Fold {fold_idx + 1}/{cv_folds}:")
                print(f"      Train groups: {sorted(train_roi_set)}")
                print(f"      Val groups:   {sorted(val_roi_set)}")

                if overlap:
                    print(f"      ❌ OVERLAP DETECTED: {overlap} (DATA LEAKAGE!)")
                else:
                    print(f"      ✅ No overlap (clean split)")

                print(f"      Train samples: {dict(train_counts)}")
                print(f"      Val samples:   {dict(val_counts)}")

                if missing_classes:
                    print(f"      ⚠️  MISSING CLASSES IN VAL: {missing_classes}")

            # 🔥 NEW: REFUSE fold if validation is missing any class
            if missing_classes:
                error_msg = (
                    f"❌ FOLD {fold_idx + 1} INVALID: Validation set missing classes {missing_classes}!\n"
                    f"   This makes CV scores unreliable.\n"
                    f"   Try reducing cv_folds or use Leave-One-Bomb-Out strategy.\n"
                    f"   Current fold has: {sorted(val_roi_set)}"
                )
                raise ValueError(error_msg)

            # Train SVM on this fold
            if optimize_params and len(X_fold_train) > 0:
                param_grid = {
                    "C": [0.1, 1, 10, 100, 1000],
                    "gamma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, "scale", "auto"],
                }
                # 🔥 NEW: Use GroupKFold for inner CV in GridSearchCV, but only if enough groups
                n_train_groups = len(train_roi_set)
                n_train_samples = len(X_fold_train)

                # Check if we have enough samples for inner CV
                # For 2-fold CV: need at least 2 samples per class (1 per fold)
                # For safer CV: need at least 4 samples per class (2 per fold)
                min_samples_per_class = (
                    min(train_counts.values()) if train_counts else 0
                )
                can_do_grouped_cv = n_train_groups >= 3
                can_do_stratified_cv = (
                    min_samples_per_class >= 4
                )  # Need 2 per fold minimum

                if can_do_grouped_cv:
                    # Use GroupKFold for inner CV (grouped splitting)
                    inner_cv = GroupKFold(n_splits=min(3, n_train_groups))
                    grid_search = GridSearchCV(
                        SVC(
                            kernel=svm_kernel,
                            class_weight="balanced",
                            probability=True,
                            random_state=42,
                        ),
                        param_grid,
                        cv=inner_cv,
                        scoring="f1_macro",
                        n_jobs=-1,
                    )
                    grid_search.fit(
                        X_fold_train, y_fold_train, groups=groups_fold_train
                    )
                    fold_model = grid_search.best_estimator_
                elif can_do_stratified_cv:
                    # Use StratifiedKFold for inner CV (no group constraint)
                    grid_search = GridSearchCV(
                        SVC(
                            kernel=svm_kernel,
                            class_weight="balanced",
                            probability=True,
                            random_state=42,
                        ),
                        param_grid,
                        cv=2,  # Minimum CV splits
                        scoring="f1_macro",
                        n_jobs=-1,
                    )
                    grid_search.fit(
                        X_fold_train, y_fold_train
                    )  # No groups for inner CV
                    fold_model = grid_search.best_estimator_
                else:
                    # Not enough samples for inner CV - use default params
                    if not quiet:
                        print(
                            f"      ⚠️  Only {n_train_samples} samples ({n_train_groups} groups) - skipping inner CV, using default params"
                        )
                    fold_model = SVC(
                        kernel=svm_kernel,
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    )
                    fold_model.fit(X_fold_train, y_fold_train)
            else:
                fold_model = SVC(
                    kernel=svm_kernel,
                    C=svm_C,
                    gamma=svm_gamma,
                    class_weight="balanced",
                    probability=True,
                    random_state=42,  # 🔥 NEW: class_weight="balanced"
                )
                fold_model.fit(X_fold_train, y_fold_train)

            # Predict on validation set
            y_fold_pred = fold_model.predict(X_fold_val)

            # Calculate metrics
            fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
            fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(
                y_fold_val, y_fold_pred, average="macro", zero_division=0
            )
            fold_cm = confusion_matrix(y_fold_val, y_fold_pred)

            cv_accuracy.append(fold_accuracy)
            cv_precision.append(fold_precision)
            cv_recall.append(fold_recall)
            cv_f1.append(fold_f1)
            cv_confusion_matrices.append(fold_cm)

            if not quiet:
                print(
                    f"      📊 Results: Accuracy={fold_accuracy:.3f}, Precision={fold_precision:.3f}, Recall={fold_recall:.3f}, F1={fold_f1:.3f}"
                )

        # Calculate mean CV metrics
        cv_mean_metrics = {
            "accuracy_mean": np.mean(cv_accuracy),
            "accuracy_std": np.std(cv_accuracy),
            "precision_mean": np.mean(cv_precision),
            "precision_std": np.std(cv_precision),
            "recall_mean": np.mean(cv_recall),
            "recall_std": np.std(cv_recall),
            "f1_mean": np.mean(cv_f1),
            "f1_std": np.std(cv_f1),
        }

        if not quiet:
            print(f"\n📊 Cross-Validation Results (mean ± std):")
            print(
                f"   Accuracy:  {cv_mean_metrics['accuracy_mean']:.3f} ± {cv_mean_metrics['accuracy_std']:.3f}"
            )
            print(
                f"   Precision: {cv_mean_metrics['precision_mean']:.3f} ± {cv_mean_metrics['precision_std']:.3f}"
            )
            print(
                f"   Recall:    {cv_mean_metrics['recall_mean']:.3f} ± {cv_mean_metrics['recall_std']:.3f}"
            )
            print(
                f"   F1 Score:  {cv_mean_metrics['f1_mean']:.3f} ± {cv_mean_metrics['f1_std']:.3f}"
            )

        # Train final model on ALL training data
        if not quiet:
            print(f"\n🏋️  Training final SVM on all {len(X_train)} training pixels...")

        start_time = time.time()

        # 🔥 NEW: Setup class weights (custom dict or balanced)
        if class_weight_dict is not None:
            # Convert class names to encoded labels
            class_weight_encoded = {}
            for class_name, weight in class_weight_dict.items():
                if class_name in class_names:
                    encoded_label = le.transform([class_name])[0]
                    class_weight_encoded[encoded_label] = weight
                else:
                    print(
                        f"   ⚠️  Warning: Class '{class_name}' in class_weight_dict not found in training data"
                    )

            # Fill in any missing classes with weight 1.0
            for class_name in class_names:
                encoded_label = le.transform([class_name])[0]
                if encoded_label not in class_weight_encoded:
                    class_weight_encoded[encoded_label] = 1.0

            class_weight_setting = class_weight_encoded
            if not quiet:
                print(f"\n⚖️  Using custom class weights:")
                for class_name in class_names:
                    encoded_label = le.transform([class_name])[0]
                    weight = class_weight_encoded[encoded_label]
                    print(f"   {class_name}: {weight:.2f}x")
        else:
            class_weight_setting = "balanced"
            if not quiet:
                print(f"\n⚖️  Using balanced class weights (automatic)")

        if optimize_params:
            param_grid = {
                "C": [0.1, 1, 10, 100, 1000],
                "gamma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, "scale", "auto"],
            }
            # 🔥 NEW: Use GroupKFold for inner CV in final GridSearchCV
            n_unique_groups = len(np.unique(groups))
            if n_unique_groups >= 3:
                final_inner_cv = GroupKFold(n_splits=min(3, n_unique_groups))
                grid_search = GridSearchCV(
                    SVC(
                        kernel=svm_kernel,
                        class_weight=class_weight_setting,
                        probability=True,
                        random_state=42,
                    ),
                    param_grid,
                    cv=final_inner_cv,
                    scoring="f1_macro",
                    n_jobs=-1,
                )
                grid_search.fit(X_train, y_train_encoded, groups=groups)
            else:
                # Not enough groups - use standard CV (falls back to pixel-level for hyperparameter tuning only)
                if not quiet:
                    print(
                        f"   ⚠️  Only {n_unique_groups} ROI groups - using pixel-level CV for hyperparameter tuning"
                    )
                grid_search = GridSearchCV(
                    SVC(
                        kernel=svm_kernel,
                        class_weight=class_weight_setting,
                        probability=True,
                        random_state=42,
                    ),
                    param_grid,
                    cv=3,
                    scoring="f1_macro",
                    n_jobs=-1,
                )
                grid_search.fit(X_train, y_train_encoded)
            final_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            if not quiet:
                print(
                    f"   Best parameters: C={best_params['C']}, gamma={best_params['gamma']}"
                )
        else:
            final_model = SVC(
                kernel=svm_kernel,
                C=svm_C,
                gamma=svm_gamma,
                class_weight=class_weight_setting,
                probability=True,
                random_state=42,  # 🔥 NEW: class_weight="balanced"
            )
            final_model.fit(X_train, y_train_encoded)
            best_params = {"C": svm_C, "gamma": svm_gamma}

        training_time = time.time() - start_time

        # Store model in cube
        self.svm_model = final_model
        self.svm_label_encoder = le
        self.svm_class_names = class_names
        self.svm_training_segment = (segment_start, segment_end)

        # Store spatial groups for visualization
        if use_spatial_groups:
            # Create spatial_groups_roi_collection for plotting
            spatial_groups_roi = {}
            for i, (pixel, group_name) in enumerate(
                zip(filtered_training_rois_for_grouping, groups)
            ):
                if group_name not in spatial_groups_roi:
                    spatial_groups_roi[group_name] = []
                # Get the actual pixel coordinate from filtered_training_rois
                # Need to reconstruct which pixel this is
                pass

            # Better approach: reconstruct from groups array
            spatial_groups_roi_collection = {}
            pixel_idx = 0
            for class_name, class_pixels in filtered_training_rois.items():
                for pixel_coord in class_pixels:
                    if pixel_idx < len(groups):
                        group_name = groups[pixel_idx]
                        if group_name not in spatial_groups_roi_collection:
                            spatial_groups_roi_collection[group_name] = []
                        spatial_groups_roi_collection[group_name].append(pixel_coord)
                        pixel_idx += 1

            self.svm_spatial_groups = spatial_groups_roi_collection
        else:
            self.svm_spatial_groups = None

        if not quiet:
            print(f"   ✅ Training complete in {training_time:.2f} seconds")
            print("\n✅ SVM training with cross-validation complete!")
            print("=" * 60)

        return {
            "model": final_model,
            "label_encoder": le,
            "class_names": class_names,
            "cv_results": {
                "accuracy": cv_accuracy,
                "precision": cv_precision,
                "recall": cv_recall,
                "f1": cv_f1,
                "confusion_matrices": cv_confusion_matrices,
            },
            "cv_mean_metrics": cv_mean_metrics,
            "best_params": best_params,
            "training_pixels_per_class": training_pixel_counts,
            "filtered_training_rois": filtered_training_rois,
            "spatial_groups_roi_collection": (
                spatial_groups_roi_collection if use_spatial_groups else None
            ),  # 🔥 NEW
            "filtered_pixels_outside": pixels_kept_outside,
            "filtered_pixels_inside": pixels_rejected_inside,
            "training_time": training_time,
        }

    def _post_classification_filtering(
        self,
        classification_map,
        class_names,
        post_cc_bomb=True,
        post_cc_dark=True,
        post_cc_sediment=False,
        cc_connectivity=8,
        cc_min_area=20,
        morph_close_radius=1,
        morph_open_radius=1,
        merge_proximity_px=0,
        quiet=False,
    ):
        """
        Post-classification connected-component filtering to remove isolated pixels.

        This function applies morphological operations and connected-component filtering
        to clean up the classification results by removing small, isolated regions.

        Args:
            classification_map: 2D array of class names (track x slit)
            class_names: List of class names in the classification
            post_cc_bomb: Apply filtering to bomb class (default: True)
            post_cc_dark: Apply filtering to dark class (default: True)
            post_cc_sediment: Apply filtering to sediment class (default: False)
            cc_connectivity: Connectivity for connected components (4 or 8)
            cc_min_area: Minimum area (pixels) to keep a component
            morph_close_radius: Radius for morphological closing (0 to disable)
            morph_open_radius: Radius for morphological opening (0 to disable)
            merge_proximity_px: Merge components within this distance (0 to disable)
            quiet: Suppress progress messages

        Returns:
            Cleaned classification map (2D array of class names)
        """
        from scipy.ndimage import (
            label,
            binary_closing,
            binary_opening,
            binary_dilation,
            generate_binary_structure,
            iterate_structure,
        )

        # Create disk structuring element (replacement for skimage.morphology.disk)
        def create_disk(radius):
            """Create a disk-shaped structuring element."""
            if radius <= 0:
                return np.ones((1, 1), dtype=np.uint8)

            y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            disk_mask = x**2 + y**2 <= radius**2
            return disk_mask.astype(np.uint8)

        if not quiet:
            print(f"\n🧹 POST-CLASSIFICATION FILTERING")
            print(f"=" * 60)

        cleaned_map = classification_map.copy()

        # Determine which classes to filter
        filter_classes = {}
        for class_name in class_names:
            if "bomb" in class_name.lower() and post_cc_bomb:
                filter_classes[class_name] = "bomb"
            elif "dark" in class_name.lower() and post_cc_dark:
                filter_classes[class_name] = "dark"
            elif "sediment" in class_name.lower() and post_cc_sediment:
                filter_classes[class_name] = "sediment"

        if not filter_classes:
            if not quiet:
                print("   ⚠️  No classes selected for filtering")
            return cleaned_map

        if not quiet:
            print(f"   Classes to filter: {list(filter_classes.keys())}")
            print(f"   Connectivity: {cc_connectivity}")
            print(f"   Min area: {cc_min_area} pixels")
            print(
                f"   Morph close radius: {morph_close_radius} ({'disabled' if morph_close_radius == 0 else 'enabled'})"
            )
            print(
                f"   Morph open radius: {morph_open_radius} ({'disabled' if morph_open_radius == 0 else 'enabled'})"
            )
            print(
                f"   Merge proximity: {merge_proximity_px} px ({'disabled' if merge_proximity_px == 0 else 'enabled'})"
            )

        # Process each class
        for class_name, class_type in filter_classes.items():
            if not quiet:
                print(f"\n   🔍 Filtering {class_name} ({class_type})...")

            # 1. Create binary mask for this class
            binary_mask = (classification_map == class_name).astype(np.uint8)
            original_count = np.sum(binary_mask)

            if original_count == 0:
                if not quiet:
                    print(f"      ⚠️  No pixels found for {class_name}")
                continue

            # 2. Label connected components FIRST (before morphological operations)
            # This way we identify clusters based on original pixels, not eroded ones
            structure = np.ones((3, 3), dtype=int) if cc_connectivity == 8 else None
            labeled_mask, n_components = label(binary_mask, structure=structure)

            if not quiet:
                print(f"      Found {n_components} connected components")

            # 3. Filter components by area
            cleaned_mask = np.zeros_like(binary_mask)
            components_kept = 0
            components_removed = 0
            pixels_kept = 0
            pixels_removed = 0

            for component_id in range(1, n_components + 1):
                component_mask = labeled_mask == component_id
                component_area = np.sum(component_mask)

                if component_area >= cc_min_area:
                    cleaned_mask[component_mask] = 1
                    components_kept += 1
                    pixels_kept += component_area
                else:
                    components_removed += 1
                    pixels_removed += component_area

            if not quiet:
                print(
                    f"      Kept: {components_kept} components ({pixels_kept} pixels)"
                )
                print(
                    f"      Removed: {components_removed} components ({pixels_removed} pixels)"
                )

            # 4. Apply morphological operations AFTER filtering (optional cleanup)
            # Morphological closing (fills small holes in kept clusters)
            if morph_close_radius > 0:
                selem = create_disk(morph_close_radius)
                cleaned_mask = binary_closing(cleaned_mask, structure=selem).astype(
                    np.uint8
                )

            # Morphological opening (smooths edges of kept clusters)
            # WARNING: This can shrink clusters! Use with caution or disable.
            if morph_open_radius > 0:
                selem = create_disk(morph_open_radius)
                cleaned_mask = binary_opening(cleaned_mask, structure=selem).astype(
                    np.uint8
                )

            # 6. Merge nearby components (optional)
            if merge_proximity_px > 0 and components_kept > 1:
                # Dilate components and merge overlaps
                selem = create_disk(merge_proximity_px)
                dilated_mask = binary_dilation(cleaned_mask, structure=selem).astype(
                    np.uint8
                )

                # Re-label after dilation to merge components
                merged_labeled, n_merged = label(dilated_mask, structure=structure)

                # Apply merged labels back to original cleaned mask
                cleaned_mask = np.zeros_like(binary_mask)
                for component_id in range(1, n_merged + 1):
                    component_mask = merged_labeled == component_id
                    # Only keep pixels that were in original cleaned_mask
                    original_component_mask = (labeled_mask > 0) & component_mask
                    cleaned_mask[original_component_mask] = 1

                if not quiet:
                    print(f"      After merging: {n_merged} components")

            # 7. Write cleaned mask back to classification map
            # Pixels that don't pass the filter are RENAMED to filtered_* (not removed!)
            # FIXED: Use ORIGINAL classification_map, not cleaned_map!
            class_mask = classification_map == class_name
            # Pixels that should be removed: in original class but NOT in cleaned_mask
            pixels_to_remove = class_mask & (~cleaned_mask.astype(bool))
            n_pixels_to_remove = np.sum(pixels_to_remove)

            if n_pixels_to_remove > 0:
                # Create filtered class name: classified_bombs → filtered_bombs OR dark_bomb → filtered_dark_bomb
                if class_name.startswith("classified_"):
                    filtered_class_name = class_name.replace("classified_", "filtered_")
                else:
                    filtered_class_name = f"filtered_{class_name}"

                # Reclassify removed pixels with filtered_* name
                cleaned_map[pixels_to_remove] = filtered_class_name

                if not quiet:
                    print(
                        f"      ✂️  Reclassified {n_pixels_to_remove} pixels as {filtered_class_name}"
                    )

            final_count = np.sum(cleaned_mask)
            if not quiet:
                removed_pct = (
                    ((original_count - final_count) / original_count * 100)
                    if original_count > 0
                    else 0
                )
                print(
                    f"      Final: {final_count} pixels ({original_count - final_count} removed, {removed_pct:.1f}%)"
                )

        if not quiet:
            print(f"\n✅ Post-classification filtering complete")

        return cleaned_map

    def classify_segment_with_validation(
        self,
        segment_start,
        segment_end,
        validation_rois,
        validation_class_mapping=None,
        use_corrected=True,
        save_to_h5=True,
        dataset_name="svm_classification_validated",
        # 🔥 NEW: Post-classification filtering parameters
        apply_post_filtering=True,
        post_cc_bomb=True,
        post_cc_dark=True,
        post_cc_sediment=False,
        cc_connectivity=8,
        cc_min_area=20,
        morph_close_radius=1,
        morph_open_radius=1,
        merge_proximity_px=0,
        confidence_threshold=None,
        quiet=False,
    ):
        """
        Classify segment pixels and evaluate using validation ROIs INSIDE the segment.

        This method:
        1. Classifies all pixels in [segment_start, segment_end] using trained SVM
        2. (Optional) Applies post-classification filtering to remove isolated pixels
        3. Extracts validation pixels from ROIs that are INSIDE the segment
        4. Computes confusion matrix and per-class metrics on validation pixels
        5. Returns classification map + validation metrics + filtered validation ROI coordinates

        Args:
            segment_start: Start track index of segment to classify
            segment_end: End track index of segment to classify
            validation_rois: List of ROI names for validation (e.g., ["sediment", "dark spots", "all bombs"])
            validation_class_mapping: Dict mapping validation ROI names to training class names
                                     (e.g., {"sediment": "training_sediment", "dark spots": "training_dark"})
                                     If None, assumes validation ROI names match training class names
            use_corrected: Use corrected data (True) or raw data (False)
            save_to_h5: Save classification map to H5 files
            dataset_name: Name of dataset in H5 file
            apply_post_filtering: Apply post-classification connected-component filtering (default: True)
            post_cc_bomb: Filter bomb class to remove isolated pixels (default: True)
            post_cc_dark: Filter dark class to remove isolated pixels (default: True)
            post_cc_sediment: Filter sediment class to remove isolated pixels (default: False)
            cc_connectivity: Connectivity for connected components - 4 or 8 (default: 8)
            cc_min_area: Minimum component area in pixels (default: 20)
            morph_close_radius: Morphological closing radius in pixels, 0 to disable (default: 1)
            morph_open_radius: Morphological opening radius in pixels, 0 to disable (default: 1)
            merge_proximity_px: Merge components within this distance, 0 to disable (default: 0)
            quiet: Suppress progress messages

        Returns:
            dict with keys:
                - 'classification_map': 2D array of class names (track x slit)
                - 'classification_map_encoded': 2D array of class indices (track x slit)
                - 'class_names': List of class names
                - 'track_range': Tuple (segment_start, segment_end)
                - 'validation_metrics': Dict with validation results
                    - 'confusion_matrix': 3×3 confusion matrix
                    - 'accuracy': Overall accuracy on validation pixels
                    - 'precision_per_class': Dict of precision per class
                    - 'recall_per_class': Dict of recall per class
                    - 'f1_per_class': Dict of F1 per class
                    - 'support_per_class': Dict of pixel counts per class
                - 'filtered_validation_rois': Dict with ROI coordinates actually used (for visualization)
                - 'validation_pixels_inside': Number of validation pixels kept (inside segment)
                - 'validation_pixels_outside': Number of validation pixels rejected (outside segment)
        """
        from sklearn.metrics import (
            confusion_matrix,
            classification_report,
            accuracy_score,
            precision_recall_fscore_support,
        )
        import time

        if not quiet:
            print("=" * 60)
            print("🎯 SEGMENT CLASSIFICATION WITH VALIDATION")
            print("=" * 60)

        # Check if model is trained
        if not hasattr(self, "svm_model"):
            raise ValueError(
                "No SVM model found. Train a model first using train_svm_with_cv()"
            )

        # Get datacube
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )

        if cube_data is None:
            raise ValueError(
                "No data loaded. Call cube.load_data() or cube.apply_illumination_correction_v2() first."
            )

        n_tracks, n_slits, n_wavelengths = cube_data.shape

        # Apply same wavelength filtering as training
        if hasattr(self, "svm_wavelength_indices"):
            cube_data = cube_data[:, :, self.svm_wavelength_indices]
            if not quiet:
                print(
                    f"\n📊 Using wavelength subset: {len(self.svm_wavelength_indices)} wavelengths"
                )

        if not quiet:
            print(f"\n📦 Datacube shape: {cube_data.shape}")
            print(f"📏 Using data: {'corrected' if use_corrected else 'raw'}")
            print(f"🎯 Segment range: tracks {segment_start} to {segment_end}")

        # Extract segment
        if not quiet:
            print(f"\n🔪 Extracting segment...")

        segment_data = cube_data[segment_start : segment_end + 1, :, :]
        segment_shape = segment_data.shape

        if not quiet:
            print(f"   Segment shape: {segment_shape}")

        # Classify segment
        total_pixels = segment_shape[0] * segment_shape[1]
        if not quiet:
            print(f"\n🤖 Classifying {total_pixels} pixels...")

        start_time = time.time()

        X_classify = segment_data.reshape(-1, segment_data.shape[2])

        # 🔥 NEW: Intensity-only mode - replace full spectrum with mean intensity
        if hasattr(self, "svm_use_intensity_only") and self.svm_use_intensity_only:
            intensity = np.mean(
                X_classify, axis=1, keepdims=True
            )  # Shape: (n_pixels, 1)
            X_classify = intensity  # Replace spectrum with single intensity value
            if not quiet:
                print(
                    f"   💡 Intensity-only mode (range: [{intensity.min():.4f}, {intensity.max():.4f}])"
                )

        # 🔥 NEW: Add brightness feature if it was used during training
        elif (
            hasattr(self, "svm_add_brightness_feature")
            and self.svm_add_brightness_feature
        ):
            brightness = np.mean(
                X_classify, axis=1, keepdims=True
            )  # Shape: (n_pixels, 1)
            X_classify = np.hstack([X_classify, brightness])  # Append as last column
            if not quiet:
                print(
                    f"   💡 Added brightness feature (range: [{brightness.min():.4f}, {brightness.max():.4f}])"
                )

        # Classify with progress bar (batch processing for better visualization)
        batch_size = 10000  # Classify 10k pixels at a time
        n_batches = int(np.ceil(len(X_classify) / batch_size))
        y_pred_encoded = np.zeros(len(X_classify), dtype=int)

        # 🔥 NEW: Compute probabilities if confidence threshold is set
        if confidence_threshold is not None:
            y_pred_proba = np.zeros((len(X_classify), len(self.svm_class_names)))
        else:
            y_pred_proba = None

        if not quiet:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=len(X_classify), desc="   Classifying", unit="pixels", ncols=100
            )

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_classify))
            batch = X_classify[start_idx:end_idx]
            y_pred_encoded[start_idx:end_idx] = self.svm_model.predict(batch)

            # 🔥 NEW: Get probabilities if needed
            if confidence_threshold is not None:
                y_pred_proba[start_idx:end_idx] = self.svm_model.predict_proba(batch)

            if not quiet:
                progress_bar.update(len(batch))

        if not quiet:
            progress_bar.close()

        y_pred = self.svm_label_encoder.inverse_transform(y_pred_encoded)

        # 🔥 RENAME: training_* → classified_* (these are predictions, not training data!)
        y_pred_classified = np.array(
            [name.replace("training_", "classified_") for name in y_pred]
        )

        # 🔥 NEW: Apply confidence threshold
        confidence_map = None
        unknown_count = 0
        if confidence_threshold is not None:
            max_proba = y_pred_proba.max(axis=1)  # Best probability per pixel
            low_confidence_mask = max_proba < confidence_threshold
            y_pred_classified[low_confidence_mask] = "classified_unknown"
            unknown_count = low_confidence_mask.sum()

            # Store confidence map for visualization
            confidence_map = max_proba.reshape(segment_shape[0], segment_shape[1])

            if not quiet:
                print(f"\n🎲 Confidence Thresholding:")
                print(f"   Threshold: {confidence_threshold:.2f}")
                print(
                    f"   Unknown pixels: {unknown_count} ({100.0 * unknown_count / len(y_pred_classified):.1f}%)"
                )
                print(
                    f"   Confidence range: [{max_proba.min():.3f}, {max_proba.max():.3f}]"
                )

        classification_map = y_pred_classified.reshape(
            segment_shape[0], segment_shape[1]
        )
        classification_map_encoded = y_pred_encoded.reshape(
            segment_shape[0], segment_shape[1]
        )

        classification_time = time.time() - start_time

        if not quiet:
            print(f"   ✅ Classification complete in {classification_time:.2f} seconds")

            # Class distribution (before filtering)
            unique, counts = np.unique(y_pred_classified, return_counts=True)
            print(f"\n📊 Classification distribution (before filtering):")
            for class_name, count in zip(unique, counts):
                percentage = (count / len(y_pred_classified)) * 100
                print(f"   {class_name}: {count} pixels ({percentage:.1f}%)")

        # 🔥 NEW: Store "before filtering" version for comparison
        classification_map_before_filtering = classification_map.copy()
        classification_map_encoded_before_filtering = classification_map_encoded.copy()

        # 🔥 NEW: Apply post-classification filtering
        if apply_post_filtering:
            classification_map = self._post_classification_filtering(
                classification_map,
                self.svm_class_names,
                post_cc_bomb=post_cc_bomb,
                post_cc_dark=post_cc_dark,
                post_cc_sediment=post_cc_sediment,
                cc_connectivity=cc_connectivity,
                cc_min_area=cc_min_area,
                morph_close_radius=morph_close_radius,
                morph_open_radius=morph_open_radius,
                merge_proximity_px=merge_proximity_px,
                quiet=quiet,
            )

            # Update encoded map and y_pred for validation
            # Re-encode the cleaned classification map
            classification_map_encoded = np.zeros_like(classification_map, dtype=int)
            for idx, class_name in enumerate(self.svm_class_names):
                classification_map_encoded[classification_map == class_name] = idx

            # Update y_pred for validation
            y_pred = classification_map.flatten()
            y_pred_encoded = classification_map_encoded.flatten()

            if not quiet:
                # Class distribution (after filtering)
                unique, counts = np.unique(y_pred[y_pred != ""], return_counts=True)
                print(f"\n📊 Classification distribution (after filtering):")
                for class_name, count in zip(unique, counts):
                    if class_name:  # Skip empty strings
                        percentage = (count / len(y_pred)) * 100
                        print(f"   {class_name}: {count} pixels ({percentage:.1f}%)")

        # Validation on inside-segment ROIs
        if validation_rois:
            if not quiet:
                print(f"\n✅ Validating on ROIs inside segment...")

            # Convert list to dict
            if isinstance(validation_rois, list):
                val_roi_dict = {}
                for roi_name in validation_rois:
                    if roi_name in self.roi_collection:
                        val_roi_dict[roi_name] = self.roi_collection[roi_name]
                    else:
                        print(f"   ⚠️  Warning: Validation ROI '{roi_name}' not found")
                validation_rois = val_roi_dict

            # Extract validation pixels INSIDE segment
            X_val = []
            y_val_true = []
            filtered_validation_rois = {}
            val_pixels_inside = 0
            val_pixels_outside = 0

            for roi_name, roi_pixels in validation_rois.items():
                # Map validation ROI name to training class name
                if validation_class_mapping and roi_name in validation_class_mapping:
                    class_name = validation_class_mapping[roi_name]
                else:
                    class_name = roi_name  # Assume same name

                # Check if this class was in training
                if class_name not in self.svm_class_names:
                    print(
                        f"   ⚠️  Warning: ROI '{roi_name}' maps to class '{class_name}' which was not in training. Skipping."
                    )
                    continue

                class_pixels = []
                filtered_pixels = []

                for slit_idx, track_idx in roi_pixels:  # ROIs stored as (slit, track)
                    # Check if pixel is INSIDE segment
                    if segment_start <= track_idx <= segment_end:
                        if 0 <= track_idx < n_tracks and 0 <= slit_idx < n_slits:
                            spectrum = cube_data[track_idx, slit_idx, :]
                            class_pixels.append(spectrum)
                            filtered_pixels.append(
                                (slit_idx, track_idx)
                            )  # Store as (slit, track)
                            y_val_true.append(class_name)  # Use mapped class name
                            val_pixels_inside += 1
                    else:
                        # Pixel is outside segment - reject it
                        val_pixels_outside += 1

                if len(class_pixels) > 0:
                    X_val.extend(class_pixels)
                    filtered_validation_rois[roi_name] = (
                        filtered_pixels  # Store with original ROI name
                    )

                    if not quiet:
                        display_name = (
                            f"{roi_name} → {class_name}"
                            if validation_class_mapping
                            and roi_name in validation_class_mapping
                            else class_name
                        )
                        print(
                            f"   {display_name}: {len(class_pixels)} validation pixels (rejected {len(roi_pixels) - len(class_pixels)} outside segment)"
                        )
                else:
                    print(
                        f"   ⚠️  Warning: No validation pixels for '{class_name}' inside segment!"
                    )

            if X_val:
                X_val = np.array(X_val)
                y_val_true = np.array(y_val_true)

                # Predict on validation pixels
                y_val_pred_encoded = self.svm_model.predict(X_val)
                y_val_pred = self.svm_label_encoder.inverse_transform(
                    y_val_pred_encoded
                )

                # Compute metrics
                val_accuracy = accuracy_score(y_val_true, y_val_pred)
                val_cm = confusion_matrix(
                    y_val_true, y_val_pred, labels=self.svm_class_names
                )

                # Per-class metrics
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_val_true,
                    y_val_pred,
                    labels=self.svm_class_names,
                    zero_division=0,
                )

                val_metrics = {
                    "confusion_matrix": val_cm,
                    "accuracy": val_accuracy,
                    "precision_per_class": dict(zip(self.svm_class_names, precision)),
                    "recall_per_class": dict(zip(self.svm_class_names, recall)),
                    "f1_per_class": dict(zip(self.svm_class_names, f1)),
                    "support_per_class": dict(zip(self.svm_class_names, support)),
                }

                if not quiet:
                    print(f"\n📈 Validation Results:")
                    print(f"   Overall Accuracy: {val_accuracy:.3f}")
                    print(f"\n   Confusion Matrix:")
                    print(f"   Classes: {self.svm_class_names}")
                    print(val_cm)
                    print(f"\n   Per-Class Metrics:")
                    for class_name in self.svm_class_names:
                        print(
                            f"   {class_name}: P={val_metrics['precision_per_class'][class_name]:.3f}, "
                            f"R={val_metrics['recall_per_class'][class_name]:.3f}, "
                            f"F1={val_metrics['f1_per_class'][class_name]:.3f}, "
                            f"Support={val_metrics['support_per_class'][class_name]}"
                        )
            else:
                val_metrics = None
                filtered_validation_rois = {}
                val_pixels_inside = 0
                if not quiet:
                    print(f"   ⚠️  No validation pixels found inside segment")
        else:
            val_metrics = None
            filtered_validation_rois = {}
            val_pixels_inside = 0
            val_pixels_outside = 0

        # Save to H5
        if save_to_h5:
            if not quiet:
                print(f"\n💾 Saving classification to H5 files...")

            for geofile in self.geofiles:
                try:
                    with h5py.File(geofile.path, "a") as f:
                        ds_path = f"processed/{dataset_name}"

                        if ds_path in f:
                            del f[ds_path]

                        ds = f.create_dataset(
                            ds_path,
                            data=classification_map_encoded,
                            compression="gzip",
                        )
                        ds.attrs["class_names"] = self.svm_class_names
                        ds.attrs["track_start"] = segment_start
                        ds.attrs["track_end"] = segment_end

                        if not quiet:
                            print(f"   💾 Saved to: {geofile.name} → {ds_path}")

                except Exception as e:
                    print(f"   ⚠️  Could not save to {geofile.name}: {e}")

        # Store results
        self.svm_classification_map = classification_map
        self.svm_classification_map_encoded = classification_map_encoded
        self.svm_classification_range = (segment_start, segment_end)
        self.svm_validation_class_mapping = (
            validation_class_mapping  # Store mapping for plotting colors
        )

        # 🔥 FIX: Return classified_* names, not training_* names
        classified_class_names = [
            name.replace("training_", "classified_") for name in self.svm_class_names
        ]

        # 🔥 NEW: Add "classified_unknown" to class names if confidence threshold was used
        if confidence_threshold is not None and unknown_count > 0:
            classified_class_names.append("classified_unknown")

        results = {
            "classification_map": classification_map,
            "classification_map_encoded": classification_map_encoded,
            "classification_map_before_filtering": classification_map_before_filtering,  # 🔥 NEW
            "classification_map_encoded_before_filtering": classification_map_encoded_before_filtering,  # 🔥 NEW
            "class_names": classified_class_names,  # 🔥 FIX: Use renamed version
            "track_range": (segment_start, segment_end),
            "classification_time": classification_time,
            "validation_metrics": val_metrics,
            "filtered_validation_rois": filtered_validation_rois,
            "validation_pixels_inside": val_pixels_inside,
            "validation_pixels_outside": val_pixels_outside,
        }

        # 🔥 NEW: Add confidence info if threshold was used
        if confidence_threshold is not None:
            results["confidence_map"] = confidence_map
            results["unknown_count"] = unknown_count

        if not quiet:
            print("\n✅ Classification and validation complete!")
            print("=" * 60)

        return results

    # ============================================================================
    # 🔥 NEW: CLEANER WRAPPER FUNCTIONS FOR BETTER SEPARATION
    # ============================================================================

    def classify_segment(
        self,
        segment_start,
        segment_end,
        use_corrected=True,
        confidence_threshold=None,
        quiet=False,
    ):
        """
        Classify segment pixels using trained SVM model (NO post-filtering, NO validation).

        This is a CLEAN function that ONLY does classification.
        For filtering, use filter_classification() afterwards.
        For validation, use validate_classification() afterwards.

        Args:
            segment_start: Start track index of segment to classify
            segment_end: End track index of segment to classify
            use_corrected: Use corrected data (True) or raw data (False)
            confidence_threshold: Minimum probability threshold (0.0-1.0). If set, pixels
                                 with max probability below this become "classified_unknown".
                                 Default None (all pixels classified, no threshold).
            quiet: Suppress progress messages

        Returns:
            dict with keys:
                - 'classification_map': 2D array of class names (segment_size x n_slits)
                - 'classification_map_encoded': 2D array of encoded labels
                - 'class_names': List of class names (includes "classified_unknown" if threshold used)
                - 'segment_shape': Shape of classified segment
                - 'track_range': (segment_start, segment_end)
                - 'confidence_map': 2D array of max probabilities (only if threshold used)
                - 'unknown_count': Number of pixels classified as unknown (only if threshold used)
        """
        if not quiet:
            print("=" * 60)
            print("🎯 SEGMENT CLASSIFICATION (no filtering, no validation)")
            print("=" * 60)
            if confidence_threshold is not None:
                print(f"🎲 Confidence threshold: {confidence_threshold:.2f}")
                print(
                    f"   (Pixels with max_prob < {confidence_threshold:.2f} → classified_unknown)"
                )
                print("=" * 60)

        # Call the full function with filtering and validation disabled
        results = self.classify_segment_with_validation(
            segment_start=segment_start,
            segment_end=segment_end,
            validation_rois=None,  # No validation
            use_corrected=use_corrected,
            save_to_h5=False,  # Don't save (let user decide)
            apply_post_filtering=False,  # No filtering
            confidence_threshold=confidence_threshold,  # Pass through
            quiet=quiet,
        )

        # Get shape from classification map
        classification_map = results["classification_map_before_filtering"]
        segment_shape = classification_map.shape

        # Base return dict
        return_dict = {
            "classification_map": classification_map,
            "classification_map_encoded": results[
                "classification_map_encoded_before_filtering"
            ],
            "class_names": results["class_names"],
            "segment_shape": segment_shape,
            "track_range": results["track_range"],
        }

        # Add confidence info if threshold was used
        if confidence_threshold is not None:
            return_dict["confidence_map"] = results["confidence_map"]
            return_dict["unknown_count"] = results["unknown_count"]

        return return_dict

    def merge_unknown_to_sediment(
        self,
        classification_results,
        quiet=False,
    ):
        """
        Merge 'classified_unknown' pixels into 'classified_sediment'.

        Converts 4-class classification output (with confidence threshold) to 3-class
        output compatible with the standard pipeline. This allows you to:
        1. Run classify_segment with confidence_threshold to identify uncertain pixels
        2. Analyze/visualize the uncertainty (optional)
        3. Merge unknown pixels into sediment (conservative assumption)
        4. Continue with standard filtering/merge/validation pipeline

        Args:
            classification_results: Dict returned from classify_segment() with confidence_threshold
            quiet: Suppress progress messages

        Returns:
            dict: Modified classification_results with 3 classes
                - 'classification_map': Updated 2D array (unknown → sediment)
                - 'class_names': Updated list (3 classes, no 'classified_unknown')
                - 'merge_stats': Dict with pixel counts before/after merge
                - All other original keys preserved
        """
        if not quiet:
            print("=" * 60)
            print("🔀 MERGING UNKNOWN PIXELS TO SEDIMENT")
            print("=" * 60)

        # Copy results to avoid modifying original
        merged_results = classification_results.copy()
        classification_map = classification_results["classification_map"].copy()
        class_names = classification_results["class_names"].copy()

        # Check if unknown class exists
        if "classified_unknown" not in class_names:
            if not quiet:
                print("⚠️  No 'classified_unknown' class found - nothing to merge")
                print("   Returning original results unchanged")
            return classification_results

        # Count pixels before merge
        unknown_mask = classification_map == "classified_unknown"
        sediment_mask = classification_map == "classified_sediment"
        unknown_count = np.sum(unknown_mask)
        sediment_count_before = np.sum(sediment_mask)

        if not quiet:
            print(f"\n📊 Pixel counts BEFORE merge:")
            print(f"   classified_sediment: {sediment_count_before}")
            print(f"   classified_unknown:  {unknown_count}")

        # Merge: unknown → sediment
        classification_map[unknown_mask] = "classified_sediment"
        sediment_count_after = np.sum(classification_map == "classified_sediment")

        # Remove 'classified_unknown' from class names
        class_names_merged = [c for c in class_names if c != "classified_unknown"]

        if not quiet:
            print(f"\n📊 Pixel counts AFTER merge:")
            print(f"   classified_sediment: {sediment_count_after} (+{unknown_count})")
            print(f"\n✅ Merge complete! Classes: {class_names_merged}")

        # Update results dict
        merged_results["classification_map"] = classification_map
        merged_results["class_names"] = class_names_merged
        merged_results["merge_stats"] = {
            "unknown_pixels_merged": unknown_count,
            "sediment_count_before": sediment_count_before,
            "sediment_count_after": sediment_count_after,
        }

        return merged_results

    def filter_classification(
        self,
        classification_map,
        class_names,
        filter_bombs=True,
        filter_dark=True,
        filter_sediment=False,
        min_area_px=20,
        connectivity=8,
        morph_close_radius=1,
        morph_open_radius=1,
        merge_proximity_px=0,
        quiet=False,
    ):
        """
        Apply post-classification filtering to remove isolated pixels.

        This is a CLEAN function that ONLY does filtering (connected-component analysis).
        Use after classify_segment() to clean up noisy predictions.

        Args:
            classification_map: 2D array of class names from classify_segment()
            class_names: List of class names (from classify_segment())
            filter_bombs: Apply filtering to bomb pixels
            filter_dark: Apply filtering to dark spot pixels
            filter_sediment: Apply filtering to sediment pixels
            min_area_px: Minimum pixels per connected component (smaller = removed)
            connectivity: 4 or 8 (how pixels connect: orthogonal or diagonal)
            morph_close_radius: Fill small holes (0 = disabled)
            morph_open_radius: Remove small protrusions (0 = disabled)
            merge_proximity_px: Merge nearby components (0 = disabled)
            quiet: Suppress progress messages

        Returns:
            dict with keys:
                - 'filtered_map': 2D array of filtered class names
                - 'original_map': Original classification map (copy)
                - 'pixels_removed_per_class': Dict of removed pixel counts
        """
        if not quiet:
            print("=" * 60)
            print("🧹 POST-CLASSIFICATION FILTERING")
            print("=" * 60)

        # Map class names to filter flags
        filter_flags = {}
        for class_name in class_names:
            if "bomb" in class_name.lower():
                filter_flags[class_name] = filter_bombs
            elif "dark" in class_name.lower():
                filter_flags[class_name] = filter_dark
            elif "sediment" in class_name.lower():
                filter_flags[class_name] = filter_sediment
            else:
                filter_flags[class_name] = False  # Default: no filtering

        # Count original pixels
        original_counts = {}
        for class_name in class_names:
            original_counts[class_name] = np.sum(classification_map == class_name)

        # Apply filtering using internal function
        filtered_map = self._post_classification_filtering(
            classification_map,
            class_names,
            post_cc_bomb=filter_bombs,
            post_cc_dark=filter_dark,
            post_cc_sediment=filter_sediment,
            cc_connectivity=connectivity,
            cc_min_area=min_area_px,
            morph_close_radius=morph_close_radius,
            morph_open_radius=morph_open_radius,
            merge_proximity_px=merge_proximity_px,
            quiet=quiet,
        )

        # Count pixels in filtered map (including new filtered_* classes)
        pixels_removed = {}
        filtered_class_counts = {}

        # Get all unique classes in filtered map (includes filtered_* classes)
        all_classes_after = np.unique(filtered_map)

        for class_name in class_names:
            final_count = np.sum(filtered_map == class_name)
            pixels_removed[class_name] = original_counts[class_name] - final_count

            # Check if filtered version exists
            if class_name.startswith("classified_"):
                filtered_class_name = class_name.replace("classified_", "filtered_")
            else:
                filtered_class_name = f"filtered_{class_name}"
            if filtered_class_name in all_classes_after:
                filtered_class_counts[filtered_class_name] = np.sum(
                    filtered_map == filtered_class_name
                )

        if not quiet:
            print(f"\n📊 Filtering summary:")
            for class_name in class_names:
                if filter_flags[class_name]:
                    pct = (
                        (pixels_removed[class_name] / original_counts[class_name] * 100)
                        if original_counts[class_name] > 0
                        else 0
                    )
                    print(
                        f"   {class_name}: {pixels_removed[class_name]} pixels removed ({pct:.1f}%)"
                    )

                    # Show filtered class count
                    if class_name.startswith("classified_"):
                        filtered_class_name = class_name.replace(
                            "classified_", "filtered_"
                        )
                    else:
                        filtered_class_name = f"filtered_{class_name}"
                    if filtered_class_name in filtered_class_counts:
                        print(
                            f"      → Created {filtered_class_name}: {filtered_class_counts[filtered_class_name]} pixels"
                        )
                else:
                    print(f"   {class_name}: No filtering applied")

            print(f"\n📊 Final class distribution:")
            for class_name in sorted(all_classes_after):
                count = np.sum(filtered_map == class_name)
                print(f"   {class_name}: {count} pixels")

        return {
            "filtered_map": filtered_map,
            "original_map": classification_map.copy(),
            "pixels_removed_per_class": pixels_removed,
            "filtered_class_counts": filtered_class_counts,
            "all_classes": list(all_classes_after),
        }

    def merge_filtered_to_sediment(
        self,
        filtered_map,
        quiet=False,
    ):
        """
        Merge filtered_* classes and classified_sediment into new_sediment.

        This creates a 3-class map for validation by combining:
        - classified_sediment + filtered_bombs + filtered_dark → new_sediment
        - classified_bombs → stays as is
        - classified_dark → stays as is

        Args:
            filtered_map: 2D array from filter_classification() with 5 classes
            quiet: Suppress progress messages

        Returns:
            dict with keys:
                - 'validation_map': 2D array with 3 classes (for validation)
                - 'class_names': List of 3 class names
                - 'merge_summary': Dict showing what was merged
        """
        if not quiet:
            print("=" * 60)
            print("🔀 MERGING FILTERED CLASSES TO SEDIMENT")
            print("=" * 60)

        validation_map = filtered_map.copy()

        # Find all unique classes in input
        unique_classes = np.unique(filtered_map)

        if not quiet:
            print(f"\n📊 Before merge:")
            for class_name in sorted(unique_classes):
                count = np.sum(filtered_map == class_name)
                print(f"   {class_name}: {count} pixels")

        # Merge: classified_sediment + filtered_* → new_sediment
        merge_summary = {}
        total_merged = 0

        for class_name in unique_classes:
            if "sediment" in class_name.lower() or "filtered_" in class_name:
                pixels = np.sum(filtered_map == class_name)
                merge_summary[class_name] = pixels
                total_merged += pixels
                validation_map[filtered_map == class_name] = "new_sediment"

        # Final class names for validation
        final_classes = ["classified_bombs", "classified_dark", "new_sediment"]

        if not quiet:
            print(f"\n🔀 Merged into new_sediment:")
            for class_name, count in merge_summary.items():
                print(f"   {class_name}: {count} pixels")
            print(f"   ─────────────────────────")
            print(f"   Total: {total_merged} pixels")

            print(f"\n📊 After merge (3 classes for validation):")
            for class_name in final_classes:
                count = np.sum(validation_map == class_name)
                print(f"   {class_name}: {count} pixels")

        return {
            "validation_map": validation_map,
            "class_names": final_classes,
            "merge_summary": merge_summary,
        }

    def validate_classification(
        self,
        classification_map,
        segment_start,
        segment_end,
        validation_rois,
        validation_class_mapping=None,
        quiet=False,
    ):
        """
        Validate classification results using ground-truth ROIs.

        This is a CLEAN function that ONLY does validation (metrics calculation).
        Use after classify_segment() or filter_classification().

        Args:
            classification_map: 2D array of class names (from classify or filter)
            segment_start: Start track index of segment
            segment_end: End track index of segment
            validation_rois: List of ROI names or dict of ROI pixels
            validation_class_mapping: Dict mapping validation ROI names to training class names
            quiet: Suppress progress messages

        Returns:
            dict with keys:
                - 'accuracy': Overall accuracy
                - 'confusion_matrix': Confusion matrix
                - 'precision_per_class': Dict of precision per class
                - 'recall_per_class': Dict of recall per class
                - 'f1_per_class': Dict of F1 score per class
                - 'support_per_class': Dict of pixel counts per class
                - 'filtered_validation_rois': Dict of validation pixels used
        """
        from sklearn.metrics import (
            confusion_matrix,
            accuracy_score,
            precision_recall_fscore_support,
        )

        if not quiet:
            print("=" * 60)
            print("📊 CLASSIFICATION VALIDATION")
            print("=" * 60)

        # Convert validation ROIs if needed
        if isinstance(validation_rois, list):
            val_roi_dict = {}
            for roi_name in validation_rois:
                if roi_name in self.roi_collection:
                    val_roi_dict[roi_name] = self.roi_collection[roi_name]
            validation_rois = val_roi_dict

        # Filter validation pixels (keep only those INSIDE segment)
        filtered_val_rois = {}
        n_tracks = classification_map.shape[0]
        n_slits = classification_map.shape[1]

        for val_roi_name, val_pixels in validation_rois.items():
            filtered_pixels = []
            for slit_idx, track_idx in val_pixels:
                if segment_start <= track_idx <= segment_end:
                    rel_track = track_idx - segment_start
                    if 0 <= rel_track < n_tracks and 0 <= slit_idx < n_slits:
                        filtered_pixels.append((slit_idx, track_idx))
            if len(filtered_pixels) > 0:
                filtered_val_rois[val_roi_name] = filtered_pixels

        if not filtered_val_rois:
            if not quiet:
                print("⚠️  No validation pixels found inside segment!")
            return None

        # Map validation ROI names to training class names
        if validation_class_mapping is None:
            validation_class_mapping = {name: name for name in filtered_val_rois.keys()}

        # Extract predictions and ground truth
        y_val_pred = []
        y_val_true = []

        for val_roi_name, val_pixels in filtered_val_rois.items():
            training_class = validation_class_mapping.get(val_roi_name, val_roi_name)

            for slit_idx, track_idx in val_pixels:
                rel_track = track_idx - segment_start
                if 0 <= rel_track < n_tracks and 0 <= slit_idx < n_slits:
                    predicted_class = classification_map[rel_track, slit_idx]
                    y_val_pred.append(predicted_class)
                    y_val_true.append(training_class)

        # Determine unique classes in classification_map
        unique_pred_classes = np.unique(classification_map)
        unique_true_classes = np.unique(y_val_true)
        all_classes = sorted(set(unique_pred_classes) | set(unique_true_classes))

        if not quiet:
            print(f"\n   Classes in classification: {list(unique_pred_classes)}")
            print(f"   Classes in validation: {list(unique_true_classes)}")

        # Calculate metrics
        val_cm = confusion_matrix(y_val_true, y_val_pred, labels=all_classes)
        val_accuracy = accuracy_score(y_val_true, y_val_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val_true, y_val_pred, labels=all_classes, zero_division=0
        )

        val_metrics = {
            "accuracy": val_accuracy,
            "confusion_matrix": val_cm,
            "precision_per_class": dict(zip(all_classes, precision)),
            "recall_per_class": dict(zip(all_classes, recall)),
            "f1_per_class": dict(zip(all_classes, f1)),
            "support_per_class": dict(zip(all_classes, support)),
            "filtered_validation_rois": filtered_val_rois,
            "class_names": all_classes,
        }

        if not quiet:
            print(f"\n📈 Validation Results:")
            print(f"   Overall Accuracy: {val_accuracy:.3f}")
            print(f"\n   Per-Class Metrics:")
            for idx, class_name in enumerate(all_classes):
                print(
                    f"   {class_name}: P={precision[idx]:.3f}, "
                    f"R={recall[idx]:.3f}, "
                    f"F1={f1[idx]:.3f}, "
                    f"Support={support[idx]}"
                )

        return val_metrics

    # ============================================================================
    # END OF CLEANER WRAPPER FUNCTIONS
    # ============================================================================

    def plot_classification_map(
        self,
        coordinate_system="NED",
        figsize=(30, 10),
        cmap="tab10",
        show_legend=True,
        quiet=True,
    ):
        """
        Plot the classified map with class colors.

        Args:
            coordinate_system: 'NED', 'ECEF', or 'LATLON'
            figsize: Figure size (width, height)
            cmap: Colormap for classes
            show_legend: Show legend with class names
            quiet: Suppress plot_georef debug output

        Example:
            cube.plot_classification_map(coordinate_system="NED", figsize=(40, 10))
        """
        if not hasattr(self, "svm_classification_map_encoded"):
            raise ValueError(
                "No classification results found. Run classify_svm() first."
            )

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        track_start, track_end = self.svm_classification_range

        # Get georeferencing info (reuse plot_georef logic)
        # Note: track_start and track_end are absolute indices
        # ECEF arrays are 2D: (tracks, slits)
        X_ecef = self.X_ecef[track_start : track_end + 1, :]
        Y_ecef = self.Y_ecef[track_start : track_end + 1, :]
        Z_ecef = self.Z_ecef[track_start : track_end + 1, :]

        # Transform coordinates based on coordinate system
        if coordinate_system.upper() == "LATLON":
            from pyproj import Transformer

            tf_ecef_to_geo = Transformer.from_crs(
                "EPSG:4978", "EPSG:4979", always_xy=True
            )
            lon, lat, height = tf_ecef_to_geo.transform(X_ecef, Y_ecef, Z_ecef)
            Xp, Yp = lon, lat
            xlabel, ylabel = "Longitude (°)", "Latitude (°)"
        elif coordinate_system.upper() == "NED":
            # Get origin from config
            if (
                "config" in globals()
                and hasattr(config, "LAT0")
                and hasattr(config, "LON0")
            ):
                lat0 = float(config.LAT0)
                lon0 = float(config.LON0)
                h0 = float(getattr(config, "H0", 0.0))
            else:
                lat0, lon0, h0 = 60.8011575, 10.7122345, 0.0

            N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)
            Xp, Yp = E, N
            xlabel = f"East (m) from {lat0}°, {lon0}°"
            ylabel = "North (m)"
        elif coordinate_system.upper() == "ECEF":
            Xp, Yp = X_ecef, Y_ecef
            xlabel, ylabel = "ECEF X (m)", "ECEF Y (m)"
        else:
            raise ValueError("coordinate_system must be 'LATLON', 'NED', or 'ECEF'")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get classification map
        class_map = self.svm_classification_map_encoded

        # Create colormap and display labels
        n_classes = len(self.svm_class_names)
        colors = []
        display_labels = []

        # Use validation_class_mapping to find correct validation ROI colors
        # The mapping is: {"validation_roi_name": "training_class_name"}
        # We need reverse: {"training_class_name": "validation_roi_name"}
        reverse_mapping = {}
        if (
            hasattr(self, "svm_validation_class_mapping")
            and self.svm_validation_class_mapping
        ):
            reverse_mapping = {
                v: k for k, v in self.svm_validation_class_mapping.items()
            }

        for class_name in self.svm_class_names:
            # Create better display label: "Classified: Sediment" instead of "training_sediment"
            if class_name.startswith("training_"):
                clean_name = (
                    class_name.replace("training_", "").replace("_", " ").title()
                )
            else:
                clean_name = class_name.replace("_", " ").title()
            display_labels.append(f"Classified: {clean_name}")

            color_found = False

            # Priority 1: Use validation ROI color via reverse mapping
            # E.g., training_sediment → sediment, training_dark → "dark spots"
            if class_name in reverse_mapping:
                validation_roi_name = reverse_mapping[class_name]
                if validation_roi_name in self.roi_color_map:
                    colors.append(self.roi_color_map[validation_roi_name])
                    color_found = True

            # Priority 2: Training ROI color
            if not color_found and class_name in self.roi_color_map:
                colors.append(self.roi_color_map[class_name])
                color_found = True

            # Priority 3: Fallback to default colors
            if not color_found:
                if isinstance(cmap, str):
                    base_cmap = plt.cm.get_cmap(cmap, n_classes)
                    colors.append(base_cmap(len(colors)))
                else:
                    colors.append(
                        cmap[len(colors)] if len(colors) < len(cmap) else "gray"
                    )

        cmap_discrete = ListedColormap(colors[:n_classes])

        # Plot
        im = ax.pcolormesh(
            Xp,
            Yp,
            class_map,
            cmap=cmap_discrete,
            shading="auto",
            vmin=0,
            vmax=n_classes - 1,
        )

        # Legend with improved labels
        if show_legend:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=colors[i], label=display_labels[i])
                for i in range(n_classes)
            ]
            ax.legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=12,
                framealpha=0.9,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"SVM Classification Map - {self.name}\n{coordinate_system} coordinates"
        )
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

    def plot_classification_overlay(
        self,
        red_wl=654.2,
        green_wl=560,
        blue_wl=440.3,
        coordinate_system="NED",
        use_corrected=True,
        figsize=(30, 10),
        alpha=0.5,
        cmap="tab10",
        show_legend=True,
        quiet=True,
    ):
        """
        Plot RGB image with semi-transparent classification overlay.

        Args:
            red_wl, green_wl, blue_wl: Wavelengths for RGB
            coordinate_system: 'NED', 'ECEF', or 'LATLON'
            use_corrected: Use corrected data for RGB
            figsize: Figure size
            alpha: Transparency of classification overlay (0=transparent, 1=opaque)
            cmap: Colormap for classes
            show_legend: Show legend
            quiet: Suppress debug output

        Example:
            cube.plot_classification_overlay(
                coordinate_system="NED",
                alpha=0.6,
                figsize=(40, 10)
            )
        """
        if not hasattr(self, "svm_classification_map_encoded"):
            raise ValueError(
                "No classification results found. Run classify_svm() first."
            )

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        track_start, track_end = self.svm_classification_range

        # Get data
        cube_data = (
            self.data_corrected
            if (use_corrected and hasattr(self, "data_corrected"))
            else self.data
        )

        n_tracks, n_slits, n_wavelengths = cube_data.shape

        # Get segment (track_start and track_end are absolute indices)
        segment_data = cube_data[track_start : track_end + 1, :, :]

        # Create RGB (no transpose - keep as tracks x slits x wavelength)
        red_idx = np.argmin(np.abs(self.wavelengths - red_wl))
        green_idx = np.argmin(np.abs(self.wavelengths - green_wl))
        blue_idx = np.argmin(np.abs(self.wavelengths - blue_wl))

        R = segment_data[:, :, red_idx].copy()
        G = segment_data[:, :, green_idx].copy()
        B = segment_data[:, :, blue_idx].copy()

        # Normalize
        for C in (R, G, B):
            if C.max() != C.min():
                C[:] = (C - C.min()) / (C.max() - C.min())

        # RGB should be (T, S, 3) for pcolormesh
        RGB = np.dstack([R, G, B])

        # Get coordinates (absolute indices, ECEF arrays are 2D: tracks x slits)
        X_ecef = self.X_ecef[track_start : track_end + 1, :]
        Y_ecef = self.Y_ecef[track_start : track_end + 1, :]
        Z_ecef = self.Z_ecef[track_start : track_end + 1, :]

        # Transform coordinates based on coordinate system
        if coordinate_system.upper() == "LATLON":
            from pyproj import Transformer

            tf_ecef_to_geo = Transformer.from_crs(
                "EPSG:4978", "EPSG:4979", always_xy=True
            )
            lon, lat, height = tf_ecef_to_geo.transform(X_ecef, Y_ecef, Z_ecef)
            Xp, Yp = lon, lat
            xlabel, ylabel = "Longitude (°)", "Latitude (°)"
        elif coordinate_system.upper() == "NED":
            # Get origin from config
            if (
                "config" in globals()
                and hasattr(config, "LAT0")
                and hasattr(config, "LON0")
            ):
                lat0 = float(config.LAT0)
                lon0 = float(config.LON0)
                h0 = float(getattr(config, "H0", 0.0))
            else:
                lat0, lon0, h0 = 60.8011575, 10.7122345, 0.0

            N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)
            Xp, Yp = E, N
            xlabel = f"East (m) from {lat0}°, {lon0}°"
            ylabel = "North (m)"
        elif coordinate_system.upper() == "ECEF":
            Xp, Yp = X_ecef, Y_ecef
            xlabel, ylabel = "ECEF X (m)", "ECEF Y (m)"
        else:
            raise ValueError("coordinate_system must be 'LATLON', 'NED', or 'ECEF'")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Pad coordinates for pcolormesh (needs cell edges)
        Xc = np.pad(Xp, ((0, 1), (0, 1)), mode="edge")
        Yc = np.pad(Yp, ((0, 1), (0, 1)), mode="edge")

        # Plot RGB as background using pcolormesh (same as plot_georef)
        ax.pcolormesh(Xc, Yc, RGB, shading="flat")

        # Plot classification overlay
        class_map = self.svm_classification_map_encoded
        n_classes = len(self.svm_class_names)

        # Use same colors and labels as classification map
        colors = []
        display_labels = []

        # Use validation_class_mapping to find correct validation ROI colors
        reverse_mapping = {}
        if (
            hasattr(self, "svm_validation_class_mapping")
            and self.svm_validation_class_mapping
        ):
            reverse_mapping = {
                v: k for k, v in self.svm_validation_class_mapping.items()
            }

        for class_name in self.svm_class_names:
            # Create better display label: "Classified: Sediment" instead of "training_sediment"
            if class_name.startswith("training_"):
                clean_name = (
                    class_name.replace("training_", "").replace("_", " ").title()
                )
            else:
                clean_name = class_name.replace("_", " ").title()
            display_labels.append(f"Classified: {clean_name}")

            color_found = False

            # Priority 1: Use validation ROI color via reverse mapping
            if class_name in reverse_mapping:
                validation_roi_name = reverse_mapping[class_name]
                if validation_roi_name in self.roi_color_map:
                    colors.append(self.roi_color_map[validation_roi_name])
                    color_found = True

            # Priority 2: Training ROI color
            if not color_found and class_name in self.roi_color_map:
                colors.append(self.roi_color_map[class_name])
                color_found = True

            # Priority 3: Fallback to default colors
            if not color_found:
                if isinstance(cmap, str):
                    base_cmap = plt.cm.get_cmap(cmap, n_classes)
                    colors.append(base_cmap(len(colors)))
                else:
                    colors.append(
                        cmap[len(colors)] if len(colors) < len(cmap) else "gray"
                    )

        # Create overlay with alpha
        cmap_overlay = ListedColormap(colors[:n_classes])

        # Plot classification overlay using same padded coordinates
        ax.pcolormesh(
            Xc,
            Yc,
            class_map,
            cmap=cmap_overlay,
            shading="flat",
            vmin=0,
            vmax=n_classes - 1,
            alpha=alpha,
        )

        # Legend with improved labels
        if show_legend:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=colors[i], alpha=alpha, label=display_labels[i])
                for i in range(n_classes)
            ]
            ax.legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=12,
                framealpha=0.9,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"RGB + Classification Overlay - {self.name}\n"
            f"(R={red_wl}nm, G={green_wl}nm, B={blue_wl}nm, α={alpha})"
        )
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

    def apply_mnf_transform(self, use_corrected=True, n_components=20, quiet=False):
        """
        Apply MNF (Minimum Noise Fraction) transformation to hyperspectral data.
        MEMORY-EFFICIENT VERSION: Uses incremental PCA and chunked processing.

        MNF is a two-step process:
        1. Noise whitening: Decorrelate and scale noise based on noise covariance
        2. Signal transformation: PCA on whitened data to maximize SNR

        Parameters:
        -----------
        use_corrected : bool
            Use illumination-corrected data (default: True)
        n_components : int
            Number of MNF components to compute (default: 20)
        quiet : bool
            Suppress progress messages (default: False)

        Stores:
        -------
        self.mnf_data : ndarray
            Transformed data (n_tracks, n_slits, n_components)
        self.mnf_eigenvalues : ndarray
            MNF eigenvalues (signal-to-noise ratios)
        self.mnf_explained_variance : ndarray
            Fraction of total variance explained by each component
        self.mnf_cumulative_variance : ndarray
            Cumulative explained variance
        self.mnf_transformation_matrix : ndarray
            Combined transformation matrix (noise whitening + PCA)

        Example:
        --------
            cube.apply_mnf_transform(use_corrected=True, n_components=20)
            print(f"Top 3 components explain {cube.mnf_cumulative_variance[2]*100:.1f}% of variance")
        """
        from sklearn.decomposition import IncrementalPCA
        import numpy as np

        if not quiet:
            print("🔬 Computing MNF transformation (memory-efficient)...")
            print(f"   Using {'corrected' if use_corrected else 'raw'} data")

        # Get data source
        if (
            use_corrected
            and hasattr(self, "data_corrected")
            and self.data_corrected is not None
        ):
            data = self.data_corrected
            if not quiet:
                print(f"   ✅ Using corrected data")
        else:
            data = self.data
            if not quiet:
                print(f"   ⚠️  Using raw data (no correction available)")

        n_tracks, n_slits, n_bands = data.shape
        if not quiet:
            print(
                f"   Data shape: {n_tracks} tracks × {n_slits} slits × {n_bands} bands"
            )
            print(f"   💾 Memory-efficient mode: Processing in chunks")

        # Step 1: Estimate noise covariance (MEMORY EFFICIENT VERSION)
        if not quiet:
            print("\n📊 Step 1: Estimating noise covariance...")
            print("   Using difference between adjacent pixels (track direction)")

        # Compute differences along track direction WITHOUT loading all data
        noise_samples = []

        # Process in chunks to limit memory
        chunk_size = 500  # Process 500 tracks at a time
        for chunk_start in range(0, n_tracks - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_tracks - 1)

            # Load this chunk + next track
            data_chunk = data[chunk_start : chunk_end + 1, :, :].astype(np.float32)

            for i in range(chunk_end - chunk_start):
                # Get valid pixels for track i and i+1 within chunk
                track_i_data = data_chunk[i, :, :]
                track_i_next_data = data_chunk[i + 1, :, :]

                # Only use pixels where both are finite
                valid_i = np.isfinite(track_i_data).all(axis=1)
                valid_i_next = np.isfinite(track_i_next_data).all(axis=1)
                both_valid = valid_i & valid_i_next

                if both_valid.any():
                    # Compute differences (noise estimate)
                    diff = (
                        track_i_next_data[both_valid, :] - track_i_data[both_valid, :]
                    )

                    # Subsample to save memory (keep at most 50 samples per track pair)
                    if len(diff) > 50:
                        step = max(1, len(diff) // 50)
                        diff = diff[::step, :]

                    noise_samples.append(diff)

            # Free chunk memory
            del data_chunk

        # Combine noise samples
        noise_data = np.vstack(noise_samples)
        del noise_samples

        if not quiet:
            print(f"   Noise samples collected: {noise_data.shape[0]:,}")

        # Subsample if still too many (for covariance calculation)
        max_noise_samples = 5000  # Sufficient for good covariance estimate
        if noise_data.shape[0] > max_noise_samples:
            if not quiet:
                print(
                    f"   Subsampling to {max_noise_samples:,} samples for covariance..."
                )
            indices = np.random.choice(
                noise_data.shape[0], max_noise_samples, replace=False
            )
            noise_data = noise_data[indices, :]

        # Noise covariance matrix (differences have 2x noise variance)
        noise_cov = np.cov(noise_data.T.astype(np.float32)) / 2.0
        del noise_data

        # Step 2: Noise whitening transformation
        if not quiet:
            print("\n🔄 Step 2: Computing noise whitening transformation...")

        # Eigendecomposition of noise covariance
        noise_eigenvalues, noise_eigenvectors = np.linalg.eigh(noise_cov)

        # Sort in descending order
        idx = np.argsort(noise_eigenvalues)[::-1]
        noise_eigenvalues = noise_eigenvalues[idx]
        noise_eigenvectors = noise_eigenvectors[:, idx]

        # Whitening matrix: D^(-1/2) * V^T where noise_cov = V * D * V^T
        epsilon = 1e-10
        noise_inv_sqrt = np.diag(1.0 / np.sqrt(noise_eigenvalues + epsilon))
        whitening_matrix = (noise_inv_sqrt @ noise_eigenvectors.T).astype(np.float32)

        del noise_cov, noise_eigenvalues, noise_eigenvectors, noise_inv_sqrt

        # Step 3: Incremental PCA on whitened data (memory-efficient!)
        if not quiet:
            print("\n📈 Step 3: Incremental PCA on whitened data...")
            print(f"   Processing data in chunks to minimize memory usage...")

        ipca = IncrementalPCA(n_components=min(n_components, n_bands))

        # First pass: Fit the PCA model incrementally
        if not quiet:
            print("   Phase 3a: Fitting PCA model...")

        chunk_size = 500  # Process 500 tracks at a time
        n_chunks = (n_tracks + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_tracks)

            # Load chunk
            data_chunk = data[chunk_start:chunk_end, :, :].astype(np.float32)

            # Reshape to 2D and get valid pixels
            chunk_2d = data_chunk.reshape(-1, n_bands)
            valid_mask_chunk = np.isfinite(chunk_2d).all(axis=1)

            if valid_mask_chunk.any():
                chunk_valid = chunk_2d[valid_mask_chunk]

                # Apply whitening
                chunk_whitened = chunk_valid @ whitening_matrix.T

                # Partial fit
                ipca.partial_fit(chunk_whitened)

                del chunk_valid, chunk_whitened

            del data_chunk, chunk_2d, valid_mask_chunk

            if not quiet and (chunk_idx + 1) % 5 == 0:
                print(f"      Processed {chunk_idx + 1}/{n_chunks} chunks")

        if not quiet:
            print(f"   Phase 3b: Transforming data to MNF space...")
            print(f"   MNF components: {ipca.n_components_}")
            print(
                f"   Explained variance (top 5): {ipca.explained_variance_ratio_[:5]}"
            )

        # Second pass: Transform data and store results
        mnf_data_3d = np.full(
            (n_tracks, n_slits, ipca.n_components_), np.nan, dtype=np.float32
        )

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_tracks)

            # Load chunk
            data_chunk = data[chunk_start:chunk_end, :, :].astype(np.float32)
            chunk_shape = data_chunk.shape

            # Reshape to 2D and get valid pixels
            chunk_2d = data_chunk.reshape(-1, n_bands)
            valid_mask_chunk = np.isfinite(chunk_2d).all(axis=1)

            if valid_mask_chunk.any():
                chunk_valid = chunk_2d[valid_mask_chunk]

                # Apply whitening + PCA transformation
                chunk_whitened = chunk_valid @ whitening_matrix.T
                chunk_mnf = ipca.transform(chunk_whitened).astype(np.float32)

                # Store results
                chunk_output = np.full(
                    (chunk_shape[0] * chunk_shape[1], ipca.n_components_),
                    np.nan,
                    dtype=np.float32,
                )
                chunk_output[valid_mask_chunk] = chunk_mnf
                chunk_output_3d = chunk_output.reshape(
                    chunk_shape[0], chunk_shape[1], ipca.n_components_
                )

                mnf_data_3d[chunk_start:chunk_end, :, :] = chunk_output_3d

                del (
                    chunk_valid,
                    chunk_whitened,
                    chunk_mnf,
                    chunk_output,
                    chunk_output_3d,
                )

            del data_chunk, chunk_2d, valid_mask_chunk

            if not quiet and (chunk_idx + 1) % 5 == 0:
                print(f"      Transformed {chunk_idx + 1}/{n_chunks} chunks")

        # Store results
        self.mnf_data = mnf_data_3d
        self.mnf_eigenvalues = ipca.explained_variance_
        self.mnf_explained_variance = ipca.explained_variance_ratio_
        self.mnf_cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
        self.mnf_transformation_matrix = whitening_matrix.T @ ipca.components_.T
        self.mnf_pca_model = ipca
        self.mnf_whitening_matrix = whitening_matrix

        if not quiet:
            print(f"\n✅ MNF transformation complete!")
            print(f"   Output shape: {mnf_data_3d.shape}")
            print(
                f"   Top 3 components explain {self.mnf_cumulative_variance[2]*100:.1f}% of variance"
            )

        return {
            "mnf_data": mnf_data_3d,
            "eigenvalues": self.mnf_eigenvalues,
            "explained_variance": self.mnf_explained_variance,
            "cumulative_variance": self.mnf_cumulative_variance,
        }

    def plot_mnf_rgb(
        self,
        components=[1, 2, 3],
        flip_axes=True,
        flip_horizontal=False,
        flip_vertical=False,
        crop_center_track=None,
        crop_center_slit=None,
        crop_width=None,
        crop_aspect_ratio=3.5,
        roi_collection=None,
        roi_marker_size=200,
        roi_legend_loc="outside",
        roi_marker_edgewidth=0,
        roi_legend_markersize=10,
        roi_show_numbers=False,
        figsize=(8, 8),
        title=None,
        percentile_stretch=(2, 98),
    ):
        """
        Plot MNF components as RGB composite (like plot_rgb but with MNF data).

        Parameters:
        -----------
        components : list of int
            Which MNF components to use for [Red, Green, Blue]. Default [1,2,3]
            uses MNF1=R, MNF2=G, MNF3=B. Component indices are 1-based.
        flip_axes : bool
            Transpose the image (swap track and slit axes)
        flip_horizontal : bool
            Mirror the image horizontally
        flip_vertical : bool
            Mirror the image vertically
        crop_center_track : int, optional
            Track index for crop center (in original coordinates)
        crop_center_slit : int, optional
            Slit index for crop center (in original coordinates)
        crop_width : int, optional
            Width of crop in slit pixels
        crop_aspect_ratio : float
            Physical aspect ratio (track:slit pixel size, default 3.5 for UHI)
        roi_collection : str, list, or dict, optional
            ROI collection to overlay. Can be "all", list of names, or dict
        roi_marker_size : int
            Size of ROI markers (default: 200)
        roi_legend_loc : str
            Legend location ("outside", "upper right", etc.)
        roi_marker_edgewidth : int
            Edge width of ROI markers (default: 0)
        roi_legend_markersize : int
            Size of markers in legend (default: 10)
        roi_show_numbers : bool
            Show ROI numbers on plot (default: False)
        figsize : tuple
            Figure size (width, height) in inches
        title : str, optional
            Plot title (auto-generated if None)
        percentile_stretch : tuple
            Percentile values for contrast stretching (default 2nd-98th percentile)

        Example:
        --------
            # Basic usage
            cube.plot_mnf_rgb(components=[1, 2, 3])

            # With cropping and ROIs
            cube.plot_mnf_rgb(
                components=[1, 3, 4],
                flip_axes=True,
                flip_horizontal=True,
                crop_center_track=5592,
                crop_center_slit=765,
                crop_width=500,
                roi_collection=roi_3,
                roi_legend_loc="outside"
            )
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if not hasattr(self, "mnf_data"):
            print("❌ MNF data not found! Run apply_mnf_transform() first.")
            return

        # Validate components
        max_comp = self.mnf_data.shape[2]
        for comp in components:
            if comp < 1 or comp > max_comp:
                print(f"❌ Component {comp} out of range! Available: 1-{max_comp}")
                return

        print(f"🎨 Plotting MNF RGB composite:")
        print(f"   Red   = MNF{components[0]}")
        print(f"   Green = MNF{components[1]}")
        print(f"   Blue  = MNF{components[2]}")

        # Extract MNF components (convert to 0-indexed)
        R = self.mnf_data[:, :, components[0] - 1].copy()
        G = self.mnf_data[:, :, components[1] - 1].copy()
        B = self.mnf_data[:, :, components[2] - 1].copy()

        # Stack into RGB
        rgb_image = np.stack([R, G, B], axis=2)
        n_tracks, n_slits = R.shape

        print(f"   Image shape: {n_tracks} tracks × {n_slits} slits")

        # Handle cropping
        crop_applied = False
        if (
            crop_center_track is not None
            and crop_center_slit is not None
            and crop_width is not None
        ):
            half_width_slit = crop_width // 2
            half_width_track = int(crop_width / crop_aspect_ratio / 2)

            crop_track_min = max(0, crop_center_track - half_width_track)
            crop_track_max = min(n_tracks, crop_center_track + half_width_track)
            crop_slit_min = max(0, crop_center_slit - half_width_slit)
            crop_slit_max = min(n_slits, crop_center_slit + half_width_slit)

            rgb_image = rgb_image[
                crop_track_min:crop_track_max, crop_slit_min:crop_slit_max, :
            ]
            crop_applied = True
            print(
                f"   📦 Cropped to: track [{crop_track_min}:{crop_track_max}], slit [{crop_slit_min}:{crop_slit_max}]"
            )

        # Apply flips
        if flip_axes:
            rgb_image = np.transpose(rgb_image, (1, 0, 2))
        if flip_horizontal:
            rgb_image = np.flip(rgb_image, axis=1)
        if flip_vertical:
            rgb_image = np.flip(rgb_image, axis=0)

        # Normalize each channel independently using percentile stretch
        rgb_normalized = np.zeros_like(rgb_image)
        for i in range(3):
            channel = rgb_image[:, :, i]
            valid = channel[np.isfinite(channel)]
            if len(valid) > 0:
                p_low, p_high = np.percentile(valid, percentile_stretch)
                channel_stretched = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
                rgb_normalized[:, :, i] = channel_stretched
                print(
                    f"   Channel {i} ({['Red','Green','Blue'][i]}): [{p_low:.3f}, {p_high:.3f}]"
                )

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        if crop_applied:
            if flip_axes:
                extent = [crop_slit_min, crop_slit_max, crop_track_min, crop_track_max]
            else:
                extent = [crop_track_min, crop_track_max, crop_slit_min, crop_slit_max]
        else:
            if flip_axes:
                extent = [0, n_slits, 0, n_tracks]
            else:
                extent = [0, n_tracks, 0, n_slits]

        ax.imshow(rgb_normalized, aspect="auto", origin="lower", extent=extent)

        if crop_applied:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        # Add ROIs if requested
        if roi_collection is not None and hasattr(self, "roi_collection"):
            # Determine which ROIs to plot
            if roi_collection == "all":
                rois_to_plot = self.roi_collection
            elif isinstance(roi_collection, list):
                rois_to_plot = {
                    name: self.roi_collection[name]
                    for name in roi_collection
                    if name in self.roi_collection
                }
            elif isinstance(roi_collection, dict):
                rois_to_plot = roi_collection
            else:
                rois_to_plot = {}

            # Plot each ROI
            for roi_name, roi_pixels in rois_to_plot.items():
                if crop_applied:
                    # Filter ROIs within crop
                    valid_rois = [
                        (slit, track)
                        for slit, track in roi_pixels
                        if crop_slit_min <= slit < crop_slit_max
                        and crop_track_min <= track < crop_track_max
                    ]
                else:
                    valid_rois = [
                        (slit, track)
                        for slit, track in roi_pixels
                        if 0 <= slit < n_slits and 0 <= track < n_tracks
                    ]

                if valid_rois:
                    roi_tracks = [track for slit, track in valid_rois]
                    roi_slits = [slit for slit, track in valid_rois]

                    # Get color (use same color method as plot_rgb)
                    color = self._get_roi_color(
                        roi_name, ["yellow", "cyan", "magenta"], None
                    )

                    # Transform coordinates
                    if flip_axes:
                        x_coords = roi_slits
                        y_coords = roi_tracks
                        if flip_horizontal:
                            if crop_applied:
                                x_coords = [
                                    crop_slit_min + crop_slit_max - x for x in x_coords
                                ]
                            else:
                                x_coords = [n_slits - 1 - x for x in x_coords]
                    else:
                        x_coords = roi_tracks
                        y_coords = roi_slits
                        if flip_horizontal:
                            if crop_applied:
                                x_coords = [
                                    crop_track_min + crop_track_max - x
                                    for x in x_coords
                                ]
                            else:
                                x_coords = [n_tracks - 1 - x for x in x_coords]

                    if flip_vertical:
                        if flip_axes:
                            if crop_applied:
                                y_coords = [
                                    crop_track_min + crop_track_max - y
                                    for y in y_coords
                                ]
                            else:
                                y_coords = [n_tracks - 1 - y for y in y_coords]
                        else:
                            if crop_applied:
                                y_coords = [
                                    crop_slit_min + crop_slit_max - y for y in y_coords
                                ]
                            else:
                                y_coords = [n_slits - 1 - y for y in y_coords]

                    ax.scatter(
                        x_coords,
                        y_coords,
                        c=color,
                        s=roi_marker_size,
                        marker="s",
                        edgecolors="black" if roi_marker_edgewidth > 0 else "none",
                        linewidths=roi_marker_edgewidth,
                        alpha=0.8,
                        label=f"{roi_name} ({len(valid_rois)})",
                    )

                    # Show ROI numbers if requested
                    if roi_show_numbers and len(valid_rois) > 0:
                        center_x = np.mean(x_coords)
                        center_y = np.mean(y_coords)
                        ax.text(
                            center_x,
                            center_y,
                            roi_name,
                            color="white",
                            fontsize=10,
                            ha="center",
                            va="center",
                            bbox=dict(
                                facecolor="black", alpha=0.5, edgecolor="none", pad=2
                            ),
                        )

            # Add legend
            if roi_legend_loc and roi_legend_loc != "none":
                if roi_legend_loc == "outside":
                    ax.legend(
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left",
                        markerscale=roi_legend_markersize / 100,
                    )
                else:
                    ax.legend(
                        loc=roi_legend_loc, markerscale=roi_legend_markersize / 100
                    )

        # Labels
        if flip_axes:
            ax.set_xlabel("Slit Pixel Index", fontsize=11)
            ax.set_ylabel("Track Index", fontsize=11)
        else:
            ax.set_xlabel("Track Index", fontsize=11)
            ax.set_ylabel("Slit Pixel Index", fontsize=11)

        if title:
            ax.set_title(title, fontsize=13, fontweight="bold")
        else:
            ax.set_title(
                f"MNF RGB Composite (MNF{components[0]}-{components[1]}-{components[2]})",
                fontsize=13,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

        print(f"\n✅ MNF RGB plot complete!")


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


# ============================================================================
# 🎨 HELPER FUNCTIONS FOR CLASSIFICATION VISUALIZATION
# ============================================================================


def plot_classification_map(
    cube,
    classification_map,
    class_names,
    track_start,
    track_end,
    figsize=(50, 20),
    class_name_mapping=None,
    show_pixel_counts=True,
    roi_legend_loc="upper left",
    roi_marker_size=1,
    roi_marker_edgewidth=0,
    roi_legend_markersize=50,
    roi_solid_pixel_size=1,
    **plot_kwargs,
):
    """
    Plot a classification map with customizable class names and legend.

    Parameters
    ----------
    cube : CombinedTransectCube
        The cube object with plot_georef method
    classification_map : np.ndarray
        2D array with class labels (strings)
    class_names : list
        List of class names present in the map
    track_start : int
        Starting track index
    track_end : int
        Ending track index
    figsize : tuple, optional
        Figure size (width, height)
    class_name_mapping : dict, optional
        Dictionary to rename classes for display, e.g.:
        {"classified_bombs": "bombs", "classified_dark": "dark", "new_sediment": "sediment"}
        If None, original names are used
    show_pixel_counts : bool, optional
        If True, show pixel counts in legend labels. Default True.
    roi_legend_loc : str, optional
        Legend location. Default "upper left"
    roi_marker_size : int, optional
        Marker size for ROI points. Default 1
    roi_marker_edgewidth : int, optional
        Edge width for ROI markers. Default 0
    roi_legend_markersize : int, optional
        Marker size in legend. Default 50
    roi_solid_pixel_size : int, optional
        Pixel size for solid overlay. Default 1
    **plot_kwargs : optional
        Additional arguments passed to plot_georef

    Returns
    -------
    None
        Displays the plot
    """
    import numpy as np

    # 🎨 SMART COLOR INHERITANCE: Map validation/training ROI colors to classification names
    # This ensures colors stay consistent across validation and classification plots
    roi_color_map = {}

    if hasattr(cube, "roi_color_map") and cube.roi_color_map:
        # Build mapping from class names to their display names (after mapping)
        for class_name in class_names:
            display_name = class_name
            if class_name_mapping and class_name in class_name_mapping:
                display_name = class_name_mapping[class_name]

            # Check if we can inherit color from validation/training ROIs
            # e.g., "classified_bombs" → look for "validation_bombs" or "training_bombs" color
            possible_sources = []

            # Extract base name (e.g., "bombs" from "classified_bombs")
            if class_name.startswith("classified_"):
                base_name = class_name.replace("classified_", "")
                possible_sources.extend(
                    [
                        f"validation_{base_name}",
                        f"training_{base_name}",
                        class_name,  # exact match
                        display_name,  # mapped name
                    ]
                )
            elif class_name.startswith("new_"):
                # "new_sediment" → look for "validation_sediment" or "training_sediment"
                base_name = class_name.replace("new_", "")
                possible_sources.extend(
                    [
                        f"validation_{base_name}",
                        f"training_{base_name}",
                        class_name,
                        display_name,
                    ]
                )
            else:
                possible_sources = [class_name, display_name]

            # Find first matching color
            for source_name in possible_sources:
                if source_name in cube.roi_color_map:
                    roi_color_map[display_name] = cube.roi_color_map[source_name]
                    break

    # Create ROI collection from classification map
    classification_rois = {}
    for class_name in class_names:
        mask = classification_map == class_name
        rows, cols = np.where(mask)
        pixels = [(col, row + track_start) for row, col in zip(rows, cols)]
        if len(pixels) > 0:
            # Apply name mapping if provided
            display_name = class_name
            if class_name_mapping and class_name in class_name_mapping:
                display_name = class_name_mapping[class_name]

            # Add pixel count to name if requested
            display_name_with_count = display_name
            if show_pixel_counts:
                display_name_with_count = f"{display_name} ({len(pixels)} px)"

            # 🔥 UPDATE roi_color_map: Copy color from base name to name with pixel count
            if display_name in roi_color_map:
                roi_color_map[display_name_with_count] = roi_color_map[display_name]

            classification_rois[display_name_with_count] = pixels

    print(f"📊 Plotting classification map...")
    print(f"   Classes: {list(classification_rois.keys())}")

    # Debug: Show inherited colors
    if roi_color_map:
        print(f"   🎨 Inherited colors from validation/training ROIs:")
        for display_name_with_count in classification_rois.keys():
            if display_name_with_count in roi_color_map:
                print(
                    f"      {display_name_with_count}: {roi_color_map[display_name_with_count]}"
                )

    # Plot with plot_georef (pass inherited color map)
    cube.plot_georef(
        use_corrected=True,
        coordinate_system="NED",
        track_start=track_start,
        track_end=track_end,
        figsize=figsize,
        roi_collection=classification_rois,
        roi_color_map=(
            roi_color_map if roi_color_map else None
        ),  # 🎨 Pass inherited colors
        roi_marker_size=roi_marker_size,
        roi_legend_loc=roi_legend_loc,
        roi_marker_edgewidth=roi_marker_edgewidth,
        roi_legend_markersize=roi_legend_markersize,
        roi_legend_show_counts=False,  # ✅ Don't add counts in plot_georef (already added by show_pixel_counts)
        roi_solid_pixel_size=roi_solid_pixel_size,
        roi_overlay_mode="solid",
        **plot_kwargs,
    )


def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    class_name_mapping=None,
    figsize=(10, 8),
    cmap="Greens",
    title="Confusion Matrix",
    normalize=True,
    show_counts=True,
    fontsize_labels=12,
    fontsize_title=14,
    fontsize_cells=12,
    show_colorbar=True,
):
    """
    Plot a confusion matrix with customizable appearance and class names.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        2D confusion matrix (rows=true class, cols=predicted class)
    class_names : list
        List of class names corresponding to matrix rows/cols
    class_name_mapping : dict, optional
        Dictionary to rename classes for display, e.g.:
        {"classified_bombs": "bombs", "classified_dark": "dark", "new_sediment": "sediment"}
        If None, original names are used
    figsize : tuple, optional
        Figure size (width, height). Default (10, 8)
    cmap : str, optional
        Colormap name. Default "Greens"
    title : str, optional
        Plot title. Default "Confusion Matrix"
    normalize : bool, optional
        If True, also print normalized confusion matrix (percentages by row).
        Default True
    show_counts : bool, optional
        If True, show pixel counts in cells. Default True
    fontsize_labels : int, optional
        Font size for axis labels. Default 12
    fontsize_title : int, optional
        Font size for title. Default 14
    fontsize_cells : int, optional
        Font size for cell text. Default 12
    show_colorbar : bool, optional
        If True, show colorbar. Default True

    Returns
    -------
    fig, ax : matplotlib figure and axes
        The figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Apply class name mapping if provided
    display_names = []
    for name in class_names:
        if class_name_mapping and name in class_name_mapping:
            display_names.append(class_name_mapping[name])
        else:
            display_names.append(name)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix as image
    im = ax.imshow(confusion_matrix, cmap=cmap, aspect="auto")

    # Add colorbar if requested
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Pixel Count", fontsize=fontsize_labels)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(display_names)))
    ax.set_yticks(np.arange(len(display_names)))
    ax.set_xticklabels(display_names, fontsize=fontsize_labels)
    ax.set_yticklabels(display_names, fontsize=fontsize_labels)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations if requested
    if show_counts:
        for i in range(len(display_names)):
            for j in range(len(display_names)):
                text = ax.text(
                    j,
                    i,
                    f"{confusion_matrix[i, j]}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=fontsize_cells,
                )

    ax.set_xlabel("Predicted Class", fontsize=fontsize_labels)
    ax.set_ylabel("True Class", fontsize=fontsize_labels)
    ax.set_title(title, fontsize=fontsize_title, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Print normalized confusion matrix (percentages) if requested
    if normalize:
        cm_normalized = (
            confusion_matrix.astype("float")
            / confusion_matrix.sum(axis=1)[:, np.newaxis]
            * 100
        )

        print(f"\n📊 Confusion Matrix (Normalized by True Class - Row %):")
        print(f"{'':20s}", end="")
        for pred_class in display_names:
            print(f"{pred_class:>20s}", end="")
        print()
        print("-" * (20 + 20 * len(display_names)))

        for i, true_class in enumerate(display_names):
            print(f"{true_class:20s}", end="")
            for j in range(len(display_names)):
                print(f"{cm_normalized[i, j]:19.1f}%", end="")
            print()

    return fig, ax


def compute_confusion_matrix_metrics(
    confusion_matrix,
    class_names,
    validation_results=None,
    class_name_mapping=None,
    print_analysis=True,
):
    """
    Compute comprehensive confusion matrix metrics and return structured data.

    This function calculates all standard metrics from a confusion matrix:
    - Raw confusion matrix (pixel counts)
    - Row/column totals
    - Producer's Accuracy (PA) = Recall per class
    - User's Accuracy (UA) = Precision per class
    - Normalized confusion matrix (percentages)
    - Overall Accuracy (OA)
    - Cohen's Kappa (κ)
    - Summary statistics

    Parameters
    ----------
    confusion_matrix : np.ndarray
        2D confusion matrix (rows=true class, cols=predicted class)
    class_names : list
        List of class names corresponding to matrix rows/cols
    validation_results : dict, optional
        Validation results dict from validate_classification() containing
        'f1_per_class' for F1 scores. If None, F1 scores won't be included.
    class_name_mapping : dict, optional
        Dictionary to rename classes for display, e.g.:
        {"classified_bombs": "bombs", "classified_dark": "dark", "new_sediment": "sediment"}
        If None, original names are used.
    print_analysis : bool, optional
        If True, print detailed analysis to console. Default True.

    Returns
    -------
    dict
        Dictionary containing all computed metrics:
        {
            'confusion_matrix': np.ndarray,  # Original confusion matrix
            'class_names': list,  # Original class names
            'display_names': list,  # Mapped class names (if mapping provided)
            'row_totals': np.ndarray,  # Pixels per true class
            'col_totals': np.ndarray,  # Pixels per predicted class
            'total_pixels': int,  # Total pixels
            'cm_normalized': np.ndarray,  # Normalized confusion matrix (percentages)
            'producers_accuracy': dict,  # PA per class {class_name: value}
            'users_accuracy': dict,  # UA per class {class_name: value}
            'overall_accuracy': float,  # Overall accuracy
            'cohens_kappa': float,  # Cohen's Kappa
            'kappa_interpretation': str,  # Kappa interpretation string
            'f1_per_class': dict,  # F1 scores per class (if validation_results provided)
        }

    Examples
    --------
    >>> # After running validation
    >>> metrics = compute_confusion_matrix_metrics(
    ...     confusion_matrix=validation_results["confusion_matrix"],
    ...     class_names=merge_results["class_names"],
    ...     validation_results=validation_results,
    ...     class_name_mapping={"classified_bombs": "bombs", "classified_dark": "dark"},
    ...     print_analysis=True
    ... )
    >>> # Access specific metrics
    >>> print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    >>> print(f"Kappa: {metrics['cohens_kappa']:.3f}")
    >>> for class_name in metrics['class_names']:
    ...     print(f"{class_name}: PA={metrics['producers_accuracy'][class_name]:.1%}")
    """
    import numpy as np
    from sklearn.metrics import cohen_kappa_score

    cm = confusion_matrix

    # Apply class name mapping if provided
    display_names = []
    for name in class_names:
        if class_name_mapping and name in class_name_mapping:
            display_names.append(class_name_mapping[name])
        else:
            display_names.append(name)

    # Compute basic statistics
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    total_pixels = cm.sum()

    # Compute normalized confusion matrix (percentages by row)
    cm_normalized = cm.astype("float") / row_totals[:, np.newaxis] * 100

    # Compute Producer's Accuracy (PA) = correctness from producer's perspective
    # PA = diagonal / column total (predicted totals)
    producers_accuracy = {}
    for i, class_name in enumerate(class_names):
        pa = cm[i, i] / col_totals[i] if col_totals[i] > 0 else 0
        producers_accuracy[class_name] = pa

    # Compute User's Accuracy (UA) = correctness from user's perspective
    # UA = diagonal / row total (ground truth totals)
    users_accuracy = {}
    for i, class_name in enumerate(class_names):
        ua = cm[i, i] / row_totals[i] if row_totals[i] > 0 else 0
        users_accuracy[class_name] = ua

    # Compute Overall Accuracy (OA)
    correct_pixels = np.trace(cm)
    overall_accuracy = correct_pixels / total_pixels

    # Compute Cohen's Kappa (κ)
    y_true = []
    y_pred = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            y_true.extend([i] * int(count))
            y_pred.extend([j] * int(count))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Kappa interpretation
    if kappa > 0.8:
        kappa_interpretation = "Strong agreement"
    elif kappa > 0.6:
        kappa_interpretation = "Substantial agreement"
    elif kappa > 0.4:
        kappa_interpretation = "Moderate agreement"
    elif kappa > 0.2:
        kappa_interpretation = "Fair agreement"
    else:
        kappa_interpretation = "Slight agreement"

    # Extract F1 scores if validation_results provided
    f1_per_class = None
    if validation_results and "f1_per_class" in validation_results:
        f1_per_class = validation_results["f1_per_class"]

    # Build results dictionary
    results = {
        "confusion_matrix": cm,
        "class_names": class_names,
        "display_names": display_names,
        "row_totals": row_totals,
        "col_totals": col_totals,
        "total_pixels": int(total_pixels),
        "cm_normalized": cm_normalized,
        "producers_accuracy": producers_accuracy,
        "users_accuracy": users_accuracy,
        "overall_accuracy": overall_accuracy,
        "cohens_kappa": kappa,
        "kappa_interpretation": kappa_interpretation,
        "f1_per_class": f1_per_class,
    }

    # Print analysis if requested
    if print_analysis:
        print("=" * 80)
        print("📊 CONFUSION MATRIX ANALYSIS")
        print("=" * 80)

        # 1. Raw confusion matrix (pixel counts)
        print("\n1️⃣ CONFUSION MATRIX - PIXEL COUNTS")
        print("-" * 80)
        print(f"{'':>15} | ", end="")
        for dn in display_names:
            print(f"{dn:>12} | ", end="")
        print("Row Total")
        print("-" * 80)

        for i, true_class in enumerate(display_names):
            print(f"{true_class:>15} | ", end="")
            for j in range(len(display_names)):
                print(f"{cm[i, j]:>12,} | ", end="")
            print(f"{row_totals[i]:>12,}")

        print("-" * 80)
        print(f"{'Column Total':>15} | ", end="")
        for ct in col_totals:
            print(f"{ct:>12,} | ", end="")
        print(f"{total_pixels:>12,}")
        print("=" * 80)

        # 2. Row totals (true pixels per class)
        print("\n2️⃣ ROW TOTALS (True pixels per class)")
        print("-" * 80)
        for i, true_class in enumerate(display_names):
            print(f"  {true_class:>15}: {row_totals[i]:>10,} pixels")
        print(f"  {'TOTAL':>15}: {total_pixels:>10,} pixels")
        print("=" * 80)

        # 3. Column totals (predicted pixels per class)
        print("\n3️⃣ COLUMN TOTALS (Predicted pixels per class)")
        print("-" * 80)
        for j, pred_class in enumerate(display_names):
            print(f"  {pred_class:>15}: {col_totals[j]:>10,} pixels")
        print(f"  {'TOTAL':>15}: {total_pixels:>10,} pixels")
        print("=" * 80)

        # 4. Total pixels N
        print(f"\n4️⃣ TOTAL PIXELS (N)")
        print("-" * 80)
        print(f"  N = {total_pixels:,} pixels")
        print("=" * 80)

        # 5. Producer's Accuracy (PA) per class = Recall
        print("\n5️⃣ PRODUCER'S ACCURACY (PA) per class")
        print("-" * 80)
        print("  PA = diagonal / row_total = recall")
        print("  (How many true class pixels were correctly predicted)")
        print()
        for i, (class_name, true_class) in enumerate(zip(class_names, display_names)):
            pa = producers_accuracy[class_name]
            print(f"  {true_class:>15}: {pa:>6.1%}  ({cm[i, i]:,} / {row_totals[i]:,})")
        print("=" * 80)

        # 6. User's Accuracy (UA) per class = Precision
        print("\n6️⃣ USER'S ACCURACY (UA) per class")
        print("-" * 80)
        print("  UA = diagonal / column_total = precision")
        print("  (How many predicted class pixels were actually correct)")
        print()
        for j, (class_name, pred_class) in enumerate(zip(class_names, display_names)):
            ua = users_accuracy[class_name]
            print(f"  {pred_class:>15}: {ua:>6.1%}  ({cm[j, j]:,} / {col_totals[j]:,})")
        print("=" * 80)

        # 7. Normalized percentages (per cell)
        print("\n7️⃣ NORMALIZED CONFUSION MATRIX (Percentages)")
        print("-" * 80)
        print(f"{'':>15} | ", end="")
        for dn in display_names:
            print(f"{dn:>12} | ", end="")
        print()
        print("-" * 80)

        for i, true_class in enumerate(display_names):
            print(f"{true_class:>15} | ", end="")
            for j in range(len(display_names)):
                print(f"{cm_normalized[i, j]:>11.1f}% | ", end="")
            print()
        print("=" * 80)

        # 8. Overall Accuracy (OA)
        print("\n8️⃣ OVERALL ACCURACY (OA)")
        print("-" * 80)
        print(
            f"  OA = {overall_accuracy:>6.1%}  ({correct_pixels:,} / {total_pixels:,})"
        )
        print("  (Total correctly classified pixels / Total pixels)")
        print("=" * 80)

        # 9. Cohen's Kappa (κ)
        print("\n9️⃣ COHEN'S KAPPA (κ)")
        print("-" * 80)
        print(f"  κ = {kappa:.3f}")
        print()
        print("  Interpretation:")
        print(f"  → {kappa_interpretation}")
        print()
        print("  Scale:")
        print("    κ > 0.8  : Strong agreement")
        print("    0.6-0.8  : Substantial agreement")
        print("    0.4-0.6  : Moderate agreement")
        print("    0.2-0.4  : Fair agreement")
        print("    κ < 0.2  : Slight agreement")
        print("=" * 80)

        # 10. Summary table
        print("\n🔟 SUMMARY TABLE")
        print("-" * 80)
        if f1_per_class:
            print(
                f"{'Class':>15} | {'N pixels':>12} | {'PA (%)':>8} | {'UA (%)':>8} | {'F1':>8}"
            )
        else:
            print(f"{'Class':>15} | {'N pixels':>12} | {'PA (%)':>8} | {'UA (%)':>8}")
        print("-" * 80)

        for i, (class_name, display_name) in enumerate(zip(class_names, display_names)):
            pa = producers_accuracy[class_name] * 100
            ua = users_accuracy[class_name] * 100
            if f1_per_class:
                f1 = f1_per_class[class_name]
                print(
                    f"{display_name:>15} | {row_totals[i]:>12,} | {pa:>7.1f}% | {ua:>7.1f}% | {f1:>7.3f}"
                )
            else:
                print(
                    f"{display_name:>15} | {row_totals[i]:>12,} | {pa:>7.1f}% | {ua:>7.1f}%"
                )

        print("-" * 80)
        if f1_per_class:
            print(
                f"{'OVERALL':>15} | {total_pixels:>12,} | {'':>8} | {'':>8} | {overall_accuracy:>7.1%}"
            )
            print(f"{'':>15} | {'':>12} | {'':>8} | {'':>8} | κ={kappa:.3f}")
        else:
            print(
                f"{'OVERALL':>15} | {total_pixels:>12,} | {'':>8} | {'':>8} | {overall_accuracy:>7.1%}"
            )
            print(f"{'':>15} | {'':>12} | {'':>8} | {'':>8} | κ={kappa:.3f}")
        print("=" * 80)

    return results
