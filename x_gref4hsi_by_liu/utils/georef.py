import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pyproj import Transformer

try:
    import config  # optional, used only to provide nicer defaults
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

    def __init__(self, path: str, use_corrected: bool = False):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.use_corrected = use_corrected

        self.shape = None  # (T, S, B)
        self.wavelengths = None
        self.has_georef = False
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

        except Exception as e:
            print(f"⚠️  Could not read metadata from {self.name}: {e}")
            self.shape = None
            self.wavelengths = None
            self.has_georef = False

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

    def select_files(self, names: List[str]) -> "CombinedTransectCube":
        missing = [n for n in names if n not in self.files]
        if missing:
            print(f"⚠️  Missing files: {missing}")
        chosen = [self.files[n] for n in names if n in self.files]
        if not chosen:
            raise ValueError("No valid files selected")
        return CombinedTransectCube(chosen, self.folder, self.use_corrected)

    def select_all_files(self) -> "CombinedTransectCube":
        return self.select_files(list(self.files.keys()))

    def select_files_by_pattern(self, pattern: str) -> "CombinedTransectCube":
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

        # combined grids (filled in on demand)
        self.X_ecef = None
        self.Y_ecef = None
        self.Z_ecef = None
        self.R = None
        self.G = None
        self.B = None
        self.wavelengths = geofiles[0].wavelengths if geofiles else None

        self._build_combined()

    def _build_combined(self):
        print("🔄 Rebuilding grids from georef hits for selected files...")
        xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
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
            rs.append(d["R"])
            gs.append(d["G"])
            bs.append(d["B"])
            T_accum += T

        # combine along track axis
        self.X_ecef = np.concatenate(xs, axis=0)
        self.Y_ecef = np.concatenate(ys, axis=0)
        self.Z_ecef = np.concatenate(zs, axis=0)
        self.R = np.concatenate(rs, axis=0)
        self.G = np.concatenate(gs, axis=0)
        self.B = np.concatenate(bs, axis=0)

        print(f"✅ Combined shapes: X/Y/Z {self.X_ecef.shape}, RGB {self.R.shape}")

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
        interactive=True,  # Enable click-to-show coordinates
        track_start=None,  # Start track index (inclusive), None = from beginning
        track_end=None,  # End track index (exclusive), None = to end
        use_corrected=False,  # Use illumination-corrected data if available
        **pcolor_kwargs,
    ):
        """
        Render an RGB composite in top-down 2D coordinates.
        Click on the plot to display coordinates (ECEF, NED, and lat/lon).

        Parameters:
        -----------
        track_start : int, optional
            Starting track index (inclusive). If None, starts from beginning (0).
        track_end : int, optional
            Ending track index (exclusive). If None, goes to the end.
            Example: track_start=100, track_end=200 plots tracks 100-199.
        use_corrected : bool, optional
            If True, uses illumination-corrected RGB data (requires apply_illumination_correction() to be called first).
            If False (default), uses original wavelength-based RGB data.

        - If coordinate_system == None or "LATLON" (DEFAULT):
            * Plots in Latitude/Longitude (WGS84 degrees)
            * No origin needed - converts ECEF directly to lat/lon
        - If coordinate_system == "NED":
            * Displays East vs North (meters) relative to origin
            * Requires origin (lat, lon, h)
        - If coordinate_system == "ECEF":
            * use_local_origin=True -> ΔECEF from origin (meters)
            * use_local_origin=False -> absolute ECEF X,Y (rarely useful)
        - If interactive=True:
            * Click anywhere to see all coordinate representations
        """
        # Default to LATLON if not specified
        if coordinate_system is None:
            coordinate_system = "LATLON"

        # Only get/require origin for NED or ECEF modes
        if coordinate_system.upper() in ["NED", "ECEF"]:
            if origin is None:
                if (
                    config is not None
                    and hasattr(config, "LAT0")
                    and hasattr(config, "LON0")
                    and hasattr(config, "H0")
                ):
                    origin = (
                        float(config.LAT0),
                        float(config.LON0),
                        float(getattr(config, "H0", 0.0)),
                    )
                else:
                    origin = (60.8011575, 10.7122345, 0.0)
        else:
            # For LATLON, origin is not used for plotting but keep for reference
            if origin is None:
                # Compute center from data for display purposes
                origin = None  # Will be computed from data later

        # Rebuild with requested wavelengths (or use corrected data)
        if use_corrected:
            if not hasattr(self, "data_corrected") or self.data_corrected is None:
                raise ValueError(
                    "Corrected data not available. Run apply_illumination_correction() first."
                )
            # Extract RGB from corrected data cube
            X_ecef, Y_ecef, Z_ecef = self.X_ecef, self.Y_ecef, self.Z_ecef
            R, G, B = self._extract_rgb_from_cube(
                self.data_corrected, red_wl, green_wl, blue_wl
            )
            # Optionally normalize
            if normalize:
                for C in (R, G, B):
                    m, M = np.nanmin(C), np.nanmax(C)
                    if np.isfinite(m) and np.isfinite(M) and M > m:
                        C[:] = (C - m) / (M - m)
        else:
            # Use original data with requested wavelengths
            X_ecef, Y_ecef, Z_ecef, R, G, B = self._rgb_from_wavelengths(
                red_wl, green_wl, blue_wl, normalize
            )

        T, S = X_ecef.shape

        # Apply track slicing if specified
        if track_start is not None or track_end is not None:
            start_idx = track_start if track_start is not None else 0
            end_idx = track_end if track_end is not None else T

            # Validate indices
            if start_idx < 0 or start_idx >= T:
                raise ValueError(f"track_start={start_idx} out of range [0, {T})")
            if end_idx <= start_idx or end_idx > T:
                raise ValueError(f"track_end={end_idx} must be in ({start_idx}, {T}]")

            # Slice all arrays
            X_ecef = X_ecef[start_idx:end_idx, :]
            Y_ecef = Y_ecef[start_idx:end_idx, :]
            Z_ecef = Z_ecef[start_idx:end_idx, :]
            R = R[start_idx:end_idx, :]
            G = G[start_idx:end_idx, :]
            B = B[start_idx:end_idx, :]
            T, S = X_ecef.shape  # Update dimensions

            print(f"📊 Plotting tracks {start_idx}–{end_idx-1} (shape: {T} × {S})")

        # Build RGBA (alpha 0 where RGB or XY are NaN)
        RGB = np.dstack([R, G, B]).astype(np.float64)  # (T,S,3)
        # Start fully opaque
        alpha = np.ones((T, S), dtype=np.float64)
        # Transparent where RGB is missing
        alpha[~(np.isfinite(R) & np.isfinite(G) & np.isfinite(B))] = 0.0

        # Coordinates
        from pyproj import Transformer

        if coordinate_system.upper() == "LATLON":
            # Convert ECEF to Lat/Lon
            tf_ecef_to_geo = Transformer.from_crs(
                "EPSG:4978", "EPSG:4979", always_xy=True
            )
            lon, lat, height = tf_ecef_to_geo.transform(X_ecef, Y_ecef, Z_ecef)
            Xp, Yp = lon, lat
            xlabel, ylabel = "Longitude (°)", "Latitude (°)"
            # Compute center for display
            valid_mask = np.isfinite(lon) & np.isfinite(lat)
            if valid_mask.any():
                lat0 = float(np.nanmean(lat[valid_mask]))
                lon0 = float(np.nanmean(lon[valid_mask]))
                h0 = 0.0
                origin = (lat0, lon0, h0)
            else:
                origin = (0.0, 0.0, 0.0)
        elif coordinate_system.upper() == "NED":
            lat0, lon0, h0 = origin
            N, E, D = _ecef_to_ned_arrays(X_ecef, Y_ecef, Z_ecef, lat0, lon0, h0)
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

        # Also hide cells where XY are invalid
        alpha[~(np.isfinite(Xp) & np.isfinite(Yp))] = 0.0

        # pcolormesh needs cell corners; pad edges once
        Xc = np.pad(Xp, ((0, 1), (0, 1)), mode="edge")
        Yc = np.pad(Yp, ((0, 1), (0, 1)), mode="edge")

        # Set RGB to NaN where alpha is 0 (invalid data)
        # This makes matplotlib render those cells transparently
        RGB[alpha == 0] = np.nan

        # FIX: Replace NaN/inf in Xc, Yc with nearest valid neighbor
        # This prevents pcolormesh from crashing on non-finite coordinates
        mask_valid = np.isfinite(Xc) & np.isfinite(Yc)
        if not mask_valid.all():
            # Replace NaN coordinates with nearby valid values
            from scipy.ndimage import distance_transform_edt

            invalid_mask = ~mask_valid
            if invalid_mask.any():
                # Find nearest valid coordinate for each invalid point
                indices = distance_transform_edt(
                    invalid_mask, return_distances=False, return_indices=True
                )
                Xc[invalid_mask] = Xc[tuple(indices[:, invalid_mask])]
                Yc[invalid_mask] = Yc[tuple(indices[:, invalid_mask])]

        plt.figure(figsize=figsize)
        # Pass RGB directly as the color array (like original implementation)
        # pcolormesh with shading='flat' accepts (T, S, 3) RGB array
        plt.pcolormesh(Xc, Yc, RGB, shading="flat", **pcolor_kwargs)

        # File boundary markers
        if show_file_boundaries and len(self.file_boundaries) > 1:
            for i, b in enumerate(self.file_boundaries[1:], 1):
                j = b["start_track"]
                if j < T:
                    # draw along the track j (all slits)
                    plt.plot(
                        Xp[j, :],
                        Yp[j, :],
                        color="yellow",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.9,
                        label="File boundary" if i == 1 else "",
                    )

        # Aspect ratio
        if coordinate_system.upper() == "LATLON":
            # For lat/lon, use adjusted aspect ratio to account for latitude distortion
            # At mid-latitudes, longitude degrees are ~cos(lat) times shorter
            if origin and origin[0] != 0:
                lat_center = origin[0]
                aspect_ratio = 1.0 / np.cos(np.radians(lat_center))
                plt.gca().set_aspect(aspect_ratio, adjustable="box")
            else:
                plt.gca().set_aspect("equal", adjustable="box")
        else:
            plt.gca().set_aspect("equal", adjustable="box")

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
        if interactive:
            title += "\n(Click to show coordinates)"
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show_file_boundaries and len(self.file_boundaries) > 1:
            plt.legend()

        # Add interactive click handler
        if interactive:
            from pyproj import Transformer

            # Store coordinate data for click handler
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

            # Create transformer for ECEF to lat/lon
            tf_ecef_to_geo = Transformer.from_crs(
                "EPSG:4978", "EPSG:4979", always_xy=True
            )

            def on_click(event):
                if event.inaxes is None:
                    return

                # Get click position in plot coordinates
                x_click, y_click = event.xdata, event.ydata

                # Find nearest grid point
                dist = (click_data["Xp"] - x_click) ** 2 + (
                    click_data["Yp"] - y_click
                ) ** 2
                min_idx = np.nanargmin(dist)
                track_idx, slit_idx = np.unravel_index(min_idx, click_data["Xp"].shape)

                # Get coordinates at this point
                x_plot = click_data["Xp"][track_idx, slit_idx]
                y_plot = click_data["Yp"][track_idx, slit_idx]
                x_ecef = click_data["X_ecef"][track_idx, slit_idx]
                y_ecef = click_data["Y_ecef"][track_idx, slit_idx]
                z_ecef = click_data["Z_ecef"][track_idx, slit_idx]

                # Get RGB values
                r_val = click_data["R"][track_idx, slit_idx]
                g_val = click_data["G"][track_idx, slit_idx]
                b_val = click_data["B"][track_idx, slit_idx]

                # Check if valid point
                if not (
                    np.isfinite(x_ecef) and np.isfinite(y_ecef) and np.isfinite(z_ecef)
                ):
                    print(f"⚠️  Invalid point at track={track_idx}, slit={slit_idx}")
                    return

                # Convert ECEF to lat/lon/height
                lon_deg, lat_deg, height = tf_ecef_to_geo.transform(
                    x_ecef, y_ecef, z_ecef
                )

                # Print info
                print("\n" + "=" * 70)
                print(f"📍 Clicked at track={track_idx}, slit={slit_idx}")

                # Show coordinates based on current display mode
                if click_data["coord_system"] == "LATLON":
                    print(f"   Lat/Lon: {y_plot:.6f}°, {x_plot:.6f}°")
                elif click_data["coord_system"] == "NED":
                    print(f"   NED: E={x_plot:.2f}m, N={y_plot:.2f}m")
                else:
                    print(f"   Plot coords: ({x_plot:.2f}, {y_plot:.2f})")

                # Always show ECEF and WGS84 for reference
                print(f"   ECEF: X={x_ecef:.2f}m, Y={y_ecef:.2f}m, Z={z_ecef:.2f}m")
                print(
                    f"   WGS84: Lat={lat_deg:.6f}°, Lon={lon_deg:.6f}°, h={height:.2f}m"
                )

                # Compute NED if origin is available and not already in NED mode
                if click_data["origin"] and click_data["coord_system"] != "NED":
                    lat0, lon0, h0 = click_data["origin"]
                    N, E, D = _ecef_to_ned_arrays(
                        np.array([[x_ecef]]),
                        np.array([[y_ecef]]),
                        np.array([[z_ecef]]),
                        lat0,
                        lon0,
                        h0,
                    )
                    print(
                        f"   NED (from center): E={E[0,0]:.2f}m, N={N[0,0]:.2f}m, D={D[0,0]:.2f}m"
                    )

                if np.isfinite(r_val):
                    print(f"   RGB: R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")
                else:
                    print(f"   RGB: No data")
                print("=" * 70)

            # Connect click event
            fig = plt.gcf()
            fig.canvas.mpl_connect("button_press_event", on_click)
            print("💡 Interactive mode: Click on the plot to display coordinates")

        plt.tight_layout()
        plt.show()

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
        import config
        from utils import utils
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
            return self.load_illumination_correction()

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

    def has_illumination_correction(self, window_size=1000, strength=1.0):
        """
        Check if illumination correction with these parameters has already been applied.
        Returns True if all files have the corrected dataset with matching metadata.
        """
        import h5py

        for gf in self.geofiles:
            try:
                with h5py.File(gf.path, "r") as f:
                    # Check if corrected dataset exists
                    if "processed/radiance/dataCube_illum_corrected" not in f:
                        return False

                    # Check metadata matches
                    dset = f["processed/radiance/dataCube_illum_corrected"]
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

    def load_illumination_correction(self):
        """
        Load previously saved illumination-corrected data from HDF5 files.
        Populates self.data_corrected.
        """
        import h5py

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
                if "processed/radiance/dataCube_illum_corrected" not in f:
                    raise RuntimeError(f"{gf.name}: corrected data not found")

                dset = f["processed/radiance/dataCube_illum_corrected"]
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
        Splits self.data_corrected back into individual files and saves as
        'processed/radiance/dataCube_illum_corrected' with metadata.
        """
        import h5py

        if self.data_corrected is None:
            raise RuntimeError(
                "No corrected data to save. Run apply_illumination_correction first."
            )

        print(f"💾 Saving illumination correction to {len(self.geofiles)} files...")

        t_offset = 0
        for gf in self.geofiles:
            T_file = gf.shape[0]
            corrected_chunk = self.data_corrected[t_offset : t_offset + T_file, :, :]

            with h5py.File(gf.path, "a") as f:  # 'a' = read/write, create if not exists
                dset_path = "processed/radiance/dataCube_illum_corrected"

                # Remove old dataset if it exists
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
                dset.attrs["window_size"] = window_size
                dset.attrs["strength"] = strength
                dset.attrs["description"] = "Illumination-corrected radiance data"

                print(f"   ✓ {gf.name}: saved {corrected_chunk.shape}")

            t_offset += T_file

        print(f"✅ Illumination correction saved to disk")

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
