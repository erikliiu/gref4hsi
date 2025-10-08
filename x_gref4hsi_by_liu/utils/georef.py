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
        **pcolor_kwargs,
    ):
        """
        Render an RGB composite in top-down 2D coordinates.
        Click on the plot to display coordinates (ECEF, NED, and lat/lon).

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

        # Rebuild with requested wavelengths
        X_ecef, Y_ecef, Z_ecef, R, G, B = self._rgb_from_wavelengths(
            red_wl, green_wl, blue_wl, normalize
        )
        T, S = X_ecef.shape

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
