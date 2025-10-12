import os
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import gc
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
import matplotlib.ticker as ticker
import numpy as np


class ROIReader:
    @staticmethod
    def get_roi_data(path):
        roi_data = []
        with open(path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    tokens = line.split()
                    wavelength = float(tokens[0])
                except ValueError:
                    continue
                if len(tokens) >= 4:
                    try:
                        mean_refl = float(tokens[3])
                    except ValueError:
                        continue
                    roi_data.append([wavelength, mean_refl])
        return roi_data


class UHIFile:
    def __init__(self, filepath, track_offset=0, data=None, wavelengths=None):
        self.filepath = filepath
        self.track_offset = track_offset
        self.name = os.path.splitext(os.path.basename(filepath))[0]
        if data is not None:
            self.data = data
            # If wavelengths are provided (for example, from a slicing operation), use them.
            if wavelengths is not None:
                self.wavelengths = wavelengths
            else:
                self.wavelengths = np.arange(data.shape[2])
        else:
            self.data = self._load_data()

    def _load_data(self):
        with h5py.File(self.filepath, "r") as f:
            try:
                cube = f["processed"]["radiance"]["dataCube"][()]
            except KeyError as e:
                print(f"Missing expected path in {self.name}: {e}")
                return None
            try:
                # Load the band-to-wavelength mapping
                self.wavelengths = f["processed"]["radiance"]["calibration"][
                    "spectral"
                ]["band2Wavelength"][()]
            except KeyError as e:
                print(f"Missing wavelength data in {self.name}: {e}")
                self.wavelengths = np.arange(cube.shape[2])
            return cube

    def describe(self, track_times=None, start_time=None, end_time=None):
        if self.data is None:
            return

        n_tracks, n_slit, n_wave = self.data.shape
        print(f"\n=== {self.name} ===")
        print(f"Shape: {self.data.shape}")
        print(f" - Tracks (time steps):     {n_tracks}")
        print(
            f" - Actual track indices:    {self.track_offset} to {self.track_offset + n_tracks - 1}"
        )
        print(f" - Slit pixels:             {n_slit}")
        print(f" - Wavelengths (bands):     {n_wave}")
        print(
            f" - Intensity range:         {self.data.min():.3f} to {self.data.max():.3f}"
        )

        if track_times is not None:
            if len(track_times) != n_tracks:
                print("[!] track_times length mismatch.")
            else:
                print(f"Track times: {track_times[:3]} ... {track_times[-3:]}")
                print(f"Start: {track_times[0]} | End: {track_times[-1]}")

        if start_time is not None:
            print(f"Start time: {start_time}")
        if end_time is not None:
            print(f"End time:   {end_time}")

    def plot_heatmap(self, spacing=1, track_index=None, slit_index=None):
        if self.data is None:
            return

        # Compute mean intensity across wavelengths.
        intensity_map = np.mean(self.data, axis=2).T  # shape: (n_slit, n_tracks)
        n_tracks = self.data.shape[0]
        n_slits = intensity_map.shape[0]
        # Create x_coords using the track offset.
        x_coords = np.arange(self.track_offset, self.track_offset + n_tracks)

        plt.figure(figsize=(12 * spacing, 6))
        plt.imshow(
            intensity_map,
            aspect="auto",
            origin="lower",
            extent=[x_coords[0], x_coords[-1], 0, n_slits],
            cmap="inferno",
        )
        plt.colorbar(label="Average Intensity (across wavelengths)")

        # Draw the provided special lines if any.
        if track_index is not None:
            plt.axvline(x=track_index, color="cyan", linestyle="--", linewidth=2)
        if slit_index is not None:
            plt.axhline(y=slit_index, color="lime", linestyle="--", linewidth=2)

        # Add grid lines every 50 units.
        # Vertical lines for track indices.
        for x in np.arange(x_coords[0], x_coords[-1] + 1, 50):
            plt.axvline(x=x, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        # Horizontal lines for slit pixels.
        for y in np.arange(0, n_slits + 1, 50):
            plt.axhline(y=y, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

        plt.xlabel("Track Index")
        plt.ylabel("Slit Pixel Index")
        plt.title(f"Heatmap for {self.name}")
        plt.tight_layout()
        plt.show()

    def plot_spectrum(self, track_index, slit_index, ylabel="Intensity"):
        if self.data is None:
            return

        # Convert the absolute track_index into the relative index within this instance.
        relative_index = track_index - self.track_offset
        if relative_index < 0 or relative_index >= self.data.shape[0]:
            print("Error: track_index is outside of the current cube's range.")
            return

        spectrum = self.data[relative_index, slit_index, :]

        plt.figure(figsize=(10, 5))
        plt.plot(self.wavelengths, spectrum, linestyle="-")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(ylabel)
        plt.title(f"Spectrum at Track {track_index}, Slit {slit_index} ({self.name})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def slice_tracks(self, start, end):
        """
        Create a new UHIFile instance that is a slice of the original data.
        The slice takes track indices from start (inclusive) to end (exclusive) and updates the track offset.
        """
        if self.data is None:
            print("No data to slice.")
            return None
        new_data = self.data[start:end, :, :]
        new_track_offset = self.track_offset + start
        return UHIFile(
            self.filepath,
            track_offset=new_track_offset,
            data=new_data,
            wavelengths=self.wavelengths,
        )

    def get_reflectance(self, roi_file_path):

        if self.data is None:
            print("No data available for reflectance calculation.")
            return None

        # Load ROI data using the ROIReader class.
        roi_data = ROIReader.get_roi_data(roi_file_path)
        if not roi_data:
            print("No ROI data loaded.")
            return None

        roi_data = np.array(roi_data)
        # The ROI data is assumed to have two columns: [wavelength, mean_reflectance]
        refl_plate = roi_data[:, 1]

        n_wave_data = self.data.shape[2]
        if len(refl_plate) != n_wave_data:
            print(
                "Warning: Mismatch in the number of wavelengths between ROI data and data cube."
            )
            common_length = min(len(refl_plate), n_wave_data)
            data = self.data[:, :, :common_length]
            refl_plate = refl_plate[:common_length]
        else:
            data = self.data

        # Compute reflectance using broadcasting.
        reflectance = data / refl_plate[np.newaxis, np.newaxis, :]
        return UHIFile(
            self.filepath,
            track_offset=self.track_offset,
            data=reflectance,
            wavelengths=self.wavelengths,
        )

    def get_reflectance_normalize(self, roi_file_path):
        """
        i call it normalize since it normalize, then divide and then unnormalize.
        Normalize both data and plate, divide to get reflectance, then un‑normalize
        back to raw reflectance units.

        - For a full spatial cube: does per‑band min–max across (tracks, slits).
        - For a 1×1×n spectral cube: does min–max across the wavelength axis.
        """
        import numpy as np

        if self.data is None:
            print("No data available for reflectance calculation.")
            return None

        # --- Load plate spectrum from ROI file
        roi_data = ROIReader.get_roi_data(roi_file_path)
        if not roi_data:
            print("No ROI data loaded.")
            return None
        roi_data = np.array(roi_data)  # shape (n_wave, 2)
        refl_plate = roi_data[:, 1]  # vector length n_wave

        # --- Trim to common spectral length
        data = self.data
        n_wave = data.shape[2]
        if len(refl_plate) != n_wave:
            common = min(len(refl_plate), n_wave)
            data = data[:, :, :common]
            refl_plate = refl_plate[:common]
            wavelengths = self.wavelengths[:common]
        else:
            wavelengths = self.wavelengths

        # --- 1) Normalize data
        if data.shape[0] == 1 and data.shape[1] == 1:
            # Single-pixel cube
            spec = data.reshape(-1)
            dmin = spec.min()
            dmax = spec.max()
            drange = (dmax - dmin) if (dmax != dmin) else 1.0
            data_norm = ((spec - dmin) / drange).reshape((1, 1, -1))
        else:
            # Per-band (spatial) normalization
            dmin = data.min(axis=(0, 1))  # shape (n_wave,)
            dmax = data.max(axis=(0, 1))
            drange = dmax - dmin
            drange[drange == 0] = 1.0  # Avoid zero division
            data_norm = (data - dmin[np.newaxis, np.newaxis, :]) / drange[
                np.newaxis, np.newaxis, :
            ]

        # --- 2) Normalize plate spectrum
        pmin = refl_plate.min()
        pmax = refl_plate.max()
        plate_range = pmax - pmin

        if plate_range == 0:
            print("⚠️ Plate spectrum has no variation. Skipping normalization.")
            plate_norm = np.ones_like(refl_plate)  # Avoid divide-by-zero
        else:
            plate_norm = (refl_plate - pmin) / plate_range

        # --- 3) Divide
        with np.errstate(divide="ignore", invalid="ignore"):
            refl_norm = data_norm / plate_norm[np.newaxis, np.newaxis, :]

        # --- 4) Un-normalize
        if np.ndim(drange) == 0:
            drange_arr = np.full_like(refl_plate, drange)
        else:
            drange_arr = drange

        if plate_range == 0:
            reflectance = refl_norm  # Already unitless
        else:
            reflectance = refl_norm * (
                drange_arr[np.newaxis, np.newaxis, :] / plate_range
            )

        return UHIFile(
            self.filepath,
            track_offset=self.track_offset,
            data=reflectance,
            wavelengths=wavelengths,
        )

    def change_wavelength_interval(self, wl_start, wl_end):
        """
        Create a new UHIFile instance with a subset of wavelengths.
        Only wavelengths (and the corresponding data) between wl_start and wl_end (inclusive)
        are retained.

        Parameters:
            wl_start (float): The starting wavelength in nm.
            wl_end (float): The ending wavelength in nm.

        Returns:
            UHIFile: A new instance with the updated data cube and wavelengths.
        """
        # Find indices corresponding to the desired wavelength interval.
        indices = np.where(
            (self.wavelengths >= wl_start) & (self.wavelengths <= wl_end)
        )[0]

        if len(indices) == 0:
            print("No wavelengths found in the specified interval.")
            return None

        # Slice the data cube along the third axis (wavelength axis).
        new_data = self.data[:, :, indices]
        new_wavelengths = self.wavelengths[indices]

        # Return a new UHIFile instance preserving file path and track offset.
        return UHIFile(
            self.filepath,
            track_offset=self.track_offset,
            data=new_data,
            wavelengths=new_wavelengths,
        )

    def plot_rgb(
        self,
        red_wl=654.2,
        green_wl=560,
        blue_wl=440.3,
        spacing=1,
        track_index=None,
        slit_index=None,
        normalize=True,
        roi_radii=None,  # tuple (radius_track, radius_slit)
        roi_edgecolor="white",
        roi_linewidth=2,
    ):
        """
        Create an RGB composite using three wavelengths.
        If you pass roi_radii=(rt,rs) *and* track_index, slit_index,
        this will draw an elliptical ROI overlay of half‑axes rt, rs.
        """
        # ─ pick channels ──────────────────────────────────────────────────────────
        red_idx = np.argmin(np.abs(self.wavelengths - red_wl))
        green_idx = np.argmin(np.abs(self.wavelengths - green_wl))
        blue_idx = np.argmin(np.abs(self.wavelengths - blue_wl))

        self.rgb_mapping = {"red": red_idx, "green": green_idx, "blue": blue_idx}

        R = self.data[:, :, red_idx].T
        G = self.data[:, :, green_idx].T
        B = self.data[:, :, blue_idx].T

        if normalize:
            R = (R - R.min()) / (R.max() - R.min()) if R.max() != R.min() else R
            G = (G - G.min()) / (G.max() - G.min()) if G.max() != G.min() else G
            B = (B - B.min()) / (B.max() - B.min()) if B.max() != B.min() else B

        rgb_image = np.stack([R, G, B], axis=-1)

        # ─ axes extents ─────────────────────────────────────────────────────────
        n_tracks = self.data.shape[0]
        x_coords = np.arange(self.track_offset, self.track_offset + n_tracks)
        n_slits = R.shape[0]

        # ─ draw the image ───────────────────────────────────────────────────────
        plt.figure(figsize=(12 * spacing, 6))
        plt.imshow(
            rgb_image,
            aspect="auto",
            origin="lower",
            extent=[x_coords[0], x_coords[-1], 0, n_slits],
        )

        # ─ optional cross‑hairs ─────────────────────────────────────────────────
        if track_index is not None:
            plt.axvline(track_index, color="cyan", linestyle="--", linewidth=2)
        if slit_index is not None:
            plt.axhline(slit_index, color="lime", linestyle="--", linewidth=2)

        # ─ ellipse overlay ─────────────────────────────────────────────────────
        if roi_radii and track_index is not None and slit_index is not None:
            rt, rs = roi_radii
            ell = Ellipse(
                (track_index, slit_index),
                width=2 * rt,
                height=2 * rs,
                edgecolor=roi_edgecolor,
                facecolor="none",
                linewidth=roi_linewidth,
            )
            plt.gca().add_patch(ell)

        # ─ grid lines every 50 px ───────────────────────────────────────────────
        for x in np.arange(x_coords[0], x_coords[-1] + 1, 50):
            plt.axvline(x, color="black", linewidth=0.5, alpha=0.3)
        for y in np.arange(0, n_slits + 1, 50):
            plt.axhline(y, color="black", linewidth=0.5, alpha=0.3)

        # ─ ticks & formatting ───────────────────────────────────────────────────
        xticks = np.arange(x_coords[0], x_coords[-1] + 1, 50)
        plt.xticks(xticks, fontsize=6, rotation=90)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{int(v)}"))

        # ─ labels & title ───────────────────────────────────────────────────────
        plt.xlabel("Track Index")
        plt.ylabel("Slit Pixel Index")
        plt.title(
            f"RGB \n {self.name}\n" f"(R={red_wl}nm, G={green_wl}nm, B={blue_wl}nm)"
        )

        plt.tight_layout()
        plt.show()

    def plot_roi_rgb(self, red_wl=654.2, green_wl=560, blue_wl=440.3, normalize=True):
        """
        Display this UHIFile’s data (presumably a small ROI cube) as an RGB image.
        Any NaNs (outside the ellipse) will appear blank.
        """
        import matplotlib.pyplot as plt

        # pick channels
        ri = np.argmin(np.abs(self.wavelengths - red_wl))
        gi = np.argmin(np.abs(self.wavelengths - green_wl))
        bi = np.argmin(np.abs(self.wavelengths - blue_wl))

        R = self.data[:, :, ri].T
        G = self.data[:, :, gi].T
        B = self.data[:, :, bi].T

        if normalize:
            for C in (R, G, B):
                valid = ~np.isnan(C)
                if valid.any():
                    m, M = C[valid].min(), C[valid].max()
                    if M > m:
                        C[valid] = (C[valid] - m) / (M - m)

        rgb = np.stack([R, G, B], axis=-1)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(rgb, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"ROI @ {self.name}\n"
            f"Tracks {self.track_offset}–{self.track_offset+self.data.shape[0]-1}"
        )
        plt.tight_layout()
        plt.show()

    def plot_mean_spectrum(self, ylabel="Intensity"):
        """
        Compute and plot the mean spectrum over all spatial pixels in this cube.
        Useful for your ROI cubes.
        """
        spec = np.nanmean(self.data, axis=(0, 1))
        plt.figure(figsize=(10, 5))
        plt.plot(self.wavelengths, spec, linestyle="-")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(ylabel)
        plt.title(f"Mean Spectrum over ROI\n{self.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def init_rgb_view(
        self,
        red_wl=620,
        green_wl=520,
        blue_wl=470,
        normalize=True,
        spacing=1,
        figsize=(200, 10),
    ):
        """
        Draw the RGB composite + grid once, cache the background, and
        return (fig, ax, background, vert_start, vert_end).
        """
        import matplotlib.ticker as ticker

        # build the RGB array
        red_idx = np.argmin(np.abs(self.wavelengths - red_wl))
        green_idx = np.argmin(np.abs(self.wavelengths - green_wl))
        blue_idx = np.argmin(np.abs(self.wavelengths - blue_wl))
        R = self.data[:, :, red_idx].T.copy()
        G = self.data[:, :, green_idx].T.copy()
        B = self.data[:, :, blue_idx].T.copy()
        if normalize:
            for C in (R, G, B):
                if C.max() != C.min():
                    C[:] = (C - C.min()) / (C.max() - C.min())
        rgb = np.stack([R, G, B], axis=-1)

        # set up axes, draw image + grid once
        n_tracks = self.data.shape[0]
        x0, x1 = self.track_offset, self.track_offset + n_tracks
        n_slits = R.shape[0]

        fig, ax = plt.subplots(figsize=(figsize[0] * spacing, figsize[1]))
        ax.imshow(rgb, aspect="auto", origin="lower", extent=[x0, x1, 0, n_slits])

        # grid every 50
        for x in np.arange(x0, x1 + 1, 50):
            ax.axvline(x=x, color="black", linestyle="-", lw=0.5, alpha=0.3)
        for y in np.arange(0, n_slits + 1, 50):
            ax.axhline(y=y, color="black", linestyle="-", lw=0.5, alpha=0.3)

        # force ticks every 50
        xticks = np.arange(x0, x1 + 1, 50)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{int(v)}"))
        plt.xticks(fontsize=6, rotation=90)
        ax.set_xlabel("Track Index")
        ax.set_ylabel("Slit Pixel Index")
        ax.set_title(
            f"RGB Composite for {self.name}\n"
            f"(R={red_wl}nm, G={green_wl}nm, B={blue_wl}nm)"
        )

        # overlay artists: two vertical lines for start/end
        vert_start = ax.axvline(x=x0, color="cyan", linestyle="--", lw=2)
        vert_end = ax.axvline(x=x1, color="cyan", linestyle="--", lw=2)

        # draw & cache
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)
        fig.tight_layout()
        plt.ion()
        return fig, ax, background, vert_start, vert_end

    def update_overlay(
        self,
        fig,
        ax,
        background,
        vert_start,
        vert_end,
        highlight_start=None,
        highlight_end=None,
    ):
        """
        Restore the background, move the two vertical lines, and blit.
        """
        canvas = fig.canvas
        canvas.restore_region(background)

        if highlight_start is not None:
            vert_start.set_xdata([highlight_start, highlight_start])
            ax.draw_artist(vert_start)
        if highlight_end is not None:
            vert_end.set_xdata([highlight_end, highlight_end])
            ax.draw_artist(vert_end)

        canvas.blit(ax.bbox)
        plt.pause(0.001)

    def create_training_data(
        self,
        out_dir,
        segment_length=50,
        reuse_labels=False,
        labels_file=None,
        red_wl=654.2,
        green_wl=560,
        blue_wl=440.3,
        normalize=True,
        spacing=1,
    ):
        """
        Slice into chunks, show an RGB+grid once, then for each segment:
          • move the two cyan overlay lines,
          • prompt for (or re‑use) label (“back” to re‑label, “skip” to leave unlabeled),
          • save .npz,
        finally writes labels.csv.

        Args:
          out_dir (str): output folder for this transect
          segment_length (int): tracks per segment
          reuse_labels (bool): if True, read labels from labels_file
          labels_file (str): path to existing labels.csv
          red_wl,green_wl,blue_wl,normalize,spacing: passed to init_rgb_view
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        # categories = {"bomb", "dark", "little_dark", "sediment", "other"}
        categories = {"bomb", "dark", "sediment", "other", "fish"}
        valid = categories | {"back", "skip"}

        # 1) Prepare the static RGB view + two overlay lines
        fig, ax, background, vert_start, vert_end = self.init_rgb_view(
            red_wl=red_wl,
            green_wl=green_wl,
            blue_wl=blue_wl,
            normalize=normalize,
            spacing=spacing,
        )

        # 2) Load existing labels if requested
        existing = {}
        if reuse_labels and labels_file and os.path.exists(labels_file):
            df0 = pd.read_csv(labels_file)
            existing = {(int(r.start), int(r.end)): r.label for r in df0.itertuples()}

        # 3) Build list of (start,end) segments
        n_tracks = self.data.shape[0]
        segments = [
            (self.track_offset + i, self.track_offset + i + segment_length)
            for i in range(0, n_tracks, segment_length)
            if i + segment_length <= n_tracks
        ]

        idx = 0
        # 4) Loop with “back” and “skip”
        while idx < len(segments):
            start, end = segments[idx]
            key = (start, end)

            # update overlay
            self.update_overlay(
                fig,
                ax,
                background,
                vert_start=vert_start,
                vert_end=vert_end,
                highlight_start=start,
                highlight_end=end,
            )

            # skip if reusing
            if reuse_labels and key in existing:
                idx += 1
                continue

            # prompt
            print(f"\n🔹 Segment {start}–{end}")
            # print("   Categories:", categories, "| back = previous, skip = no label")
            while True:
                choice = input("   → Label/back/skip: ").strip().lower()
                if choice == "back":
                    if idx > 0:
                        idx -= 1
                    else:
                        print("   [!] Already at first segment.")
                    break

                if choice == "skip":
                    print(f"   ⏭  Skipping {start}–{end}")
                    idx += 1
                    break

                if choice in categories:
                    existing[key] = choice

                    # remove any old file for this segment
                    pattern = f"_{start}_{end}.npz"
                    for old in os.listdir(out_dir):
                        if old.endswith(pattern):
                            os.remove(os.path.join(out_dir, old))
                            print(f"   🗑  Removed old {old}")

                    # save new
                    rel_start = start - self.track_offset
                    rel_end = end - self.track_offset
                    seg = self.slice_tracks(rel_start, rel_end)
                    fname = f"{choice}_{start}_{end}.npz"
                    path = os.path.join(out_dir, fname)
                    np.savez(
                        path,
                        data=seg.data,
                        wavelengths=seg.wavelengths,
                        track_offset=seg.track_offset,
                    )
                    print(f"   ✅ Saved {fname}")

                    idx += 1
                    break

                print(f"   ❌ '{choice}' not valid; try again.")

        # 5) Write labels.csv for everything *labeled*
        records = [
            {"start": s, "end": e, "label": lab} for (s, e), lab in existing.items()
        ]
        df = pd.DataFrame(records)
        labels_csv = os.path.join(out_dir, "labels.csv")
        df.to_csv(labels_csv, index=False)
        print(f"\n🗂  Labels written to {labels_csv}")

        # cleanup
        plt.ioff()
        plt.close(fig)

    def get_ROI_ellipse(
        self,
        track_index: int,
        slit_index: int,
        radius_track: int,
        radius_slit: int = None,
        elliptical: bool = True,
    ):
        """
        Extract an ROI around (track_index, slit_index).
        • radius_track: half‐width along the track axis
        • radius_slit:  half‐width along the slit axis (if None, equals radius_track → circle)
        • elliptical:   if True, zero (→ NaN) everything outside the ellipse
                        if False, you get a rectangular window
        Returns a new UHIFile whose .data is that little subcube.
        """
        if radius_slit is None:
            radius_slit = radius_track

        # 1) convert absolute to relative track index
        rel_t = int(track_index - self.track_offset)
        if not (0 <= rel_t < self.data.shape[0]):
            raise ValueError("track_index out of range")
        if not (0 <= slit_index < self.data.shape[1]):
            raise ValueError("slit_index out of range")

        # 2) compute bounding box
        t0 = max(rel_t - radius_track, 0)
        t1 = min(rel_t + radius_track + 1, self.data.shape[0])
        s0 = max(slit_index - radius_slit, 0)
        s1 = min(slit_index + radius_slit + 1, self.data.shape[1])

        # 3) slice out the block
        roi_block = self.data[t0:t1, s0:s1, :].copy()

        # 4) mask to ellipse if requested
        if elliptical:
            tt = np.arange(t0, t1)[:, None]
            ss = np.arange(s0, s1)[None, :]
            norm2 = ((tt - rel_t) / radius_track) ** 2 + (
                (ss - slit_index) / radius_slit
            ) ** 2
            mask2d = norm2 <= 1.0  # True inside ellipse
            # broadcast to all wavelengths & set outside→NaN
            roi_block = np.where(mask2d[:, :, None], roi_block, np.nan)

        # 5) return a new UHIFile
        return UHIFile(
            filepath=self.filepath,
            track_offset=self.track_offset + t0,
            data=roi_block,
            wavelengths=self.wavelengths,
        )

    def apply_flat_field_correction(self):
        with h5py.File(self.filepath, "r") as f:
            dark = f["processed/radiance/calibration/radiometric/darkFrame"][
                ()
            ]  # (968, 210)
            flat = f["processed/radiance/calibration/radiometric/radiometricFrame"][
                ()
            ]  # (968, 210)

        # Broadcast dark and flat to match the shape of data
        dark = dark[
            np.newaxis, :, :
        ]  # (1, 968, 210) → broadcastable to (2478, 968, 210)
        flat = flat[np.newaxis, :, :]

        correction_factor = np.median(flat - dark)
        denom = flat - dark
        denom[denom == 0] = 1e-6  # avoid division by zero

        corrected = (self.data - dark) * correction_factor / denom

        return UHIFile(
            self.filepath,
            track_offset=self.track_offset,
            data=corrected,
            wavelengths=self.wavelengths,
        )


class UHIDataSet:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = self._load_all_files()

    def _load_all_files(self):
        uhi_files = {}
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".h5"):
                path = os.path.join(self.folder_path, filename)
                uhi = UHIFile(path)
                if uhi.data is not None:
                    uhi_files[uhi.name] = uhi
        return uhi_files

    def describe_all(self):
        print(f"\n📂 Total files loaded: {len(self.files)}")
        for uhi in self.files.values():
            uhi.describe()
