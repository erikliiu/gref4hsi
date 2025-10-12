import os
import numpy as np
import matplotlib.pyplot as plt
from classes import *
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import gc


def load_roi_as_cube(roi_file_path):
    """
    Load a ROI file (e.g., perfect_vertical_bomb.txt) using ROIReader,
    and convert the 2D ROI data (wavelength & mean reflectance)
    into a dummy UHIFile instance with a data cube of shape (1, 1, n).

    Parameters:
        roi_file_path (str): Path to the ROI file.

    Returns:
        UHIFile: An instance containing the ROI data as a cube.
    """
    roi_data = ROIReader.get_roi_data(roi_file_path)
    if not roi_data:
        print("No ROI data loaded.")
        return None

    roi_data = np.array(
        roi_data
    )  # Expecting shape (n, 2) with columns [wavelength, mean_reflectance]
    wavelengths = roi_data[:, 0]
    reflectance = roi_data[:, 1]

    # Create a data cube with shape (1,1,n)
    data_cube = reflectance.reshape((1, 1, len(reflectance)))
    # Create a dummy filepath for the ROI instance
    dummy_filepath = f"ROI:{os.path.basename(roi_file_path)}"

    # Create and return a new UHIFile instance with this "cube"
    return UHIFile(
        dummy_filepath, track_offset=0, data=data_cube, wavelengths=wavelengths
    )


def plot_segments_random(
    folder,
    num_samples=10,
    red_wl=654.2,
    green_wl=560,
    blue_wl=440.3,
    spacing=0.4,
    normalize=True,
    mode="rgb",  # or "heatmap"
):
    """
    Visualize `num_samples` random .npz training cubes using RGB or heatmap.
    """
    # 1. Find all .npz files
    npz_files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    if not npz_files:
        print("No .npz files found in folder.")
        return

    # 2. Pick random files
    chosen = random.sample(npz_files, min(num_samples, len(npz_files)))

    for fname in chosen:
        fpath = os.path.join(folder, fname)
        print(f"\n[📦] Showing {fname}")

        # 3. Load into a UHIFile
        arr = np.load(fpath)
        data = arr["data"]
        wls = arr["wavelengths"]
        toffset = int(arr["track_offset"])
        cube = UHIFile(filepath=fpath, track_offset=toffset, data=data, wavelengths=wls)

        # 4. Visualize
        if mode == "rgb":
            cube.plot_rgb(
                red_wl=red_wl,
                green_wl=green_wl,
                blue_wl=blue_wl,
                spacing=spacing,
                normalize=normalize,
            )
            plt.show()
        elif mode == "heatmap":
            cube.plot_heatmap(spacing=spacing)
        else:
            print("Unknown mode:", mode)
            return


def plot_labeled_segments(
    root_dir,
    transect=None,
    label=None,
    red_wl=654.2,
    green_wl=560,
    blue_wl=440.3,
    normalize=True,
    spacing=0.4,
):
    """
    Plot saved segments in root_dir, filtering by transect and/or label.

    Parameters:
      root_dir (str): e.g. 'training_data/radiance' or '…/reflectance'
      transect (str, optional): only this transect folder (by name)
      label (str, optional): only this label category, e.g. 'dark','bomb', etc.
      red_wl, green_wl, blue_wl: wavelengths for RGB
      normalize (bool): normalize each channel before plotting
      spacing (float): figure size scale
    """
    # 1) Print summary of what we’re about to do:
    print(f"\n▶️ Plotting segments from: {root_dir!r}")
    if transect:
        print(f"   • Transect filter: {transect!r}")
    if label:
        print(f"   • Label filter:    {label!r}")
    print()

    # 2) Loop over transect subfolders
    for tr_name in sorted(os.listdir(root_dir)):
        tr_dir = os.path.join(root_dir, tr_name)
        if not os.path.isdir(tr_dir):
            continue
        if transect and tr_name != transect:
            continue

        # 3) Loop over .npz files in that transect
        for fname in sorted(os.listdir(tr_dir)):
            if not fname.endswith(".npz"):
                continue
            seg_label = fname.split("_")[0]
            if label and seg_label != label:
                continue

            fpath = os.path.join(tr_dir, fname)
            arr = np.load(fpath)
            data = arr["data"]
            wls = arr["wavelengths"]
            toff = int(arr["track_offset"])

            # Reconstruct a UHIFile for plotting
            cube = UHIFile(
                filepath=fpath, track_offset=toff, data=data, wavelengths=wls
            )

            # 4) Show small RGB plot
            cube.plot_rgb(
                red_wl=red_wl,
                green_wl=green_wl,
                blue_wl=blue_wl,
                spacing=spacing,
                normalize=normalize,
            )
            plt.title(f"{tr_name} | {fname}", fontsize=8)
            plt.show()
