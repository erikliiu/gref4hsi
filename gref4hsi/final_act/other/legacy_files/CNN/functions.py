import os
import numpy as np
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt
from training_data.classes import UHIFile  # adjust path if needed
from sklearn.model_selection import train_test_split
import os
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import os
import numpy as np
from collections import defaultdict


def load_data(root_dir):
    """
    Walks through each transect folder under root_dir, loads all .npz files,
    and groups their 'data' arrays by class name (filename before the first underscore).

    Returns:
        dict: { class_name: [np.ndarray, ...], ... }
    """
    data_by_class = defaultdict(list)
    for transect in os.listdir(root_dir):
        transect_path = os.path.join(root_dir, transect)
        if not os.path.isdir(transect_path):
            continue
        for fname in os.listdir(transect_path):
            if not fname.lower().endswith(".npz"):
                continue
            file_path = os.path.join(transect_path, fname)
            arr = np.load(file_path)
            segment = arr["data"]  # 3D hyperspectral cube
            class_name = fname.split("_", 1)[0]
            data_by_class[class_name].append(segment)
    return data_by_class


def prepare_X_y(data_by_class):
    """
    Flattens the data_by_class dict into X and y arrays for CNN training.

    Returns:
        X (np.ndarray): shape (N, tracks, slits, bands)
        y (np.ndarray): one-hot labels shape (N, num_classes)
        class_names (list): sorted list of class label names
    """
    X, y, class_names = [], [], sorted(data_by_class.keys())
    label_to_index = {c: i for i, c in enumerate(class_names)}

    for c in class_names:
        for cube in data_by_class[c]:
            X.append(cube)
            y.append(label_to_index[c])

    X = np.stack(X, axis=0)
    y = to_categorical(y, num_classes=len(class_names))
    return X, y, class_names


def augment_all_segments(base_root, out_root, max_workers=4):
    """
    Recursively augments all .npz files in base_root with flips and saves them
    to a mirror directory in out_root. Keeps original filename + flip suffix.
    Shows progress and timing.
    """

    def augment_and_save(path, save_dir):
        with np.load(path, mmap_mode="r", allow_pickle=False) as arr:
            data = arr["data"]
            wls = arr.get("wavelengths")
            toff = int(arr.get("track_offset", 0))

        base = os.path.splitext(os.path.basename(path))[0]
        flips = {
            "flip_slit": data[:, ::-1],
            "flip_track": data[::-1, :],
            "flip_both": data[::-1, ::-1],
        }

        os.makedirs(save_dir, exist_ok=True)
        for suffix, flipped in flips.items():
            save_path = os.path.join(save_dir, f"{base}_{suffix}.npz")
            np.savez(save_path, data=flipped, wavelengths=wls, track_offset=toff)

    # Gather tasks
    tasks = []
    for root, _, files in os.walk(base_root):
        npz_files = [f for f in files if f.endswith(".npz")]
        if not npz_files:
            continue
        rel_dir = os.path.relpath(root, base_root)
        save_subdir = os.path.join(out_root, rel_dir)
        for fname in npz_files:
            tasks.append((os.path.join(root, fname), save_subdir))

    # Run with progress bar and timing
    print(f"🛠️ Augmenting {len(tasks)} files with {max_workers} workers...")
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(
            tqdm(ex.map(lambda args: augment_and_save(*args), tasks), total=len(tasks))
        )

    end = time.time()
    print(f"✅ Done in {end - start:.2f} seconds")


def plot_all_npz_segments(folder_path):
    """
    Plots all .npz UHI segments in the given folder using UHIFile.plot_rgb().
    Title includes class name and track index range from filename.

    Args:
        folder_path (str): Path to the folder containing .npz files.
    """
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npz")])

    for fname in files:
        fpath = os.path.join(folder_path, fname)

        # Parse info from filename
        base = os.path.splitext(fname)[0]  # remove .npz
        parts = base.split("_")
        if len(parts) < 3:
            print(f"⚠️ Skipping unrecognized filename format: {fname}")
            continue
        label = parts[0]
        track_start = parts[1]
        track_end = parts[2]

        # Load cube
        arr = np.load(fpath)
        data = arr["data"]
        wls = arr["wavelengths"]
        toff = int(arr["track_offset"])

        # Recreate UHIFile and plot
        cube = UHIFile(filepath=fpath, track_offset=toff, data=data, wavelengths=wls)
        # plt.title(f"{label} | Tracks {track_start}-{track_end}", fontsize=10)
        cube.plot_rgb(spacing=0.25)
