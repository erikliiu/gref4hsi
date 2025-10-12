"""
TransectDataSet - A class for handling entire transects composed of multiple .h5 files.

Similar to UHIDataSet, but operates at the transect level where each transect
contains multiple .h5 files that can be selected and combined.

Author: Generated for UHI post-processing workflow
Date: 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import List, Dict, Optional, Union
from pyproj import Transformer


def ecef_to_ned(x_ecef, y_ecef, z_ecef, lat0, lon0, h0):
    """
    Convert ECEF coordinates to NED (North-East-Down) coordinates.

    Args:
        x_ecef, y_ecef, z_ecef: ECEF coordinates (arrays or scalars)
        lat0, lon0, h0: Reference point (latitude in degrees, longitude in degrees, height in meters)

    Returns:
        north, east, down: NED coordinates relative to the reference point
    """
    # Convert reference point to ECEF
    tf = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = tf.transform(lon0, lat0, h0)

    # Translate to origin
    dx = x_ecef - x0
    dy = y_ecef - y0
    dz = z_ecef - z0

    # Convert to radians
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    # Rotation matrix from ECEF to NED
    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # NED transformation matrix
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    east = -sin_lon * dx + cos_lon * dy
    down = -cos_lat * cos_lon * dx - cos_lat * sin_lon * dy - sin_lat * dz

    return north, east, down


class TransectFile:
    """
    A wrapper around individual .h5 files within a transect.
    Similar to UHIFile but designed for transect-level operations.
    """

    def __init__(
        self, filepath: str, track_offset: int = 0, use_corrected: bool = False
    ):
        self.filepath = filepath
        self.track_offset = track_offset
        self.name = os.path.splitext(os.path.basename(filepath))[0]
        self.data = None
        self.wavelengths = None
        self.metadata = {}
        self.use_corrected = use_corrected
        self.dataset_path = (
            "processed/radiance/dataCube_corrected"
            if use_corrected
            else "processed/radiance/dataCube"
        )
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata without loading the full data cube."""
        try:
            with h5py.File(self.filepath, "r") as f:
                # Check for the specified dataset (corrected or original)
                if self.dataset_path in f:
                    self.metadata["shape"] = f[self.dataset_path].shape
                    self.metadata["has_data"] = True
                    self.metadata["dataset_used"] = self.dataset_path
                elif "processed/radiance/dataCube" in f:
                    # Fallback to original dataset if corrected not found
                    self.dataset_path = "processed/radiance/dataCube"
                    self.metadata["shape"] = f[self.dataset_path].shape
                    self.metadata["has_data"] = True
                    self.metadata["dataset_used"] = self.dataset_path
                    if self.use_corrected:
                        print(
                            f"⚠️  dataCube_corrected not found in {self.name}, using dataCube instead"
                        )
                else:
                    self.metadata["has_data"] = False

                if self.metadata.get("has_data", False):
                    # Load wavelengths if available
                    try:
                        self.wavelengths = f[
                            "processed/radiance/calibration/spectral/band2Wavelength"
                        ][()]
                    except KeyError:
                        if self.metadata["shape"]:
                            self.wavelengths = np.arange(self.metadata["shape"][2])

                # Extract timestamp or other relevant metadata
                self.metadata["filepath"] = self.filepath

        except Exception as e:
            print(f"Warning: Could not load metadata from {self.name}: {e}")
            self.metadata["has_data"] = False

    def load_data(self):
        """Load the actual data cube when needed."""
        if self.data is not None:
            return self.data

        try:
            with h5py.File(self.filepath, "r") as f:
                self.data = f[self.dataset_path][()]
                if self.wavelengths is None:
                    try:
                        self.wavelengths = f[
                            "processed/radiance/calibration/spectral/band2Wavelength"
                        ][()]
                    except KeyError:
                        self.wavelengths = np.arange(self.data.shape[2])
            return self.data
        except Exception as e:
            print(f"Error loading data from {self.name} using {self.dataset_path}: {e}")
            return None

    def describe(self):
        """Describe this file's metadata and data if loaded."""
        print(f"\n=== {self.name} ===")
        print(f"File path: {self.filepath}")
        print(f"Track offset: {self.track_offset}")
        print(f"Dataset used: {self.metadata.get('dataset_used', 'Unknown')}")

        if self.metadata.get("has_data", False):
            shape = self.metadata["shape"]
            print(f"Data shape: {shape}")
            print(f" - Tracks: {shape[0]}")
            print(f" - Slit pixels: {shape[1]}")
            print(f" - Wavelengths: {shape[2]}")

            if self.data is not None:
                print(
                    f" - Intensity range: {self.data.min():.3f} to {self.data.max():.3f}"
                )
            else:
                print(" - Data not loaded (use .load_data() to load)")
        else:
            print("❌ No valid data found")


class TransectDataSet:
    """
    A class for handling entire transects composed of multiple .h5 files.

    Usage:
        # Load all files from a transect folder
        transect = TransectDataSet("/path/to/transect/folder")

        # Select specific files to work with
        selected_cube = transect.select_files(["file1", "file2", "file3"])

        # Or select by pattern
        selected_cube = transect.select_files_by_pattern("rad_uhi_20241029_11*")

        # Access the combined data as a single cube-like object
        selected_cube.plot_rgb()
        selected_cube.describe()
    """

    def __init__(self, folder_path: str, use_corrected: bool = False):
        self.folder_path = folder_path
        self.use_corrected = use_corrected
        self.files = self._discover_files()
        dataset_type = "corrected" if use_corrected else "original"
        print(
            f"📂 Discovered {len(self.files)} .h5 files in transect (using {dataset_type} data)"
        )

    def _discover_files(self) -> Dict[str, TransectFile]:
        """Discover all .h5 files in the folder and create TransectFile objects."""
        files = {}

        if not os.path.exists(self.folder_path):
            print(f"❌ Folder not found: {self.folder_path}")
            return files

        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith(".h5"):
                filepath = os.path.join(self.folder_path, filename)
                name = os.path.splitext(filename)[0]
                transect_file = TransectFile(filepath, use_corrected=self.use_corrected)

                if transect_file.metadata.get("has_data", False):
                    files[name] = transect_file
                else:
                    print(f"⚠️  Skipping {filename} (no valid data)")

        return files

    def list_files(self):
        """List all available files in the transect."""
        dataset_type = "corrected" if self.use_corrected else "original"
        print(
            f"\n📋 Available files in {os.path.basename(self.folder_path)} (using {dataset_type} data):"
        )
        print("-" * 70)

        for i, (name, tfile) in enumerate(self.files.items(), 1):
            shape = tfile.metadata.get("shape", "Unknown")
            dataset_used = tfile.metadata.get("dataset_used", "Unknown")
            status = "✅" if tfile.metadata.get("has_data", False) else "❌"
            print(f"{i:2d}. {status} {name}")
            print(f"     Shape: {shape}")
            print(f"     Dataset: {dataset_used}")

        if not self.files:
            print("No valid .h5 files found")

    def describe_all(self):
        """Describe all files in detail."""
        print(f"\n📂 Transect: {os.path.basename(self.folder_path)}")
        print(f"Total files: {len(self.files)}")

        for tfile in self.files.values():
            tfile.describe()

    def select_files(self, file_names: List[str]) -> "CombinedTransectCube":
        """
        Select specific files by name and create a combined cube.

        Args:
            file_names: List of file names (without .h5 extension)

        Returns:
            CombinedTransectCube: Object that behaves like a single cube
        """
        selected_files = []
        missing_files = []

        for name in file_names:
            if name in self.files:
                selected_files.append(self.files[name])
            else:
                missing_files.append(name)

        if missing_files:
            print(f"⚠️  Files not found: {missing_files}")

        if not selected_files:
            raise ValueError("No valid files selected")

        print(f"📦 Selected {len(selected_files)} files for combination")
        return CombinedTransectCube(selected_files, self.folder_path)

    def select_files_by_pattern(self, pattern: str) -> "CombinedTransectCube":
        """
        Select files that match a pattern.

        Args:
            pattern: Glob-like pattern (* and ? wildcards supported)

        Returns:
            CombinedTransectCube: Object that behaves like a single cube
        """
        import fnmatch

        matched_files = []
        for name in self.files.keys():
            if fnmatch.fnmatch(name, pattern):
                matched_files.append(name)

        if not matched_files:
            print(f"❌ No files match pattern: {pattern}")
            print("Available files:")
            for name in self.files.keys():
                print(f"  - {name}")
            raise ValueError(f"No files match pattern: {pattern}")

        print(f"🔍 Pattern '{pattern}' matched {len(matched_files)} files:")
        for name in matched_files:
            print(f"  ✅ {name}")

        return self.select_files(matched_files)

    def select_all_files(self) -> "CombinedTransectCube":
        """Select all available files in the transect."""
        return self.select_files(list(self.files.keys()))


class CombinedTransectCube:
    """
    A cube-like object that combines multiple TransectFiles.
    Provides similar interface to UHIFile but for combined transect data.
    """

    def __init__(self, transect_files: List[TransectFile], transect_path: str):
        self.transect_files = transect_files
        self.transect_path = transect_path
        self.name = f"Combined_{os.path.basename(transect_path)}"

        # Combined properties
        self.data = None
        self.wavelengths = None
        self.track_offset = 0
        self.file_boundaries = []  # Track where each file starts/ends

        self._combine_data()

    def _combine_data(self):
        """Combine data from all selected files into a single cube."""
        print("🔄 Loading and combining data from selected files...")

        all_data = []
        all_wavelengths = []
        current_offset = 0

        for i, tfile in enumerate(self.transect_files):
            print(f"   Loading {tfile.name}...")
            data = tfile.load_data()

            if data is None:
                print(f"   ❌ Failed to load {tfile.name}")
                continue

            # Set track offset for this file
            tfile.track_offset = current_offset

            # Store boundary information
            self.file_boundaries.append(
                {
                    "file": tfile.name,
                    "start_track": current_offset,
                    "end_track": current_offset + data.shape[0] - 1,
                    "n_tracks": data.shape[0],
                }
            )

            all_data.append(data)
            all_wavelengths.append(tfile.wavelengths)
            current_offset += data.shape[0]

        if not all_data:
            raise ValueError("No data could be loaded from selected files")

        # Combine along track axis
        self.data = np.concatenate(all_data, axis=0)

        # Use wavelengths from first file (assuming they're consistent)
        self.wavelengths = all_wavelengths[0]

        # Verify wavelength consistency
        for i, wl in enumerate(all_wavelengths[1:], 1):
            if not np.allclose(wl, self.wavelengths):
                print(f"⚠️  Wavelength mismatch in file {self.transect_files[i].name}")

        print(f"✅ Combined cube shape: {self.data.shape}")
        print(f"   Total tracks: {self.data.shape[0]}")
        print(f"   Slit pixels: {self.data.shape[1]}")
        print(f"   Wavelengths: {self.data.shape[2]}")

    def describe(self):
        """Describe the combined cube."""
        print(f"\n=== {self.name} ===")
        print(f"Combined from {len(self.transect_files)} files")
        print(f"Total shape: {self.data.shape}")
        print(f" - Total tracks: {self.data.shape[0]}")
        print(f" - Slit pixels: {self.data.shape[1]}")
        print(f" - Wavelengths: {self.data.shape[2]}")
        print(f" - Intensity range: {self.data.min():.3f} to {self.data.max():.3f}")

        print(f"\n📋 File composition:")
        for boundary in self.file_boundaries:
            print(
                f"  {boundary['file']}: tracks {boundary['start_track']}-{boundary['end_track']} ({boundary['n_tracks']} tracks)"
            )

    def apply_illumination_correction(self):
        """
        Compute per-slit, along-track normalization and store only in self.data_corrected.
        Original self.data remains untouched.
        """
        data = self.data.astype(float)
        # Reference spectrum per slit (median over tracks)
        s_ref = np.median(data, axis=0)
        s_ref[s_ref == 0] = 1.0
        # Corrected cube
        self.data_corrected = data / s_ref[np.newaxis, :, :]
        print("✅ data_corrected ready")
        return self.data_corrected

    def slice_tracks(self, start: int, end: int) -> "CombinedTransectCube":
        """
        Create a new CombinedTransectCube with a subset of tracks.
        """
        if start < 0 or end > self.data.shape[0] or start >= end:
            raise ValueError(
                f"Invalid slice range [{start}:{end}] for data with {self.data.shape[0]} tracks"
            )

        # Create a dummy transect file for the sliced data
        from types import SimpleNamespace

        sliced_file = SimpleNamespace()
        sliced_file.name = f"{self.name}_slice_{start}_{end}"
        sliced_file.track_offset = start
        sliced_file.load_data = lambda: self.data[start:end, :, :]
        sliced_file.wavelengths = self.wavelengths

        # Create new combined cube
        sliced_cube = CombinedTransectCube([sliced_file], self.transect_path)
        sliced_cube.name = f"{self.name}_slice_{start}_{end}"

        return sliced_cube

    def get_file_info(self, track_index: int) -> Dict:
        """Get information about which file a specific track belongs to."""
        for boundary in self.file_boundaries:
            if boundary["start_track"] <= track_index <= boundary["end_track"]:
                return boundary
        return None


# Convenience function for quick usage
def load_transect(folder_path: str, use_corrected: bool = False) -> TransectDataSet:
    """
    Convenience function to quickly load a transect.

    Args:
        folder_path: Path to the folder containing .h5 files
        use_corrected: If True, loads from dataCube_corrected; if False, loads from dataCube

    Usage:
        # Load original data
        transect = load_transect("/path/to/transect/folder")

        # Load corrected data
        transect = load_transect("/path/to/transect/folder", use_corrected=True)

        cube = transect.select_files(["file1", "file2"])
        cube.plot_rgb()
    """
    return TransectDataSet(folder_path, use_corrected=use_corrected)
