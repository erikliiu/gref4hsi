"""
Quick test to check H5 file structure and get timestamps for track range
"""

import sys
from pathlib import Path
import h5py
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from gref_pipeline import config

h5_folder = Path(config.H5_FOLDER)

print("Checking H5 files:")
for uhi_file in config.UHI_FILES:
    h5_path = h5_folder / f"{uhi_file}.h5"
    print(f"\n{'='*60}")
    print(f"File: {uhi_file}.h5")

    if not h5_path.exists():
        print(f"❌ File not found!")
        continue

    with h5py.File(h5_path, "r") as hf:
        print(f"\nDatasets in file:")

        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

        hf.visititems(print_structure)
