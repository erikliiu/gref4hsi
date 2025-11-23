"""
Test script to verify the illumination correction optimization works correctly.
Compares results between old approach and new approach on small dataset.
"""

import numpy as np
import h5py
import tempfile
import os
from pathlib import Path


def create_test_data():
    """Create small test HDF5 files to verify logic."""
    # Create 2 small test files
    # Shape: (10 tracks, 5 slits, 3 bands)
    T1, T2 = 10, 8
    S, B = 5, 3

    files = []
    temp_dir = tempfile.mkdtemp()

    for i, T in enumerate([T1, T2]):
        filepath = Path(temp_dir) / f"test_file_{i}.h5"

        # Create synthetic data with known pattern
        # Each slit-band has values = track_index + slit + band + file_offset
        data = np.zeros((T, S, B), dtype=np.float32)
        for t in range(T):
            for s in range(S):
                for b in range(B):
                    data[t, s, b] = t + s * 10 + b * 100 + i * 1000

        with h5py.File(filepath, "w") as f:
            f.create_dataset("rad", data=data)

        files.append(str(filepath))

    return files, temp_dir, S, B, T1 + T2


def test_old_approach(files, S, B):
    """Simulate old approach: open files S*B times."""
    print("Testing OLD approach (per slit-band)...")

    ref_values = {}
    file_opens = 0

    for s in range(S):
        for b in range(B):
            # Collect data for this slit-band from all files
            values = []
            for file_path in files:
                with h5py.File(file_path, "r") as f:
                    file_opens += 1
                    values.append(f["rad"][:, s, b])

            ts = np.concatenate(values)
            ref = np.nanmedian(ts)
            ref_values[(s, b)] = ref

    print(f"  File opens: {file_opens}")
    return ref_values, file_opens


def test_new_approach(files, S, B):
    """Simulate new approach: open files once each."""
    print("Testing NEW approach (file-by-file)...")

    ref_values = {}
    file_opens = 0

    # Collect all slit data
    all_slit_data = {s: [] for s in range(S)}

    for file_path in files:
        with h5py.File(file_path, "r") as f:
            file_opens += 1
            file_data = f["rad"][:, :, :]

            for s in range(S):
                all_slit_data[s].append(file_data[:, s, :])

    # Compute references
    for s in range(S):
        slit_data = np.concatenate(all_slit_data[s], axis=0)
        for b in range(B):
            ts = slit_data[:, b]
            ref = np.nanmedian(ts)
            ref_values[(s, b)] = ref

    print(f"  File opens: {file_opens}")
    return ref_values, file_opens


def main():
    print("=" * 60)
    print("TESTING ILLUMINATION CORRECTION OPTIMIZATION")
    print("=" * 60)

    # Create test data
    files, temp_dir, S, B, T_total = create_test_data()
    print(f"\nTest data created:")
    print(f"  Files: {len(files)}")
    print(f"  Shape per file: varies × {S} slits × {B} bands")
    print(f"  Total tracks: {T_total}")

    # Test old approach
    print(f"\n{'-'*60}")
    old_refs, old_opens = test_old_approach(files, S, B)

    # Test new approach
    print(f"\n{'-'*60}")
    new_refs, new_opens = test_new_approach(files, S, B)

    # Compare results
    print(f"\n{'-'*60}")
    print("COMPARISON:")
    print(f"  Old approach file opens: {old_opens}")
    print(f"  New approach file opens: {new_opens}")
    print(f"  Speedup factor: {old_opens / new_opens:.1f}x")

    # Verify correctness
    print(f"\n{'='*60}")
    print("CORRECTNESS CHECK:")
    all_match = True
    for key in old_refs:
        if not np.isclose(old_refs[key], new_refs[key]):
            print(f"  ❌ MISMATCH at {key}: {old_refs[key]} vs {new_refs[key]}")
            all_match = False

    if all_match:
        print("  ✅ All reference values MATCH!")
        print("  ✅ Optimization is CORRECT!")
    else:
        print("  ❌ ERRORS FOUND - optimization has bugs!")

    # Print sample values
    print(f"\nSample reference values:")
    for s in range(min(2, S)):
        for b in range(min(2, B)):
            print(
                f"  Slit {s}, Band {b}: {old_refs[(s,b)]:.2f} (old) vs {new_refs[(s,b)]:.2f} (new)"
            )

    # Cleanup
    for f in files:
        os.remove(f)
    os.rmdir(temp_dir)

    print(f"\n{'='*60}")
    if all_match:
        print("✅ TEST PASSED - Safe to use optimized version!")
    else:
        print("❌ TEST FAILED - Do NOT use optimized version!")
    print("=" * 60)


if __name__ == "__main__":
    main()
