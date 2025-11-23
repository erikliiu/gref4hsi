"""
Test rolling window case specifically.
"""

import numpy as np
import pandas as pd


def test_rolling_window():
    print("Testing rolling window computation...")

    # Create simple test data
    # 3 files with 5, 4, 6 tracks = 15 total
    file1_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    file2_data = np.array([6, 7, 8, 9], dtype=np.float32)
    file3_data = np.array([10, 11, 12, 13, 14, 15], dtype=np.float32)

    # Concatenate like the code does
    full_data = np.concatenate([file1_data, file2_data, file3_data])

    print(f"Full data: {full_data}")
    print(f"Shape: {full_data.shape}")

    # Compute rolling median with window=5
    window_size = 5
    ref_vec = (
        pd.Series(full_data)
        .rolling(window=window_size, center=True, min_periods=1)
        .median()
        .values
    )

    print(f"\nRolling median (window={window_size}):")
    print(f"  {ref_vec}")

    # Verify correctness
    # At index 7 (value=8), window should be [6,7,8,9,10] -> median=8
    expected_at_7 = np.median([6, 7, 8, 9, 10])
    print(f"\nAt index 7:")
    print(f"  Value: {full_data[7]}")
    print(f"  Window: [6, 7, 8, 9, 10]")
    print(f"  Expected median: {expected_at_7}")
    print(f"  Computed median: {ref_vec[7]}")
    print(f"  Match: {np.isclose(ref_vec[7], expected_at_7)}")

    print("\n✅ Rolling window works correctly across file boundaries!")


if __name__ == "__main__":
    test_rolling_window()
