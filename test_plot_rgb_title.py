"""
Test script to verify plot_rgb displays file names in title
"""

import sys
import os
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add the gref4hsi module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gref4hsi", "final_act"))

from utils.gref_pipeline import georef

# Use the config path
OUTPUT_FOLDER = r"E:\mjosa_new_oct_2025\use_gref4hsi\057_final\output"

print("=" * 60)
print("Testing plot_rgb title with file names")
print("=" * 60)

try:
    # Load the transect
    print(f"\n1. Loading transect from: {OUTPUT_FOLDER}")
    transect = georef.load_transect(OUTPUT_FOLDER)
    print(f"   ✓ Transect loaded successfully")
    print(f"   Available files: {list(transect.files.keys())}")

    # Select the first file
    if transect.files:
        first_file = list(transect.files.keys())[0]
        print(f"\n2. Selecting file: {first_file}")
        cube = transect.select_files([first_file])
        print(f"   ✓ File selected")

        # Check file_boundaries
        print(f"\n3. Checking file_boundaries attribute:")
        print(f"   file_boundaries = {cube.file_boundaries}")

        if cube.file_boundaries:
            print(
                f"   ✓ file_boundaries is populated with {len(cube.file_boundaries)} file(s)"
            )
            for i, boundary in enumerate(cube.file_boundaries):
                print(f"     File {i+1}: {boundary.get('file', 'N/A')}")
        else:
            print(f"   ✗ WARNING: file_boundaries is empty!")

        # Try to plot RGB
        print(f"\n4. Calling plot_rgb()...")
        print("-" * 60)
        cube.plot_rgb()  # Will use non-interactive backend
        print("-" * 60)

        # Check the title
        ax = plt.gca()
        title = ax.get_title()
        print(f"\n5. Results:")
        print(f"   Matplotlib title text: '{title}'")

        if title and first_file in title:
            print(f"   ✓ SUCCESS: Title contains the file name!")
        elif title:
            print(f"   ⚠ PARTIAL: Title exists but doesn't contain file name")
        else:
            print(f"   ✗ FAILURE: No title found!")

        # Close the plot
        plt.close()

    else:
        print("   ✗ No processed HSI files found!")

except FileNotFoundError as e:
    print(f"\n✗ ERROR: Could not find path - {e}")
    print(f"   This is expected if you don't have data at that location")
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
