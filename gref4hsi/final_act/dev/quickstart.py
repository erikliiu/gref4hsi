"""
QUICK START GUIDE - Use this script to test your new module!

This script demonstrates the simplest way to use dev3_timestamp_complete.py
Replace the paths and file names with your actual data.
"""

import sys
import os

# Make sure we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import the module
import dev3_timestamp_complete as geo

print("=" * 70)
print("🚀 QUICK START - Georeferenced HSI Plotting with Timestamps")
print("=" * 70)

# ============================================================================
# OPTION 1: Use config.OUTPUT_FOLDER (if you have config.py set up)
# ============================================================================

try:
    print("\n📂 Attempting to load from config.OUTPUT_FOLDER...")
    transect = geo.load_transect()  # Uses config.OUTPUT_FOLDER

    print("\n📋 Available files:")
    transect.list_files()

    if not transect.files:
        print("\n⚠️  No files found in config.OUTPUT_FOLDER")
    else:
        print(f"\n✅ Found {len(transect.files)} file(s)")

        # Select all files
        cube = transect.select_all_files()

        # Plot the data
        print("\n📊 Creating plot...")
        cube.plot_georef(
            coordinate_system="NED", interactive=True, show_file_boundaries=True
        )

        print("\n✅ Plot complete! Click on pixels to inspect coordinates.")

except Exception as e:
    print(f"\n❌ Error using config.OUTPUT_FOLDER: {e}")
    print("\n💡 Try Option 2 below - specify folder path directly")


# ============================================================================
# OPTION 2: Specify folder path directly
# ============================================================================

print("\n" + "=" * 70)
print("📂 OPTION 2: Specify folder path directly")
print("=" * 70)

# CHANGE THIS to your actual data folder:
FOLDER_PATH = r"C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\gref4hsi\data\processed"

print(f"\n📂 Loading from: {FOLDER_PATH}")

try:
    transect = geo.load_transect(folder_path=FOLDER_PATH)

    print("\n📋 Available files:")
    transect.list_files()

    if not transect.files:
        print("\n⚠️  No valid HDF5 files found")
        print("Make sure the folder contains .h5 files with:")
        print("  - processed/radiance/dataCube")
        print("  - processed/georef/points_ecef_crs")
        print("  - processed/radiance/timestamps (optional)")
    else:
        print(f"\n✅ Found {len(transect.files)} file(s)")

        # Select all files
        cube = transect.select_all_files()

        # Plot specific track range (adjust these numbers for your data)
        print("\n📊 Creating plot with track range...")
        cube.plot_georef(
            coordinate_system="NED",
            track_start=0,  # Start from beginning
            track_end=100,  # Plot first 100 tracks (change this!)
            interactive=True,
            show_file_boundaries=True,
        )

        print("\n✅ Plot complete!")
        print("\n💡 Tips:")
        print("  - Click on pixels to see coordinates and RGB values")
        print("  - The title shows UTC time range for displayed tracks")
        print("  - Yellow dotted lines mark file boundaries")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n🔍 Troubleshooting:")
    print("  1. Check that FOLDER_PATH exists and contains .h5 files")
    print("  2. Verify HDF5 files have required datasets (see README)")
    print("  3. Try with interactive Python to debug:")
    print("     >>> import dev3_timestamp_complete as geo")
    print("     >>> transect = geo.load_transect(r'YOUR_PATH')")
    print("     >>> transect.list_files()")


# ============================================================================
# OPTION 3: Select specific files
# ============================================================================

print("\n" + "=" * 70)
print("📂 OPTION 3: Select specific files by name")
print("=" * 70)

# CHANGE THESE to your actual file names (without .h5 extension):
FILE_NAMES = [
    "rad_uhi_20241029_115057_1",
    "rad_uhi_20241029_115057_2",
    "rad_uhi_20241029_115057_3",
]

try:
    transect = geo.load_transect()  # or specify folder_path=...

    print(f"\n📂 Selecting {len(FILE_NAMES)} specific files...")
    cube = transect.select_files(FILE_NAMES)

    print("\n📊 Creating plot in LATLON coordinates...")
    cube.plot_georef(coordinate_system="LATLON", interactive=True)

    print("\n✅ Plot complete!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("💡 Make sure the file names match exactly (without .h5)")


# ============================================================================
# Summary and Next Steps
# ============================================================================

print("\n" + "=" * 70)
print("📚 NEXT STEPS")
print("=" * 70)
print(
    """
1️⃣  Edit this script:
   - Change FOLDER_PATH to your actual data folder
   - Change FILE_NAMES to match your file names
   - Adjust track_start and track_end for your data

2️⃣  Run the script:
   python quickstart.py

3️⃣  Experiment with options:
   - Try coordinate_system="LATLON", "NED", or "ECEF"
   - Adjust red_wl, green_wl, blue_wl for different RGB combinations
   - Use track_start/track_end to plot specific regions

4️⃣  Read the documentation:
   - See README_TIMESTAMP_MODULE.md for detailed usage
   - Check example_usage.py for more advanced examples

5️⃣  Integrate into your workflow:
   import dev3_timestamp_complete as geo
   # Use geo.load_transect(), cube.plot_georef(), etc.

❓ Need help?
   - Check README_TIMESTAMP_MODULE.md
   - Verify HDF5 file structure matches expected format
   - Test with interactive Python to debug
"""
)

print("=" * 70)
print("✨ Happy plotting!")
print("=" * 70)
