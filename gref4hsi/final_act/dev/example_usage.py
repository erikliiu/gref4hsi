"""
Example usage of dev3_timestamp_complete module

This script demonstrates how to:
1. Load HDF5 files with georeferenced hyperspectral data
2. Select specific files or patterns
3. Plot with UTC timestamp display
4. Use different coordinate systems
"""

import dev3_timestamp_complete as geo

# ============================================================================
# Example 1: Load from config.OUTPUT_FOLDER
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Load all files from config.OUTPUT_FOLDER")
print("=" * 70)

try:
    # This uses config.OUTPUT_FOLDER if available
    transect = geo.load_transect()
    transect.list_files()

    # Select all files
    cube = transect.select_all_files()

    # Plot entire dataset with timestamps
    print("\n📊 Plotting full dataset...")
    cube.plot_georef(coordinate_system="NED", interactive=True)

except Exception as e:
    print(f"❌ Error: {e}")


# ============================================================================
# Example 2: Load from specific folder
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Load from specific folder")
print("=" * 70)

folder_path = r"C:\path\to\your\h5\files"  # CHANGE THIS

try:
    transect = geo.load_transect(folder_path=folder_path)
    transect.list_files()

    # Select files by pattern
    cube = transect.select_files_by_pattern("rad_uhi_*")

    # Plot specific track range
    print("\n📊 Plotting tracks 100-200...")
    cube.plot_georef(
        coordinate_system="NED",
        track_start=100,
        track_end=200,
        interactive=True,
        show_file_boundaries=True,
    )

except Exception as e:
    print(f"❌ Error: {e}")


# ============================================================================
# Example 3: Select specific files
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Select specific files")
print("=" * 70)

try:
    transect = geo.load_transect()
    transect.list_files()

    # Select specific files by name
    file_names = [
        "rad_uhi_20241029_115057_1",
        "rad_uhi_20241029_115057_2",
        "rad_uhi_20241029_115057_3",
    ]

    cube = transect.select_files(file_names)

    # Plot in LATLON coordinate system
    print("\n📊 Plotting in Lat/Lon coordinates...")
    cube.plot_georef(coordinate_system="LATLON", interactive=True)

except Exception as e:
    print(f"❌ Error: {e}")


# ============================================================================
# Example 4: Different coordinate systems and options
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Advanced options")
print("=" * 70)

try:
    transect = geo.load_transect()
    cube = transect.select_all_files()

    # Plot in ECEF with custom origin
    print("\n📊 Plotting in ECEF coordinates...")
    fig, ax = cube.plot_georef(
        coordinate_system="ECEF",
        use_local_origin=True,
        origin=(60.8, 10.7, 0.0),  # Custom origin (lat, lon, h)
        red_wl=670.0,  # Custom wavelengths
        green_wl=550.0,
        blue_wl=450.0,
        normalize=True,
        figsize=(14, 10),
        interactive=True,
        return_fig=True,  # Return figure for further customization
    )

    # Further customize the plot
    ax.grid(True, alpha=0.3)
    ax.set_title(ax.get_title() + "\n[Custom wavelengths]", fontsize=10)

except Exception as e:
    print(f"❌ Error: {e}")


print("\n" + "=" * 70)
print("✅ Examples complete!")
print("=" * 70)
