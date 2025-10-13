import sys

sys.path.insert(0, "..")
from utils import georef
import config

# Load full transect
transect = georef.load_transect(config.OUTPUT_FOLDER)

print("=" * 70)
print("ALL FILES IN TRANSECT:")
print("=" * 70)
cumulative_tracks = 0
for i, (name, gf) in enumerate(transect.files.items()):
    # Load the h5 file to get number of frames
    import h5py

    h5_path = f"{config.OUTPUT_FOLDER}/{name}.h5"
    try:
        with h5py.File(h5_path, "r") as f:
            n_frames = f["georef"]["grid_ecef"].shape[0]
            print(f"{i}: {name}")
            print(f"   Frames: {n_frames}")
            print(
                f"   Global track range: {cumulative_tracks} to {cumulative_tracks + n_frames - 1}"
            )
            cumulative_tracks += n_frames
    except Exception as e:
        print(f"{i}: {name} - Error: {e}")

print(f"\n{'=' * 70}")
print(f"TOTAL TRACKS: {cumulative_tracks}")
print(f"{'=' * 70}\n")

# Now check the selected files
print("SELECTED FILES (4 and 5):")
print("=" * 70)
cube = transect.select_files(["rad_uhi_20241029_115057_4", "rad_uhi_20241029_115057_5"])
print(
    f"Combined cube shape: {cube.X_ecef.shape if cube.X_ecef is not None else 'Not built yet'}"
)

# Track range in question
track_start = 3039
track_end = 4029
print(
    f"\nRequested track range: {track_start} to {track_end} ({track_end - track_start} tracks)"
)
print(f"This refers to GLOBAL indices across all files, not local to selected files!")
