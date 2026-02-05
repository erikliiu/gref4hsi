"""Test PDF quality with different interpolation settings."""

import sys
from pathlib import Path

# Setup path like notebook does
mjosa_code_root = Path(__file__).parent
sys.path.insert(0, str(mjosa_code_root))

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np

# Load the cube like notebook does
print("Loading cube...")
from utils.uhi import georef
from utils.uhi.georef import load_transect
from utils.common import config

transect_028 = load_transect(config.TRANSECT_028_OUTPUT)
cube = transect_028.select_files(
    [
        "rad_uhi_20241029_125028_1",
        "rad_uhi_20241029_125028_2",
        "rad_uhi_20241029_125028_3",
        "rad_uhi_20241029_125028_4",
        "rad_uhi_20241029_125028_5",
    ]
)
cube.apply_illumination_correction_v2()

print("Creating plot...")

test_dir = mjosa_code_root / "images_for_publication" / "TEST_028"

# Get the RGB data directly
import numpy as np

# Get RGB composite (full)
rgb = cube._cube_corrected[
    :, :, [cube._band_indices[677], cube._band_indices[549], cube._band_indices[490]]
]

# Normalize
for i in range(3):
    band = rgb[:, :, i]
    p2, p98 = np.percentile(band[~np.isnan(band)], [2, 98])
    rgb[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)

# Crop a region (adjust to valid range)
rgb_crop = rgb[100:200, 400:900, :]
print(f"Crop shape: {rgb_crop.shape}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(rgb_crop, aspect="auto")
ax.plot([50, 450], [50, 50], "r-", linewidth=3)  # Add a line
ax.set_xticks([])
ax.set_yticks([])

# Save PNG first (reference)
png_path = test_dir / "real_test.png"
plt.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"PNG saved: {png_path.stat().st_size / 1024:.1f} KB")

# Save PDF without modification
pdf_default = test_dir / "real_test_default.pdf"
plt.savefig(pdf_default, bbox_inches="tight", dpi=300)
print(f"PDF (default): {pdf_default.stat().st_size / 1024:.1f} KB")

# Now change interpolation to 'none' before saving PDF
for artist in ax.get_children():
    if isinstance(artist, AxesImage):
        artist.set_interpolation("none")
        print(f"Changed interpolation to none")

pdf_none = test_dir / "real_test_interp_none.pdf"
plt.savefig(pdf_none, bbox_inches="tight", dpi=300)
print(f"PDF (interp=none): {pdf_none.stat().st_size / 1024:.1f} KB")

plt.close()
print("\nDone! Compare these files:")
print(f"  - {png_path.name} (reference - should look same as notebook)")
print(f"  - {pdf_default.name} (might be blurry)")
print(f"  - {pdf_none.name} (should match PNG quality)")
