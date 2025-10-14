import sys, os

sys.path.append(os.path.abspath("../"))
from utils.other.detrend_mbes import MBESDetrender, AutoTuneMBES, ScoreWeights
from gref_pipeline import config
import matplotlib.pyplot as plt

plt.ion()

tif_path = config.MBES_GEOTIFF

# Simplified! UHI footprint is now loaded automatically in load()
# Just create the MBESDetrender and plot with footprint_style
mb_ned = (
    MBESDetrender(
        tif_path,
        coord_system="ned",
        ned_origin=(config.LON0, config.LAT0, config.H0),
        epsg_utm=config.EPSG_UTM,
        epsg_geo=config.EPSG_GEOGRAPHIC,
    )
    .load()  # Automatically loads UHI footprint using config defaults
    .detrend(
        order_x=3,
        smooth_baseline_m=0.80,
        smooth_tilt_m=0.70,
        smooth_center_m=1.0,
        robust=True,
        central_frac=0.8,
    )
)

# Plot with UHI footprint overlay - that's it!
fig = mb_ned.plot_triptych(footprint_style="outline", show=True)
print("✅ Done! UHI footprint loaded automatically and displayed as black outline.")

# %matplotlib qt
# fig = mb_ned.plot_triptych(footprint_style="outline", show=True, use_adjusted=True)

fig1 = mb_ned.plot_uhi_mbes_comparison(
    use_adjusted=True, normalization_method="zscore", show=True
)

# %matplotlib qt
# fig_scatter1 = mb_ned.plot_density_scatter(
#     use_adjusted=True, normalization_method="zscore", show=True, figsize=(4, 4)
# )
