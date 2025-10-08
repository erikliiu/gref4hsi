"""
Diagnostic: Check vehicle speed to verify if 10-second buffer is sufficient
"""

import pandas as pd
import numpy as np

# Load CSV
csv_path = (
    r"E:\mj osa_new_oct_2025\use_gref4hsi\057\input\NavData\EELY_2024_mission.csv"
)
df = pd.read_csv(csv_path)

# Calculate speed from position changes
from pyproj import Geod

geod = Geod(ellps="WGS84")

speeds = []
for i in range(1, len(df)):
    _, _, dist = geod.inv(
        df["longitude [deg]"].iloc[i - 1],
        df["latitude [deg]"].iloc[i - 1],
        df["longitude [deg]"].iloc[i],
        df["latitude [deg]"].iloc[i],
    )
    dt = (
        df["timestamp [unix epoch s]"].iloc[i]
        - df["timestamp [unix epoch s]"].iloc[i - 1]
    )
    if dt > 0:
        speed = dist / dt  # m/s
        speeds.append(speed)

speeds = np.array(speeds)

print(f"\n🚤 Vehicle Speed Analysis:")
print(f"   Mean speed: {np.mean(speeds):.2f} m/s")
print(f"   Median speed: {np.median(speeds):.2f} m/s")
print(f"   Max speed: {np.max(speeds):.2f} m/s")
print(f"   95th percentile: {np.percentile(speeds, 95):.2f} m/s")

print(f"\n⏱️  Time Buffer Analysis (for 2.5m HSI forward offset):")
print(
    f"   At mean speed ({np.mean(speeds):.2f} m/s): need {2.5/np.mean(speeds):.1f} sec buffer"
)
print(
    f"   At max speed ({np.max(speeds):.2f} m/s): need {2.5/np.max(speeds):.1f} sec buffer"
)
print(f"   Current buffer: 10.0 sec")

if 10.0 < 2.5 / np.mean(speeds):
    print(f"\n   ❌ PROBLEM: Buffer is TOO SMALL!")
else:
    print(f"\n   ✅ Buffer should be sufficient")
