"""
Quick diagnostic: Check altitude statistics from navigation CSV
to determine appropriate max_ray_length setting.
"""

import pandas as pd
import numpy as np

# Load the CSV navigation data
# csv_path = r"C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\gref4hsi\Input\NavData\EELY_2024_mission.csv"
csv_path = r"C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\gref4hsi\Input\NavData\EELY_2024_mission.csv"

print("Loading navigation data...")
df = pd.read_csv(csv_path)

print(f"\n📊 Navigation Data Summary:")
print(f"   Total records: {len(df)}")
print(f"\n🌊 Depth Statistics (meters below surface):")
print(f"   Mean: {df['depth'].mean():.1f} m")
print(f"   Min:  {df['depth'].min():.1f} m")
print(f"   Max:  {df['depth'].max():.1f} m")
print(f"   Std:  {df['depth'].std():.1f} m")

print(f"\n📏 Altitude Statistics (meters above seafloor):")
print(f"   Mean: {df['altitude'].mean():.1f} m")
print(f"   Min:  {df['altitude'].min():.1f} m")
print(f"   Max:  {df['altitude'].max():.1f} m")
print(f"   Std:  {df['altitude'].std():.1f} m")

# Calculate recommended max_ray_length
max_altitude = df["altitude"].max()
recommended = max_altitude * 1.5  # 50% safety margin
print(f"\n💡 Recommendation:")
print(f"   Current max_ray_length: 20 m")
print(f"   Maximum altitude above seafloor: {max_altitude:.1f} m")
print(f"   Recommended max_ray_length: {recommended:.1f} m (150% of max altitude)")
print(
    f"   Suggested setting: {int(np.ceil(recommended / 10) * 10)} m (rounded up to nearest 10)"
)

# Check how many records have altitude > 20m
above_20 = (df["altitude"] > 20).sum()
pct_above = above_20 / len(df) * 100
print(f"\n⚠️  Records with altitude > 20m: {above_20} ({pct_above:.1f}%)")
if pct_above > 50:
    print(f"   ❌ CRITICAL: More than half your data has altitude > 20m!")
    print(f"   With max_ray_length=20, these rays cannot reach the seafloor.")
