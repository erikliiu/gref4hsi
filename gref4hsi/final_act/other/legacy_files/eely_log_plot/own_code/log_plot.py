import sys
import os

sys.path.append(os.path.abspath("../"))

from Eelume import PyPost as pp
import matplotlib.pyplot as plt
import datetime as datetime
import os
import numpy as np
import helper

# Load the database file
filename = r"E:\mjosa\29\log_files\LOG_2024-10-29_10-13-32.db3"
db = pp.DatabaseHandler(filename)
# xyz_file = r"C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\test_georef\mbes_20k.txt"

# Set the origin for latitude/longitude to UTM conversion
origin = (60.8011575, 10.7122345)

# Set the time interval for loading data

# this time interval is good for start and stop
start_time = datetime.datetime(2024, 10, 29, 10, 10, 00, tzinfo=datetime.timezone.utc)
end_time = datetime.datetime(2024, 10, 29, 13, 15, 00, tzinfo=datetime.timezone.utc)


# Create the MotionAnalyzer object
# analyzer = pp.MotionAnalyzer(db, north_east_origin=origin)
analyzer = pp.MotionAnalyzer(
    db, north_east_origin=origin, start_time=start_time, end_time=end_time
)
# analyzer = pp.MotionAnalyzer(db, start_time=start_time, end_time=end_time)

# moved forward 8.5 min
# this is UTC
transect_dict_uhi = {
    "2024-10-29": [
        ("10:50:51", "00:08:15", "uhi_20241029_104221"),
        ("11:23:50", "00:01:14", "uhi_20241029_111520_from1"),
        ("11:26:32", "00:05:03", "uhi_20241029_111520_from3"),
        ("11:59:27", "00:08:50", "uhi_20241029_115057"),
        ("12:25:09", "00:08:24", "uhi_20241029_121639"),
        ("12:58:58", "00:08:22", "uhi_20241029_125028"),
        ("13:10:05", "00:03:17", "uhi_20241029_130135"),
        ("13:43:34", "00:01:26", "uhi_20241029_133504"),
        ("13:45:18", "00:03:11", "uhi_20241029_133648"),
    ]
}

# transect_dict = {
#     "2024-10-29": [
#         ("10:42:21", "00:08:15", "uhi_20241029_104221"),
#         ("11:15:20", "00:01:14", "uhi_20241029_111520_from1"),
#         ("11:18:02", "00:05:03", "uhi_20241029_111520_from3"),
#         ("11:50:57", "00:08:50", "uhi_20241029_115057"),
#         ("12:16:39", "00:08:24", "uhi_20241029_121639"),
#         ("12:50:28", "00:08:22", "uhi_20241029_125028"),
#         ("13:01:35", "00:03:17", "uhi_20241029_130135"),
#         ("13:35:04", "00:01:26", "uhi_20241029_133504"),
#         ("13:36:48", "00:03:11", "uhi_20241029_133648"),
#     ],
#     # Add additional dates and their transect lists as needed.
# }|


######################### PLOTS #########################
# analyzer.plot_position_scatter(figure_number=1, skip_non_control=True)
# analyzer.correct_navigation_data()
analyzer.plot_position_scatter(figure_number=1, skip_non_control=True)

analyzer.plot_pose_timeseries(figure_number=2)
analyzer.plot_pose_timeseries_separate()  # yaw is not goodr
# analyzer.plot_altitude_timeseries(date_str="2024-10-29", transect_dict=transect_dict)
# analyzer.plot_altitude_and_depth()
# analyzer.plot_3d_trajectory()
analyzer.plot_yaw_angle()  # yaw is good
helper.plot_live_3d_highlighted(
    analyzer,
    "2024-10-29",
    transect_dict=transect_dict_uhi,
    invert_depth=True,
    interval=100,  # milliseconds per frame (100ms = 10 frames per second)
    # xyz_file=xyz_file,
)

plt.show()

######################### LOAD .nav for immersion #########################
# nav_file_path = r"C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\29\.nav file\nav_file.txt"
# analyzer.create_nav_file(nav_file_path)
# nav_file_path = r"C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\29\.nav file\nav_file.nav"
# analyzer.create_nav_file(nav_file_path)


######################### LOAD .csv for Håvards repo #########################
# load new analyzer without origo so we can get ECEF
# analyzer = pp.MotionAnalyzer(db, start_time=start_time, end_time=end_time)
# csv_file_path = r"E:\mjosa\mission_dir\Input\nav_data.csv"
# analyzer.create_csv_file(csv_file_path)
