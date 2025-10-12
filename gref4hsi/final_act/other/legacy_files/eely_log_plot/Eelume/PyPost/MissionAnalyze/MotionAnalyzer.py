# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

import pandas as pd
from scipy.interpolate import interp1d

import Eelume.PyPost as pp
import numpy as np
from Eelume.PyPost.CoordinateConversion import lat_long_2_utm
import fnmatch
import datetime
import matplotlib.pyplot as plt
import os  # Add this import for file operations
import matplotlib.dates as mdates  # Add this import for date formatting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # Needed for 3D projection in some matplotlib versions
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider  # Add this import for the slider
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D projection is available
import Eelume.PyPost as pp  # Assuming your library is imported as pp
from math import cos, sin, radians

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.dates as mdates

# from datetime import datetime
from math import cos, sin, radians

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import datetime
from math import cos, sin, radians

from own_code import helper
from pyproj import Transformer

import pandas as pd

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings

"""
Class that contains tools for loading and plotting motion data (pose, velocity, etc)
"""


class MotionAnalyzer(object):
    def __init__(
        self,
        db: pp.DatabaseHandler,
        start_time=None,
        end_time=None,
        north_east_origin=None,
    ):
        """
        Constructor

        Parameters
        ----------
        db : pp.DatabaseHandler
            Log database object
        start_time : datetime
            Start time for analysis. The default is None.
        end_time : datetime
            End time for analysis. The default is None.
        north_east_origin : tuple, optional
            The north/east origin given as longitude and latitude coordinates.
            The default is None and data will be plotted as longitude/latitude.
        """
        self.__db = db
        self.set_time_interval(start_time, end_time)

        self.north_east_origin = north_east_origin
        if north_east_origin is not None:
            north, east, temp_num, temp_letter = (
                lat_long_2_utm(  # ? UTM uses meters as unit instread of degrees that latitude and longitude uses
                    north_east_origin[0], north_east_origin[1]
                )
            )
            self.offset_ned_origin = (-north, -east)
            # print(f"UTM num {temp_num} letter {temp_letter}")
        else:
            self.offset_ned_origin = None

        self.__set_time_axis_plot_format()

    def set_time_interval(self, start_time=None, end_time=None):
        """
        Changes the start and end time for analysis and reloads data.

        Parameters
        ----------
        start_time : datetime
            Start time for analysis. The default is None.
        end_time : datetime
            End time for analysis. The default is None.
        """
        self.__start_time = start_time
        self.__end_time = end_time
        self.__reload()

    def plot_pose_timeseries(self, figure_number=None, **kwargs):
        """
        Plots timeseries for position/attitude navigation data and
        position/attitude motion controller references.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs :
            key-value argument which is passed directly to the
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.
        """
        if self.offset_ned_origin is None:
            ylabel = (
                "Latitude [deg]",
                "Longitude [deg]",
                "Depth [m]",
                "Roll [deg]",
                "Pitch [deg]",
                "Yaw [deg]",
            )
        else:
            ylabel = (
                "North [m]",
                "East [m]",
                "Down [m]",
                "Roll [deg]",
                "Pitch [deg]",
                "Yaw [deg]",
            )

        pp.plot_line(
            *pp.interleave_signals(self.load_pose(), self.load_pose_refs()),
            figure_number=figure_number,
            n_signals_pr_subplot=2,
            title="Pose",
            ylabel=ylabel,
            graph_type="step",
            time_axis_format=self.__time_axis_plot_format,
            **kwargs,
        )

    def plot_pose_timeseries_separate(self, figure_number_start=None, **kwargs):
        """
        Plots timeseries for position/attitude navigation data and
        position/attitude motion controller references, each in its own figure.

        Parameters
        ----------
        figure_number_start : int, optional
            Sets the starting figure number. If provided, subsequent figures will
            increment from this starting number. If None, matplotlib will choose.
        **kwargs :
            Key-value arguments that are passed directly to the matplotlib.axes.Axes.plot
            function, controlling how each line is plotted.
        """
        # Define y-axis labels based on whether a local NED origin is used.
        if self.offset_ned_origin is None:
            ylabel = (
                "Latitude [deg]",
                "Longitude [deg]",
                "Depth [m]",
                "Roll [deg]",
                "Pitch [deg]",
                "Yaw [deg]",
            )
        else:
            ylabel = (
                "North [m]",
                "East [m]",
                "Down [m]",
                "Roll [deg]",
                "Pitch [deg]",
                "Yaw [deg]",
            )

        # Load the navigation data and reference signals.
        pose_data = self.load_pose()  # Returns a tuple of 6 Signal objects.
        pose_refs = self.load_pose_refs()  # Returns a tuple of 6 Signal objects.

        # For each degree of freedom, create a separate figure with its own plot.
        for i in range(len(pose_data)):
            # Determine the figure number if a starting number is provided.
            fig_number = (
                figure_number_start + i if figure_number_start is not None else None
            )

            pp.plot_line(
                pose_data[i],
                pose_refs[i],
                figure_number=fig_number,
                title=f"{ylabel[i]}",
                ylabel=ylabel[i],
                graph_type="step",
                time_axis_format=self.__time_axis_plot_format,
                **kwargs,
            )

    def plot_position_scatter(
        self, figure_number=None, skip_non_control=False, equal_axis=False, **kwargs
    ):
        """
        Creates a scatter plot of the horisontal navigation and motion control reference position.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        skip_non_control : bool
            Skips samples where motion control is inactive. The default is False.
        equal_axis : bool
            Axis scaling for x and y axis will be equal. The default is False.
        **kwargs :
            key-value argument which is passed directly to the
            matplotlib.pyplot.scatter function.
        """
        # ? if skip_non_control is True then drifting data will be removed
        # ? if platform is drifting then skip_non_control is False -> ie plots also when the vehcile is just drifting (not dp or transect)
        if skip_non_control:
            pose = self.__filter_control_is_on(*self.load_pose())
        else:
            # loads pose of vehicle -> 6 signals from class Signal (just val. and corresponding time etc -> see Signal class for more info)
            # pose = (lat, long, depth, roll, pitch, yaw) -> its a tuple
            pose = self.load_pose()
        refs = self.load_pose_refs()

        if self.offset_ned_origin is not None:
            xlabel = "East [m]"
            ylabel = "North [m]"
        else:
            xlabel = "longitude [deg]"
            ylabel = "latitude [deg]"

        # xy plot thus only 2 signals are given as inputs
        pp.plot_scatter2d(
            pose[1],  # elem. of pose are all Signals -> x and y in meters
            pose[0],
            refs[1],
            refs[0],
            figure_number=figure_number,
            title="Position",
            xlabel=xlabel,
            ylabel=ylabel,
            plot_as_line=(False, True),
            equal_axis=equal_axis,
            **kwargs,
        )

    def plot_velocity_body_timeseries(self, figure_number=None, **kwargs):
        """
        Plots linear and angular navigation velocity in body frame.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs :
            key-value argument which is passed directly to the
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.
        """
        ylabel = (
            "Surge [m/s]",
            "Sway [m/s]",
            "Heave [m/s]",
            "Roll [rad/s]",
            "Pitch [rad/s]",
            "Yaw [rad/s]",
        )
        pp.plot_line(
            *self.load_velocity_body(),
            figure_number=figure_number,
            n_signals_pr_subplot=1,
            title="Velocity body",
            ylabel=ylabel,
            time_axis_format=self.__time_axis_plot_format,
            **kwargs,
        )

    def plot_position_std_dev(self, figure_number=None, **kwargs):
        """
        Plots the standard deviation for position reported by the navigation system.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs :
            key-value argument which is passed directly to the
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.
        """
        ylabel = ("Latitude [m]", "Longitude [m]", "Down [m]")
        pp.plot_line(
            *self.load_pos_std_dev(),
            figure_number=figure_number,
            n_signals_pr_subplot=1,
            title="Position standard deviation",
            ylabel=ylabel,
            time_axis_format=self.__time_axis_plot_format,
            **kwargs,
        )

    def plot_thrust_forces_and_moments(self, figure_number=None, **kwargs):
        """
        Plots thrust forces and moments from motion controller and applied forces and moments

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs :
            key-value argument which is passed directly to the
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.
        """
        ylabel = (
            "Surge [N]",
            "Sway [N]",
            "Heave [N]",
            "Roll [Nm]",
            "Pitch [Nm]",
            "Yaw [Nm",
        )
        pp.plot_line(
            *pp.interleave_signals(
                self.load_thrust_forces_and_moments_ref(),
                self.load_thrust_forces_and_moments_applied(),
            ),
            figure_number=figure_number,
            n_signals_pr_subplot=2,
            ylabel=ylabel,
            title="Thrust forces and moments",
            time_axis_format=self.__time_axis_plot_format,
            **kwargs,
        )

    def plot_thruster_cmd(self, figure_number=None, **kwargs):
        """
        Plots the thruster commands.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs :
            key-value argument which is passed directly to the
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.
        """
        ylabel = (
            "Front lateral bottom",
            "Front starboard",
            "Front lateral top",
            "Front port",
            "Back lateral bottom",
            "Back starboard",
            "Back lateral top",
            "Back port",
        )
        pp.plot_line(
            *self.load_thruster_cmd(),
            n_signals_pr_subplot=1,
            figure_number=figure_number,
            title="Thruster commands",
            graph_type="step",
            ylabel=ylabel,
            time_axis_format=self.__time_axis_plot_format,
            **kwargs,
        )

    def plot_joint_angles(self, figure_number=None, **kwargs):
        """
        Plots the joint angle position

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs :
            key-value argument which is passed directly to the
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.
        """
        pp.plot_line(
            *self.load_joint_angles(),
            figure_number=figure_number,
            n_signals_pr_subplot=1,
            title="Joint angle position",
            time_axis_format=self.__time_axis_plot_format,
            **kwargs,
        )

    def plot_altitude_timeseries(
        self,
        figure_number=None,
        date_str=None,
        transect_dict=None,
        **kwargs,
    ):
        # Plot altitude
        pp.plot_line(
            self.load_altitude(),
            figure_number=figure_number,
            n_signals_pr_subplot=1,
            title="Altitude",
            ylabel="Altitude [m]",
            time_axis_format=self.__time_axis_plot_format,
            **kwargs,
        )
        ax = plt.gca()

        if date_str and transect_dict and (date_str in transect_dict):
            color_list = ["red", "green", "blue", "orange", "purple", "cyan", "magenta"]
            for i, (start_str, duration_str, label) in enumerate(
                transect_dict[date_str]
            ):
                color = color_list[i % len(color_list)]

                # Parse times
                start_dt = self._parse_date_time(date_str, start_str)
                delta = self._parse_duration(duration_str)
                end_dt = start_dt + delta

                # If needed, convert them to numeric epoch if altitude data is numeric
                # start_val = self._datetime_to_epoch(start_dt)
                # end_val   = self._datetime_to_epoch(end_dt)

                # Vertical shading
                ax.axvspan(start_dt, end_dt, color=color, alpha=0.1)

                # Thin vertical lines
                ax.axvline(start_dt, color=color, linestyle="--")
                ax.axvline(end_dt, color=color, linestyle="--")

                # Single label in the middle
                y_min, y_max = ax.get_ylim()
                mid_dt = start_dt + (delta / 2)
                mid_alt = y_min + 0.05 * (y_max - y_min)
                ax.text(mid_dt, mid_alt, label, ha="center", color=color, fontsize=8)

        # Optionally show
        # plt.show()

    def _parse_date_time(self, date_str, time_str):
        """
        Convert "2024-10-29" and "HH:MM:SS" into a timezone-aware datetime object.
        """
        import datetime

        # Split the date and time components
        year, month, day = map(int, date_str.split("-"))
        hh, mm, ss = map(int, time_str.split(":"))

        # Return a timezone-aware datetime (using UTC)
        return datetime.datetime(
            year, month, day, hh, mm, ss, tzinfo=datetime.timezone.utc
        )

    def _parse_duration(self, duration_str):
        """
        Convert a string like "00:08:15" into a datetime.timedelta.
        """
        import datetime

        hh, mm, ss = map(int, duration_str.split(":"))
        return datetime.timedelta(hours=hh, minutes=mm, seconds=ss)

    def plot_multiple_altitude_timeseries(self, start_figure_number=4, **kwargs):
        """
        Plots timeseries for multiple altitude data paths, each in its own figure.

        Parameters
        ----------
        start_figure_number : int
            Sets the starting plot window figure number. Each altitude path will be plotted in a separate figure.
            The default is 4.
        **kwargs :
            key-value argument which is passed directly to the
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.
        """
        altitude_paths = [
            "/sensors/dvl/D__altitude",
            "/sensors/dvl/N__altitude",  # ok
            "/guidance/hover/altitude_ref",  # wierd
            "/navigation/altitude_timestamp",  # not timeseries
            "/sensors/altimeter/N__altitude",  # ok
            "/sensors/altimeter/altitude",  # noisy
            "/sensors/altimeter/altitude_confidence",  # not timeseries
            "/navigation/N__altitude",  # noisy
        ]

        for i, path in enumerate(altitude_paths):
            figure_number = start_figure_number + i
            try:
                signal = self.__db.get_signal(
                    path, start_time=self.__start_time, end_time=self.__end_time
                )
                pp.plot_line(
                    signal,
                    figure_number=figure_number,
                    title=f"Altitude Timeseries - {path}",
                    ylabel="Altitude [m]",
                    time_axis_format=self.__time_axis_plot_format,
                    **kwargs,
                )
            except KeyError:
                print(f"Path {path} not found in the database.")

    def load_altitude(self) -> pp.Signal:
        """
        Loads altitude data.

        Returns
        -------
        pp.Signal
            Signal object containing altitude data.
        """

        # /sensors/dvl/D__altitude
        # /sensors/dvl/N__altitude
        # /guidance/hover/altitude_ref
        # /navigation/altitude_timestamp
        # /sensors/altimeter/N__altitude
        # /sensors/altimeter/altitude
        # /sensors/altimeter/altitude_confidence
        # /navigation/N__altitude

        if self.__altitude is None:
            self.__altitude = self.__db.get_signal(
                # "/sensors/altitude",
                "/sensors/dvl/D__altitude",
                start_time=self.__start_time,
                end_time=self.__end_time,
            )
        return self.__altitude

    # def plot_atitude_highlights(self, figure_number=None, **kwargs):

    def load_pose_refs(self) -> tuple:
        """
        Loads pose motion control references

        Returns
        -------
        tuple
            A tuple of Signal objects.

        """
        # if not loaded then load it
        if self.__pose_refs is None:
            # use same get_signal as for pose but with different path
            pos_ref = self.__db.get_signal(
                "/guidance/wgs84_ref",
                start_time=self.__start_time,
                end_time=self.__end_time,
            ).create_1d_signals()  # ref. signals are oten stored as multi-dimensional signals -> need to convert them into separate one-dim. signals for each comp.
            # one path contains 3 ref. signals, thus the data(values) are multi-dim. and has to be split using the create_1d_signals() function shown above

            if self.offset_ned_origin is not None:
                ii_lat = np.isclose(pos_ref[0].data, np.zeros(pos_ref[0].size))
                ii_long = np.isclose(pos_ref[1].data, np.zeros(pos_ref[1].size))
                pos_ref[0].data[ii_lat] = self.north_east_origin[
                    0
                ]  # cannot have elements where zone number differs for correct utm calc. Removes zeros and replaces by ned_origin.
                pos_ref[1].data[ii_long] = self.north_east_origin[1]
                pos_ref = pp.convert_signal2ned(
                    *pos_ref,
                    offset_ned_origin=self.offset_ned_origin,
                    name_prefix="ref",
                    convert_depth=True,
                )
            else:
                pos_ref[2].data *= -1

            # same steps as done for position
            attitude_ref = self.__db.get_signal(
                "/guidance/euler_ref",
                start_time=self.__start_time,
                end_time=self.__end_time,
            ).create_1d_signals()
            if len(attitude_ref) != 3:
                assert pos_ref[0].size == 0
                empty = pp.Signal.create_empty(attitude_ref[0].name)
                attitude_ref = (empty, empty, empty)
                pos_ref = (empty, empty, empty)

            self.__pose_refs = pos_ref + attitude_ref
            # removes data where control is off ie when drifting
            self.__pose_refs = self.__filter_control_is_on(*self.__pose_refs)

        return self.__pose_refs

    def load_pose(self) -> tuple:
        """
        Loads pose navigation data.

        Returns
        -------
        tuple with Signal objects.
        """
        # ? if pose is not loaded then load it
        if self.__pose is None:
            # pos is a list of 3 Signal objects -> lat, long (both in degree), depth (meter?)
            pos = list(
                pp.get_linear_pos_signals(
                    self.__db, start_time=self.__start_time, end_time=self.__end_time
                )
            )

            # due to sensor issues, missing data, or timing differences, they might have different lengths -> resample one of them to fix
            if pos[0].size != pos[1].size:
                # use interpolation since its effective for slowly changing signals
                pos[0], pos[1] = pp.resample_to_same_length(
                    pos[0], pos[1], length="highest", method="interpolate"
                )

            # if we have a origin, we can transform the data from global (latitude/longitude in degrees) to local (NED in meters)
            if self.offset_ned_origin is not None:
                # converts the position data from latitude/longitude (global coordinates) to NED (local coordinates in meters).
                pos = pp.convert_signal2ned(
                    *pos, offset_ned_origin=self.offset_ned_origin, name_prefix="pos"
                )

            # loads the attitude -> same as done from position (lat, long, depth)
            attitude = pp.get_attitude_signals(
                self.__db, start_time=self.__start_time, end_time=self.__end_time
            )

            # Remove the first data point for all signals
            for signal in list(pos) + list(attitude):
                signal.data = signal.data[2:]
                signal.axis = signal.axis[2:]

            # Convert attitude to a list before concatenation
            self.__pose = tuple(pos) + tuple(attitude)

        # returns pose which is a tuple of 6 Signal objects -> lat, long, depth, roll, pitch, yaw
        return self.__pose

    def ssa(self, angle):
        """
        Normalize an angle to the range [0, 360] degrees.

        Parameters:
            angle (float or np.array): Input angle(s) in degrees.

        Returns:
            float or np.array: Normalized angle(s).
        """
        return angle % 360

    def unwrap_degrees(self, angle_deg):
        import numpy as np

        # Convert the entire array from degrees to radians.
        angle_rad = np.deg2rad(angle_deg)
        # Unwrap the entire array.
        unwrapped_rad = np.unwrap(angle_rad)
        # Convert back to degrees.
        return np.rad2deg(unwrapped_rad)

    def plot_3d_trajectory(self, figure_number=None, invert_depth=True, **kwargs):
        """
        Plots a 3D trajectory using North, East, and Depth data if offset_ned_origin is set;
        otherwise, uses latitude, longitude, and depth in degrees/meters.

        Parameters
        ----------
        figure_number : int, optional
            Sets the figure number (window ID). If None, matplotlib auto-creates one.
        invert_depth : bool, optional
            If True (default), multiplies depth by -1 so that 'up' is positive in the 3D plot.
            If False, the z-axis will increase downward.
        **kwargs :
            Additional keyword arguments passed directly to matplotlib's plot function
            (e.g. color='red', linestyle='--', etc.).
        """
        # Load the pose signals: (north or lat, east or long, depth, roll, pitch, yaw).
        pose = self.load_pose()

        # Resample signals to the same length
        north, east = pp.resample_to_same_length(
            pose[0], pose[1], length="highest", method="interpolate"
        )
        north, depth = pp.resample_to_same_length(
            north, pose[2], length="highest", method="interpolate"
        )
        east, depth = pp.resample_to_same_length(
            east, depth, length="highest", method="interpolate"
        )

        # Decide how to label and interpret the first two signals
        if self.offset_ned_origin is not None:
            x_data = north.data  # North [m]
            y_data = east.data  # East [m]
            z_data = depth.data  # Depth [m], typically positive downward
            x_label = "North [m]"
            y_label = "East [m]"
            z_label = "Depth [m]"
        else:
            x_data = north.data  # Latitude [deg]
            y_data = east.data  # Longitude [deg]
            z_data = depth.data  # Depth [m]
            x_label = "Latitude [deg]"
            y_label = "Longitude [deg]"
            z_label = "Depth [m]"

        # Optionally invert the depth so that the z-axis points "up" in the 3D plot
        if invert_depth:
            z_data = -z_data
            z_label = "Height [m]"  # or "Altitude [m]" if you prefer

        # Create a new figure (or overwrite the one specified)
        fig = plt.figure(num=figure_number)
        ax = fig.add_subplot(111, projection="3d")

        # Plot the 3D line or scatter, depending on kwargs (e.g. marker='o')
        ax.plot(x_data, y_data, z_data, label="3D Trajectory", **kwargs)

        # Label axes
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        # Optional: set a nice view angle if desired
        # ax.view_init(elev=30, azim=-60)

        ax.legend()
        plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation
    # from matplotlib.widgets import Slider, Button
    # from mpl_toolkits.mplot3d import Axes3D  # ensure 3D projection is available
    # from math import cos, sin, radians

    #  import numpy as np
    #     import matplotlib.pyplot as plt
    #     from matplotlib.widgets import Slider, Button
    #     from mpl_toolkits.mplot3d import Axes3D
    #     from mpl_toolkits.mplot3d.proj3d import proj_transform
    #     from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    #     import datetime
    #     from math import cos, sin, radians

    def plot_live_3d_trajectory(
        self,
        figure_number=None,
        invert_depth=True,
        interval=50,
        fix_vertical=False,
        highlight_intervals=None,
        **kwargs,
    ):
        """
        Animates a live 3D trajectory with orientation arrows representing the vehicle.
        The trajectory is drawn progressively with a text label displaying the current UTC timestamp.

        Three arrows indicate the vehicle's body axes:
        - Red: local X (forward)
        - Green: local Y (sideward)
        - Blue: local Z (vertical)

        If 'fix_vertical' is False (default), all three arrows rotate together.
        If 'highlight_intervals' is provided (a list of (start, end) frame tuples),
        the corresponding sections of the trajectory are highlighted with a thick, semi-transparent line.

        A slider lets you jump to any point in time, and a play/pause button controls the animation.

        Parameters
        ----------
        figure_number : int, optional
            Figure window number (if None, a new figure is created).
        invert_depth : bool, optional
            If True, multiplies depth by -1 so that z points upward.
        interval : int, optional
            Delay in milliseconds between frames.
        fix_vertical : bool, optional
            If True, forces the blue arrow to remain fixed to world vertical;
            if False (default), all three arrows rotate according to roll, pitch, and yaw.
        highlight_intervals : list of tuples, optional
            A list of (start_frame, end_frame) pairs. For each pair, the segment of the trajectory
            between start_frame and end_frame is highlighted.
        **kwargs :
            Additional keyword arguments passed to the trajectory line plot.
        """
        # --- Load and resample data ---
        # Pose signals: (north/lat, east/long, depth, roll, pitch, yaw)
        pose = self.load_pose()
        # Resample north, east, depth so they share the same time axis.
        north, east = pp.resample_to_same_length(
            pose[0], pose[1], length="highest", method="interpolate"
        )
        north, depth = pp.resample_to_same_length(
            north, pose[2], length="highest", method="interpolate"
        )
        east, depth = pp.resample_to_same_length(
            east, depth, length="highest", method="interpolate"
        )
        # Resample orientation signals.
        _, roll = pp.resample_to_same_length(
            north, pose[3], length="highest", method="interpolate"
        )
        _, pitch = pp.resample_to_same_length(
            north, pose[4], length="highest", method="interpolate"
        )
        _, yaw = pp.resample_to_same_length(
            north, pose[5], length="highest", method="interpolate"
        )

        # Choose coordinate interpretation.
        if self.offset_ned_origin is not None:
            x_data = north.data  # North [m]
            y_data = east.data  # East [m]
            z_data = depth.data  # Depth [m]
            x_label = "North [m]"
            y_label = "East [m]"
            z_label = "Depth [m]"
        else:
            x_data = north.data  # Latitude [deg]
            y_data = east.data  # Longitude [deg]
            z_data = depth.data  # Depth [m]
            x_label = "Latitude [deg]"
            y_label = "Longitude [deg]"
            z_label = "Depth [m]"

        # if invert_depth:
        #     z_data = -z_data
        #     z_label = "Depth [m]"

        n_frames = len(x_data)

        # --- Create figure, 3D axis, and widgets ---
        fig = plt.figure(num=figure_number)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        # 1) If user provided an xyz_file, load seafloor data:
        xyz_file = kwargs.pop("xyz_file", None)
        if xyz_file is not None:
            try:
                # Use the same 'invert_depth' argument to keep them consistent
                sea_x, sea_y, sea_z = self.load_seafloor_data(
                    xyz_file, invert_depth=invert_depth
                )

                # 2) Combine the min/max from vehicle + seafloor so you can see both
                all_x_min = min(np.min(x_data), np.min(sea_x))
                all_x_max = max(np.max(x_data), np.max(sea_x))
                all_y_min = min(np.min(y_data), np.min(sea_y))
                all_y_max = max(np.max(y_data), np.max(sea_y))
                all_z_min = min(np.min(z_data), np.min(sea_z))
                all_z_max = max(np.max(z_data), np.max(sea_z))

                ax.set_xlim([all_x_min, all_x_max])
                ax.set_ylim([all_y_min, all_y_max])
                ax.set_zlim([all_z_min, all_z_max])

                # 3) Plot seafloor as scatter:
                ax.scatter(
                    sea_x,
                    sea_y,
                    sea_z,
                    c="brown",
                    marker=".",
                    alpha=0.5,
                    s=1,
                    label="Sea Floor",
                )
                ax.legend()

            except Exception as e:
                print(f"Error loading sea floor data: {e}")
        else:
            # If no seafloor data is provided, set limits from vehicle alone:
            ax.set_xlim([np.min(x_data), np.max(x_data)])
            ax.set_ylim([np.min(y_data), np.max(y_data)])
            ax.set_zlim([np.min(z_data), np.max(z_data)])

        if invert_depth:
            ax.invert_zaxis()  # Invert z-axis if depth is inverted
            ax.invert_yaxis()  # Invert y-axis if depth is inverted

        # ─── INSERT THIS RIGHT HERE ────────────────────────────────────────────
        # for each named transect, draw a thick colored slice and label it
        cmap = plt.cm.get_cmap("tab10", len(highlight_intervals or []))
        for idx, (i0, i1, name) in enumerate(highlight_intervals or []):
            color = cmap(idx)
            ax.plot(
                x_data[i0:i1],
                y_data[i0:i1],
                z_data[i0:i1],
                color=color,
                linewidth=3,
                alpha=0.6,
            )
            if name:
                mid = (i0 + i1) // 2
                ax.text(
                    x_data[mid], y_data[mid], z_data[mid], name, color=color, fontsize=8
                )
        # ────────────────────────────────────────────────────────────────────────
        # Initialize an empty trajectory line.
        (line,) = ax.plot([], [], [], label="3D Trajectory", **kwargs)
        # List to hold orientation arrow quivers.
        arrow_quivers = []
        # List to hold highlight line objects.
        highlight_lines = []

        # Create a text annotation for the UTC timestamp.
        timestamp_text = ax.text2D(
            0.80,
            0.92,
            "",
            transform=ax.transAxes,
            fontsize=10,
            color="black",
            backgroundcolor="w",
        )

        # Create a slider widget.
        slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor="lightgoldenrodyellow")
        slider = Slider(slider_ax, "Time", 0, n_frames - 1, valinit=0, valstep=1)

        # Create a play/pause button.
        button_ax = plt.axes([0.87, 0.02, 0.1, 0.04])
        play_button = Button(button_ax, "Pause")
        anim_running = True
        current_frame = 0

        # --- Update function ---
        def update(frame):
            nonlocal arrow_quivers, current_frame, highlight_lines
            current_frame = frame

            # # Ensure frame index is within bounds
            # if frame >= len(x_data):
            #     frame = len(x_data) - 1

            # Update trajectory line.
            line.set_data(x_data[:frame], y_data[:frame])
            line.set_3d_properties(z_data[:frame])

            # Remove previous orientation arrows.
            for q in arrow_quivers:
                try:
                    q.remove()
                except Exception:
                    pass
            arrow_quivers.clear()

            # Remove previous highlighted segments.
            for hl in highlight_lines:
                try:
                    hl.remove()
                except Exception:
                    pass
            highlight_lines.clear()

            # Get current orientation values (in degrees).
            current_roll = roll.data[frame]
            current_pitch = pitch.data[frame]
            current_yaw = yaw.data[frame]

            # Convert to radians.
            r = radians(current_roll)
            p = radians(current_pitch)
            y_angle = radians(current_yaw)

            # Compute rotation matrices.
            Rx = np.array([[1, 0, 0], [0, cos(r), -sin(r)], [0, sin(r), cos(r)]])
            Ry = np.array([[cos(p), 0, sin(p)], [0, 1, 0], [-sin(p), 0, cos(p)]])
            Rz = np.array(
                [
                    [cos(y_angle), -sin(y_angle), 0],
                    [sin(y_angle), cos(y_angle), 0],
                    [0, 0, 1],
                ]
            )
            R_combined = Rz @ Ry @ Rx

            # Define base arrow length.
            L = 0.1 * (np.max(x_data) - np.min(x_data))
            if L == 0:
                L = 1.0
            scale_factor = 1.5  # adjust as needed
            arrow_length = L * scale_factor

            # Compute body axis vectors.
            arrow_x_vec = R_combined @ np.array([arrow_length, 0, 0])
            arrow_y_vec = R_combined @ np.array([0, arrow_length, 0])
            if fix_vertical:
                arrow_z_vec = np.array([0, 0, arrow_length])
            else:
                arrow_z_vec = R_combined @ np.array([0, 0, arrow_length])

            pos_x = x_data[frame]
            pos_y = y_data[frame]
            pos_z = z_data[frame]

            # Draw orientation arrows.
            qx = ax.quiver(
                pos_x,
                pos_y,
                pos_z,
                arrow_x_vec[0],
                arrow_x_vec[1],
                arrow_x_vec[2],
                color="r",
                length=1,
                normalize=False,
            )
            qy = ax.quiver(
                pos_x,
                pos_y,
                pos_z,
                arrow_y_vec[0],
                arrow_y_vec[1],
                arrow_y_vec[2],
                color="g",
                length=1,
                normalize=False,
            )
            qz = ax.quiver(
                pos_x,
                pos_y,
                pos_z,
                arrow_z_vec[0],
                arrow_z_vec[1],
                arrow_z_vec[2],
                color="b",
                length=1,
                normalize=False,
            )
            arrow_quivers.extend([qx, qy, qz])

            # Highlight trend segments if provided.
            if highlight_intervals is not None:
                for start_idx, end_idx, name in highlight_intervals:
                    if frame >= start_idx:
                        seg_end = min(frame, end_idx)
                        (hl,) = ax.plot(
                            x_data[start_idx : seg_end + 1],
                            y_data[start_idx : seg_end + 1],
                            z_data[start_idx : seg_end + 1],
                            color="yellow",
                            linewidth=3,
                            alpha=0.5,
                        )
                        highlight_lines.append(hl)

            # Update UTC timestamp text.
            t_val = north.axis[frame]
            utc_str = datetime.datetime.utcfromtimestamp(t_val).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            timestamp_text.set_text(utc_str)

            # Update the slider value without triggering its callback.
            slider.eventson = False
            slider.set_val(frame)
            slider.eventson = True

            fig.canvas.draw_idle()
            return (line, *arrow_quivers, timestamp_text, *highlight_lines)

        # --- Slider callback ---
        def slider_update(val):
            frame = int(slider.val)
            update(frame)
            fig.canvas.draw_idle()

        slider.on_changed(slider_update)

        # --- Play/Pause callback ---

        def toggle_animation(event):
            nonlocal anim_running
            if anim_running:
                print("Pausing animation")
                timer.stop()  # Stop the timer
                anim_running = False
                play_button.label.set_text("Play")
            else:
                print("Resuming animation")
                anim_running = True
                play_button.label.set_text("Pause")
                timer.start()  # Restart the timer
            fig.canvas.draw_idle()  # Force a redraw to update the button label

        play_button.on_clicked(toggle_animation)

        # --- Animation via timer ---
        timer = fig.canvas.new_timer(interval=interval)

        def timer_callback():
            nonlocal current_frame
            if anim_running:
                current_frame = (current_frame + 1) % n_frames
                update(current_frame)
            # timer.start()  # restart timer

        timer.add_callback(timer_callback)
        timer.start()

        # plt.sh()

    def load_seafloor_data(self, file_path, invert_depth=False):
        """
        Loads seafloor data from a file that has columns in the order:
        (East, North, Depth).

        We reorder them so that x=North, y=East, z=Depth to match
        the vehicle data. If invert_depth=True, we flip the sign of depth
        so that a positive depth (down) becomes negative (down) to match
        the vehicle's 'Height [m]' axis.
        """
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 3:
            raise ValueError(
                "Sea floor file must have at least 3 columns (East, North, Depth)."
            )

        # File columns: [0]=East, [1]=North, [2]=Depth
        sea_e = data[:, 0]
        sea_n = data[:, 1]
        sea_z = data[:, 2]

        # Reorder to match vehicle's (x=North, y=East, z=Depth)
        sea_x = sea_n
        sea_y = sea_e

        # If you want the seafloor to appear below zero in 'Height [m]', invert the depth:
        if invert_depth:
            sea_z = -sea_z

        return sea_x, sea_y, sea_z  # inside your MotionAnalyzer class

    def plot_live_3d_trajectory_for_date(
        self,
        date_str,
        transect_list,  # now a list of (start_time, duration, name)
        figure_number=None,
        invert_depth=True,
        interval=50,
        fix_vertical=False,
        **kwargs,
    ):
        """
        Plots a live 3D trajectory and highlights specific named transects on a given date.
        """
        # load & resample pose → north, east, depth arrays + their common time axis
        pose = self.load_pose()
        north, east = pp.resample_to_same_length(
            pose[0], pose[1], length="highest", method="interpolate"
        )
        north, depth = pp.resample_to_same_length(
            north, pose[2], length="highest", method="interpolate"
        )
        east, depth = pp.resample_to_same_length(
            east, depth, length="highest", method="interpolate"
        )
        time_axis = north.axis

        # convert each (start, dur, name) → (i0, i1, name)
        named_indices = []
        for start_str, dur_str, name in transect_list:
            t0 = self._parse_date_time(date_str, start_str).timestamp()
            t1 = (
                self._parse_date_time(date_str, start_str)
                + self._parse_duration(dur_str)
            ).timestamp()
            i0 = np.searchsorted(time_axis, t0, side="left")
            i1 = np.searchsorted(time_axis, t1, side="right")
            if i0 >= i1 or i0 >= len(time_axis):
                continue
            named_indices.append((max(i0, 0), min(i1, len(time_axis) - 1), name))

        # hand off to the core plot routine, passing names too
        self.plot_live_3d_trajectory(
            figure_number=figure_number,
            invert_depth=invert_depth,
            interval=interval,
            fix_vertical=fix_vertical,
            highlight_intervals=named_indices,
            **kwargs,
        )

    def load_velocity_body(self) -> tuple:
        """
        Loads navigation linear and angular velocity

        Returns
        -------
        tuple with Signal objects

        """
        if self.__vel_body is None:
            self.__vel_body = self.__db.get_signal(
                "/sensors/sunstone/navigation/velocity",
                start_time=self.__start_time,
                end_time=self.__end_time,
            ).create_1d_signals()
            self.__vel_body += self.__db.get_signal(
                "/sensors/sunstone/navigation/angular_vel",
                start_time=self.__start_time,
                end_time=self.__end_time,
            ).create_1d_signals()
        return self.__vel_body

    def load_pos_std_dev(self) -> tuple:
        """
        Loads position standard deviation reported by the navigation system.

        Returns
        -------
        tuple with Signal objects

        """
        if self.__pos_std_dev is None:
            self.__pos_std_dev = self.__db.get_multiple_signals(
                "/sensors/sunstone/position/std_dev/lat_deg",
                "/sensors/sunstone/position/std_dev/lon_deg",
                "/sensors/sunstone/position/std_dev/depth",
                start_time=self.__start_time,
                end_time=self.__end_time,
            )
        return self.__pos_std_dev

    def load_control_modes(self) -> tuple:
        """
        Loads control modes for 6 DOF motion controller. The modes indicates if the control is on or off.

        Returns
        -------
        tuple with Signal objects

        """
        if self.__control_modes is None:
            self.__control_modes = self.__db.get_multiple_signals(
                "/guidance/auto_modes/x",
                "/guidance/auto_modes/y",
                "/guidance/auto_modes/z",
                "/guidance/auto_modes/roll",
                "/guidance/auto_modes/pitch",
                "/guidance/auto_modes/yaw",
                start_time=self.__start_time,
                end_time=self.__end_time,
            )
        return self.__control_modes

    def load_thruster_cmd(self) -> tuple:
        """
        Loads thruster normalized (-100 - 100) commands generated from thruster allocation

        Returns:
            tuple: tuple with Signal objects
        """

        if self.__thruster_cmd is None:
            all_paths = self.__db.search_for_str_in_paths("/thrusters")
            cmd_paths = ()
            for p in all_paths:
                if fnmatch.fnmatch(p, "/thrusters/0x*/ref"):
                    cmd_paths += (p,)
            self.__thruster_cmd = self.__db.get_multiple_signals(
                *cmd_paths, start_time=self.__start_time, end_time=self.__end_time
            )
        return self.__thruster_cmd

    def load_thrust_forces_and_moments_ref(self) -> tuple:
        """
        Loads thruster forces and moments calculated by the motion controller

        Returns:
            tuple: tuple with Signal objects
        """
        if self.__thruster_forces_and_moments_ref is None:
            self.__thruster_forces_and_moments_ref = self.__db.get_signal(
                "/thruster_alloc/V__forceRef",
                start_time=self.__start_time,
                end_time=self.__end_time,
            ).create_1d_signals()
            self.__thruster_forces_and_moments_ref += self.__db.get_signal(
                "/thruster_alloc/V__torqueRef",
                start_time=self.__start_time,
                end_time=self.__end_time,
            ).create_1d_signals()
        return self.__thruster_forces_and_moments_ref

    def load_thrust_forces_and_moments_applied(self) -> tuple:
        """
        Loads thruster forces and moments applied to thrusters (actual commanded thrust)

        Returns:
            tuple: tuple with Signal objects
        """
        if self.__thruster_forces_and_moments_applied is None:
            self.__thruster_forces_and_moments_applied = self.__db.get_signal(
                "/thruster_alloc/V__forceUsed",
                start_time=self.__start_time,
                end_time=self.__end_time,
            ).create_1d_signals()
            self.__thruster_forces_and_moments_applied += self.__db.get_signal(
                "/thruster_alloc/V__torqueUsed",
                start_time=self.__start_time,
                end_time=self.__end_time,
            ).create_1d_signals()
        return self.__thruster_forces_and_moments_applied

    def load_joint_angles(self) -> tuple:
        """
        Loads joint angle position

        Returns:
            tuple: tuple with Signal objects
        """
        if self.__joint_angles is None:
            angle_paths = self.__db.search_for_str_in_paths("/joints/angle_measured/")
            self.__joint_angles = self.__db.get_multiple_signals(
                *angle_paths, start_time=self.__start_time, end_time=self.__end_time
            )
        return self.__joint_angles

    def create_nav_file(self, file_path):
        """
        Creates a .nav file with the specified structure.

        Parameters
        ----------
        file_path : str
            The path where the .nav file will be saved.
        """
        pose = self.load_pose()
        altitude = self.load_altitude()

        _, _, zoneNumber, zoneLetter = lat_long_2_utm(pose[0].data[0], pose[1].data[0])
        print(f"UTM: zone number {zoneNumber}, zone letter {zoneLetter}")

        with open(file_path, "w") as file:
            for i in range(1, len(pose[0].axis)):
                date_time = datetime.datetime.utcfromtimestamp(pose[0].axis[i])
                date_str = date_time.strftime("%d.%m.%Y")
                time_str = date_time.strftime("%H:%M:%S.%f")[:-2]

                lat = pose[0].data[i]
                lon = pose[1].data[i]
                north, east, _, _ = lat_long_2_utm(lat, lon)

                # northing = pose[0].data[i]
                # easting = pose[1].data[i]

                pitch = pose[4].data[i]
                roll = pose[3].data[i]
                heading = pose[5].data[i]
                depth = pose[2].data[i]
                alt = altitude.data[i] if i < len(altitude.data) else float("nan")

                if np.isnan(alt):
                    alt = 1
                    print(
                        f"Warning: Altitude data missing at index {i}, substituting with 1"
                    )

                file.write(
                    f"{date_str}\t{time_str}\t{north:.3f}\t{east:.3f}\t{pitch:.3f}\t{roll:.3f}\t{heading:.3f}\t{depth:.2f}\t{alt:.2f}\n"
                )

    def plot_altitude_and_depth(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Altitude", color="tab:blue")
        altitude_data = self.load_altitude()
        ax1.plot(altitude_data.axis, altitude_data.data, color="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Depth", color="tab:red")
        pose_data = self.load_pose()
        depth_data = pose_data[2]  # Assuming depth is the third element in the tuple
        ax2.plot(
            depth_data.axis, -depth_data.data, color="tab:red"
        )  # Reverse the depth data

        fig.tight_layout()
        # plt.show()

    def plot_yaw_angle(self, figure_number=None, **kwargs):
        """
        Plots actual vs. reference yaw over time (step plot), properly aligned.
        """
        # 1) grab the two pp.Signal objects
        actual_sig = self.load_pose()[5]  # navigation yaw
        ref_sig = self.load_pose_refs()[5]  # guidance yaw

        # 2) unwrap both so +179→–179 discontinuities go away
        actual_sig.data = self.unwrap_degrees(actual_sig.data)

        # 4) call the same plotting utility as plot_pose_timeseries_separate
        pp.plot_line(
            actual_sig,
            ref_sig,
            figure_number=figure_number,
            title="Yaw",
            ylabel="Yaw [deg]",
            graph_type="step",
            time_axis_format=self.__time_axis_plot_format,
            **kwargs,
        )

    # def plot_yaw_angle(self, figure_number=None, **plot_kwargs):
    #     """
    #     Plots actual vs. reference yaw over time (step plot), properly aligned,
    #     and marks any big jumps (>60°) in the reference signal.
    #     """
    #     from datetime import datetime

    #     # 1) grab the two pp.Signal objects
    #     # actual_sig = self.load_pose()[5]  # navigation yaw
    #     reference_sig = self.load_pose_refs()[5]  # guidance yaw
    #     print("shape" + str(reference_sig.shape))
    #     # 2) unwrap only the actual signal
    #     # actual = self.unwrap_degrees(actual_sig.data)
    #     reference = reference_sig.data

    #     # 3) build separate UTC time axes
    #     # times_act = [datetime.utcfromtimestamp(t) for t in actual_sig.axis]
    #     times_ref = [datetime.utcfromtimestamp(t) for t in reference_sig.axis]

    #     # 4) create figure/axis
    #     fig = plt.figure(figure_number) if figure_number is not None else plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)

    #     # 5) plot both
    #     # ax.step(times_act, actual, label="Actual yaw", where="post", **plot_kwargs)
    #     ax.step(
    #         times_ref, reference, label="Reference yaw", where="post", **plot_kwargs
    #     )

    #     # 6) mark big jumps in reference (>60°)
    #     diffs = []
    #     for i in range(len(reference) - 1):
    #         print(reference[i])
    #         diffs.append(reference[i + 1] - reference[i])
    #     print("shape diffs" + str(len(diffs)))
    #     for i in range(len(diffs) - 1):
    #         if abs(diffs[i]) > 60:
    #             print("found at" + str(i))
    #             t_jump = times_ref[i]
    #             y_jump = reference[i]
    #             ax.plot(t_jump, y_jump, "o", markersize=6, color="red")
    #             t_jump = times_ref[i + 1]
    #             y_jump = reference[i + 1]
    #             ax.plot(t_jump, y_jump, "o", markersize=3, color="blue")

    #     # 7) format UTC datetime x-axis
    #     ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    #     ax.xaxis.get_offset_text().set_visible(False)
    #     fig.autofmt_xdate()

    #     # 8) finish
    #     ax.set_xlabel("UTC time")
    #     ax.set_ylabel("Yaw [deg]")
    #     ax.set_title("Yaw")
    #     ax.grid(True)
    #     ax.legend()
    #     plt.show()

    def __reload(self):
        # Lazy loading of signals later
        self.__pose_refs = None
        self.__pose = None
        self.__vel_body = None
        self.__pos_std_dev = None
        self.__control_modes = None
        self.__control_is_on_time_ranges = None
        self.__thruster_cmd = None
        self.__thruster_forces_and_moments_ref = None
        self.__thruster_forces_and_moments_applied = None
        self.__joint_angles = None
        self.__altitude = None

    def __load_control_is_on_time_ranges(self):
        if self.__control_is_on_time_ranges is None:
            control_mode = self.load_control_modes()
            self.__control_is_on_time_ranges = ()

            for i in range(6):
                is_active_axis = control_mode[i].axis
                is_active_data = control_mode[i].data > 0.5
                axis_ranges = []

                for j in range(is_active_data.size):
                    next_axis = is_active_axis[min(j + 1, is_active_axis.size - 1)]
                    assert is_active_data[j] != next_axis, "Control mode is not valid"

                    if not is_active_data[j]:
                        axis_ranges.append(
                            (
                                is_active_axis[j] - 0.1,
                                (
                                    next_axis - 0.1
                                    if j < is_active_axis.size - 1
                                    else float("inf")
                                ),
                            )
                        )

                self.__control_is_on_time_ranges += (axis_ranges,)

        return self.__control_is_on_time_ranges

    def __filter_control_is_on(self, *signals: pp.Signal):
        assert len(signals) == 6
        time_ranges = self.__load_control_is_on_time_ranges()

        removed_signals = ()
        for i, sig in enumerate(signals):

            data = sig.data.copy()
            for range in time_ranges[i]:
                ii_remove = np.logical_and(sig.axis >= range[0], sig.axis < range[1])
                data[ii_remove] = np.nan

            removed_signals += (pp.Signal.create(sig.name, sig.axis, data),)

        return removed_signals

    def __set_time_axis_plot_format(self):
        """
        Determines how to display time on a graph based on the time zone
        of start_time and end_time.

        Process:
        1. If both start_time and end_time exist, ensure they use the same time zone.
        2. Choose a reference time (whichever is available).
        3. Set the format based on the time zone:
        - If UTC → Display as "datetime_utc".
        - If no time zone → Assume local time, display as "datetime_local".
        - If an unknown time zone → Raise an error.
        4. If no time is provided, default to UTC.

        Example Scenarios:
        -----------------
        1. start_time = 12:00 UTC, end_time = 14:00 UTC → "datetime_utc"
        2. start_time = 12:00 (no time zone), end_time = 14:00 (no time zone) → "datetime_local"
        3. start_time = None, end_time = None → Defaults to "datetime_utc"
        4. start_time and end_time have different time zones → ERROR
        """
        if self.__start_time is not None and self.__end_time is not None:
            assert self.__start_time.tzinfo == self.__end_time.tzinfo

        time = None
        if self.__start_time is not None:
            time = self.__start_time
        if self.__end_time is not None:
            time = self.__end_time

        if time is not None:
            if time.tzinfo == datetime.timezone.utc:
                self.__time_axis_plot_format = "datetime_utc"
            elif time.tzinfo is None:
                self.__time_axis_plot_format = "datetime_local"
            else:
                raise Exception("Not able to set time axis plot format")
        else:
            self.__time_axis_plot_format = "datetime_utc"  # defaults to utc plot

    """
    this is the old one that also gave ecef, unsure if it works, so i will just follow same as Håvard
    def create_csv_file(self, csv_file_path: str):
        # loads pose (lat, lon, depth, roll, pitch, yaw)
        pose = self.load_pose()

        # Define labels before usage
        labels = [
            "latitude [deg]",
            "longitude [deg]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
        ]

        # Extract each signal's data and determine the shortest common length.
        arrays = [signal.data for signal in pose]
        min_len = min(len(arr) for arr in arrays)

        # Build a data dictionary with truncated arrays and add the shared timestamp.
        data = {label: signal.data[:min_len] for label, signal in zip(labels, pose)}
        data["timestamp [unix epoch s]"] = pose[0].axis[:min_len]

        # # Convert data to numpy arrays for transformation.
        # lat = np.array(data["latitude [deg]"])
        # lon = np.array(data["longitude [deg]"])
        # depth = np.array(data["depth [m]"])

        # # Calculate height above ellipsoid.
        # geoid_height = 38.84  # from geoid calculator for Mjøsa
        # height = geoid_height - depth

        # # Create a transformer from WGS84 (lat, lon, height) to ECEF (x, y, z)
        # transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
        # x, y, z = transformer.transform(lon, lat, height)

        # # Add the ECEF coordinates into the data dictionary.
        # data["position_ecef_x [m]"] = x
        # data["position_ecef_y [m]"] = y
        # data["position_ecef_z [m]"] = z

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Reorder so timestamp is first, then save to CSV
        cols = ["timestamp [unix epoch s]"] + [
            c for c in df.columns if c != "timestamp [s]"
        ]
        df.to_csv(csv_file_path, columns=cols, index=False)
    """

    def create_csv_file(self, csv_file_path: str):
        # Load pose data
        pose = self.load_pose()
        labels = [
            "latitude [deg]",
            "longitude [deg]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
        ]
        arrays = [sig.data for sig in pose]
        min_len = min(len(a) for a in arrays)
        main_timestamps = np.array(pose[0].axis[:min_len])

        # Create data dict and unwrap yaw before saving
        data = {}
        for lbl, sig in zip(labels, pose):
            values = sig.data[:min_len]
            if lbl == "yaw [deg]":
                values = self.unwrap_degrees(values)
            data[lbl] = values

        data["timestamp [unix epoch s]"] = main_timestamps

        # Load altitude and interpolate
        alt_sig = self.load_altitude()
        alt_ts = np.array(alt_sig.axis)
        alt_data = np.array(alt_sig.data)
        sort_idx = np.argsort(alt_ts)
        alt_ts = alt_ts[sort_idx]
        alt_data = alt_data[sort_idx]

        interp = interp1d(
            alt_ts,
            alt_data,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        data["altitude [m]"] = interp(main_timestamps)

        # Export to CSV
        df = pd.DataFrame(data)
        cols = ["timestamp [unix epoch s]"] + [
            c for c in df.columns if c != "timestamp [unix epoch s]"
        ]
        df.to_csv(csv_file_path, columns=cols, index=False)

    def create_aligned_csv_file(self, csv_file_path: str):
        """
        Create CSV with properly time-aligned data.
        Uses interpolation to align all signals to a common time grid.
        """
        from scipy.interpolate import interp1d
        import pandas as pd
        import numpy as np

        print("Creating time-aligned CSV...")

        # Load pose data
        pose = self.load_pose()
        labels = [
            "latitude [deg]",
            "longitude [deg]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
        ]

        # Find the common time range (intersection of all signals)
        time_ranges = []
        for sig in pose:
            time_ranges.append((sig.axis.min(), sig.axis.max()))

        # Use the intersection of all time ranges
        common_start = max(t[0] for t in time_ranges)
        common_end = min(t[1] for t in time_ranges)
        print(f"Common time range: {common_start:.1f} to {common_end:.1f}")

        # Create a common time grid using the highest sample rate
        all_timestamps = []
        for sig in pose:
            mask = (sig.axis >= common_start) & (sig.axis <= common_end)
            all_timestamps.extend(sig.axis[mask])

        # Use unique sorted timestamps as the reference grid
        reference_times = np.unique(np.array(all_timestamps))
        print(f"Reference time grid: {len(reference_times)} points")

        # Interpolate all pose signals to the reference time grid
        data = {"timestamp [unix epoch s]": reference_times}

        for label, sig in zip(labels, pose):
            # Sort the signal data by time (just in case)
            sort_idx = np.argsort(sig.axis)
            sorted_times = sig.axis[sort_idx]
            sorted_data = sig.data[sort_idx]

            # Unwrap yaw angles if needed
            if label == "yaw [deg]":
                sorted_data = self.unwrap_degrees(sorted_data)

            # Interpolate to reference grid
            interp_func = interp1d(
                sorted_times,
                sorted_data,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            data[label] = interp_func(reference_times)

            # Count NaN values
            nan_count = np.isnan(data[label]).sum()
            if nan_count > 0:
                print(
                    f"  ⚠️  {label}: {nan_count} NaN values ({100*nan_count/len(reference_times):.1f}%)"
                )

        # Add altitude data
        alt_sig = self.load_altitude()
        alt_sort_idx = np.argsort(alt_sig.axis)
        alt_interp = interp1d(
            alt_sig.axis[alt_sort_idx],
            alt_sig.data[alt_sort_idx],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        data["altitude [m]"] = alt_interp(reference_times)

        alt_nan_count = np.isnan(data["altitude [m]"]).sum()
        if alt_nan_count > 0:
            print(
                f"  ⚠️  altitude [m]: {alt_nan_count} NaN values ({100*alt_nan_count/len(reference_times):.1f}%)"
            )

        # Create DataFrame and save
        df = pd.DataFrame(data)

        # Remove rows with any NaN values to ensure clean data
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)

        print(f"Removed {initial_len - final_len} rows with NaN values")
        print(f"Final dataset: {final_len} rows")

        # Reorder columns
        cols = ["timestamp [unix epoch s]"] + [
            c for c in df.columns if c != "timestamp [unix epoch s]"
        ]
        df.to_csv(csv_file_path, columns=cols, index=False)
        print(f"✅ Saved aligned CSV to: {csv_file_path}")

        return df

    def diagnose_timestamp_alignment(self):
        """
        Diagnostic function to analyze timestamp alignment issues in pose data.
        """
        print("=== TIMESTAMP ALIGNMENT DIAGNOSTIC ===")

        # Load pose data
        pose = self.load_pose()
        labels = [
            "latitude [deg]",
            "longitude [deg]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
        ]

        print(f"Loaded {len(pose)} pose signals:")
        for i, (label, sig) in enumerate(zip(labels, pose)):
            print(
                f"  [{i}] {label}: {sig.data.shape} data points, {sig.axis.shape} timestamps"
            )
            print(f"      Time range: {sig.axis.min():.1f} to {sig.axis.max():.1f}")
            if i > 0:
                time_diff = sig.axis.min() - pose[0].axis.min()
                print(f"      Time offset vs signal[0]: {time_diff:.3f} seconds")
            print()

        # Check how original create_csv_file truncates
        arrays = [sig.data for sig in pose]
        min_len = min(len(a) for a in arrays)
        print(f"Minimum length across all pose signals: {min_len}")
        print(f"Original create_csv_file uses timestamps from signal[0]: {labels[0]}")
        main_timestamps = pose[0].axis[:min_len]
        print(
            f"Selected timestamp range: {main_timestamps.min():.1f} to {main_timestamps.max():.1f}"
        )

        # Check altitude data
        alt_sig = self.load_altitude()
        print(
            f"\nAltitude signal: {alt_sig.data.shape} points, {alt_sig.axis.shape} timestamps"
        )
        print(
            f"Altitude time range: {alt_sig.axis.min():.1f} to {alt_sig.axis.max():.1f}"
        )

        # Check timestamp overlap
        alt_start, alt_end = alt_sig.axis.min(), alt_sig.axis.max()
        main_start, main_end = main_timestamps.min(), main_timestamps.max()
        print(f"\nTimestamp overlap analysis:")
        print(f"  Main pose:  {main_start:.1f} to {main_end:.1f}")
        print(f"  Altitude:   {alt_start:.1f} to {alt_end:.1f}")
        print(
            f"  Overlap:    {max(main_start, alt_start):.1f} to {min(main_end, alt_end):.1f}"
        )

        if alt_start > main_start or alt_end < main_end:
            print(
                "  ⚠️  WARNING: Altitude data does not fully cover pose data timespan!"
            )
            print("  ⚠️  This will result in NaN values in the CSV altitude column!")

        return {
            "pose_signals": pose,
            "labels": labels,
            "min_length": min_len,
            "main_timestamps": main_timestamps,
            "altitude_signal": alt_sig,
        }

    def create_altitude_csv(self, csv_file_path: str):
        import pandas as pd

        # Load altitude signal
        altitude_signal = self.load_altitude()
        # Build a data dictionary with the altitude data and its own timestamps.
        data = {
            "timestamp [s]": altitude_signal.axis,  # assuming altitude_signal.axis holds its timestamps
            "altitude [m]": altitude_signal.data,
        }
        pd.DataFrame(data).to_csv(csv_file_path, index=False)

    # def test_dbl_data(self):
    #     """
    #     Plot all DVL signals vs. time, plus a position scatter:
    #      • Position (first two dims) as X vs Y scatter
    #      • D__altitude & N__altitude vs time
    #      • orientation (roll, pitch, yaw) vs time
    #      • seabed_depth vs time
    #      • velocity components D__x, D__y, D__z vs time
    #     """
    #     db = self.__db
    #     t0, t1 = self.__start_time, self.__end_time

    #     # ── Position scatter (first two components of /sensors/dvl/position) ──
    #     pos_vec = db.get_signal(
    #         "/sensors/dvl/position",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     pos_signals = pos_vec.create_1d_signals()
    #     x_sig, y_sig = pos_signals[0], pos_signals[1]
    #     plt.figure()
    #     plt.scatter(x_sig.data, y_sig.data, s=10)
    #     plt.title("DVL Position (X vs Y)")
    #     plt.xlabel(f"{x_sig.name}")
    #     plt.ylabel(f"{y_sig.name}")
    #     plt.axis("equal")

    #     # ── D__altitude & N__altitude vs time ──
    #     d_alt = db.get_signal(
    #         "/sensors/dvl/D__altitude",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     n_alt = db.get_signal(
    #         "/sensors/dvl/N__altitude",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     plt.figure()
    #     plt.plot(d_alt.axis, d_alt.data, label="D__altitude")
    #     plt.plot(n_alt.axis, n_alt.data, label="N__altitude")
    #     plt.title("DVL Altitude vs Time")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Altitude [m]")
    #     plt.legend()

    #     # ── Orientation vs time ──
    #     ori_vec = db.get_signal(
    #         "/sensors/dvl/orientation",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     ori_signals = ori_vec.create_1d_signals()  # roll, pitch, yaw
    #     plt.figure()
    #     for sig, name in zip(ori_signals, ("roll", "pitch", "yaw")):
    #         plt.plot(sig.axis, sig.data, label=name)
    #     plt.title("DVL Orientation vs Time")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Orientation [°]")
    #     plt.legend()

    #     # ── Seabed depth vs time ──
    #     seabed = db.get_signal(
    #         "/sensors/dvl/seabed_depth",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     plt.figure()
    #     plt.plot(seabed.axis, seabed.data)
    #     plt.title("DVL Seabed Depth vs Time")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Depth [m]")

    #     # ── Velocity components vs time ──
    #     vx = db.get_signal(
    #         "/sensors/dvl/velocity/D__x",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     vy = db.get_signal(
    #         "/sensors/dvl/velocity/D__y",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     vz = db.get_signal(
    #         "/sensors/dvl/velocity/D__z",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     plt.figure()
    #     plt.plot(vx.axis, vx.data, label="D__x")
    #     plt.plot(vy.axis, vy.data, label="D__y")
    #     plt.plot(vz.axis, vz.data, label="D__z")
    #     plt.title("DVL Velocity vs Time")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Velocity [m/s]")
    #     plt.legend()

    #     plt.show()

    def test_dbl_data(self):
        """
        Plot all DVL signals vs. time, plus a position scatter:
         • Position (first two dims) as X vs Y scatter
         • D__altitude & N__altitude vs time
         • orientation (roll, pitch, yaw) vs time
         • seabed_depth vs time
         • velocity components D__x, D__y, D__z vs time
        """
        db = self.__db
        t0, t1 = self.__start_time, self.__end_time

        # ── Position scatter (first two components of /sensors/dvl/position) ──

        # ── D__altitude & N__altitude vs time ──
        d_alt = db.get_signal(
            "/sensors/dvl/D__altitude",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        n_alt = db.get_signal(
            "/sensors/dvl/N__altitude",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        plt.figure()
        plt.plot(d_alt.axis, d_alt.data, label="D__altitude")
        plt.plot(n_alt.axis, n_alt.data, label="N__altitude")
        plt.title("DVL Altitude vs Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Altitude [m]")
        plt.legend()

        # ── Orientation vs time ──
        ori_vec = db.get_signal(
            "/sensors/dvl/orientation",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        ori_signals = ori_vec.create_1d_signals()  # roll, pitch, yaw
        plt.figure()
        for sig, name in zip(ori_signals, ("roll", "pitch", "yaw")):
            plt.plot(sig.axis, sig.data, label=name)
        plt.title("DVL Orientation vs Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Orientation [°]")
        plt.legend()

        # ── Seabed depth vs time ──
        seabed = db.get_signal(
            "/sensors/dvl/seabed_depth",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        plt.figure()
        plt.plot(seabed.axis, seabed.data)
        plt.title("DVL Seabed Depth vs Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Depth [m]")

        # ── Velocity components vs time ──
        vx = db.get_signal(
            "/sensors/dvl/velocity/D__x",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        vy = db.get_signal(
            "/sensors/dvl/velocity/D__y",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        vz = db.get_signal(
            "/sensors/dvl/velocity/D__z",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        plt.figure(figsize=(12, 8))
        plt.plot(vx.axis, vx.data, label="D__x")
        plt.plot(vy.axis, vy.data, label="D__y")
        plt.plot(vz.axis, vz.data, label="D__z")
        plt.title("DVL Velocity vs Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.legend()

        plt.show()

    def plot_dvl_trajectory_from_velocity(self):
        import numpy as np
        import matplotlib.pyplot as plt

        db = self.__db
        t0, t1 = self.__start_time, self.__end_time

        # 1. load raw velocities + valid flags
        vx_sig = db.get_signal(
            "/sensors/dvl/velocity/D__x",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        vy_sig = db.get_signal(
            "/sensors/dvl/velocity/D__y",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        xv_sig = db.get_signal(
            "/sensors/dvl/velocity/x_valid",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )
        yv_sig = db.get_signal(
            "/sensors/dvl/velocity/y_valid",
            start_time=t0,
            end_time=t1,
            reset_time_stamp_to_zero=False,
        )

        # raw axes + data
        ax_vx, vx = vx_sig.axis, vx_sig.data
        ax_vy, vy = vy_sig.axis, vy_sig.data
        ax_xv, xv = xv_sig.axis, xv_sig.data.astype(float)
        ax_yv, yv = yv_sig.axis, yv_sig.data.astype(float)

        # 2. resample vy onto ax_vx
        vy_on_vx = np.interp(ax_vx, ax_vy, vy, left=0, right=0)

        # 3. resample valid flags onto ax_vx
        xv_on_vx = np.interp(ax_vx, ax_xv, xv, left=0, right=0) > 0.5
        yv_on_vx = np.interp(ax_vx, ax_yv, yv, left=0, right=0) > 0.5
        valid = xv_on_vx & yv_on_vx

        # 4. integrate
        dt = np.diff(ax_vx, prepend=ax_vx[0])
        disp_x = np.cumsum(vx * dt)
        disp_y = np.cumsum(vy_on_vx * dt)

        # 5. scatter only the valid points
        plt.figure()
        plt.scatter(disp_x[valid], disp_y[valid], s=3)
        plt.title("DVL Trajectory (integrated D__x/D__y)")
        plt.xlabel("Cumulative X [m]")
        plt.ylabel("Cumulative Y [m]")
        plt.axis("equal")
        plt.show()

    def debug_print_dvl_position(self):
        """
        Load /sensors/dvl/position and print raw and per-component data.
        """
        # 1) grab the raw vector signal
        vec = self.__db.get_signal(
            "/sensors/dvl/position",
            start_time=self.__start_time,
            end_time=self.__end_time,
            reset_time_stamp_to_zero=False,
        )
        print("Raw vector .data shape:", vec.data.shape)
        print("Raw vector .data:\n", vec.data)
        print("Raw vector .axis:\n", vec.axis)

        # 2) split into 1D components (x,y,z,…)
        comps = vec.create_1d_signals()
        for i, sig in enumerate(comps):
            print(f"\nComponent #{i}: name={sig.name!r}")
            print("  axis:", sig.axis)
            print("  data:", sig.data)

    # inside your MotionAnalyzer class

    # def _mark_spikes(self, north, east, max_jump):
    #     """
    #     Return a boolean mask where True = that index is a 2D spike.
    #     """
    #     import numpy as np

    #     mask = np.zeros(len(north), dtype=bool)
    #     for i in range(1, len(north)):
    #         if np.hypot(north[i] - north[i - 1], east[i] - east[i - 1]) > max_jump:
    #             mask[i] = True

    #     return mask

    # def _interp_gaps(self, arr, mask):
    #     """
    #     Linearly interpolate over any True regions in mask.
    #     """
    #     out = arr.copy()
    #     i = 0
    #     n = len(arr)
    #     while i < n:
    #         if mask[i]:
    #             start = i - 1
    #             # find end of gap
    #             j = i + 1
    #             while j < n and mask[j]:
    #                 j += 1
    #             end = j
    #             # if we ran off the end, just backfill with the last good
    #             if end >= n:
    #                 out[start + 1 :] = out[start]
    #                 break
    #             # interpolate between out[start] and out[end]
    #             for k in range(start + 1, end):
    #                 t = (k - start) / (end - start)
    #                 out[k] = out[start] * (1 - t) + out[end] * t
    #             i = end
    #         else:
    #             i += 1
    #     return out

    # def correct_navigation_data(self, max_jump=0.5, alpha=0.9):
    #     """
    #     1) Mark & interpolate any >max_jump spikes (fills them back onto the trend line)
    #     2) Small EMA‐smooth to polish corners (alpha near 1 preserves shape)
    #     3) Overwrite pose[0].data (north) & pose[1].data (east)
    #     """
    #     # load & copy
    #     pose = list(self.load_pose())
    #     north = pose[0].data.copy()
    #     east = pose[1].data.copy()

    #     # 1) repair spikes
    #     spikes = self._mark_spikes(north, east, max_jump)
    #     north_i = self._interp_gaps(north, spikes)
    #     east_i = self._interp_gaps(east, spikes)

    #     # 2) light smoothing
    #     north_s = north_i.copy()
    #     east_s = east_i.copy()
    #     for idx in range(1, len(north_s)):
    #         north_s[idx] = alpha * north_s[idx - 1] + (1 - alpha) * north_i[idx]
    #         east_s[idx] = alpha * east_s[idx - 1] + (1 - alpha) * east_i[idx]

    #     # 3) overwrite
    #     pose[0].data = north_s
    #     pose[1].data = east_s
    #     self.__pose = tuple(pose)
    # def correct_navigation_data(self):
    #     pose = list(self.load_pose())
    #     north = pose[0].data.copy()
    #     east = pose[1].data.copy()
    #     yaw = pose[5].data.copy()

    #     ref_pose = list(self.load_pose_refs())
    #     yaw_ref = ref_pose[5].data.copy()

    #     # Detect corner events where yaw exceeds 70 degrees
    #     for i, val in enumerate(yaw_ref.data):
    #         if val > 70:
    #             print("Corner detected at time", yaw_ref.axis[i])

    def correct_navigation_data(self, corner_thr: float = 30.0):

        # 1) load the reference yaw signal (6th element of load_pose_refs)
        ref_pose = list(self.load_pose_refs())
        yaw_ref_sig = ref_pose[5]  # pp.Signal
        yaw_ref = yaw_ref_sig.data  # numpy.ndarray
        time_ref = yaw_ref_sig.axis  # numpy timestamps

        # 2) unwrap to avoid 360° discontinuities, then diff
        diff_yaw = np.diff(yaw_ref)
        for i in range(len(diff_yaw)):
            if abs(diff_yaw[i]) > 60:
                t_utc = datetime.datetime.utcfromtimestamp(time_ref[i + 1])
                ts = t_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                print(
                    f"Corner detected at time {ts}: yaw changed by {diff_yaw[i]} degrees"
                )

    # def plot_dvl_velocity(self):
    #     """
    #     Plot DVL velocity components vs. time:
    #     • D__x, D__y, D__z vs time
    #     """
    #     db = self.__db
    #     t0, t1 = self.__start_time, self.__end_time

    #     # Retrieve velocity components
    #     vx = db.get_signal(
    #         "/sensors/dvl/velocity/D__x",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     vy = db.get_signal(
    #         "/sensors/dvl/velocity/D__y",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )
    #     vz = db.get_signal(
    #         "/sensors/dvl/velocity/D__z",
    #         start_time=t0,
    #         end_time=t1,
    #         reset_time_stamp_to_zero=False,
    #     )

    #     # Plot velocity vs time

    #     plt.figure(figsize=(10, 6))
    #     plt.plot(vx.axis, vx.data, label="D__x")
    #     plt.plot(vy.axis, vy.data, label="D__y")
    #     plt.plot(vz.axis, vz.data, label="D__z")
    #     plt.title("DVL Velocity vs Time")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Velocity [m/s]")
    #     plt.legend()
    #     plt.show()

    def plot_dvl_velocity(self, highlight_intervals=None):
        """
        Plot DVL velocity components vs. UTC time, with optional highlighted intervals.
        • D__x, D__y, D__z vs time
        • highlight_intervals: list of (start_str, end_str) in "HH:MM:SS" UTC
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime, timezone

        db = self.__db
        # t0 and t1 are datetime.datetime already
        t0_dt, t1_dt = self.__start_time, self.__end_time

        # Fetch using numeric timestamps
        vx = db.get_signal(
            "/sensors/dvl/velocity/D__x",
            start_time=t0_dt.timestamp(),
            end_time=t1_dt.timestamp(),
            reset_time_stamp_to_zero=False,
        )
        vy = db.get_signal(
            "/sensors/dvl/velocity/D__y",
            start_time=t0_dt.timestamp(),
            end_time=t1_dt.timestamp(),
            reset_time_stamp_to_zero=False,
        )
        vz = db.get_signal(
            "/sensors/dvl/velocity/D__z",
            start_time=t0_dt.timestamp(),
            end_time=t1_dt.timestamp(),
            reset_time_stamp_to_zero=False,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, sig in (("D__x", vx), ("D__y", vy), ("D__z", vz)):
            # convert each axis separately to UTC datetime
            times = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in sig.axis]
            ax.plot(times, sig.data, label=name)

        # Highlight intervals (use date from t0_dt)
        if highlight_intervals:
            base_date = t0_dt.date()
            for start_str, end_str in highlight_intervals:
                start_time = datetime.combine(
                    base_date,
                    datetime.strptime(start_str, "%H:%M:%S").time(),
                    tzinfo=timezone.utc,
                )
                end_time = datetime.combine(
                    base_date,
                    datetime.strptime(end_str, "%H:%M:%S").time(),
                    tzinfo=timezone.utc,
                )
                ax.axvspan(start_time, end_time, color="yellow", alpha=0.3)

        ax.set_title("DVL Velocity vs UTC Time")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Velocity [m/s]")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
        fig.autofmt_xdate()
        plt.show()

    def plot_dvl_velocity_intervals(self, intervals):
        """
        For each (start_str, end_str) in intervals, plot DVL velocity vs. UTC time on its own figure.
        intervals: list of (start_str, end_str) tuples in "HH:MM:SS" UTC format
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime, timezone

        db = self.__db

        for start_str, end_str in intervals:
            # Parse interval bounds as UTC datetimes using the date from self.__start_time
            base_date = self.__start_time.date()
            start_dt = datetime.combine(
                base_date,
                datetime.strptime(start_str, "%H:%M:%S").time(),
                tzinfo=timezone.utc,
            )
            end_dt = datetime.combine(
                base_date,
                datetime.strptime(end_str, "%H:%M:%S").time(),
                tzinfo=timezone.utc,
            )

            # Convert datetimes to timestamps for signal retrieval
            t0 = start_dt.timestamp()
            t1 = end_dt.timestamp()

            # Retrieve velocity signals in the interval
            vx = db.get_signal(
                "/sensors/dvl/velocity/D__x",
                start_time=t0,
                end_time=t1,
                reset_time_stamp_to_zero=False,
            )
            vy = db.get_signal(
                "/sensors/dvl/velocity/D__y",
                start_time=t0,
                end_time=t1,
                reset_time_stamp_to_zero=False,
            )
            vz = db.get_signal(
                "/sensors/dvl/velocity/D__z",
                start_time=t0,
                end_time=t1,
                reset_time_stamp_to_zero=False,
            )

            # Convert each axis to UTC datetime for plotting
            times_x = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in vx.axis]
            times_y = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in vy.axis]
            times_z = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in vz.axis]

            # Plot each interval in its own figure
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(times_x, vx.data, label="D__x")
            ax.plot(times_y, vy.data, label="D__y")
            ax.plot(times_z, vz.data, label="D__z")

            ax.set_title(f"DVL Velocity\n{start_str}–{end_str} UTC")
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Velocity [m/s]")
            ax.legend()
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M:%S", tz=timezone.utc)
            )
            fig.autofmt_xdate()

            plt.show()

    @staticmethod
    def create_datetime_from_time_string(time_str, reference_datetime):
        """Convert time string like '10:50:49' to datetime object for same date as reference"""
        hour, minute, second = map(int, time_str.split(":"))
        return reference_datetime.replace(
            hour=hour, minute=minute, second=second, microsecond=0
        )

    def create_individual_transect_csvs(self, base_path: str, transect_times: list):
        """Create individual CSV files for each transect in separate folders"""
        print(
            "⚠️  WARNING: create_individual_transect_csvs uses the OLD timestamp alignment method!"
        )
        print(
            "⚠️  This function has the same timestamp misalignment issues as the original create_csv_file."
        )
        print("⚠️  Consider using create_aligned_individual_transect_csvs instead.")
        print()

        # Load all data once (same as original create_csv_file method)
        pose = self.load_pose()
        labels = [
            "latitude [deg]",
            "longitude [deg]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
        ]
        arrays = [sig.data for sig in pose]
        min_len = min(len(a) for a in arrays)
        main_timestamps = np.array(pose[0].axis[:min_len])

        # Create data dict and unwrap yaw
        data = {}
        for lbl, sig in zip(labels, pose):
            values = sig.data[:min_len]
            if lbl == "yaw [deg]":
                values = self.unwrap_degrees(values)
            data[lbl] = values

        data["timestamp [unix epoch s]"] = main_timestamps

        # Load altitude and interpolate
        alt_sig = self.load_altitude()
        alt_ts = np.array(alt_sig.axis)
        alt_data = np.array(alt_sig.data)
        sort_idx = np.argsort(alt_ts)
        alt_ts = alt_ts[sort_idx]
        alt_data = alt_data[sort_idx]

        interp = interp1d(
            alt_ts,
            alt_data,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        data["altitude [m]"] = interp(main_timestamps)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(
            df["timestamp [unix epoch s]"], unit="s", utc=True
        )

        # Process each transect individually
        for start_time_str, end_time_str in transect_times:
            # Create folder name from start time (remove colons)
            folder_name = start_time_str.replace(":", "")
            folder_path = os.path.join(base_path, folder_name)

            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Get the date from the first timestamp
            first_datetime = df["datetime"].iloc[0]

            # Create datetime objects for start and end of this transect
            start_dt = self.create_datetime_from_time_string(
                start_time_str, first_datetime
            )
            end_dt = self.create_datetime_from_time_string(end_time_str, first_datetime)

            # Filter data for this transect
            transect_mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
            transect_df = df[transect_mask].copy()

            if len(transect_df) == 0:
                print(
                    f"Warning: No data found for transect {start_time_str}-{end_time_str}"
                )
                continue

            # Drop the helper datetime column
            transect_df = transect_df.drop("datetime", axis=1)

            # Reorder columns (timestamp first)
            cols = ["timestamp [unix epoch s]"] + [
                c for c in transect_df.columns if c != "timestamp [unix epoch s]"
            ]

            # Create CSV filename
            csv_filename = f"nav_data_{folder_name}.csv"
            csv_path = os.path.join(folder_path, csv_filename)

            # Export to CSV
            transect_df.to_csv(csv_path, columns=cols, index=False)

            print(
                f"Transect {start_time_str}-{end_time_str}: {len(transect_df)} data points"
            )
            print(f"Saved to: {csv_path}")
            print()

    def create_aligned_individual_transect_csvs(
        self, base_path: str, transect_times: list
    ):
        """
        Create individual CSV files for each transect in separate folders.
        Uses proper timestamp alignment to avoid the issues in create_individual_transect_csvs.
        """
        from scipy.interpolate import interp1d
        import pandas as pd
        import numpy as np
        import os

        print("Creating time-aligned individual transect CSVs...")

        # Load pose data
        pose = self.load_pose()
        labels = [
            "latitude [deg]",
            "longitude [deg]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
        ]

        # Find the common time range (intersection of all signals)
        time_ranges = []
        for sig in pose:
            time_ranges.append((sig.axis.min(), sig.axis.max()))

        # Use the intersection of all time ranges
        common_start = max(t[0] for t in time_ranges)
        common_end = min(t[1] for t in time_ranges)
        print(f"Common time range: {common_start:.1f} to {common_end:.1f} s")

        # Create reference time grid using the signal with most data in common range
        best_signal = None
        max_points = 0
        for sig in pose:
            in_range = (sig.axis >= common_start) & (sig.axis <= common_end)
            n_points = in_range.sum()
            if n_points > max_points:
                max_points = n_points
                best_signal = sig

        # Use times from the best signal as reference
        mask = (best_signal.axis >= common_start) & (best_signal.axis <= common_end)
        reference_times = best_signal.axis[mask]
        print(
            f"Using {len(reference_times)} reference time points from signal with most data"
        )

        # Interpolate all pose signals to reference times
        data = {"timestamp [unix epoch s]": reference_times}

        for label, sig in zip(labels, pose):
            print(f"Interpolating {label}...")

            # Sort signal by time to ensure monotonic
            sort_idx = np.argsort(sig.axis)
            sig_times = sig.axis[sort_idx]
            sig_values = sig.data[sort_idx]

            # Create interpolator
            interp_func = interp1d(
                sig_times,
                sig_values,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

            # Interpolate to reference times
            interpolated_values = interp_func(reference_times)

            # For individual transects, use smart wrapping on the entire array
            # This analyzes the whole transect to choose the best wrapping strategy
            if label == "yaw [deg]":
                # Apply smart wrapping to the entire array at once
                interpolated_values = self._smart_wrap_angle_array(interpolated_values)

            data[label] = interpolated_values

        # Add altitude data
        print("Interpolating altitude [m]...")
        alt_sig = self.load_altitude()
        alt_sort_idx = np.argsort(alt_sig.axis)
        alt_interp = interp1d(
            alt_sig.axis[alt_sort_idx],
            alt_sig.data[alt_sort_idx],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        data["altitude [m]"] = alt_interp(reference_times)

        alt_nan_count = np.isnan(data["altitude [m]"]).sum()
        if alt_nan_count > 0:
            print(
                f"  ⚠️  altitude [m]: {alt_nan_count} NaN values ({100*alt_nan_count/len(reference_times):.1f}%)"
            )

        # Create full DataFrame
        df = pd.DataFrame(data)

        # Remove rows with any NaN values to ensure clean data
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)

        print(f"Removed {initial_len - final_len} rows with NaN values")
        print(f"Final aligned dataset: {final_len} rows")

        # Add datetime column for filtering
        df["datetime"] = pd.to_datetime(
            df["timestamp [unix epoch s]"], unit="s", utc=True
        )

        # Process each transect individually
        for start_time_str, end_time_str in transect_times:
            # Create folder name from start time (remove colons)
            folder_name = start_time_str.replace(":", "")
            folder_path = os.path.join(base_path, folder_name)

            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Get the date from the first timestamp
            first_datetime = df["datetime"].iloc[0]

            # Create datetime objects for start and end of this transect
            start_dt = self.create_datetime_from_time_string(
                start_time_str, first_datetime
            )
            end_dt = self.create_datetime_from_time_string(end_time_str, first_datetime)

            # Filter data for this transect
            transect_mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
            transect_df = df[transect_mask].copy()

            if len(transect_df) == 0:
                print(
                    f"Warning: No data found for transect {start_time_str}-{end_time_str}"
                )
                continue

            # Drop the helper datetime column
            transect_df = transect_df.drop("datetime", axis=1)

            # Reorder columns (timestamp first)
            cols = ["timestamp [unix epoch s]"] + [
                c for c in transect_df.columns if c != "timestamp [unix epoch s]"
            ]

            # Create CSV filename
            csv_filename = f"nav_data_{folder_name}.csv"
            csv_path = os.path.join(folder_path, csv_filename)

            # Export to CSV
            transect_df.to_csv(csv_path, columns=cols, index=False)

            print(
                f"✅ Transect {start_time_str}-{end_time_str}: {len(transect_df)} data points"
            )
            print(f"   Saved to: {csv_path}")
            print()

    # -----------------------------------------------------------------------

    def get_individual_transect_csvs(
        self, main_csv_path: str, base_path: str, transect_times: list
    ):
        """
        Simple approach: Extract transect time ranges directly from existing CSV file.
        This preserves all the good properties of the main navigation data (proper yaw, etc.)
        without any additional processing that could introduce artifacts.

        Parameters:
        -----------
        main_csv_path : str
            Path to the main navigation CSV file (already processed with proper alignment)
        base_path : str
            Base directory where individual transect folders will be created
        transect_times : list
            List of (start_time_str, end_time_str) tuples, e.g. [("10:50:49", "10:59:00")]
        """
        import pandas as pd
        import os
        from datetime import datetime, timezone, timedelta

        # print("🔧 Extracting individual transect CSVs from main navigation file...")
        # print(f"   Source: {main_csv_path}")
        # print(f"   Output base: {base_path}")
        # print()

        # Load the main CSV file
        try:
            df = pd.read_csv(main_csv_path)
        except Exception as e:
            print(f"❌ Error reading main CSV file: {e}")
            return

        # Ensure we have timestamp column
        if "timestamp [unix epoch s]" not in df.columns:
            print("❌ No timestamp column found in main CSV file")
            return

        # Convert timestamp to datetime for easier filtering
        df["datetime"] = pd.to_datetime(
            df["timestamp [unix epoch s]"], unit="s", utc=True
        )

        # Get the date from the first timestamp for creating time strings
        first_datetime = df["datetime"].iloc[0]

        # print(f"📊 Main CSV contains {len(df)} data points")
        # print(f"   Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        # print()

        # Process each transect
        for start_time_str, end_time_str in transect_times:
            # Create folder name from start time (remove colons)
            folder_name = start_time_str.replace(":", "")
            folder_path = os.path.join(base_path, folder_name)

            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Create datetime objects for start and end of this transect
            start_dt = self.create_datetime_from_time_string(
                start_time_str, first_datetime
            )
            end_dt = self.create_datetime_from_time_string(end_time_str, first_datetime)

            # Filter data for this transect
            transect_mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
            transect_df = df[transect_mask].copy()

            if len(transect_df) == 0:
                print(
                    f"⚠️  Warning: No data found for transect {start_time_str}-{end_time_str}"
                )
                continue

            # Drop the helper datetime column
            transect_df = transect_df.drop("datetime", axis=1)

            # Create CSV filename
            csv_filename = f"nav_data_{folder_name}.csv"
            csv_path = os.path.join(folder_path, csv_filename)

            # Export to CSV (preserve original column order)
            transect_df.to_csv(csv_path, index=False)

            # Quick yaw range check
            if "yaw [deg]" in transect_df.columns:
                yaw_min = transect_df["yaw [deg]"].min()
                yaw_max = transect_df["yaw [deg]"].max()
                yaw_range_str = f"(yaw range: {yaw_min:.1f}° to {yaw_max:.1f}°)"
            else:
                yaw_range_str = "(no yaw data)"

            print(
                f"✅ Transect {start_time_str}-{end_time_str}: {len(transect_df)} points {yaw_range_str}"
            )
            print(f"   📁 Saved to: {csv_path}")
            print()

        print("🎉 All transect CSVs extracted successfully!")
        print("   ➡️  These should inherit the clean yaw values from main CSV")

    def create_dvl_velocity_dead_reckoned_csv(
        self,
        input_csv_path: str,
        output_suffix: str = "_dr_dvl",
        adjust_speed_to_hit_end: bool = False,
    ):
        """
        Create a dead-reckoned CSV using actual DVL velocity measurements instead of constant velocity.
        This should be much more accurate than the constant velocity approach.

        Parameters:
        -----------
        input_csv_path : str
            Path to the input transect CSV file
        output_suffix : str
            Suffix to add to the output filename (default: "_dr_dvl")
        adjust_speed_to_hit_end : bool
            Whether to adjust speeds to hit the end position exactly (default: False)

        Returns:
        --------
        str : Path to the created dead-reckoned CSV file
        """
        import pandas as pd
        import numpy as np
        import os
        from scipy.interpolate import interp1d

        print(
            f"🚀 Creating DVL-based dead reckoning for: {os.path.basename(input_csv_path)}"
        )

        # Load the input CSV
        df = pd.read_csv(input_csv_path)

        if len(df) < 2:
            print("❌ Not enough data points for dead reckoning")
            return None

        # Get time range for this transect
        start_timestamp = df["timestamp [unix epoch s]"].iloc[0]
        end_timestamp = df["timestamp [unix epoch s]"].iloc[-1]

        print(f"   📊 Transect time range: {start_timestamp} to {end_timestamp}")
        print(f"   📊 Duration: {end_timestamp - start_timestamp:.1f} seconds")

        # Load DVL velocity data for this time range
        import datetime

        start_dt = datetime.datetime.fromtimestamp(
            start_timestamp, tz=datetime.timezone.utc
        )
        end_dt = datetime.datetime.fromtimestamp(
            end_timestamp, tz=datetime.timezone.utc
        )

        try:
            # Get DVL velocity signals
            vx_dvl = self.__db.get_signal(
                "/sensors/dvl/velocity/D__x",
                start_time=start_dt,
                end_time=end_dt,
                reset_time_stamp_to_zero=False,
            )
            vy_dvl = self.__db.get_signal(
                "/sensors/dvl/velocity/D__y",
                start_time=start_dt,
                end_time=end_dt,
                reset_time_stamp_to_zero=False,
            )

            print(
                f"   📊 DVL data points: vx={len(vx_dvl.data)}, vy={len(vy_dvl.data)}"
            )

            if len(vx_dvl.data) == 0 or len(vy_dvl.data) == 0:
                print("❌ No DVL velocity data found for this time range")
                return None

        except Exception as e:
            print(f"❌ Error loading DVL data: {e}")
            return None

        # Interpolate DVL velocities to match the CSV timestamps
        try:
            vx_interp = interp1d(
                vx_dvl.axis,
                vx_dvl.data,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            vy_interp = interp1d(
                vy_dvl.axis,
                vy_dvl.data,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            # Get velocities at CSV timestamps
            csv_timestamps = df["timestamp [unix epoch s]"].values
            vx_body = vx_interp(csv_timestamps)
            vy_body = vy_interp(csv_timestamps)

            print(
                f"   📊 Interpolated velocities - vx range: {np.min(vx_body):.3f} to {np.max(vx_body):.3f} m/s"
            )
            print(
                f"   📊 Interpolated velocities - vy range: {np.min(vy_body):.3f} to {np.max(vy_body):.3f} m/s"
            )

        except Exception as e:
            print(f"❌ Error interpolating DVL velocities: {e}")
            return None

        # Start position and heading
        start_lat = df["latitude [deg]"].iloc[0]
        start_lon = df["longitude [deg]"].iloc[0]

        # Calculate bearing from first to last point
        end_lat = df["latitude [deg]"].iloc[-1]
        end_lon = df["longitude [deg]"].iloc[-1]
        bearing_rad = np.arctan2(end_lon - start_lon, end_lat - start_lat)
        bearing_deg = np.degrees(bearing_rad)

        print(f"   🧭 Initial bearing: {bearing_deg:.2f}°")

        # Dead reckoning using actual DVL velocities
        new_lats = [start_lat]
        new_lons = [start_lon]

        # Earth radius (approximate)
        R_earth = 6.378137e6  # meters

        for i in range(1, len(df)):
            dt = csv_timestamps[i] - csv_timestamps[i - 1]

            # Get body velocities at this timestep
            vx = vx_body[i]  # forward velocity
            vy = vy_body[i]  # lateral velocity

            # Get heading (yaw) for body-to-earth frame transformation
            yaw_deg = df["yaw [deg]"].iloc[i]
            yaw_rad = np.radians(yaw_deg)

            # Transform body velocities to earth frame
            # vx_body = forward, vy_body = starboard
            # ve_earth = east, vn_earth = north
            ve_earth = vx * np.sin(yaw_rad) + vy * np.cos(yaw_rad)
            vn_earth = vx * np.cos(yaw_rad) - vy * np.sin(yaw_rad)

            # Update position using earth-frame velocities
            prev_lat = new_lats[-1]
            prev_lon = new_lons[-1]

            # Convert velocity to lat/lon changes
            dlat = (vn_earth * dt) / R_earth * 180.0 / np.pi
            dlon = (
                (ve_earth * dt)
                / (R_earth * np.cos(np.radians(prev_lat)))
                * 180.0
                / np.pi
            )

            new_lat = prev_lat + dlat
            new_lon = prev_lon + dlon

            new_lats.append(new_lat)
            new_lons.append(new_lon)

        # Optionally adjust to hit the end point exactly
        if adjust_speed_to_hit_end:
            # Scale the entire trajectory to end at the correct point
            actual_end_lat = new_lats[-1]
            actual_end_lon = new_lons[-1]

            lat_error = end_lat - actual_end_lat
            lon_error = end_lon - actual_end_lon

            print(
                f"   🎯 End position error: lat={lat_error*1e5:.1f}e-5°, lon={lon_error*1e5:.1f}e-5°"
            )

            # Apply linear correction
            for i in range(len(new_lats)):
                progress = i / (len(new_lats) - 1)
                new_lats[i] += lat_error * progress
                new_lons[i] += lon_error * progress

        # Create the output dataframe
        df_out = df.copy()
        df_out["latitude [deg]"] = new_lats
        df_out["longitude [deg]"] = new_lons

        # Add DVL velocity columns for reference
        df_out["dvl_vx_body [m/s]"] = vx_body
        df_out["dvl_vy_body [m/s]"] = vy_body

        # Calculate effective speed
        total_speed = np.sqrt(vx_body**2 + vy_body**2)
        avg_speed = np.mean(total_speed)

        # Save to CSV
        base_name = os.path.splitext(input_csv_path)[0]
        out_path = f"{base_name}{output_suffix}.csv"
        df_out.to_csv(out_path, index=False)

        # Calculate final statistics
        original_distance = self._calculate_distance(
            start_lat, start_lon, end_lat, end_lon
        )
        dr_distance = self._calculate_distance(
            start_lat, start_lon, new_lats[-1], new_lons[-1]
        )

        print(f"   ✅ DVL dead reckoning complete!")
        print(f"   📏 Original distance: {original_distance:.2f} m")
        print(f"   📏 DR distance: {dr_distance:.2f} m")
        print(f"   📏 Distance error: {abs(dr_distance - original_distance):.2f} m")
        print(f"   🏃 Average speed: {avg_speed:.3f} m/s")
        print(f"   💾 Saved: {os.path.basename(out_path)}")

        return out_path

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two lat/lon points in meters using Haversine formula"""
        import math

        R = 6.378137e6  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def create_transect_csvs_and_dvl_dead_reckon(
        self,
        base_path: str,
        transect_times: list,
        main_csv_path: str,
        adjust_speed_to_hit_end: bool = False,
    ):
        """
        • Extract transect CSVs from main navigation file (simple approach)
        • Run DVL-velocity-based dead reckoning on each transect

        Parameters:
        -----------
        base_path : str
            Base directory where transect folders will be created
        transect_times : list
            List of (start_time_str, end_time_str) tuples
        main_csv_path : str
            Path to the main navigation CSV file
        adjust_speed_to_hit_end : bool
            Whether to adjust speeds to hit end position exactly
        #"""
        print("🎯 Creating transect CSVs and applying DVL-based dead reckoning...")
        print("   ✅ Using actual DVL velocities instead of constant values")
        print("   ✅ Should be much more accurate!")
        print()

        # First, extract the individual transect CSVs
        self.get_individual_transect_csvs(main_csv_path, base_path, transect_times)

        print()
        print("🔄 Now applying DVL-based dead reckoning to each transect...")
        print()

        # Then apply DVL-based dead reckoning to each
        for start_time_str, end_time_str in transect_times:
            folder_name = start_time_str.replace(":", "")
            csv_path = os.path.join(
                base_path, folder_name, f"nav_data_{folder_name}.csv"
            )

            if os.path.exists(csv_path):
                try:
                    dr_path = self.create_dvl_velocity_dead_reckoned_csv(
                        csv_path,
                        output_suffix="_dr_dvl",
                        adjust_speed_to_hit_end=adjust_speed_to_hit_end,
                    )
                    if dr_path:
                        print(f"   ✅ DVL DR completed: {os.path.basename(dr_path)}")
                    print()
                except Exception as e:
                    print(f"   ❌ DVL DR failed for {folder_name}: {e}")
                    print()
            else:
                print(f"   ❌ Transect CSV not found: {csv_path}")

        print("🎉 All DVL-based dead reckoning completed!")

    # -----------------------------------------------------------------------
    # Geodesy helpers (spherical Earth, plenty accurate for short transects)
    @staticmethod
    def _initial_bearing_rad(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, atan2

        lat1 = radians(lat1)
        lat2 = radians(lat2)
        dlon = radians(lon2 - lon1)
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        return atan2(y, x)

    @staticmethod
    def _great_circle_distance_m(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos

        _EARTH_RADIUS_M = 6_378_137.0  # WGS-84 semi-major axis
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * _EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @staticmethod
    def _destination_point(lat, lon, bearing_rad, distance_m):
        from math import radians, degrees, sin, cos, asin, atan2

        _EARTH_RADIUS_M = 6_378_137.0  # WGS-84 semi-major axis
        lat1 = radians(lat)
        lon1 = radians(lon)
        ang = distance_m / _EARTH_RADIUS_M
        sin_lat1, cos_lat1 = sin(lat1), cos(lat1)
        sin_ang, cos_ang = sin(ang), cos(ang)

        sin_lat2 = sin_lat1 * cos_ang + cos_lat1 * sin_ang * cos(bearing_rad)
        lat2 = asin(sin_lat2)
        y = sin(bearing_rad) * sin_ang * cos_lat1
        x = cos_ang - sin_lat1 * sin_lat2
        lon2 = lon1 + atan2(y, x)
        lon2_deg = (degrees(lon2) + 540) % 360 - 180
        return degrees(lat2), lon2_deg

    @staticmethod
    def _wrap_to_180(deg_val):  # for yaw / bearing columns
        return (deg_val + 180.0) % 360.0 - 180.0

    @staticmethod
    @staticmethod
    def _smart_wrap_angle_array(deg_array):
        """
        Apply smart wrapping to an entire array of angles.
        Analyzes the distribution to choose optimal wrapping range.
        """
        import numpy as np

        # Remove NaN values for analysis
        valid_angles = deg_array[~np.isnan(deg_array)]
        if len(valid_angles) == 0:
            return deg_array

        # Normalize to [0, 360] for analysis
        normalized = valid_angles % 360.0

        # Check if data spans across 0°/360° boundary
        # This happens when we have angles both near 0° and near 360°
        has_low_angles = np.any(normalized <= 90)  # Angles near 0°
        has_high_angles = np.any(normalized >= 270)  # Angles near 360°

        # If data spans across 0°/360° boundary, use [-180, 180] wrapping
        # This prevents jumps between 0° and 360°
        if has_low_angles and has_high_angles:
            # Use [-180, 180] wrapping for data crossing 0°/360° boundary
            return (deg_array + 180.0) % 360.0 - 180.0
        else:
            # Check which range has more data
            near_zero = np.sum((normalized >= 270) | (normalized <= 90))
            near_180 = np.sum((normalized > 90) & (normalized < 270))

            if near_zero >= near_180:
                # Data clustered around 0° - use [-180, 180] wrapping
                return (deg_array + 180.0) % 360.0 - 180.0
            else:
                # Data clustered around 180° - use [0, 360] wrapping
                return deg_array % 360.0

    def ssa_180(self, angle):
        """
        Normalize an angle to the range [-180, 180] degrees.

        Parameters:
            angle (float or np.array): Input angle(s) in degrees.

        Returns:
            float or np.array: Normalized angle(s) in range [-180, 180].
        """
        return (angle + 180.0) % 360.0 - 180.0

    def ssa(self, angle):
        return angle % 360.0

    def create_constant_body_vel_dead_reckoned_csv(
        self,
        input_csv_path: str,
        vx_body_mps: float = 0.2,
        vy_body_mps: float = 0.0,
        adjust_speed_to_hit_end: bool = False,
        output_suffix: str = "_dr",
    ):
        """
        Re-write <input_csv_path> with lat/lon replaced by dead-reckoned positions
        along a straight-line heading, constant body-axis velocity (vx, vy).
        Returns path of new CSV.
        """
        import pandas as pd
        from math import degrees

        if not os.path.isfile(input_csv_path):
            raise FileNotFoundError(input_csv_path)

        df = pd.read_csv(input_csv_path)
        req = ["timestamp [unix epoch s]", "latitude [deg]", "longitude [deg]"]
        if any(c not in df.columns for c in req):
            raise ValueError(f"CSV missing required cols: {req}")

        valid = df["latitude [deg]"].notna() & df["longitude [deg]"].notna()
        if not valid.any():
            raise ValueError("No valid lat/lon in CSV")

        first_idx = valid.idxmax()
        last_idx = valid[::-1].idxmax()

        lat0, lon0 = map(
            float, df.loc[first_idx, ["latitude [deg]", "longitude [deg]"]]
        )
        lat1, lon1 = map(float, df.loc[last_idx, ["latitude [deg]", "longitude [deg]"]])

        bearing = self._initial_bearing_rad(lat0, lon0, lat1, lon1)
        bearing_deg = degrees(bearing)

        # --- constant speed handling --------------------------------------
        t = df["timestamp [unix epoch s]"].astype(float).values
        dt_total = max(float(t[last_idx] - t[first_idx]), 0.0)

        if adjust_speed_to_hit_end and dt_total > 0:
            dist = self._great_circle_distance_m(lat0, lon0, lat1, lon1)
            vx_body_mps = dist / dt_total
            vy_body_mps = 0.0

        speed = float(np.hypot(vx_body_mps, vy_body_mps))

        # --- integrate -----------------------------------------------------
        lat_new = df["latitude [deg]"].astype(float).values.copy()
        lon_new = df["longitude [deg]"].astype(float).values.copy()
        lat_cur, lon_cur = lat0, lon0
        lat_new[first_idx] = lat_cur
        lon_new[first_idx] = lon_cur

        for i in range(first_idx + 1, last_idx + 1):
            step = float(t[i] - t[i - 1])
            dist = speed * step
            lat_cur, lon_cur = self._destination_point(lat_cur, lon_cur, bearing, dist)
            lat_new[i] = lat_cur
            lon_new[i] = lon_cur

        # --- write out -----------------------------------------------------
        df["latitude [deg] (orig)"] = df["latitude [deg]"]
        df["longitude [deg] (orig)"] = df["longitude [deg]"]
        df["latitude [deg]"] = lat_new
        df["longitude [deg]"] = lon_new
        # if "yaw [deg]" in df.columns:
        #     df["yaw [deg]"] = self._smart_wrap_angle(bearing_deg)
        # df["dr_bearing [deg]"] = self._smart_wrap_angle(bearing_deg)
        df["dr_speed [m/s]"] = speed

        out_path = os.path.splitext(input_csv_path)[0] + output_suffix + ".csv"

        # # Check if file already exists to avoid duplicate processing
        # if os.path.exists(out_path):
        #     print(f"[DR] Skipping {os.path.basename(input_csv_path)} - output already exists")
        #     return out_path

        df.to_csv(out_path, index=False)

        print(
            f"[DR] {os.path.basename(input_csv_path)} → {os.path.basename(out_path)} "
            f"({bearing_deg:.2f}°, {speed:.3f} m/s)"
        )
        return out_path

    def create_dvl_velocity_dead_reckoned_csv_v2(
        self,
        input_csv_path: str,
        output_suffix: str = "_dr_dvl",
        adjust_speed_to_hit_end: bool = False,
    ):
        """
        DVL-based dead reckoning using YOUR original approach:
        - Calculate constant heading from start to end position
        - Use actual DVL speeds but along the straight-line bearing
        - Much simpler and follows your proven method

        Parameters:
        -----------
        input_csv_path : str
            Path to the input transect CSV file
        output_suffix : str
            Suffix to add to the output filename (default: "_dr_dvl")
        adjust_speed_to_hit_end : bool
            Whether to adjust speeds to hit the end position exactly (default: False)

        Returns:
        --------
        str : Path to the created dead-reckoned CSV file
        """
        import pandas as pd
        import numpy as np
        import os
        from scipy.interpolate import interp1d
        from math import degrees

        print(
            f"🚀 DVL dead reckoning (constant heading): {os.path.basename(input_csv_path)}"
        )

        # Load the input CSV
        if not os.path.isfile(input_csv_path):
            raise FileNotFoundError(input_csv_path)

        df = pd.read_csv(input_csv_path)
        req = ["timestamp [unix epoch s]", "latitude [deg]", "longitude [deg]"]
        if any(c not in df.columns for c in req):
            raise ValueError(f"CSV missing required cols: {req}")

        valid = df["latitude [deg]"].notna() & df["longitude [deg]"].notna()
        if not valid.any():
            raise ValueError("No valid lat/lon in CSV")

        first_idx = valid.idxmax()
        last_idx = valid[::-1].idxmax()

        lat0, lon0 = map(
            float, df.loc[first_idx, ["latitude [deg]", "longitude [deg]"]]
        )
        lat1, lon1 = map(float, df.loc[last_idx, ["latitude [deg]", "longitude [deg]"]])

        # Calculate constant bearing (same as your original method)
        bearing = self._initial_bearing_rad(lat0, lon0, lat1, lon1)
        bearing_deg = degrees(bearing)

        print(f"   🧭 Constant bearing: {bearing_deg:.2f}°")

        # Get time range for DVL data loading
        start_timestamp = df["timestamp [unix epoch s]"].iloc[first_idx]
        end_timestamp = df["timestamp [unix epoch s]"].iloc[last_idx]

        import datetime

        start_dt = datetime.datetime.fromtimestamp(
            start_timestamp, tz=datetime.timezone.utc
        )
        end_dt = datetime.datetime.fromtimestamp(
            end_timestamp, tz=datetime.timezone.utc
        )

        # Load DVL velocity data
        try:
            vx_dvl = self.__db.get_signal(
                "/sensors/dvl/velocity/D__x",
                start_time=start_dt,
                end_time=end_dt,
                reset_time_stamp_to_zero=False,
            )
            vy_dvl = self.__db.get_signal(
                "/sensors/dvl/velocity/D__y",
                start_time=start_dt,
                end_time=end_dt,
                reset_time_stamp_to_zero=False,
            )

            if len(vx_dvl.data) == 0 or len(vy_dvl.data) == 0:
                print("❌ No DVL velocity data - falling back to constant velocity")
                return self.create_constant_body_vel_dead_reckoned_csv(
                    input_csv_path,
                    vx_body_mps=0.2,
                    vy_body_mps=0.0,
                    adjust_speed_to_hit_end=adjust_speed_to_hit_end,
                    output_suffix=output_suffix,
                )

        except Exception as e:
            print(f"❌ DVL data loading failed: {e}")
            print("   Falling back to constant velocity")
            return self.create_constant_body_vel_dead_reckoned_csv(
                input_csv_path,
                vx_body_mps=0.2,
                vy_body_mps=0.0,
                adjust_speed_to_hit_end=adjust_speed_to_hit_end,
                output_suffix=output_suffix,
            )

        # Interpolate DVL velocities to CSV timestamps
        try:
            vx_interp = interp1d(
                vx_dvl.axis,
                vx_dvl.data,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            vy_interp = interp1d(
                vy_dvl.axis,
                vy_dvl.data,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            csv_timestamps = df["timestamp [unix epoch s]"].values
            vx_body = vx_interp(csv_timestamps)
            vy_body = vy_interp(csv_timestamps)

            # Calculate total speed (magnitude) - this is what we'll use for dead reckoning
            dvl_speeds = np.sqrt(vx_body**2 + vy_body**2)
            avg_speed = np.mean(dvl_speeds[first_idx : last_idx + 1])

            print(
                f"   📊 DVL speed range: {np.min(dvl_speeds):.3f} to {np.max(dvl_speeds):.3f} m/s"
            )
            print(f"   📊 Average DVL speed: {avg_speed:.3f} m/s")

        except Exception as e:
            print(f"❌ DVL interpolation failed: {e}")
            print("   Falling back to constant velocity")
            return self.create_constant_body_vel_dead_reckoned_csv(
                input_csv_path,
                vx_body_mps=0.2,
                vy_body_mps=0.0,
                adjust_speed_to_hit_end=adjust_speed_to_hit_end,
                output_suffix=output_suffix,
            )

        # --- Dead reckoning integration (following YOUR method) ---
        t = df["timestamp [unix epoch s]"].astype(float).values
        lat_new = df["latitude [deg]"].astype(float).values.copy()
        lon_new = df["longitude [deg]"].astype(float).values.copy()

        lat_cur, lon_cur = lat0, lon0
        lat_new[first_idx] = lat_cur
        lon_new[first_idx] = lon_cur

        # Integrate along constant bearing using actual DVL speeds
        for i in range(first_idx + 1, last_idx + 1):
            step = float(t[i] - t[i - 1])  # time step
            speed = dvl_speeds[i]  # actual DVL speed at this timestep
            dist = speed * step  # distance traveled

            # Move along constant bearing (same as your original method)
            lat_cur, lon_cur = self._destination_point(lat_cur, lon_cur, bearing, dist)
            lat_new[i] = lat_cur
            lon_new[i] = lon_cur

        # --- Create output (same format as your original) ---
        df["latitude [deg] (orig)"] = df["latitude [deg]"]
        df["longitude [deg] (orig)"] = df["longitude [deg]"]
        df["latitude [deg]"] = lat_new
        df["longitude [deg]"] = lon_new

        # Add DVL data for reference
        df["dvl_vx_body [m/s]"] = vx_body
        df["dvl_vy_body [m/s]"] = vy_body
        df["dvl_speed [m/s]"] = dvl_speeds

        # if "yaw [deg]" in df.columns:
        #     df["yaw [deg]"] = self._smart_wrap_angle(bearing_deg)
        # df["dr_bearing [deg]"] = self._smart_wrap_angle(bearing_deg)

        out_path = os.path.splitext(input_csv_path)[0] + output_suffix + ".csv"
        df.to_csv(out_path, index=False)

        # Calculate statistics
        original_distance = self._great_circle_distance_m(lat0, lon0, lat1, lon1)
        dr_distance = self._great_circle_distance_m(
            lat0, lon0, lat_new[last_idx], lon_new[last_idx]
        )

        print(
            f"   ✅ DVL DR complete: {bearing_deg:.2f}°, avg speed {avg_speed:.3f} m/s"
        )
        print(f"   📏 Original distance: {original_distance:.1f} m")
        print(f"   📏 DR distance: {dr_distance:.1f} m")
        print(f"   💾 Saved: {os.path.basename(out_path)}")

        return out_path

    def create_transect_csvs_and_dead_reckon(
        self,
        base_path: str,
        transect_times: list,
        vx_body_mps: float = 0.2,
        vy_body_mps: float = 0.0,
        adjust_speed_to_hit_end: bool = False,
    ):
        """
        • Makes one CSV per (start,end) in `transect_times`
        • Immediately runs constant-velocity DR on each, producing *_dr.csv.
        """
        import pandas as pd
        from scipy.interpolate import interp1d

        # ---------- load full log once ------------------------------------
        pose = self.load_pose()
        labels = [
            "latitude [deg]",
            "longitude [deg]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
        ]
        arrays = [sig.data for sig in pose]
        min_len = min(len(a) for a in arrays)
        ts_main = np.array(pose[0].axis[:min_len])

        data = {}
        for lbl, sig in zip(labels, pose):
            v = sig.data[:min_len]
            if lbl == "yaw [deg]":
                v = self.unwrap_degrees(v)
            data[lbl] = v
        data["timestamp [unix epoch s]"] = ts_main

        # altitude
        alt = self.load_altitude()
        alt_ts = np.array(alt.axis)
        sort_i = np.argsort(alt_ts)

        alt_interp = interp1d(
            alt_ts[sort_i],
            np.array(alt.data)[sort_i],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        data["altitude [m]"] = alt_interp(ts_main)

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(
            df["timestamp [unix epoch s]"], unit="s", utc=True
        )

        # ---------- iterate transects -------------------------------------
        saved = []
        for start_str, end_str in transect_times:
            folder = start_str.replace(":", "")
            out_dir = os.path.join(base_path, folder)
            os.makedirs(out_dir, exist_ok=True)

            dt0 = df["datetime"].iloc[0]
            sh, sm, ss = map(int, start_str.split(":"))
            eh, em, es = map(int, end_str.split(":"))
            t_start = dt0.replace(hour=sh, minute=sm, second=ss, microsecond=0)
            t_end = dt0.replace(hour=eh, minute=em, second=es, microsecond=0)

            m = (df["datetime"] >= t_start) & (df["datetime"] <= t_end)
            seg = df[m].copy()
            if seg.empty:
                print(f"!! No data {start_str}-{end_str}")
                continue

            seg.drop(columns="datetime", inplace=True)
            # Keep original yaw values for individual transect CSVs
            # Smart wrapping will be applied during dead reckoning

            cols = ["timestamp [unix epoch s]"] + [
                c for c in seg.columns if c != "timestamp [unix epoch s]"
            ]
            fname = f"nav_data_{folder}.csv"
            path = os.path.join(out_dir, fname)
            seg.to_csv(path, columns=cols, index=False)
            print(f"Transect {start_str}-{end_str}: {len(seg)} pts → {path}")
            saved.append(path)

        # ---------- dead-reckon each --------------------------------------
        for p in saved:
            self.create_constant_body_vel_dead_reckoned_csv(
                p,
                vx_body_mps=vx_body_mps,
                vy_body_mps=vy_body_mps,
                adjust_speed_to_hit_end=adjust_speed_to_hit_end,
                output_suffix="_dr",
            )
