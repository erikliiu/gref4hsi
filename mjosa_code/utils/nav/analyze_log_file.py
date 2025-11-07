import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from geopy.distance import geodesic


class LogData:
    def __init__(self, csv_path: str):
        """
        Loads CSV, drops any rows where altitude is NaN,
        and stores each column as a NumPy array on self.
        """
        df = pd.read_csv(csv_path)
        df = df.rename(
            columns={
                "timestamp [unix epoch s]": "timestamp",
                "latitude [deg]": "latitude",
                "longitude [deg]": "longitude",
                "depth [m]": "depth",
                "roll [deg]": "roll",
                "pitch [deg]": "pitch",
                "yaw [deg]": "yaw",
                "altitude [m]": "altitude",
            }
        )
        # remove rows without altitude
        df = df[df["altitude"].notna()]
        # store arrays
        for col in df.columns:
            setattr(self, col, df[col].to_numpy())
        self._data = df

    def to_dict(self) -> dict:
        return {col: getattr(self, col) for col in self._data.columns}

    # def plot_test(self):
    #     """
    #     Generates quick diagnostic plots, all laid out horizontally:
    #     - Position scatter
    #     - Pose time series (lat/lon/depth/roll/pitch/yaw)
    #     - Altitude time series
    #     - 3D trajectory
    #     """
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa

    #     # Set up a horizontal layout with 9 plots: 1 position + 6 pose + 1 altitude + 1 3D
    #     fig = plt.figure(figsize=(36, 4))  # wide horizontal figure

    #     # 1. Position scatter (subplot 1)
    #     ax1 = fig.add_subplot(1, 9, 1)
    #     ax1.scatter(self.longitude, self.latitude, s=1)
    #     ax1.set_xlabel("Lon [deg]")
    #     ax1.set_ylabel("Lat [deg]")
    #     ax1.set_title("Position")

    #     # 2. Pose time series (subplot 2-7)
    #     signals = [
    #         ("latitude", "Lat [deg]"),
    #         ("longitude", "Lon [deg]"),
    #         ("depth", "Depth [m]"),
    #         ("roll", "Roll [deg]"),
    #         ("pitch", "Pitch [deg]"),
    #         ("yaw", "Yaw [deg]"),
    #     ]
    #     for i, (attr, label) in enumerate(signals, start=2):
    #         ax = fig.add_subplot(1, 9, i)
    #         ax.plot(self.timestamp, getattr(self, attr))
    #         ax.set_title(label)
    #         ax.set_xlabel("Time [s]")

    #     # 3. Altitude time series (subplot 8)
    #     ax8 = fig.add_subplot(1, 9, 8)
    #     ax8.plot(self.timestamp, self.altitude)
    #     ax8.set_xlabel("Time [s]")
    #     ax8.set_ylabel("Alt [m]")
    #     ax8.set_title("Altitude")

    #     # 4. 3D trajectory (subplot 9)
    #     ax9 = fig.add_subplot(1, 9, 9, projection="3d")
    #     ax9.plot(self.longitude, self.latitude, self.altitude)
    #     ax9.set_xlabel("Lon")
    #     ax9.set_ylabel("Lat")
    #     ax9.set_zlabel("Alt")
    #     ax9.set_title("3D Trajectory")

    #     plt.tight_layout()
    #     plt.show()

    @staticmethod
    def deg_to_meter_simple(latitudes, longitudes):
        """
        Simple conversion of lat/lon to relative meters using mathematical approximation.
        Returns (east, north) coordinates in meters relative to first point.
        """
        import numpy as np

        # Use first point as origin
        lat0, lon0 = latitudes[0], longitudes[0]

        # Earth radius in meters
        R = 6378137.0

        # Convert to radians
        lat_rad = np.radians(latitudes)
        lon_rad = np.radians(longitudes)
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)

        # Calculate relative distances in meters
        east = (lon_rad - lon0_rad) * R * np.cos(lat0_rad)
        north = (lat_rad - lat0_rad) * R

        return east, north

    def plot_test(self, use_meters=False):
        """
        Generates quick diagnostic plots, each in its own figure:
        1) Position scatter
        2) Pose time series (lat, lon, depth, roll, pitch, yaw)
        3) Altitude time series
        4) 3D trajectory

        Parameters:
        -----------
        use_meters : bool, optional
            If True, converts lat/lon to meters for spatial plots.
            If False (default), uses degrees.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        # Convert coordinates if requested
        if use_meters:
            x_coords, y_coords = self.deg_to_meter_simple(self.latitude, self.longitude)
            x_label = "East [m]"
            y_label = "North [m]"
            x_3d_label = "East [m]"
            y_3d_label = "North [m]"
        else:
            x_coords = self.longitude
            y_coords = self.latitude
            x_label = "Lon [deg]"
            y_label = "Lat [deg]"
            x_3d_label = "Lon"
            y_3d_label = "Lat"

        # 1. Position scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x_coords, y_coords, s=5, alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Position")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

        # 2. Pose time series
        signals = [
            ("latitude", "Lat [deg]"),
            ("longitude", "Lon [deg]"),
            ("depth", "Depth [m]"),
            ("roll", "Roll [deg]"),
            ("pitch", "Pitch [deg]"),
            ("yaw", "Yaw [deg]"),
        ]

        for attr, label in signals:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(self.timestamp, getattr(self, attr))
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(label)
            ax.set_title(label + " over Time")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.show()

        # 3. Altitude time series
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.timestamp, self.altitude)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Alt [m]")
        ax.set_title("Altitude over Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

        # 4. 3D trajectory
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_coords, y_coords, self.altitude)
        ax.set_xlabel(x_3d_label)
        ax.set_ylabel(y_3d_label)
        ax.set_zlabel("Alt")
        ax.set_title("3D Trajectory")
        fig.tight_layout()
        plt.show()

    def plot_with_transects(
        self, transect_csv_paths: list, transect_labels: list = None
    ):
        """
        Generates diagnostic plots with main data and overlaid transect segments.

        Parameters:
        -----------
        transect_csv_paths : list
            List of paths to transect CSV files
        transect_labels : list, optional
            Labels for each transect. If None, will use filename
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import pandas as pd
        import os

        # Load transect data
        transects = []
        labels = []
        colors = [
            "red",
            "orange",
            "green",
            "blue",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        for i, csv_path in enumerate(transect_csv_paths):
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = df.rename(
                    columns={
                        "timestamp [unix epoch s]": "timestamp",
                        "latitude [deg]": "latitude",
                        "longitude [deg]": "longitude",
                        "depth [m]": "depth",
                        "roll [deg]": "roll",
                        "pitch [deg]": "pitch",
                        "yaw [deg]": "yaw",
                        "altitude [m]": "altitude",
                    }
                )
                df = df[df["altitude"].notna()]
                transects.append(df)

                # Create label
                if transect_labels and i < len(transect_labels):
                    labels.append(transect_labels[i])
                else:
                    filename = os.path.basename(csv_path)
                    labels.append(filename.replace(".csv", ""))
            else:
                print(f"Warning: {csv_path} not found")

        if not transects:
            print("No valid transect files found. Plotting main data only.")
            self.plot_test()
            return

        # 1. Position scatter with transects overlaid
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            self.longitude,
            self.latitude,
            s=3,
            alpha=0.5,
            color="blue",
            label="All data",
        )

        for i, (transect, label) in enumerate(zip(transects, labels)):
            color = colors[i % len(colors)]
            ax.scatter(
                transect["longitude"],
                transect["latitude"],
                s=1,
                alpha=0.8,
                color=color,
                label=label,
            )

        ax.set_xlabel("Lon [deg]")
        ax.set_ylabel("Lat [deg]")
        ax.set_title("Position - Main Data with Transect Highlights")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        plt.show()

        # 2. Time series plots with transects overlaid
        signals = [
            ("latitude", "Lat [deg]"),
            ("longitude", "Lon [deg]"),
            ("depth", "Depth [m]"),
            ("roll", "Roll [deg]"),
            ("pitch", "Pitch [deg]"),
            ("yaw", "Yaw [deg]"),
        ]

        for attr, ylabel in signals:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                self.timestamp,
                getattr(self, attr),
                color="lightgray",
                alpha=0.7,
                linewidth=1,
                label="All data",
            )

            for i, (transect, label) in enumerate(zip(transects, labels)):
                color = colors[i % len(colors)]
                ax.plot(
                    transect["timestamp"],
                    transect[attr],
                    color=color,
                    linewidth=2,
                    alpha=0.9,
                    label=label,
                )

            ax.set_xlabel("Time [s]")
            ax.set_ylabel(ylabel)

            # Invert y-axis for depth to show 0 at top, positive values downward
            if attr == "depth":
                ax.invert_yaxis()

            ax.set_title(f"{ylabel} over Time - with Transect Highlights")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.tight_layout()
            plt.show()

        # 3. Altitude time series with transects
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            self.timestamp,
            self.altitude,
            color="blue",
            alpha=1,
            linewidth=1,
            label="All data",
        )

        for i, (transect, label) in enumerate(zip(transects, labels)):
            color = colors[i % len(colors)]
            ax.plot(
                transect["timestamp"],
                transect["altitude"],
                color=color,
                linewidth=2,
                alpha=0.9,
                label=label,
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Alt [m]")
        ax.set_title("Altitude over Time - with Transect Highlights")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        plt.show()

        # 4. 3D trajectory with transects (depth inverted)
        fig = plt.figure(figsize=(20, 18))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            self.longitude,
            self.latitude,
            -self.depth,  # Negative depth to show 0 at top, positive going down
            color="blue",
            alpha=0.5,
            linewidth=2,
            label="All data",
        )

        for i, (transect, label) in enumerate(zip(transects, labels)):
            color = colors[i % len(colors)]
            ax.plot(
                transect["longitude"],
                transect["latitude"],
                -transect["depth"],  # Negative depth for proper orientation
                color=color,
                linewidth=2,
                alpha=0.9,
                label=label,
            )

        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.set_zlabel("Depth [m]")

        # Invert z-axis to show depth properly (0 at top, positive downward)
        ax.invert_zaxis()

        ax.set_title("3D Trajectory - Main Data with Transect Highlights")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        plt.show()

    def plot_transects_only(
        self, transect_csv_paths: list, transect_labels: list = None
    ):
        """
        Plot only the transect segments (no main data background)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import pandas as pd
        import os

        # Load transect data
        transects = []
        labels = []
        colors = [
            "red",
            "orange",
            "green",
            "blue",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        for i, csv_path in enumerate(transect_csv_paths):
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = df.rename(
                    columns={
                        "timestamp [unix epoch s]": "timestamp",
                        "latitude [deg]": "latitude",
                        "longitude [deg]": "longitude",
                        "depth [m]": "depth",
                        "roll [deg]": "roll",
                        "pitch [deg]": "pitch",
                        "yaw [deg]": "yaw",
                        "altitude [m]": "altitude",
                    }
                )
                df = df[df["altitude"].notna()]
                transects.append(df)

                # Create label
                if transect_labels and i < len(transect_labels):
                    labels.append(transect_labels[i])
                else:
                    filename = os.path.basename(csv_path)
                    labels.append(filename.replace(".csv", ""))

        if not transects:
            print("No valid transect files found.")
            return

        # 1. Position scatter - transects only
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (transect, label) in enumerate(zip(transects, labels)):
            color = colors[i % len(colors)]
            ax.scatter(
                transect["longitude"],
                transect["latitude"],
                s=10,
                alpha=0.8,
                color=color,
                label=label,
            )

        ax.set_xlabel("Lon [deg]")
        ax.set_ylabel("Lat [deg]")
        ax.set_title("Position - Transects Only")
        ax.legend()
        fig.tight_layout()
        plt.show()

        # 2. 3D trajectory - transects only
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for i, (transect, label) in enumerate(zip(transects, labels)):
            color = colors[i % len(colors)]
            ax.plot(
                transect["longitude"],
                transect["latitude"],
                transect["altitude"],
                color=color,
                linewidth=3,
                alpha=0.9,
                label=label,
                marker="o",
                markersize=2,
            )

        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.set_zlabel("Alt")
        ax.set_title("3D Trajectory - Transects Only")
        ax.legend()
        fig.tight_layout()
        plt.show()
