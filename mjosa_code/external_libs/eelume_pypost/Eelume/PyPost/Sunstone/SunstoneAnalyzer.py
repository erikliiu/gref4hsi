# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

import Eelume.PyPost as pp
import numpy as np
from Eelume.PyPost.CoordinateConversion import lat_long_2_utm
import datetime

"""
Class that contains method for plotting and loading a selection of Sunstone data
"""
class SunstoneAnalyzer(object):
    def __init__(self, digest_file : str, *cp_database_files : str, 
                start_time : datetime = None, end_time : datetime = None, north_east_origin :tuple = None):
        """
        Constructor

        Parameters
        ----------
        digest_file : str
            The digest filename found in SunstoneLog.
        *cp_database_files : str
            The cp filenames found in the SunstoneLog. 
            Note that it's optional to pass any arguments here. 
            Some of the functionality in SunstoneAnalyzer will not work if 
            cp filenames are not given.
        start_time : datetime
            Start time for analysis. The default is None.
        end_time : datetime
            End time for analysis. The default is None.
        north_east_origin : tuple
            The north/east origin given as longitude and latitude coordinates. 
            The default is None and data will be plotted as longitude/latitude,
            else data is converted to UTM.

        """
        self.__load_digest_file(digest_file)    
        self.__load_cp_files(*cp_database_files)

        self.set_time_interval(start_time,end_time)
        self.set_ned_origin(north_east_origin)
        self.__set_time_axis_plot_format()

    def set_time_interval(self,start_time=None, end_time=None):
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

    def set_ned_origin(self,north_east_origin):
        """
        Changes the north/east origin and reloads data.

        Parameters
        ----------
        north_east_origin : tuple
            The north/east origin given as longitude and latitude coordinates. 
            The default is None and data will be plotted as longitude/latitude.

        """
        if north_east_origin is not None:
            north,east,_,_ = lat_long_2_utm(north_east_origin[0],north_east_origin[1])
            self.__offset_ned_origin = (-north,-east)
        else:
            self.__offset_ned_origin = None
        self.__reload()

    def has_cp_data(self):
        """
        Returns true if cp data is available
        """
        return self.__cp_handler is not None

    def plot_pose(self, figure_number=None, include_sensors=True, **kwargs):
        """
        Plots pose for navigation data and optionally navigation sensors.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        include_sensors : bool
            Flag which includes navigation sensors in the plot (GPS, DVL, etc) in the plot. The default is True.
        **kwargs : 
            key-value argument which is passed directly to the 
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.

        """
        if include_sensors:
            dummy = pp.Signal.create('_',np.array([]),np.array([]))
            nav = self.load_nav_pose()
            gps = self.load_gps_pos()
            acoustic = self.load_acoustic_pos()
            depth = self.load_depth_sensor()
            
            signals = (nav[0],gps[0],acoustic[0],nav[1],gps[1],acoustic[1],nav[2],dummy,depth,
                        nav[3],dummy,dummy,nav[4],dummy,dummy,nav[5],dummy,dummy)

            n_signals_pr_subplot = 3
        else:
            signals = self.load_nav_pose()
            n_signals_pr_subplot = 1

        pp.plot_line(*signals,figure_number=figure_number,n_signals_pr_subplot=n_signals_pr_subplot,title='Sunstone pose',
        ylabel=self.__get_pose_ylabel(),time_axis_format=self.__time_axis_plot_format, **kwargs)

    def plot_position_scatter(self, figure_number=None, include_acoustics=False, include_gps=False, equal_axis=False, **kwargs):
        """
        Creates a scatter plot of the horizontal position.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        include_acoustics : bool
            Flag which includes acoustic position fixes in the plot. The default is False.
        include_gps : bool
            Flag which includes GPS position fixes in the plot. The default is False.
        equal_axis : bool
            Axis scaling for x and y axis will be equal. The default is False.
        **kwargs : 
            key-value argument which is passed directly to the 
            matplotlib.pyplot.scatter function.

        """
        pose = self.load_nav_pose()
        signals = (pose[1],pose[0])
        
        if include_acoustics:
            acoustics = self.load_acoustic_pos()
            signals += (acoustics[1],acoustics[0])
        if include_gps:
            gps = self.load_gps_pos()
            signals += (gps[1],gps[0])
        
        if self.__offset_ned_origin is not None:
            xlabel = 'East [m]'
            ylabel = 'North [m]'
        else:
            xlabel = 'longitude [deg]'
            ylabel = 'latitude [deg]'

        pp.plot_scatter2d(*signals, figure_number=figure_number, title='Position', xlabel=xlabel, ylabel=ylabel, equal_axis=equal_axis,**kwargs)

    def plot_velocity_body(self, figure_number=None, include_sensors=True, **kwargs):     
        """
        Plots linear and angular velocity in body frame.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        include_sensors : bool
            Flag which includes navigation sensors in the plot (DVL) in the plot. The default is True.
        **kwargs : 
            key-value argument which is passed directly to the 
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.

        """           
        if include_sensors:
            dummy = pp.Signal.create('_',np.array([]),np.array([]))
            nav = self.load_nav_vel()
            rel_vel = self.load_dvl_rel_vel_body()
            abs_vel = self.load_dvl_abs_vel_body()
            
            signals = (rel_vel[0],abs_vel[0],nav[0],rel_vel[1],abs_vel[1],nav[1],rel_vel[2],abs_vel[2],nav[2],
                            nav[3],dummy,dummy,nav[4],dummy,dummy,nav[5],dummy,dummy)

            n_signals_pr_subplot = 3
        else:
            signals = self.load_nav_vel()
            n_signals_pr_subplot = 1
                
        ylabel = ('Surge [m/s]','Sway [m/s]', 'Heave [m/s]', 'Roll [rad/s]', 'Pitch [rad/s]', 'Yaw [rad/s]')
        pp.plot_line(*signals,figure_number=figure_number,n_signals_pr_subplot=n_signals_pr_subplot,title='Sunstone velocity body',
                    ylabel=ylabel,time_axis_format=self.__time_axis_plot_format,**kwargs)

    def plot_pose_jumps(self, figure_number=None, include_attitude=True, **kwargs):
        """
        Plots jumps in navigation signal from one sample to another

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        include_attitude : bool
            Flag which includes attitude as part of the plot. The default is True.
        **kwargs : 
            key-value argument which is passed directly to the 
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.

        """
        pose = self.load_nav_pose()
        signals = pose[:3]

        if include_attitude:
            signals += pp.unwrap(pose[3],period=np.pi)
            signals += pp.unwrap(pose[4],period=np.pi)
            signals += pp.unwrap(pose[5],period=2*np.pi)

        jumps = pp.calc_rate_of_change(*signals,calc_as_jump=True)
        pp.plot_line(*jumps,figure_number=figure_number,n_signals_pr_subplot=1,title='Sunstone pose jumps',
                    ylabel=self.__get_pose_ylabel()[:len(signals)],time_axis_format=self.__time_axis_plot_format,**kwargs)

    def plot_std_dev_pose(self, figure_number=None, **kwargs):
        """
        Plots the standard deviation of the pose navigation data.

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
        ylabel = ('Latitude [m]','Longitude [m]','Depth [m]','Roll [deg]','Pitch [deg]','Yaw [deg]')
        pp.plot_line(*self.load_std_dev_pose(),figure_number=figure_number,n_signals_pr_subplot=1,title='Sunstone pose standard deviation',
                    ylabel=ylabel,time_axis_format=self.__time_axis_plot_format,**kwargs)

    def plot_std_dev_vel_body(self, figure_number=None, **kwargs):
        """
        Plots the standard deviation of the velocity navigation data (surge, sway, yaw).

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs : TYPE
            key-value argument which is passed directly to the 
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.

        """
        ylabel = ('Surge [m/s]', 'Sway [m/s]','Heave [m/s]')
        pp.plot_line(*self.load_std_dev_vel_body(),figure_number=figure_number,n_signals_pr_subplot=1,title='Sunstone velocity standard deviation',
                    ylabel=ylabel,time_axis_format=self.__time_axis_plot_format,**kwargs)

    def plot_acceleration(self, figure_number=None, **kwargs):
        """
        Plots the acceleration navigation data.

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
        ylabel = ('Surge [m/s^2]','Sway [m/s^2]','Heave [m/s^2]')
        pp.plot_line(*self.load_nav_acc(),figure_number=figure_number,n_signals_pr_subplot=1,title='Sunstone acceleration',
                    ylabel=ylabel,time_axis_format=self.__time_axis_plot_format,**kwargs)

    def plot_imu_raw(self, figure_number=None, **kwargs):
        """
        Plots IMU raw data (accelerometers and gyros)

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
        ylabel = ('Surge [m/s^2]','Sway [m/s^2]','Heave [m/s^2]','Roll [rad/s]', 'Pitch [rad/s]', 'Yaw [rad/s]')
        
        imu_raw = self.load_imu_raw()
        rms = pp.rms(*imu_raw,window_length=200)
        signals = pp.interleave_signals(imu_raw,rms)

        pp.plot_line(*signals,figure_number=figure_number,n_signals_pr_subplot=2,title='Sunstone IMU raw (acceleration and gyro rate)',
                    ylabel=ylabel,time_axis_format=self.__time_axis_plot_format,**kwargs)

    def plot_imu_raw_acc_norm(self, figure_number=None, **kwargs):
        """
        Plots the norm of IMU raw data accelerometers. 
        Convinient to verify accelerometer drift, where the norm is expected to be close to 9.81 except for the short
        periods of robot acceleration.

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

        imu_raw = self.load_imu_raw()
        abs_data = np.sqrt(imu_raw[0].data**2 + imu_raw[1].data**2 + imu_raw[2].data**2)
        abs_sig = pp.Signal.create('/imu_raw/acc/abs',imu_raw[0].axis,abs_data)
        pp.plot_line(abs_sig,figure_number=figure_number,title='Sunstone norm of IMU raw acceleration measurements',
                    time_axis_format=self.__time_axis_plot_format,**kwargs)     

    def plot_imu_raw_roll_pitch_from_g_vector(self, figure_number=None, **kwargs):
        """
        Plots roll/pitch calculated from g-vector based on IMU raw data accelerometers and roll/pitch from navigation.
        Convinient to analyze Sunstone drift issues.

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
        acc = self.load_imu_raw()[:3]
        roll_g = pp.rad2deg(pp.Signal.create('/from_g_vec/roll', acc[0].axis, -np.arctan2(acc[1].data, -acc[2].data)))
        pitch_g = pp.rad2deg(pp.Signal.create('/from_g_vec/pitch', acc[0].axis, np.arctan2(acc[0].data, np.sqrt(acc[1].data**2 + acc[2].data**2))))
        g_att = (*roll_g,*pitch_g)

        signals = pp.interleave_signals(self.load_nav_pose()[3:5],g_att)
        ylabel = ('Roll [deg]', 'Pitch [deg]')
        pp.plot_line(*signals,figure_number=figure_number,n_signals_pr_subplot=2,ylabel=ylabel, title='Sunstone IMU g-vector calculated roll/pitch',
                        time_axis_format=self.__time_axis_plot_format, **kwargs)
        
    def plot_kf_status_pose(self, figure_number=None, use_gps_as_sensor=False, **kwargs):
        """
        Plots the pose aiding used in the Sunstone kalman filter along with the navigation solution data.

        Parameters
        ----------
        figure_number : int
            Sets the plot window figure number.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        use_gps_as_sensor : bool
            Will use GPS as sensor if True, else acoustics.
        **kwargs : TYPE
            key-value argument which is passed directly to the 
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.

        """
        self.__assert_cp()

        nav_pos = self.load_nav_pose()

        if use_gps_as_sensor:
            sensor_data = self.load_gps_pos()
        else:
            sensor_data = self.load_acoustic_pos()
        
        hori_pos = self.__kf_status_creator.create_horisontal_position_aiding(nav_pos[0:2], sensor_data)
        depth = self.__kf_status_creator.create_depth_aiding((nav_pos[2],),(self.load_depth_sensor(),))
        attitude = self.__kf_status_creator.create_attitude_aiding(nav_pos[3:])

        ylabel = self.__get_pose_ylabel()
        pp.plot_line(*hori_pos,*depth,*attitude,figure_number=figure_number,n_signals_pr_subplot=3,title='Sunstone pose KF-status',
                    ylabel=ylabel, time_axis_format=self.__time_axis_plot_format, marker='.', **kwargs)

    def plot_kf_status_abs_vel_body(self, figure_number=None, **kwargs):
        """
        Plots absolute velocity aiding used in the Sunstone kalman filter along with the navigation solution data.

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
        self.__assert_cp()

        nav_vel = self.load_nav_vel()
        abs_vel = self.__kf_status_creator.create_abs_vel_aiding(nav_vel[:3],self.load_dvl_abs_vel_body())
        ylabel = ('Surge [m/s]', 'Sway [m/s]','Heave [m/s]')
        pp.plot_line(*abs_vel,figure_number=figure_number,n_signals_pr_subplot=3,title='Sunstone absolute velocity KF-status',
                    ylabel=ylabel,time_axis_format=self.__time_axis_plot_format,marker='.',**kwargs)

    def plot_kf_status_rel_vel_body(self, figure_number=None, **kwargs):
        """
        Plots relative velocity aiding used in the Sunstone kalman filter along with the navigation solution data.

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
        self.__assert_cp()

        nav_vel = self.load_nav_vel()
        abs_vel = self.__kf_status_creator.create_rel_vel_aiding(nav_vel[:3],self.load_dvl_rel_vel_body())
        ylabel = ('Surge [m/s]', 'Sway [m/s]','Heave [m/s]')
        pp.plot_line(*abs_vel,figure_number=figure_number,n_signals_pr_subplot=3,title='Sunstone relative velocity KF-status',
                    ylabel=ylabel,time_axis_format=self.__time_axis_plot_format,marker='.',**kwargs)

    def plot_sensor_update_interval_pos(self, figure_number_update_rate=None, figure_number_histogram=None, **kwargs):
        """
        Plots sensor update interval for position sensors.

        Parameters
        ----------
        figure_number_update_rate : int
            Sets the plot window figure number for timeseries plot.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        figure_number_histogram : int
            Sets the plot window figure number for the histogram.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs : TYPE
            key-value argument which is passed directly to the 
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.

        """
        signals = (self.load_acoustic_pos()[0],)
        signals += (self.load_depth_sensor(),)
        updates = self.__calc_update_interval(*signals)

        label = ('Acoustic pos [s]','Depth [s]')
        pp.plot_line(*updates,figure_number=figure_number_update_rate, n_signals_pr_subplot=1, graph_type='line', 
                    time_axis_format=self.__time_axis_plot_format, marker='.', title='Position update interval', ylabel=label, **kwargs)
        pp.plot_histogram(*updates,figure_number=figure_number_histogram,title='Histogram of position update interval', xlabel=label, **kwargs)

    def plot_sensor_update_interval_vel(self, figure_number_update_rate=None, figure_number_histogram=None, **kwargs):
        """
        Plots sensor update interval for velocity sensors.

        Parameters
        ----------
        figure_number_update_rate : int
            Sets the plot window figure number for timeseries plot.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        figure_number_histogram : int
            Sets the plot window figure number for the histogram.
            If a figure with the given number exists, the content in the window will
            be overwritten, else a new figure window will be created. The default is None.
        **kwargs : 
            key-value argument which is passed directly to the 
            matplotlib.axes.Axes.plot function. Controls how each line is plotted.
        """
        signals = self.load_dvl_rel_vel_body()
        signals += self.load_dvl_abs_vel_body()
        updates = self.__calc_update_interval(*signals)
        
        label = ('Rel vel surge [s]', 'Rel vel sway [s]', 'Rel vel heave [s]', 'Abs vel surge [s]', 'Abs vel sway [s]', 'Abs vel heave [s]')
        pp.plot_line(*updates,figure_number=figure_number_update_rate, n_signals_pr_subplot=1, graph_type='line',
                    time_axis_format=self.__time_axis_plot_format, title='DVL update interval', ylabel=label, marker='.', **kwargs)
        pp.plot_histogram(*updates,figure_number=figure_number_histogram,title='Histogram of velocity update interval',xlabel=label,**kwargs)
        
    def load_nav_pose(self) -> tuple:
        """
        Loads navigation pose

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__nav_pose is None:
            lin_pos = list(self.__main_handler.get_multiple_signals('Navigation','latitude','longitude','depth',start_time=self.__start_time,end_time=self.__end_time))

            if self.__offset_ned_origin is not None:
                lin_pos = pp.convert_signal2ned(*lin_pos,offset_ned_origin=self.__offset_ned_origin,name_prefix='pos',input_is_radians=True,convert_depth=False)
            else:
                lin_pos[0], lin_pos[1] = pp.rad2deg(lin_pos[0], lin_pos[1])
            
            attitude = self.__main_handler.get_multiple_signals('Navigation','roll','pitch','heading',start_time=self.__start_time,end_time=self.__end_time)
            attitude = pp.rad2deg(*attitude)

            self.__nav_pose = tuple(lin_pos) + attitude
        
        return self.__nav_pose

    def load_acoustic_pos(self) -> tuple:
        """
        Loads acoustic position data

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__acoustic_pos is None:
            self.__acoustic_pos = self.__main_handler.get_multiple_signals('uPAP', 'latitude_rad','longitude_rad',start_time=self.__start_time,end_time=self.__end_time)
            if self.__offset_ned_origin is not None:
                self.__acoustic_pos = pp.convert_signal2ned(*self.__acoustic_pos,input_is_radians=True,name_prefix='uPap',offset_ned_origin=self.__offset_ned_origin)
            else:
                self.__acoustic_pos = pp.rad2deg(*self.__acoustic_pos)
        return self.__acoustic_pos

    def load_depth_sensor(self):
        """
        Loads depth sensor data

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__depth_sensor_pos is None:
            self.__depth_sensor_pos = self.__main_handler.get_signal('depth_sensor', 'depth', start_time=self.__start_time,end_time=self.__end_time)
        return self.__depth_sensor_pos

    def load_gps_pos(self):
        """
        Loads gps position data

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__gps_pos is None:
            self.__gps_pos = self.__main_handler.get_multiple_signals('GPS', 'latitude','longitude',start_time=self.__start_time,end_time=self.__end_time)
            if self.__offset_ned_origin is not None:
                self.__gps_pos = pp.convert_signal2ned(*self.__gps_pos,input_is_radians=True,name_prefix='GPS',offset_ned_origin=self.__offset_ned_origin)     
            else:
                self.__gps_pos = pp.rad2deg(*self.__gps_pos)  
        return self.__gps_pos

    def load_nav_vel(self):
        """
        Loads navigation velocity

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__nav_vel_body is None:
            self.__nav_vel_body = self.__main_handler.get_multiple_signals('Navigation','xVelocityB','yVelocityB','zVelocityB','xAngRateB','yAngRateB','zAngRateB',start_time=self.__start_time,end_time=self.__end_time)
        return self.__nav_vel_body

    def load_dvl_rel_vel_body(self):
        """
        Loads relative (water track) DVL velocty in body frame.

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__dvl_rel_vel_body is None:
            self.__dvl_rel_vel_body = self.__main_handler.get_multiple_signals('dvl_wtr', 'xVelBody_mps','yVelBody_mps','zVelBody_mps',start_time=self.__start_time,end_time=self.__end_time)
        return self.__dvl_rel_vel_body

    def load_dvl_abs_vel_body(self):
        """
        Loads absolute (bottom track) DVL velocity in body frame.

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__dvl_abs_vel_body is None:
            self.__dvl_abs_vel_body = self.__main_handler.get_multiple_signals('dvl_btm', 'xVelBody_mps','yVelBody_mps','zVelBody_mps',start_time=self.__start_time,end_time=self.__end_time)
        return self.__dvl_abs_vel_body

    def load_nav_acc(self):
        """
        Loads navigation acceleration.

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__nav_acc is None:
            self.__nav_acc = self.__main_handler.get_multiple_signals('Navigation','xAccB','yAccB','zAccB',start_time=self.__start_time,end_time=self.__end_time)
        return self.__nav_acc

    def load_std_dev_pose(self):
        """
        Loads navigation pose standard deviation.

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__std_dev_pose is None:
            lin = self.__main_handler.get_multiple_signals('Navigation', 'latitudeStdDev','longitudeStdDev','depthStdDev',start_time=self.__start_time,end_time=self.__end_time)
            att = self.__main_handler.get_multiple_signals('Navigation', 'rollStdDev','pitchStdDev','headingStdDev',start_time=self.__start_time,end_time=self.__end_time)
            att = pp.rad2deg(*att)
            self.__std_dev_pose = lin + att
        return self.__std_dev_pose

    def load_std_dev_vel_body(self):
        """
        Loads navigation velocity standard deviation.

        Returns
        -------
        tuple with Signal objects
            
        """
        if self.__std_dev_vel_body is None:
            self.__std_dev_vel_body = self.__main_handler.get_multiple_signals('Navigation','xVelocityBStdDev','yVelocityBStdDev','zVelocityBStdDev',start_time=self.__start_time,end_time=self.__end_time)
        return self.__std_dev_vel_body

    def load_imu_raw(self):
        """
        Loads IMU raw data for accelerometers and gyros.

        Returns
        -------
        tuple with Signal objects
            
        """
        self.__assert_cp()
        if self.__imu_raw is None:
            self.__imu_raw = self.__cp_handler.get_multiple_signals('imu','deltaVelocityBodyX_mps','deltaVelocityBodyY_mps','deltaVelocityBodyZ_mps',
                                                                    'deltaRotationBodyX_rad','deltaRotationBodyY_rad','deltaRotationBodyZ_rad',
                                                                    start_time=self.__start_time,end_time=self.__end_time)
            for i in range(6):
                self.__imu_raw[i].data *= 200 # scales by Fs to get unit in m/s^2 and rad/s
            self.__imu_raw[0].name = '/imu_raw/acc_surge'
            self.__imu_raw[1].name = '/imu_raw/acc_sway'
            self.__imu_raw[2].name = '/imu_raw/acc_heave'
            self.__imu_raw[3].name = '/imu_raw/gyro_roll'
            self.__imu_raw[4].name = '/imu_raw/gyro_pitch'
            self.__imu_raw[5].name = '/imu_raw/gyro_yaw'

        return self.__imu_raw

    def __reload(self):
        # Lazy loading of signals later 
        self.__nav_pose = None
        self.__acoustic_pos = None
        self.__depth_sensor_pos = None
        self.__gps_pos = None
        self.__nav_vel_body = None
        self.__dvl_rel_vel_body = None
        self.__dvl_abs_vel_body = None
        self.__nav_acc = None
        self.__std_dev_pose = None
        self.__std_dev_vel_body = None
        self.__imu_raw = None

    def __get_pose_ylabel(self):
        if self.__offset_ned_origin is None:
            ylabel = ('Latitude [deg]','Longitude [deg]','Depth [m]','Roll [deg]','Pitch [deg]','Yaw [deg]')
        else:
            ylabel = ('North [m]','East [m]','Down [m]','Roll [deg]','Pitch [deg]','Yaw [deg]')
        return ylabel

    def __load_digest_file(self,filename):
        self.__main_handler = pp.SunstoneFileHandler(filename,
            hdf5_groups=(pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/CalculatedData/NavigationSolutionData/slotID0/','dataValidTime','Navigation'),
                        pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/SensorData/GenericPositionData/slotID0/','acquisitionTimestamp','uPAP'),
                        pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/SensorData/DvlXYZData/Dvl4BeamData/Dvl4BeamWtrTrackData/slotID0/','acquisitionTimestamp','dvl_wtr'),
                        pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/SensorData/DvlXYZData/Dvl4BeamData/Dvl4BeamBtmTrackData/slotID0/','acquisitionTimestamp','dvl_btm'),
                        pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/SensorData/PressureSensorData/slotID4/','acquisitionTimestamp','depth_sensor'),
                        pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/SensorData/MotionSensorData/slotID3/','acquisitionTimestamp','MGC'),
                        pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/SensorData/GPSSensorData/slotID4/','acquisitionTimestamp','GPS'),
              ))

    def __load_cp_files(self,*filenames):
        if len(filenames) > 0:
            self.__cp_handler = pp.SunstoneFileHandler(*filenames,
            hdf5_groups=(                    
                    pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/KFStatusData/slotID0/','databaseStoreTimestamp','KFStatus'),                       
                    pp.SunstoneFileHandler.HdfGroupNamer('DatagramBase/SensorData/ImuData/ImuDataGeneric/slotID0','acquisitionTimestamp','imu')
            ))
            self.__kf_status_creator = pp.SunstoneKfAidingCreator(self.__main_handler,self.__cp_handler)
        else:
            self.__cp_handler = None
            self.__kf_status_creator = None

    def __assert_cp(self):
        assert self.__cp_handler is not None, 'CP files must be loaded to use this feature'    

    def __calc_update_interval(self,*signals):
        updates = ()
        for sig in signals:
            updates += (pp.Signal.create(sig.name + '/update_interval',sig.axis[1:],np.diff(sig.axis)),)   
        return updates

    def __set_time_axis_plot_format(self):
        if self.__start_time is not None and self.__end_time is not None:
                assert self.__start_time.tzinfo == self.__end_time.tzinfo

        time = None
        if self.__start_time is not None:
            time = self.__start_time
        if self.__end_time is not None:
            time = self.__end_time

        if time is not None:
            if time.tzinfo == datetime.timezone.utc:
                self.__time_axis_plot_format = 'datetime_utc'
            elif time.tzinfo is None:
                self.__time_axis_plot_format = 'datetime_local'
            else:
                raise Exception('Not able to set time axis plot format')
        else:
            self.__time_axis_plot_format = 'datetime_utc' # defaults to utc plot