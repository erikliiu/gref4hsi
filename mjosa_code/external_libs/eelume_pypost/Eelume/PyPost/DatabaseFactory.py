# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

"""
Overview of database paths pr 01.06.2022:
    
Pose that is used by control system (source is either xsens or primarly Sunstone):
    '/navigation/robot_position/lat_deg',
    '/navigation/robot_position/lon_deg',
    '/navigation/robot_position/depth',
    '/navigation/roll',
    '/navigation/pitch',
    '/navigation/heading',
    
Pose controller references:
    Not in db
    
Thruster forces and moments (output from motion control)
    Not in db
    
Thruster commands from thruster allocation:
    /thrusters/refs/0x51
    /thrusters/refs/0x52
    /thrusters/refs/0x53
    /thrusters/refs/0x55
    /thrusters/refs/0x71
    /thrusters/refs/0x72
    /thrusters/refs/0x73
    /thrusters/refs/0x74

Thruster measured thrust (estimated by FW):
    /thrusters/measure/0x51/thrust
    /thrusters/measure/0x52/thrust
    /thrusters/measure/0x53/thrust
    /thrusters/measure/0x55/thrust
    /thrusters/measure/0x71/thrust
    /thrusters/measure/0x72/thrust
    /thrusters/measure/0x73/thrust
    /thrusters/measure/0x74/thrust

Xsense fallback IMU angular position:
    /dynamics/Eul_N_B_roll
    /dynamics/Eul_N_B_pitch
    /dynamics/Eul_N_B_yaw

Sunstone angular position:    
    /sensors/sunstone/navigation/log_euler_deg/x
    /sensors/sunstone/navigation/log_euler_deg/y
    /sensors/sunstone/navigation/log_euler_deg/z

DVL velocity:
    /sensors/dvl/velocity/D__x
    /sensors/dvl/velocity/D__y
    /sensors/dvl/velocity/D__z
    
ALP meldinger:
    /acoustic_link/ALP/received_messages (counter for recived messages, incremented for each recived message)
    /sensors/cnode/LUP/received_messages (counter for recived messages from uPAP)


"""

import Eelume.PyPost as pp
import numpy as np


def remove_head_zeros(x: pp.Signal, limit=np.finfo(np.float32).eps):
    """
    Removes data samples that are zero and returns the result.
    """
    if x.size > 0:
        index = np.argmax(np.abs(x.data) > limit)
        x.remove_head(index)
    return x


def get_pose_signals(
    dataBase: pp.DatabaseHandler,
    start_time=None,
    end_time=None,
    reset_time_stamp_to_zero=False,
    convert2ned=False,
    remove_lat_long_samples_before_first_pos_fix=True,
) -> tuple:
    """
    Factory that loads the robot pose.
    NB! Depth sensor and not Sunstone is used for depth

    Parameters
    ----------
    dataBase : pp.DatabaseHandler
    start_time : float or a datatime object. The default is None.
    end_time : float or a datatime object. The default is None.
    reset_time_stamp_to_zero : bool. The default is False.
    convert2ned : bool. The default is False.
    remove_lat_long_samples_before_first_pos_fix: bool
        Option to removes long/lat samples that are zeros.
        It is reasonable to assume that long/lat with a value of zero is
        data that was sampled before the first position fix was available.

    Returns
    -------
    A tuple of Signal
    """

    pose = list(
        dataBase.get_multiple_signals(
            "/navigation/robot_position/lat_deg",
            "/navigation/robot_position/lon_deg",
            "/envsens/depth",  #'/navigation/robot_position/depth',
            "/navigation/roll",
            "/navigation/pitch",
            "/navigation/heading",
            start_time=start_time,
            end_time=end_time,
            reset_time_stamp_to_zero=reset_time_stamp_to_zero,
        )
    )

    if remove_lat_long_samples_before_first_pos_fix:
        pose[0] = remove_head_zeros(pose[0], 1e-3)
        pose[1] = remove_head_zeros(pose[1], 1e-3)

    if convert2ned:
        pose[0], pose[1] = pp.convert_signal2ned(pose[0], pose[1])

    return tuple(pose)


def get_linear_pos_signals(
    dataBase: pp.DatabaseHandler,
    start_time=None,
    end_time=None,
    reset_time_stamp_to_zero=False,
    convert2ned=False,
    remove_lat_long_samples_before_first_pos_fix=True,
) -> tuple:
    """
    Factory function that loads the robot linear positions.
    NB! Depth sensor and not Sunstone is used for depth

    Parameters
    ----------
    dataBase : pp.DatabaseHandler
    start_time : float or a datetime object. The default is None.
    end_time : float or a datetime object. The default is None.
    reset_time_stamp_to_zero : bool. The default is False.
    convert2ned : bool. The default is False.
    remove_lat_long_samples_before_first_pos_fix: bool
        Option to removes long/lat samples that are zeros.
        It is reasonable to assume that long/lat with a value of zero is
        data that was sampled before the first position fix was available.

    Returns
    -------
    A tuple of Signal
    """

    # pos is a list of signals, one signal for each of 3 specified paths
    pos = list(
        dataBase.get_multiple_signals(
            "/navigation/robot_position/lat_deg",
            "/navigation/robot_position/lon_deg",
            "/envsens/depth",  #'/navigation/robot_position/depth',
            start_time=start_time,
            end_time=end_time,
            reset_time_stamp_to_zero=reset_time_stamp_to_zero,
        )
    )

    """
    eg
    before: 
    pos[0] = [0.0, 0.0, 0.0005, 60.123, 60.124]  # Latitude
    pos[1] = [0.0, 0.0, 0.0003, 10.567, 10.568]  # Longitude

    after:
    pos[0] = [60.123, 60.124]  # Latitude (Real data only)
    pos[1] = [10.567, 10.568]  # Longitude (Real data only)

    initial values are removed since they are zeros at the start .> when INS syst start -> dont know exact position, need some movement/calibration to correct itself -> during this time the pos. data might be zero or very small -> we remove this 
    """
    if remove_lat_long_samples_before_first_pos_fix:
        pos[0] = remove_head_zeros(
            pos[0], 1e-3
        )  # 1e-3 is the limit, if below it is removed
        pos[1] = remove_head_zeros(pos[1], 1e-3)

    # if convert2ned is True, then convert the latitude and longitude (degrees) to NED frame (meteres)
    if convert2ned:
        pos[0], pos[1] = pp.convert_signal2ned(pos[0], pos[1])

    return tuple(pos)


def get_attitude_signals(
    dataBase: pp.DatabaseHandler,
    start_time=None,
    end_time=None,
    reset_time_stamp_to_zero=False,
) -> tuple:
    """
    Factory function that loads the robot attitude.

    Parameters
    ----------
    dataBase : pp.DatabaseHandler
    start_time : float or a datatime object. The default is None.
    end_time : float or a datatime object. The default is None.
    reset_time_stamp_to_zero : bool. The default is False.

    Returns
    -------
    A tuple of Signal

    """
    return dataBase.get_multiple_signals(
        "/navigation/roll",
        "/navigation/pitch",
        "/navigation/heading",
        start_time=start_time,
        end_time=end_time,
        reset_time_stamp_to_zero=reset_time_stamp_to_zero,
    )
