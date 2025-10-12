# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

import utm
import Eelume.PyPost as pp
import numpy as np

def lat_long_2_utm(lat, long, handle_mixed_signed_latitude_as_postive=False):
    '''
    Converts from latitude, longitude (wgs84) to UTM.

    Parameters
    ----------
    lat : Scalar or numpy array
    long : Scalar or numpy array
    handle_mixed_signed_latitude_as_postive : bool
        The utm library does not accept mixed sign in 'lat' (if 'lat' is an numpy array).
        Set this flag to True to perform abs(lat). (Negative sign on latitude means that 
        we are on the southern hemisphere.)

    Returns
    -------
    Northing : Scalar or numpy array
    Easting : Scalar or numpy array
    zoneNumber : scalar
    zoneLetter : char
    '''
    
    is_numpy = lambda x : type(x).__module__ == np.__name__

    if is_numpy(lat) or is_numpy(long):
        if lat.size == 0 or long.size == 0:
            return np.empty(0),np.empty(0),0,0

    if handle_mixed_signed_latitude_as_postive:
        sign = np.sign(lat)
        if not (np.all(sign >= 0) or np.all(sign < 0)):
            lat = np.abs(lat)
    
    east,north,zoneNumber,zoneLetter = utm.from_latlon(lat, long)
    return north,east,zoneNumber,zoneLetter

def convert_signal2ned(*signals : pp.Signal, offset_ned_origin=None, input_is_radians=False, 
                      name_prefix=None, handle_mixed_signed_latitude_as_postive = False, convert_depth=False):
    '''
    Converts latitude and longitude, given as two Signal objects, to NED frame.

    Parameters
    ----------
    lat : pp.Signal
        Latitude input
    long : pp.Signal
        Longitude input
    offset_ned_origin : tuple
        The default is (None,None). Offsets the NE origin by a 
        North,East offset in metres. The default value will reset NE by first sample 
        such that first sample becomes (0,0) NE origin.
    handle_mixed_signed_latitude_as_postive : bool
        The utm library does not accept mixed sign in 'lat' (if 'lat' is an numpy array).
        Set this flag to True to perform abs(lat). (Negative sign on latitude means that 
        we are on the southern hemisphere.)
    convert_depth : bool
        Changes sign of depth. Set this to true for wgs84 signals.

    Returns
    -------
    (Signal,Signal)
        A tuple of the converted NE Signal objects

    '''

    assert len(signals) >= 2
    assert all(v.ndim == 1 for v in signals), 'Signals must be 1-dimensional'
    lat = signals[0].data.copy()
    long = signals[1].data.copy()
    assert lat.size == long.size

    if input_is_radians:
        lat *= 180/np.pi
        long *= 180/np.pi
    
    north,east,_,_ = lat_long_2_utm(lat,long,handle_mixed_signed_latitude_as_postive)
    
    if offset_ned_origin is None:
        north -= north[0]        
        east -= east[0]
    else:      
        north += offset_ned_origin[0]
        east += offset_ned_origin[1]  
    
    prefix = name_prefix + '/' if name_prefix is not None else '/'

    ret = ()
    ret += (pp.Signal(signals[0].axis, north, prefix + 'north_pos'),) 
    ret += (pp.Signal(signals[1].axis, east, prefix + 'east_pos'),)

    if len(signals) >= 3:
        ret +=  (pp.Signal(signals[2].axis, -signals[2].data if convert_depth else signals[2].data, prefix + 'depth_pos'),)  

    for i in range(3,len(signals)):
        ret += (signals[i],)
        
    return ret
