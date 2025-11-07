# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

from Eelume.PyPost.Signal import Signal
import numpy as np

def rad2deg(*signals : Signal) -> tuple:
    '''
    Converts signals from radians to degrees, returns the converted signals.
    Note that the signals must be in radians for this function to work as expected, will fail silently.
    '''    
    ret = ()
    for s in signals:
        ret += (Signal.create(s.name + '/deg',s.axis,s.data*180/np.pi),)
    return ret

def deg2rad(*signals : Signal) -> tuple:
    '''
    Converts signals from degrees to radians, returns the converted signals.
    Note that the signals must be in degrees for this function to work as expected, will fail silently.
    '''
    ret = ()
    for s in signals:
        ret += (Signal.create(s.name + '/rad',s.axis,s.data*np.pi/180),)
    return ret

def unwrap(*signals : Signal, period) -> tuple:
    """
    Unwraps signal with respect to period. 
    Convenient to avoid discontinuties in Euler angles in radians, e.g. unwrap yaw with period=2*np.pi.
    """
    ret = ()
    for sig in signals:
        ret += (Signal.create(sig.name + '/unwrap',sig.axis,np.unwrap(sig.data,period=period)),)
    return ret




