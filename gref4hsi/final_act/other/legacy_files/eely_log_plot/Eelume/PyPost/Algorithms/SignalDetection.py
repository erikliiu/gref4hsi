# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

import Eelume.PyPost as pp
import numpy as np

def calc_rate_of_change(*signals : pp.Signal, dt=None, calc_as_jump=False) -> tuple:
    """
    Calculates the rate of change on signals.

    Parameters
    ----------
    *signals : pp.Signal
        Input signals
    dt : float, optional
        Sample interval. The default is None and 1/signal.sample_freq will be used.
    calc_as_jump : bool
        Will not use dt to scale the result (pure jump calculation). The dt argument is ignored.

    Returns
    -------
    res : tuple
        A tuple of signals with the calculated rate of change.

    """
    res = ()
    suffix = '/rate_of_change'
    for sig in signals:
        if calc_as_jump:
            dt = 1
            suffix = '/jump'
        elif dt is None:
            assert sig.sample_freq is not None, 'Signal is not periodic, dt could not be None in this case'
            dt = 1/sig.sample_freq
                    
        res += (pp.Signal.create(sig.name + suffix,sig.axis[1:],np.diff(sig.data) / dt),)
    return res