# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************
import Eelume.PyPost as pp
import numpy as np

def calc_signal_time_delta(*signals : pp.Signal, time_delta_to_next = False) -> tuple:
    """Takes one or more Signals and calculates the time difference between timestamps for consecutive 
    data points. Returns a Signal for each input Signal.

    Args:
        *signals (Signal):   One or more Signals to analyse.
        time_delta_to_next (bool, optional): How to calculate time-axis.
                True: timestamp for a delta time value is timestamp for the first data point, e.g. timestamp means that the time delta is __from__ a data-point to the next.
                False: timestamp for a delta time value is timestamp for the second data point, e.g. timestamp means that the time delta is __from__ a data-point to the previous.
                Defaults to False.

    Returns:
        A tuple of signals with the time difference between data-points. Length will be equal to len(signals)
    """
    res = ()
    for signal in signals:        
        diff = np.diff(signal.axis)
        if time_delta_to_next == True:        
            times = signal.axis[0:-1]     # Use timestamp of first data point
        else:
            times = signal.axis[1:]       # Use timestamp of second data point

        res_signal = pp.Signal(times, diff, signal.name + "/time_diff")
        res += (res_signal,)

    return res

def rms(*signals : pp.Signal, window_length=100, detrend=True, high_pass_cut_off=None, output_spectral_density=False) -> tuple:
    """
    Calculates RMS of signals.

    Parameters
    ----------
    master : Signal
        The master signal which decides the sample rate
    *signals : Signal
        Input signals.
    window_length : int
        Window length in number of samples. The default is 100.
    detrend : bool, optional
        Will remove mean based trend from data before the RMS calculation. 
        High pass filtering must be turned off to use this option. The default is True.
    high_pass_cut_off : float
        Turns on high pass filtering of data before the RMS calculation. 
        Detrending must be off to use this option. The default is None.
    output_spectral_density : bool, optional
        Scales output by sqrt of signal bandwidht. The default is False.

    Returns
    -------
    tuple
        tuple with signals.

    """
    from scipy import signal as sp_signal

    y = ()

    for sig in signals:
        assert sig.ndim == 1, 'Signal must be 1-dimensional'    

        if high_pass_cut_off is not None:
            assert not detrend
            nyq = 0.5 * sig.sample_freq
            normal_cutoff = high_pass_cut_off / nyq
            b_coeff, a_coeff = sp_signal.butter(6, normal_cutoff, btype = "high", analog = False)
    
        nRms = sig.size // window_length
        rms = np.empty(nRms)
        axis = np.empty(nRms)
        
        for i in range(nRms):
            x = sig.data[i*window_length:i*window_length+window_length]
            if detrend:
                x = sp_signal.detrend(x)
                assert high_pass_cut_off is None
            elif high_pass_cut_off is not None:
                x = sp_signal.filtfilt(b_coeff, a_coeff, x)    
            rms[i] = np.sqrt(np.mean(np.square(x)))
            axis[i] = np.mean(sig.axis[i*window_length:i*window_length+window_length])
            if output_spectral_density:
                bw_low = high_pass_cut_off if high_pass_cut_off is not None else 0
                rms[i] /= np.sqrt(sig.sample_freq/2 - bw_low)

        y += (pp.Signal.create(sig.name + '/RMS', axis,rms),)
    
    return y