# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

from signal import signal
import numpy as np
from Eelume.PyPost.Signal import Signal


def resample_interpolate(master: Signal, *signals: Signal) -> tuple:
    """
    Resamples signals to same length as a master signal by linear interpolation.

    Parameters
    ----------
    master : Signal
        The master signal which decides the sample rate
    *signals : Signal
        The signals to resamples

    Returns
    -------
    tuple
        A tuple of the resampled signals. Length will be equal to len(signals)

    """
    ret = ()
    for s in signals:
        ret += (
            Signal.create(s.name, master.axis, np.interp(master.axis, s.axis, s.data)),
        )
    return ret


def resample_fft(*signals: Signal, N: int) -> tuple:
    """
    Resamples signals to N number of samples using scipy.signal.resample (Fourier method).
    Be careful to use this function with signal that have discontinouties, such as steps.
    Note that all the signals must start and end approximatly at the same timestamp for this
    function to work as intended.

    Parameters
    ----------
    *signals : Signal
        Input signals
    N : int
        Resamples

    Returns
    -------
    ret : tuple
        Returns a tuple of signals with the resampled results.

    """
    from scipy import signal as sp_signal

    ret = ()
    for s in signals:
        axis = np.linspace(s.axis[0], s.axis[-1], N)
        data = sp_signal.resample(s.data, N)
        ret += (Signal(axis, data, s.name),)
    return ret


def resample_to_same_length(
    *signals: Signal, length: str = "lowest", method: str = "interpolate"
) -> tuple:
    """
    Resamples all signals to the same number of samples using the scipy fft resample function.
    Be careful to use this function with signal that have discontinouties, such as steps.
    Note that all the signals must start and end approximatly at the same timestamp for this
    function to work as intended.

    Parameters
    ----------
    *signals : Signal
        The input signals to resample
    length : string
        The length of the resampled signals must either be specified as 'lowest' or 'highest'.
        'lowest' resamples the signals to the same length as the signal with fewest samples, while 'highest'
        the same length as the signal with most samples.
        The default is 'lowest'.
    method : string
        Method used for interpolation. Specify either 'fft' or 'interpolate',
        which will call PyPost.resample_interpolate or PyPost.resample_fft, respectively.

    Returns
    -------
    ret : tuple
        DESCRIPTION. A tuple of resampled signals.

    """
    assert type(length) is str

    if length.lower() == "lowest":
        N = 100000000
        for s in signals:
            if s.size < N:
                master = s
            N = min(N, s.size)
    elif length.lower() == "highest":
        N = 0
        for s in signals:
            if s.size > N:
                master = s
            N = max(N, s.size)
    else:
        raise Exception("length argument '{}' is not valid".format(length))

    if method.lower() == "fft":
        ret = resample_fft(*signals, N=N)
    elif method.lower() == "interpolate":
        ret = resample_interpolate(master, *signals)
    else:
        raise Exception("method argument '{}' is not valid".format(method))

    return ret
