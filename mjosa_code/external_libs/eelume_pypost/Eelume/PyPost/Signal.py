# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************
import numpy as np
import math
from datetime import datetime
import numbers


class Signal(object):
    """
    A Signal represents the full output from a specific sensor over time.

    Example:
    --------
    A speedometer in a car records speed while driving from **time 0 to time 100**.
    - The **axis** is the time values (each time instant).
    - The **data** is the sensor readings (e.g., velocity at each time).

    A Signal stores both **axis (time)** and **data (sensor values)** together,
    allowing trimming (start and end time) and resetting time to zero.
    """

    @classmethod
    def create(
        cls,
        name: str,
        axis,
        data,
        start_time=None,
        end_time=None,
        reset_time_stamp_to_zero=False,
    ):
        """
        Factory for creating a signal object.

        Parameters
        ----------
        name : str
            Signal name.
        axis : np.array
            Array that holds axis ticks, e.g. seconds if timeseries.
        data : np.array
            The data in signal
        start_time : double or a datatime object.
            The default is None. Removes data before this time.
        end_time : double or a datatime object.
            The default is None. Removes data after this time.
        reset_time_stamp_to_zero : bool.
            The default is False. Resets timestamps in such a way
            that the first sample starts at zero.

        Returns
        -------
        signal : Signal
            The Signal object containing the specified axis and data.

        """

        assert type(name) is str, "name argument must be str"

        if start_time is not None:
            index_remove_before = cls.__find_index_from_timestamp(start_time, axis)
            axis = axis[index_remove_before:]
            data = data[..., index_remove_before:]

        if end_time is not None:
            index_remove_after = cls.__find_index_from_timestamp(end_time, axis)
            axis = axis[:index_remove_after]
            data = data[..., :index_remove_after]

        signal = cls(axis, data, name)

        if reset_time_stamp_to_zero:
            signal.reset_time_axis_to_zero()

        return signal

    @classmethod
    def create_empty(cls, name: str):
        """Creates empty signal signal object

        Args:
            name (str): Signal name

        Returns:
            A signal object
        """
        return cls(np.array([]), np.array([]), name)

    def __init__(self, axis: np.array, data: np.array, name: str, axis_is_time=True):
        """
        Signal ctor.

        Parameters
        ----------
        axis : np.array
            Array that holds axis ticks, e.g. seconds.
        data : np.array
            The Signal data
        name : str
            Signal name info
        axis_is_time : bool. The default is True.

        Returns
        -------
        None.

        """
        assert (
            axis.size == data.shape[-1]
        ), "size of axis must be equal to size of last dim in data (data.shape[-1])"
        assert type(name) is str

        self.axis = axis
        self.data = data
        self.name = name
        self.axis_is_time = axis_is_time
        self.__calcSampleFreq()

    def __getitem__(self, indexes):
        """
        Overloads brackets, which makes it possible to call Signal with slices just like in Numpy, e.g.:
        - signal[0] or signal[0,0]
        - signal[1:4] or signal[1:4,2:5]
        - signal[0:10:2] or signal[0:10:2,2:8:2]

        Args:
            indexes: Integer, slice object or tuple of integer or slices. Can be given as "start:stop:step"

        Returns:
            Signal: A new signal object
        """
        return self.create(
            self.name + "/" + self.__create_str_from_slice_tuple(indexes),
            self.axis.copy(),
            self.data[indexes].squeeze(),
        )

    def create_1d_signals(self) -> tuple:
        """
        Creates 1d signals from multi-dim. signal (2D or more) and returns them as a tuple of signals.

        Returns:
            tuple(Signal)
        """

        if self.data.ndim < 2:
            return (self.create(self.name, self.axis, self.data),)
        return self.__create_1d_signals_impl(self.data)

    def copy(self):
        """
        Returns a copy this object.
        """
        return self.create(self.name.copy(), self.axis.copy(), self.data.copy())

    @property
    def size(self) -> int:
        """
        Returns the number of samples in the signal
        -------
        int

        """
        return self.axis.size

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the signal.

        Returns
        -------
        tuple
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        """
        Returns the dimension of the signal

        Returns:
            int:
        """
        return self.data.ndim

    def reset_time_axis_to_zero(self) -> None:
        """
        Resets axis to zero so that axis ticks starts at zero

        Returns
        -------
        None.

        """
        if self.axis_is_time:
            self.axis = self.axis - self.axis[0]

    def remove_head(self, nRemove: int) -> None:
        """
        Removes the n first samples from the Signal

        Parameters
        ----------
        nRemove : int

        Returns
        -------
        None.

        """
        assert nRemove < self.size
        self.axis = self.axis[..., nRemove:]
        self.data = self.data[..., nRemove:]

    def remove_tail(self, nRemove: int) -> None:
        """
        Removes the n last samples from the Signal

        Parameters
        ----------
        nRemove : int

        Returns
        -------
        None.

        """
        assert nRemove < self.size
        self.axis = self.axis[..., :-nRemove]
        self.data = self.data[..., :-nRemove]

    def __create_1d_signals_impl(self, data, base_name=None):
        res = ()
        for i, d in enumerate(data):
            if len(d.shape) == 1:
                if base_name is None:
                    name = "%s/[%d]" % (self.name, i)
                else:
                    name = "%s/[%s,%d]" % (self.name, base_name, i)
                res += (self.create(name, self.axis.copy(), d.copy()),)
            else:
                res += self.__create_1d_signals_impl(d, base_name=str(i))
        return res

    def __create_str_from_slice_tuple(self, slices):
        res = ""
        if type(slices) is tuple:
            for s in slices:
                if len(res) != 0:
                    res += ","
                res += self.__create_str_from_slice(s)
        else:
            res += self.__create_str_from_slice(slices)

        return "[%s]" % (res)

    def __create_str_from_slice(self, x):
        if type(x) is slice:
            sl_start = "" if x.start is None else str(x.start)
            sl_stop = "" if x.stop is None else str(x.stop)
            if x.step is None:
                sl_str = "%s:%s" % (sl_start, sl_stop)
            else:
                sl_str = "%s:%s:%s" % (sl_start, sl_stop, x.step)
            return sl_str
        return str(x)

    def __calcSampleFreq(self):
        self.sample_freq = None

        if self.axis_is_time and self.axis.size > 10:
            sample_rates = np.diff(self.axis)
            mean_sample_rate = np.mean(sample_rates)
            rel_std_dev = np.std(sample_rates) / mean_sample_rate

            if rel_std_dev < 0.2:
                ndesimals = 10 - math.floor(math.log(mean_sample_rate * 10e10, 10)) + 1
                self.sample_freq = 1 / round(mean_sample_rate, ndesimals)

    @staticmethod
    def __find_index_from_timestamp(timeStamp, timeArray):
        if type(timeStamp) is datetime:
            unixTime = timeStamp.timestamp()
        elif isinstance(timeStamp, numbers.Number):
            unixTime = timeStamp
        else:
            raise Exception(
                "timeStamp argument must be either number of datetime object"
            )

        unixTime = np.clip(unixTime, timeArray[0], timeArray[-1])
        return np.argmax(timeArray >= unixTime)


def convert_signal_tuple_to_dict(*signals: Signal):
    """
    Converts Signals to a dictionary, where the Signal name is key.
    """
    ret = {}
    for sig in signals:
        ret[sig.dataNames.name] = sig
    return ret


def interleave_signals(*signal_tuples: tuple) -> tuple:
    """
    Interleaves the signals in the signal_tuples input argument and returns a
    single tuple with the interleaved signals.
    """
    assert all(
        len(v) == len(signal_tuples[0]) for v in signal_tuples
    ), "Size of all tuples must be equal"
    return [val for tup in zip(*signal_tuples) for val in tup]
