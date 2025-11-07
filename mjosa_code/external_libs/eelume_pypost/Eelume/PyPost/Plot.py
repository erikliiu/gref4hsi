# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

from matplotlib import pyplot as plt
from Eelume.PyPost.Signal import Signal
from Eelume.PyPost.SubplotHandler import SubplotHandler
from Eelume.PyPost.Algorithms.SimpleAnalysis import rms
from datetime import datetime
import numpy as np
from collections.abc import Iterable


import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from datetime import datetime
import pandas as pd
from datetime import timedelta


def plot_line(
    *signals: Signal,
    time_axis_format: str = "datetime_utc",
    n_signals_pr_subplot: int = 0,
    figure_number: int = None,
    title=None,
    graph_type: str = "line",
    ylabel=None,
    **kwargs
):
    share_x = all(v.size == signals[0].size for v in signals)
    subplot = SubplotHandler(
        len(signals),
        figure_number=figure_number,
        n_signals_pr_subplot=n_signals_pr_subplot,
        title=title,
        share_x_axis=share_x,
    )

    for i, sig in enumerate(signals):
        assert sig.ndim == 1, "Signal must be 1-dimensional"

        ax = subplot.get_subplot(i)  # Get the current axis

        # Build the time axis
        if time_axis_format == "datetime_local":
            assert sig.axis_is_time
            ticks = [datetime.fromtimestamp(e) for e in sig.axis]
            subplot.set_xlabel(i, "Local timestamp")
        elif time_axis_format == "datetime_utc":
            assert sig.axis_is_time
            ticks = [datetime.utcfromtimestamp(e) for e in sig.axis]
            subplot.set_xlabel(i, "UTC timestamp")
        elif time_axis_format == "seconds":
            ticks = sig.axis
            subplot.set_xlabel(i, "Seconds since 1.1.1970")
        elif time_axis_format == "seconds_zero":
            # Resets time axis to zero based on the first sample
            ticks = sig.axis - signals[0].axis[0]
            subplot.set_xlabel(i, "Seconds since first sample in signal")
        else:
            raise ValueError(
                "Invalid time_axis_format. Choose from: "
                "datetime_local, datetime_utc, seconds, seconds_zero."
            )

        # Plot the data
        if graph_type == "step":
            ax.step(ticks, sig.data, label=sig.name, where="post", **kwargs)
        elif graph_type == "line":
            ax.plot(ticks, sig.data, label=sig.name, **kwargs)
        elif graph_type == "scatter":
            ax.scatter(ticks, sig.data, label=sig.name, **kwargs)
        else:
            raise ValueError("Invalid graph_type. Use 'line', 'step', or 'scatter'.")

        # If we’re dealing with date/time ticks, set the locator/formatter
        if time_axis_format in ("datetime_local", "datetime_utc"):
            locator = (
                mdates.AutoDateLocator()
            )  # or mdates.HourLocator(interval=1), etc.
            formatter = DateFormatter("%H:%M:%S")  # Show only hours:minutes:seconds
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            # Turn off the automatic date offset label (which shows e.g. "29")
            ax.xaxis.get_offset_text().set_visible(False)

        # Configure subplot labels, grid, legend
        subplot.set_ylabel(i, ylabel)
        subplot.show_grid(i)
        subplot.show_legend(i)


def plot_scatter2d(
    *signals: Signal,
    figure_number=None,
    title=None,
    xlabel="x",
    ylabel="y",
    plot_as_line=None,
    equal_axis=False,
    **kwargs
):
    """
    Plots a 2D scatter plot.

    Parameters
    ----------
    *signals : Signal
        Variadic number of arguments with x and y Signal objects to plot along x and y axis,
        give as x0, y0, x1, y1, ..... , xn, yn. Number of arguments must be multiple of 2.
    figure_number : int. The default is None.
        Sets the plot window figure number.
        If a figure with the given number exists, the content in the window will
        be overwritten, else a new figure window will be created.
    title : str
        Figure title. The default is None.
    xlabel : str
        x-axis label string. The default is 'x'.
    ylabel : str
        y-axis label string. The default is 'y'.
    plot_as_line : bool or tuple(bool,).
        Plots lines between markers. Give as bool to plot for all x,y pairs or a tuple or list
        to specify for each x,y pair separatly. The default is None.
    equal_axis : bool
        Sets equal axis aspect ratio for x and y axis
    **kwargs :
        key-value argument which is passed directly to the
        matplotlib.pyplot.scatter function. See: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    """

    assert len(signals) % 2 == 0
    assert all(v.ndim == 1 for v in signals), "Signals must be 1-dimensional"

    # Create a new figure or activate an existing figure with the given figure number
    fig = plt.figure(figure_number)
    fig.clf()  # Clear the current figure
    # Add a subplot to the figure with a single subplot (1 row, 1 column, first subplot)
    ax = fig.add_subplot(111)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if equal_axis:
        ax.axis("equal")

    # for scatter plot (xy plot) we should iterate twice -> since (x, y) and then (x_ref, y_ref)
    for i in range(0, len(signals), 2):

        # signals is a list of Signal objecst, where signals[i] is x and signals[i+1] is y
        # Ensure that the x and y signals have the same number of samples
        assert signals[i].size == signals[i + 1].size

        use_line_plot = False
        if plot_as_line is not None:
            if isinstance(plot_as_line, Iterable):
                # Ensure the length of plot_as_line matches the number of signal pairs
                assert len(plot_as_line) == len(signals) // 2
                # Ensure each entry in plot_as_line is a boolean
                assert type(plot_as_line[i // 2]) is bool
                use_line_plot = plot_as_line[i // 2]
            else:
                # Ensure plot_as_line is a boolean
                assert type(plot_as_line) is bool
                use_line_plot = plot_as_line

        # Set linestyle based on whether to plot as line or not
        if use_line_plot:
            linestyle = "-"
        else:
            linestyle = ""

        # Set default marker if not provided in kwargs
        if "marker" not in kwargs:
            kwargs["marker"] = "."

        # Plot the x and y signals
        plt.plot(
            signals[i].data,
            signals[i + 1].data,
            label="x: %s, y: %s" % (signals[i].name, signals[i + 1].name),
            linestyle=linestyle,
            **kwargs
        )

    plt.legend()


def plot_psd(*signals: Signal, figure_number=None, title=None, **kwargs):
    """
    Plots the Power Spectral Density (fft). It is recommended to set these kwargs parameters (values as examples):
    detrend='mean',NFFT=256,noverlap=128

    Parameters
    ----------
    signals : Signal
        Input time series Signal object
    figure_number : int. The default is None.
        Sets the plot window figure number.
        If a figure with the given number exists, the content in the window will
        be overwritten, else a new figure window will be created.
    title : str.
        Figure title text
    **kwargs :
        key-value argument which is passed directly to the
        matplotlib.pyplot.psd function. See:
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.psd.html

    Returns
    -------
    None.

    """
    plt.figure(figure_number)
    plt.cla()
    plt.grid()
    plt.title(title)

    for x in signals:
        assert x.axis_is_time

        if x.sample_freq is None:
            assert kwargs["Fs"]
        else:
            kwargs["Fs"] = x.sample_freq

        plt.psd(x.data, label=x.name, **kwargs)
    plt.legend()


def plot_rms(
    *signals: Signal,
    figure_number=None,
    n_signals_pr_subplot=0,
    title=None,
    window_length=100,
    detrend=True,
    high_pass_cut_off=None,
    output_spectral_density=False
):
    """
    Plots the RMS of data

    Parameters
    ----------
    signals : Signal
        Input time series Signal object
    figure_number : int. The default is None.
        Sets the plot window figure number.
        If a figure with the given number exists, the content in the window will
        be overwritten, else a new figure window will be created.
    n_signals_pr_subplot : int
        The default is 0. The number of signals to plot per
        sub-plot. Set to zero to all signals in one plot (no sub-plots)
    title : str. The default is None.
        Plot title showned in the figure window.
    window_length : int. The default is 100
        The windows length in samples to take RMS over.
    detrend : bool. The default is True.
        Remove linear trend from data.
    high_pass_cut_off : float. The default is None (high pass filtering turned off)
        High pass filter cut off frequency in Hz. Must be None if detrend=True.
    output_spectral_density : bool
        Output RMS spectral density with unit SignalUnit/rt(Hz). The default is False.

    Returns
    -------
    None.

    """

    y = rms(
        *signals,
        window_length=window_length,
        detrend=detrend,
        high_pass_cut_off=high_pass_cut_off,
        output_spectral_density=output_spectral_density
    )
    plot_line(
        *y,
        n_signals_pr_subplot=n_signals_pr_subplot,
        figure_number=figure_number,
        title=title
    )


def plot_histogram(
    *signals: Signal, figure_number=None, title=None, xlabel=None, bins=100, **kwargs
):
    """
    Plots a histogram of a single Signal

    Parameters
    ----------
    signals : Signal. Input signals to plot
    figure_number : int. The default is None.
        Sets the plot window figure number.
        If a figure with the given number exists, the content in the window will
        be overwritten, else a new figure window will be created.
    xlabel : str or iterable of strings
    bins : int
        The number of bins in histogram
    **kwargs :
        key-value argument which is passed directly to the
        matplotlib.pyplot.hist function. See:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    """
    subplots = SubplotHandler(
        len(signals), figure_number=figure_number, n_signals_pr_subplot=1, title=title
    )

    for i, sig in enumerate(signals):
        subplots.show_grid(i)
        subplots.set_xlabel(i, xlabel)
        subplots.set_ylabel(i, "Counts")
        subplots.get_subplot(i).hist(sig.data, label=sig.name, bins=bins, **kwargs)
        subplots.show_legend(i)
