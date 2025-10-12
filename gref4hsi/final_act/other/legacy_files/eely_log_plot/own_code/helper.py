from Eelume import PyPost as pp
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from math import cos, sin, radians
import numpy as np


def plot_live_3d_highlighted(analyzer, date_str, transect_dict, **kwargs):
    """
    Helper function to plot a live 3D trajectory with highlighted transects for a given date.
    Each transect is drawn in a unique color and labelled in the 3D view.
    """
    if date_str not in transect_dict:
        print(f"No transect data available for date {date_str}.")
        return

    # build a list of (start, end, name) tuples
    raw = transect_dict[date_str]
    named_list = []
    for item in raw:
        if len(item) == 3:
            start, dur, name = item
        elif len(item) == 2:
            start, dur = item
            name = ""
        else:
            raise ValueError("Each tuple must be (start, duration, [name])")
        named_list.append((start, dur, name))

    analyzer.plot_live_3d_trajectory_for_date(
        date_str=date_str, transect_list=named_list, **kwargs
    )
