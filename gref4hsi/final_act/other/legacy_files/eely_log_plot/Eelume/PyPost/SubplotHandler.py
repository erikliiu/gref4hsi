# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

from matplotlib import pyplot as plt
from collections.abc import Iterable   
import numpy as np
from Eelume.PyPost.Signal import Signal


class SubplotHandler(object):
    """
    Helper class for managing sub-plots
    """
    def __init__(self, n_signals : int, figure_number:int=None, n_signals_pr_subplot:int=0, title:str=None, share_x_axis:bool=False):
        """Constructor

        Args:
            n_signals (int): number of signals to plot
            figure_number (int): The figure number of the plot window. Defaults to None.
            n_signals_pr_subplot (int): Number of singals in each subplot. Defaults to 0.
            title (str): Main title of the plot. Defaults to None.
            share_x_axis (bool): If True x-axis is shared for the subplots. Defaults to False.
        """
        fig = plt.figure(figure_number)
        fig.clf()
        
        if title is not None:
            fig.suptitle(title)
        
        self.n_signals_pr_subplot = n_signals_pr_subplot
        if self.n_signals_pr_subplot <= 0:
            self.n_signals_pr_subplot = n_signals

        self.sub_plot_map = np.zeros(n_signals,dtype=int)
        self.n_sub_plots = max(1,int(np.ceil(n_signals / self.n_signals_pr_subplot)) )
        for i in range(n_signals):
            self.sub_plot_map[i] = i // self.n_signals_pr_subplot

        if self.n_sub_plots >= 6:
            subplotShape = (int(np.ceil(self.n_sub_plots / 3)), 3)
        elif self.n_sub_plots >= 4:
            subplotShape = (int(np.ceil(self.n_sub_plots / 2)), 2)
        else:
            subplotShape = (self.n_sub_plots,1)

        self.fig, self.ax = plt.subplots(subplotShape[0], subplotShape[1], sharex=share_x_axis, num=plt.gcf().number)
    
        if type(self.ax) is not np.ndarray:
            self.ax = np.array([self.ax])

    def get_subplot(self,index : int):
        """Gets matplotlib axes subplot handle

        Args:
            index (int): Index of the subplot

        Returns:
            plt.subplot.ax: Axes handle
        """
        return self.ax.flatten()[self.sub_plot_map[index]]     

    def set_xlabel(self, index : int, xlabel:str):
        """Sets xlabel of a specific subplot

        Args:
            index (int): index for subplot
            xlabel (str): label
        """
        assert index < self.sub_plot_map.size
        
        if xlabel is not None:
            if type(xlabel) is not str:
                assert isinstance(xlabel, Iterable)
                assert len(xlabel) == self.n_sub_plots, 'xlabel must either be scalar or an iterable that match number of sub-plots'
                self.ax.flatten()[self.sub_plot_map[index]].set_xlabel(xlabel[self.sub_plot_map[index]])
            else:
                self.ax.flatten()[self.sub_plot_map[index]].set_xlabel(xlabel)

    def set_ylabel(self, index : int, ylabel:str):
        """Sets ylabel for a specific subplot

        Args:
            index (int): index for subplot
            ylabel (str): label
        """
        assert index < self.sub_plot_map.size
        
        if ylabel is not None:
            if type(ylabel) is not str:
                assert isinstance(ylabel, Iterable)
                assert len(ylabel) == self.n_sub_plots, 'ylabel must either be scalar or an iterable that match number of sub-plots'
                self.ax.flatten()[self.sub_plot_map[index]].set_ylabel(ylabel[self.sub_plot_map[index]])
            else:
                self.ax.flatten()[self.sub_plot_map[index]].set_ylabel(ylabel)

    def show_legend(self, index : int):
        """
        Shows the legend for a specific subplot.
        Note that legends must be set by the label kwarg in matplotlip plot function prior to calling this method.

        Args:
            index (int): Subplot index.
        """
        assert index < self.sub_plot_map.size
        self.ax.flatten()[self.sub_plot_map[index]].legend()   

    def show_grid(self, index : int, on:bool=True):
        """Turns grid on/off

        Args:
            index (int): Subplot index
            on (bool): Turns grid on if True. Defaults to True.
        """
        assert index < self.sub_plot_map.size
        self.ax.flatten()[self.sub_plot_map[index]].grid(on)