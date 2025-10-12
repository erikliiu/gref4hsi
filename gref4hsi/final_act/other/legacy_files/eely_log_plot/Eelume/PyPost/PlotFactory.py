# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

import Eelume.PyPost as pp


def create_plot_scatter_horisontal_pos(
    dataBase: pp.DatabaseHandler,
    start_time=None,
    end_time=None,
    figure_number=None,
    convert2ned=False,
):
    pos = pp.factory_linear_pos_signals(
        dataBase, start_time, end_time, convert2_ned=convert2ned
    )
    pp.plotScatter2D(pos[1], pos[0], figure_number=figure_number)


def plot_aml_ct_sensor(
    data_base: pp.DatabaseHandler, start_time=None, end_time=None, figure_number=None
):
    conductivity = data_base.get_signal(
        "/sensors/aml/0/conductivity", start_time=start_time, end_time=end_time
    )
    temperature = data_base.get_signal(
        "/sensors/aml/0/temperature", start_time=start_time, end_time=end_time
    )
    ylabel = ("conductivity", "temperature [deg C]")  # ToDo: add unit for conductivity
    pp.plot_line(
        conductivity,
        temperature,
        figure_number=figure_number,
        n_signals_pr_subplot=1,
        title="CT sensor",
        ylabel=ylabel,
    )


def plot_aml_sound_speed_sensor(
    data_base: pp.DatabaseHandler, start_time=None, end_time=None, figure_number=None
):
    soundspeed = data_base.get_signal(
        "/sensors/aml/1/soundspeed", start_time=start_time, end_time=end_time
    )
    temperature = data_base.get_signal(
        "/sensors/aml/1/temperature", start_time=start_time, end_time=end_time
    )
    ylabel = (
        "Speed of sound",
        "temperature [deg C]",
    )  # ToDo: add unit for conductivity
    pp.plot_line(
        soundspeed,
        temperature,
        figure_number=figure_number,
        n_signals_pr_subplot=1,
        title="Sound speed sensor",
        ylabel=ylabel,
    )
