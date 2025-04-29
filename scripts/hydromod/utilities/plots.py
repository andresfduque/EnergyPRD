# -*- coding: utf-8 -*-
"""
Plot tools for model evaluation

Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
"""

import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

from . import time_series as ts
from . import metrics


# ==============================================================================
# Plot tools
# ==============================================================================

DEFAULT_COLORS = ["k", "b", "r", "g", "c", "y"] * 100


def model_evaluation(prec, qobs, qsim, start=None, end=None, units="m3/s",
                     ocolor="#212f3c", scolor="#c0392b", **kwargs):
    """
    Generate multiple plots for model evaluation

    :param prec:      [Serie] precipitation in a catchment
    :param qobs:      [Serie] observed streamflow
    :param qsim:      [Serie] simulated streamflow
    :param start:     [str] initial date for plotting
    :param end:       [str] final date for plotting
    :param units:     [str] streamflow units for labels. Options: "m3/s" (default), "mm"
    :param ocolor:    [str] line color for observations
    :param scolor:    [str] line color for simulations
    :param kwargs:    additional plot options (figsize, fontsize, linewidth)
    :return:          figure, {axis}
    """
    if start is None:
        start = max(prec.index.min(), qobs.index.min(), qsim.index.min())
    if end is None:
        end = min(prec.index.max(), qobs.index.max(), qsim.index.max())

    # Daily data
    prec = prec.loc[start:end]
    qobs = qobs.loc[start:end]
    qsim = qsim.loc[start:end]
    # Monthly data
    if "mm" in units.lower():
        qobsm = ts.monthly_stats(qobs, method="sum")["mean"]
        qsimm = ts.monthly_stats(qsim, method="sum")["mean"]
    else:
        qobsm = ts.monthly_stats(qobs, method="mean")["mean"]
        qsimm = ts.monthly_stats(qsim, method="mean")["mean"]
    # Flow duration curve
    qobs_fdc = ts.flow_duration_curve(qobs)
    qsim_fdc = ts.flow_duration_curve(qsim)

    # Create Axis
    fig = plt.figure(figsize=kwargs.get("figsize", (10, 8)))
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :])  # precipitation axis
    ax2 = fig.add_subplot(gs[1, :])  # streamflow axis
    ax3 = fig.add_subplot(gs[2, 0])  # mean monthly streamflow
    ax4 = fig.add_subplot(gs[2, 1])  # flow duration curve
    ax5 = fig.add_subplot(gs[2, 2])  # streamflow correlation

    # plot precipitation
    prec.plot(color=ocolor, ax=ax1)
    ax1.set_ylabel("Precipitation (mm)", fontsize=kwargs.get("fontsize", 12))
    ax1.set_xlabel("")

    # plot streamflow
    kge = metrics.kling_gupta_efficiency(qobs, qsim)
    nse = metrics.nash_sutcliffe_efficiency(qobs, qsim)
    rmse = metrics.root_mean_square_error(qobs, qsim)
    title = f"KGE:{kge:.2f}, NSE:{nse:.2f}, RMSE:{rmse:.2f}"
    qobs.plot(color=ocolor, label="Obs", ax=ax2)
    qsim.plot(color=scolor, label="Sim", ax=ax2)
    ax2.set_ylabel(f"Streamflow ({units})", fontsize=kwargs.get("fontsize", 12))
    ax2.set_xlabel("")
    ax2.set_title(title, fontsize=kwargs.get("fontsize", 12))
    ax2.legend(["Obs", "Sim"], loc=1)

    # monthly streamflow
    nse = metrics.nash_sutcliffe_efficiency(qobsm, qsimm)
    rbias = metrics.bias_relative(qobsm, qsimm)
    cc = metrics.correlation(qobsm, qsimm)
    title = f"NSE:{nse:.2f}, RBias:{rbias:.3f}, CC:{cc:.2f}"
    qobsm.plot(color=ocolor, label="Obs", ax=ax3)
    qsimm.plot(color=scolor, label="Sim", ax=ax3)
    ax3.set_ylabel(f"Streamflow  ({units}/month)", fontsize=kwargs.get("fontsize", 12))
    ax3.set_xlabel("")
    ax3.set_title(title, fontsize=kwargs.get("fontsize", 12))
    ax3.set_xticks(np.arange(1, 13))

    # flow duration curve
    lognse = metrics.log_nash_sutcliffe_efficiency(qobs_fdc, qsim_fdc)
    logrmse = metrics.root_mean_square_error(np.log(qobs_fdc), np.log(qsim_fdc))
    title = f"LogNSE:{lognse:.2f}, LogRMSE:{logrmse:.2f}"
    qobs_fdc.plot(color=ocolor, label="Obs", ax=ax4)
    qsim_fdc.plot(color=scolor, label="Sim", ax=ax4)
    ax4.set_ylabel(f"Streamflow ({units})", fontsize=kwargs.get("fontsize", 12))
    ax4.set_xlabel("")
    ax4.set_title(title, fontsize=kwargs.get("fontsize", 12))
    ax4.set_yscale("log")

    # correlation
    r2 = metrics.determination_coeff(qobs, qsim)
    cc = metrics.correlation(qobs, qsim)
    rbias = metrics.bias_relative(qobs, qsim)
    title = f"R2:{r2:.2f}, CC:{cc:.2f}, RBias:{rbias:.3f}"
    ax5.plot(qobs, qsim, '.', color=ocolor, label="Streamflow", alpha=0.5)
    vmax = max(qobs.max(), qsim.max())
    ax5.plot([0, vmax], [0, vmax], 'k', label="line 1:1")
    ax5.set_xlim(0, vmax)
    ax5.set_ylim(0, vmax)
    ax5.set_xlabel(f"Streamflow Obs ({units})", fontsize=kwargs.get("fontsize", 12))
    ax5.set_ylabel(f"Streamflow Sim ({units})", fontsize=kwargs.get("fontsize", 12))
    ax5.set_title(title, fontsize=kwargs.get("fontsize", 12))

    fig.tight_layout()
    axes = {
        "prec": ax1,
        "qt": ax2,
        "qtm": ax3,
        "fdc": ax4,
        "corr": ax5
    }
    plt.show()

    return fig, axes


def plot_correlation(yobs, ysim, reg=False, log=False, scores=True, ax=None, **kwargs):
    """
    Create correlation plot

    :param yobs:     [Serie] observed time serie
    :param ysim:     [Serie] simulated time serie
    :param reg:      [bool] if True, fits a linear regression
    :param log:      [bool] if True, uses log scale
    :param scores:   [Boolean, list, tuple] if True, show the KGE, NSE and CC metrics
                        Also, a list of metrics can be defined
                            "kge"   > Kling-Gupta Efficiency
                            "nse"   > Nash-Sutcliffe Efficiency
                            "lnse"  > Nash-Sutcliffe Efficiency using logarithms
                            "cc"    > Correlacion coefficient
                            "mae"   > Mean Absolute Error
                            "r2"    > Determination coefficient
                            "rmse"  > Root Mean Squared Error
                            "bias"  > Bias
                            "rbias" > Relative bias
    :param ax:       [axis] use an existing axis
    :param kwargs:   additional matplotlib plot parameters (figsize, fontsize)
    :return:         Matplotlib axis
    """

    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    corr = data.corr().iloc[1, 0]

    lscores = []
    if type(scores) in (list, tuple):
        lscores = [str(x).lower() for x in scores]
    else:
        if scores:
            lscores = ["kge", "nse", "cc"]
    title = ""
    if lscores:
        for key in lscores:
            if key in metrics.METRICS:
                error = metrics.METRICS[key](data.iloc[:, 0], data.iloc[:, 1])
            title += f"{key.upper()}:{error:.2f} "

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 5)))
    ax.plot(data.iloc[:, 0], data.iloc[:, 1], '.', color="gray",
            alpha=0.5, label=f"$p$={corr:.2f}")

    vmax = data.max().max()
    ax.plot([0, vmax], [0, vmax], 'k', label="1:1 line")

    if reg:
        eq = np.polyfit(data.iloc[:, 0], data.iloc[:, 1], 1)
        yreg = np.polyval(eq, data.iloc[:, 0].values)
        ax.plot(data.iloc[:, 0], yreg, color="r", alpha=0.5,
                label=f"Y={eq[0]:.3f}X{eq[1]:+.3f}")

    ax.set_xlabel(kwargs.get("xlabel", data.columns[0]),
                  fontsize=kwargs.get("fontsize", 12))
    ax.set_ylabel(kwargs.get("ylabel", data.columns[1]),
                  fontsize=kwargs.get("fontsize", 12))
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.grid(kwargs.get("grid", True))
    ax.legend(loc=1)

    if title:
        ax.set_title(title, fontsize=kwargs.get("fontsize", 12))
    if fig is not None:
        fig.tight_layout()
    return ax


def daily_median(serie, ci=None, color="k", ylabel=None, ax=None, **kwargs):
    """
    Plots the median daily value of a timeserie

    :param serie:    [Serie] input daily timeseries
    :param ci:       [float] percentile to plot bounds. Lower bound=ci, Upper bound=1-ci
                       ci can vary from 0 to 1. If None, boundaries are not ploted
    :param color:    [string] color of line and areas
    :param ylabel:   [string] title at yaxis. By default None
    :param ax:       [axis] use an existing axis
    :param kwargs:   additional matplotlib plot parameters (figsize, fontsize, alpha,
                      yscale, grid)
    :return:         Matplotlib axis
    """
    days = serie.index.dayofyear
    sm = serie.groupby(days).median()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 4)))

    if ci is not None:
        c1 = ci
        c2 = 1.0 - ci
        c1, c2 = min(c1, c2), max(c1, c2)
        sl = serie.groupby(days).quantile(c1)
        su = serie.groupby(days).quantile(c2)
        ax.fill_between(sm.index, sl, su, color=color, alpha=kwargs.get("alpha", 0.1))

    sm.plot(ax=ax, color=color, label="Median")
    ax.legend(loc=1)
    ax.set_xlim(0, 365)
    ax.set_yscale(kwargs.get("yscale", "linear"))
    ax.set_ylabel(ylabel, fontsize=kwargs.get("fontsize", 12))
    ax.set_xlabel("Day of year", fontsize=kwargs.get("fontsize", 12))
    ax.grid(kwargs.get("grid", True))
    if fig is not None:
        fig.tight_layout()
    plt.show()
    return ax


def double_yaxis(left, right, ylabel=None, rylabel=None, revert=False,
                 colors=None, rcolors=None, ax=None, **kwargs):
    """
    Plots a double yaxis graph using multiple timeseries.

    :param left:        [Serie, DataFrame] left axis data to plot
    :param right:       [Serie, DataFrame] right axis data to plot
    :param ylabel:      [string] title at left yaxis. By default None
    :param rylabel:     [string] title at right yaxis. By default None
    :param revert:      [Bool] if True, right yaxis is reverted
    :param colors:      [list, tuple] color name for each serie in left axis
    :param rcolors:     [list, tuple] color name for each serie in right axis
    :param ax:          [axis] use an existing axis
    :param kwargs:      additional matplotlib plot parameters (figsize=(8, 4), fontsize=12,
                         grid=False, ylim=None, rylim=None, yscale="log" or "linear",
                         ryscale="log" or "linear", linestyle="-", rlinestyle="--")
    :return:            Matplotlib axis
    """
    if colors is None:
        colors = DEFAULT_COLORS
    if rcolors is None:
        rcolors = DEFAULT_COLORS

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 4)))

    left.plot(ax=ax, color=colors, linestyle=kwargs.get("linestyle", "-"))
    axr = ax.twinx()
    right.plot(ax=axr, color=rcolors, linestyle=kwargs.get("rlinestyle", "--"))

    axr.set_xlabel("")
    ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=kwargs.get("fontsize", 12))
    ax.set_ylabel(ylabel, fontsize=kwargs.get("fontsize", 12))
    axr.set_ylabel(rylabel, fontsize=kwargs.get("fontsize", 12))
    ax.set_yscale(kwargs.get("yscale", "linear"))
    axr.set_yscale(kwargs.get("ryscale", "linear"))
    ax.set_ylim(kwargs.get("ylim", None))
    axr.set_ylim(kwargs.get("rylim", None))
    ax.grid(kwargs.get("grid", False))
    axr.grid(kwargs.get("grid", False))
    ax.legend(loc=2)
    axr.legend(loc=1)
    if revert:
        axr.invert_yaxis()
    plt.show()
    return ax


def flow_duration_curves(streamflow, attrs=None, labels=None, ylabel="Streamflow (m3/s)",
                         colors=None, cmap="viridis", ax=None, **kwargs):
    """
    Plots the flow duration curve (FDC) of multiple streamflow timeseries

    :param streamflow:  [Serie, DataFrame, list] streamflow time-series. If streamflow can be a DataFrame
                          or a list/tuple of multiple series
    :param attrs:       [Serie, array, list] add a color acoording to the input attrs parameter.
                          The number of values in attrs must be equal than numer of streamflow timeseries
    :param labels:      [list, tuple] label for each streamflow timeserie to use in legend
    :param ylabel:      [string] title at yaxis. By default "Streamflow (m3/s)"
    :param colors:      [list, tuple] color name for each streamflow timeseie
    :param cmap:        [string] color map when attrs parameter is used. By default "viridis"
    :param ax:          [axis] use an existing axis
    :param kwargs:      additional matplotlib plot parameters (figsize, fontsize)
    :return:            Matplotlib axis
    """

    # Sort data to plot
    if isinstance(streamflow, pd.Series):
        data = [streamflow]
    elif isinstance(streamflow, pd.DataFrame):
        data = [streamflow.iloc[:, i] for i in range(streamflow.shape[1])]
    elif type(streamflow) in (list, tuple):
        data = []
        for row in streamflow:
            if isinstance(row, pd.Series):
                data.append(row)
            else:
                for i in range(row.shape[1]):
                    data.append(row.iloc[:, i])

    # Compute flow duration curves
    data = [ts.flow_duration_curve(x) for x in data]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 5)))

    # In case of input attributes
    if attrs is not None:
        if isinstance(attrs, pd.Series):
            attrs = attrs.values
        else:
            attrs = np.array(attrs)

        z = [[0, 0], [0, 0]]
        vmin = kwargs.get("vmin", np.floor(attrs.min()))
        vmax = kwargs.get("vmax", np.ceil(attrs.max()))
        step = (vmax - vmin) / 10.0
        levels = np.arange(vmin, vmax+step, step)
        pbase = ax.contourf(z, levels, cmap=cmap)
        colormap = cm.get_cmap(cmap, len(attrs))
        ax.cla()

    for i in range(len(data)):
        if labels:
            label = labels[i]
        else:
            label = None

        if attrs is not None:
            color = colormap(i)
        elif colors:
            color = colors[i]
        else:
            color = DEFAULT_COLORS[i]

        data[i].plot(
            color=color,
            ax=ax,
            label=label,
            linewidth=kwargs.get("linewidth", 1),
            linestyle=kwargs.get("linestyle", "-"),
        )

    if attrs is not None:
        plt.colorbar(pbase)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency", fontsize=kwargs.get("fontsize", 12))
    ax.set_ylabel(ylabel, fontsize=kwargs.get("fontsize", 12))
    ax.grid(True, which="both")
    if labels:
        ax.legend(loc=0)
    if fig is not None:
        fig.tight_layout()
    plt.show()
    return ax


