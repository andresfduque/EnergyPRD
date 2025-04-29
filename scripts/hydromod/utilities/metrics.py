# -*- coding: utf-8 -*-
"""
Metrics for model validation

Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
"""

import numpy as np
import pandas as pd


# ==============================================================================
# Metrics functions
# ==============================================================================

def correlation(yobs, ysim):
    """
    Pearson correlation coefficient

    cc = cov(yobs, ysim) / std(yobs) / std(ysim)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    return data.corr().iloc[1, 0]


def mean_absolute_error(yobs, ysim):
    """
    Mean Absolute Error (MAE)

    mae = sum(abs(yobs - ysim)) / n
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    return (np.abs(data.iloc[:, 0] - data.iloc[:, 1]) / data.shape[0]).sum()


def root_mean_square_error(yobs, ysim):
    """
    Root Mean Square Error (RMSE)

    rmse = sqrt(sum((yobs - ysim) ^ 2) / n)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    return (np.sum((data.iloc[:, 0] - data.iloc[:, 1]) ** 2.0) / data.shape[0]) ** 0.5


def nash_sutcliffe_efficiency(yobs, ysim):
    """
    Nash Sutcliffe Efficiency Criteria (NSE)

    nse = 1 - sum((yobs - ysim) ^ 2) / sum((yobs - mean(yobs)) ^ 2)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    rm = data.iloc[:, 0].mean()
    part1 = ((data.iloc[:, 1] - data.iloc[:, 0]) ** 2).sum()
    part2 = ((data.iloc[:, 0] - rm) ** 2).sum()

    return 1.0 - part1 / part2


def log_nash_sutcliffe_efficiency(yobs, ysim):
    """
    Nash Sutcliffe Efficiency Criteria (NSE) using logarithms
    Zero and negative values are removed from data

    xobs = log(yobs)  if yobs > 0
    xsim = log(ysim)  if ysim > 0
    lnse = 1 - sum((xobs - xsim) ^ 2) / sum((xobs - mean(xobs)) ^ 2)
    """
    data = pd.concat((yobs, ysim), axis=1)
    data[(np.isclose(data, 0)) | (data < 0)] = np.nan
    data = np.log(data.dropna(how="any"))
    rm = data.iloc[:, 0].mean()
    part1 = ((data.iloc[:, 1] - data.iloc[:, 0]) ** 2).sum()
    part2 = ((data.iloc[:, 0] - rm) ** 2).sum()

    return 1.0 - part1 / part2


def kling_gupta_efficiency(yobs, ysim):
    """
    Kling Gupta Efficiency Criteria (KGE)

    a = std(ysim) / std(yobs)
    b = mean(ysim) / mean(yobs)
    kge = 1 - sqrt((cc - 1) ^ 2 + (a - 1) ^ 2 + (b - 1) ^ 2)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    mean = data.mean().values
    std = data.std().values
    cc = data.corr().iloc[0, 1]
    part1 = (cc - 1.0) ** 2.0
    part2 = (std[1] / std[0] - 1.0) ** 2.0
    part3 = (mean[1] / mean[0] - 1.0) ** 2.0
    return 1 - (part1 + part2 + part3) ** 0.5


def bias(yobs, ysim):
    """
    Bias respect the mean value

    bias = mean_sim / mean_obs
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    mean = data.mean().values
    return mean[1] / mean[0]


def bias_relative(yobs, ysim):
    """
    Relative bias

    rbias = sum(sim - obs) / sum(obs)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    return np.sum(data.iloc[:, 1].values - data.iloc[:, 0].values) / np.sum(data.iloc[:, 0].values)


def determination_coeff(yobs, ysim):
    """
    Coefficient of Determination (R2)
    """
    r2 = correlation(yobs, ysim) ** 2.0
    return r2


#%% Summary

METRICS = {
    "cc": correlation,
    "nse": nash_sutcliffe_efficiency,
    "lnse": log_nash_sutcliffe_efficiency,
    "kge": kling_gupta_efficiency,
    "r2": determination_coeff,
    "mae": mean_absolute_error,
    "rmse": root_mean_square_error,
    "bias": bias,
    "rbias": bias_relative,
}


def summary(yobs, ysim):
    """
    Summary of metrics applied to observed and modeled timeseries
    """
    keys = list(METRICS.keys())
    n = len(keys)
    data = pd.Series(np.zeros(n, dtype=np.float32), index=keys)
    for key in keys:
        data[key] = METRICS[key](yobs, ysim)
    return data

