# -*- coding: utf-8 -*-
"""
Infiltration models

Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
"""

import numpy as np
import pandas as pd
import numba as nb


# ==============================================================================
# Infiltration methods
# ==============================================================================

@nb.jit(nopython=True)
def constant(inflow, imax):
    """
    Constant infiltration rate

    :param inflow:    [float] input flow rate (mm/time)
    :param imax:      [float] maximum infiltration rate (mm/time)
    :return:          [float] computed infiltration rate (mm/time)
    """
    return min(inflow, imax)


@nb.jit(nopython=True)
def exponential(inflow, w, wmax, imax, p):
    """
    Exponential infiltration rate

    :param inflow:    [float] input flow rate (mm/time)
    :param w:         [float] soil layer water storage (mm)
    :param wmax:      [float] maximum soil layer water storage (mm)
    :param imax:      [float] maximum infiltration rate (mm/time)
    :param p:         [float] exponential scaling parameter (-)
    :return:          [float] computed infiltration rate (mm/time)
    """
    return min(inflow, imax * np.exp(-1. * p * max(0., w)) / wmax)


@nb.jit(nopython=True)
def non_linear_scaled(inflow, w, wmax, p1, p2):
    """
    Infiltration rate non-linear scaled by relative storage
    for water balance

    :param inflow:    [float] input flow rate (mm/time)
    :param w:         [float] soil layer water storage (mm)
    :param wmax:      [float] maximum soil layer water storage (mm)
    :param p1:        [float] base infiltration rate (mm/time)
    :param p2:        [float] exponential scaling parameter (-)
    :return:          [float] computed infiltration rate (mm/time)
    """
    return min(inflow, inflow * p1 * max(0.0, max(0., w) / wmax) ** p2)


@nb.jit(nopython=True)
def non_linear(inflow, w, wmax, alpha):
    """
    Non-linear infiltration rate

    :param inflow:    [float] input flow rate (mm/time)
    :param w:         [float] soil layer water storage (mm)
    :param wmax:      [float] maximum soil layer water storage (mm)
    :param alpha:     [float] model exponent (-)
    :return:          [float] computed infiltration rate (mm/time)
    """
    return max(0.0, inflow * (1. - (max(0., w) / wmax) ** alpha))


@nb.jit(nopython=True)
def water_excess(inflow, w, wmax):
    """
    Compute the soil saturation

    :param inflow:    [float] input flow rate (mm/time)
    :param w:         [float] soil layer water storage (mm)
    :param wmax:      [float] maximum soil layer water storage (mm)
    :return:          [float] corrected inflow and water excess
    """
    water_excess = 0.0
    if inflow + w >= wmax:
        water_excess = inflow + w - wmax
        inflow -= water_excess
    return inflow, water_excess


def green_ampt(prec, psi=-100, theta0=0.2, ks=30, wmax=500, error=0.0001):
    """
    Green-Ampt infiltration method for short continuous time series

    :param prec:      [serie] precipitation (mm)
    :param psi:       [float] suction head (mm)
    :param theta0:    [float] initial water content (-) [0, 1]
    :param ks:        [float] saturated hydraulic conductivity (mm/hr)
    :param wmax:      [float] maximum soil water content (mm)
    :param error:     [float] error threshold for iterative computation
    :return:          [serie] infiltration (mm)
    """
    dt = (prec.index[1] - prec.index[0]).seconds / 3600.
    data = _green_ampt(prec.values, psi=psi, theta0=theta0, ks=ks, wmax=wmax, dt=dt, error=error)
    return pd.Series(data, index=prec.index)


def curve_number(prec, nc, k=0.2):
    """
    USGS Curve Number for losses estimation
    for short continuous time series

    :param prec:      [serie] precipitation (mm)
    :param nc:        [float] curve number (-) [0, 100]
    :param k:         [float] initial losses (-) [0, 1]
    :return:          [serie] infiltration (mm)
    """
    data = _curve_number(prec.values, nc, k=k)
    return pd.Series(data, index=prec.index)


# ==============================================================================
# Core functions
# ==============================================================================

@nb.jit(nopython=True)
def _green_ampt(prec, psi=-100, theta0=0.2, ks=30, wmax=500, dt=0.1, error=0.0001):
    """
    Core function for Green-Ampt infiltration method
    """
    n = len(prec)
    infil = np.zeros(n, dtype=np.float32)
    ks = ks * dt
    psi = abs(psi)
    theta = theta0
    w0 = theta * wmax
    F = 0.0001
    for i in range(n):
        prec = prec[i]
        dtheta = 1.0 - theta
        F0 = F
        cnt = 0
        # Green-Ampt aproach
        while True:
            p = psi * dtheta
            F1 = ks + p * np.log(1 + F0 / p)
            err = abs(F1 - F0)
            F0 = F1
            cnt += 1
            if err <= error or cnt > 10:
                break
        f = max(min(F1 - F, prec), 0)
        w1 = min(w0 + f, wmax)
        f = w1 - w0  # infiltration
        F += f       # cumulative infiltration
        theta = w1 / wmax
        w0 = w1
        infil[i] = f

    return infil


@nb.jit(nopython=True)
def _curve_number(prec, nc, k=0.2):
    """
    Core function for USGS Curve Number for losses estimation
    """
    n = len(prec)
    pacum = prec.cumsum() / 10.0  # convert mm to cm
    peff = np.zeros_like(prec)
    peffacum = np.zeros_like(prec)
    s = 2.54 * (1000 / nc - 10)
    ia = k * s
    for i in range(n):
        s1 = pacum[i] - ia
        if s1 > 0:
            peffacum[i] = s1 ** 2. / (pacum.iloc[i] + s - ia)
        elif i > 0:
            peffacum[i] = peffacum[i - 1]
    peff[1:] = np.diff(peffacum) * 10.  # convert cm to mm
    infil = prec - peff
    return infil

