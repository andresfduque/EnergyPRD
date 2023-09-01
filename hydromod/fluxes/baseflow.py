# -*- coding: utf-8 -*-
"""
Baseflow and soil discharge models

Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
"""

import numpy as np
import numba as nb


# ==============================================================================
# Baseflow methods
# ==============================================================================

@nb.jit(nopython=True)
def linear(w, k):
    """
    Output from a linear reservoir

    :param w:         [float] soil layer water storage (mm)
    :param k:         [float] discharge parameter (1/d)
    :return:          [float] computed discharge (mm/d)
    """
    return w * k


@nb.jit(nopython=True)
def linear_threshold(w, k, h0):
    """
    Output from a linear reservoir with threshold

    :param w:         [float] soil layer water storage (mm)
    :param k:         [float] discharge parameter (1/d)
    :return:          [float] computed discharge (mm/d)
    """
    return k * max(0., w - h0)


@nb.jit(nopython=True)
def exponential(w, k, b):
    """
    Exponential outflow from deficit store

    :param w:         [float] soil layer water storage (mm)
    :param k:         [float] base outflow rate (mm/d)
    :param b:         [float] exponent parameter (1/mm)
    :return:          [float] computed discharge (mm/d)
    """
    return max(0., k * np.exp(- b * w))


@nb.jit(nopython=True)
def exponential_scaled(w, wmax, k, b):
    """
    Exponential scaled outflow from deficit store

    :param w:         [float] soil layer water storage (mm)
    :param wmax:      [float] maximum soil layer water storage (mm)
    :param k:         [float] base outflow rate (mm/d)
    :param b:         [float] exponent scaling parameter (1/mm)
    :return:          [float] computed discharge (mm/d)
    """
    return max(0., k * (np.exp(b * min(w, 0.) / wmax) - 1.))


@nb.jit(nopython=True)
def non_linear(w, a, b):
    """
    Output from a non-linear reservoir

    :param w:         [float] soil layer water storage (mm)
    :param a:         [float] time coefficient (d)
    :param b:         [float] exponential scaling parameter (-)
    :return:          [float] computed discharge (mm/d)
    """
    return max(0., 1. / a * max(w, 0.) ** (1. / b))


@nb.jit(nopython=True)
def non_linear_scaled(w, wmax, k, b):
    """
    Non-linear scaled outflow from a reservoir

    :param w:         [float] soil layer water storage (mm)
    :param wmax:      [float] maximum soil layer water storage (mm)
    :param k:         [float] base outflow rate (mm/d)
    :param b:         [float] exponential scaling parameter (-)
    :return:          [float] computed discharge (mm/d)
    """
    return min(w, k * (max(w, 0.) / wmax) ** b)


@nb.jit(nopython=True)
def linear_two_outputs(w, k0, k1, h0):
    """
    Two linear output from a reservoir

    :param w:         [float] soil layer water storage (mm)
    :param k0:        [float] discharge parameter upper discharge (1/d)
    :param k1:        [float] discharge parameter lower discharge (1/d)
    :param h0:        [float] threshold storage for upper discharge (mm)
    :return:          [float] computed upper and lower discharges (mm/d)
    """
    q0 = max(0., (w-h0) * k0)
    w = max(0., w - q0)
    q1 = max(0., w * k1)
    return q0, q1


