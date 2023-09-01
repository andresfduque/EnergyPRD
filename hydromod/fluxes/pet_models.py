# -*- coding: utf-8 -*-
"""
Potential evapotranspiration models

Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
"""

import os
import math
from math import tanh
import calendar
import numpy as np
import pandas as pd
import numba as nb


#%% Global parameters
SOLAR_CONSTANT = 0.0820  # [MJ m-2 min-1]
PRUITT_COEF = np.array([0.2100, 0.2200, 0.2300, 0.2800, 0.3000, 0.3100,
                        0.3000, 0.2900, 0.2700, 0.2500, 0.2200, 0.2000])


# ==============================================================================
# Potential evapotranspiration methods
# ==============================================================================

def pet_oudin(tmean, lat):
    """
    Estimate daily potential evapotranspiration (PET) using the Oudin method

    :param tmean:     [Serie] input mean daily temperature (°C)
    :param lat:       [float] latitude (degrees)
    :return:          [Serie] computed potential evapotranspiration (mm)

    Reference:
    Ludovic Oudin, Frédéric Hervieu, Claude Michel, Charles Perrin, Vazken Andréassian, François Anctil, Cécile Loumagne,
    Which potential evapotranspiration input for a lumped rainfall–runoff model?: Part 2—Towards a simple and efficient
    potential evapotranspiration model for rainfall–runoff modelling. Journal of Hydrology, Volume 303, Issues 1–4, 2005,
    Pages 290-306, https://doi.org/10.1016/j.jhydrol.2004.08.026.
    """
    doy = tmean.index.dayofyear.values  # day of year
    pet = pd.Series(_pet_oudin(tmean.values, doy, lat), index=tmean.index)
    return pet


def pet_hargreaves(tmin, tmax, lat, tmean=None):
    """
    Estimate reference daily evapotranspiration over grass (ETo) using
    the Hargreaves equation

    :param tmin:    [Serie] input minimum daily temperature (°C)
    :param tmax:    [Serie] input maximum daily temperature (°C)
    :param lat:     [float] latitude (degrees)
    :param tmean:   [Serie] Optional. Input mean daily temperature in degrees
    :return:        [Serie] computed reference evapotranspiration (mm)

    Reference:
    George H. Hargreaves, ; Zohrab A. Samani, (1985). Reference Crop Evapotranspiration from Temperature.
    Applied Engineering in Agriculture, 1(2), 96–99. https://doi:10.13031/2013.26773
    """
    if tmean is None:
        tmean = (tmin + tmax) / 2.
    data = pd.concat((tmin, tmax, tmean), axis=1).dropna(how="all")
    doy = data.index.dayofyear.values
    pet = pd.DataFrame(
        _pet_hargreaves(
            data.iloc[:, 0].values,
            data.iloc[:, 1].values,
            data.iloc[:, 2].values,
            doy,
            lat
        ),
        index=data.index
    )
    return pet


def pet_pruitt(tmean, kc):
    """
    Potential evaporation computation using Doorenbos and Pruitt (1977) method

    :param tmean:   [Serie] input mean daily temperature (°C)
    :param kc:      [float] crop coefficient (-)
    :return:        [Serie] computed potential evapotranspiration (mm)

    Reference:
    Doorenbos, J. and Pruitt, W.O. (1977) Guidelines for Predicting Crop Water Requirements.
    FAO Irrigation and Drainage Paper. Food and Agriculture Organization of the United Nations,
    Rome, 145 p. http://www.fao.org/3/f2430e/f2430e.pdf
    """
    month = tmean.index.month.values  # month of year
    pet = pd.Series(_pet_pruitt(tmean.values, month, kc), index=tmean.index)
    return pet


def pet_bergstrom(tmean, cevp, cevpam, ttmp=0.0, cevpph=0):
    """
    Potential evaporation computation using Bergström method implemented in HBV model

    :param tmean:    [Serie] input mean daily temperature in degrees
    :param cevp:     [float] evapotranspiration parameter that depends of land use (mm/day.°C)
    :param cevpam:   [float] amplitude of sinus function that corrects pet
    :param ttmp:     [float] threshold temperature for snow melt and evapotranspiration (°C)
    :param cevpph:   [int] phase of sinus function that corrects pet (days)
    :return:         [Serie] computed potential evapotranspiration (mm)

    Reference:
    Lindstrom, G., Johansson, B., Persson, M., Gardelin, M. & Bergstrom, S. (1997).
    Development and test of the distributed HBV-96 model. J. Hydrol. 201, 272–288.
    https://doi.org/10.1016/S0022-1694(97)00041-3
    """
    doy = tmean.index.dayofyear.values  # day of year
    pet = pd.Series(_pet_bergstrom(tmean.values, doy, cevp, cevpam, ttmp, cevpph), index=tmean.index)
    return pet


def pet_thornthwaite(temp, lat):
    """
    Monthly potential evaporation computation using the Thornthwaite (1948) method

    :param temp:    [Serie] input mean daily air temperature for each month of the year (deg C)
    :param lat:     [float] latitude in degrees
    :return:        [Serie] computed otential evapotranspiration (mm)

    Reference:
    Thornthwaite, C. W. (1948). An approach toward a rational classification of climate.
    Geographical Review. 38 (1): 55–94. https://doi.org/10.2307/210739
    """
    # required parameters
    year = temp.index.year.values
    month = temp.index.month.values
    month_days = [calendar.monthrange(year[i], month[i])[1] for i in range(len(year))]
    lat_rad = degrees2rad(lat)
    # correct temperature
    temp_corr = temp
    temp_corr[temp_corr < 0.0] = 0.0
    # indices
    hlh = monthly_mean_daylight_hours(lat_rad, year, month)
    hi = (temp_corr / 5.0) ** 1.514
    hi = hi.groupby(year).sum()  # heat index
    alpha = (6.75e-07 * hi ** 3) - (7.71e-05 * hi ** 2) + (1.792e-02 * hi) + 0.49239
    # pet calculation
    tc = temp_corr.values
    pet = temp.copy()
    for i in range(len(year)):
        pet.iloc[i] = 1.6 * (hlh[i] / 12.0) * (month_days[i] / 30.0) \
                      * ((10.0 * tc[i] / hi.loc[year[i]]) ** alpha.loc[year[i]]) * 10.0
    return pet


# ==============================================================================
# Radiation methods
# ==============================================================================

@nb.jit(nopython=True)
def global_radiation(lat, doy):
    """
    Global radiation computed for Oudin method
    """
    glob_rad = np.zeros(len(doy), dtype=np.float32)

    for i in range(len(doy)):
        teta = 0.4093 * np.sin(doy[i] / 58.1 - 1.405)
        cosgz = max(0.001, np.cos(lat / 57.3 - teta))
        cosom = max(
            -1.0,
            min(
                1.0 - cosgz / np.cos(lat / 57.3) / np.cos(teta),
                1.0
            )
        )
        om = np.arccos(cosom)
        eta = 1.0 + np.cos(doy[i] / 58.1) / 30.0
        cospz = cosgz + np.cos(lat / 57.3) * np.cos(teta) * (np.sin(om) / om - 1.0)
        glob_rad[i] = 446 * om * eta * cospz

    return glob_rad


@nb.jit(nopython=True)
def degrees2rad(degrees):
    """
    Converts degress to radians
    """
    return degrees * (np.pi / 180.0)


@nb.jit(nopython=True)
def rad2degrees(radians):
    """
    Converts radians to degrees
    """
    return radians * (180.0 * np.pi)


@nb.jit(nopython=True)
def sun_dec(doy):
    """
    Computes sun decaiment
    """
    return 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)


@nb.jit(nopython=True)
def sunset_hour_angle(latitude, sun_decline):
    """
    Computes sun angle
    """
    cos_sha = - np.tan(latitude) * np.tan(sun_decline)
    out_var = np.zeros_like(cos_sha)
    for i in range(len(cos_sha)):
        out_var[i] = np.arccos(min(max(cos_sha[i], -1.0), 1.0))
    return out_var


@nb.jit(nopython=True)
def daylight_hours(sha):
    """
    Calculate daylight hours from sunset hour angle (sha)
    """
    return (24.0 / np.pi) * sha


@nb.jit(nopython=True)
def inv_rel_dist_earth_sun(doy):
    """
    Computes relative distance from earth to sun
    """
    return 1.0 + (0.033 * np.cos(2.0 * np.pi / 365.0 * doy))


@nb.jit(nopython=True)
def et_radiation(latitude, doy):
    """
    Global radiation computed for Hargreaves method
    """
    sd = sun_dec(doy)
    sha = sunset_hour_angle(latitude, sd)
    ird = inv_rel_dist_earth_sun(doy)

    tmp1 = (24.0 * 60.0) / np.pi
    tmp2 = sha * np.sin(latitude) * np.sin(sd)
    tmp3 = np.cos(latitude) * np.cos(sd) * np.sin(sha)
    return tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)


def month_dates(year, month):
    """
    Creates a Pandas Timestep series of all days in a month
    """
    month_days = calendar.monthrange(year, month)[1]
    dates = pd.to_datetime({
        "year": np.full(month_days, year),
        "month": np.full(month_days, month),
        "day": np.arange(1, month_days + 1)
    })
    return dates


def monthly_mean_daylight_hours(latitude, year, month):
    """
    Calculate the monthly mean daylight hours for one or more year-month combinations
    """
    if np.ndim(month) == 0:
        dates = month_dates(year, month)
        multiple = False
    else:
        n = len(year)
        dates = pd.concat([month_dates(year[i], month[i]) for i in range(n)], axis=0)
        multiple = True

    doy = dates.dt.dayofyear.values

    sd = sun_dec(doy)
    sha = sunset_hour_angle(latitude, sd)
    dlh = daylight_hours(sha)

    dlh_serie = pd.Series(dlh, index=dates)
    dlh_mean = dlh_serie.resample("1M").mean()
    if multiple:
        return dlh_mean.values
    else:
        return dlh_mean.values[0]


# ==============================================================================
# Core potential evapotranspiration methods
# ==============================================================================

@nb.jit(nopython=True)
def _pet_oudin(tmean, doy, lat):
    """
    Estimate potential evapotranspiration (PET) using the Oudin method

    :param tmean:     [array] input mean daily temperature in degrees
    :param doy:       [array] day of year for each tmean record
    :param lat:       [float] latitude
    :return:          [array] computed pet
    """
    radiation = global_radiation(lat, doy)
    pet = radiation * (tmean + 5.0) / 28.5 / 100.0
    pet[pet < 0.] = 0.0
    return pet


@nb.jit(nopython=True)
def _pet_hargreaves(tmin, tmax, tmean, doy, lat):
    """
    Estimate reference evapotranspiration over grass (ETo) using the Hargreaves
    equation and Allen method to compute potential evapotranspiration

    :param tmin:    [array] input minimum daily temperature in degrees
    :param tmax:    [array] input maximum daily temperature in degrees
    :param tmean:   [array] input mean daily temperature in degrees
    :param doy:     [array] day of year for each tmean record
    :param lat:     [float] latitude
    :return:        [array] computed eto
    """
    lat_rad = degrees2rad(lat)
    radiation = et_radiation(lat_rad, doy)
    pet = 0.0023 * (tmean + 17.8) * (tmax - tmin) ** 0.5 * 0.408 * radiation
    return pet


@nb.jit(nopython=True)
def _pet_pruitt(tmean, month, kc):
    """
    Potential evaporation computation using Doorenbos and Pruitt (1977) method

    :param tmean:   [array] input mean daily temperature in degrees
    :param month:   [array] input month for each tmean
    :param kc:      [float] crop coefficient
    :return:        [array] computed pet
    """
    ka = 1.26
    n = len(tmean)
    pet = np.zeros(n, dtype=np.float32)
    # Compute potential evaporation
    for t in range(n):
        if tmean[t] > 0:
            pet[t] = (kc * (ka * PRUITT_COEF[month[t] - 1]) * (0.46 * tmean[t] + 8.) - 2.)
    return pet


@nb.jit(nopython=True)
def _pet_bergstrom(tmean, doy, cevp, cevpam, ttmp=0.0, cevpph=0):
    """
    Potential evaporation computation using Bergström method implemented in HBV model

    :param tmean:    [array] input mean daily temperature in degrees
    :param doy:      [array] day of year for each tmean record
    :param cevp:     [float] evapotranspiration parameter that depends of land use (mm/day.°C)
    :param cevpam:   [float] amplitude of sinus function that corrects pet
    :param ttmp:     [float] threshold temperature for snow melt and evapotranspiration (°C)
    :param cevpph:   [int] phase of sinus function that corrects pet (days)
    :return:         [array] computed pet
    """
    n = len(tmean)
    pet = np.zeros(n, dtype=np.float32)
    # Compute potential evaporation
    for t in range(n):
        cseason = 1.0 + cevpam * np.sin(2.0 * np.pi * (doy[t] - cevpph) / 365.0)
        pet[t] = cevp * cseason * (tmean[t] - ttmp)
    return pet


