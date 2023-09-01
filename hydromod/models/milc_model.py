# -*- coding: utf-8 -*-
"""
Modello Idrologico Lumped in Continuo (MILC)

Rain-Runoff Model for daily simulation


Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx


Based on:
MISDc Rainfall-Runoff Model (https://github.com/IRPIhydrology/MISDc)

Reference:
Brocca, L., Liersch, S., Melone, F., Moramarco, T., Volk, M. (2013).
Application of a model-based rainfall-runoff database as efficient tool for flood risk management.
Hydrology and Earth System Sciences Discussion, 10, 2089-2115.
"""

# %% Import libraries
import os
import numpy as np
import pandas as pd
import numba as nb
from ..fluxes import pet_models
from .base_model import BaseModel
import warnings

warnings.filterwarnings("ignore")

# Load IUH data
IUH_DATA = np.loadtxt(os.path.join(os.path.dirname(__file__), "milc_iuh.txt"))
LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


# ==============================================================================
# Main class
# ==============================================================================

class MILC(BaseModel):

    def __init__(self, area=100, lat=0, params=None):
        """
        Modello Idrologico Lumped in Continuo (MILC)


        INPUTS:
            area    >     [float] catchment area in km2
            lat     >     [float] catchment latitude at centroid
            params  >     [dict] model parameters

        Model Parameters
            w0      >     [float] initial water content as a fraction of W_max (-)
            wmax    >     [float] maximum water content (mm)
            gamma   >     [float] routing coefficient for HUI lag-time relationship
            kc      >     [float] pet parameter for Doorenbos and Pruitt (1977) method (-)
            alpha   >     [float] runoff exponent (-)
            m       >     [float] drainage exponent of percolation (-)
            ks      >     [float] satured hydraulic conductivity (mm/d)
            nu      >     [float] fraction of drainage vs interflow (-)
            kc      >     [float] crop coefficient (-)


        Based on:
        MISDc Rainfall-Runoff Model (https://github.com/IRPIhydrology/MISDc)

        Reference:
        Brocca, L., Liersch, S., Melone, F., Moramarco, T., Volk, M. (2013).
        Application of a model-based rainfall-runoff database as efficient tool for flood risk management.
        Hydrology and Earth System Sciences Discussion, 10, 2089-2115.
        """
        super().__init__(area, lat)

        self.params = {
            "gamma": 5.0,  # routing coefficient for HUI lag-time relationship
            "w0": 0.5,  # initial water content as a fraction of W_max (-)
            "wmax": 1000.0,  # maximum water content (mm)
            "m": 10.0,  # drainage exponent of percolation (-)
            "ks": 100,  # satured hydraulic conductivity (mm/d)
            "kc": 1.0,  # crop coefficient
            "nu": 0.5,  # fraction of drainage vs interflow (-)
            "alpha": 2.0  # runoff exponent (-)
        }

        if params is not None:
            self.set_parameters(**params)

    def __repr__(self):
        text = "\n______________MILC structure______________\n"
        text += "Catchment properties:\n"
        text += "    Area (km2): {:.3f}\n".format(self.area)
        text += "    Latitude  : {:.4f}\n".format(self.lat)
        text += "Model Parameters:\n"

        text += "    w0    > Initial Water Content (adim)             : {:.3f}\n".format(self.params["w0"])
        text += "    wmax  > Maximum Water Capacity (mm)              : {:.3f}\n".format(self.params["wmax"])
        text += "    gamma > Routing coefficient (adim)               : {:.3f}\n".format(self.params["gamma"])
        text += "    kc    > Pot. evapotranspiration parameter (adim) : {:.3f}\n".format(self.params["kc"])
        text += "    alpha > Runoff parameter (adim)                  : {:.3f}\n".format(self.params["alpha"])
        text += "    m     > Drainage exponent (adim)                 : {:.3f}\n".format(self.params["m"])
        text += "    ks    > Satured hydraulic conductivity (mm/d)    : {:.3f}\n".format(self.params["ks"])
        text += "    nu    > Fraction of drainage vs interflow (adim) : {:.3f}\n".format(self.params["nu"])
        return text

    def __str__(self):
        gamma = self.params["gamma"]
        w0 = self.params["w0"]
        wmax = self.params["wmax"]
        alpha = self.params["alpha"]
        kc = self.params["kc"]
        m = self.params["m"]
        ks = self.params["ks"]
        nu = self.params["nu"]

        text = f"MILC(area={self.area:.2f},lat={self.lat:.3f},"
        text += f"gamma={gamma:.3f},w0={w0:.3f},wmax={wmax:.2f},alpha={alpha:.2f},"
        text += f"kc={kc:.2f},m={m:.3f},ks={ks:.3f},nu={nu:.2f})"
        return text

    def run(self, forcings, start=None, end=None, save_state=False, dt=0.2, **kwargs):
        """
        Run the MILC model


        Parameters
        ----------
        forcings : DataFrame
            Input data with columns prec (precipitation), tmean, and
            pet(potential evapotranspiration, optional)
        start : string, optional
            Start date for simulation in format. Example: '2001-01-01'
        end : string, optional
            End date for simulation in format. Example: '2010-12-31'
        save_state : bool, optional
            If True (default), last storage is saved as w0 parameter
        dt : float, optional
            Time step in hours to compute IUH
        **kwargs :
            Model parameters can be changed for the simulation
                area    >     [float] catchment area in km2
                lat     >     [float] catchment latitude at centroid
                w0      >     [float] initial water content as a fraction of W_max (-)
                wmax    >     [float] maximum water content (mm)
                gamma   >     [float] routing coefficient for HUI lag-time relationship
                kc      >     [float] pet parameter for Doorenbos and Pruitt (1977) method (-)
                alpha   >     [float] runoff exponent (-)
                m       >     [float] drainage exponent of percolation (-)
                ks      >     [float] saturated hydraulic conductivity (mm/d)
                nu      >     [float] fraction of drainage vs interflow (-)
                kc      >     [float] crop coefficient (-)

        Returns
        -------
        Simulations : DataFrame
            qt       > Streamflow (Qd+Qb) at catchment output (m3/s)
            qd       > Direct flow at catchment output (m3/s)
            qb       > Baseflow at catchment output (m3/s)
            pet      > Potential Evapotranspiration (mm)
            et       > Evapotranspiration (mm)
            runoff   > Runoff (mm)
            baseflow > Baseflow in (mm)
            infil    > Infiltration (mm)
            perc     > Percolation (mm)
            ws       > Water Soil content dimensionless (w/wmax)
        """
        if start is None:
            start = forcings.index.min()
        if end is None:
            end = forcings.index.max()
        forcings = forcings.loc[start:end, :]

        # Load new parameters
        if kwargs:
            self.area = kwargs.get("area", self.area)
            self.lat = kwargs.get("lat", self.lat)
            self.set_parameters(**kwargs)

        # Get Forcings
        prec = forcings["prec"].values
        if "pet" in forcings.columns:
            pet = forcings["pet"].values
        else:
            month = forcings.index.month.values
            tmean = forcings["tmean"].values
            pet = pet_models._pet_pruitt(
                tmean,
                month,
                self.params["kc"]
            )

        # Compute Water Balance
        simulations = _milc(
            prec,
            pet,
            self.area,
            self.params["gamma"],
            self.params["w0"],
            self.params["wmax"],
            self.params["alpha"],
            self.params["m"],
            self.params["ks"],
            self.params["nu"],
            dt
        )

        # Save final storage state
        if save_state:
            self.params["w0"] = simulations[-1][-1]

        # Save Outputs
        outputs = pd.DataFrame(
            {
                "qt": simulations[0],
                "qd": simulations[1],
                "qb": simulations[2],
                "runoff": simulations[3],
                "baseflow": simulations[4],
                "pet": pet,
                "et": simulations[5],
                "infil": simulations[6],
                "perc": simulations[7],
                "ws": simulations[8],
            },
            index=forcings.index
        )

        return outputs


# ==============================================================================
# Subroutines for model processes
# ==============================================================================

@nb.jit(nopython=True)
def factorial(n):
    # Factorial function to be used in numba functions
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]


@nb.jit(nopython=True)
def _actual_evapotranspiration(pet, w, wmax):
    # Compute evpotranspiration
    return max(0.0, pet * w / wmax)


@nb.jit(nopython=True)
def _runoff(prec, infil):
    # Compute runoff
    return max(0.0, prec - infil)


@nb.jit(nopython=True)
def _infiltration(prec, w, wmax, alpha):
    # Compute infiltlration
    infil = max(0.0, prec * (1. - (w / wmax) ** alpha))
    # Check soil saturation
    if infil + w >= wmax:
        water_excess = infil + w - wmax
        infil -= water_excess
    return infil


@nb.jit(nopython=True)
def _baseflow(w, wmax, m, ks, nu):
    # Compute baseflow
    baseflow = (1.0 - nu) * ks * (w / wmax) ** m
    return max(0.0, baseflow)


@nb.jit(nopython=True)
def _percolation(w, wmax, m, ks, nu):
    # Compute percolation
    perc = nu * ks * (w / wmax) ** m
    return max(0.0, perc)


@nb.jit(nopython=True)
def _water_content(w, wmax):
    # Compute Water Content
    return w / wmax


@nb.jit(nopython=True)
def _iuh_comp(gamma, area, dt, delta_T):
    """
    Calculation of geomorphological Instantaneos Unit Hydrograph
    using the geomorphological approach of Gupta et al. (1980)

    Inputs:
        gamma      >   [float] coefficient lag-time relationship
                           Lag = gamma * 1.19 * area ^ 0.33
        area       >   [float] basin area in squared kilometers
        dt         >   [float] computational time step for flood
                           event simulation, in hours
        delta_T    >   [float] input time step of the time series

    Outputs:
        IUH        >   [array] output Instaneous Unit Hydrograph
    """
    lag = (gamma * 1.19 * area ** 0.33) / delta_T
    hp = 0.8 / lag
    t = IUH_DATA[:, 0] * lag
    IUH0 = IUH_DATA[:, 1] * hp
    t_i = np.arange(0, np.max(t), dt)
    IUH = np.interp(t_i, t, IUH0)
    return IUH


@nb.jit(nopython=True)
def _iuh_nash(n, gamma, area, dt, delta_T):
    """
    Nash Instantaneos Unit Hydrograph used for baseflow routine

    Inputs:
        n          >   [float] model parameter
        gamma      >   [float] coefficient lag-time relationship
                           Lag = gamma * 1.19 * area ^ 0.33
        area       >   [float] basin area in squared kilometers
        dt         >   [float] computational time step for flood
                           event simulation, in hours
        delta_T    >   [float] input time step of the time series
    """
    K = (gamma * 1.19 * area ** 0.33) / delta_T
    t = np.arange(0, 100, dt)
    IUH = (t / K) ** (n - 1) * np.exp(-t / K) / factorial(int(n - 1)) / K
    return IUH


@nb.jit(nopython=True)
def _convolution_giuh(runoff, baseflow, area, gamma, dt):
    """
    Compute streamflow hydrograph at the catchment outlet
    using Geomorphological Instantaneos Unit Hydrograph for runoff and
    Nash Instantaneos Unit Hydrograph for baseflow.

    Inputs:
        runoff     >   [array] runoff serie
        baseflow   >   [array] baseflow serie
        dt         >   [float] computational time step for flood
                           event simulation, in hours
        delta_T    >   [float] input time step of the time series
    """
    delta_T = 24.0    # time delta in hours
    n = len(runoff)

    # Compute runoff IUH
    IUH1 = _iuh_comp(gamma, area, dt, delta_T) * dt
    IUH1 /= np.sum(IUH1)

    # Compute baseflow IUH
    IUH2 = _iuh_nash(1.0, 0.5 * gamma, area, dt, delta_T) * dt
    IUH2 /= np.sum(IUH2)

    # Convolution to compute hydrographs
    qs_int = np.interp(np.arange(0, n, dt), np.arange(n), runoff)
    bf_int = np.interp(np.arange(0, n, dt), np.arange(n), baseflow)

    temp1 = np.convolve(IUH1, qs_int)
    temp2 = np.convolve(IUH2, bf_int)

    # Compute total flow and baseflow
    dt1 = np.round(1. / dt)
    idx = np.arange(0, n * dt1, dt1, dtype=np.int32)

    factor = area * 1000. / delta_T / 3600.
    qd = temp1[idx] * factor  # routed runoff
    qb = temp2[idx] * factor  # routed baseflow

    return qd, qb


@nb.jit(nopython=True)
def _milc(prec, pet, area, gamma, w0, wmax, alpha, m, ks, nu, dt):
    """
    Modello Idrologico Lumped in Continuo (MILC)
    """

    # Initial parameters
    n = len(prec)
    w = w0 * wmax  # water storage in mm
    # ks * 24      # convert mm/hr to mm

    # Create empty arrays
    runoff = np.zeros(n, dtype=np.float32)    # direct flow
    baseflow = np.zeros(n, dtype=np.float32)  # baseflow
    et = np.zeros(n, dtype=np.float32)        # evapotranspiration
    infil = np.zeros(n, dtype=np.float32)     # infiltration
    perc = np.zeros(n, dtype=np.float32)      # percolation
    ww = np.zeros(n, dtype=np.float32)        # water content

    for t in range(n):
        # Surface processes
        infil[t] = _infiltration(prec[t], w, wmax, alpha)
        runoff[t] = _runoff(prec[t], infil[t])
        w += infil[t]  # update Water Storage
        # Subsurface processes
        et[t] = _actual_evapotranspiration(pet[t], w, wmax)
        w -= et[t]  # update Water Storage
        # Deep processes
        baseflow[t] = _baseflow(w, wmax, m, ks, nu)
        perc[t] = _percolation(w, wmax, m, ks, nu)
        w -= baseflow[t] + perc[t]  # update Water Storage
        # Compute Water Content
        ww[t] = _water_content(w, wmax)

    # Flow routing using IUH
    qd, qb = _convolution_giuh(runoff, baseflow, area, gamma, dt)
    qt = qb + qd

    return qt, qd, qb, runoff, baseflow, et, infil, perc, ww

