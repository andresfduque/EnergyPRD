# -*- coding: utf-8 -*-
"""
HYdrological MODel (HYMOD)

Rain-Runoff Model for daily simulation


Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx

Based on:
Roy, T., H. V. Gupta, A. Serrat-Capdevila, J. B. Valdes (2017). HYMOD2 Model MATLAB Code,
HydroShare, https://doi.org/10.4211/hs.26c1d7a19e544718851181ac6e9f0fdc

Reference:
Tirthankar Roy (royt@email.arizona.edu)
Copyright: Hoshin V Gupta and Tirthankar Roy (University of Arizona)

Roy, T., Gupta, H. V., Serrat-Capdevila, A. and Valdes, J. B.: Using satellite-based
evapotranspiration estimates to improve the structure of a simple conceptual
rainfall-runoff model, Hydrol. Earth Syst. Sci., 21(2), 879–896,
doi:10.5194/hess-21-879-2017, 2017.

Quan, Z., Teng, J., Sun, W., Cheng, T., & Zhang, J. (2015).
Evaluation of the HYMOD model for rainfall-runoff simulation using the GLUE method.
IAHS-AISH Proceedings and Reports, 368(August 2014), 180–185.
https://doi.org/10.5194/piahs-368-180-2015
"""

#%% Import libraries
import numpy as np
import pandas as pd
import numba as nb
from ..fluxes import pet_models
from .base_model import BaseModel
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# Main class
# ==============================================================================

class HYMOD(BaseModel):

    def __init__(self, area=100, lat=0, params=None):
        """
        simple HYdrological MODel (HYMOD)


        INPUTS:
            area    >     [float] catchment area in km2
            lat     >     [float] catchment latitude at centroid
            params  >     [dict] model parameters

        Model Parameters
            wmax    >     [float] maximum water content (mm)
            w0      >     [float] initial soil water content as a fraction of wmax (-)
            wq0     >     [float] mean initial water in quick reservoirs (mm)
            ws0     >     [float] initial water in slow reservoir (mm)
            alpha   >     [float] quick-slow split parameter (-) [0,1]
            beta    >     [float] distribution function shape parameter (-) [0,2]
            cexp    >     [float] exponent on ratio of maximum storage capacities (-) [0,2]
            nres    >     [int] number of quickflow routing reservoirs (integer, -)
            ks      >     [float] residence time of the slow release reservoir (1/days)
            kq      >     [float] residence time of the quick release reservoirs (1/days)
            kmax    >     [float] upper limit of ET resistance parameter (-)
            llet    >     [float] lower limit of ET resistance parameter (-) [0, 1.0)


        Based on:
        Roy, T., H. V. Gupta, A. Serrat-Capdevila, J. B. Valdes (2017). HYMOD2 Model MATLAB Code,
        HydroShare, https://doi.org/10.4211/hs.26c1d7a19e544718851181ac6e9f0fdc

        HYMOD moddified version from:
        Roy, T., Gupta, H. V., Serrat-Capdevila, A. and Valdes, J. B. (2017). Using satellite-based
        evapotranspiration estimates to improve the structure of a simple conceptual rainfall-runoff model,
        Hydrol. Earth Syst. Sci., 21(2), 879–896, https://doi.org/10.5194/hess-21-879-2017
        """
        super().__init__(area, lat)

        self.params = {
            "wmax": 800.0,  # maximum water content (mm)
            "w0": 0.5,  # initial soil water content as a fraction of wmax (-)
            "wq0": 0.0,  # mean initial water in quick reservoirs (mm)
            "ws0": 0.0,  # initial water in slow reservoir (mm)
            "alpha": 0.3,  # quick-slow split parameter (-) [0,1]
            "beta": 1.0,  # distribution function shape parameter (-) [0,2]
            "cexp": 0.7,  # exponent on ratio of maximum storage capacities (-) [0,2]
            "nres": 3,  # number of quickflow routing reservoirs (integer, -)
            "ks": 0.05,  # residence time of the slow release reservoir (days)
            "kq": 0.3,  # residence time of the quick release reservoirs (days)
            "kmax": 0.9,  # upper limit of ET resistance parameter (-)
            "llet": 0.2,  # lower limit of ET resistance parameter (-) [0, 1.0)
        }

        if params is not None:
            self.set_parameters(**params)

    def __repr__(self):
        text = "\n____________HYMOD structure_____________\n"
        text += "Catchment properties:\n"
        text += "    Area (km2): {:.3f}\n".format(self.area)
        text += "    Latitude  : {:.4f}\n".format(self.lat)
        text += "Model Parameters:\n"
        text += "    wmax  > Maximum Water Capacity (mm)              : {:.3f}\n".format(self.params["wmax"])
        text += "    w0    > Initial Water Content (adim)             : {:.3f}\n".format(self.params["w0"])
        text += "    wq0   > Initial water in quick reservoirs (mm)   : {:.3f}\n".format(self.params["wq0"])
        text += "    ws0   > Initial water in slow reservoir (mm)     : {:.3f}\n".format(self.params["ws0"])
        text += "    alpha > Quick-slow split parameter (-) [0,1]     : {:.3f}\n".format(self.params["alpha"])
        text += "    beta  > Function shape parameter (-) [0,2]       : {:.3f}\n".format(self.params["beta"])
        text += "    cexp  > Exponent of maximum storage (-) [0,2]    : {:.3f}\n".format(self.params["cexp"])
        text += "    nres  > Number of quickflow reservoirs (int, -)  : {:.3f}\n".format(self.params["nres"])
        text += "    ks    > Residence time of the slow flow (1/days) : {:.3f}\n".format(self.params["ks"])
        text += "    kq    > Residence time of the quick flow (1/days): {:.3f}\n".format(self.params["kq"])
        text += "    kmax  > Lower limit of ET resistance (-) [0, 1.0): {:.3f}\n".format(self.params["kmax"])
        text += "    llet  > Lower limit of ET resistance (-) [0, 1.0): {:.3f}\n".format(self.params["llet"])
        return text

    def __str__(self):
        text = f"HYDMOD(area={self.area:.2f},lat={self.lat:.3f},...)"
        return text

    def run(self, forcings, start=None, end=None, save_state=False, **kwargs):
        """
        Run HYMOD


        Parameters
        ----------
        forcings : DataFrame
            Input data with columns prec (precipitation), tmin, tmax, and
            pet(potential evapotranspiration, optional).
        start : string, optional
            Start date for simulation in format. Example: '2001-01-01'
        end : string, optional
            End date for simulation in format. Example: '2010-12-31'
        save_state : bool, optional
            If True, save the last simulated state of reservoirs
        **kwargs :
            Model parameters can be changed for the simulation
                area    >     [float] catchment area in km2
                wmax    >     [float] maximum water content (mm)
                w0      >     [float] initial soil water content as a fraction of wmax (-)
                wq0     >     [float] mean initial water in quick reservoirs (mm)
                ws0     >     [float] initial water in slow reservoir (mm)
                alpha   >     [float] quick-slow split parameter (-) [0,1]
                beta    >     [float] distribution function shape parameter (-) [0,2]
                cexp    >     [float] exponent on ratio of maximum storage capacities (-) [0,2]
                nres    >     [int] number of quickflow routing reservoirs (integer, -)
                ks      >     [float] residence time of the slow release reservoir (1/days)
                kq      >     [float] residence time of the quick release reservoirs (1/days)
                kmax    >     [float] upper limit of ET resistance parameter (-)
                llet    >     [float] lower limit of ET resistance parameter (-) [0, 1.0)

        Returns
        -------
        Simulations : DataFrame
            qt       > Streamflow at catchment outlet (m3/s)
            qd       > Quickflow at catchment outlet (m3/s)
            qb       > Slow flow at catchment outlet (m3/s)
            peff     > Effective precipitation (mm)
            pet      > Potential Evapotranspiration (mm)
            et       > Evapotranspiration (mm)
            infil    > Infiltration (mm)
            ww       > Water Soil Moisture Content dimensionless (w/wmax)
            wq       > Water in quick reservoir (mm)
            ws       > Water in slow reservoir (mm)
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
            doy = forcings.index.dayofyear.values
            tmin = forcings["tmin"].values
            tmax = forcings["tmax"].values
            if "tmean" in forcings.columns:
                tmean = forcings["tmean"].values
            else:
                tmean = (tmin + tmax) / 2.0
            pet = pet_models._pet_hargreaves(
                tmin,
                tmax,
                tmean,
                doy,
                self.lat
            )

        # Compute Water Balance
        simulations = _hymod(
            prec,
            pet,
            self.area,
            self.params["wmax"],
            self.params["w0"],
            self.params["wq0"],
            self.params["ws0"],
            self.params["alpha"],
            self.params["beta"],
            self.params["cexp"],
            self.params["nres"],
            self.params["ks"],
            self.params["kq"],
            self.params["kmax"],
            self.params["llet"]
        )

        # Save final storage state
        if save_state:
            self.params["w0"] = simulations[6][-1]
            self.params["wq0"] = simulations[7][-1]
            self.params["ws0"] = simulations[8][-1]

        # Save Outputs
        outputs = pd.DataFrame(
            {
                "qt": simulations[0],
                "qd": simulations[1],
                "qb": simulations[2],
                "peff": simulations[3],
                "pet": pet,
                "et": simulations[5],
                "infil": simulations[4],
                "ww": simulations[6],
                "wq": simulations[7],
                "ws": simulations[8]
            },
            index=forcings.index,
        )

        return outputs


# ==============================================================================
# Subroutines for model processes
# ==============================================================================

@nb.jit(nopython=True)
def _maximum_storage_capacity(wmax, b):
    # Compute cmax
    return wmax / (1.0 + b)


@nb.jit(nopython=True)
def _soil_moisture_content(w, wmax, cmax, b):
    return cmax * (1.0 - (1.0 - (w / wmax) ** (1.0 + b)))


@nb.jit(nopython=True)
def _soil_moisture_module(prec, w, wmax, cmax, b):

    cbeg = _soil_moisture_content(w, wmax, cmax, b)  # Contents at begining
    peff2 = max(0.0, prec + w - wmax)  # Compute overland flow if enough precipitation
    infil = prec - peff2  # precipitation that does not go to overland flow
    w = min(wmax, infil + w)  # Intermediate height
    cinit = _soil_moisture_content(w, wmax, cmax, b)  # Intermediate contents
    peff1 = max(0.0, infil + cbeg - cinit)
    peff = peff1 + peff2  # compute overland flow
    return w, cinit, peff, infil


@nb.jit(nopython=True)
def _compute_et(pet, cinit, cmax, kmin, kmax, ce):
    """
    Compute evpotranspiration related to the soil moisture state
    """
    k = kmin + (kmax - kmin) * (cinit / cmax) ** ce
    return k * min(pet, cinit)


@nb.jit(nopython=True)
def _flow_partitioning(alpha, peff):
    xs = (1.0 - alpha) * peff  # slow flow
    xq = alpha * peff          # quick flow
    return xq, xs


@nb.jit(nopython=True)
def _linear_reservoir(storage, inflow, k):
    """
    Compute linear reservoir simulation
    """
    outflow = k * storage
    sto = max(0.0, storage - outflow + inflow)
    return sto, outflow


@nb.jit(nopython=True)
def _hymod(prec, pet, area, wmax, w0, wq0, ws0, alpha, beta, cexp, nres, ks, kq, kmax, llet):

    # Initial parameters
    n = len(prec)
    nres = int(nres)  # number of quick reservoirs
    w = w0 * wmax  # water storage in mm
    x_slow = ws0   # initial slow storage
    x_quick = wq0 + np.zeros(nres, dtype=np.float32)  # initial quick storage

    # Convert from scaled B (0-2) to unscaled b (0 - Inf)
    beta = min(max(beta, 0), 2)
    if beta == 2:
        b = 10.0 ** 6.0
    else:
        b = np.log(1.0 - beta / 2.0) / np.log(0.5)
    # Convert from scaled CE (0-2) to unscaled ce (0 - Inf)
    cexp = min(max(cexp, 0), 2)
    if cexp == 2:
        ce = 10.0 ** 6.0
    else:
        ce = np.log(1.0 - cexp / 2.0) / np.log(0.5)
    # Maximum capacity of soil zone
    cmax = _maximum_storage_capacity(wmax, b)
    kmin = llet * kmax

    # Create empty arrays
    qd = np.zeros(n, dtype=np.float32)  # routed quick flow
    qb = np.zeros(n, dtype=np.float32)  # routed slow flow
    peff = np.zeros(n, dtype=np.float32)  # effective precipitation
    infil = np.zeros(n, dtype=np.float32)  # infiltration
    et = np.zeros(n, dtype=np.float32)  # evapotranspiration
    ww = np.zeros(n, dtype=np.float32)  # water content
    wq = np.zeros(n, dtype=np.float32)  # mean storage in quick storages
    ws = np.zeros(n, dtype=np.float32)  # storage in slow storage

    for t in range(n):

        # Soil moisture computation
        w, cinit, peff[t], infil[t] = _soil_moisture_module(prec[t], w, wmax, cmax, b)
        # Compute evapotranspiration
        et[t] = _compute_et(pet[t], cinit, cmax, kmin, kmax, ce)
        # Update storage
        cend = min(max(cinit - et[t], 0.0), cmax)
        w = wmax * (1.0 - (1.0 - cend / cmax) ** (1.0 / (1.0 + b)))
        # peff partitioning
        uq, us = _flow_partitioning(alpha, peff[t])
        # Slow reservoir
        x_slow, qsout = _linear_reservoir(x_slow, us, ks)
        # Quick reservoir
        inflow = uq
        for i in range(nres):
            x_quick[i], qqout = _linear_reservoir(x_quick[i], inflow, kq)
            inflow = qqout
        # Save results
        qd[t] = qqout
        qb[t] = qsout
        ww[t] = w / wmax
        wq[t] = np.mean(x_quick)
        ws[t] = x_slow

    factor = area * 1000. / 86400.  # convert mm/d to m3/s
    qd *= factor
    qb *= factor
    qt = qd + qb
    # qt=1, qd=1, qb=2, peff=3, infil=4, et=5, ww=6, wq=7, ws=8
    return qt, qd, qb, peff, infil, et, ww, wq, ws

