# -*- coding: utf-8 -*-
"""
modele du Genie Rural a 4 parametres Journalier (GR4j)

Rain-Runoff Model

Author:
Andres Felipe Duque Pérez
INGENEVO S.A.S
andresfduque@gmail.com | aduquep@ingenevo.com.co

Based on:
Saul Arciniega Esparza
https://gitlab.com/Zaul_AE/lumod

Reference:
Perrin, C. (2002). Vers une amélioration d'un modèle global pluie-débit au travers d'une approche comparative.
La Houille Blanche, n°6/7, 84-91.
Perrin, C., Michel, C., Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation.
Journal of Hydrology 279(1-4), 275-289.
"""

import math
from math import tanh
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

class GR4J(BaseModel):
    """GR4J Runoff Model

    Args:
        BaseModel (object): Base model object
    """
    def __init__(self, area=100, lat=0, params=None):
        """
        GR4J hydrological model for daily simulation


        Inputs:
            area    >     [float] catchment area in km2
            lat     >     [float] catchment latitude at centroid
            params  >     [dict] model parameters

        Model Parameters
            ps0     >     [float] initial production storage as a fraction ps0=(ps/X1)
            rs0     >     [float] initial routing storage as a fraction pr0=(rs/X3)
            x1      >     [float] maximum production capacity (mm)
            x2      >     [float] discharge parameter (mm)
            x3      >     [float] routing maximum capacity (mm)
            x4      >     [float] HU1 unit hydrograph time base (days)

        Reference:
        Perrin, C. (2002). Vers une amélioration d'un modèle global pluie-débit au travers d'une approche comparative.
        La Houille Blanche, n°6/7, 84-91.
        Perrin, C., Michel, C., Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation.
        Journal of Hydrology 279(1-4), 275-289.
        """
        super().__init__(area, lat)

        self.params = {
            "ps0": 1.0, 
            "rs0": 0.5,
            "x1": 500.0,
            "x2": 3.0,
            "x3": 200.0,
            "x4": 5.0,
        }

        if params is not None:
            self.set_parameters(**params)

    def __repr__(self):
        text = "\n______________GR4J structure______________\n"
        text += "Catchment properties:\n"
        text += f"    Area (km2): {self.area:.3f}\n"
        text += f"    Latitude  : {self.lat:.4f}\n"
        text += "Model Parameters:\n"
        text += f"    x1  > Maximum production capacity (mm)     : {self.params['x1']:.3f}\n"
        text += f"    x2  > Discharge parameter (mm)             : {self.params['x2']:.3f}\n"
        text += f"    x3  > Routing maximum capacity (mm)        : {self.params['x3']:.3f}\n"
        text += f"    x4  > Delay (days)                         : {self.params['x4']:.3f}\n"
        text += f"    ps0 > Initial production storage (ps/x1)   : {self.params['ps0']:.3f}\n"
        text += f"    rs0 > Initial routing storage (rs/x3)      : {self.params['rs0']:.3f}\n"
        return text

    def __str__(self):
        x1 = self.params["x1"]
        x2 = self.params["x2"]
        x3 = self.params["x3"]
        x4 = self.params["x4"]
        ps0 = self.params["ps0"]
        rs0 = self.params["rs0"]
        text = f"GR4J(area={self.area:.2f},lat={self.lat:.3f},"
        text += f"x1={x1:.3f},x2={x2:.3f},x3={x3:.3f},x4={x4:.3f},"
        text += f"ps0={ps0:.3f},rs0={rs0:.3f})"
        return text

    def run(self, forcings, start=None, end=None, save_state=False, **kwargs):
        """
        Run the GR4J model


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
        **kwargs :
            Model parameters can be changed for the simulation
                area    >     [float] catchment area in km2
                lat     >     [float] catchment latitude
                ps0     >     [float] initial production storage as a fraction ps0=(ps/X1)
                rs0     >     [float] initial routing storage as a fraction pr0=(rs/X3)
                x1      >     [float] maximum production capacity (mm)
                x2      >     [float] discharge parameter (mm)
                x3      >     [float] routing maximum capacity (mm)
                x4      >     [float] HU1 unit hydrograph time base (days)

        Returns
        -------
        Simulations : DataFrame
            qt       > Streamflow (qd+qb) at catchment output (m3/s)
            qd       > Direct flow at catchment output (m3/s)
            qb       > Base flow at catchment output (m3/s)
            pet      > Potential Evapotranspiration (mm)
            gwe      > Groundwater Exchange (mm)
            ps       > Production storage as a fraction of x1 (-)
            rs       > Routing storage as a fraction of x3 (-)
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
            tmean = forcings["tmean"].values
            doy = forcings.index.dayofyear.values  # day of year
            pet = pet_models._pet_oudin(tmean, doy, self.lat)

        simulations = _gr4j(
            prec,
            pet,
            self.params["x1"],
            self.params["x2"],
            self.params["x3"],
            self.params["x4"],
            self.params["ps0"],
            self.params["rs0"]
        )

        # Create Output DataFrame
        outputs = pd.DataFrame(
            {
                "qt": simulations[0],
                "qd": simulations[1],
                "qb": simulations[2],
                "pet": pet,
                "gwe": simulations[3],
                "ps": simulations[4],
                "rs": simulations[5],
            },
            index=forcings.index,
        )

        # Convert units mm/d to m3/s
        outputs.loc[:, "qt"] = outputs.loc[:, "qt"] * self.area / 86.4
        outputs.loc[:, "qd"] = outputs.loc[:, "qd"] * self.area / 86.4
        outputs.loc[:, "qb"] = outputs.loc[:, "qb"] * self.area / 86.4

        # Save final storage state
        if save_state:
            psto = simulations[4][-1]
            rsto = simulations[5][-1]
            self.params["ps0"] = psto
            self.params["rs0"] = rsto

        return outputs


# ==============================================================================
# Subroutines for model processes
# ==============================================================================

@nb.jit(nopython=True)
def _reservoirs_evaporation(prec, pet, ps, x1):
    """
    Estimate net evapotranspiration and reservoir production
    """
    if prec > pet:
        evap = 0.
        snp = (prec - pet) / x1     # scaled net precipitation
        snp = min(snp, 13.)         # minimum (why 13?)
        tsnp = tanh(snp)            # tanh scaled net precipitation

        # reservoir production
        res_prod = ((x1 * (1. - (ps / x1) ** 2.) * tsnp)
                    / (1. + ps / x1 * tsnp))

        # percolation (can be ignored)
        if ps != 0:
            perc = ps * (1 - (1 + 4 * ps / (9 * x1)) ** 4) ** (-0.25)
        else: perc = 0

        # routing pattern
        rout_pat = perc + (prec - pet) - res_prod   # rout path = Pr

    else:
        sne = (pet - prec) / x1     # scaled net evapotranspiration
        sne = min(sne, 13.)         # minimum (why 13?)
        tsne = tanh(sne)            # tanh scaled net evapotranspiration
        ps_div_x1 = (2. - ps / x1) * tsne
        evap = ps * ps_div_x1 / (1. + (1. - ps / x1) * tsne)    # evap = Es

        # percolation (can be ignored)
        if ps != 0:
            perc = ps * (1 - (1 + 4 * ps / (9 * x1)) ** 4) ** (-0.25)
        else: perc = 0

        # reservoir production and flow routing
        res_prod = 0
        rout_pat = 0

        # routing pattern
        rout_pat = perc     # if no effective precipitation then the routing is given by percolation

    return evap, res_prod, rout_pat, perc


@nb.jit(nopython=True)
def _s_curves1(t, x4):
    """
    Unit hydrograph ordinates for UH1 derived from S-curves.
    """
    if t <= 0:
        return 0
    elif t < x4:
        return (t / x4) ** 2.5
    else:  # t >= x4
        return 1


@nb.jit(nopython=True)
def _s_curves2(t, x4):
    """
    Unit hydrograph ordinates for UH2 derived from S-curves.
    """
    if t <= 0:
        return 0
    elif t < x4:
        return 0.5 * (t / x4) ** 2.5
    elif t < 2 * x4:
        return 1 - 0.5 * (2 - t / x4) ** 2.5
    else:  # t >= x4
        return 1


@nb.jit(nopython=True)
def _compute_unitary_hydrograph(x4):

    nuh1 = int(math.ceil(x4))
    nuh2 = int(math.ceil(2.0 * x4))
    uh1 = np.zeros(nuh1)
    uh2 = np.zeros(nuh2)
    uh1_ordinates = np.zeros(nuh1)
    uh2_ordinates = np.zeros(nuh2)

    for t in range(1, nuh1 + 1):
        uh1_ordinates[t - 1] = _s_curves1(t, x4) - _s_curves1(t - 1, x4)

    for t in range(1, nuh2 + 1):
        uh2_ordinates[t - 1] = _s_curves2(t, x4) - _s_curves2(t - 1, x4)

    ouh1 = uh1_ordinates
    ouh2 = uh2_ordinates
    return ouh1, ouh2, uh1, uh2


@nb.jit(nopython=True)
def _compute_hydrograph(rout_pat, ouh1, ouh2, uh1, uh2):
    """
    Daily hydrograph for catchment routine
    """
    for i in range(0, len(uh1) - 1):
        uh1[i] = uh1[i + 1] + ouh1[i] * rout_pat
    uh1[-1] = ouh1[-1] * rout_pat

    for j in range(0, len(uh2) - 1):
        uh2[j] = uh2[j + 1] + ouh2[j] * rout_pat
    uh2[-1] = ouh2[-1] * rout_pat

    return uh1, uh2


@nb.jit(nopython=True)
def _compute_exchange(uh1, rout_sto, x2, x3):
    # groundwater exchange
    gw_exc = x2 * (rout_sto / x3) ** 3.5    # F
    rout_sto = max(0, rout_sto + uh1[0] * 0.9 + gw_exc) # R
    return gw_exc, rout_sto


@nb.jit(nopython=True)
def _compute_discharge(uh2, gw_exc, rout_sto, x3):
    new_rout_sto = rout_sto / (1. + (rout_sto / x3) ** 4.0) ** 0.25
    qr = rout_sto - new_rout_sto
    rout_sto = new_rout_sto
    qd = max(0, uh2[0] * 0.1 + gw_exc)
    return qr, qd, rout_sto


@nb.jit(nopython=True)
def _gr4j(prec, pet, x1, x2, x3, x4, ps0, rs0):

    # Create empty arrays
    n = len(prec)
    qtarray = np.zeros(n, dtype=np.float32)
    qdarray = np.zeros(n, dtype=np.float32)
    qrarray = np.zeros(n, dtype=np.float32)
    gwarray = np.zeros(n, dtype=np.float32)
    psarray = np.zeros(n, dtype=np.float32)
    rsarray = np.zeros(n, dtype=np.float32)

    # Initial parameters
    ouh1, ouh2, uh1, uh2 = _compute_unitary_hydrograph(x4)
    psto = ps0 * x1
    rsto = rs0 * x3

    # Compute water partioning
    for t in range(n):
        res = _reservoirs_evaporation(prec[t], pet[t], psto, x1)
        evap, res_prod, rout_pat = res

        psto = psto - evap + res_prod
        perc = psto / (1. + (psto / 2.25 / x1) ** 4.) ** 0.25
        rout_pat = rout_pat + (psto - perc)
        psto = perc

        uh1, uh2 = _compute_hydrograph(rout_pat, ouh1, ouh2, uh1, uh2)

        gw_exc, rsto = _compute_exchange(uh1, rsto, x2, x3)

        qr, qd, rsto = _compute_discharge(uh2, gw_exc, rsto, x3)
        qt = qr + qd

        # Save outputs
        qtarray[t] = qt  # total flow
        qdarray[t] = qd  # runoff
        qrarray[t] = qr  # baseflow
        gwarray[t] = gw_exc  # groundwater exchange
        psarray[t] = psto / x1  # production storage
        rsarray[t] = rsto / x3  # routing storage

    return qtarray, qdarray, qrarray, gwarray, psarray, rsarray

