# -*- coding: utf-8 -*-
"""
Conceptual Hydrologiska Byråns Vattenbalansavdelning (HBV) model

Rain-Runoff Model


Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx

Based on:
HRL (2021). HBV-EDU Hydrologic Model (https://www.mathworks.com/matlabcentral/fileexchange/41395-hbv-edu-hydrologic-model),
MATLAB Central File Exchange. Retrieved November 4, 2021.

Reference:
AghaKouchak A., Habib E., 2010, Application of a Conceptual Hydrologic
Model in Teaching Hydrologic Processes, International Journal of Engineering Education, 26(4), 963-973.
"""

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

class HBV(BaseModel):

    def __init__(self, area=100, lat=0, params=None):
        """
        Conceptual Hydrologiska Byråns Vattenbalansavdelning (HBV) model


        INPUTS:
            area    >     [float] catchment area in km2
            lat     >     [float] catchment latitude at centroid
            params  >     [dict] model parameters

        Model Parameters
            maxbas  >     [int] weighting parameter used for triangular unit hydrograph (days)
            tthres  >     [float] threshold temperature for snow melt initiation (°C)
            dd      >     [float] degree-day factor for snow accumulation (mm/°C.d)
            cevp    >     [float] PET parameter that depends of land use (mm/day.°C)
            cevpam  >     [float] amplitude of sinus function that corrects PET (-)
            cevpph  >     [int] phase of sinus function that corrects pet (days)
            beta    >     [float] shape coefficient for runoff (-)
            fc      >     [float] maximum soil storage capacity (mm)
            pwp     >     [float] soil permanente wilting point as a fraction of fc (-) [0,1]
            k0      >     [float] recession coefficient of surface flow (1/d)
            k1      >     [float] recession coefficient of interflow (1/d)
            k2      >     [float] recession coefficient of baseflow (1/d)
            kp      >     [float] recession coefficient of percolation (1/d)
            lthres  >     [float] threshold water level for generating surface flow (mm)
            snow0   >     [float] initial snow equivalent thickness (mm)
            s0      >     [float] initial soil moisture storage (s/fc) [0-1]
            w01     >     [float] upper reservoir storage (mm)
            w02     >     [float] lower reservoir storage (mm)

        Based on:
        HRL (2021). HBV-EDU Hydrologic Model (https://www.mathworks.com/matlabcentral/fileexchange/41395-hbv-edu-hydrologic-model),
        MATLAB Central File Exchange. Retrieved November 4, 2021.

        References:
        AghaKouchak A., Habib E., 2010, Application of a Conceptual Hydrologic Model in Teaching Hydrologic Processes,
        International Journal of Engineering Education, 26(4), 963-973.
        Bergström S. (1976). Development and application of a conceptual runoff model for Scandinavian catchments,
        SMHI RHO 7, Norrköping, Sweden.
        """
        super().__init__(area, lat)

        self.params = {
            "maxbas": 3,
            "tthres": 5.0,
            "dd": 2.0,
            "cevp": 2.0,
            "cevpam": 1.0,
            "cevpph": 0,
            "beta": 2.0,
            "fc": 500.0,
            "pwp": 0.8,
            "k0": 0.5,
            "k1": 0.1,
            "k2": 0.01,
            "kp": 0.05,
            "lthres": 50,
            "snow0": 0,
            "s0": 0.5,
            "w01": 20,
            "w02": 100,
        }

        if params is not None:
            self.set_parameters(**params)

    def __repr__(self):
        text = "\n_______________HBV structure______________\n"
        text += "Catchment properties:\n"
        text += "    Area (km2): {:.3f}\n".format(self.area)
        text += "    Latitude  : {:.4f}\n".format(self.lat)
        text += "Model Parameters:\n"
        text += "    maxbas > time for triangular u. hydrograph (days): {:.3f}\n".format(self.params["maxbas"])
        text += "    tthres > threshold temp for snow melt init. (°C) : {:.3f}\n".format(self.params["tthres"])
        text += "    dd     > degree-day for snow accum. (mm/°C.d)    : {:.3f}\n".format(self.params["dd"])
        text += "    cevp   > PET parameter of land use (mm/day.°C)   : {:.3f}\n".format(self.params["cevp"])
        text += "    cevpam > amplitude of sinus function for PET (-) : {:.3f}\n".format(self.params["cevpam"])
        text += "    cevpph > phase of sinus function for pet (days)  : {:.3f}\n".format(self.params["cevpph"])
        text += "    beta   > shape coefficient for runoff (-)        : {:.3f}\n".format(self.params["beta"])
        text += "    fc     > maximum soil storage capacity (mm)      : {:.3f}\n".format(self.params["fc"])
        text += "    pwp    > soil permanente wilting point (1/fc) (-): {:.3f}\n".format(self.params["pwp"])
        text += "    k0     > recession coef. of surface flow (1/d)   : {:.3f}\n".format(self.params["k0"])
        text += "    k1     > recession coef. of interflow (1/d)      : {:.3f}\n".format(self.params["k1"])
        text += "    k2     > recession coef. of baseflow (1/d)       : {:.3f}\n".format(self.params["k2"])
        text += "    kp     > recession coef. of percolation (1/d)    : {:.3f}\n".format(self.params["kp"])
        text += "    lthres > thres. water level for surface flow (mm): {:.3f}\n".format(self.params["lthres"])
        text += "    snow0  > initial snow equivalent thickness (mm)  : {:.3f}\n".format(self.params["snow0"])
        text += "    s0     > initial soil moisture storage (s/fc)    : {:.3f}\n".format(self.params["s0"])
        text += "    w01    > upper reservoir storage (mm)            : {:.3f}\n".format(self.params["w01"])
        text += "    w02    > lower reservoir storage (mm)            : {:.3f}\n".format(self.params["w02"])
        return text

    def __str__(self):
        text = f"HBV(area={self.area:.2f},lat={self.lat:.3f},...)"
        return text

    def run(self, forcings, start=None, end=None, save_state=False, **kwargs):
        """
        Run the Hydrologiska Byråns Vattenbalansavdelning (HBV) model


        Parameters
        ----------
        forcings : DataFrame
            Input data with columns prec (precipitation), tmean, and
            pet (potential evapotranspiration, optional)
        start : string, optional
            Start date for simulation in format. Example: '2001-01-01'
        end : string, optional
            End date for simulation in format. Example: '2010-12-31'
        save_state : bool, optional
            If True (default), last storage is saved as s0, w01, w02 parameter
        **kwargs :
            Model parameters can be changed for the simulation
                maxbas  >     [int] weighting parameter used for triangular unit hydrograph (days)
                tthres  >     [float] threshold temperature for snow melt initiation (°C)
                dd      >     [float] degree-day factor for snow accumulation (mm/°C.d)
                cevp    >     [float] PET parameter that depends of land use (mm/day.°C)
                cevpam  >     [float] amplitude of sinus function that corrects PET (-)
                cevpph  >     [int] phase of sinus function that corrects pet (days)
                beta    >     [float] shape coefficient for runoff (-)
                fc      >     [float] maximum soil storage capacity (mm)
                pwp     >     [float] soil permanent wilting point as a fraction of fc (-) [0,1]
                k0      >     [float] recession coefficient of surface flow (1/d)
                k1      >     [float] recession coefficient of interflow (1/d)
                k2      >     [float] recession coefficient of baseflow (1/d)
                kp      >     [float] recession coefficient of percolation (1/d)
                lthres  >     [float] threshold water level for generating surface flow (mm)
                snow0   >     [float] initial snow equivalent thickness (mm)
                s0      >     [float] initial soil moisture storage (s/fc) [0-1]
                w01     >     [float] upper reservoir storage (mm)
                w02     >     [float] lower reservoir storage (mm)

        Returns
        -------
        Simulations : DataFrame
            qt        > Streamflow (runoff+interflow+baseflow) routed at catchment output (m3/s)
            runoff    > Near surface flow (mm)
            interflow > Interflow from upper soil storage (mm)
            baseflow  > Baseflow from lower soil storage (mm)
            snow      > Snow depth (mm)
            lqw       > Liquid water from precipitation (mm)
            peff      > Effective precipitation (mm)
            pet       > Potential Evapotranspiration (mm)
            et        > Evapotranspiration (mm)
            infil     > Infiltration (mm)
            perc      > Percolation from upper to lower storage (mm)
            ws        > Water Soil content as a fraction of fc parameter (adim)
            ws1       > Water in upper soil storage (mm)
            ws2       > Water in lower soil storage (mm)
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
        tmean = forcings["tmean"].values
        if "pet" in forcings.columns:
            pet = forcings["pet"].values
        else:
            doy = forcings.index.dayofyear.values
            pet = pet_models._pet_bergstrom(
                tmean,
                doy,
                self.params["cevp"],
                self.params["cevpam"],
                self.params["tthres"],
                self.params["cevpph"]
            )

        # Compute Water Balance
        simulations = _hbv_light(
            prec,
            tmean,
            pet,
            self.area,
            self.params["maxbas"],
            self.params["tthres"],
            self.params["dd"],
            self.params["fc"],
            self.params["beta"],
            self.params["pwp"],
            self.params["k0"],
            self.params["k1"],
            self.params["k2"],
            self.params["kp"],
            self.params["lthres"],
            self.params["snow0"],
            self.params["s0"],
            self.params["w01"],
            self.params["w02"]
        )

        # Save final storage state
        if save_state:
            self.params["s0"] = simulations[10][-1]
            self.params["w01"] = simulations[11][-1]
            self.params["w02"] = simulations[12][-1]

        # Save Outputs
        outputs = pd.DataFrame(
            {
                "qt": simulations[0],
                "runoff": simulations[2],
                "interflow": simulations[3],
                "baseflow": simulations[4],
                "snow": simulations[9],
                "lqw": simulations[6],
                "peff": simulations[1],
                "pet": pet,
                "et": simulations[5],
                "infil": simulations[7],
                "perc": simulations[8],
                "ws": simulations[10],
                "ws1": simulations[11],
                "ws2": simulations[12],
            },
            index=forcings.index
        )

        return outputs


# ==============================================================================
# Subroutines for model processes
# ==============================================================================

@nb.jit(nopython=True)
def _snow_accumulation(sn, prec, tmed, tthres, dd):
    # snow accumulation and snowmelt
    if tmed < tthres:
        sn += prec
        lw = 0.0
    else:
        sn = max(0, sn - dd * (tmed - tthres))
        lw = prec + min(sn, dd * (tmed - tthres))
    return sn, lw  # snow, liquid water


@nb.jit(nopython=True)
def _effective_precipitation(w, inflow, fc, beta):
    # Compute effective precipitation
    return inflow * (w / fc) ** beta


@nb.jit(nopython=True)
def _infiltration(prec, peff):
    # Compute infiltration
    return max(0.0, prec - peff)


@nb.jit(nopython=True)
def _actual_evapotranspiration(pet, w, pwp, fc):
    # Compute actual evapotranspiration
    if w > pwp:
        et = pet
    else:
        et = pet * (w / (pwp * fc))
    return et


@nb.jit(nopython=True)
def _near_surface_flow(w, k, lthres):
    # near surface flow computation
    return max(0.0, (w - lthres) * k)


@nb.jit(nopython=True)
def _linear_reservoir(w, k):
    # linear reservoir discharge
    return w * k


@nb.jit(nopython=True)
def _uh_traingular(maxbas):
    # triangular unitary hydrograph
    delay = maxbas
    tt = np.arange(1, np.ceil(delay) + 1)
    ff = 0.5 / (0.5 * (0.5 * delay) ** 2.0)
    d50 = 0.5 * delay
    tri = lambda t: max(ff * (t - d50) * np.sign(d50 - t) + ff * d50, 0)
    qu = np.zeros(len(tt))
    for i in range(len(tt)):
        h1 = tri(i)
        h2 = tri(i + 1)
        area = (h1 + h2) / 2.0
        qu[i] = area
    qu = qu / np.sum(qu)
    return qu


@nb.jit(nopython=True)
def _convolution_giuh(inflow, area, maxbas):
    # Unitary hydrograph convulution
    n = len(inflow)
    qu = _uh_traingular(maxbas)
    qt = np.convolve(qu, inflow)
    qt = qt[np.arange(n)]
    factor = area / 86.4
    qt *= factor
    return qt


@nb.jit(nopython=True)
def _hbv_light(prec, tmean, pet, area, maxbas, tthres, dd, fc, beta, pwp, k0, k1, k2, kp, lthres,
               snow0, s0, w01, w02):
    # Initial parameters
    n = len(prec)
    sm = s0 * fc  # initial soil moisture storage in mm
    s1 = w01  # upper reservoir storage in mm
    s2 = w02  # lower reservoir storage in mm
    sn = snow0  # initial snow equivalent thickness in mm

    # Create empty arrays
    snow = np.zeros(n, dtype=np.float32)  # snow accumulation
    lqw = np.zeros(n, dtype=np.float32)  # liquid water
    runoff = np.zeros(n, dtype=np.float32)  # direct flow
    et = np.zeros(n, dtype=np.float32)  # evapotranspiration
    infil = np.zeros(n, dtype=np.float32)  # infiltration
    totalflow = np.zeros(n, dtype=np.float32)  # total flow
    nsurflow = np.zeros(n, dtype=np.float32)  # near surface flow
    interflow = np.zeros(n, dtype=np.float32)  # interflow
    baseflow = np.zeros(n, dtype=np.float32)  # baseflow
    perc = np.zeros(n, dtype=np.float32)  # percolation
    ws = np.zeros(n, dtype=np.float32)  # soil moisture as a fraction of fc
    ws1 = np.zeros(n, dtype=np.float32)  # upper reservoir storage
    ws2 = np.zeros(n, dtype=np.float32)  # upper reservoir storage

    for t in range(n):
        # Snow accumulation
        sn, lqw[t] = _snow_accumulation(sn, prec[t], tmean[t], tthres, dd)
        # Surface processes
        runoff[t] = _effective_precipitation(sm, lqw[t], fc, beta)
        # Infiltration
        infil[t] = _infiltration(lqw[t], runoff[t])
        sm += infil[t]
        # Actual evapotranspiration
        et[t] = _actual_evapotranspiration(pet[t], s0, pwp, fc)
        sm -= et[t]
        # Upper soil layer discharge routine
        s1 += runoff[t]
        nsurflow[t] = _near_surface_flow(s1, k0, lthres)
        s1 -= nsurflow[t]
        interflow[t] = _linear_reservoir(s1, k1)
        s1 -= interflow[t]
        perc[t] = _linear_reservoir(s1, kp)
        s1 -= perc[t]
        # Lower soil layer discharge
        s2 += perc[t]
        baseflow[t] = _linear_reservoir(s2, k2)
        s2 -= baseflow[t]
        # Total Flow
        totalflow[t] = nsurflow[t] + interflow[t] + baseflow[t]
        # Save reservoirs
        snow[t] = sn
        ws[t] = sm / fc
        ws1[t] = s1
        ws2[t] = s2

    totalflow[:] = _convolution_giuh(totalflow, area, maxbas)
    # totalflow=0, runoff=1, nsurflow=2, interflow=3, baseflow=4
    # et=5, lqw=6, infil=7, perc=8, snow=9, ws=10, ws1=11, ws2=12
    return totalflow, runoff, nsurflow, interflow, baseflow, et, lqw, infil, perc, snow, ws, ws1, ws2

