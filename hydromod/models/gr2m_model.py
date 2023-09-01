# -*- coding: utf-8 -*-
"""
modèle pluie-débit mensuel GR2M

Rain-Runoff Model


Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx

Reference:
Mouelhi, S., 2003. Vers une chaîne cohérente de modèles pluie-débit conceptuels
globaux aux pas de temps pluriannuel, annuel, mensuel et journalier. Thèse de Doctorat,
ENGREF, Cemagref Antony, France, 323 pp.

Mouelhi, S., C. Michel, C. Perrin, and V. Andréassian (2006), Stepwise development of a two-parameter
monthly water balance model, J. Hydrol., 318, 200-214, https://doi.org/10.1016/j.jhydrol.2005.06.014
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

class GR2M(BaseModel):

    def __init__(self, area=100, lat=0, params=None):
        """
        modèle pluie-débit mensuel GR2M


        INPUTS:
            area    >     [float] catchment area in km2
            lat     >     [float] catchment latitude at centroid
            params  >     [dict] model parameters

        Model parameters
            x1      >     [float] production capacity (mm)
            x2      >     [float] discharge parameter (mm)
            s0      >     [float] First storage level as a fraction of x1 (-)
            r0      >     [float] River storage level as a fraction of 60 mm (-)

        Reference:
        Mouelhi, S., C. Michel, C. Perrin, and V. Andréassian (2006), Stepwise development of a two-parameter
        monthly water balance model, J. Hydrol., 318, 200-214, https://doi.org/10.1016/j.jhydrol.2005.06.014
        """
        super().__init__(area, lat)

        self.params = {
            "x1": 500,  # Production capacity (mm)
            "x2": 0.8,  # Discharge parameter (mm)
            "s0": 0.5,  # Initial storage (adim)
            "r0": 0.5,  # Initial filling (adim)
        }

        if params is not None:
            self.set_parameters(**params)

    def __repr__(self):
        text = "\n____________GR2M structure_____________\n"
        text += "Catchment properties:\n"
        text += "    Area (km2): {:.3f}\n".format(self.area)
        text += "    Latitude  : {:.4f}\n".format(self.lat)
        text += "Model Parameters:\n"
        text += "    x1 > Production capacity (mm)    : {:.3f}\n".format(self.params["x1"])
        text += "    x2 > Discharge parameter (mm)    : {:.3f}\n".format(self.params["x2"])
        text += "    s0 > Initial storage level (adim): {:.3f}\n".format(self.params["s0"])
        text += "    r0 > Initial river level (adim)  : {:.3f}\n".format(self.params["r0"])
        return text

    def __str__(self):
        x1 = self.params["x1"]
        x2 = self.params["x2"]
        s0 = self.params["s0"]
        r0 = self.params["r0"]
        text = f"GR2M(area={self.area:.2f},lat={self.lat:.3f},"
        text += f"x1={x1:.3f},x2={x2:.3f},s0={s0:.3f},r0={r0:.3f})"
        return text

    def run(self, forcings, start=None, end=None, save_state=False, **kwargs):
        """
        Run the GR2M model


        Parameters
        ----------
        forcings : DataFrame
            Input data with columns prec (precipitation, mm),tmean (mean temperature, deg)
            pet (potential evapotranspiration, mm, optional)
        start : string, optional
            Start date for simulation in format. Example: '2001-01-01'
        end : string, optional
            End date for simulation in format. Example: '2010-12-31'
        save_state : bool, optional
            If True (default), last storage is saved as w0 parameter
        **kwargs :
            Model parameters can be changed for the simulation
                area    >     [float] catchment area in km2
                lat     >     [float] catchment latitude at centroid
                x1      >     [float] production capacity (mm)
                x2      >     [float] discharge parameter (mm)
                s0      >     [float] First storage level as a fraction of x1 (-)
                r0      >     [float] River storage level as a fraction of 60 mm (-)

        Returns
        -------
        Simulations : DataFrame
            qt       > Monthly streamflow at catchment output (mm)
            pet      > Monthly potential evapotranspiration (mm)
            fs       > First storage level as a fraction of x1 (-)
            rs       > River storage level as a fraction of 60 mm (-)
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
            tmean = forcings["tmean"]
            pet = pet_models.pet_thornthwaite(
                tmean,
                self.lat
            ).values

        # Compute Water Balance
        simulations = _gr2m(
            prec,
            pet,
            self.params["s0"],
            self.params["r0"],
            self.params["x1"],
            self.params["x2"]
        )

        # Save final storage state
        if save_state:
            self.params["s0"] = simulations[1][-1]
            self.params["r0"] = simulations[2][-1]

        # Save Outputs
        outputs = pd.DataFrame(
            {
                "qt": simulations[0],
                "pet": pet,
                "fs": simulations[1],
                "rs": simulations[2],
            },
            index=forcings.index
        )

        return outputs


# ==============================================================================
# Subroutines for model processes
# ==============================================================================

@nb.jit(nopython=True)
def _gr2m(prec, pet, s0, r0, x1, x2):
    """
    modèle pluie-débit mensuel GR2M
    """
    # Initial parameters
    s0 = s0 * x1
    r0 = r0 * 60

    # Output series
    n = len(prec)
    s = np.zeros(n, dtype=np.float32)
    r = np.zeros(n, dtype=np.float32)
    qt = np.zeros(n, dtype=np.float32)

    # Main loop
    for i in range(n):
        phi = np.tanh(prec[i] / x1)
        psi = np.tanh(pet[i] / x1)
        s1 = (s0 + x1 * phi) / (1.0 + phi * s0 / x1)
        p1 = prec[i] + s0 - s1
        s2 = (s1 * (1.0 - psi)) / (1.0 + psi * (1.0 - s1 / x1))
        s0 = s2 / (1.0 + (s2 / x1) ** 3.0) ** (1.0 / 3.0)
        s[i] = s0 / x1  # save state
        p2 = s2 - s0
        p3 = p1 + p2
        r1 = r0 + p3
        r2 = x2 * r1
        qt[i] = r2 ** 2.0 / (r2 + 60.0)
        r0 = r2 - qt[i]
        r[i] = r0 / 60.0  # save state

    return qt, s, r

