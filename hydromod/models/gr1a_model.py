# -*- coding: utf-8 -*-
"""
modèle pluie-débit annual GR1A

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

Mouelhi, S., Michel , C., Perrin, C. & Andreassian, V. (2006) Linking stream flow to
rainfall at the annual time step: the Manabe bucket model revisited. J. Hydrol. 328,
283-296, doi:10.1016/j.jhydrol.2005.12.022.

Modified by:
Andres Duque Perez
Universidad Nacional de Colombia, Sede Medellin
andresfduque@gmail.com | aduquep@ingenevo.com.co
"""

import numpy as np
import pandas as pd
import numba as nb
from .base_model import BaseModel
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# Main class
# ==============================================================================

class GR1A(BaseModel):

    def __init__(self, area=100, lat=0, params=None):
        """
        modèle pluie-débit annual GR1A

        INPUTS:
            area    >     [float] catchment area in km2
            lat     >     [float] catchment latitude at centroid
            params  >     [dict] model parameters

        Model parameters
            x       >     [float] production capacity (mm)

        Reference:
        Mouelhi, S., C. Michel, C. Perrin, and V. Andréassian (2006), Stepwise development of a two-parameter
        monthly water balance model, J. Hydrol., 318, 200-214, doi:10.1016/j.jhydrol.2005.1006.1014.
        """
        super().__init__(area, lat)

        self.params = {
            "x": 0.5,  # flux parameter (-)
        }

        if params is not None:
            self.set_parameters(**params)

    def __repr__(self):
        text = "\n____________GR1A structure_____________\n"
        text += "Catchment properties:\n"
        text += "    Area (km2): {:.3f}\n".format(self.area)
        text += "    Latitude  : {:.4f}\n".format(self.lat)
        text += "Model Parameters:\n"
        text += "    x > Flux parameter (adim): {:.3f}\n".format(self.params["x"])
        return text

    def __str__(self):
        x = self.params["x"]
        text = f"GR1A(area={self.area:.2f},lat={self.lat:.3f},"
        text += f"x={x:.3f})"
        return text

    def run(self, forcings, start=None, end=None, save_state=False, **kwargs):
        """
        Run the GR1A model


        Parameters
        ----------
        forcings : DataFrame
            Input annual data with columns prec (precipitation, mm), tmean (mean temperature, deg)
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
                x       >     [float] production capacity (mm)

        Returns
        -------
        Simulations : DataFrame
            qt       > Annual streamflow at catchment output (mm)
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
        pet = forcings["pet"].values

        # Compute Water Balance
        simulations = _gr1a(
            prec,
            pet,
            self.params["x"]
        )

        # Save Outputs
        outputs = pd.DataFrame(
            {
                "qt": simulations,
            },
            index=forcings.index,
        )

        return outputs


# ==============================================================================
# Subroutines for model processes
# ==============================================================================

@nb.jit(nopython=True)
def _gr1a(prec, pet, x):
    """
    modèle pluie-débit annual GR1A
    """
    n = len(prec)
    qt = np.zeros(n, dtype=np.float32)
    for t in range(n):
        if t == 0:
            sub = prec[t] / (x * pet[t])
        else:
            sub = (0.7 * prec[t] + 0.3 * prec[t-1]) / (x * pet[t])
        qt[t] = prec[t] * (1.0 - 1.0 / (1.0 + sub ** 2.0) ** 0.5)
    return qt

