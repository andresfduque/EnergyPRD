# -*- coding: utf-8 -*-
"""
Builder Class for hydrological models


Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
"""

# %% Import libraries
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Main class
# ==============================================================================

class BaseModel(object):

    def __init__(self, area=100, lat=0):
        self.area = area
        self.lat = lat
        self.params = {}

    def get_parameters(self, asdict=True):
        """
        Return the model parameters as dict or Pandas Series
        """
        if asdict:
            return self.params.copy()
        else:
            return pd.Series(self.params)

    def set_parameters(self, params=None, **kwargs):
        """
        Set model parameters as dict {"x": 1}, Series (pd.Series(1, index=["x"])) or
        parameter by parameter (x=1)
        """
        if isinstance(params, dict):
            params = kwargs.update(params)
        elif isinstance(params, pd.Series):
            params = kwargs.update(params.to_dict())

        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

