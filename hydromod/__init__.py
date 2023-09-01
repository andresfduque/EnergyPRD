# -*- coding: utf-8 -*-
"""
Lumped Models (LuMod) for hydrology

Author:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
"""

__version__ = '0.1.3.0'
__author__ = 'Saul Arciniega Esparza'
__citation__ = 'Coming soon'

from lumod.tools.monte_carlo import MonteCarlo
from lumod.tools.time_series import load_example
from .fluxes import pet_models
from . import models
from . import tools

