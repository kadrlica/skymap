"""
Skymap
======

Provides utilities for plotting skymaps.
"""
__author__ = 'Alex Drlica-Wagner'
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from skymap.core import Skymap, McBrydeSkymap, OrthoSkymap
from skymap.survey import SurveySkymap,SurveyMcBryde,SurveyOrtho
from skymap.survey import DESSkymap, BlissSkymap

import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore",category=MatplotlibDeprecationWarning)
