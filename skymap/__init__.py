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

def get_datadir():
    from os.path import abspath,dirname,join
    return join(dirname(abspath(__file__)),'data')

import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore",category=MatplotlibDeprecationWarning)
