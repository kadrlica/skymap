#!/usr/bin/env python
"""
Unit tests for skymap
"""
__author__ = "Alex Drlica-Wagner"

import os,sys
import unittest

import matplotlib

if not sys.flags.interactive:
    matplotlib.use('Agg')
else:
    if not os.getenv('DISPLAY'): matplotlib.use('Agg')

import pylab as plt
import numpy as np
import healpy as hp

from skymap import Skymap,McBrydeSkymap,OrthoSkymap
from skymap import SurveySkymap,SurveyMcBryde,SurveyOrtho
from skymap import DESSkymap,BlissSkymap

nside = 8

SKYMAPS = [Skymap,McBrydeSkymap,OrthoSkymap]
SURVEYS = [SurveySkymap,SurveyMcBryde,SurveyOrtho]
ZOOMS = [DESSkymap,BlissSkymap]

class TestSkymap(unittest.TestCase):

    def test_skymap(self):
        for cls in SKYMAPS:
            plt.figure()
            m = cls()
            m.draw_milky_way()

    def test_survey_skymap(self):
        for cls in SURVEYS:
            plt.figure()
            m = cls()
            m.draw_des()
            m.draw_maglites()
            m.draw_bliss()

    def test_zoom_skymap(self):
        for cls in ZOOMS:
            plt.figure()
            m = cls()
            m.draw_des()
            m.draw_maglites()
            m.draw_bliss()

    def test_draw_hpxmap(self):
        """ Test drawing a full healpix skymap """
        hpxmap = np.arange(hp.nside2npix(nside))
        for cls in SKYMAPS:
            plt.figure()
            m = cls()
            m.draw_hpxmap(hpxmap,xsize=400)

    def test_draw_explicit_hpxmap(self):
        """ Test an explicit healpix map """
        pix = hpxmap = np.arange(525,535)
        for cls in SKYMAPS:
            plt.figure()
            m = cls()
            m.draw_hpxmap(hpxmap,pix,nside,xsize=400)

if __name__ == '__main__':
    plt.ion()
    unittest.main()

