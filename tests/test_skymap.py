#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import unittest
import os

import matplotlib
#if not os.getenv('DISPLAY'): matplotlib.use('Agg')
matplotlib.use('Agg')
import pylab as plt

from skymap import Skymap,McBrydeSkymap,OrthoSkymap
from skymap import SurveySkymap,SurveyMcBryde,SurveyOrtho
from skymap import DESSkymap

class TestSkymap(unittest.TestCase):

    def test_skymap(self):
        for cls in [Skymap,McBrydeSkymap,OrthoSkymap]:
            plt.figure()
            m = cls()
            m.draw_milky_way()

    def test_survey_skymap(self):
        for cls in [SurveySkymap,SurveyMcBryde,SurveyOrtho]:
            plt.figure()
            m = cls()
            m.draw_des()
            m.draw_maglites()
            m.draw_bliss()

    def test_des_skymap(self):
        for cls in [DESSkymap]:
            plt.figure()
            m = cls()
            m.draw_des()
            m.draw_maglites()
            m.draw_bliss()

if __name__ == '__main__':
    plt.ion()

    unittest.main()

