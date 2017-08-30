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

import skymap
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
            plt.title('Milky Way (%s)'%cls.__name__)

    def test_survey_skymap(self):
        for cls in SURVEYS:
            plt.figure()
            m = cls()
            m.draw_des()
            m.draw_maglites()
            m.draw_bliss()
            plt.title('Footprints (%s)'%cls.__name__)

    def test_zoom_skymap(self):
        for cls in ZOOMS:
            plt.figure()
            m = cls()
            m.draw_des()
            m.draw_maglites()
            m.draw_bliss()
            plt.title('Zoom Footprints (%s)'%cls.__name__)

    def test_draw_hpxmap(self):
        """ Test drawing a full healpix skymap """
        hpxmap = np.arange(hp.nside2npix(nside))
        for cls in SKYMAPS:
            plt.figure()
            m = cls()
            m.draw_hpxmap(hpxmap,xsize=400)
            plt.title('HEALPix Map (%s)'%cls.__name__)

    def test_draw_explicit_hpxmap(self):
        """ Test an explicit healpix map """
        pix = hpxmap = np.arange(525,535)
        for cls in SKYMAPS:
            plt.figure()
            m = cls()
            m.draw_hpxmap(hpxmap,pix,nside,xsize=400)
            plt.title('Partial HEALPix Map (%s)'%cls.__name__)

    def test_draw_hpxbin(self):
        """ Test drawing hpxbin from points """
        kwargs = dict(nside=64)
        for cls in SKYMAPS:
            plt.figure()
            m = cls()
            # This is not uniform
            size = int(1e6)
            lon = np.random.uniform(0,360,size=size)
            lat = np.random.uniform(-90,90,size=size)
            m.draw_hpxbin(lon,lat,**kwargs)
            plt.title('HEALPix Bin (%s)'%cls.__name__)

    def test_draw_hires(self):
        """ Draw a partial healpix map with very large nside """
        nside = 4096*2**5
        ra,dec = 45,-45
        radius = 0.05
        pixels = skymap.healpix.ang2disc(nside,ra,dec,radius)
        values = pixels

        # Use the Cassini projection (because we can)
        m = Skymap(projection='cass', lon_0=ra, lat_0=dec, celestial=False,
                   llcrnrlon=ra-2*radius,urcrnrlon=ra+2*radius,
                   llcrnrlat=dec-2*radius,urcrnrlat=dec+2*radius,
        )

        m.draw_hpxmap(values,pixels,nside=nside,xsize=400)
        m.draw_parallels(np.linspace(dec-2*radius,dec+2*radius,5),
                         labelstyle='+/-',labels=[1,0,0,0])
        m.draw_meridians(np.linspace(ra-2*radius,ra+2*radius,5),
                         labelstyle='+/-',labels=[0,0,0,1])
        plt.title('HEALPix Zoom (nside=%i)'%nside)


    def test_draw_focal_planes(self):
        """ Draw a DECam focal planes """
        ra,dec = 45,-45
        radius = 1.5
        delta = 1.0

        # Use the Cassini projection (because we can)
        m = Skymap(projection='cass', lon_0=ra, lat_0=dec, celestial=False,
                   llcrnrlon=ra+2*radius,urcrnrlon=ra-2*radius,
                   llcrnrlat=dec-2*radius,urcrnrlat=dec+2*radius,
        )

        # Can plot individual fields
        m.draw_focal_planes([ra+delta/2],[dec-delta/2],color='g')
        # Or as arrays
        m.draw_focal_planes([ra,ra-delta,ra-delta],[dec,dec+delta,dec-delta],color='r')
        # Draw the grid lines
        m.draw_parallels(np.linspace(dec-2*radius,dec+2*radius,5),
                         labelstyle='+/-',labels=[1,0,0,0])
        m.draw_meridians(np.linspace(ra-2*radius,ra+2*radius,5),
                         labelstyle='+/-',labels=[0,0,0,1])
        plt.title('DECam Focal Planes')


if __name__ == '__main__':
    plt.ion()
    unittest.main()

