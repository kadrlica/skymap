#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os

import numpy as np

from skymap.utils import get_datadir
from skymap.utils import SphericalRotator

class DECamFocalPlane(object):
    """Class for storing and manipulating the corners of the DECam CCDs.
    """

    filename = os.path.join(get_datadir(),'ccd_corners_xy_fill.dat')

    def __init__(self):
        # This is not safe. Use yaml instead (extra dependency)
        self.ccd_dict = eval(''.join(open(self.filename).readlines()))

        # These are x,y coordinates
        self.corners = np.array(list(self.ccd_dict.values()))

        # Since we don't know the original projection of the DECam
        # focal plane into x,y it is probably not worth trying to
        # deproject it right now...

        #x,y = self.ccd_array[:,:,0],self.ccd_array[:,:,1]
        #ra,dec = Projector(0,0).image2sphere(x.flat,y.flat)
        #self.corners[:,:,0] = ra.reshape(x.shape)
        #self.corners[:,:,1] = dec.reshape(y.shape)

    def rotate(self, ra, dec):
        """Rotate the corners of the DECam CCDs to a given sky location.

        Parameters:
        -----------
        ra      : The right ascension (deg) of the focal plane center
        dec     : The declination (deg) of the focal plane center

        Returns:
        --------
        corners : The rotated corner locations of the CCDs
        """
        corners = np.copy(self.corners)

        R = SphericalRotator(ra,dec)
        _ra,_dec = R.rotate(corners[:,:,0].flat,corners[:,:,1].flat,invert=True)

        corners[:,:,0] = _ra.reshape(corners.shape[:2])
        corners[:,:,1] = _dec.reshape(corners.shape[:2])
        return corners

    def project(self, basemap, ra, dec):
        """Apply the given basemap projection to the DECam focal plane at a
        location given by ra,dec.

        Parameters:
        -----------
        basemap : The Basemap to project to.
        ra      : The right ascension (deg) of the focal plane center
        dec     : The declination (deg) of the focal plane center

        Returns:
        --------
        corners : Projected corner locations of the CCDs
        """
        corners = self.rotate(ra,dec)

        x,y = basemap.proj(corners[:,:,0],corners[:,:,1])

        # Remove CCDs that cross the map boundary
        x[(np.ptp(x,axis=1) > np.pi)] = np.nan

        corners[:,:,0] = x
        corners[:,:,1] = y
        return corners


