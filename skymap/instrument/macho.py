#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os

import numpy as np

from skymap.utils import get_datadir
from skymap.utils import SphericalRotator
from skymap.instrument import FocalPlane

class MachoFocalPlane(FocalPlane):
    """Class for storing and manipulating the corners of the DECam CCDs.
    """

    filename = os.path.join(get_datadir(),'macho_corners_xy.dat')

    def __init__(self):
        # This is not safe. Use yaml instead (extra dependency)
        self.ccd_dict = eval(''.join(open(self.filename).readlines()))

        # These are x,y coordinates
        self.corners = np.array(list(self.ccd_dict.values()))
