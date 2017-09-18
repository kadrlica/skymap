#!/usr/bin/env python
"""
Various constants used for plotting
"""
__author__ = "Alex Drlica-Wagner"

import numpy as np

# Plotting DECam
DECAM=1.1 # DECam radius (deg)

# Marker size depends on figsize and DPI
FIGSIZE=(10.5,8.5)
SCALE=np.sqrt((8.0*6.0)/(FIGSIZE[0]*FIGSIZE[1]))
DPI=80;

# LMC and SMC
RA_LMC = 80.8939
DEC_LMC = -69.7561
RADIUS_LMC = 5.3667 # semi-major axis (deg)
RA_SMC = 13.1867
DEC_SMC = -72.8286
RADIUS_SMC = 2.667 # semi_major axis (deg)

