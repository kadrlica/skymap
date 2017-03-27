#!/usr/bin/env python
"""
Random utilities
"""
import os, os.path

import numpy as np
import healpy as hp

def get_datadir():
    from os.path import abspath,dirname,join
    return join(dirname(abspath(__file__)),'data')

def setdefaults(kwargs,defaults):
    for k,v in defaults.items():
        kwargs.setdefault(k,v)
    return kwargs


# Astropy is still way to inefficient with coordinate transformations:
# https://github.com/astropy/astropy/issues/1717

def gal2cel(glon, glat):
    """
    Converts Galactic (deg) to Celestial J2000 (deg) coordinates
    """
    glat = np.radians(glat)
    sin_glat = np.sin(glat)
    cos_glat = np.cos(glat)

    glon = np.radians(glon)
    ra_gp = np.radians(192.85948)
    de_gp = np.radians(27.12825)
    lcp = np.radians(122.932)

    sin_lcp_glon = np.sin(lcp - glon)
    cos_lcp_glon = np.cos(lcp - glon)

    sin_d = (np.sin(de_gp) * sin_glat) \
            + (np.cos(de_gp) * cos_glat * cos_lcp_glon)
    ramragp = np.arctan2(cos_glat * sin_lcp_glon,
                         (np.cos(de_gp) * sin_glat) \
                         - (np.sin(de_gp) * cos_glat * cos_lcp_glon))
    dec = np.arcsin(sin_d)
    ra = (ramragp + ra_gp + (2. * np.pi)) % (2. * np.pi)
    return np.degrees(ra), np.degrees(dec)

def cel2gal(ra, dec):
    """
    Converts Celestial J2000 (deg) to Calactic (deg) coordinates
    """
    dec = np.radians(dec)
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)

    ra = np.radians(ra)
    ra_gp = np.radians(192.85948)
    de_gp = np.radians(27.12825)

    sin_ra_gp = np.sin(ra - ra_gp)
    cos_ra_gp = np.cos(ra - ra_gp)

    lcp = np.radians(122.932)
    sin_b = (np.sin(de_gp) * sin_dec) \
            + (np.cos(de_gp) * cos_dec * cos_ra_gp)
    lcpml = np.arctan2(cos_dec * sin_ra_gp,
                       (np.cos(de_gp) * sin_dec) \
                       - (np.sin(de_gp) * cos_dec * cos_ra_gp))
    glat = np.arcsin(sin_b)
    glon = (lcp - lcpml + (2. * np.pi)) % (2. * np.pi)
    return np.degrees(glon), np.degrees(glat)

def phi2lon(phi): return np.degrees(phi)
def lon2phi(lon): return np.radians(lon)

def theta2lat(theta): return 90. - np.degrees(theta)
def lat2theta(lat): return np.radians(90. - lat)

def hpx_gal2cel(galhpx):
    npix = len(galhpx)
    nside = hp.npix2nside(npix)
    pix = np.arange(npix)

    ra,dec = pix2ang(nside,pix)
    glon,glat = cel2gal(ra,dec)

    return galhpx[ang2pix(nside,glon,glat)]

def pix2ang(nside, pix):
    """
    Return (lon, lat) in degrees instead of (theta, phi) in radians
    """
    theta, phi =  hp.pix2ang(nside, pix)
    lon = phi2lon(phi)
    lat = theta2lat(theta)
    return lon, lat

def ang2pix(nside, lon, lat, coord='GAL'):
    """
    Input (lon, lat) in degrees instead of (theta, phi) in radians
    """
    theta = np.radians(90. - lat)
    phi = np.radians(lon)
    return hp.ang2pix(nside, theta, phi)
