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

def angsep(lon1,lat1,lon2,lat2):
    """
    Angular separation (deg) between two sky coordinates.
    Borrowed from astropy (www.astropy.org)

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1],
    which is slighly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    [1] http://en.wikipedia.org/wiki/Great-circle_distance
    """
    lon1,lat1 = np.radians([lon1,lat1])
    lon2,lat2 = np.radians([lon2,lat2])

    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.degrees(np.arctan2(np.hypot(num1,num2), denominator))

class SphericalRotator:
    """
    Base class for rotating points on a sphere.

    The input is a fiducial point (deg) which becomes (0, 0) in rotated coordinates.
    """

    def __init__(self, lon_ref, lat_ref, zenithal=False):
        self.setReference(lon_ref, lat_ref, zenithal)

    def setReference(self, lon_ref, lat_ref, zenithal=False):

        if zenithal:
            phi = (np.pi / 2.) + np.radians(lon_ref)
            theta = (np.pi / 2.) - np.radians(lat_ref)
            psi = 0.
        if not zenithal:
            phi = (-np.pi / 2.) + np.radians(lon_ref)
            theta = np.radians(lat_ref)
            psi = np.radians(90.) # psi = 90 corresponds to (0, 0) psi = -90 corresponds to (180, 0)


        cos_psi,sin_psi = np.cos(psi),np.sin(psi)
        cos_phi,sin_phi = np.cos(phi),np.sin(phi)
        cos_theta,sin_theta = np.cos(theta),np.sin(theta)

        self.rotation_matrix = np.matrix([
            [cos_psi * cos_phi - cos_theta * sin_phi * sin_psi,
             cos_psi * sin_phi + cos_theta * cos_phi * sin_psi,
             sin_psi * sin_theta],
            [-sin_psi * cos_phi - cos_theta * sin_phi * cos_psi,
             -sin_psi * sin_phi + cos_theta * cos_phi * cos_psi,
             cos_psi * sin_theta],
            [sin_theta * sin_phi,
             -sin_theta * cos_phi,
             cos_theta]
        ])

        self.inverted_rotation_matrix = np.linalg.inv(self.rotation_matrix)

    def cartesian(self,lon,lat):
        lon = np.radians(lon)
        lat = np.radians(lat)

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z =  np.sin(lat)
        return np.array([x,y,z])


    def rotate(self, lon, lat, invert=False):
        vec = self.cartesian(lon,lat)

        if invert:
            vec_prime = np.dot(np.array(self.inverted_rotation_matrix), vec)
        else:
            vec_prime = np.dot(np.array(self.rotation_matrix), vec)

        lon_prime = np.arctan2(vec_prime[1], vec_prime[0])
        lat_prime = np.arcsin(vec_prime[2])

        return (np.degrees(lon_prime) % 360.), np.degrees(lat_prime)
