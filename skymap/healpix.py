#!/usr/bin/env python
"""
Tools for working with healpix.
"""
__author__ = "Alex Drlica-Wagner"
import numpy as np
import healpy as hp
import pandas as pd

import warnings
warnings.simplefilter("always")

def masked_array(array,badval=hp.UNSEEN):
    if isinstance(array,np.ma.MaskedArray):
        return array
    mask = ~np.isfinite(array) | (array==badval)
    return np.ma.MaskedArray(array,mask=mask)

def check_hpxmap(hpxmap,pixel,nside):
    if pixel is None and not hp.isnpixok(hpxmap.shape[-1]):
        msg = "'hpxmap' has invalid dimension: %s"%(hpxmap.shape)
        raise Exception(msg)

    if pixel is not None and nside is None:
        msg = "'nside' must be specified for explicit maps"
        raise Exception(msg)

    if pixel is not None and (hpxmap.shape != pixel.shape):
        msg = "'hpxmap' and 'pixel' must have same shape"
        raise Exception(msg)
    
def create_map(hpxmap,pixel,nside,badval=hp.UNSEEN):
    """ Create the full map from hpxmap,pixel,nside combo
    """
    if pixel is None: return hpxmap
    m = badval*np.ones(hp.nside2npix(nside),dtype=hpxmap.dtype)
    m[pixel] = hpxmap
    return m
    
def get_map_range(hpxmap, pixel=None, nside=None, wrap_angle=180):
    """ Calculate the longitude and latitude range for a map. """
    check_hpxmap(hpxmap,pixel,nside)
    if isinstance(hpxmap,np.ma.MaskedArray):
        hpxmap = hpxmap.data

    if pixel is None:
        nside = hp.get_nside(hpxmap)
        pixel = np.arange(len(hpxmap),dtype=int)

    ipring,=np.where(np.isfinite(hpxmap) & (hpxmap!=hp.UNSEEN))
    theta,phi = hp.pix2ang(nside, pixel[ipring])
    lon = np.mod(np.degrees(phi),360)
    lat = 90.0-np.degrees(theta)

    # Small offset to add to make sure we get the whole pixel
    eps = np.degrees(hp.max_pixrad(nside))

    # CHECK ME
    hi,=np.where(lon > wrap_angle)
    lon[hi] -= 360.0

    lon_min = max(np.nanmin(lon)-eps,wrap_angle-360)
    lon_max = min(np.nanmax(lon)+eps,wrap_angle)
    lat_min = max(np.nanmin(lat)-eps,-90)
    lat_max = min(np.nanmax(lat)+eps,90)

    return (lon_min,lon_max), (lat_min,lat_max)

def hpx2xy(hpxmap, pixel=None, nside=None, xsize=800, aspect=1.0,
           lonra=None, latra=None):
    """ Convert a healpix map into x,y pixels and values"""
    check_hpxmap(hpxmap,pixel,nside)

    if lonra is None and latra is None:
        lonra,latra = get_map_range(hpxmap,pixel,nside)
    elif (lonra is None) or (latra is None):
        msg = "Both lonra and latra must be specified"
        raise Exception(msg)

    lon = np.linspace(lonra[0],lonra[1], xsize)
    lat = np.linspace(latra[0],latra[1], int(aspect*xsize))
    lon, lat = np.meshgrid(lon, lat)

    # Calculate the value at the average location for pcolormesh
    # ADW: How does this play with RA = 360 boundary?
    llon = (lon[1:,1:]+lon[:-1,:-1])/2.
    llat = (lat[1:,1:]+lat[:-1,:-1])/2.

    if nside is None:
        if isinstance(hpxmap,np.ma.MaskedArray):
            nside = hp.get_nside(hpxmap.data)
        else:
            nside = hp.get_nside(hpxmap)

    # Old version of healpy
    try:
        pix = hp.ang2pix(nside,llon,llat,lonlat=True)
    except TypeError:
        pix = hp.ang2pix(nside,np.radians(90-llat),np.radians(llon))

    if pixel is None:
        values = masked_array(hpxmap[pix])
    else:
        # Things get fancy here...
        # Match the arrays on the pixel index
        pixel_df = pd.DataFrame({'pix':pixel,'idx':np.arange(len(pixel))})
        # Pandas warns about type comparison.
        # It probably doesn't like `pix.flat`, but it would save space
        #pix_df = pd.DataFrame({'pix':pix.flat},dtype=int)
        pix_df = pd.DataFrame({'pix':pix.ravel()},dtype=int)
        idx = pix_df.merge(pixel_df,on='pix',how='left')['idx'].values
        mask = np.isnan(idx)

        # Index the values by the matched index
        values = np.nan*np.ones(pix.shape,dtype=hpxmap.dtype)
        values[np.where(~mask.reshape(pix.shape))] = hpxmap[idx[~mask].astype(int)]
        values = np.ma.array(values,mask=mask)

    return lon,lat,values

def pd_index_pix_in_pixels(pix,pixels):
    pixel_df = pd.DataFrame({'pix':pixel,'idx':np.arange(len(pixel))})
    # Pandas warns about type comparison (probably doesn't like `pix.flat`)...
    pix_df = pd.DataFrame({'pix':pix.flat},dtype=int)
    idx = pix_df.merge(pixel_df,on='pix',how='left')['idx'].values
    return idx

def np_index_pix_in_pixels(pix,pixels):
    """
    Find the indices of a set of pixels into another set of pixels
    """
    # Are the pixels always sorted? Is it quicker to check?
    pixels = np.sort(pixels)
    # Assumes that 'pixels' is pre-sorted, otherwise...???
    index = np.searchsorted(pixels,pix)
    if np.isscalar(index):
        if not np.in1d(pix,pixels).any(): index = np.nan
    else:
        # Find objects that are outside the pixels
        index[~np.in1d(pix,pixels)] = np.nan
    return index

def index_lonlat_in_pixels(lon,lat,pixels,nside):
    pix = ang2pix(nside,lon,lat)
    return index_pix_in_pixels(pix,pixels)

def ang2disc(nside, lon, lat, radius, inclusive=False, fact=4, nest=False):
    """
    Wrap `query_disc` to use lon, lat, and radius in degrees.
    """
    vec = hp.ang2vec(lon,lat,lonlat=True)
    return hp.query_disc(nside,vec,np.radians(radius),inclusive,fact,nest)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
