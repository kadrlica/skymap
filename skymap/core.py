#!/usr/bin/env python
"""
Core skymap classes
"""
import os
from os.path import expandvars
import logging
from collections import OrderedDict as odict

import matplotlib
from matplotlib import mlab
import pylab as plt
import numpy as np
import ephem
import healpy as hp
import scipy.ndimage as nd

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import pyproj
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from skymap.utils import setdefaults,get_datadir
from skymap.utils import cel2gal, gal2cel
from skymap import healpix

# TODO:
# - abstract interface between new and old healpy pixfuncs
# - abstract wrapper around pixfuncs for masked and unmasked arrays
# - abstract function for coordinate wrapping

class Skymap(Basemap):
    """ Skymap base class. """

    COLORS = odict([
            ('none','black'),
            ('u','blue'),
            ('g','green'),
            ('r','red'),
            ('i','#EAC117'),
            ('z','darkorchid'),
            ('Y','black'),
            ('VR','gray'),
            ])

    defaults = dict(celestial=True, rsphere=1.0, lon_0=0, lat_0=0,
                    parallels=True,meridians=True)

    def __init__(self, *args, **kwargs):
        self.set_observer(kwargs.pop('observer',None))
        self.set_date(kwargs.pop('date',None))

        setdefaults(kwargs,self.defaults)
        parallels = kwargs.pop('parallels',True)
        meridians = kwargs.pop('meridians',True)
        super(Skymap,self).__init__(*args,**kwargs)

        if parallels:
            self.draw_parallels()
        if meridians:
            self.draw_meridians()

        # Coordinate formatter
        # This is creating an axes (which we really don't want)
        # Better to stick in set_axes_limits
        ax = self._check_ax()
        def format_coord(x, y):
            return 'lon=%1.4f, lat=%1.4f'%self(x,y,inverse=True)
        plt.gca().format_coord = format_coord

        self.wrap_angle = np.mod(kwargs['lon_0'] + 180,360)

        self.resolution = 'c'

    def set_observer(self, observer):
        observer = observer.copy() if observer else ephem.Observer()
        self.observer = observer

    def set_date(self,date):
        date = ephem.Date(date) if date else ephem.now()
        self.observer.date = date

    def draw_parallels(self,*args,**kwargs):
        defaults = dict(labels=[1,0,0,1],labelstyle='+/-',dashes=(2,3),
                        color='gray',linewidth=0.75)
        if not args:
            defaults.update(circles=np.arange(-60,90,30))
        if self.projection in ['ortho','geos','nsper','aeqd','vandg']:
            defaults.update(labels=[0,0,0,0])
        setdefaults(kwargs,defaults)
        return self.drawparallels(*args, **kwargs)

    def draw_meridians(self,*args,**kwargs):
        defaults = dict(labels=[1,0,0,1],labelstyle='+/-',dashes=(2,3),
                        color='gray',linewidth=0.75)
        if self.projection in ['ortho','geos','nsper','aeqd','vandg',
                               'sinu','hammer']:
            defaults.update(labels=[0,0,0,0])
        if not args:
            #defaults.update(meridians=np.arange(0,420,60))
            defaults.update(meridians=np.arange(0,360,60))
        setdefaults(kwargs,defaults)
        return self.drawmeridians(*args,**kwargs)

    def cartesian(self, ra, dec):
        x = np.cos(ra) * np.cos(dec)
        y = np.sin(ra) * np.cos(dec)
        z = np.sin(dec)
        return x, y, z

    def spherical(self, x, y, z):
        ra = np.arctan2(y, x)
        dec = np.pi / 2 - np.arccos(z)
        return ra, dec

    def proj(self,lon,lat):
        """ Remove points outside of projection """
        # Should this overload __call__?
        x, y = self(np.atleast_1d(lon),np.atleast_1d(lat))
        x[np.abs(x) > 1e29] = None
        y[np.abs(y) > 1e29] = None
        return x, y

    def get_zenith(self):
        # RA and Dec of zenith
        lon_zen, lat_zen = np.degrees(self.observer.radec_of(0,'90'))
        return -lon_zen

    @staticmethod
    def wrap_index(lon, lat, wrap=180.):
        """ Find the index where the array wraps.
        """
        # No wrap: ignore
        if wrap is None:  return None

        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)

        # No array: ignore
        if len(lon)==1 or len(lat)==1: return None

        # Map [0,360)
        lon = np.mod(lon,360)
        wrap = np.mod(wrap,360)

        # Find the index of the entry closest to the wrap angle
        idx = np.abs(lon - wrap).argmin()
        # First or last index: ignore
        if idx == 0 or idx+1 == len(lon): return None
        # Value exactly equals wrap, choose next value
        elif (lon[idx] == wrap): idx += 1
        # Wrap angle sandwiched
        elif (lon[idx]<wrap) and (lon[idx+1]>wrap): idx += 1
        elif (lon[idx]<wrap) and (lon[idx-1]>wrap): idx += 0
        elif (lon[idx]>wrap) and (lon[idx+1]<wrap): idx += 1
        elif (lon[idx]>wrap) and (lon[idx-1]<wrap): idx += 0
        # There is no wrap: ignore
        else: return None

        return idx

    @classmethod
    def roll(cls,lon,lat,wrap=180.):
        """ Roll an lon,lat combination to split 180 boundary
        Parameters:
        -----------
        lon : right ascension (deg)
        lat: declination (deg)
        wrap_angle : angle to wrap at (deg)
        """
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)

        # Do nothing
        if wrap is None: return lon,lat
        if len(lon)==1 or len(lat)==1: return lon,lat

        idx = cls.wrap_index(lon,lat,wrap)
        if idx is None: return lon, lat

        return np.roll(lon,-idx), np.roll(lat,-idx)

    @classmethod
    def split(cls,lon,lat,wrap=180.):
        """ Split an lon,lat combination into lists across a wrap boundary
        Parameters:
        -----------
        lon : right ascension (deg)
        lat: declination (deg)
        wrap_angle : angle to wrap at (deg)
        """
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)

        # Do nothing
        if wrap is None: return [lon],[lat]
        if len(lon)==1 or len(lat)==1: return [lon],[lat]

        idx = cls.wrap_index(lon,lat,wrap)
        if idx is None: return [lon], [lat]

        return np.split(lon,[idx]), np.split(lat,[idx])

    def great_circle(self, lon1, lat1, lon2, lat2, arc='full'):
        """
        Create a great circle between two points.

        Parameters:
        -----------
        lon1, lat1 : The longitude/latitude of first point
        lon2, lat2 : The longitude/latitude of second point
        arc        : ['full','long','short'] which arc segment to draw

        Returns:
        --------
        matplotlib.plot
        """

        x1, y1, z1 = self.cartesian(np.radians(lon1), np.radians(lat1))
        x2, y2, z2 = self.cartesian(np.radians(lon2), np.radians(lat2))

        u = np.array([x1, y1, z1])
        v = np.array([x2, y2, z2])
        w = np.cross(np.cross(u, v), u)
        w /= np.linalg.norm(w)

        tt = np.linspace(0, 2 * np.pi, 360)
        r = u[:,np.newaxis] * np.cos(tt) + w[:,np.newaxis] * np.sin(tt)
        lon,lat = np.degrees(self.spherical(r[0], r[1], r[2]))

        # ADW: Is there any way this could not split into two elements?
        lons,lats = self.split(lon,lat,lon2)

        lons[0][-1] = lon2
        lats[0][-1] = lat2
        lons[1][0]  = lon2
        lats[1][0]  = lat2

        if len(lons[0]) < len(lons[1]):
            (slon,llon),(slat,llat) = lons,lats
        else:
            (llon,slon), (llat,slat) = lons,lats

        if arc in ['full','both']:
            return lon,lat
        elif arc == 'short':
            return slon,slat
        elif arc == 'long':
            return llon,llat
        else:
            msg = 'Unrecognized arc type: %s'%arc
            raise ValueError(msg)


    def draw_great_circle(self, lon1, lat1, lon2, lat2, arc='full', **kwargs):
        """
        Draw a great circle between two points.

        Parameters:
        -----------
        lon1, lat1 : The longitude/latitude of first point
        lon2, lat2 : The longitude/latitude of first point
        arc        : ['both','long','short'] which arc segment to draw
        kwargs     : keyword arguments to matplotlib.plot

        Returns:
        --------
        matplotlib.plot
        """
        # ADW: Using zip here isn't great
        lon,lat = self.great_circle(lon1,lat1,lon2,lat2,arc)
        
        for _lon,_lat in zip(*self.split(lon,lat,self.wrap_angle)):
            self.plot(*self(_lon,_lat),**kwargs)

        return lon,lat
        
    def draw_great_circle2(self, lon1, lat1, lon2, lat2, arc='both', **kwargs):
        """
        Draw a great circle between two points.

        Parameters:
        -----------
        lon1, lat1 : The longitude/latitude of first point
        lon2, lat2 : The longitude/latitude of first point
        arc        : ['both','long','short'] which arc segment to draw
        kwargs     : keyword arguments to matplotlib.plot

        Returns:
        --------
        matplotlib.plot
        """

        #defaults = dict(color='#1F618D')
        #setdefaults(kwargs,defaults)

        x1, y1, z1 = self.cartesian(np.radians(lon1), np.radians(lat1))
        x2, y2, z2 = self.cartesian(np.radians(lon2), np.radians(lat2))

        u = np.array([x1, y1, z1])
        v = np.array([x2, y2, z2])
        w = np.cross(np.cross(u, v), u)
        w /= np.linalg.norm(w)

        tt = np.linspace(0, 2 * np.pi, 360)
        r = u[:,np.newaxis] * np.cos(tt) + w[:,np.newaxis] * np.sin(tt)
        lon,lat = np.degrees(self.spherical(r[0], r[1], r[2]))

        # ADW: Is there any way this could not split into two elements?
        lons,lats = self.split(lon,lat,lon2)

        lons[0][-1] = lon2
        lats[0][-1] = lat2
        lons[1][0]  = lon2
        lats[1][0]  = lat2

        if len(lons[0]) < len(lons[1]):
            (slon,llon),(slat,llat) = lons,lats
        else:
            (llon,slon), (llat,slat) = lons,lats

        # ADW: Using zip here isn't great
        if arc in ['short','both']:
            #print 'short'
            for _lon,_lat in zip(*self.split(slon,slat,self.wrap_angle)):
                self.plot(*self(_lon,_lat),**kwargs)

        if arc in ['long','both']:
            #print 'long'
            for _lon,_lat in zip(*self.split(llon,llat,self.wrap_angle)):
                self.plot(*self(_lon,_lat),**kwargs)

        return lon,lat


    def draw_polygon(self,filename,**kwargs):
        """ Draw a polygon footprint. """
        defaults=dict(color='k', lw=2)
        setdefaults(kwargs,defaults)

        poly = np.loadtxt(filename,dtype=[('ra',float),('dec',float)])
        return self.draw_polygon_radec(poly['ra'],poly['dec'],**kwargs)

    def draw_polygon_radec(self,ra,dec,**kwargs):
        xy = self.proj(*self.roll(ra,dec,self.wrap_angle))
        self.plot(*xy,**kwargs)
        return xy

    def draw_polygons(self,filename,**kwargs):
        """Draw a text file containing multiple polygons"""
        data = np.genfromtxt(filename,names=['ra','dec','poly'])

        ret = []
        for p in np.unique(data['poly']):
            poly = data[data['poly'] == p]
            xy = self.draw_polygon_radec(poly['ra'],poly['dec'],**kwargs)
            ret += [xy]
            kwargs.pop('label',None)

        return ret

    def draw_paths(self,filename,**kwargs):
        """Draw a text file containing multiple polygons"""
        try:
            data = np.genfromtxt(filename,names=['ra','dec','poly'])
        except ValueError:
            data = np.genfromtxt(filename,names=['ra','dec'])
            data = mlab.rec_append_fields(data,'poly',np.zeros(len(data)))

        ret = []
        for p in np.unique(data['poly']):
            poly = data[data['poly'] == p]
            path,patch = self.draw_path_radec(poly['ra'],poly['dec'],**kwargs)
            ret += [(path,patch)]
        return ret

    def draw_path_radec(self,ra,dec,**kwargs):
        xy = self.proj(*self.roll(ra,dec,self.wrap_angle))
        vertices = np.vstack(xy).T
        path = matplotlib.path.Path(vertices)
        patch = matplotlib.patches.PathPatch(path,**kwargs)
        plt.gca().add_artist(patch)
        return path,patch

    def draw_zenith(self, radius=1.0, **kwargs):
        """
        Plot a to-scale representation of the zenith.
        """
        defaults = dict(color='green',alpha=0.75,lw=1.5)
        setdefaults(kwargs,defaults)

        # RA and Dec of zenith
        zra, zdec = np.degrees(self.observer.radec_of(0, '90'))
        xy = self.proj(zra, zdec)

        self.plot(*xy,marker='+',ms=10,mew=1.5, **kwargs)
        if radius:
            self.tissot(zra,zdec,radius,npts=100,fc='none', **kwargs)

    def draw_airmass(self, airmass=1.4, npts=360, **kwargs):
        defaults = dict(color='green', lw=2)
        setdefaults(kwargs,defaults)

        altitude_radians = (0.5 * np.pi) - np.arccos(1. / airmass)
        ra_contour = np.zeros(npts)
        dec_contour = np.zeros(npts)
        for ii, azimuth in enumerate(np.linspace(0., 2. * np.pi, npts)):
            ra_radians, dec_radians = self.observer.radec_of(azimuth, '%.2f'%(np.degrees(altitude_radians)))
            ra_contour[ii] = np.degrees(ra_radians)
            dec_contour[ii] = np.degrees(dec_radians)
        xy = self.proj(ra_contour, dec_contour)
        self.plot(*xy, **kwargs)

        self.draw_zenith(**kwargs)

    def draw_moon(self, date):
        moon = ephem.Moon()
        moon.compute(date)
        ra_moon = np.degrees(moon.ra)
        dec_moon = np.degrees(moon.dec)

        x,y = self.proj(np.array([ra_moon]), np.array([dec_moon]))
        if np.isnan(x).all() or np.isnan(y).all(): return

        self.scatter(x,y,color='%.2f'%(0.01*moon.phase),edgecolor='black',s=600)
        color = 'black' if moon.phase > 50. else 'white'
        #text = '%.2f'%(0.01 * moon.phase)
        text = '%2.0f%%'%(moon.phase)
        plt.text(x, y, text, fontsize=10, ha='center', va='center', color=color)

    def draw_milky_way(self,width=10,**kwargs):
        """ Draw the Milky Way galaxy. """
        defaults = dict(color='k',lw=1.5,ls='-')
        setdefaults(kwargs,defaults)

        glon = np.linspace(0,360,500)
        glat = np.zeros_like(glon)
        ra,dec = self.roll(*gal2cel(glon,glat),wrap=self.wrap_angle)
        ra -= 360*(ra > 180)

        self.draw_polygon_radec(ra,dec,**kwargs)

        if width:
            kwargs.update(dict(ls='--',lw=1))
            for delta in [+width,-width]:
                ra,dec = self.roll(*gal2cel(glon,glat+delta))
                ra -= 360*(ra > 180)
                self.draw_polygon_radec(ra,dec,**kwargs)

    def draw_magellanic_stream(self,**kwargs):
        import fitsio
        defaults = dict(xsize=800, vmin=17., vmax=25.0, rasterized=True,
                        cmap=plt.cm.binary)
        setdefaults(kwargs,defaults)

        dirname  = get_datadir()
        filename = 'allms_coldens_gal_nside_1024.fits'
        galhpx = fitsio.read(os.path.join(dirname,filename))['coldens']
        celhpx = obztak.utils.projector.hpx_gal2cel(galhpx)
        return self.draw_hpxmap(celhpx,**kwargs)

    def draw_sfd(self,**kwargs):
        defaults = dict(rasterized=True,cmap=plt.cm.binary)
        setdefaults(kwargs,defaults)
        dirname  = get_datadir()
        filename = 'lambda_sfd_ebv.fits'

        galhpx = hp.read_map(os.path.join(dirname,filename))
        celhpx = obztak.utils.projector.hpx_gal2cel(galhpx)
        return self.draw_hpxmap(np.log10(celhpx),**kwargs)

    def draw_lmc(self,**kwargs):
        from skymap.constants import RA_LMC, DEC_LMC, RADIUS_LMC
        defaults = dict(npts=100,fc='0.7',ec='0.5')
        setdefaults(kwargs,defaults)
        proj = self.proj(RA_LMC,DEC_LMC)
        self.tissot(RA_LMC,DEC_LMC,RADIUS_LMC,**kwargs)
        plt.text(proj[0],proj[1], 'LMC', weight='bold',
                 fontsize=10, ha='center', va='center', color='k')

    def draw_smc(self,**kwargs):
        from skymap.constants import RA_SMC, DEC_SMC, RADIUS_SMC
        defaults = dict(npts=100,fc='0.7',ec='0.5')
        setdefaults(kwargs,defaults)
        proj = self.proj(RA_SMC,DEC_SMC)
        self.tissot(RA_SMC,DEC_SMC,RADIUS_SMC,**kwargs)
        plt.text(proj[0],proj[1], 'SMC', weight='bold',
                 fontsize=8, ha='center', va='center', color='k')

    def draw_fields(self,fields,**kwargs):
        # Scatter point size is figsize dependent...
        defaults = dict(edgecolor='none',s=15)
        # case insensitive without changing input array
        names = dict([(n.lower(),n) for n in fields.dtype.names])

        if self.projection == 'ortho': defaults.update(s=50)
        if 'filter' in names:
            colors = [self.COLORS[b] for b in fields[names['filter']]]
            defaults.update(c=colors)
        elif 'band' in names:
            colors = [self.COLORS[b] for b in fields[names['band']]]
            defaults.update(c=colors)

        setdefaults(kwargs,defaults)
        ra,dec = fields[names['ra']],fields[names['dec']]
        self.scatter(*self.proj(ra,dec),**kwargs)

    def draw_hpxbin(self, lon, lat, nside=256, **kwargs):
        """
        Create a healpix histogram of the counts.

        Like `hexbin` from matplotlib

        Parameters:
        -----------
        lon : input longitude (deg)
        lat : input latitude (deg)
        nside : heaplix nside resolution
        kwargs : passed to draw_hpxmap and plt.pcolormesh

        Returns:
        --------
        hpxmap, im : healpix map and image
        """
        try:
            pix = hp.ang2pix(nside,lon,lat,lonlat=True)
        except TypeError:
            pix = hp.ang2pix(nside,np.radians(90-lat),np.radians(lon))

        npix = hp.nside2npix(nside)
        hpxmap = hp.UNSEEN*np.ones(npix)
        idx,cts = np.unique(pix,return_counts=True)
        hpxmap[idx] = cts

        return hpxmap,self.draw_hpxmap(hpxmap,**kwargs)

    def get_map_range(self, hpxmap, pixel=None, nside=None):
        """ Calculate the longitude and latitude range for an implicit map. """
        return healpix.get_map_range(hpxmap,pixel,nside,wrap_angle=self.wrap_angle)

    def hpx2xy(self, hpxmap, pixel=None, nside=None, xsize=800,
               lonra=None, latra=None):
        """ Convert from healpix map to longitude and latitude coordinates """
        return healpix.hpx2xy(hpxmap,pixel=pixel,nside=nside,
                              xsize=xsize,aspect=self.aspect,
                              lonra=lonra,latra=latra)


    def smooth(self,hpxmap,badval=hp.UNSEEN,sigma=None):
        """ Smooth a healpix map """
        healpix.check_hpxmap(hpxmap,None,None)
        hpxmap = healpix.masked_array(hpxmap,badval)
        hpxmap.fill_value = np.ma.median(hpxmap)
        smooth = hp.smoothing(hpxmap,sigma=np.radians(sigma),verbose=False)
        return np.ma.array(smooth,mask=hpxmap.mask)

    def draw_hpxmap(self, hpxmap, pixel=None, nside=None, xsize=800,
                    lonra=None, latra=None, badval=hp.UNSEEN, smooth=None, **kwargs):
        """
        Use pcolor/pcolormesh to draw healpix map.

        Parameters:
        -----------
        hpxmap: input healpix map
        pixel:  explicit pixel indices (required for partial maps)
        nside:  explicit nside of the map (required for partial maps)
        xsize:  resolution of the output image
        lonra:  longitude range [-180,180] (deg)
        latra:  latitude range [-90,90] (deg)
        badval: set of values considered "bad"
        smooth: gaussian smoothing kernel (deg)

        Returns:
        --------
        im,lon,lat,values : mpl image with pixel longitude, latitude (deg), and values
        """
        healpix.check_hpxmap(hpxmap,pixel,nside)
        hpxmap = healpix.masked_array(hpxmap,badval)

        if smooth:
            # To smooth we need the full map
            hpxmap = healpix.create_map(hpxmap,pixel,nside,badval)
            pixel,nside = None,None
            hpxmap = healpix.masked_array(hpxmap,badval)
            hpxmap = self.smooth(hpxmap,sigma=smooth)

        #if pixel is None:
        #    nside = hp.get_nside(hpxmap.data)
        #    pixel = np.arange(len(hpxmap),dtype=int)
        #elif nside is None:
        #    msg = "'nside' must be specified for explicit maps"
        #    raise Exception(msg)

        vmin,vmax = np.percentile(hpxmap.compressed(),[2.5,97.5])

        defaults = dict(latlon=True, rasterized=True, vmin=vmin, vmax=vmax)
        setdefaults(kwargs,defaults)

        lon,lat,values = healpix.hpx2xy(hpxmap,pixel=pixel,nside=nside,
                                        xsize=xsize,
                                        lonra=lonra,latra=latra)

        # pcolormesh doesn't work in Ortho...
        if self.projection == 'ortho':
            im = self.pcolor(lon,lat,values,**kwargs)
        else:
            # Why were we plotting the values.data?
            #im = self.pcolormesh(lon,lat,values.data,**kwargs)

            # pcolormesh recommends that values be larger than x,y
            # but basemap has problems with this (sometimes?)
            # https://github.com/matplotlib/basemap/issues/182
            try: 
                im = self.pcolormesh(lon,lat,values,**kwargs)
            except IndexError:
                im = self.pcolormesh(lon[:-1,:-1],lat[:-1,:-1],values,**kwargs)

        return im,lon,lat,values

    def draw_hpxmap_rgb(self, r, g, b, xsize=800, **kwargs):
        hpxmap = healpix.masked_array(np.array([r,g,b]))

        vmin,vmax = np.percentile(hpxmap.compressed(),[0.1,99.9])

        defaults = dict(latlon=True, rasterized=True, vmin=vmin, vmax=vmax)
        setdefaults(kwargs,defaults)

        #ax = plt.gca()
        #lonra = [-180.,180.]
        #latra = [-90.,90]
        
        #lon = np.linspace(lonra[0], lonra[1], xsize)
        #lat = np.linspace(latra[0], latra[1], xsize*self.aspect)
        #lon, lat = np.meshgrid(lon, lat)

        #nside = hp.get_nside(hpxmap.data)
        #try:
        #    pix = hp.ang2pix(nside,lon,lat,lonlat=True)
        #except TypeError:
        #    pix = hp.ang2pix(nside,np.radians(90-lat),np.radians(lon))

        lonra,latra = self.get_map_range(r)

        lon, lat, R = self.hpx2xy(r,lonra=lonra,latra=latra,xsize=xsize)
        _, _, G = self.hpx2xy(g,lonra=lonra,latra=latra,xsize=xsize)
        _, _, B = self.hpx2xy(b,lonra=lonra,latra=latra,xsize=xsize)

        # Colors are all normalized to R... probably not desired...
        #norm = np.nanmax(R)
        #r = self.set_scale(R,norm=norm)
        #g = self.set_scale(G,norm=norm)
        #b = self.set_scale(B,norm=norm)

        # Better?
        norm=np.percentile(R[~np.isnan(R)],97.5)
        kw = dict(log=False,sigma=0.5,norm=norm)
        r = self.set_scale(R,**kw)
        g = self.set_scale(G,**kw)
        b = self.set_scale(B,**kw)
        
        #rgb = np.array([r,g,b]).T
        color_tuples = np.array([r[:-1,:-1].filled(np.nan).flatten(), 
                                 g[:-1,:-1].filled(np.nan).flatten(), 
                                 b[:-1,:-1].filled(np.nan).flatten()]).T
        color_tuples[np.where(np.isnan(color_tuples))] = 0.0
        setdefaults(kwargs,{'color':color_tuples})

        if self.projection is 'ortho':
            im = self.pcolor(lon,lat,r,**kwargs)
        else:
            im = self.pcolormesh(lon,lat,r,**kwargs)
        plt.gca().set_facecolor((0,0,0))
        plt.draw()

        return im,lon,lat,r,color_tuples

    def draw_focal_planes(self, ra, dec, **kwargs):
        from skymap.instrument.decam import DECamFocalPlane
        defaults = dict(alpha=0.2,color='red',edgecolors='none',lw=0)
        setdefaults(kwargs,defaults)
        ra,dec = np.atleast_1d(ra,dec)
        if len(ra) != len(dec):
            msg = "Dimensions of 'ra' and 'dec' do not match"
            raise ValueError(msg)
        decam = DECamFocalPlane()
        # Should make sure axis exists....
        ax = plt.gca()
        for _ra,_dec in zip(ra,dec):
            corners = decam.project(self,_ra,_dec)
            collection = matplotlib.collections.PolyCollection(corners,**kwargs)
            ax.add_collection(collection)
        plt.draw()
    draw_decam = draw_focal_planes

    def draw_macho(self, ra, dec, **kwargs):
        from skymap.instrument.macho import MachoFocalPlane
        defaults = dict(alpha=0.2,color='gray',edgecolors='none',lw=0)
        setdefaults(kwargs,defaults)
        ra,dec = np.atleast_1d(ra,dec)
        if len(ra) != len(dec):
            msg = "Dimensions of 'ra' and 'dec' do not match"
            raise ValueError(msg)
        camera = MachoFocalPlane()
        # Should make sure axis exists....
        ax = plt.gca()
        for _ra,_dec in zip(ra,dec):
            corners = camera.project(self,_ra,_dec)
            collection = matplotlib.collections.PolyCollection(corners,**kwargs)
            ax.add_collection(collection)
        plt.draw()


    # Adapted from Reed Essick
    # https://github.com/reedessick/skymap_statistics

    @classmethod
    def read_constellations(cls):
        import json
        dirname  = get_datadir()
        filename = 'constellationsANDstars.json'
        with open(os.path.join(dirname,filename), 'r') as f:
            data = json.load(f)

        # Constellation shapes
        shapes = []
        for const in data['constellations'].values():
            for shape in const:
                shapes.append( np.degrees(shape) )
        shapes = np.array(shapes)

        # Stars
        stars = np.array(data['stars'])
        stars[:,:2] = np.degrees(stars[:,:2])

        # Boundaries
        boundaries = [np.degrees(e) for e in data['boundaries']]

        # Centers
        centers = data['centers']
        for k,v in centers.items():
            centers[k] = np.degrees(v)

        return shapes, stars, boundaries, centers

    def draw_constellations(self,**kwargs):
        defaults = dict(color='k',alpha=1.0)
        setdefaults(kwargs,defaults)
        ax = plt.gca()

        shapes, stars, boundaries, centers = self.read_constellations()

        ### add constellations
        for shape in shapes:
            self.plot(*self.proj(*shape.T),lw=0.5,**kwargs)

        ### add stars FIXME: hard coded...bad?
        mag = stars[:,-1]
        size = 7*np.max([np.ones_like(mag), 5-mag], axis=0)
        self.scatter(*self.proj(stars[:,0],stars[:,1]),
                     s=size,marker='o',edgecolor='none',**kwargs)

        ### add constellation boundaries
        # Use the safe projection
        bound = [np.array(self.proj(*b)).T for b in boundaries]
        collect = LineCollection(bound,linestyle='--',linewidth=0.5,**kwargs)
        ax.add_collection(collect)

        ### add constellation centers
        for (name, (x, y))  in centers.items():
            ax.text(*self(x, y), s=name,
                    ha='center',va='center',fontsize=8,
                    **kwargs)

    def set_scale(self, array, log=False, sigma=1.0, norm=None):
        if isinstance(array,np.ma.MaskedArray):
            out = np.ma.copy(array)
        else:
            out = np.ma.array(array,mask=np.isnan(array),fill_value=np.nan)

        if sigma > 0:
            out.data[:] = nd.gaussian_filter(out.filled(0),sigma=sigma)[:]
            
        if norm is None:
            norm = np.percentile(out.compressed(),97.5)

        if log: 
            out = np.log10(out)
            if norm: norm = np.log10(norm)

        out /= norm
        out = np.clip(out,0.0,1.0)
        return out

    def draw_inset_colorbar(self,format=None,label=None,ticks=None,fontsize=11,**kwargs):
        defaults = dict(width="25%", height="5%", loc=7,
                        bbox_to_anchor=(0.,-0.04,1,1))
        setdefaults(kwargs,defaults)

        ax = plt.gca()
        im = plt.gci()
        cax = inset_axes(ax,bbox_transform=ax.transAxes,**kwargs)
        cmin,cmax = im.get_clim()

        if (ticks is None) and (cmin is not None) and (cmax is not None):
            cmed = (cmax+cmin)/2.
            delta = (cmax-cmin)/10.
            ticks = np.array([cmin+delta,cmed,cmax-delta])

        tmin = np.min(np.abs(ticks[0]))
        tmax = np.max(np.abs(ticks[1]))

        if format is None:
            if (tmin < 1e-2) or (tmax > 1e3):
                format = '$%.1e$'
            elif (tmin > 0.1) and (tmax < 100):
                format = '$%.1f$'
            elif (tmax > 100):
                format = '$%i$'
            else:
                format = '$%.2g$'
                #format = '%.2f'

        kwargs = dict(format=format,ticks=ticks,orientation='horizontal')

        if format == 'custom':
            ticks = np.array([cmin,0.85*cmax])
            kwargs.update(format='$%.0e$',ticks=ticks)

        cbar = plt.colorbar(cax=cax,**kwargs)
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='x', labelsize=fontsize)

        if format == 'custom':
            ticklabels = cax.get_xticklabels()
            for i,l in enumerate(ticklabels):
                val,exp = ticklabels[i].get_text().split('e')
                ticklabels[i].set_text(r'$%s \times 10^{%i}$'%(val,int(exp)))
            cax.set_xticklabels(ticklabels)

        if label is not None:
            cbar.set_label(label,size=fontsize)
            cax.xaxis.set_label_position('top')

        plt.sca(ax)
        return cbar,cax

    def zoom_to_fit(self, hpxmap, pixel=None, nside=None):
        lonra, latra = self.get_map_range(hpxmap, pixel, nside)
        self.zoom_to(lonra,latra)

    def zoom_to(self, lonra, latra):
        """ Zoom the map to a specific longitude and latitude range.

        Parameters:
        -----------
        lonra : Longitude range [lonmin,lonmax]
        latra : Latitude range [latmin,latmax]

        Returns:
        --------
        None
        """

        (lonmin,lonmax), (latmin,latmax) = lonra, latra

        ax = plt.gca()
        self.llcrnrx,self.llcrnry = self(lonmin,latmin)
        self.urcrnrx,self.urcrnry = self(lonmax,latmax)

        ax.set_xlim(self.llcrnrx,self.urcrnrx)
        ax.set_ylim(self.llcrnry,self.urcrnry)

        self.set_axes_limits(ax=ax)

class McBrydeSkymap(Skymap):
    defaults = dict(Skymap.defaults,projection='mbtfpq')

    def __init__(self,*args,**kwargs):
        setdefaults(kwargs,self.defaults)
        if np.abs(kwargs['lon_0']) > 180:
            raise Exception("Basemap requires: -180 < lon_0 < 180")
        super(McBrydeSkymap,self).__init__(*args, **kwargs)

class OrthoSkymap(Skymap):

    # To get oriented on zenith:
    #lon_0=self.get_zenith(),lat_0=self.observer.lat

    defaults = dict(Skymap.defaults,projection='ortho',celestial=False)

    def __init__(self,*args,**kwargs):
        setdefaults(kwargs,self.defaults)
        super(OrthoSkymap,self).__init__(*args, **kwargs)

    def draw_meridians(self,*args,**kwargs):
        cardinal = kwargs.pop('cardinal',False)
        meridict = super(OrthoSkymap,self).draw_meridians(*args,**kwargs)
        ax = plt.gca()
        for mer in meridict.keys():
            ax.annotate(r'$%i^{\circ}$'%mer,self.proj(mer,5),ha='center')
        if cardinal:
            ax.annotate('West',xy=(1.0,0.5),ha='left',xycoords='axes fraction')
            ax.annotate('East',xy=(0.0,0.5),ha='right',xycoords='axes fraction')
        return meridict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    # Deprecated
    #def draw_great_circle(self, lon1, lat1, lon2, lat2, **kwargs):
    #    """
    #    Draw a great circle between two points.
    #
    #    Parameters:
    #    -----------
    #    lon1, lat1 : The longitude/latitude of first point
    #    lon2, lat2 : The longitude/latitude of first point
    #    kwargs     : keyword arguments to matplotlib.plot
    #
    #    Returns:
    #    --------
    #    matplotlib.plot
    #    """
    #
    #    defaults = dict(color='#1F618D')
    #    setdefaults(kwargs,defaults)
    #
    #    lon1 = np.radians(lon1)
    #    lat1 = np.radians(lat1)
    #    lon2 = np.radians(lon2)
    #    lat2 = np.radians(lat2)
    #
    #    x1, y1, z1 = self.cartesian(lon1, lat1)
    #    x2, y2, z2 = self.cartesian(lon2, lat2)
    #
    #    u = np.array([x1, y1, z1])
    #    v = np.array([x2, y2, z2])
    #    w = np.cross(np.cross(u, v), u)
    #    w /= np.linalg.norm(w)
    #
    #    tt = np.linspace(0, 2 * np.pi, 100)
    #
    #    xx = []
    #    yy = []
    #
    #    # ADW: This is where the full circle is generated. Would like
    #    # to eventually have the option to just draw the points
    #    # between lon1,lat1 and lon2,lat2
    #    for t in tt:
    #        r = u * np.cos(t) + w * np.sin(t)
    #        lon, lat = self.spherical(r[0], r[1], r[2])
    #        xx.append(np.degrees(lon))
    #        yy.append(np.degrees(lat))
    #
    #    # ADW: Deal with the axis break. This would be better be
    #    # done consistently with whatever mechanism is used to draw
    #    # countries etc.
    #    x2 = np.copy(xx)
    #    y2 = np.copy(yy)
    #    for i in range(len(xx) - 1):
    #        # jump would be 360, but could be smaller bc finite number of points
    #        # ADW: should respect wrap_angle
    #        #if np.abs(xx[i] - xx[i + 1]) > 300:
    #        if np.abs(xx[i] - xx[i + 1]) > self.wrap_angle:
    #            x1 = xx[:i + 1]
    #            x2 = xx[i + 1:]
    #            y1 = yy[:i + 1]
    #            y2 = yy[i + 1:]
    #
    #            x1, y1 = self(x1, y1)
    #            self.plot(x1, y1, **kwargs)
    #
    #    x2, y2 = self(x2, y2)
    #    return self.plot(x2, y2, **kwargs)
