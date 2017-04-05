#!/usr/bin/env python
"""
Core skymap classes
"""
import os
from os.path import expandvars
import logging
from collections import OrderedDict as odict

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import pyproj
import matplotlib
import pylab as plt
import numpy as np
import ephem
import healpy as hp

from skymap.utils import setdefaults,get_datadir
from skymap.utils import cel2gal, gal2cel

class Skymap(Basemap):
    """ Skymap base class. """

    def __init__(self, *args, **kwargs):
        self.set_observer(kwargs.pop('observer',None))
        self.set_date(kwargs.pop('date',None))

        defaults = dict(celestial=True,rsphere=1.0)
        setdefaults(kwargs,defaults)

        super(Skymap,self).__init__(self,*args,**kwargs)

        self.draw_parallels()
        self.draw_meridians()

    def set_observer(self, observer):
        observer = ephem.Observer(observer) if observer else ephem.Observer()
        self.observer = observer

    def set_date(self,date):
        date = ephem.Date(date) if date else ephem.now()
        self.observer.date = date

    def draw_parallels(self,*args,**kwargs):
        defaults = dict(labels=[1,0,0,1])
        if not args: defaults.update(circles=np.arange(-90,120,30))
        if self.projection in ['ortho','geos','nsper','aeqd']:
            defaults.update(labels=[0,0,0,0])
        setdefaults(kwargs,defaults)
        return self.drawparallels(*args, **kwargs)

    def draw_meridians(self,*args,**kwargs):
        defaults = dict(labels=[1,0,0,1])
        if self.projection in ['ortho','geos','nsper','aeqd']:
            defaults.update(labels=[0,0,0,0])
        if not args: defaults.update(meridians=np.arange(0,420,60))
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

    def draw_great_circle(self, lon1, lat1, lon2, lat2, **kwargs):
        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)

        x1, y1, z1 = self.cartesian(lon1, lat1)
        x2, y2, z2 = self.cartesian(lon2, lat2)

        u = np.array([x1, y1, z1])
        v = np.array([x2, y2, z2])
        w = np.cross(np.cross(u, v), u)
        w /= np.linalg.norm(w)
 
        tt = np.linspace(0, 2 * np.pi, 100)

        xx = []
        yy = []
        for t in tt:
            r = u * np.cos(t) + w * np.sin(t)
            lon, lat = self.spherical(r[0], r[1], r[2])
            xx.append(np.degrees(lon))
            yy.append(np.degrees(lat))

        x2 = np.copy(xx)
        y2 = np.copy(yy)
        for i in range(len(xx) - 1):
            if np.abs(xx[i] - xx[i + 1]) > 300: # jump would be 360, but could be smaller bc finite number of points
                x1 = xx[:i + 1]
                x2 = xx[i + 1:]
                y1 = yy[:i + 1]
                y2 = yy[i + 1:]

                x1, y1 = self(x1, y1)
                if 'c' not in kwargs:
                    self.plot(x1, y1, c='#1F618D', **kwargs)
                else:
                    self.plot(x1, y1, **kwargs)
        
        x2, y2 = self(x2, y2)
        if 'c' not in kwargs:
            return self.plot(x2, y2, c='#1F618D', **kwargs)
        else:
            return self.plot(x2, y2, **kwargs)

    def proj(self,lon,lat):
        """ Remove points outside of projection """
        # Should this overload __call__?
        x, y = self(np.atleast_1d(lon),np.atleast_1d(lat))
        x[x > 1e29] = None
        y[y > 1e29] = None
        return x, y

    def get_zenith(self):
        # RA and Dec of zenith
        lon_zen, lat_zen = np.degrees(self.observer.radec_of(0,'90'))
        return -lon_zen

    @staticmethod
    def roll(ra,dec,wrap=180.):
        """ Roll an ra,dec combination to split 180 boundary """
        idx = np.abs(ra - wrap).argmin()
        if idx+1 == len(ra): idx = 0
        elif (ra[idx] == wrap): idx += 1
        elif (ra[idx]<wrap) and (ra[idx+1]>wrap): idx += 1
        elif (ra[idx]>wrap) and (ra[idx+1]<wrap): idx += 1

        return np.roll(ra,-idx), np.roll(dec,-idx)

    def draw_polygon(self,filename,**kwargs):
        """ Draw a polygon footprint. """
        defaults=dict(color='k', lw=2)
        setdefaults(kwargs,defaults)

        poly = np.loadtxt(filename,dtype=[('ra',float),('dec',float)])
        self.draw_polygon_radec(poly['ra'],poly['dec'],**kwargs)

    def draw_polygon_radec(self,ra,dec,**kwargs):
        xy = self.proj(*self.roll(ra,dec))
        self.plot(*xy,**kwargs)

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
        ra,dec = self.roll(*gal2cel(glon,glat))
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

    def draw_lmc(self):
        proj = self.proj(RA_LMC,DEC_LMC)
        self.tissot(RA_LMC,DEC_LMC,RADIUS_LMC,100,fc='0.7',ec='0.5')
        plt.text(proj[0],proj[1], 'LMC', weight='bold',
                 fontsize=10, ha='center', va='center', color='k')

    def draw_smc(self):
        proj = self.proj(RA_SMC,DEC_SMC)
        self.tissot(RA_SMC,DEC_SMC,RADIUS_SMC,100,fc='0.7',ec='0.5')
        plt.text(proj[0],proj[1], 'SMC', weight='bold',
                 fontsize=8, ha='center', va='center', color='k')

    def draw_fields(self,fields,**kwargs):
        defaults = dict(edgecolor='none',s=15)
        if self.projection == 'ortho': defaults.update(s=50)
        if 'FILTER' in fields.dtype.names:
            colors = [COLORS[b] for b in fields['FILTER']]
        defaults.update(c=colors)
        setdefaults(kwargs,defaults)
        self.scatter(*self.proj(fields['RA'],fields['DEC']),**kwargs)

    def draw_hist2d(self, lon, lat, nside=256, **kwargs):
        """
        Draw a 2d histogram of coordinantes x,y.

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

    def draw_hpxmap(self, hpxmap, xsize=800, **kwargs):
        """
        Use pcolormesh to draw healpix map
        """
        if not isinstance(hpxmap,np.ma.MaskedArray):
            mask = ~np.isfinite(hpxmap) | (hpxmap==hp.UNSEEN)
            hpxmap = np.ma.MaskedArray(hpxmap,mask=mask)

        vmin,vmax = np.percentile(hpxmap.compressed(),[0.1,99.9])

        defaults = dict(latlon=True, rasterized=True, vmin=vmin, vmax=vmax)
        setdefaults(kwargs,defaults)

        ax = plt.gca()

        lon = np.linspace(0, 360., xsize)
        lat = np.linspace(-90., 90., xsize)
        lon, lat = np.meshgrid(lon, lat)

        nside = hp.get_nside(hpxmap.data)
        try:
            pix = hp.ang2pix(nside,lon,lat,lonlat=True)
        except TypeError:
            pix = hp.ang2pix(nside,np.radians(90-lat),np.radians(lon))

        values = hpxmap[pix]
        #mask = ((values == hp.UNSEEN) | (~np.isfinite(values)))
        #values = np.ma.array(values,mask=mask)
        if self.projection is 'ortho':
            im = self.pcolor(lon,lat,values,**kwargs)
        else:
            im = self.pcolormesh(lon,lat,values,**kwargs)

        return im

    def draw_focal_planes(self, ra, dec, **kwargs):
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

class McBrydeSkymap(Skymap):
    def __init__(self,*args,**kwargs):
        defaults = dict(projection='mbtfpq',lon_0=0,lat_0=0)
        setdefaults(kwargs,defaults)
        super(McBrydeSkymap,self).__init__(*args, **kwargs)

class OrthoSkymap(Skymap):
    def __init__(self,*args,**kwargs):
        self.set_observer(kwargs.pop('observer',None))
        self.set_date(kwargs.pop('date',None))

        defaults = dict(projection='ortho',lon_0=self.get_zenith(),
                        lat_0=self.observer.lat)
        setdefaults(kwargs,defaults)

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
