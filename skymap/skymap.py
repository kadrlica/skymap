import os
from os.path import expandvars
import shutil
import time
import logging
import tempfile
import subprocess
import warnings
from collections import OrderedDict as odict

from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.cm
import pylab as plt
import numpy as np
import ephem

from skymap import get_datadir, setdefaults


class Skymap(Basemap):
    """ Skymap class """

    def __init__(self, *args, **kwargs):
        self.observer = kwargs.pop('observer',ephem.Observer())
        self.set_date(kwargs.pop('date'))
        super(Skymap,self).__init__(self,*args,**kwargs)

        self.draw_parallels()
        self.draw_meridians()

    def draw_parallels(self,*args,**kwargs):
        defaults = dict()
        if not args: defaults.update(circles=np.arange(-90,120,30))
        setdefaults(kwargs,defaults)
        return self.drawparallels(*args, **kwargs)

    def draw_meridians(self,*args,**kwargs):
        defaults = dict(labels=[1,0,0,1])
        if self.projection in ['ortho','geos','nsper','aeqd']:
            defaults.update(labels=[0,0,0,0])
        if not args: defaults.update(meridians=np.arange(0,420,60))
        setdefaults(kwargs,defaults)
        return self.drawmeridians(*args,**kwargs)
        
    def proj(self,lon,lat):
        """ Remove points outside of projection """
        # Should this overload __call__?
        x, y = self(np.atleast_1d(lon),np.atleast_1d(lat))
        x[x > 1e29] = None
        y[y > 1e29] = None
        return x, y

    def set_date(self,date):
        date = ephem.Date(date) if date else ephem.now()
        self.observer.date = date

    def get_zenith(self):
        # RA and Dec of zenith
        lon_zen, lat_zen = np.degrees(self.observer.radec_of(0,'90'))
        return -lon_zen

    @staticmethod
    def roll(ra,dec):
        idx = np.abs(ra - 180).argmin()
        if   (ra[idx]<180) and (ra[idx+1]>180): idx += 1
        elif (ra[idx]>180) and (ra[idx+1]<180): idx += 1
        return np.roll(ra,-idx), np.roll(dec,-idx)

    @staticmethod
    def split(ra,angle=180):
        pass

    def draw_polygon(self,filename,**kwargs):
        """ Draw a polygon footprint. """
        defaults=dict(color='k', lw=2)
        setdefaults(kwargs,defaults)

        perim = np.loadtxt(filename,dtype=[('ra',float),('dec',float)])
        self.draw_polygon_radec(perim['ra'],perim['dec'],**kwargs)

    def draw_polygon_radec(self,ra,dec,**kwargs):
        xy = self.proj(ra,dec)
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

        glon = np.linspace(0,360,200)
        glat = np.zeros_like(glon)
        ra,dec = self.roll(*gal2cel(glon,glat))

        self.draw_polygon_radec(ra,dec,**kwargs)
        
        if width:
            kwargs.update(dict(ls='--',lw=1))
            for delta in [+width,-width]:
                ra,dec = self.roll(*gal2cel(glon,glat+delta))
                self.draw_polygon_radec(ra,dec,**kwargs)
            
    def draw_magellanic_stream(self,**kwargs):
        import fitsio
        defaults = dict(xsize=800, vmin=17., vmax=25.0, rasterized=True,
                        cmap=plt.cm.binary)
        setdefaults(kwargs,defaults)

        dirname  = '/Users/kadrlica/bliss/observing/data'
        filename = 'allms_coldens_gal_nside_1024.fits'
        galhpx = fitsio.read(os.path.join(dirname,filename))['coldens']
        celhpx = obztak.utils.projector.hpx_gal2cel(galhpx)
        return self.draw_hpxmap(celhpx,**kwargs)

    def draw_sfd(self,**kwargs):
        import healpy as hp
        defaults = dict(rasterized=True,cmap=plt.cm.binary)
        setdefaults(kwargs,defaults)
        dirname  = '/Users/kadrlica/bliss/observing/data'
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

    def draw_hpxmap(self, hpxmap, xsize=800, **kwargs):
        """
        Use pcolormesh to draw healpix map
        """
        import healpy
        if not isinstance(hpxmap,np.ma.MaskedArray):
            mask = ~np.isfinite(hpxmap) | (hpxmap==healpy.UNSEEN)
            hpxmap = np.ma.MaskedArray(hpxmap,mask=mask)

        vmin,vmax = np.percentile(hpxmap.compressed(),[0.1,99.9])

        defaults = dict(latlon=True, rasterized=True, vmin=vmin, vmax=vmax)
        setdefaults(kwargs,defaults)

        ax = plt.gca()

        lon = np.linspace(0, 360., xsize)
        lat = np.linspace(-90., 90., xsize)
        lon, lat = np.meshgrid(lon, lat)

        nside = healpy.get_nside(hpxmap.data)
        try:
            pix = healpy.ang2pix(nside,lon,lat,lonlat=True)
        except TypeError:
            pix = healpy.ang2pix(nside,np.radians(90-lat),np.radians(lon))

        values = hpxmap[pix]
        #mask = ((values == healpy.UNSEEN) | (~np.isfinite(values)))
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

class McBrideSkymap(Skymap):
    def __init__(self,*args,**kwargs):
        defaults = dict(projection='mbtfpq',lon_0=0,rsphere=1.0,celestial=True)
        setdefaults(kwargs,defaults)
        super(McBrideSkymap,self).__init__(*args, **kwargs)

class OrthoSkymap(Skymap):
    def __init__(self,*args,**kwargs):
        defaults = dict(projection='ortho',celestial=True,rsphere=1.0,
                        lon_0=self.get_zenith(),lat_0=self.observer.lat)
        setdefaults(kwargs,defaults)

        super(OrthoSkymap,self).__init__(*args, **kwargs)

    def draw_meridians(self,*args,**kwargs):
        cardinal = kwargs.pop('cardinal',False)
        meridict = super(DECamOrtho,self).draw_meridians(*args,**kwargs)
        ax = plt.gca()
        for mer in meridict.keys():
            ax.annotate(r'$%i^{\circ}$'%mer,self.proj(mer,5),ha='center')
        if cardinal:
            ax.annotate('West',xy=(1.0,0.5),ha='left',xycoords='axes fraction')
            ax.annotate('East',xy=(0.0,0.5),ha='right',xycoords='axes fraction')
        return meridict
