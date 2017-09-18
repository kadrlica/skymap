#!/usr/bin/env python
"""
Core skymap classes
"""
import os
from os.path import expandvars
import logging
from collections import OrderedDict as odict


import matplotlib
import pylab as plt
import numpy as np
import ephem
import healpy as hp
import scipy.ndimage as nd

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

    defaults = dict(celestial=True,rsphere=1.0)

    def __init__(self, *args, **kwargs):
        self.set_observer(kwargs.pop('observer',None))
        self.set_date(kwargs.pop('date',None))

        setdefaults(kwargs,self.defaults)

        super(Skymap,self).__init__(*args,**kwargs)

        self.draw_parallels()
        self.draw_meridians()
        self.wrap_angle = 180

    def set_observer(self, observer):
        observer = ephem.Observer(observer) if observer else ephem.Observer()
        self.observer = observer

    def set_date(self,date):
        date = ephem.Date(date) if date else ephem.now()
        self.observer.date = date

    def draw_parallels(self,*args,**kwargs):
        defaults = dict(labels=[1,0,0,1],labelstyle='+/-')
        if not args:
            defaults.update(circles=np.arange(-90,120,30))
        if self.projection in ['ortho','geos','nsper','aeqd']:
            defaults.update(labels=[0,0,0,0])
        setdefaults(kwargs,defaults)
        return self.drawparallels(*args, **kwargs)

    def draw_meridians(self,*args,**kwargs):
        defaults = dict(labels=[1,0,0,1],labelstyle='+/-')
        if self.projection in ['ortho','geos','nsper','aeqd']:
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
        """ Roll an ra,dec combination to split 180 boundary 
        Parameters:
        -----------
        ra : right ascension (deg)
        dec: declination (deg)
        wrap_angle : angle to wrap at (deg)
        """
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        # Find the index of the entry closest to the wrap angle
        idx = np.abs(ra - wrap).argmin()
        # First or last index: ignore
        if idx == 0 or idx+1 == len(ra): return ra, dec
        # Value exactly equals wrap, choose next value
        elif (ra[idx] == wrap): idx += 1
        # Wrap angle sandwiched
        elif (ra[idx]<wrap) and (ra[idx+1]>wrap): idx += 1
        elif (ra[idx]<wrap) and (ra[idx-1]>wrap): idx += 0
        elif (ra[idx]>wrap) and (ra[idx+1]<wrap): idx += 1
        elif (ra[idx]>wrap) and (ra[idx-1]<wrap): idx += 0
        # There is no wrap: ignore
        else: return ra, dec

        return np.roll(ra,-idx), np.roll(dec,-idx)

    def draw_polygon(self,filename,**kwargs):
        """ Draw a polygon footprint. """
        defaults=dict(color='k', lw=2)
        setdefaults(kwargs,defaults)

        poly = np.loadtxt(filename,dtype=[('ra',float),('dec',float)])
        self.draw_polygon_radec(poly['ra'],poly['dec'],**kwargs)

    def draw_polygon_radec(self,ra,dec,**kwargs):
        xy = self.proj(*self.roll(ra,dec,self.wrap_angle))
        self.plot(*xy,**kwargs)

    def draw_polygons(self,filename,**kwargs):
        """Draw a text file containing multiple polygons"""
        data = np.genfromtxt(filename,names=['ra','dec','poly'])
        for p in np.unique(data['poly']):
            poly = data[data['poly'] == p]
            self.draw_polygon_radec(poly['ra'],poly['dec'],**kwargs)


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

    def draw_hpxmap(self, hpxmap, pixel=None, nside=None, xsize=800,
                    lonra=None, latra=None, **kwargs):
        """
        Use pcolor/pcolormesh to draw healpix map.
        """
        if not isinstance(hpxmap,np.ma.MaskedArray):
            mask = ~np.isfinite(hpxmap) | (hpxmap==hp.UNSEEN)
            hpxmap = np.ma.MaskedArray(hpxmap,mask=mask)

        if pixel is None:
            nside = hp.get_nside(hpxmap.data)
            pixel = np.arange(len(hpxmap),dtype=int)
        elif nside is None:
            msg = "'nside' must be specified for explicit maps"
            raise Exception(msg)

        vmin,vmax = np.percentile(hpxmap.compressed(),[2.5,97.5])

        defaults = dict(latlon=True, rasterized=True, vmin=vmin, vmax=vmax)
        setdefaults(kwargs,defaults)

        lon,lat,values = healpix.hpx2xy(hpxmap,pixel=pixel,nside=nside,
                                        xsize=xsize,
                                        lonra=lonra,latra=latra)

        if self.projection == 'ortho':
            im = self.pcolor(lon,lat,values,**kwargs)
        else:
            # Why were we plotting the values.data?
            #im = self.pcolormesh(lon,lat,values.data,**kwargs)
            im = self.pcolormesh(lon,lat,values,**kwargs)

        return im,lon,lat,values

    def draw_hpxmap_rgb(self, r, g, b, xsize=800, **kwargs):
        hpxmap = np.array([r,g,b])
        if not isinstance(hpxmap,np.ma.MaskedArray):
            mask = ~np.isfinite(hpxmap) | (hpxmap==hp.UNSEEN)
            hpxmap = np.ma.MaskedArray(hpxmap,mask=mask)

        vmin,vmax = np.percentile(hpxmap.compressed(),[0.1,99.9])

        defaults = dict(latlon=True, rasterized=True, vmin=vmin, vmax=vmax)
        setdefaults(kwargs,defaults)

        ax = plt.gca()
        lonra = [-180.,180.]
        latra = [-90.,90]

        lon = np.linspace(lonra[0], lonra[1], xsize)
        lat = np.linspace(latra[0], latra[1], xsize*self.aspect)
        lon, lat = np.meshgrid(lon, lat)

        nside = hp.get_nside(hpxmap.data)
        try:
            pix = hp.ang2pix(nside,lon,lat,lonlat=True)
        except TypeError:
            pix = hp.ang2pix(nside,np.radians(90-lat),np.radians(lon))

        lon,lat,R = self.hpx2xy(r,lonra=lonra,latra=latra,xsize=xsize)
        G = self.hpx2xy(g,lonra=lonra,latra=latra,xsize=xsize)[-1]
        B = self.hpx2xy(b,lonra=lonra,latra=latra,xsize=xsize)[-1]

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
        color_tuples = np.array([r.filled(np.nan).flatten(), 
                                 g.filled(np.nan).flatten(), 
                                 b.filled(np.nan).flatten()]).T
        color_tuples[np.where(np.isnan(color_tuples))] = 0.0
        setdefaults(kwargs,{'color':color_tuples})
        
        if self.projection is 'ortho':
            im = self.pcolor(lon,lat,r,**kwargs)
        else:
            im = self.pcolormesh(lon,lat,r,**kwargs)

        return im

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

    def set_scale(self, array, log=False, sigma=1.0, norm=None):
        if isinstance(array,np.ma.MaskedArray):
            out = copy.deepcopy(array)
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

    def draw_inset_colorbar(self,format=None,label=None,ticks=None):
        ax = plt.gca()
        im = plt.gci()
        cax = inset_axes(ax, width="25%", height="5%", loc=7,
                         bbox_to_anchor=(0.,-0.04,1,1),
                         bbox_transform=ax.transAxes
                         )
        cmin,cmax = im.get_clim()
        cmed = (cmax+cmin)/2.
        delta = (cmax-cmin)/10.

        if not ticks:
            ticks = np.array([cmin+delta,cmed,cmax-delta])

        tmin = np.min(np.abs(ticks[0]))
        tmax = np.max(np.abs(ticks[1]))

        if format is None:
            if (tmin < 1e-2) or (tmax > 1e3):
                format = '%.1e'
            elif (tmin > 0.1) and (tmax < 100):
                format = '%.1f'
            elif (tmax > 100):
                format = '%i'
            else:
                format = '%.2g'
                #format = '%.2f'

        kwargs = dict(format=format,ticks=ticks,orientation='horizontal')

        if format == 'custom':
            ticks = np.array([cmin,0.85*cmax])
            kwargs.update(format='%.0e',ticks=ticks)

        cbar = plt.colorbar(cax=cax,**kwargs)
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='x', labelsize=11)

        if format == 'custom':
            ticklabels = cax.get_xticklabels()
            for i,l in enumerate(ticklabels):
                val,exp = ticklabels[i].get_text().split('e')
                ticklabels[i].set_text(r'$%s \times 10^{%i}$'%(val,int(exp)))
            cax.set_xticklabels(ticklabels)

        if label is not None:
            cbar.set_label(label,size=11)
            cax.xaxis.set_label_position('top')

        plt.sca(ax)
        return cbar,cax

    def zoom_to_fit(self, hpxmap, pixel=None, nside=None):
        ax = plt.gca()
        (lonmin,lonmax), (latmin,latmax) = self.get_map_range(hpxmap, pixel, nside)
        self.llcrnrx,self.llcrnry = self(lonmin,latmin)
        self.urcrnrx,self.urcrnry = self(lonmax,latmax)

        ax.set_xlim(self.llcrnrx,self.urcrnrx)
        ax.set_ylim(self.llcrnry,self.urcrnry)

        self.set_axes_limits(ax=ax)

class McBrydeSkymap(Skymap):
    defaults = dict(projection='mbtfpq',lon_0=0,lat_0=0,celestial=True)

    def __init__(self,*args,**kwargs):
        setdefaults(kwargs,self.defaults)
        super(McBrydeSkymap,self).__init__(*args, **kwargs)

class OrthoSkymap(Skymap):

    # To get oriented on zenith:
    #lon_0=self.get_zenith(),lat_0=self.observer.lat

    defaults = dict(projection='ortho',lon_0=0,lat_0=0)

    def __init__(self,*args,**kwargs):
        self.set_observer(kwargs.pop('observer',None))
        self.set_date(kwargs.pop('date',None))

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
