#!/usr/bin/env python
"""
Extension for individual surveys.
"""
import os

import numpy as np
import pylab as plt
import pandas as pd
from collections import OrderedDict as odict

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot
import mpl_toolkits.axisartist as axisartist
import  mpl_toolkits.axisartist.angle_helper as angle_helper

from skymap.utils import setdefaults,get_datadir,hpx_gal2cel
from skymap.core import Skymap,McBrydeSkymap,OrthoSkymap
from skymap.constants import DECAM

# Derived from telra,teldec of 10000 exposures
DES_SN = odict([
    ('E1',dict(ra=7.874,  dec=-43.010)),
    ('E2',dict(ra=9.500,  dec=-43.999)),
    ('X1',dict(ra=34.476, dec=-4.931 )),
    ('X2',dict(ra=35.664, dec=-6.413 )),
    ('X3',dict(ra=36.449, dec=-4.601 )),
    ('S1',dict(ra=42.818, dec=0.000  )),
    ('S2',dict(ra=41.193, dec=-0.991 )),
    ('C1',dict(ra=54.274, dec=-27.113)),
    ('C2',dict(ra=54.274, dec=-29.090)),
    ('C3',dict(ra=52.647, dec=-28.101)),
])

DES_SN_LABELS = odict([
    ('SN-E',   dict(ra=15, dec=-38, ha='center')),
    ('SN-X',   dict(ra=35, dec=-13, ha='center')),
    ('SN-S',   dict(ra=55, dec=0,   ha='center')),
    ('SN-C',   dict(ra=57, dec=-36, ha='center')),
])


class SurveySkymap(Skymap):
    """Extending to survey specific functions.
    """
    def draw_maglites(self,**kwargs):
        """Draw the MagLiteS footprint"""
        defaults=dict(color='blue', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'maglites-poly.txt')
        self.draw_polygon(filename,**kwargs)

    def draw_bliss(self,**kwargs):
        """Draw the BLISS footprint"""
        defaults=dict(color='magenta', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'bliss-poly.txt')
        self.draw_polygons(filename,**kwargs)

        #data = np.genfromtxt(filename,names=['ra','dec','poly'])
        #for p in np.unique(data['poly']):
        #    poly = data[data['poly'] == p]
        #    self.draw_polygon_radec(poly['ra'],poly['dec'],**kwargs)

    def draw_des(self,**kwargs):
        """ Draw the DES footprint. """
        return self.draw_des17(**kwargs)

    def draw_des13(self,**kwargs):
        """ Draw the DES footprint. """
        defaults=dict(color='red', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'des-round13-poly.txt')
        return self.draw_polygon(filename,**kwargs)

    def draw_des17(self,**kwargs):
        """ Draw the DES footprint. """
        defaults=dict(color='blue', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'des-round17-poly.txt')
        return self.draw_polygon(filename,**kwargs)

    def draw_des_sn(self,**kwargs):
        defaults = dict(facecolor='none',edgecolor='k',lw=1,zorder=10)
        setdefaults(kwargs,defaults)
        for v in DES_SN.values():
            # This does the projection correctly, but fails at boundary
            self.tissot(v['ra'],v['dec'],DECAM,100,**kwargs)

    def draw_smash(self,**kwargs):
        """ Draw the SMASH fields. """
        defaults=dict(facecolor='none',color='k')
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'smash_fields_final.txt')
        smash=np.genfromtxt(filename,dtype=[('ra',float),('dec',float)],usecols=[4,5])
        xy = self.proj(smash['ra'],smash['dec'])
        self.scatter(*xy,**kwargs)

    def draw_decals(self,**kwargs):
        defaults=dict(color='red', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'decals-poly.txt')
        return self.draw_polygon(filename,**kwargs)

    def draw_jethwa(self,filename=None,log=True,**kwargs):
        import healpy as hp
        if not filename:
            datadir = '/home/s1/kadrlica/projects/bliss/v0/data/'
            datadir = '/Users/kadrlica/bliss/observing/data'
            filename = os.path.join(datadir,'jethwa_satellites_n256.fits.gz')
        hpxmap = hp.read_map(filename)
        if log:
            return self.draw_hpxmap(np.log10(hpxmap),**kwargs)
        else:
            return self.draw_hpxmap(hpxmap,**kwargs)

    def draw_planet9(self,**kwargs):
        from scipy.interpolate import interp1d
        from scipy.interpolate import UnivariateSpline
        defaults=dict(color='b',lw=3)
        setdefaults(kwargs,defaults)
        datadir = '/home/s1/kadrlica/projects/bliss/v0/data/'
        datadir = '/Users/kadrlica/bliss/observing/data/'
        ra_lo,dec_lo=np.genfromtxt(datadir+'p9_lo.txt',usecols=(0,1)).T
        ra_lo,dec_lo = self.roll(ra_lo,dec_lo)
        ra_lo += -360*(ra_lo > 180)
        ra_lo,dec_lo = ra_lo[::-1],dec_lo[::-1]
        ra_hi,dec_hi=np.genfromtxt(datadir+'p9_hi.txt',usecols=(0,1)).T
        ra_hi,dec_hi = self.roll(ra_hi,dec_hi)
        ra_hi += -360*(ra_hi > 180)
        ra_hi,dec_hi = ra_hi[::-1],dec_hi[::-1]

        spl_lo = UnivariateSpline(ra_lo,dec_lo)
        ra_lo_smooth = np.linspace(ra_lo[0],ra_lo[-1],360)
        dec_lo_smooth = spl_lo(ra_lo_smooth)

        spl_hi = UnivariateSpline(ra_hi,dec_hi)
        ra_hi_smooth = np.linspace(ra_hi[0],ra_hi[-1],360)
        dec_hi_smooth = spl_hi(ra_hi_smooth)

        #self.plot(ra_lo,dec_lo,latlon=True,**kwargs)
        #self.plot(ra_hi,dec_hi,latlon=True,**kwargs)
        self.plot(ra_lo_smooth,dec_lo_smooth,latlon=True,**kwargs)
        self.plot(ra_hi_smooth,dec_hi_smooth,latlon=True,**kwargs)

        orb = pd.read_csv(datadir+'P9_orbit_Cassini.csv').to_records(index=False)[::7]
        kwargs = dict(marker='o',s=40,edgecolor='none',cmap='jet_r')
        self.scatter(*self.proj(orb['ra'],orb['dec']),c=orb['cassini'],**kwargs)

    def draw_ligo(self,filename=None, log=True,**kwargs):
        import healpy as hp
        from astropy.io import fits as pyfits
        if not filename:
            datadir = '/home/s1/kadrlica/projects/bliss/v0/data/'
            datadir = '/Users/kadrlica/bliss/observing/data'
            filename = datadir + 'obsbias_heatmap_semesterA.fits'
        hpxmap = pyfits.open(filename)[0].data
        if log: self.draw_hpxmap(np.log10(hpxmap))
        else:   self.draw_hpxmap(hpxmap)

    def draw_sfd(self,filename=None,**kwargs):
        import healpy as hp
        defaults = dict(rasterized=True,cmap=plt.cm.binary)
        setdefaults(kwargs,defaults)
        if not filename:
            datadir  = '/Users/kadrlica/bliss/observing/data/'
            filename = datadir+'lambda_sfd_ebv.fits'

        galhpx = hp.read_map(filename)
        celhpx = hpx_gal2cel(galhpx)
        return self.draw_hpxmap(np.log10(celhpx),**kwargs)

class SurveyMcBryde(SurveySkymap,McBrydeSkymap): pass
class SurveyOrtho(SurveySkymap,OrthoSkymap): pass

# Original DES Formatter
# ADW: Why doesn't ZoomFormatter180 work?
class ZoomFormatterDES(angle_helper.FormatterDMS):

    def __call__(self, direction, factor, values):
        values = np.asarray(values)
        ss = np.where(values>=0, 1, -1)
        values = np.mod(np.abs(values),360)
        values -= 360*(values > 180)
        return [self.fmt_d % (s*int(v),) for (s, v) in zip(ss, values)]

class ZoomFormatter(angle_helper.FormatterDMS):
    def _wrap_angle(self, angle):
        return angle

    def __call__(self, direction, factor, values):
        values = np.asarray(values)
        values = self._wrap_angle(values)
        ticks = [self.fmt_d % (int(v),) for v in values]
        return ticks

class ZoomFormatter360(ZoomFormatter):
    def _wrap_angle(self, angle):
        """Ticks go from 0 to 360"""
        angle = np.mod(angle,360)
        return angle

class ZoomFormatter180(ZoomFormatter):
    def _wrap_angle(self, angle):
        """Ticks go from -180 to 180"""
        angle = np.mod(np.abs(angle),360)
        angle -= 360*(angle > 180)
        return angle
    
class SurveyZoom(SurveyMcBryde):
    FRAME = [[-50,-50,90,90],[10,-75,10,-75]]
    FIGSIZE=(8,5)

    def __init__(self, rect=None, *args, **kwargs):
        super(SurveyZoom,self).__init__(*args, **kwargs)
        self.create_axes(rect)

    @classmethod
    def figure(cls,**kwargs):
        """ Create a figure of proper size """
        defaults=dict(figsize=cls.FIGSIZE)
        setdefaults(kwargs,defaults)
        return plt.figure(**kwargs)

    def draw_parallels(*args, **kwargs): return
    def draw_meridians(*args, **kwargs): return

    def set_axes_limits(self, ax=None):
        if ax is None: ax = plt.gca()

        x,y = self(*self.FRAME)
        ax.set_xlim(min(x),max(x))
        ax.set_ylim(min(y),max(y))
        ax.grid(True,linestyle=':',color='k',lw=0.5)

        # Fix the aspect ratio for full-sky projections
        if self.fix_aspect:
            ax.set_aspect('equal',anchor=self.anchor)
        else:
            ax.set_aspect('auto',anchor=self.anchor)

        return ax.get_xlim(),ax.get_ylim()

    def create_tick_formatter(self):
        return ZoomFormatter()

    def create_axes(self,rect=111):
        """
        Create a special AxisArtist to overlay grid coordinates.

        Much of this taken from the examples here:
        http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html
        """

        # from curved coordinate to rectlinear coordinate.
        def tr(x, y):
            x, y = np.asarray(x), np.asarray(y)
            return self(x,y)

        # from rectlinear coordinate to curved coordinate.
        def inv_tr(x,y):
            x, y = np.asarray(x), np.asarray(y)
            return self(x,y,inverse=True)


        # Cycle the coordinates
        extreme_finder = angle_helper.ExtremeFinderCycle(20, 20)

        # Find a grid values appropriate for the coordinate.
        # The argument is a approximate number of grid lines.
        grid_locator1 = angle_helper.LocatorD(9,include_last=False)
        #grid_locator1 = angle_helper.LocatorD(8,include_last=False)
        grid_locator2 = angle_helper.LocatorD(6,include_last=False)

        # Format the values of the grid
        tick_formatter1 = self.create_tick_formatter()
        tick_formatter2 = angle_helper.FormatterDMS()

        grid_helper = GridHelperCurveLinear((tr, inv_tr),
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=tick_formatter2,
        )

        fig = plt.gcf()
        if rect is None:
            # This doesn't quite work. Need to remove the existing axis...
            rect = plt.gca().get_position()
            plt.gca().axis('off')
            ax = axisartist.Axes(fig,rect,grid_helper=grid_helper)
            fig.add_axes(ax)
        else:
            ax = axisartist.Subplot(fig,rect,grid_helper=grid_helper)
            fig.add_subplot(ax)

        ## Coordinate formatter
        def format_coord(x, y):
            return 'lon=%1.4f, lat=%1.4f'%inv_tr(x,y)
        ax.format_coord = format_coord
        ax.axis['left'].major_ticklabels.set_visible(True)
        ax.axis['right'].major_ticklabels.set_visible(False)
        ax.axis['bottom'].major_ticklabels.set_visible(True)
        ax.axis['top'].major_ticklabels.set_visible(True)

        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")
        #self.set_axes_limits()

        self.axisartist = ax
        return fig,ax

class DESSkymapMcBryde(SurveyZoom):
    """Class for plotting a zoom on DES. This is relatively inflexible."""
    # RA, DEC frame limits
    FRAME = [[-50,-50,90,90],[10,-75,10,-75]]
    FIGSIZE=(8,5)

    def __init__(self, *args, **kwargs):
        defaults = dict(lon_0=0,celestial=True)
        setdefaults(kwargs,defaults)
        super(DESSkymap,self).__init__(*args, **kwargs)

    def create_tick_formatter(self):
        return ZoomFormatterDES()
        #return ZoomFormatter180()

DESSkymap = DESSkymapMcBryde

### These should be moved into streamlib

class DESSkymapQ1(DESSkymapMcBryde):
    """Class for plotting a zoom on DES. This is relatively inflexible."""
    # RA, DEC frame limits
    FRAME = [[10,-46],[-68,-38]]

    def draw_inset_colorbar(self, *args, **kwargs):
        defaults = dict(loc=4,height="6%",width="20%",bbox_to_anchor=(0,0.05,1,1))
        setdefaults(kwargs,defaults)
        super(DESSkymapMcBryde,self).draw_inset_colorbar(*args,**kwargs)

class DESSkymapQ2(DESSkymapMcBryde):
    """Class for plotting a zoom on DES. This is relatively inflexible."""
    # RA, DEC frame limits
    FRAME = [[60,0],[8,-45]]

    def draw_inset_colorbar(self, *args, **kwargs):
        defaults = dict(loc=2,width="30%",height="4%",bbox_to_anchor=(0,-0.1,1,1))
        setdefaults(kwargs,defaults)
        super(DESSkymapMcBryde,self).draw_inset_colorbar(*args,**kwargs)

class DESSkymapQ3(DESSkymapMcBryde):
    """Class for plotting a zoom on DES. This is relatively inflexible."""
    # RA, DEC frame limits
    FRAME = [[5,60],[-68,-38]]

    def draw_inset_colorbar(self, *args, **kwargs):
        defaults = dict(loc=3,height="7%",bbox_to_anchor=(0,0.05,1,1))
        setdefaults(kwargs,defaults)
        super(DESSkymapMcBryde,self).draw_inset_colorbar(*args,**kwargs)

class DESSkymapQ4(DESSkymapMcBryde):
    """Class for plotting a zoom on DES. This is relatively inflexible."""
    # RA, DEC frame limits
    FRAME = [[90,70],[-15,-55]]

    def draw_inset_colorbar(self, *args, **kwargs):
        defaults = dict(loc=3,width="30%",height="4%",bbox_to_anchor=(0,0.05,1,1))
        setdefaults(kwargs,defaults)
        super(DESSkymapMcBryde,self).draw_inset_colorbar(*args,**kwargs)

class DESSkymapCart(SurveyZoom):
    """Class for plotting a zoom on DES. This is relatively inflexible."""
    # RA, DEC frame limits
    FRAME = [[-60,-60,100,100],[10,-75,10,-75]]
    FIGSIZE=(8,5)

    def __init__(self, *args, **kwargs):
        defaults = dict(projection='cyl',celestial=True)
        setdefaults(kwargs,defaults)
        super(DESSkymapCart,self).__init__(*args, **kwargs)

    def create_tick_formatter(self):
        return ZoomFormatterDES()
        #return ZoomFormatter180()


class DESLambert(SurveySkymap):
    """Class for plotting a zoom on DES. This is relatively inflexible."""
    # RA, DEC frame limits
    FIGSIZE=(8,5)

    def __init__(self, *args, **kwargs):
        defaults = dict(projection='laea',lon_0=120,lat_0=-90,
                        llcrnrlon=-110,llcrnrlat=8,
                        urcrnrlon=60,urcrnrlat=-15,
                        round=False,celestial=False)

        setdefaults(kwargs,defaults)
        super(SurveySkymap,self).__init__(*args, **kwargs)


    def draw_meridians(self,*args,**kwargs):

        def lon2str(deg):
            # This is a function just to remove some weird string formatting
            deg -= 360. * (deg >= 180)
            if (np.abs(deg) == 0):
                return r"$%d{}^{\circ}$"%(deg)
            elif (np.abs(deg) == 180):
                return r"$%+d{}^{\circ}$"%(np.abs(deg))
            else:
                return r"$%+d{}^{\circ}$"%(deg)

        #defaults = dict(labels=[1,1,1,1],labelstyle='+/-',
        #                fontsize=14,fmt=lon2str)
        defaults = dict(fmt=lon2str,labels=[1,1,1,1],fontsize=14)
        if not args:
            defaults.update(meridians=np.arange(0,360,60))
        setdefaults(kwargs,defaults)

        #return self.drawmeridians(*args,**kwargs)
        return super(DESLambert,self).draw_meridians(*args,**kwargs)

    def draw_parallels(self,*args,**kwargs):
        defaults = dict(labels=[0,0,0,0])
        setdefaults(kwargs,defaults)
        ret =  super(DESLambert,self).draw_parallels(*args,**kwargs)

        ax = plt.gca()
        for l in ret.keys():
            ax.annotate(r"$%+d{}^{\circ}$"%(l), self(0,l),xycoords='data',
                        xytext=(+5,+5),textcoords='offset points',
                        va='top',ha='left',fontsize=12)
        return ret

    def draw_inset_colorbar(self,*args,**kwargs):
        defaults = dict(bbox_to_anchor=(-0.01,0.07,1,1))
        setdefaults(kwargs,defaults)
        return super(DESLambert,self).draw_inset_colorbar(*args,**kwargs)


class DESPolarLambert(DESLambert):
    """Class for plotting a zoom on DES. This is relatively inflexible."""
    # RA, DEC frame limits
    FIGSIZE=(8,8)

    def __init__(self, *args, **kwargs):
        defaults = dict(projection='splaea',lon_0=60,boundinglat=-20,
                        round=True,celestial=True,parallels=True)
        setdefaults(kwargs,defaults)
        super(SurveySkymap,self).__init__(*args, **kwargs)



class BlissSkymap(SurveyZoom):
    """Class for plotting a zoom on BLISS. This is relatively inflexible."""
    # RA, DEC frame limits
    FRAME = [[130,130,0,0],[-5,-55,-5,-55]]
    FIGSIZE = (12,3)
    defaults = dict(lon_0=-100)
    wrap_angle = 60

    def __init__(self, *args, **kwargs):
        setdefaults(kwargs,self.defaults)
        super(BlissSkymap,self).__init__(*args, **kwargs)

    def create_tick_formatter(self):
        return ZoomFormatter360()
        
class MaglitesSkymap(SurveyOrtho):
    defaults = dict(SurveyOrtho.defaults,lat_0=-90)

    def draw_meridians(self,*args,**kwargs):
        defaults = dict(labels=[1,1,1,1],fontsize=14,labelstyle='+/-')
        setdefaults(kwargs,defaults)
        cardinal = kwargs.pop('cardinal',False)
        meridict = super(OrthoSkymap,self).draw_meridians(*args,**kwargs)
        return meridict
