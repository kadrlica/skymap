#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

from skymapy.skymap import Skymap

class DESSkymap(Skymap):

    def draw_maglites(self,**kwargs):
        defaults=dict(color='blue', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'maglites-poly.txt')
        self.draw_polygon(filename,**kwargs)

    def draw_bliss(self,**kwargs):
        defaults=dict(color='magenta', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'bliss-poly.txt')
        data = np.genfromtxt(filename,names=['ra','dec','poly'])
        for p in np.unique(data['poly']):
            poly = data[data['poly'] == p]
            self.draw_polygon_radec(poly['ra'],poly['dec'],**kwargs)

    def draw_des(self,**kwargs):
        """ Draw the DES footprint on this Basemap instance.
        """
        defaults=dict(color='red', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'round13-poly.txt')
        self.draw_polygon(filename,**kwargs)

    def draw_smash(self,**kwargs):
        defaults=dict(facecolor='none',color='k')
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'smash_fields_final.txt')
        smash=np.genfromtxt(filename,dtype=[('ra',float),('dec',float)],usecols=[4,5])
        xy = self.proj(smash['ra'],smash['dec'])
        self.scatter(*xy,**kwargs)

    def draw_decals(self,**kwargs):
        defaults=dict(color='red', lw=2)
        setdefaults(kwargs,defaults)

        filename = os.path.join(get_datadir(),'decals-perimeter.txt')
        decals = np.genfromtxt(filename,names=['poly','ra','dec'])
        poly1 = decals[decals['poly'] == 1]
        poly2 = decals[decals['poly'] == 2]
        #self.draw_polygon_radec(poly1['ra'],poly1['dec'],**kwargs)
        #self.draw_polygon_radec(poly2['ra'],poly2['dec'],**kwargs)
        self.scatter(*self.proj(poly1['ra'],poly1['dec']))
        self.scatter(*self.proj(poly2['ra'],poly2['dec']))

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

        orb = fileio.csv2rec(datadir+'P9_orbit_Cassini.csv')[::7]
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

    def draw_sfd(self,**kwargs):
        import healpy as hp
        defaults = dict(rasterized=True,cmap=plt.cm.binary)
        setdefaults(kwargs,defaults)
        dirname  = '/Users/kadrlica/bliss/observing/data'
        filename = 'lambda_sfd_ebv.fits'

        galhpx = hp.read_map(os.path.join(dirname,filename))
        celhpx = obztak.utils.projector.hpx_gal2cel(galhpx)
        return self.draw_hpxmap(np.log10(celhpx),**kwargs)
