#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

class DECamFocalPlane(object):
    """Class for storing and manipulating the corners of the DECam CCDs.
    """

    filename = os.path.join(fileio.get_datadir(),'ccd_corners_xy_fill.dat')

    def __init__(self):
        # This is not safe. Use yaml instead (extra dependency)
        self.ccd_dict = eval(''.join(open(self.filename).readlines()))

        # These are x,y coordinates
        self.corners = np.array(self.ccd_dict.values())

        # Since we don't know the original projection of the DECam
        # focal plane into x,y it is probably not worth trying to
        # deproject it right now...

        #x,y = self.ccd_array[:,:,0],self.ccd_array[:,:,1]
        #ra,dec = Projector(0,0).image2sphere(x.flat,y.flat)
        #self.corners[:,:,0] = ra.reshape(x.shape)
        #self.corners[:,:,1] = dec.reshape(y.shape)

    def rotate(self, ra, dec):
        """Rotate the corners of the DECam CCDs to a given sky location.

        Parameters:
        -----------
        ra      : The right ascension (deg) of the focal plane center
        dec     : The declination (deg) of the focal plane center

        Returns:
        --------
        corners : The rotated corner locations of the CCDs
        """
        corners = np.copy(self.corners)

        R = SphericalRotator(ra,dec)
        _ra,_dec = R.rotate(corners[:,:,0].flat,corners[:,:,1].flat,invert=True)

        corners[:,:,0] = _ra.reshape(corners.shape[:2])
        corners[:,:,1] = _dec.reshape(corners.shape[:2])
        return corners

    def project(self, basemap, ra, dec):
        """Apply the given basemap projection to the DECam focal plane at a
        location given by ra,dec.

        Parameters:
        -----------
        basemap : The DECamBasemap to project to.
        ra      : The right ascension (deg) of the focal plane center
        dec     : The declination (deg) of the focal plane center

        Returns:
        --------
        corners : Projected corner locations of the CCDs
        """
        corners = self.rotate(ra,dec)

        x,y = basemap.proj(corners[:,:,0],corners[:,:,1])

        # Remove CCDs that cross the map boundary
        x[(np.ptp(x,axis=1) > np.pi)] = np.nan

        corners[:,:,0] = x
        corners[:,:,1] = y
        return corners

############################################################
# Depricated module functions

def drawDES(basemap, color='red'):
    msg = "drawDES is depricated; use DECamBasemap.draw_des instead."
    warnings.warn(msg)
    basemap.draw_des(color=color)

def drawSMASH(basemap, color='none', edgecolor='black', marker='h', s=50):
    msg = "drawSMASH is depricated; use DECamBasemap.draw_smash instead."
    warnings.warn(msg)
    basemap.draw_smash(color=color, edgecolor=edgecolor, marker=marker, s=s)

def drawMAGLITES(basemap, color='blue'):
    msg = "drawMAGLITES is depricated; use DECamBasemap.draw_smash instead."
    warnings.warn(msg)
    basemap.draw_maglites(color=color)

def drawAirmassContour(basemap, observatory, airmass, n=360, s=50):
    msg = "drawAirmassContour is depricated; use DECamBasemap.draw_airmass instead."
    warnings.warn(msg)
    basemap.draw_airmass(observatory=observatory, airmass=airmass, n=n, s=s)

def drawZenith(basemap, observatory):
    """
    Plot a to-scale representation of the focal plane size at the zenith.
    """
    msg = "drawZenith is depricated; use DECamBasemap.draw_zenith instead."
    warnings.warn(msg)
    basemap.draw_zenith(observatory)

def drawMoon(basemap, date):
    msg = "drawMoon is depricated; use DECamBasemap.draw_moon instead."
    warnings.warn(msg)
    basemap.draw_moon(date)

############################################################

def makePlot(date=None, name=None, figsize=(10.5,8.5), dpi=80, s=50, center=None, airmass=True, moon=True, des=True, smash=False, maglites=None, bliss=None, galaxy=True):
    """
    Create map in orthographic projection
    """
    if date is None: date = ephem.now()
    if type(date) != ephem.Date:
        date = ephem.Date(date)

    survey = get_survey()
    if survey == 'maglites':
        if maglites is None: maglites = True
        if airmass is True: airmass = 2.0
    if survey == 'bliss':
        if bliss is None: bliss = True
        if airmass is True: airmass = 1.4
    if des:
        if airmass is True: airmass = 1.4

    fig = plt.figure(name, figsize=figsize, dpi=dpi)
    plt.cla()

    proj_kwargs = dict()
    if center: proj_kwargs.update(lon_0=center[0], lat_0=center[1])
    basemap = DECamOrtho(date=date, **proj_kwargs)
    observatory = basemap.observatory

    if des:      basemap.draw_des()
    if smash:    basemap.draw_smash(s=s)
    if maglites: basemap.draw_maglites()
    if bliss:    basemap.draw_bliss()
    if airmass:
        airmass = 2.0 if isinstance(airmass,bool) else airmass
        basemap.draw_airmass(observatory, airmass)
    if moon:     basemap.draw_moon(date)
    if galaxy:   basemap.draw_galaxy()

    plt.title('%s UTC'%(datestring(date)))

    return fig, basemap

def plotField(field, target_fields=None, completed_fields=None, options_basemap={}, **kwargs):
    """
    Plot a specific target field.

    Parameters:
    -----------
    field            : The specific field of interest.
    target_fields    : The fields that will be observed
    completed_fields : The fields that have been observed
    options_basemap  : Keyword arguments to the basemap constructor
    kwargs           : Keyword arguments to the matplotlib.scatter function

    Returns:
    --------
    basemap : The basemap object
    """
    if isinstance(field,np.core.records.record):
        tmp = FieldArray(1)
        tmp[0] = field
        field = tmp
    band = field[0]['FILTER']
    cmap = matplotlib.cm.get_cmap(CMAPS[band])
    defaults = dict(marker='H',s=100,edgecolor='',vmin=-1,vmax=4,cmap=cmap)
    #defaults = dict(edgecolor='none', s=50, vmin=0, vmax=4, cmap='summer_r')
    #defaults = dict(edgecolor='none', s=50, vmin=0, vmax=4, cmap='gray_r')
    setdefaults(kwargs,defaults)

    msg="%s: id=%10s, "%(datestring(field['DATE'][0],0),field['ID'][0])
    msg +="ra=%(RA)-6.2f, dec=%(DEC)-6.2f, secz=%(AIRMASS)-4.2f"%field[0]
    logging.info(msg)

    defaults = dict(date=field['DATE'][0], name='ortho')
    options_basemap = dict(options_basemap)
    setdefaults(options_basemap,defaults)
    fig, basemap = makePlot(**options_basemap)
    plt.subplots_adjust(left=0.03,right=0.97,bottom=0.03,top=0.97)

    # Plot target fields
    if target_fields is not None and len(target_fields):
        sel = target_fields['FILTER']==band
        x,y = basemap.proj(target_fields['RA'], target_fields['DEC'])
        kw = dict(kwargs,c='w',edgecolor='0.6',s=0.8*kwargs['s'])
        basemap.scatter(x[sel], y[sel], **kw)
        kw = dict(kwargs,c='w',edgecolor='0.8',s=0.8*kwargs['s'])
        basemap.scatter(x[~sel], y[~sel], **kw)

    # Plot completed fields
    if completed_fields is not None and len(completed_fields):
        sel = completed_fields['FILTER']==band
        x,y = basemap.proj(completed_fields['RA'],completed_fields['DEC'])
        kw = dict(kwargs)
        basemap.scatter(x[~sel], y[~sel], c='0.6', **kw)
        basemap.scatter(x[sel], y[sel], c=completed_fields['TILING'][sel], **kw)

    # Try to draw the colorbar
    try:
        if len(fig.axes) == 2:
            # Draw colorbar in existing axis
            colorbar = plt.colorbar(cax=fig.axes[-1])
        else:
            colorbar = plt.colorbar()
        colorbar.set_label('Tiling (%s-band)'%band)
    except TypeError:
        pass
    plt.sca(fig.axes[0])

    # Show the selected field
    x,y = basemap.proj(field['RA'], field['DEC'])
    kw = dict(kwargs,edgecolor='k')
    basemap.scatter(x,y,c=COLORS[band],**kw)

    return basemap

def plotFields(fields=None,target_fields=None,completed_fields=None,options_basemap={},**kwargs):
    # ADW: Need to be careful about the size of the marker. It
    # does not change with the size of the frame so it is
    # really safest to scale to the size of the zenith circle
    # (see PlotPointings). That said, s=50 is probably roughly ok.
    if fields is None:
        fields = completed_fields[-1]

    if isinstance(fields,np.core.records.record):
        tmp = FieldArray(1)
        tmp[0] = fields
        fields = tmp

    for i,f in enumerate(fields):
        basemap = plotField(fields[i],target_fields,completed_fields,options_basemap,**kwargs)
        if completed_fields is None: completed_fields = FieldArray()
        completed_fields = completed_fields + fields[[i]]
        plt.pause(0.001)

    return basemap

def movieFields(outfile,fields=None,target_fields=None,completed_fields=None,**kwargs):
    if os.path.splitext(outfile)[-1] not in ['.gif']:
        msg = "Only animated gif currently supported."
        raise Exception(msg)

    tmpdir = tempfile.mkdtemp()

    if fields is None:
        fields = completed_fields[-1]

    if isinstance(fields,np.core.records.record):
        tmp = FieldArray(1)
        tmp[0] = fields
        fields = tmp

    plt.ioff()
    for i,f in enumerate(fields):
        plotField(fields[i],target_fields,completed_fields,**kwargs)
        png = os.path.join(tmpdir,'field_%08i.png'%i)
        plt.savefig(png,dpi=DPI)
        if completed_fields is None: completed_fields = FieldArray()
        completed_fields = completed_fields + fields[[i]]
    plt.ion()

    cmd = 'convert -delay 10 -loop 0 %s/*.png %s'%(tmpdir,outfile)
    logging.info(cmd)
    subprocess.call(cmd,shell=True)
    shutil.rmtree(tmpdir)
    return outfile

def plotWeights(date, target_fields, weights,options_basemap={},**kwargs):
    defaults = dict(c=weights, edgecolor='none', s=50, vmin=np.min(weights), vmax=np.min(weights) + 300., cmap='Spectral')
    setdefaults(kwargs,defaults)

    defaults = dict(date=date, name='ortho')
    options_basemap = dict(options_basemap)
    setdefaults(options_basemap,defaults)
    fig, basemap = makePlot(**options_basemap)

    proj = basemap.proj(target_fields['RA'], target_fields['DEC'])
    basemap.scatter(*proj, **kwargs)

    # Try to draw the colorbar
    try:
        if len(fig.axes) == 2:
            # Draw colorbar in existing axis
            colorbar = plt.colorbar(cax=fig.axes[-1])
        else:
            colorbar = plt.colorbar()
        colorbar.set_label('Tiling')
    except TypeError:
        pass
    plt.sca(fig.axes[0])

def plotWeight(field, target_fields, weight, **kwargs):
    if isinstance(field,FieldArray):
        field = field[-1]

    date = ephem.Date(field['DATE'])

    if plt.get_fignums(): plt.cla()
    fig, basemap = obztak.utils.ortho.makePlot(date,name='weight')

    index_sort = np.argsort(weight)[::-1]
    proj = basemap.proj(target_fields['RA'][index_sort], target_fields['DEC'][index_sort])
    weight_min = np.min(weight)
    basemap.scatter(*proj, c=weight[index_sort], edgecolor='none', s=50, vmin=weight_min, vmax=weight_min + 300., cmap='Spectral')

    #cut_accomplished = np.in1d(self.target_fields['ID'], self.accomplished_field_ids)
    #proj = obztak.utils.ortho.safeProj(basemap, self.target_fields['RA'][cut_accomplished], self.target_fields['DEC'][cut_accomplished])
    #basemap.scatter(*proj, c='0.75', edgecolor='none', s=50)

    """
    cut_accomplished = np.in1d(self.target_fields['ID'],self.accomplished_fields['ID'])
    proj = obztak.utils.ortho.safeProj(basemap,
                                         self.target_fields['RA'][~cut_accomplished],
                                         self.target_fields['DEC'][~cut_accomplished])
    basemap.scatter(*proj, c=np.tile(0, np.sum(np.logical_not(cut_accomplished))), edgecolor='none', s=50, vmin=0, vmax=4, cmap='summer_r')

    proj = obztak.utils.ortho.safeProj(basemap, self.target_fields['RA'][cut_accomplished], self.target_fields['DEC'][cut_accomplished])
    basemap.scatter(*proj, c=self.target_fields['TILING'][cut_accomplished], edgecolor='none', s=50, vmin=0, vmax=4, cmap='summer_r')
    """

    # Draw colorbar in existing axis
    if len(fig.axes) == 2:
        colorbar = plt.colorbar(cax=fig.axes[-1])
    else:
        colorbar = plt.colorbar()
    colorbar.set_label('Weight')

    # Show the selected field
    proj = basemap.proj([field['RA']], [field['DEC']])
    basemap.scatter(*proj, c='magenta', edgecolor='none', s=50)

    #plt.draw()
    plt.pause(0.001)
    #fig.canvas.draw()

############################################################

def plot_progress(outfile=None,**kwargs):
    defaults = dict(edgecolor='none', s=50, vmin=0, vmax=4, cmap='summer_r')
    for k,v in defaults.items():
        kwargs.setdefault(k,v)

    fields = FieldArray.load_database()

    nites = [get_nite(date) for date in fields['DATE']]
    nite = ephem.Date(np.max(nites))
    date = '%d/%02d/%d 00:00:00'%(nite.tuple()[:3])

    fig,basemap = makePlot(date=date,moon=False,airmass=False,center=(0,-90),smash=False)
    proj = basemap.proj(fields['RA'],fields['DEC'])
    basemap.scatter(*proj, c=fields['TILING'],  **kwargs)
    colorbar = plt.colorbar()
    colorbar.set_label('Tiling')
    plt.title('Maglites Coverage (%d/%02d/%d)'%nite.tuple()[:3])

    if outfile is not None:
        plt.savefig(outfile,bbox_inches='tight')

    return fig,basemap

def plot_bliss_coverage(fields,outfile=None,**kwargs):
    BANDS = ['g','r','i','z']
    filename = os.path.join(fileio.get_datadir(),'bliss-target-fields.csv')
    target = FieldArray.read(filename)
    target = target[~np.in1d(target.unique_id,fields.unique_id)]

    fig,ax = plt.subplots(2,2,figsize=(16,9))
    plt.subplots_adjust(wspace=0.01,hspace=0.02,left=0.01,right=0.99,bottom=0.01,top=0.99)
    defaults = dict(edgecolor='none', s=12, alpha=0.2, vmin=-1, vmax=2)
    setdefaults(kwargs,defaults)

    for i,b in enumerate(BANDS):
        plt.sca(ax.flat[i])

        f = fields[fields['FILTER'] == b]
        t = target[target['FILTER'] == b]

        bmap = DECamMcBride()
        bmap.draw_des()
        bmap.draw_galaxy(10)

        proj = bmap.proj(t['RA'],t['DEC'])
        bmap.scatter(*proj, c='0.7', **kwargs)

        proj = bmap.proj(f['RA'],f['DEC'])
        bmap.scatter(*proj, c=f['TILING'], cmap=CMAPS[b], **kwargs)
        plt.gca().set_title('BLISS %s-band'%b)


def plot_maglites_nightsum(fields,nitestr):
    #fields = FieldArray.load_database()
    #new = np.char.startswith(fields['DATE'],date)
    from obztak.utils.database import Database

    date = nite2utc(nitestr)
    new = (np.array(map(utc2nite,fields['DATE'])) == nitestr)
    new_fields = fields[new]
    old_fields = fields[~new]

    kwargs = dict(edgecolor='none', s=50, vmin=0, vmax=4)
    fig,basemap = makePlot(date=nitestr,name='nightsum',moon=False,airmass=False,center=(0,-90),bliss=False)
    plt.title('Coverage (%s)'%nitestr)
    kwargs['cmap'] = 'gray_r'
    proj = basemap.proj(old_fields['RA'], old_fields['DEC'])
    basemap.scatter(*proj, c=old_fields['TILING'],**kwargs)

    kwargs['cmap'] = 'summer_r'
    proj = basemap.proj(new_fields['RA'], new_fields['DEC'])
    basemap.scatter(*proj, c=new_fields['TILING'],  **kwargs)
    colorbar = plt.colorbar()
    colorbar.set_label('Tiling')

    plt.plot(np.nan, np.nan,'o',color='green',mec='green',label='Observed tonight')
    plt.plot(np.nan, np.nan,'o',color='0.7',mec='0.7',label='Observed previously')
    plt.legend(fontsize=10,loc='lower left',scatterpoints=1)
    plt.savefig('nightsum_coverage_%s.png'%nitestr,bbox_inches='tight')

    db = Database()
    db.connect()

    query = """
select id, qc_fwhm as psf, qc_teff as teff from exposure
where exptime = 90 and delivered = True and propid = '2016A-0366'
and qc_teff is not NULL and qc_fwhm is not NULL
and to_timestamp(utc_beg) %s '%s'
"""

    new = db.query2recarray(query%('>',date))
    old = db.query2recarray(query%('<',date))

    nbins = 35
    kwargs = dict(normed=True)
    step_kwargs = dict(kwargs,histtype='step',lw=3.5)
    fill_kwargs = dict(kwargs,histtype='stepfilled',lw=1.0,alpha=0.7)

    plt.figure()
    step_kwargs['bins'] = np.linspace(0.5,2.5,nbins)
    fill_kwargs['bins'] = np.linspace(0.5,2.5,nbins)
    plt.hist(new['psf'],color='green',zorder=10, label='Observed tonight', **fill_kwargs)
    plt.hist(new['psf'],color='green',zorder=10, **step_kwargs)
    plt.hist(old['psf'],color='0.5', label='Observed previously', **fill_kwargs)
    plt.hist(old['psf'],color='0.5', **step_kwargs)
    plt.axvline(1.20,ls='--',lw=2,color='gray')
    plt.legend()
    plt.title('Seeing (%s)'%nitestr)
    plt.xlabel('FWHM (arcsec)')
    plt.ylabel('Normalized Number of Exposures')
    plt.savefig('nightsum_psf_%s.png'%nitestr,bbox_inches='tight')

    plt.figure()
    step_kwargs['bins'] = np.linspace(0,1.5,nbins)
    fill_kwargs['bins'] = np.linspace(0,1.5,nbins)
    plt.hist(new['teff'],color='green',zorder=10,label='Observed tonight', **fill_kwargs)
    plt.hist(new['teff'],color='green',zorder=10, **step_kwargs)
    plt.hist(old['teff'],color='0.5',label='Observed previously', **fill_kwargs)
    plt.hist(old['teff'],color='0.5', **step_kwargs)
    plt.axvline(0.25,ls='--',lw=2,color='gray')
    plt.legend()
    plt.title('Effective Depth (%s)'%nitestr)
    plt.xlabel('Teff')
    plt.ylabel('Normalized Number of Exposures')
    plt.savefig('nightsum_teff_%s.png'%nitestr,bbox_inches='tight')


def plot_bliss_nightsum(fields,nitestr):
    plot_bliss_coverage(fields)
    plt.savefig('nightsum_coverage_%s.png'%nitestr)

    new = (np.array(map(utc2nite,fields['DATE'])) == nitestr)
    new_fields = fields[new]
    old_fields = fields[~new]

    db = Database()
    db.connect()

    query = """select id, qc_fwhm as psf, qc_teff as teff from exposure
where exptime = 90 and delivered = True and propid = '%s'
and qc_teff is not NULL and qc_fwhm is not NULL
and to_timestamp(utc_beg) %s '%s'
"""

    new = db.query2recarray(query%(fields.PROPID,'>',datestr(date)))
    try:
        old = db.query2recarray(query%(fields.PROPID,'<',date))
    except ValueError as e:
        print(e)
        old = np.recarray(0,dtype=new.dtype)

    nbins = 35
    kwargs = dict(normed=True)
    step_kwargs = dict(kwargs,histtype='step',lw=3.5)
    fill_kwargs = dict(kwargs,histtype='stepfilled',lw=1.0,alpha=0.7)

    plt.figure()
    step_kwargs['bins'] = np.linspace(0.5,2.5,nbins)
    fill_kwargs['bins'] = np.linspace(0.5,2.5,nbins)
    plt.hist(new['psf'],color='green',zorder=10, label='Observed tonight', **fill_kwargs)
    plt.hist(new['psf'],color='green',zorder=10, **step_kwargs)
    plt.hist(old['psf'],color='0.5', label='Observed previously', **fill_kwargs)
    plt.hist(old['psf'],color='0.5', **step_kwargs)
    plt.axvline(1.20,ls='--',lw=2,color='gray')
    plt.legend()
    plt.title('Seeing (%s)'%nitestr)
    plt.xlabel('FWHM (arcsec)')
    plt.ylabel('Normalized Number of Exposures')
    plt.savefig('nightsum_psf_%s.png'%nitestr,bbox_inches='tight')

    plt.figure()
    step_kwargs['bins'] = np.linspace(0,1.5,nbins)
    fill_kwargs['bins'] = np.linspace(0,1.5,nbins)
    plt.hist(new['teff'],color='green',zorder=10,label='Observed tonight', **fill_kwargs)
    plt.hist(new['teff'],color='green',zorder=10, **step_kwargs)
    plt.hist(old['teff'],color='0.5',label='Observed previously', **fill_kwargs)
    plt.hist(old['teff'],color='0.5', **step_kwargs)
    plt.axvline(0.25,ls='--',lw=2,color='gray')
    plt.legend()
    plt.title('Effective Depth (%s)'%nitestr)
    plt.xlabel('Teff')
    plt.ylabel('Normalized Number of Exposures')
    plt.savefig('nightsum_teff_%s.png'%nitestr,bbox_inches='tight')


if __name__ == '__main__':
    makePlot('2016/2/10 03:00')

############################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

