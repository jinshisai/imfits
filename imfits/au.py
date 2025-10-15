'''
Analysis utilities for Imfits.
'''

import copy
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import griddata
import scipy.ndimage

from imfits import Imfits
from imfits.mapunit import IbeamTOjpp, IcgsTObeam, IbeamTOjpp
from .fitfuncs import fit_lnprof


# constant
clight = 2.99792458e10  # light speed [cm s^-1]
auTOkm = 1.495978707e8  # AU --> km
auTOcm = 1.495978707e13 # AU --> cm
auTOpc = 4.85e-6        # au --> pc
pcTOau = 2.06e5         # pc --> au
pcTOcm = 3.09e18        # pc --> cm



# Functions
def match_images(image, ref_image,):
    '''
    Regrid an input image so that it matches the reference image grid.

    Parameters
    ----------
    input (Imfits): Input image to be regridded.
    reference (Imfits): Reference image.
    '''

    xx_ref = ref_image.xx.copy()
    yy_ref = ref_image.yy.copy()

    xinp = image.xx.ravel()
    yinp = image.yy.ravel()

    # interpolate
    if image.naxis == 2:
        data_reg = griddata((xinp, yinp), image.data.reshape(image.data.size), 
        (xx_ref, yy_ref), method='cubic',rescale=True)
    elif image.naxis == 3:
        data_reg = np.array([ griddata((xinp, yinp), image.data[i,:,:].reshape(image.data[i,:,:].size), 
            (xx_ref, yy_ref), method='cubic',rescale=True) for i in range(image.nv) ])
    elif image.naxis == 4:
        data_reg = np.array([[ griddata((xinp, yinp), image.data[i, j,:,:].reshape(image.data[i, j,:,:].size), 
            (xx_ref, yy_ref), method='cubic',rescale=True) 
        for j in range(image.nv) ] for i in range(image.ns) ])

    return data_reg



def getflux(image, rms=None, aptype='circle',
    r=None, semimaj = None, semimin=None, pa=None,
    istokes = 0, ivelocity=0, 
    inmode_data=False, dx=None, dy=None, beam=None, xx=None, yy=None, ):
    '''
    Get flux from an image.

    Parameters
    ----------
    image (object): Input image. Must be Imfits object,
     or ndarray when inmode_data=True.
    rms (float): rms noise level.
    aptype (str): Aperture type. Circule or ellipse are supported.
    r (float): Radial range where flux is measured (arcsec).
    semimaj (float): Semi-major axis of an elliptical aperture (arcsec).
    semimin (float): Semi-minor axis of an elliptical aperture (arcsec).
    pa (float): Position angle of an elliptical aperture measured from north to east (deg).
    inmode_data (bool): The input image will be treated as ndarray if True.
     Default False.
    istokes (int): Index for the stokes axis.
    ivelocity (int): Index for the velocity axis.
    '''
    # read data
    if inmode_data:
        _d = image.copy()
        q = [ False if i is not None else True for i in
        [dx, dy, beam, xx, yy]]
        if any(q):
            print('ERROR\tgetflux: necessary parameters are missing.')
            print('ERROR\tgetflux: all subparameters must be provided\
                when inmode_data = True.')
            return 0
        header = []
    else:
        _d = image.data.copy()
        xx = image.xx * 3600. # deg --> arcsec
        yy = image.yy * 3600. # deg --> arcsec
        dx, dy = np.abs(image.delx), np.abs(image.dely)
        dx *= 3600. # deg --> arcsec
        dy *= 3600. # deg --> arcsec
        beam = image.beam
        header = image.header
    # data shape
    _d = dropaxes(_d, istokes=istokes, ivelocity=ivelocity)
    # radial coordinates
    rr = np.sqrt(xx ** 2. + yy ** 2.)

    # radial range
    if aptype not in ['none', 'circle', 'ellipse']:
        print('ERROR\tgetflux: aptype must be circle or ellipse.')
        return 0

    rrange = None
    if aptype == 'circle':
        if r is not None:
            rrange = np.where(rr <= r)
        else:
            print("CAUTION\tgetflux: r is not given though aptype='circle'.")
            print("CAUTION\tgetflux: Entire map will be used for flux measurement.")
    elif aptype == 'ellipse':
        if [semimaj, semimin].count(None) == 0:
            # inverse rotation
            xxp = xx * np.cos(np.radians(pa)) - yy * np.sin(np.radians(pa))
            yyp = xx * np.sin(np.radians(pa)) + yy * np.cos(np.radians(pa))
            rrange = np.where( (yyp/semimaj)**2. + (xxp/semimin)**2. <= 1. )
        else:
            print("CAUTION\tgetflux: semimaj and/or semimin is not given though aptype='ellipse'.")
            print("CAUTION\tgetflux: Entire map will be used for flux measurement.")
    elif aptype == 'none':
        rrange = (~np.isnan(_d))

    if rrange is None:
        rrange = (~np.isnan(_d))

    # pixel size in units of beam area
    beam_area = beam[0] * beam[1] * np.pi/(4.*np.log(2.)) # arcsec^2
    ds = dx*dy/beam_area # in units of beam area

    # unit
    if 'BUNIT' in header:
        if header['BUNIT'] == 'Jy/beam':
            pass
        else:
            print('CAUTION\tgetflux: The unit of intensity seems not Jy/beam.')
            print('CAUTION\tgetflux: Currently only Jy/beam is supported.')
            print('CAUTION\tgetflux: The calculated flux might be wrong.')

    # get flux
    flux = np.sum(_d[rrange] * ds)

    if rms is not None:
        e_flux = np.sum( np.sqrt(_d[rrange].size) * rms * ds)
        return flux, e_flux
    else:
        return flux


def get_contour_radius(image, 
    rms, thr, showfig = True, savefig = False,
    outname = None):
    '''
    Calculate radius from a contour curve.

    Parameters
    ----------
     image (Imfits): Imfits object image.
     rms (float): rms noise level of the map.
     thr (float): Threshold to draw a contour curve. Should be given in a unit of rms.
                  E.g., thr = 3 will draw a 3sigma contour and then calculate a mean radius.
    '''
    im = copy.deepcopy(image)
    # mask data
    d = np.squeeze(im.data.copy())
    d[d < thr*rms] = np.nan
    d[d >= thr*rms] = 1.
    # intensity weighted mean (geometrical mean of 3sigma contour region)
    ra_mn = np.nansum(image.xx_wcs * d)/np.nansum(d)
    dec_mn = np.nansum(image.yy_wcs * d)/np.nansum(d)
    cc = SkyCoord(ra_mn, dec_mn, unit=(u.deg, u.deg), frame='icrs')
    cc_new = cc.to_string('hmsdms')
    #print(cc_new)
    im.shift_coord_center(cc_new)


    # figure
    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    # get contour
    cs = ax1.contour(im.xx * 3600., im.yy  * 3600., np.squeeze(im.data), [3.*rms])
    p = cs.get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]
    # to check
    ax1.plot(x, y, color='r', lw=2., alpha=0.7)

    # takeing mean
    r = np.sqrt(x*x + y*y) * dist
    #r_mn = np.mean(r)
    r_mn = np.median(r)
    r_sig = np.std(r)
    print('Core radius: %.2f +/- %.2f au'%(r_mn, r_sig))

    ax2.hist(r)

    if savefig:
        fig.savefig(outname)

    if showfig:
        plt.show()
    plt.close()

    return r_mn, r_sig



def get_rmsmap(cube, vwindows=[[]]):
    vaxis = cube.vaxis.copy()
    data = np.squeeze(cube.data.copy())
    if len(data.shape) !=3:
        print('ERROR\trmsmap: Input data shape is not right.')
        print('ERROR\trmsmap: Currently only Stokes I cube is supported.')
        return 0

    if type(vwindows) != list:
        print('ERROR\trmsmap: Input vwindows format is wrong.')
        print('ERROR\trmsmap: Must be a list object.')
        return 0

    if type(vwindows[0]) == float:
        indx = (vaxis >= vwindows[0]) & (vaxis < vwindows[1])
        d_masked = data[~indx, :, :]
    elif type(vwindows[0]) == list:
        if len(vwindows[0]) == 0:
            d_masked = data.copy()
        else:
            conditions = np.array([(vaxis >= vwindows[i][0]) & (vaxis < vwindows[i][1])
                for i in range(len(vwindows))])
            indx = conditions.any(axis=0)
            d_masked = data[~indx, :, :]
    else:
        print('ERROR\trmsmap: Format of vwindows elements is wrong.')
        print('ERROR\trmsmap: Must be float or list objects.')
        return 0

    return np.sqrt(np.nanmean(d_masked * d_masked, axis=0))




def get_1Dprofile(image, pa, average_side=False):
    '''
    Get one dimensional profile in a orientation at a position angule.

    Args:
        image (Imfits object): fits image
        pa (float): Position angle for the 1D cut.
        average_sides (bool): Take an average of profiles on \
                              positive and negative sides, or not.
    '''
    im_rot = imrotate(image, -pa)

    if len(im_rot.shape) == 2:
        x, y = image.yaxis.copy(), im_rot[:,image.nx//2]
    elif len(im_rot.shape) == 3:
        x, y = image.yaxis.copy(), im_rot[0,:,image.nx//2]
    elif len(im_rot.shape) == 4:
        x, y = image.yaxis.copy(), np.squeeze(im_rot[0,:,:,image.nx//2])
    else:
        print('ERROR\tget_1Dprofile: Naxis of the input fits file must be <= 4.')
        return 0

    # take average
    # x-axis must be symmetric with respect to zero
    if average_side:
        # get x-axis
        x_n = -x[x < 0.][::-1]
        x_p = x[x > 0.]
        n_xn = len(x_n)
        n_xp = len(x_p)
        x_mn = x_p if n_xn >= n_xp else x_n

        if len(im_rot.shape) == 2:
            y_n = y[x < 0.][::-1]
            y_p = y[x > 0.]

            if n_xn == n_xp:
                pass
            elif n_xn > n_xp:
                y_n  = y_n[:-1] # cut off excess
            else:
                y_p  = y_p[:-1] # cut off excess

            y_mn = 0.5*(y_n + y_p)
        else:
            nindx = np.nonzero(x < 0.)[0]
            y_n = y[:, nindx[0]:nindx[-1]+1][::-1]
            pindx = np.nonzero(x > 0.)[0]
            y_p = y[:, pindx[0]:pindx[-1]+1]
            if n_xn == n_xp:
                pass
            elif n_xn > n_xp:
                y_n  = y_n[:, :-1] # cut off excess
            else:
                y_p  = y_p[:, :-1] # cut off excess
            y_mn = np.array([
                0.5*(y_n[i,:] + y_p[i,:])
                for i in range(y.shape[0])])
        return x_mn, y_mn
    else:
        return x, y


def radial_profile(image, pa = None, 
    inc = None, step = 'Nyquist', 
    rmin = 0., rmax = None, delr = None,
    return_all = False,):
    '''
    under development
    '''
    # sky deprojection
    if (pa is not None) & (inc is not None):
        data = sky_deprojection(image, pa, inc).ravel()
    else:
        data = np.squeeze(image.data).ravel()


    xx, yy = image.xx * 3600., image.yy * 3600.
    rr = np.sqrt(xx*xx + yy*yy).ravel()

    # sampling
    if image.beam is not None:
        bmaj, bmin, bpa = image.beam
        #beam_area =  np.pi/(4.*np.log(2.)) * bmaj * bmin
    else:
        bmaj, bmin, bpa = [None]*3

    if rmax is None:
        rmax = np.nanmax(rr)

    rrange = rmax - rmin

    if delr is None:
        if step == 'Nyquist':
            delr = 0.5 * bmin
        else:
            delr = bmin / step

    r_bin_e = np.arange(rmin, rmax + delr, delr)
    r_bin = 0.5 * (r_bin_e[1:] + r_bin_e[:-1])
    nr = len(r_bin)


    prof = np.zeros(nr)
    e_prof = np.zeros(nr)
    for i, ri in enumerate(r_bin):
        indx = (rr >= r_bin_e[i]) * (rr < r_bin_e[i+1])
        #print(np.count_nonzero(indx))
        if np.count_nonzero(indx) >= 1:
            prof[i] = np.nanmean(data[indx])
            e_prof[i] = np.sqrt(np.nanvar(data[indx]))

    if return_all:
        d_sort = data[rr.argsort()]
        r_sort = np.sort(rr)
        return r_bin, prof, e_prof, r_sort, d_sort
    else:
        return r_bin, prof, e_prof


def cs_radial_profile(image, pa = None, 
    inc = None, step = 'Nyquist', return_all = False, npa = 361,
    pa_min = -180., pa_max = 180., istokes = None, ivelocity = None):
    '''
    Calculate radial profile from a given image by taking azimuthal average.

    Parameters
    ----------
     image (Imfits): Imfits image.
     pa (float): Position angle of the object. Used to deproject image.
     inc (float): Inclination angle of the object. Used to deproject image.
     step (str or int): Sampling step. Default is Nyquist sampling. Step must be given in unit of pixel.
     return_all (bool): Return all results if true.
     npa (int): Number of bin along azimuthal direction. Default 361, resulting in 1deg step.
    '''
    # sky deprojection
    if (pa is not None) & (inc is not None):
        data = sky_deprojection(image, pa, inc,
            istokes = istokes, ivelocity = ivelocity,
            conserve_flux = False)
    else:
        data = np.squeeze(image.data)

    # circular slice
    _, pa, cslice = _circular_slice(data, npa = npa,
        pa_min = pa_min, pa_max = pa_max)
    r = image.yaxis.copy() * 3600.
    r = r[image.ny//2:]

    # sampling
    bmaj, bmin, bpa = image.beam
    beam_area =  np.pi/(4.*np.log(2.)) * bmaj * bmin
    beam_area /= np.cos(np.radians(inc)) # take into account deprojection effect
    if step == 'Nyquist':
        step = int(0.5 * np.sqrt(beam_area) / image.dely / 3600.)
    elif type(step) == int:
        pass
    else:
        step = 1

    # weighting for correcting over sampling in error calculations
    circumference = r * np.pi * (pa_max - pa_min) / 180.
    w_e = np.sqrt(
    len(pa) / (2. * circumference / np.sqrt(beam_area)))

    # radial profile
    prof = np.nanmean(cslice, axis = 1)[::step]
    e_prof = np.sqrt(np.nanvar(cslice, axis = 1))[::step] #* w_e[::step]

    if return_all:
        return r[::step], pa, cslice[::step,:], w_e[::step], prof, e_prof
    else:
        return r[::step], prof, e_prof


def imrotate(image, angle=0):
    '''
    Rotate the input image

    Args:
        image (Imfits object): fits image.
        angle (float): Rotational Angle. Anti-clockwise direction will be positive (same to the Position Angle). in deg.
    '''
    import scipy.ndimage

    # check whether the array includes nan
    # nan --> 0 for interpolation
    if np.isnan(image.data).any() == False:
        pass
    elif np.isnan(image.data).any() == True:
        print ('CAUTION\timrotate: Input image array includes nan. Replace nan with 0 for interpolation when rotate image.')
        image.data[np.isnan(image.data)] = 0.

    # rotate image
    nx = image.nx
    ny = image.ny
    if image.naxis == 2:
        newimage = scipy.ndimage.rotate(image.data, -angle, reshape=False)
    elif image.naxis == 3:
        newimage = image.data.copy()
        for i in range(image.nv):
            newimage[i,:,:] = scipy.ndimage.rotate(image.data[i,:,:], -angle, reshape=False)
    elif image.naxis == 4:
        newimage = image.data.copy()
        for i in range(image.nv):
            newimage[0,i,:,:] = scipy.ndimage.rotate(image.data[0,i,:,:], -angle, reshape=False)
    else:
        print('Naxis must be <= 4.')
        return -1

    # resampling
    #mx = newimage.shape[0]
    #my = newimage.shape[1]
    #outimage = newimage[my//2 - ny//2:my//2 - ny//2 + ny, mx//2 - nx//2:mx//2 - nx//2 + nx]

    return newimage


# function
def imrotate_2d(d, angle=0.):
    # check whether the array includes nan
    # nan --> 0 for interpolation
    if np.isnan(d).any() == False:
        pass
    elif np.isnan(d).any() == True:
        #print ('CAUTION\timrotate: Input image array includes nan. Replace nan with 0 for interpolation when rotate image.')
        d[np.isnan(d)] = 0.

    # rotate image
    newimage = scipy.ndimage.rotate(d, -angle, reshape=False)
    return newimage


def gaussian_cube_fit(image, rms, 
    sampling = 'Nyquist', isaxis = 0,
    save_as_fits = False, outname = None, 
    overwrite = True, nthr = 3):
    '''
    Fit cube with Gaussian pixel by pixel
    '''
    cube = image.copy()

    if sampling == 'Nyquist':
        steps = [2,2]
        cube.sampling(steps, units = 'resolution', keep_center = True)
        cube.update_hdinfo()
    elif type(sampling) is list:
        cube.sampling(sampling, units = 'resolution', keep_center = True)
        cube.update_hdinfo()
    else:
        pass


    if cube.naxis == 3:
        data = cube.data.copy()
    elif cube.naxis == 4:
        data = cube.data[isaxis, :, :, :]
    nv, ny, nx = cube.nv, cube.ny, cube.nx

    # output
    pfit = np.empty((3, ny, nx))
    e_pfit = np.empty((3, ny, nx))

    for yi in range(ny):
        for xi in range(nx):
            speci = data[:, yi, xi]
            ndata = len(speci[speci >= 3. * rms])
            if ndata > nthr:
                popt, perr = fit_lnprof(cube.vaxis, speci, rms)
            else:
                popt, perr = [np.nan]*3, [np.nan]*3

            pfit[:, yi, xi] = popt
            e_pfit[:, yi, xi] = perr

    if save_as_fits:
        cube.data = pfit
        cube.nv = 3
        if 'NAXIS4' in cube.header: del cube.header['NAXIS4']
        cube.writeout(outname, overwrite = overwrite)

        cube.data = e_pfit
        cube.writeout(outname.replace('.fits', '_err.fits'), overwrite = overwrite)

    return pfit, e_pfit



def sky_deprojection(image, pa, inc,
    inmode_data=False, xx=[], yy=[], 
    method = 'cubic', conserve_flux = True,
    istokes = None, ivelocity = None):
    '''
    Deproject image from sky coordinates to local coordinates.

    Parameters
    ----------
    '''

    # Rotation angles
    rotdeg  = 90. - pa # 180. - pa
    rotrad  = np.radians(rotdeg)
    incrad  = np.radians(inc)

    # Data
    if inmode_data:
        data = image.copy()
        naxis = len(data.shape)
        if len(xx) * len(yy) == 0:
            print('ERROR:\tsky_deprojection: xx and yy must be given when inmode_data=True.')
            return 0
    else:
        data = image.data.copy()
        xx = image.xx.copy()
        yy = image.yy.copy()
        dx = image.delx
        dy = image.dely
        naxis = image.naxis

    if (naxis == 3) & (ivelocity is not None):
        data = data[ivelocity,:,:]
        naxis = 2
    elif (naxis == 4):
        if (istokes is not None) & (ivelocity is not None):
            data = data[istokes, ivelocity, :,:]
            naxis = 2
        elif istokes is not None:
            data = data[istokes, :, :,:]
            naxis = 3
        elif ivelocity is not None:
            data = data[:,ivelocity,:,:]
            naxis = 3

    # fill out nan with zero for interpolation
    data[np.isnan(data)] = 0.

    # Rotation of the coordinate by pa,
    #   in which xprime = xcos + ysin, and yprime = -xsin + ycos
    # now, x axis is positive in the left hand direction (x axis is inversed).
    # right hand (clockwise) rotation will be positive in angles.
    xxp = xx*np.cos(rotrad) + yy*np.sin(rotrad)
    yyp = (- xx*np.sin(rotrad) + yy*np.cos(rotrad))/np.cos(incrad)


    # 2D --> 1D
    xinp = xxp.reshape(xxp.size)
    yinp = yyp.reshape(yyp.size)
    print('Deprojecting image. Interpolation may take time.')
    if naxis == 2:
        data_reg = imrotate_2d(
        griddata((xinp, yinp), data.reshape(data.size), 
            (xx, yy), method=method,rescale=True), 
        angle=-rotdeg)
    elif naxis == 3:
        data_reg = np.array([ imrotate_2d(
            griddata((xinp, yinp), data[i,:,:].reshape(data[i,:,:].size), 
                (xx, yy), method=method,rescale=True), 
            angle=-rotdeg)
            for i in range(data.shape[0])])
    elif naxis == 4:
        data_reg = np.array([[ imrotate_2d(
            griddata((xinp, yinp), data[i, j,:,:].reshape(data[i, j,:,:].size), 
                (xx, yy), method=method,rescale=True), 
            angle=-rotdeg)
            for j in range(data.shape[1]) ] for i in range(data.shape[0]) ])
    else:
        print('ERROR\tsky_deprojection: NAXIS must be <= 4.')
        return 0

    # scaling for flux conservation
    if conserve_flux:
        data_reg *= np.cos(incrad)
        #data_reg *= np.nansum(data[np.where(np.abs(yyp) <= np.nanmax(yy))])/np.nansum(data_reg)

    return data_reg


# functions
# 2D linear function
def lnfunc_G93(del_ra, del_dec, v0, a, b):
    # See Goodman et al. (1993)
    return v0 + a*del_ra + b*del_dec

def _lnfunc_G93(x, *args):
    # See Goodman et al. (1993)
    return lnfunc_G93(x[0], x[1], *args)

def chi_lnfunc_G93(params, x, y, yerr):
    return (y - lnfunc_G93(x[0], x[1], *params)) / yerr


def lnfit2d(image, p0, rfit=None, dist=140.,
    vaxis=0, saxis=0, sigma_axis = 1, sigma_map = None, 
    full_output = False):
    '''
    Fit a two-dimensional linear function to a map by the least square fitting.
    Basically, the input map is assumed to be a velocity map and
     the measurement result will be treated as a velocity gradient.
    The fitting method is based on the one done by Goodman et al. (1993).

    Parameters:
     image (Imfits): Imfits image data.
     p0 (list): Initial guess of the fitting parameters
     rfit (float): Radius on the plane of the sky in which the fitting is performed.
     dist (float): Distance to the object.
     vaxis (int):
    '''

    # check type
    if type(image) == Imfits:
        pass
    else:
        print ('ERROR\tlnfit2d: Object type is not Imfits. Check the input.')
        return

    xx = image.xx.copy() * 3600. # deg --> arcsec
    yy = image.yy.copy() * 3600. # deg --> arcsec
    naxis = image.naxis

    # delta
    dx = np.abs(xx[0,1] - xx[0,0])
    dy = np.abs(yy[1,0] - yy[0,0])
    #print (xx.shape)

    # beam
    bmaj, bmin, bpa = image.beam # as, as, deg

    # radius
    rr = np.sqrt(xx*xx + yy*yy)


    # check data axes
    if naxis == 2:
        data = image.data.copy()
        sigma = None
        pass
    elif naxis == 3:
        nv, ny, nx = image.data.shape
        data = image.data[vaxis,:,:].copy()
        sigma = image.data[sigma_axis,:,:].copy() if sigma_axis <= nv else None
    elif naxis == 4:
        ns, nv, ny, nx = image.data.shape
        data = image.data[saxis,vaxis,:,:].copy()
        sigma = image.data[saxis,sigma_axis,:,:].copy() if sigma_axis <= nv else None
    else:
        print ('Error\tmeasure_vgrad: Input fits size is not corrected.\
            It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
        return


    if full_output:
        _data = data.copy() # save


    if sigma_map is not None:
        sigma = sigma_map


    # Nyquist sampling
    R_beam2pix = np.pi/(4.*np.log(2.)) * bmaj * bmin \
    / np.abs(dx * dy) # area ratio
    '''
    step = int(bmin/dx*0.5)
    ny, nx = xx.shape
    xx_fit = xx[0:ny:step, 0:nx:step] if step >= 1 else xx
    yy_fit = yy[0:ny:step, 0:nx:step] if step >= 1 else yy
    data_fit = data[0:ny:step, 0:nx:step] if step >= 1 else data
    if step <= 0:
        print('WARNING\tlnfit2d: Sampling given pixel and beam size is %i.'%step)
        print('WARNING\tlnfit2d: Sampling step is less than zero.')
        print('WARNING\tlnfit2d: Image might have not been well sampled.')
    '''

    # fitting range
    if rfit:
        where_fit = np.where(rr <= rfit)
        data = data[where_fit]
        xx   = xx[where_fit]
        yy   = yy[where_fit]


    # error
    if sigma is not None:
        absolute_sigma = True
        #sigma_fit = sigma[0:ny:step, 0:nx:step] if step >= 1 else sigma
        if rfit: sigma = sigma[where_fit]

        # exclude nan
        where_nan = np.isnan(data) | np.isnan(sigma)
        xx   = xx[~where_nan].ravel()
        yy   = yy[~where_nan].ravel()
        data = data[~where_nan].ravel()
        sigma = sigma[~where_nan].ravel()
        sigma *= np.sqrt(R_beam2pix) # to correct oversampling
    else:
        absolute_sigma = False
        # exclude nan
        where_nan = np.isnan(data)
        xx   = xx[~where_nan].ravel()
        yy   = yy[~where_nan].ravel()
        data = data[~where_nan].ravel()
        print('CAUTION\tlnfit2d: No error map was provided.')
        print('CAUTION\tlnfit2d: Parameter erros may not be correct.')


    xdata = np.vstack([xx, yy])

    # fitting
    popt, pcov = scipy.optimize.curve_fit(_lnfunc_G93, xdata, data, p0,
        sigma = sigma, absolute_sigma = absolute_sigma)
    #res = scipy.optimize.leastsq(chi_lnfunc_G93, p0, 
    #    args = (xdata, data, sigma),
    #    full_output = True)
    #popt, pcov = res[0], res[1]
    perr       = np.sqrt(np.diag(pcov))
    v0, a, b   = popt
    v0_err, a_err, b_err = perr

    # velocity gradient
    vgrad    = (a*a + b*b)**0.5/dist/auTOpc # km s^-1 pc^-1
    th_vgrad = np.arctan2(a,b)              # radians

    # error of vgrad through the error propagation
    c01       = (a*a + b*b)**(-0.5)/dist/auTOpc
    vgrad_err = c01*np.sqrt((a*a_err)*(a*a_err) + (b*b_err)*(b*b_err))

    # error of th_vgrad through the error propagation
    costh2 = np.cos(th_vgrad)*np.cos(th_vgrad)
    sinth2 = np.sin(th_vgrad)*np.sin(th_vgrad)
    th_vgrad_err = np.sqrt(
        (costh2*a_err/b)*(costh2*a_err/b)
        + (sinth2*b_err/a)*(sinth2*b_err/a))

    # unit
    th_vgrad     = th_vgrad*180./np.pi     # rad --> degree
    th_vgrad_err = th_vgrad_err*180./np.pi # rad --> degree

    # output results
    print ('(v0,a,b)=(%.2e,%.2e,%.2e)'%(popt[0],popt[1],popt[2]))
    print ('(sig_v0,sig_a,sig_b)=(%.2e,%.2e,%.2e)'%(perr[0],perr[1],perr[2]))
    print ('Vgrad: %.2f +/- %.2f km/s/pc'%(vgrad,vgrad_err))
    print ('P.A.: %.1f +/- %.1f deg'%(th_vgrad,th_vgrad_err))


    # output image for check
    #outname=None, 
    '''outfig=True
    if outfig:
        xx = image.xx.copy() * 3600. # deg --> arcsec
        yy = image.yy.copy() * 3600. # deg --> arcsec
        vlsr   = lnfunc_G93(xx,yy,*popt)
        xmin   = xx[0,0]
        xmax   = xx[-1,-1]
        ymin   = yy[0,0]
        ymax   = yy[-1,-1]
        extent = (xmin,xmax,ymin,ymax)
        #print (vlsr)

        fig = plt.figure(figsize=(11.69,8.27))
        ax  = fig.add_subplot(111)
        im = ax.imshow(vlsr, origin='lower',cmap='jet',extent=extent, vmin=6.6, vmax=7.4)

        # direction of the velocity gradient
        costh = np.cos(th_vgrad*np.pi/180.)
        sinth = np.sin(th_vgrad*np.pi/180.)
        mrot = np.array([[costh, -sinth],
            [sinth, costh]])
        p01 = np.array([0,xmax])
        p02 = np.array([0,-xmax])
        p01_rot = np.dot(p01,mrot)
        p02_rot = np.dot(p02,mrot)
        ax.plot([p01_rot[0],p02_rot[0]],[p01_rot[1],p02_rot[1]],ls='--',lw=1, c='k',sketch_params=0.5)

        # colorbar
        #divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        #cax     = divider.append_axes('right','3%', pad='0%')
        cbar    = fig.colorbar(im, ax = ax)
        cbar.set_label(r'$\mathrm{km\ s^{-1}}$')

        # labels
        ax.set_xlabel('RA offset (arcsec)')
        ax.set_ylabel('DEC offset (arcsec)')


        #if outname:
        #    pass
        #else:
        #    outname = 'measure_vgrad_res'
        #plt.savefig(outname+'.pdf',transparent=True)
        plt.show()

        return ax, popt, perr
    '''

    if full_output:
        # model
        xx = image.xx.copy() * 3600. # deg --> arcsec
        yy = image.yy.copy() * 3600. # deg --> arcsec
        model = lnfunc_G93(xx,yy,*popt)
        # residual
        residual = _data - model
        return {'popt': popt, 'perr': perr, 
        'v0': v0, 'v0_err': v0_err,
        'vgrad': vgrad, 'vgrad_err': vgrad_err,
        'th_vgrad': th_vgrad, 'th_vgrad_err': th_vgrad_err,
        'model': model, 'residual': residual}
    else:
        return v0, v0_err, vgrad, vgrad_err, th_vgrad, th_vgrad_err



def gaussian2D(x, y, A, mx, my, sigx, sigy, pa=0, peak=True):
    '''
    Generate normalized 2D Gaussian

    Parameters
    ----------
     x: x value (coordinate)
     y: y value
     A: Amplitude. Not a peak value, but the integrated value.
     mx, my: mean values
     sigx, sigy: standard deviations
     pa: position angle [deg]. Counterclockwise is positive.
    '''
    x, y   = rotate2d(x,y,pa)
    mx, my = rotate2d(mx, my, pa)

    coeff = A if peak else A/(2.0*np.pi*sigx*sigy)
    expx  = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    expy  = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
    gauss = coeff*expx*expy
    return gauss


def rotate2d(x, y, angle, deg=True, coords=False):
    '''
    Rotate Cartesian coordinates.
    Right hand direction will be positive.

    array2d: input array
    angle: rotational angle [deg or radian]
    axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
    deg (bool): If True, angle will be treated as in degree. If False, as in radian.
    '''

    # degree --> radian
    if deg:
        angle = np.radians(angle)

    if coords:
        angle = -angle

    cos = np.cos(angle)
    sin = np.sin(angle)

    xrot = x*cos - y*sin
    yrot = x*sin + y*cos

    return xrot, yrot


def regrid_image(self, 
    xlim: list = [],
    ylim: list = [],
    vlim: list = [],
    fits_tmp = None, check_fig = False, 
    center_mode='onpixel'):
    '''

    center_mode (str): Location of the coordinate center. onpixel or between.
    '''
    # regrid fits image using spline trapolation.
    from scipy.interpolate import griddata

    xlim = np.array(xlim)/3600.
    ylim = np.array(ylim)/3600.

    # Get coordinates
    if fits_tmp:
        im_tmp = Imfits(fits_tmp)
        delx = im_tmp.delx
        dely = im_tmp.dely
        delv = im_tmp.delv
    else:
        delx = self.delx
        dely = self.dely
        delv = self.delv

    if center_mode == 'onpixel':
        xc = np.concatenate([np.arange(0., xlim[1]-delx, -delx)[::-1], 
            np.arange(delx, xlim[0]+delx, delx)])
        yc = np.concatenate([np.arange(-dely, ylim[0]-dely, -dely)[::-1],
            np.arange(0., ylim[1]+dely, dely)])
    elif center_mode == 'between':
        xc = np.concatenate([np.arange(-delx*0.5, xlim[1]-delx, -delx)[::-1], 
            np.arange(delx*0.5, xlim[0]+delx, delx)])
        yc = np.concatenate([np.arange(-dely*0.5, ylim[0]-dely, -dely)[::-1],
            np.arange(dely*0.5, ylim[1]+dely, dely)])
    else:
        print('ERROR\tregrid_image: center_mode must be onpixel or between.')
        return 0

    # grid
    #vc             = np.linspace(vmin,vmax,nchan)
    xx_new, yy_new = np.meshgrid(xc, yc)

    # grid
    xc2     = np.linspace(xmin2,xmax2,nx2)
    yc2     = np.linspace(ymin2,ymax2,ny2)
    vc2     = np.linspace(vmin2,vmax2,nchan2)
    xx_reg, yy_reg = np.meshgrid(xc2, yc2)
    #xx_reg, yy_reg, vv_reg = np.meshgrid(xc2, yc2, vc2)

    # 2D --> 1D
    xinp    = self.xx.reshape(self.xx.size)
    yinp    = self.yy.reshape(self.yy.size)

    # regrid
    print ('regriding...')
    if self.naxis == 2:
        data_reg = griddata((xinp, yinp), self.data.reshape(self.data.size), (xx_new, yy_new), 
            method='cubic',rescale=True)
    elif self.naxis == 3:
        channel_range = tqdm(range(self.nv))
        data_reg = np.array([ griddata((xinp, yinp), data[i,:,:].reshape(dataxysize), (xx_tmp, yy_tmp), 
            method='cubic',rescale=True) for i in channel_range ])
    elif self.naxis == 4:
        channel_range = tqdm(range(self.nv))
        data_reg = np.array([[ griddata((xinp, yinp), data[j,i,:,:].reshape(dataxysize), (xx_tmp, yy_tmp), 
            method='cubic',rescale=True) for i in channel_range ] for j in range(self.ns)])

    return xc, yc, data_reg



### functions
def estimate_noise(_d, nitr=1000, thr=2.3):
    '''
    Estimate map noise

    _d (array): Data
    nitr (int): Number of the maximum iteration
    thr (float): Threshold of the each iteration
    '''

    d = _d.copy()
    rms = np.sqrt(np.nanmean(d*d))
    for i in range(nitr):
        rms_p = rms
        d[d >= thr*rms] = np.nan
        rms = np.sqrt(np.nanmean(d*d))

        if (rms - rms_p)*(rms - rms_p) < 1e-20:
            return rms

    print('Reach maximum number of iteration.')
    return rms


def get_1Dresolution(bmaj, bmin, bpa, pa):
    # an ellipse of the beam
    # (x/bmin)**2 + (y/bmaj)**2 = 1
    # y = x*tan(theta)
    # --> solve to get resolution in the direction of pv cut with P.A.=pa
    del_pa = pa - bpa
    del_pa = del_pa*np.pi/180. # radian
    term_sin = (np.sin(del_pa)/bmin)**2.
    term_cos = (np.cos(del_pa)/bmaj)**2.
    res_off  = np.sqrt(1./(term_sin + term_cos))
    return res_off


def dropaxes(data, istokes=0, ivelocity=0):
    if len(data.shape) == 2:
        return data
    elif len(data.shape) == 3:
        return data[ivelocity,:,:]
    elif len(data.shape) == 4:
        return data[istokes, ivelocity, :, :]
    else:
        print('ERROR\tdropaxes: Data must be 2 to 4 dimensions.')
        return 0


def generate_noisemap(rms, 
    image = None, xx = None, yy = None,
    beam = None,
    ):
    '''
    Generate noise map.
    Currently, only an imfits image input and Jy/beam unit is supported.

    rms (float): rms of the noise map (Jy/beam)
    image (imfits object): input image.
    '''

    shape = image.data.shape
    ndim = len(shape)
    beam = image.beam
    delx, dely = image.delx, image.dely
    xx, yy = image.xx, image.yy
    ny, nx = xx.shape
    xicent = nx // 2 - 1 + nx%2
    yicent = ny // 2 - 1 +ny%2


    if beam is not None:
        from scipy.signal import convolve
        bmaj, bmin, bpa = beam
        f_beamtojpp = IbeamTOjpp(1., bmaj/3600., bmin/3600., delx, dely)

        rms = rms * f_beamtojpp # Jy/beam --> Jy/pixel

        # scaling to add noise before convolution
        s_ang = np.pi / (4.*np.log(2.)) * bmaj * bmin / 3600. / 3600. # solid angle (deg^2)
        ratio = s_ang / np.abs(delx*dely) # pixels
        rms = rms * np.sqrt(ratio * 2.) # 2 is a scale factor for fine tuning
        #print (ratio)

        noise = np.random.normal(loc=0., scale=rms, size=(shape))

        sigx = 0.5*bmin/np.sqrt(2.*np.log(2.))/3600.            # in degree
        sigy = 0.5*bmaj/np.sqrt(2.*np.log(2.))/3600.            # in degree
        area = 1.
        gaussbeam = gaussian2D(xx, yy, 
            area, xx[yicent, xicent], yy[yicent, xicent], sigx, sigy, pa = bpa)
        gaussbeam /= np.sum(gaussbeam)

        if ndim == 2:
            pass
        elif ndim == 3:
            gaussbeam = np.array([gaussbeam])
        elif ndim == 4:
            gaussbeam = np.array([[gaussbeam]])

        noise = convolve(noise, gaussbeam, mode='same')
        noise /= f_beamtojpp # Jy/pixel to Jy/beam
        #print(np.std(noise))
    else:
        noise = np.random.normal(loc=0., scale=rms, size=(shape))

    return noise


def bin_data(image, nbin, axes=[0]):
    if nbin%2 == 0:
        xx, yy = self.shift()
    else:
        xx, yy = self.xx.copy(), self.yy.copy()

    xcut = self.nx%nbin
    ycut = self.ny%nbin
    _xx = xx[ycut//2:-ycut//2, xcut//2:-xcut//2]
    _yy = yy[ycut//2:-ycut//2, xcut//2:-xcut//2]
    xx_avg = np.array([
        _xx[i::nbin, i::nbin]
        for i in range(nbin)
        ])
    yy_avg = np.array([
        _yy[i::nbin, i::nbin]
        for i in range(nbin)
        ])

    return np.average(xx_avg, axis= 0), np.average(yy_avg, axis= 0)


def binning_1d(axis, data, nbin,
    axis_index = 0):
    # length
    nl = len(axis)
    icut = nl%nbin
    # shape
    shape = data.shape
    ndim = len(shape)

    _axis = axis[icut//2:-icut//2]
    if ndim == 1:
        _data = data[icut//2:-icut//2]
    elif ndim == 2:
        _data = data[icut//2:-icut//2, :] if axis_index == 0 \
        else data[:, icut//2:-icut//2]
    elif ndim == 3:
        if axis_index == 0:
            _data = data[icut//2:-icut//2, :, :]
        elif axis_index == 0:
            _data = data[icut//2:-icut//2, :, :]
    data_binned = np.array([
        data[i::nbin]
        for i in range(nbin)
        ])

    return data_binned



def _circular_slice(data, 
    npa = 361., istokes=0, ifreq=0,
    pa_min = -180., pa_max = 180.):
    '''
    Produce figure where position angle (x) vs radius (y) vs intensity (color)

    Parameters
    ----------
    data (2D array): 2D image array
    delt_pa (float): sampling step along pa axis (deg)
    '''

    pa = np.linspace(pa_min, pa_max, npa)

    # check data axes
    if len(data.shape) == 2:
        pass
    elif len(data.shape) == 3:
        data = data[istokes,:,:]
    elif len(data.shape) == 4:
        data = data[istokes,ifreq,:,:]
    else:
        print ('Error\tciruclar_slice: Input fits size is not corrected.\
            It is allowed only to have 2 to 4 axes. Check the shape of the fits file.')
        return

    # set parameters
    ny, nx = data.shape
    y0, x0 = ny//2, nx//2
    nz  = ny - y0
    r = np.arange(0, nz, 1)

    # start slicing
    print ('circular slice...')
    cslice = np.zeros([nz,npa])
    for ipa in range(npa):
        rotimage      = imrotate_2d(data, - pa[ipa])
        cslice[:,ipa] = rotimage[y0:,x0]

    return r, pa, cslice


def circular_slice(image, 
    npa = 361., istokes=0, ifreq=0):
    '''
    Produce figure where position angle (x) vs radius (y) vs intensity (color)

    Parameters
    ----------
    image (Imfits object): Imfits object.
    delt_pa (float): sampling step along pa axis (deg)
    '''

    pa = np.linspace(-180., 180., npa)

    # check data axes
    if len(image.data.shape) == 2:
        data = image.data.copy()
    if len(image.data.shape) == 3:
        data = image.data[istokes,:,:]
    elif len(data.shape) == 4:
        data = image.data[istokes,ifreq,:,:]
    else:
        print ('Error\tciruclar_slice: Input fits size is not corrected.\
            It is allowed only to have 2 to 4 axes. Check the shape of the fits file.')
        return

    # set parameters
    ny, nx = data.shape
    y0, x0 = ny//2, nx//2
    nz  = ny - y0

    # r axis
    r = image.yaxis.copy() * 3600. # in arcsec
    r = r[y0:]

    # start slicing
    print ('circular slice...')
    cslice = np.zeros([nz,npa])
    for ipa in range(npa):
        rotimage      = imrotate_2d(data, - pa[ipa])
        cslice[:,ipa] = rotimage[y0:,x0]

    return pa, r, cslice