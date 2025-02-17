'''
Fitting functions
'''

# modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, uniform




# functions
# gaussian
def gauss1d(x,amp,mean,sig):
    return amp * np.exp(-(x-mean)*(x-mean)/(2.0*sig*sig))

# for chi-square
def chi_gauss1d(param, xdata, ydata, ysig):
    return (ydata - (gauss1d(xdata, *param))) / ysig

def gaussfit(xdata, ydata, yerr):
    '''
    Gaussian fit through chi-square fit.
    '''

    # Get estimate of the initial parameters
    indx_pini = ydata >= 3.*yerr
    mx    = np.nansum(ydata[indx_pini]*xdata[indx_pini])/np.nansum(ydata[indx_pini]) # weighted mean
    sigx  = np.sqrt(np.nansum(ydata[indx_pini]*(xdata[indx_pini] - mx)**2.)/np.nansum(ydata[indx_pini])) # standerd deviation

    # if sigx is too small
    if sigx <= 1e-6:
        sigx = np.abs(xdata[1] - xdata[0])/len(xdata)

    amp  = np.nanmax(ydata)
    pinp = [amp, mx, sigx]

    if len(xdata) < 3:
        param_out = np.full(3, np.nan)
        param_err = np.full(3, np.nan)
        return param_out, param_err

    # fitting
    results   = optimize.leastsq(chi_gauss1d, pinp, args=(xdata, ydata, yerr), full_output=True)
    param_out = results[0]
    param_cov = results[1]
    #print(results)
    #print(param_out, param_cov)
    # Do not multiply covariance by reduced chi^2 to obtain absolute errors

    # parameter error estimates
    if param_cov is not None:
        param_err = np.array([
            np.abs(param_cov[j][j])**0.5 for j in range(len(pinp))
            ])
    else:
        param_err = np.full(3, np.nan)
        param_out = np.full(3, np.nan) if (param_out == pinp).all else param_out

    # print results
    #print ('Chi2: ', reduced_chi2)
    #print ('Fitting results: amp   mean   sigma')
    #print (amp_fit, mx_fit, sigx_fit)
    #print ('Errors')
    #print (amp_fit_err, mx_fit_err, sigx_fit_err)

    return param_out, param_err


# functions
def lnprof_gauss(v, amp, v0, delv):
    '''
    Generate a Gaussian lineprofile. Note that the definition of delv is the line width
    of the Doppler broadening.

    Parameters
    ----------
     v: velocity
     amp: Amplitude. Not a peak value, but the integrated value.
     v0: mean velocity
     delv: line width
    '''
    return amp * np.exp(- (v - v0)**2. / delv**2.)


# functions
def lnprof_gauss_tau(v, delT, tau, v0, delv, Tbg = 0.):
    '''
    Generate a Gaussian lineprofile. Note that the definition of delv is the line width
    of the Doppler broadening.

    Parameters
    ----------
     v: velocity
     amp: Amplitude. Not a peak value, but the integrated value.
     v0: mean velocity
     delv: line width
    '''
    return (delT - Tbg) * (1. - np.exp( - tau * np.exp(- (v - v0)**2. / delv**2.)))


def chi_lnprof_gauss(param, v, d, derr, include_tau = False):
    if include_tau:
        chi = (d - lnprof_gauss_tau(v, *param)) / derr
    else:
        chi = (d - lnprof_gauss(v, *param)) / derr
    return chi


def fit_lnprof(x, y, yerr, 
    pixrng=0, include_tau = False, p0 = None):
    '''
    Perform fitting to determine the peak position.
    '''

    # determining used data range
    if pixrng:
        # error check
        if type(pixrng) != int:
            print ('Error\tpvfit_vcut: pixrng must be integer.')
            return

        # use pixels only around intensity peak
        peakindex = np.argmax(y)
        fit_indx  = [peakindex - pixrng, peakindex + pixrng+1]
        x_fit = x[fit_indx[0]:fit_indx[1]]
        y_fit = y[fit_indx[0]:fit_indx[1]]
    else:
        x_fit = x
        y_fit = y

    # set the initial parameter
    if p0 is None:
        mx   = x_fit[np.nanargmax(y_fit)]
        amp  = np.nanmax(y_fit)
        sigx = (x[1] - x[0]) * 3. #np.sqrt(np.nansum((mx - x_fit)**2. * y_fit) / np.nansum(y_fit))
        if include_tau:
            pinp = [amp, 1., mx, sigx]
        else:
            pinp = [amp, mx, sigx]
        #print('pinp', pinp)

    f_chi = lambda p, x, y, yerr,: chi_lnprof_gauss(p, x, y, yerr, include_tau = include_tau)

    # fitting
    ndata   = len(y_fit)
    results = optimize.leastsq(
        f_chi, 
        pinp, 
        args=(x_fit, y_fit, yerr), 
        full_output=True)

    # recording output results
    # param_out: fitted parameters
    # err: error of fitted parameters
    # chi2: chi-square
    # DOF: degree of freedum
    # reduced_chi2: reduced chi-square
    param_out = results[0]
    if param_out[2] < 0:
        param_out[2] *= -1
    param_cov    = results[1]
    chi2         = np.sum(f_chi(param_out, x_fit, y_fit, yerr)**2.)
    nparam       = len(param_out)
    dof          = ndata - nparam - 1
    reduced_chi2 = chi2 / dof

    #print (ndata, nparam, dof)
    #print (param_cov)
    if (dof >= 0) and (param_cov is not None):
        pass
        #param_cov = param_cov #*reduced_chi2
    else:
        param_cov = np.full((nparam, nparam),np.inf)

    # best fit value
    #amp_fit, mx_fit, sigx_fit = param_out

    # fitting error
    param_err = np.array([
        np.abs(param_cov[j][j])**0.5 for j in range(nparam)
        ])
    #amp_fit_err, mx_fit_err, sigx_fit_err = param_err

    # print results
    #print ('Chi2: ', reduced_chi2)
    #print ('Fitting results: amp   mean   sigma')
    #print (amp_fit, mx_fit, sigx_fit)
    #print ('Errors')
    #print (amp_fit_err, mx_fit_err, sigx_fit_err)

    return param_out, param_err


def gaussian2d(x, y, A, mx, my, sigx, sigy, pa=0, peak=True):
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


def gaussian2d_dev(x, y, A, mx, my, sigx, sigy, pa=0, peak=True):
    '''
    Generate normalized 2D Gaussian.

    Parameters
    ----------
     x (float or array): x value (coordinate)
     y (float or array): y value
     A (float): Amplitude. Not a peak value, but the integrated value.
     mx, my (float): mean values
     sigx, sigy (float): standard deviations
     pa (float): position angle (deg). Measured from y positive to x positive.
                 In the plane of the sky, from north to east.
    '''
    coeff = A if peak else A/(2.0*np.pi*sigx*sigy)
    rho = np.tan(np.radians(90. - pa))
    cov = rho * sigx * sigy
    _exp = - 1. / (sigx**2. * sigy**2. - cov**2.) / 2.\
    * ((x - mx) * (x - mx) * sigy**2. - 2. * cov * (x - mx) * (y-my)\
        + (y - my) * (y - my) * sigx**2.)
    return coeff * np.exp(_exp)


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


def gaussfit2d(xx, yy, d, d_err, pini=[], peak=True):
    '''
    Gaussian fit through chi-square fit.
    '''
    # Data check
    if (len(pini) == 0) & (~(d >= 3.*d_err).any()):
        print('No data >3sigma. Give pini.')
        return 0

    if xx.size < 6:
        print('Not enough data points.')
        return 0

    # Get estimate of the initial parameters
    indx_max = np.unravel_index(np.nanargmax(d), d.shape)
    amp  = d[indx_max]
    mx, my = xx[indx_max], yy[indx_max]
    indx_sig = d >= 3.*d_err
    sigx = np.sqrt(np.nansum(d[indx_sig] * (xx[indx_sig] - mx)**2.)/np.nansum(d[indx_sig]))
    sigy = np.sqrt(np.nansum(d[indx_sig] * (yy[indx_sig] - my)**2.)/np.nansum(d[indx_sig]))

    pini = [amp, mx, my, sigx, sigy, 0., ]

    # fitting
    gauss2d = lambda xx, yy, *params: gaussian2d(xx, yy, *params, peak=peak)
    chi_gauss2d = lambda params, *args: chi2d(params, *args, gauss2d)
    results   = optimize.leastsq(chi_gauss2d, pini, args=(xx, yy, d, d_err), full_output=True)
    param_out = results[0]
    param_cov = results[1]
    #print(results)
    #print(param_out, param_cov)
    # Do not multiply covariance by reduced chi^2 to obtain absolute errors

    # parameter error estimates
    if param_cov is not None:
        param_err = np.array([
            np.abs(param_cov[j][j])**0.5 for j in range(len(pini))
            ])
    else:
        param_err = np.full(6, np.nan)
        param_out = np.full(6, np.nan) if (param_out == pini).all else param_out

    # print results
    #print ('Chi2: ', reduced_chi2)
    #print ('Fitting results: amp   mean   sigma')
    #print (amp_fit, mx_fit, sigx_fit)
    #print ('Errors')
    #print (amp_fit_err, mx_fit_err, sigx_fit_err)

    return param_out, param_err


def plaw(x, a, p, x0=1.):
    return a*(x/x0)**p

def chi(params, x, y, sig, fmodel):
    return (y - fmodel(x, *params)) / sig

def chi2d(params, xx, yy, d, d_err, fmodel):
    return ((d - fmodel(xx, yy, *params)) / d_err).ravel()

def plawfit(x, y, sig, x0=1., pini=[]):
    if len(pini) == 0:
        p0 = (np.log(y[-1]) - np.log(y[0]))/(np.log(x[-1]/x0) - np.log(x[0]/x0))
        a0 = np.nanmean(y)/(np.nanmean(x)/x0)**p0
        pini = [a0, p0]

    plaw_fit = lambda x, *params: plaw(x, *params, x0=x0)
    chi_plaw = lambda params, *args: chi(params, *args, plaw_fit)
    results   = optimize.leastsq(chi_plaw, pini, args=(x, y, sig), full_output=True)
    param_out = results[0]
    param_cov = results[1]

    if param_cov is not None:
        param_err = np.array([
            np.abs(param_cov[j][j])**0.5 for j in range(len(pini))
            ])
    else:
        param_err = np.full(3, np.nan)
        param_out = np.full(3, np.nan) if (param_out == pini).all else param_out

    return param_out, param_err



def estimate_perror(params, func, x, y, xerr, yerr, niter=3000):
    '''
    Estimate fitting-parameter errors by Monte-Carlo method.

    '''
    nparams = len(params)
    perrors = np.zeros((0,nparams), float)

    for i in range(niter):
        offest = norm.rvs(size = len(x), loc = x, scale = xerr)
        velest = norm.rvs(size = len(y), loc = y, scale = yerr)
        result = optimize.leastsq(func, params, args=(offest, velest, xerr, yerr), full_output = True)
        perrors = np.vstack((perrors, result[0]))
        #print param_esterr[:,0]

    sigmas = np.array([
        np.std(perrors[:,i]) for i in range(nparams)
        ])
    medians = np.array([
        np.median(perrors[:,i]) for i in range(nparams)
        ])


    with np.printoptions(precision=4, suppress=True):
        print ('Estimated errors (standard deviation):')
        print (sigmas)
        print ('Medians:')
        print (medians)


    # plot the Monte-Carlo results
    fig_errest = plt.figure(figsize=(11.69,8.27), frameon = False)
    gs         = GridSpec(nparams, 2)
    if nparams == 3:
        xlabels    = [r'$V_\mathrm{sys}$', r'$V_\mathrm{100}$', r'$p$' ]
    elif nparams == 4:
        xlabels    = [r'$V_\mathrm{break}$', r'$R_\mathrm{break}$', r'$p_\mathrm{in}$', r'$p_\mathrm{out}$']
    else:
        xlabels = [r'$p%i$'%(i+1) for i in range(nparams)]

    for i in range(nparams):
        # histogram
        ax1 = fig_errest.add_subplot(gs[i,0])
        ax1.set_xlabel(xlabels[i])
        if i == (nparams - 1):
            ax1.set_ylabel('Frequency')
        ax1.hist(perrors[:,i], bins = 50, cumulative = False) # density = True

        # cumulative histogram
        ax2 = fig_errest.add_subplot(gs[i,1])
        ax2.set_xlabel(xlabels[i])
        if i == (nparams - 1):
            ax2.set_ylabel('Cumulative\n frequency')
        ax2.hist(perrors[:,i], bins = 50, density=True, cumulative = True)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    #plt.show()
    fig_errest.savefig(outname + '_errest.pdf', transparent=True)
    fig_errest.clf()

    return sigmas