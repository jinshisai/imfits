### modules
import numpy as np
import matplotlib.pyplot as plt



### functions
def estimate_noise(_d, nitr=1000, thr=2.3):
    '''
    Estimate map noise
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



# 2D Gaussian
def gaussian1D(x, A, mx, sigx, norm=False):
    '''
    Generate normalized 2D Gaussian

    x: x value (coordinate)
    y: y value
    A: Amplitude. Not a peak value, but the integrated value.
    mx, my: mean values
    sigx, sigy: standard deviations
    pa: position angle [deg]. Counterclockwise is positive.
    x, y = rotate2d(x,y,pa)
    '''

    coeff =  A/(np.sqrt(2.0*np.pi*sigx*sigx)) if norm else A
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    return coeff*expx


def gaussian2D(x, y, A, mx, my, sigx, sigy, pa=0, norm=False):
    '''
    Generate normalized 2D Gaussian

    x: x value (coordinate)
    y: y value
    A: Amplitude. Not a peak value, but the integrated value.
    mx, my: mean values
    sigx, sigy: standard deviations
    pa: position angle [deg]. Counterclockwise is positive.
    x, y = rotate2d(x,y,pa)
    '''

    coeff =  A/(2.0*np.pi*sigx*sigy) if norm else A
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    expy = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
    return coeff*expx*expy


### debug
def main():
    # -------- input -----------
    ngrid = 1000
    xe = np.linspace(-1,1,ngrid+1)
    ye = np.linspace(-1,1,ngrid+1)
    xc = (xe[0:ngrid] + xe[1:ngrid+1])*0.5
    yc = (ye[0:ngrid] + ye[1:ngrid+1])*0.5
    xx, yy = np.meshgrid(xc, yc)
    gauss = gaussian2D(xx, yy, 1, 0, 0, 0.2, 0.2, norm=False)
    noise = np.random.normal(0, 0.1, gauss.shape)
    data  = gauss + noise

    gauss = gaussian1D(xc, 1, 0, 0.1, norm=False)
    noise = np.random.normal(0, 0.1, gauss.shape)
    data  = gauss + noise

    rms_true = np.sqrt(np.nanmean(noise*noise)) 
    # --------------------------


    plt.pcolor(xx, yy, data, shading='auto')
    plt.plot(xc, data, )
    plt.show()

if __name__ == '__main__':
    main()