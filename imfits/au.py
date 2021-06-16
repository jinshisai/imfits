'''
Analysis utilities for Imfits.
'''


import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from imfits import Imfits



# constant
clight = 2.99792458e10  # light speed [cm s^-1]
auTOkm = 1.495978707e8  # AU --> km
auTOcm = 1.495978707e13 # AU --> cm
auTOpc = 4.85e-6        # au --> pc
pcTOau = 2.06e5         # pc --> au
pcTOcm = 3.09e18        # pc --> cm



# functions
# 2D linear function
def func_G93(del_ra, del_dec, v0, a, b):
	# See Goodman et al. (1993)
	vlsr = v0 + a*del_ra + b*del_dec
	return vlsr

def _func_G93(xdata,*args):
	del_ra, del_dec = xdata
	#print (*args)
	ans = func_G93(del_ra, del_dec, *args)
	return ans


def lnfit2d(cube, p0, rfit=None, dist=140.,
 outname=None, outfig=True, axis=0):
	'''
	Fit a two-dimensional linear function to a map by the least square fitting.
	Basically, the input map is assumed to be a velocity map and
	 the measurement result will be treated as a velocity gradient.
	The fitting method is based on the one done by Goodman et al. (1993).

	cube: fits file
	'''

	# check type
	if type(cube) == Imfits:
		pass
	else:
		print ('ERROR\tlnfit2d: Object type is not Imfits. Check the input.')
		return

	data  = cube.data
	xx    = cube.xx
	yy    = cube.yy
	naxis = cube.naxis

	xx = xx*3600. # deg --> arcsec
	yy = yy*3600.

	# delta
	dx = np.abs(xx[0,1] - xx[0,0])
	dy = np.abs(yy[1,0] - yy[0,0])
	#print (xx.shape)

	# beam
	bmaj, bmin, bpa = cube.beam # as, as, deg

	# radius
	rr = np.sqrt(xx*xx + yy*yy)


	# check data axes
	if naxis == 2:
		pass
	elif naxis == 3:
		data = data[axis,:,:]
	elif naxis == 4:
		data = data[0,axis,:,:]
	else:
		print ('Error\tmeasure_vgrad: Input fits size is not corrected.\
			It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
		return


	# Nyquist sampling
	step = int(bmin/dx*0.5)
	ny, nx = xx.shape
	#print (step)
	xx_fit = xx[0:ny:step, 0:nx:step]
	yy_fit = yy[0:ny:step, 0:nx:step]
	data_fit = data[0:ny:step, 0:nx:step]
	#print (data_fit.shape)


	if rfit:
		where_fit = np.where(rr <= rfit)
		data_fit = data[where_fit]
		xx_fit   = xx[where_fit]
		yy_fit   = yy[where_fit]
	else:
		data_fit = data
		xx_fit   = xx
		yy_fit   = yy


	# exclude nan
	xx_fit   = xx_fit[~np.isnan(data_fit)]
	yy_fit   = yy_fit[~np.isnan(data_fit)]
	data_fit = data_fit[~np.isnan(data_fit)]
	#print (xx_fit)


	# Ravel the meshgrids of X, Y points to a pair of 1-D arrays.
	xdata = np.vstack((xx_fit, yy_fit)) # or xx.ravel()

	# fitting
	popt, pcov = scipy.optimize.curve_fit(_func_G93, xdata, data_fit, p0)
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
	'''
	if outfig:
		vlsr   = func_G93(xx,yy,*popt)
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
		p01 = np.array([0,60])
		p02 = np.array([0,-60])
		p01_rot = np.dot(p01,mrot)
		p02_rot = np.dot(p02,mrot)
		ax.plot([p01_rot[0],p02_rot[0]],[p01_rot[1],p02_rot[1]],ls='--',lw=1, c='k',sketch_params=0.5)

		# colorbar
		divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
		cax     = divider.append_axes('right','3%', pad='0%')
		cbar    = fig.colorbar(im, cax = cax)
		cbar.set_label(r'$\mathrm{km\ s^{-1}}$')

		# labels
		ax.set_xlabel('RA offset (arcsec)')
		ax.set_ylabel('DEC offset (arcsec)')


		if outname:
			pass
		else:
			outname = 'measure_vgrad_res'
		plt.savefig(outname+'.pdf',transparent=True)
		#plt.show()

		return ax, popt, perr
	'''

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