### modules
import numpy as np
import sys
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.gridspec import GridSpec
from astropy.io import fits
import matplotlib.patches as patches
import mpl_toolkits.axes_grid1
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.interpolate import griddata
import imhandle
from datetime import datetime



### setting for figures
#mpl.use('Agg')
#mpl.rcParams['agg.path.chunksize'] = 100000
plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
#plt.rcParams['xtick.major.width'] = 1.0 # x ticks width
#plt.rcParams['ytick.major.width'] = 1.0 # y ticks width
plt.rcParams['font.size'] = 11           # fontsize
#plt.rcParams['axes.linewidth'] = 1.0    # edge linewidth




### parameters
formatlist = np.array(['eps','pdf','png','jpeg'])
clight     = 2.99792458e10 # light speed [cm s^-1]



### functions
def fits_deprojection(self,inc,pa,outname=None,outformat='pdf',fitsout=True, figout=False,
	color=True, cmap='jet', relativecoords=True, imscale=np.empty([]),  locsym = 0.1):
	'''
	Modify inclination and P.A. of a disk to deproject.

	self: input fits file. The shape now can be 4D only.
	inc: inclination of the object [deg]
	pa: position angle of the object [deg]
	fitsout: If True, new fits file will be produced after correcting the inclination.
	'''
	### setting output file name & format
	if (outformat == formatlist).any():
		outfitsname = outname + '.fits'
		outfigname  = outname + '.' + outformat
	else:
	    print 'ERROR\tsingleim_to_fig: Outformat is wrong.'
	    return


	### reading fits files
	data, header = fits.getdata(self,header=True)

	# reading header info.
	xlabel   = header['CTYPE1']
	ylabel   = header['CTYPE2']
	try:
	    restfreq = header['RESTFRQ'] # Hz
	except:
	    restfreq = header['RESTFREQ'] # Hz
	refval_x = header['CRVAL1']*60.*60. # deg --> arcsec
	refval_y = header['CRVAL2']*60.*60.
	refval_v = header['CRVAL3']
	refpix_x = int(header['CRPIX1'])
	refpix_y = int(header['CRPIX2'])
	refpix_v = int(header['CRPIX3'])
	del_x    = header['CDELT1']*60.*60. # deg --> arcsec
	del_y    = header['CDELT2']*60.*60.
	del_v    = header['CDELT3']
	nx       = header['NAXIS1']
	ny       = header['NAXIS2']
	nchan    = header['NAXIS3']
	bmaj     = header['BMAJ']*60.*60.
	bmin     = header['BMIN']*60.*60.
	bpa      = header['BPA']  # [deg]
	unit     = header['BUNIT']
	print 'x, y axes are ', xlabel, ' and ', ylabel

	# setting axes in relative coordinate
	if relativecoords:
	    refval_x, refval_y = [0,0]
	    xlabel = 'RA offset (arcsec; J2000)'
	    ylabel = 'Dec offset (arcsec; J2000)'
	else:
	    pass
	xmin = refval_x + (0 - refpix_x + 1)*del_x
	xmax = refval_x + (nx - refpix_x +1)*del_x
	ymin = refval_y + (0 - refpix_y +1)*del_y
	ymax = refval_y + (ny - refpix_y +1)*del_y


	# grid
	xc     = np.linspace(xmin,xmax,nx)
	yc     = np.linspace(ymin,ymax,ny)
	xx, yy = np.meshgrid(xc, yc)


	# sky-plane --> coordinate parallel to the disk plane
	print 'P.A.: ', pa, ' deg'
	print 'inculination: ', inc, ' deg'
	rotdeg  = 90. - pa
	rotrad  = np.radians(rotdeg)
	incrad  = np.radians(inc)

	# rotation of the coordinate by pa,
	#   in which xprime = xcos + ysin, and yprime = -xsin + ycos
	# now, x axis is positive in the left hand direction (x axis is inversed).
	# right hand (clockwise) rotation will be positive in angles.
	xxparot = xx*np.cos(rotrad) + yy*np.sin(rotrad)
	yyparot = - xx*np.sin(rotrad) + yy*np.cos(rotrad)
	yyinc   = yyparot/np.cos(incrad)


	# check data axes
	#redata = data.copy
	redata = np.zeros(data.shape)
	if len(data.shape) == 2:
		print 'ERROR imdeproject: Input fits shape must be (I,v,x,y).'
		return
	elif len(data.shape) == 3:
		print 'ERROR imdeproject: Input fits shape must be (I,v,x,y).'
		return
	    #data = data[0,:,:]
	elif len(data.shape) == 4:
		print 'the number of total channels: ', nchan
		for ichan in xrange(nchan):
	    	# rotate image
			print 'channel: ', ichan+1, '/', nchan

			# nan --> 0 for interpolation
			print 'CAUTION imdeproject: Input image array includes nan. Replace nan with 0 for interpolation.'
			data[np.where(np.isnan(data))] = 0.

			# 2D --> 1D
			xinp    = xxparot.reshape(xxparot.size)
			yinp    = yyinc.reshape(xxparot.size)
			dinp    = data[0,ichan,:,:].reshape(data[0,ichan,:,:].size)
			#print len(xinp), len(yinp), data[0,ichan,:,:].size

			# regrid
			data_reg = griddata((xinp, yinp), dinp, (xx, yy), method='cubic')

			# rotate back
			redata[0,ichan,:,:] = imhandle.imrotate(data_reg,angle=-rotdeg)

	else:
	    print 'Error\tsingleim_to_fig: Input fits size is not corrected.\
	     It is allowed only to have 3 or 4 axes. Check the shape of the fits file.'
	    return



	### output results
	### fits file
	if fitsout:
		# set new header
		hdout = header
		today = datetime.today()
		today = today.strftime("%Y/%m/%d %H:%M:%S")

		# write into new header
		hdout['DATE']     = (today, 'Date FITS file was written')
		hdout['INC_DISK'] = inc
		hdout['PA_DISK']  = pa
		hdout['CRVAL1']   = 0.
		hdout['CRVAL2']   = 0.

		if os.path.exists(outfitsname):
			os.system('rm -r '+outfitsname)

		print 'writing fits file...'
		fits.writeto(outfitsname, redata, header=hdout)



	### plot results
	if figout:
		# figure
		fig = plt.figure(figsize=(11.69, 8.27))
		ax  = fig.add_subplot(111)

		# image size
		figxmax, figxmin, figymin, figymax = imscale


		# images
		imcolor   = ax.imshow(redata[0,0,:,:], cmap=cmap, origin='lower', extent=(xmin, xmax, ymin, ymax))
		#imcontour = ax.contour(data[0,0,:,:], origin='lower', colors='white', levels=clevels, extent=(xmin, xmax, ymin, ymax))
		depro_contour   = ax.contour(redata[0,0,:,:], origin='lower', colors='red', levels=clevels, linewidths=3., extent=(xmin, xmax, ymin, ymax))

		# images for test
		#imrot     = ax.contour(datarot, origin='lower', colors='cyan', levels=clevels, extent=(xmin, xmax, ymin, ymax))
		#imcontour = ax.contour(xx, yy, data[0,0,:,:], origin='lower', colors='white', levels=clevels)
		ax.contour(xxparot, yyinc, data[0,0,:,:], origin='lower', colors='cyan', levels=clevels, extent=(np.nanmax(xxparot), np.nanmin(xxparot), np.nanmin(yyinc), np.nanmax(yyinc)))


		# plot beam size
		beam    = patches.Ellipse(xy=(figxmax-locsym*figxmax, figymin-locsym*figymin), width=bmin, height=bmaj, fc='white', angle=-bpa)
		ax.add_patch(beam)

		# axes
		ax.set_xlim(figxmin,figxmax)
		ax.set_ylim(figymin,figymax)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_aspect(1)
		ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)

		fig.savefig(outfigname, transparent = True)
		#plt.show()

	return redata




if __name__ == '__main__':
	### input values
	fitsimage = 'tmc1a.c18o21.contsub.rbp05.mlt100.clean.image.reg.smoothed.croped.fits'
	inc       = 65. # from Aso et al. (2015) [deg]
	pa        = 72. # from Aso et al. (2015) [deg]
	outname   = 'tmc1a.c18o21.deprojected'
	rmsc18o   = 5.8e-3
	clevels   = np.array([3.,6.,9.,12.,15.,20.,25.,30.,40.,50.,60.])*rmsc18o
	imscale   = np.array([-3,3,-3,3])


	data_depro = fits_deprojection(fitsimage, inc, pa, outname=outname, imscale=imscale, fitsout=True)
