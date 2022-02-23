# -*- coding: utf-8 -*-
'''
Made and developed by J. Sai.

email: jn.insa.sai@gmail.com
'''



### Modules
import numpy as np
from astropy.io import fits
import astropy.wcs
from astropy.wcs import WCS
import matplotlib.pyplot as plt



### Constants (in cgs)
clight     = 2.99792458e10 # light speed [cm s^-1]




### Imfits
class Imfits():
	'''
	Read a fits file, store the information, and draw maps.
	'''


	def __init__(self, infile, pv=False):
		self.file = infile
		self.data, self.header = fits.getdata(infile, header=True)

		if pv:
			self.read_pvfits()
		else:
			self.read_header()
			self.get_coordinates()
			#self.fits_deprojection(relativecoords=relativecoords)

		self.ifpv = pv


	def read_header(self, velocity=True):
		'''
		Get axes of a fits file. x axis and y axis will be in intermediate coordinates.

		Args:
			fitsname: input fits file name

		Return:
			xaxis, yaxis, vaxis, saxis
		'''
		# reading fits files
		header = self.header
		#data, header = fits.getdata(fitsname,header=True)


		# number of axis
		naxis    = header['NAXIS']
		if naxis < 2:
			print ('ERROR\treadfits: NAXIS of fits is < 2.')
			return
		self.naxis = naxis

		naxis_i  = np.array([int(header['NAXIS'+str(i+1)]) for i in range(naxis)])
		label_i  = np.array([header['CTYPE'+str(i+1)] for i in range(naxis)])
		refpix_i = np.array([int(header['CRPIX'+str(i+1)]) for i in range(naxis)])
		refval_i = np.array([header['CRVAL'+str(i+1)] for i in range(naxis)]) # degree
		if 'CDELT1' in header:
			del_i = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree
		self.naxis_i  = naxis_i
		self.label_i  = label_i
		self.refpix_i = refpix_i
		self.refval_i = refval_i

		# beam size (degree)
		if 'BMAJ' in header:
			bmaj     = header['BMAJ'] # degree
			bmin     = header['BMIN'] # degree
			bpa      = header['BPA']  # degree
			self.beam = np.array([bmaj*3600., bmin*3600., bpa]) # Get in arcsec
		else:
			self.beam = None


		# rest frequency (Hz)
		if 'RESTFRQ' in header:
			restfreq = header['RESTFRQ']
		elif 'RESTFREQ' in header:
			restfreq = header['RESTFREQ']
		elif 'FREQ' in header:
			restfreq = header['FREQ']
		else:
			restfreq = None
		self.restfreq = restfreq

		if 'LONPOLE' in header:
		    phi_p = header['LONPOLE']
		else:
		    phi_p = 180.


		# coordinates
		# read projection type
		try:
			projection = label_i[0].replace('RA---','')
		except:
			print ('Cannot read information about projection from fits file.')
			print ('Set projection SIN for radio interferometric data.')
			projection = 'SIN'
		self.projection = projection

		# rotation of pixel coordinates
		if 'PC1_1' in header:
			pc_ij = np.array([
				[header['PC%i_%i'%(i+1,j+1)]
				if 'PC%i_%i'%(i+1,j+1) in header else 0.
				for j in range(naxis)] for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])
		elif 'CD1_1' in header:
			pc_ij = np.array([
			[header['CD%i_%i'%(i+1,j+1)]
			if 'CD%i_%i'%(i+1,j+1) in header else 0.
			for j in range(naxis)] for i in range(naxis)])
		else:
			print ('CAUTION\tchannelmap: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
			pc_ij = np.array([
				[1. if i==j else 0. for j in range(naxis)]
				 for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])


		# axes
		axes = np.array([np.dot(pc_ij, (i+1 - refpix_i))\
		 for i in range(np.max(naxis_i))]).T      # +1 in i+1 comes from 0 start index in python


		# x & y (RA & DEC)
		xaxis = axes[0]
		xaxis = xaxis[:naxis_i[0]]                # offset, relative
		yaxis = axes[1]
		yaxis = yaxis[:naxis_i[1]]                # offset, relative
		self.xaxis = xaxis
		self.yaxis = yaxis
		self.delx = xaxis[1] - xaxis[0]
		self.dely = yaxis[1] - yaxis[0]
		self.nx = naxis_i[0]
		self.ny = naxis_i[1]

		# frequency & stokes
		if naxis >= 3:
			# frequency
			vaxis = axes[2]
			vaxis = vaxis[:naxis_i[2]] + refval_i[2]  # frequency, absolute

			if naxis == 4:
				# stokes
				saxis = axes[3]
				saxis = saxis[:naxis_i[3]]
			else:
				saxis = np.array([0.])
		else:
			vaxis = np.array([0.])
			saxis = np.array([0.])


		# frequency --> velocity
		if len(vaxis) > 1:
			if velocity:
				if label_i[2] == 'VRAD' or label_i[2] == 'VELO':
					print ('The third axis is ', label_i[2])
					# m/s --> km/s
					vaxis    = vaxis*1.e-3 # m/s --> km/s
					#del_v    = del_v*1.e-3
					#refval_v = refval_v*1.e-3
					#vaxis    = vaxis*1.e-3
				else:
					print ('The third axis is ', label_i[2])
					print ('Convert frequency to velocity')
					# frequency (Hz) --> radio velocity (km/s)
					vaxis = clight*(1.-vaxis/restfreq)*1.e-5

				if len(vaxis) >= 2:
					self.delv = vaxis[1] - vaxis[0]
				else:
					self.delv = 1.

		self.vaxis = vaxis
		self.saxis = saxis

		axes = np.array([xaxis, yaxis, vaxis, saxis], dtype=object)
		self.axes  = axes


	# Read fits file of Poistion-velocity (PV) diagram
	def read_pvfits(self):
		'''
		Read fits file of pv diagram produced by CASA.
		'''
		# read header
		header = self.header

		# number of axis
		naxis    = header['NAXIS']
		if naxis < 2:
			print ('ERROR\treadfits: NAXIS of fits is < 2.')
			return
		self.naxis = naxis

		naxis_i  = np.array([int(header['NAXIS'+str(i+1)]) for i in range(naxis)])
		label_i  = np.array([header['CTYPE'+str(i+1)] for i in range(naxis)])
		refpix_i = np.array([int(header['CRPIX'+str(i+1)]) for i in range(naxis)])
		refval_i = np.array([header['CRVAL'+str(i+1)] for i in range(naxis)]) # degree
		if 'CDELT1' in header:
			del_i = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree
		self.naxis_i  = naxis_i
		self.label_i  = label_i
		self.refpix_i = refpix_i
		self.refval_i = refval_i

		# beam size (degree)
		if 'BMAJ' in header:
			bmaj     = header['BMAJ'] # degree
			bmin     = header['BMIN'] # degree
			bpa      = header['BPA']  # degree
			self.beam = np.array([bmaj*3600., bmin*3600., bpa])
		else:
			self.beam = None


		# Info. of P.A.
		if 'PA' in header:
			self.pa = header['PA']
		elif 'P.A.' in header:
			self.pa = header['P.A.']
		else:
			self.pa = None

		# Resolution along offset axis
		if self.pa is not None:
			# an ellipse of the beam
			# (x/bmin)**2 + (y/bmaj)**2 = 1
			# y = x*tan(theta)
			# --> solve to get resolution in the direction of pv cut with P.A.=pa
			bmaj, bmin, bpa = self.beam
			del_pa = self.pa - bpa
			del_pa = del_pa*np.pi/180. # radian
			term_sin = (np.sin(del_pa)/bmin)**2.
			term_cos = (np.cos(del_pa)/bmaj)**2.
			res_off  = np.sqrt(1./(term_sin + term_cos))
			self.res_off = res_off
		else:
			self.res_off = None


		# rest frequency (Hz)
		if 'RESTFRQ' in header:
			restfreq = header['RESTFRQ']
		elif 'RESTFREQ' in header:
			restfreq = header['RESTFREQ']
		elif 'FREQ' in header:
			restfreq = header['FREQ']
		else:
			restfreq = None
		self.restfreq = restfreq


		# get axes
		# rotation of pixel coordinates
		if 'PC1_1' in header:
			pc_ij = np.array([
				[header['PC%i_%i'%(i+1,j+1)]
				if 'PC%i_%i'%(i+1,j+1) in header else 0.
				for j in range(naxis)] for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])
		elif 'CD1_1' in header:
			pc_ij = np.array([
			[header['CD%i_%i'%(i+1,j+1)]
			if 'CD%i_%i'%(i+1,j+1) in header else 0.
			for j in range(naxis)] for i in range(naxis)])
		else:
			print ('CAUTION\tchannelmap: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
			pc_ij = np.array([
				[1. if i==j else 0. for j in range(naxis)]
				 for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])

		# axes
		axes = np.array([np.dot(pc_ij, (i+1 - refpix_i))\
		 for i in range(np.max(naxis_i))]).T # +1 in i+1 comes from 0 start index in python

		# x & v axes
		xaxis = axes[0]
		vaxis = axes[1]
		xaxis = xaxis[:naxis_i[0]]               # offset
		vaxis = vaxis[:naxis_i[1]] + refval_i[1] # frequency, absolute

		# check unit of offest
		if 'CUNIT1' in header:
			unit_i = np.array([header['CUNIT'+str(i+1)] for i in range(naxis)]) # degree
			if unit_i[0] == 'degree' or unit_i[0] == 'deg':
				# degree --> arcsec
				xaxis    = xaxis*3600.
				del_i[0] = del_i[0]*3600.
		else:
			print ('WARNING: No unit information in the header.\
				Assume units of arcesc and Hz for offset and frequency axes, respectively.')

		# frequency --> velocity
		if label_i[1] == 'VRAD' or label_i[1] == 'VELO':
			vaxis    = vaxis*1.e-3 # m/s --> km/s
			#del_v    = del_v*1.e-3
			#refval_v = refval_v*1.e-3
		else:
			print ('Convert frequency to velocity')
			vaxis    = clight*(1.-vaxis/restfreq) # radio velocity c*(1-f/f0) [cm/s]
			vaxis    = vaxis*1.e-5                # cm/s --> km/s
			#del_i[1] = -del_i[1]*clight/restfreq  # delf --> delv [cm/s]
			#del_i[1] = del_i[1]*1.e-5             # cm/s --> km/s

		axes_out = np.array([xaxis, vaxis], dtype=object)
		if naxis >= 2:
			saxis = axes[2]
			saxis = saxis[:naxis_i[2]]
			axes_out = np.array([xaxis, vaxis, saxis], dtype=object)


		# get delta
		delx = xaxis[1] - xaxis[0]
		delv = vaxis[1] - vaxis[0]

		self.axes  = axes_out
		self.xaxis = xaxis
		self.vaxis = vaxis
		self.delx  = delx
		self.delv  = delv


	# Get sky coordinates with astropy
	def get_coordinates(self):
		'''
		Get sky coordinates.
		'''
		# Get wcs
		wcs = WCS(self.file)
		self.wcs = wcs

		naxis_i  = self.naxis_i
		refpix_i = self.refpix_i
		refval_i = self.refval_i
		nx = naxis_i[0]
		ny = naxis_i[1]
		ref_x = refval_i[0]
		ref_y = refval_i[1]

		xaxis = np.arange(0,nx,1)
		yaxis = np.arange(0,ny,1)
		xx, yy = np.meshgrid(xaxis, yaxis)
		sc     = astropy.wcs.utils.pixel_to_skycoord(xx, yy, wcs)

		# Sky coordinates
		# world coordinates
		xx = sc.ra.deg
		yy = sc.dec.deg

		self.xx_wcs = xx
		self.yy_wcs = yy

		# relative coordinates
		# ra
		xx = xx - ref_x
		xx = xx*np.cos(np.radians(yy))
		# dec
		yy = yy - ref_y

		self.xx = xx
		self.yy = yy
		self.xaxis = xx[ny//2,:]
		self.yaxis = yy[:,nx//2]
		self.cc = np.array([ref_x, ref_y]) # coordinate center


	def shift_coord_center(self, coord_center):
		'''
		Shift the coordinate center.

		Args:
			coord_center: Put an coordinate for the map center.
			   The shape must be '00h00m00.00s 00d00m00.00s', or
			   'hh:mm:ss.ss dd:mm:ss.ss'. RA and DEC must be separated
			   by space.
		'''
		# module
		from astropy.coordinates import SkyCoord

		# ra, dec
		c_ra, c_dec = coord_center.split(' ')
		cc          = SkyCoord(c_ra, c_dec, frame='icrs')
		cra_deg     = cc.ra.degree                   # in degree
		cdec_deg    = cc.dec.degree                  # in degree
		new_cent    = np.array([cra_deg, cdec_deg])  # absolute coordinate of the new image center

		# current coordinates
		alpha = self.xx_wcs
		delta = self.yy_wcs


		# shift of the center
		alpha = (alpha - cra_deg)*np.cos(np.radians(delta))
		delta = delta - cdec_deg

		# update
		self.xx = alpha
		self.yy = delta
		self.cc = new_cent


	def getmoments(self, moment=[0], vrange=[], threshold=[],
		outfits=True, outname=None, overwrite=False):
		'''
		Calculate moment maps.

		moment (list): Index of moments that you want to calculate.
		vrange (list): Velocity range to calculate moment maps.
		threshold (list): Threshold to clip data. Shoul be given as [minimum intensity, maximum intensity]
		outfits (bool): Output as a fits file?
		outname (str): Output file name if outfits=True.
		overwrite (bool): If overwrite an existing fits file or not.
		'''

		data  = self.data
		xaxis, yaxis, vaxis, saxis = self.axes
		nx = len(xaxis)
		ny = len(yaxis)
		delv = np.abs(vaxis[1] - vaxis[0])

		if len(data.shape) <= 2:
			print ('ERROR\tgetmoments: Data must have more than three axes to calculate moments.')
			return
		elif len(data.shape) == 3:
			pass
		elif len(data.shape) == 4:
			data = data[0,:,:,:]
		else:
			print ('ERROR\tgetmoments: Data have more than five axes. Cannot recognize.')
			return

		if len(vrange) == 2:
			index = np.where( (vaxis >= vrange[0]) & (vaxis <= vrange[1]))
			data  = data[index[0],:,:]
			vaxis = vaxis[index[0]]

		if len(threshold):
			index = np.where( (data < threshold[0]) | (data > threshold[1]) )
			data[index] = 0.


		# condition
		nchan = len(vaxis)
		print ('Calculate moments of an image.')
		print ('nchan: %i'%nchan)
		print ('Velocity range: %.4f--%.4f km/s'%(vaxis[0], vaxis[-1]))
		print ('Calculating...')
		#print (vaxis)


		# start
		mom0 = np.array([[np.sum(delv*data[:,j,i]) for i in range(nx)] for j in range(ny)])
		w2   = np.array([[np.sum(delv*data[:,j,i]*delv*data[:,j,i])
			for i in range(nx)] for j in range(ny)]) # Sum_i w_i^2
		ndata = np.array([
			[len(np.nonzero(data[:,j,i])[0]) for i in range(nx)]
			 for j in range(ny)]) # number of data points used for calculations


		if any([i == 0 for i in moment ]):
			moments     = [mom0]
			#moments_err = []
		else:
			moments     = []
			#moments_err = []

		# moment 1
		if any([i >= 1 for i in moment ]):
			mom1 = np.array([[np.sum((data[:,j,i]*vaxis*delv))
				for i in range(nx)]
				for j in range(ny)])/mom0


			# calculate error
			sig_v2   = np.array([[np.sum(
				delv*((mom1[j,i] - vaxis[np.where(data[:,j,i] > 0)])**2))
			for i in range(nx)] for j in range(ny)])

			# Eq. (2.3) of Belloche 2013
			# approximated solution
			#sig_mom1_bl = rms*np.sqrt(sig_v2)/mom0 # 1 sigma, standerd deviation

			sig_mom1 = np.sqrt(w2*sig_v2/(ndata - 1.))/mom0 # accurate
			#print (sig_mom1)

			# taking into account error of weighting
			#sum_v2     = np.array([[
			#		np.sum(vaxis[np.where(data[:,j,i] > 0)]*delv*vaxis[np.where(data[:,j,i] > 0)]*delv)
			#		for i in range(nx)] for j in range(ny)])
			#term_sigv2 = w2/(mom0*mom0)*(sig_v2/(ndata - 1.))
			#term_sigw2 = (ndata*mom1*mom1 + sum_v2)*rms*rms/(mom0*mom0)
			#sig_mom1 = np.sqrt(term_sigv2 + term_sigw2)

			# check
			#sct = plt.scatter(sig_mom1_bl.ravel(), sig_mom1.ravel(), alpha=0.5,
			# c=(np.array([[np.nanmax(data[:,j,i])/rms for i in range(nx)] for j in range(ny)])/np.sqrt(ndata)).ravel(),
			# rasterized=True)
			#plt.plot(np.linspace(0.,0.2, 32), 0.5*np.linspace(0.,0.2, 32), lw=3., alpha=0.7, color='k', ls='--')
			#plt.plot(np.linspace(0.,0.2, 32), 2.*np.linspace(0.,0.2, 32), lw=3., alpha=0.7, color='k', ls='--')
			#plt.plot(np.linspace(0.,0.2, 32), np.linspace(0.,0.2, 32), lw=3., alpha=0.7, color='k')
			#plt.colorbar(sct, label=r'$\mathrm{SNR} / \sqrt{n_\mathrm{data}}$')
			#plt.aspect_ratio(1)

			#plt.xlim(0,0.2)
			#plt.ylim(0,0.2)
			#plt.xlabel(r'$\sigma_\mathrm{mom1, Belloche13}$')
			#plt.ylabel(r'$\sigma_\mathrm{mom1, accurate}$')
			#plt.show()

			#moments_err.append(sig_mom1)

			if any([i == 1 for i in moment]):
				moments.append(mom1)
				moments.append(sig_mom1)


		if any([i == 2 for i in moment]):
			mom2 = np.sqrt(np.array([[np.sum((data[:,j,i]*delv*(vaxis - mom1[j,i])**2.))
			for i in range(nx)]
			for j in range(ny)])/mom0)

			sig_mom2 = np.sqrt(np.array([[
				2./(np.count_nonzero(data[:,j,i]) - 1.) \
				* (mom0[j,i]*mom2[j,i]*mom2[j,i]/(mom0[j,i] - (w2[j,i]/mom0[j,i])))**2.
				if (np.count_nonzero(data[:,j,i]) - 1.) > 0
				else 0.
				for i in range(nx)]
				for j in range(ny)]))

			#moments_err.append(sig_mom2)
			moments.append(mom2)
			moments.append(sig_mom2)

		print ('Done.')


		# output
		#print (np.array(moments).shape)
		print ('Output: Moment '+' '.join([str(moment[i]) for i in range(len(moment))])
			+' and thier error maps except for moment 0.')

		if outfits:
			hdout = self.header
			naxis = self.naxis

			if naxis == 3:
				outmaps = np.array(moments)
			elif naxis == 4:
				outmaps = np.array([moments])
			else:
				print ('ERROR\tgetmoments: Input fits file must have 3 or 4 axes.')
				return

			# moment axis
			hdout['NAXIS3'] = len(moments)
			hdout['CTYPE3'] = 'Moments'
			hdout['CRVAL3'] = 1.
			hdout['CRPIX3'] = 1.
			hdout['CDELT3'] = 1.
			hdout['CUNIT3'] = '       '

			hdout['HISTORY'] = 'Produced by getmoments in Imfits.'
			hdout['HISTORY'] = 'Moments: '+' '.join([str(moment[i]) for i in range(len(moment))])

			if outname:
				pass
			else:
				outname = self.file.replace('.fits','_moments.fits')

			#print (outmaps.shape)
			fits.writeto(outname, outmaps, header=hdout, overwrite=overwrite)

		return moments


	# Draw pv diagram
	def draw_pvdiagram(self,outname,data=None,header=None,ax=None,outformat='pdf',color=True,cmap='Greys',
		vmin=None,vmax=None,vsys=0,contour=True,clevels=None,ccolor='k', pa=None,
		vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
		lw=1,clip=None,plot_res=True,inmode='fits',xranges=[], yranges=[],
		ln_hor=True, ln_var=True, alpha=None):
		'''
		Draw a PV diagram.

		Args:
		 - outname:
		'''

		# Modules
		import copy
		import matplotlib as mpl

		# format
		formatlist = np.array(['eps','pdf','png','jpeg'])

		# properties of plots
		#mpl.use('Agg')
		plt.rcParams['font.family']     ='Arial' # font (Times New Roman, Helvetica, Arial)
		plt.rcParams['xtick.direction'] = 'in'   # directions of x ticks ('in'), ('out') or ('inout')
		plt.rcParams['ytick.direction'] = 'in'   # directions of y ticks ('in'), ('out') or ('inout')
		plt.rcParams['font.size']       = fontsize  # fontsize

		def change_aspect_ratio(ax, ratio):
			'''
			This function change aspect ratio of figure.
			Parameters:
			    ax: ax (matplotlit.pyplot.subplots())
			        Axes object
			    ratio: float or int
			        relative x axis width compared to y axis width.
			'''
			aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
			aspect = np.abs(aspect)
			aspect = float(aspect)
			ax.set_aspect(aspect)


		# output file
		if (outformat == formatlist).any():
			outname = outname + '.' + outformat
		else:
			print ('ERROR\tsingleim_to_fig: Outformat is wrong.')
			return

		# Input
		if inmode == 'data':
			if data is None:
				print ("inmode ='data' is selected. data must be provided.")
				return
			naxis = len(data.shape)
		else:
			data   = self.data
			header = self.header
			naxis  = self.naxis


		# figures
		if ax:
			pass
		else:
			fig = plt.figure(figsize=(11.69,8.27)) # figsize=(11.69,8.27)
			ax  = fig.add_subplot(111)

		# Read
		xaxis = self.xaxis
		vaxis = self.vaxis
		delx  = self.delx
		delv  = self.delv
		nx    = len(xaxis)
		nv    = len(vaxis)

		# Beam
		bmaj, bmin, bpa = self.beam

		if self.res_off:
			res_off = self.res_off
		else:
			# Resolution along offset axis
			if self.pa:
				pa = self.pa

			if pa:
				# an ellipse of the beam
				# (x/bmin)**2 + (y/bmaj)**2 = 1
				# y = x*tan(theta)
				# --> solve to get resolution in the direction of pv cut with P.A.=pa
				del_pa = pa - bpa
				del_pa = del_pa*np.pi/180. # radian
				term_sin = (np.sin(del_pa)/bmin)**2.
				term_cos = (np.cos(del_pa)/bmaj)**2.
				res_off  = np.sqrt(1./(term_sin + term_cos))
			else:
				res_off = bmaj

		# relative velocity or LSRK
		offlabel = r'$\mathrm{Offset\ (arcsec)}$'
		if vrel:
			vaxis   = vaxis - vsys
			vlabel  = r'$\mathrm{Relative\ velocity\ (km\ s^{-1})}$'
			vcenter = 0
		else:
			vlabel  = r'$\mathrm{LSR\ velocity\ (km\ s^{-1})}$'
			vcenter = vsys


		# set extent of an image
		offmin = xaxis[0] - delx*0.5
		offmax = xaxis[-1] + delx*0.5
		velmin = vaxis[0] - delv*0.5
		velmax = vaxis[-1] + delv*0.5


		# set axes
		if x_offset:
			data   = data[0,:,:]
			extent = (offmin,offmax,velmin,velmax)
			xlabel = offlabel
			ylabel = vlabel
			hline_params = [vsys,offmin,offmax]
			vline_params = [0.,velmin,velmax]
			res_x = res_off
			res_y = delv
		else:
			data   = np.rot90(data[0,:,:])
			extent = (velmin,velmax,offmin,offmax)
			xlabel = vlabel
			ylabel = offlabel
			hline_params = [0.,velmin,velmax]
			vline_params = [vcenter,offmin,offmax]
			res_x = delv
			res_y = res_off


		# set colorscale
		if vmax:
			pass
		else:
			vmax = np.nanmax(data)


		# logscale
		if logscale:
			norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
		else:
			norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)


		# clip data at some value
		data_color = copy.copy(data)
		if clip:
			data_color[np.where(data < clip)] = np.nan

		# plot images
		if color:
			imcolor = ax.imshow(data_color, cmap=cmap, origin='lower',
				extent=extent, norm=norm, alpha=alpha)

		if contour:
			imcont  = ax.contour(data, colors=ccolor, origin='lower',
				extent=extent, levels=clevels, linewidths=lw, alpha=alpha)


		# axis labels
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)

		# set xlim, ylim
		if len(xranges) == 0:
			ax.set_xlim(extent[0],extent[1])
		elif len(xranges) == 2:
			xmin, xmax = xranges
			ax.set_xlim(xmin, xmax)
		else:
			print ('WARRING: Input xranges is wrong. Must be [xmin, xmax].')
			ax.set_xlim(extent[0],extent[1])

		if len(yranges) == 0:
			ax.set_ylim(extent[2],extent[3])
		elif len(yranges) == 2:
			ymin, ymax = yranges
			ax.set_ylim(ymin, ymax)
		else:
			print ('WARRING: Input yranges is wrong. Must be [ymin, ymax].')
			ax.set_ylim(extent[2],extent[3])


		# lines showing offset 0 and relative velocity 0
		if ln_hor:
			xline = plt.hlines(hline_params[0], hline_params[1], hline_params[2], ccolor, linestyles='dashed', linewidths = 1.)
		if ln_var:
			yline = plt.vlines(vline_params[0], vline_params[1], vline_params[2], ccolor, linestyles='dashed', linewidths = 1.)

		ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)

		# plot resolutions
		if plot_res:
			# x axis
			#print (res_x, res_y)
			res_x_plt, res_y_plt = ax.transLimits.transform((res_x*0.5, res_y*0.5)) -  ax.transLimits.transform((0, 0)) # data --> Axes coordinate
			ax.errorbar(0.1, 0.1, xerr=res_x_plt, yerr=res_y_plt, color=ccolor, capsize=3, capthick=1., elinewidth=1., transform=ax.transAxes)

		# aspect ratio
		if ratio:
			change_aspect_ratio(ax, ratio)

		# save figure
		plt.savefig(outname, transparent=True)

		return ax
