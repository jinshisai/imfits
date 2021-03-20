# Deproject fits coordinates
def fits_deprojection(self, noreg=True, relativecoords=True):
	'''
	Deproject coordinates of a fits file. Steps of transformation is as below:
	 1. Projected coordinates (x, y) --> native polar coordinates (phi, theta)
	 2. (phi, theta) --> (ra, dec)

	Args:
		noreg (bool): If False, regrid will be done. Default True.
	 	 For some projections, this will be needed to draw maps with exact coordinates.
	 	use (str): 'relative' or 'absolute'
	'''
	header = self.header


	if 'LONPOLE' in header:
		phi_p = header['LONPOLE']
	else:
		phi_p = 180.

	if 'LATPOLE' in header:
		the_p = header['LATPOLE']
	else:
		the_p = None

	# x & y (RA & DEC)
	xaxis, yaxis, vaxis, saxis = self.axes
	projection                 = self.projection

	# 1. get intermidiate coordinates before the projection correction
	xx, yy = np.meshgrid(xaxis, yaxis)
	#print xx[0,0],xx[-1,-1],yy[0,0],yy[-1,-1]


	# 2. (x,y) --> (phi, theta): native coordinates
	#  Correct projection effect, and then put them into polar coordinates
	#  For detail, look into Mark R. Calabretta and Eric W. Greisen (A&A, 2002)
	#   and Mark Calabretta (1992)
	#  Here variables are as following:
	#	phi, theta: (phi, theta) in polar coordinates
	#	alpha_0, delta_0: Celestial longitude and latitude of the fiducial point
	#	alpha_p, delta_p: Celestial longitude and latitude of the native pole
	#	phi_0, the_0: Native longitude and latitude of the fiducial point
	#   phi_p, the_p: Native longitude and latitude of the celestial pole

	# SIN: Slant orthographic
	if projection == 'SIN':
		# print ('projection: SIN')

		# exact solution of phi & theta
		# xi & eta
		xi  = header['PV1_1'] if 'PV1_1' in header else 0.
		eta = header['PV1_2'] if 'PV1_2' in header else 0.

		# eq. (38) and (44) in Calabretta and Greisen (A&A, 2002),
		#  where X' & Y' are replaced with xi and eta
		xx_rad = xx*np.pi/180.
		yy_rad = yy*np.pi/180.
		a = xi*xi + eta*eta + 1.
		b = xi*(xx_rad - xi) + eta*(yy_rad - eta)
		c = (xx_rad - xi)**2 + (yy_rad - eta)**2 - 1.

		sol_1 = np.arcsin((-b + np.sqrt(b*b - a*c))/a)*180./np.pi
		sol_2 = np.arcsin((-b - np.sqrt(b*b - a*c))/a)*180./np.pi

		if np.any(np.abs(90. - sol_1) <= np.abs(90. - sol_2)):
			theta = sol_1
		else:
			theta = sol_2

		sin_th = np.sin(theta*np.pi/180.)
		phi    = np.arctan2((xx_rad - xi*(1. - sin_th)), -(yy_rad - eta*(1. - sin_th) ))*180./np.pi

		# approximate solution
		#phi   = np.arctan2(xx,-yy)*180./np.pi
		#theta = np.arccos(np.sqrt(xx*xx + yy*yy)*np.pi/180.)*180./np.pi
		#print phi
		#print theta

		# values for converstion from (phi, theta) to (ra, dec)
		alpha_0 = self.refval_i[0] # degree
		delta_0 = self.refval_i[1]
		alpha_p = alpha_0
		delta_p = delta_0
		the_0   = 90.
		phi_0   = 0.
		reg     = False
	elif projection == 'SFL':
		# (ra, dec) of reference position is (0,0) in (phi, theta) and (x,y)
		# (0,0) is on a equatorial line, and (0, 90) is the pole in a native spherical coordinate
		print ('projection: SFL')

		# values for converstion from (phi, theta) to (ra, dec)
		alpha_0 = self.refval_i[0]
		delta_0 = self.refval_i[1]
		phi_0   = header['PV1_1'] if 'PV1_1' in header else 0.
		the_0   = header['PV1_2'] if 'PV1_2' in header else 0.
		alpha_p = None
		delta_p = None
		reg     = True

		# phi & theta
		xoff0 = phi_0 * np.cos(the_0*np.pi/180.)
		yoff0 = the_0
		cos   = np.cos(np.radians(yy))
		phi   = (xx - xoff0)/cos # deg
		theta = yy + yoff0       # deg
		#print (the_0)

		if the_p:
			pass
		else:
			the_p = 90.
	elif projection == 'GLS':
		print ('WARNING\tfits_deprojection: The projection GFL is treated as a projection SFL.')
		# values for converstion from (phi, theta) to (ra, dec)
		alpha_0 = self.refval_i[0]
		delta_0 = self.refval_i[1]
		phi_0   = header['PV1_1'] if 'PV1_1' in header else 0.
		the_0   = header['PV1_2'] if 'PV1_2' in header else 0.
		alpha_p = None
		delta_p = None
		reg     = True

		# phi & theta
		xoff0 = phi_0 * np.cos(the_0*np.pi/180.)
		yoff0 = the_0
		cos   = np.cos(np.radians(yy))
		phi   = (xx - xoff0)/cos # deg
		theta = yy + yoff0       # deg
		#print (the_0)
		if the_p:
			pass
		else:
			the_p = 90.
	elif projection == 'TAN':
		#print 'projection: TAN'
		phi   = np.arctan2(xx,-yy)*180./np.pi
		theta = np.arctan2(180.,np.sqrt(xx*xx + yy*yy)*np.pi)*180./np.pi

		# values for converstion from (phi, theta) to (ra, dec)
		alpha_0 = self.refval_i[0]
		delta_0 = self.refval_i[1]
		the_0   = 90.
		phi_0   = 0.
		alpha_p = alpha_0
		delta_p = delta_0
		reg     = False
	else:
		print ('ERROR\tfits_deprojection: Input value of projection is wrong. Can be only SIN or SFL now.')
		pass


	# 3. (phi, theta) --> (ra, dec) (sky plane)
	# Again, for detail, look into Mark R. Calabretta and Eric W. Greisen (A&A, 2002)
	# (alpha_p, delta_p): cerestial coordinate of the native coordinate pole
	# In SFL projection, reference point is not polar point

	# parameters
	sin_th0  = np.sin(np.radians(the_0))
	cos_th0  = np.cos(np.radians(the_0))
	sin_del0 = np.sin(np.radians(delta_0))
	cos_del0 = np.cos(np.radians(delta_0))


	# spherical coordinate rotation or not
	if phi_0 == 0. and the_0 == 90.:
		# case of spherical coordinate rotation
		sin_delp = np.sin(np.radians(delta_p))
		cos_delp = np.cos(np.radians(delta_p))
		pass
	else:
		# with non-polar (phi0, and theta0)
		# we have to derive delta_p and alpha_p
		argy    = sin_th0
		argx    = cos_th0*np.cos(np.radians(phi_p-phi_0))
		arg     = np.arctan2(argy,argx)
		#print (arg)

		cos_inv  = np.arccos(sin_del0/(np.sqrt(1. - cos_th0*cos_th0*np.sin(np.radians(phi_p - phi_0))*np.sin(np.radians(phi_p - phi_0)))))

		sol_1 = (arg + cos_inv)*180./np.pi
		sol_2 = (arg - cos_inv)*180./np.pi

		if np.any(np.abs(the_p - sol_1) <= np.abs(the_p - sol_2)):
			delta_p = sol_1
		else:
			delta_p = sol_2

		sin_delp = np.sin(np.radians(delta_p))
		cos_delp = np.cos(np.radians(delta_p))

		if delta_p == 90.:
			alpha_p = alpha_0 + phi_p - phi_0 - 180.
		elif delta_p == -90.:
			alpha_p = alpha_0 - phi_p + phi_0
		else:
			sin_alpha_p = np.sin(np.radians(phi_p - phi_0))*cos_th0/cos_del0
			cos_alpha_p = (sin_th0 - sin_delp*sin_del0)/(cos_delp*cos_del0)
			#print sin_alpha_p, cos_alpha_p
			#print np.arctan2(sin_alpha_p,cos_alpha_p)*180./np.pi
			alpha_p = alpha_0 - np.arctan2(sin_alpha_p,cos_alpha_p)*180./np.pi
			#print (alpha_p)


	# (phi, theta) --> (ra, dec) finally
	sin_th = np.sin(np.radians(theta))
	cos_th = np.cos(np.radians(theta))

	in_sin = sin_th*sin_delp + cos_th*cos_delp*np.cos(np.radians(phi-phi_p))
	delta  = np.arcsin(in_sin)*180./np.pi

	argy  = -cos_th*np.sin(np.radians(phi-phi_p))
	argx  = sin_th*cos_delp - cos_th*sin_delp*np.cos(np.radians(phi-phi_p))
	alpha = alpha_p + np.arctan2(argy,argx)*180./np.pi
	#print (alpha)

	alpha[np.where(alpha < 0.)]   = alpha[np.where(alpha < 0.)] + 360.
	alpha[np.where(alpha > 360.)] = alpha[np.where(alpha > 360.)] - 360.

	# coordinate type: relative or absolute
	if relativecoords:
		alpha = (alpha - alpha_0)*np.cos(np.radians(delta))
		delta = delta - delta_0
		self.ctype = 'relative'
		print ('Coordinates: relative.')
	else:
		self.ctype = 'absolute'
		print ('Coordinates: absolute.')

	# cosine term is to rescale alpha to delta at delta
	# exact alpha can be derived without cosine term

	# check
	# both must be the same
	#print sin_th[0,0]
	#print (np.sin(np.radians(delta[0,0]))*sin_delp + np.cos(np.radians(delta[0,0]))*cos_delp*np.cos(np.radians(alpha[0,0]-alpha_p)))
	# both must be the same
	#print cos_th[0,0]*np.sin(np.radians(phi[0,0]-phi_p))
	#print -np.cos(np.radians(delta[0,0]))*np.sin(np.radians(alpha[0,0]-alpha_p))
	# both must be the same
	#print cos_th[0,0]*np.cos(np.radians(phi[0,0]-phi_p))
	#print np.sin(np.radians(delta[0,0]))*cos_delp - np.cos(np.radians(delta[0,0]))*sin_delp*np.cos(np.radians(alpha[0,0]-alpha_p))



	# new x & y axis (ra, dec in degree)
	xx     = alpha
	yy     = delta
	#del_x  = xx[0,0] - xx[1,1]
	#del_y  = yy[0,0] - yy[1,0]
	#xmin   = xx[0,0] - 0.5*del_x
	#xmax   = xx[-1,-1] + 0.5*del_x
	#ymin   = yy[0,0] - 0.5*del_y
	#ymax   = yy[-1,-1] + 0.5*del_y
	#extent = (xmin, xmax, ymin, ymax)
	#print extent
	#print xx[0,0]-alpha_0, xx[-1,-1]-alpha_0,yy[0,0]-delta_0,yy[-1,-1]-delta_0

	# check
	#print (xx[0,0],yy[0,0])
	#print xx[0,-1],yy[0,-1]
	#print xx[-1,0],yy[-1,0]
	#print xx[-1,-1],yy[-1,-1]
	#print del_i[0], del_x, xx[0:naxis_i[1]-1,0:naxis_i[1]-1] - xx[1:naxis_i[1],1:naxis_i[1]]
	#print del_i[1], yy[0,0] - yy[1,0]

	# regridding for plot if the projection requires
	if noreg:
		pass
	else:
		from scipy.interpolate import griddata
		# new grid
		xc2     = np.linspace(np.nanmax(xx),np.nanmin(xx),naxis_i[0])
		yc2     = np.linspace(np.nanmin(yy),np.nanmax(yy),naxis_i[1])
		xx_new, yy_new = np.meshgrid(xc2, yc2)

		# nan --> 0
		data[np.where(np.isnan(data))] = 0.
		#print np.max(data)

		# 2D --> 1D
		xinp    = xx.reshape(xx.size)
		yinp    = yy.reshape(yy.size)


		# regrid
		print ('regriding...')


		# internal expression
		data_reg = griddata((xinp, yinp), data.reshape(data.size), (xx_new, yy_new), method='cubic',rescale=True)
		data_reg = data_reg.reshape((data.shape))


		# renew data & axes
		data = data_reg
		xx   = xx_new
		yy   = yy_new

	self.xx = xx
	self.yy = yy
	self.cc = np.array([alpha_0, delta_0])

	return
