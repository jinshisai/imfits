import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from imfits import Imfits



# intensity map
def intensitymap(self, ax=None, outname=None, imscale=[], outformat='pdf',
	color=True, cmap='Blues', colorbar=True, cbaroptions=np.array(['right','4%','0%','Jy/beam']), vmin=None,vmax=None,
	contour=True, clevels=None, ccolor='k', clw=1,
	data=None, axis=0, xticks=[], yticks=[], relativecoords=True, csize=18, scalebar=[],
	cstar=True, prop_star=[], color_norm=None, bcolor='k',figsize=(11.69,8.27),
	tickcolor='k',axiscolor='k',labelcolor='k',coord_center=None, plot_beam = True,
	interpolation=None, noreg=True, inmode=None, exact_coord=False):
	'''
	Draw a map from an self cube. You can overplot maps by giving ax where other maps are drawn.


	Parameters (must be given)
	--------------------------
	self: Imfits object read by Imfits.


	Return
	------
	ax: matplotlib axis.


	How to use
	----------
	 - Draw a map:
	 imdata = Imfits('fitsfile')
	 imdata.drawmaps.intensitymap(outname='test', imscale=[-10,10,-10,10], color=True, cmap='Greys')

	 - Overplot:
	rms = 10. # rms
	ax = imdata.drawmaps.intensitymap('object01.fits', outname='test', imscale=[-10,10,-10,10], color=True)
	ax = imdata.drawmaps.intensitymap('object02.fits', outname='test', ax=ax, color=False,
	contour=True, clevels=np.array([3,6,9,12])*rms) # put ax=ax and the same outname

	# If you want to overplot something more
	ax.plot(0, 0, marker='*', size=40)        # overplot
	plt.savefig('test.pdf', transparent=True) # use the same name

	Make a map editting data:
	fitsdata = 'name.fits'
	data, hd = fits.getdata(fitsdata, header=True) # get data
	data = data*3.                                 # edit data
	Idistmap(fitsdata, data=data, header=hd, inmode='data') # set inmode='data'


	Parameters (optional)
	----------------------
	 - Often used
	outname (str): Output file name. Do not include file extension.
	outformat (str): Extension of the output file. Default is pdf.
	imscale (ndarray): self scale [arcsec]. Input as np.array([xmin,xmax,ymin,ymax]).
	color (bool): If True, self will be described in color scale.
	cmap: Choose colortype of the color scale.
	colorbar (bool): If True, color bar will be put in a map. Default False.
	cbaroptions: Detailed setting for colorbar. np.array(['position','width','pad','label']).
	vmin, vmax: Minimun and maximun values in the color scale. Put abusolute values.
	contour (bool): If True, contour will be drawn.
	clevels (ndarray): Set contour levels. Put abusolute values.
	ccolor: Set contour color.
	coord_center (str): Put an coordinate for the map center. The shape must be '00h00m00.00s 00d00m00.00s', or
	'hh:mm:ss.ss dd:mm:ss.ss'. RA and DEC must be separated by space.

	 - Setting details
	xticks, yticks: Optional setting. If input ndarray, xticks and yticsk will be set as the input.
	relativecoords (bool): If True, the coordinate is shown in relativecoordinate. Default True.
	Absolute coordinate is currently now not supported.
	csize: Font size.
	cstar (bool): If True, a cross denoting central stellar position will be drawn.
	prop_star: Detailed setting for the cross showing stellar position.
	np.array(['length','width','color']) or np.array(['length','width','color', 'coordinates']).
	logscale (bool): If True, the color scale will be in log scale.
	figsize (tapule): figure size. Default A4 size.
	plot_beam (bool): If True, an ellipse denoting the beam size will be drawn.
	bcolor: color for the ellipse for the beam.
	tickcolor, axiscolor, labelcolor: color set for the map.
	interpolation (str): The color map is shown with interpolation.
	noreg (bool): If False, coordinates will be regrided when the deprojection is calculated.
	scalebar: Optional setting. Input ndarray([barx, bary, barlength, textx, texty, text ]).
	barx and bary are the position where scalebar will be putted. [arcsec].
	barlength is the length of the scalebar in the figure, so in arcsec.
	textx and texty are the position where a label of scalebar will be putted. [arcsec].
	text is a text which represents the scale of the scalebar.
	cstar (bool): If True, central star position will be marked by cross.
	inmode: 'fits' or 'data'. If 'data' is selected, header must be provided. Default 'fits'.
	'''

	# modules
	import matplotlib.figure as figure
	import matplotlib as mpl
	from astropy.coordinates import SkyCoord
	import matplotlib.patches as patches
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	# format
	formatlist = np.array(['eps','pdf','png','jpeg'])

	# properties of plots
	#mpl.use('Agg')
	#plt.rcParams['font.family']     ='Arial' # font (Times New Roman, Helvetica, Arial)
	plt.rcParams['xtick.direction'] = 'in'   # directions of x ticks ('in'), ('out') or ('inout')
	plt.rcParams['ytick.direction'] = 'in'   # directions of y ticks ('in'), ('out') or ('inout')
	plt.rcParams['font.size']       = csize  # fontsize


	# setting output file name & format
	outname = outname if outname else self.file.replace('.fits', '_intensitymap')

	if (outformat == formatlist).any():
		outname = outname + '.' + outformat
	else:
		print ('ERROR\tintensitymap: Outformat is wrong.')
		return

	if inmode == 'data':
		if data is None:
			print ("inmode ='data' is selected. data must be provided.")
			return

		naxis = len(data.shape)
	else:
		data  = self.data
		naxis = self.naxis


	if self.beam is None:
		plot_beam = False
	else:
		bmaj, bmin, bpa = self.beam


	if coord_center:
		self.shift_coord_center(coord_center)


	# coordinate style
	if relativecoords:
		xx = self.xx
		yy = self.yy
		cc = self.cc
		xlabel = 'RA offset (arcsec)'
		ylabel = 'DEC offset (arcsec)'
	else:
		print ('WARNING: Abusolute coordinates are still in development.')
		xlabel = self.label_i[0]
		ylabel = self.label_i[1]
		xx = self.xx
		yy = self.yy
		cc = self.cc


	# check data axes
	if len(data.shape) == 2:
		pass
	elif len(data.shape) == 3:
		data = data[axis,:,:]
	elif len(data.shape) == 4:
		data = data[0,axis,:,:]
	else:
		print ('Error\tintensitymap: Input fits size is incorrect.\
		 Must have 2 to 4 axes. Check the shape of the input image.')
		return

	# deg --> arcsec
	xx     = xx*3600.
	yy     = yy*3600.

	# figure extent
	xmin   = xx[0,0]
	xmax   = xx[-1,-1]
	ymin   = yy[0,0]
	ymax   = yy[-1,-1]
	del_x  = xx[1,1] - xx[0,0]
	del_y  = yy[1,1] - yy[0,0]
	extent = (xmin-0.5*del_x, xmax+0.5*del_x, ymin-0.5*del_y, ymax+0.5*del_y)
	#print (extent)

	# image scale
	if len(imscale) == 0:
		figxmin, figxmax, figymin, figymax = extent  # arcsec

		if figxmin < figxmax:
			cp      = figxmax
			figxmax = figxmin
			figxmin = cp

		if figymin > figymax:
			cp = figymax
			figymax = figymin
			figymin = cp
	elif len(imscale) == 4:
		figxmax, figxmin, figymin, figymax = imscale # arcsec
	else:
		print ('ERROR\tIdistmap: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')


	# !!!!! plot !!!!!
	# setting figure
	if ax is not None:
		pass
	else:
		fig = plt.figure(figsize=figsize)
		ax  = fig.add_subplot(111)


	# set colorscale
	if vmax is None: vmax = np.nanmax(data)

	# color scale
	if type(color_norm) == str:
		if color_norm == 'log':
			norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
	elif type(color_norm) == tuple:
		if hasattr(color_norm[0], '__call__'):
			norm = mpl.colors.FuncNorm(color_norm, vmin=vmin, vmax=vmax)
		else:
			print ('ERROR\tintensitymap: color_norm must be strings or tuple of functions.')
	else:
		norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)


	# color map
	if color:
		if exact_coord:
			imcolor = ax.pcolor(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)
		else:
			imcolor = ax.imshow(data, cmap=cmap, origin='lower', extent=extent, norm=norm, interpolation=interpolation, rasterized=True)

		# color bar
		if colorbar:
			cbar_loc, cbar_wd, cbar_pad, cbar_lbl = cbaroptions
			divider = make_axes_locatable(ax)
			cax     = divider.append_axes(cbar_loc, size=cbar_wd, pad=cbar_pad)
			cbar    = plt.colorbar(imcolor, cax=cax )#, ax = ax, orientation=cbar_loc, aspect=float(cbar_wd), pad=float(cbar_pad))
			cbar.set_label(cbar_lbl)

	# contour map
	if contour:
		if exact_coord:
			imcont = ax.contour(xx, yy, data, colors=ccolor,
				origin='lower', levels=clevels, linewidths=clw)
		else:
			imcont = ax.contour(data, colors=ccolor, origin='lower', levels=clevels,
				linewidths=clw, extent=(xmin,xmax,ymin,ymax))


	# set axes
	ax.set_xlim(figxmin,figxmax)
	ax.set_ylim(figymin,figymax)
	ax.set_xlabel(xlabel,fontsize=csize)
	ax.set_ylabel(ylabel, fontsize=csize)
	if len(xticks) != 0:
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticks)

	if  len(yticks) != 0:
		ax.set_yticks(yticks)
		ax.set_yticklabels(yticks)

	ax.set_aspect(1)
	ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, labelsize=csize, color=tickcolor, labelcolor=labelcolor, pad=9)

	# plot beam size
	if plot_beam:
		bmin_plot, bmaj_plot = ax.transLimits.transform((bmin,bmaj)) - ax.transLimits.transform((0,0))   # data --> Axes coordinate
		beam = patches.Ellipse(xy=(0.1, 0.1), width=bmin_plot, height=bmaj_plot, fc=bcolor, angle=bpa, transform=ax.transAxes)
		ax.add_patch(beam)

	# central star position
	if cstar:
		if len(prop_star) == 0:
			ll = 0.1*np.abs(figxmax - figxmin)
			lw = 1.
			cl = 'k'
			if relativecoords:
				pos_cstar = np.array([0,0])
			else:
				pos_cstar = cc
		elif len(prop_star) == 3:
			ll, lw, cl = prop_star
			ll = float(ll)
			lw = float(lw)
			if relativecoords:
				pos_cstar = np.array([0,0])
			else:
				pos_cstar = cc
		elif len(prop_star) == 4:
			ll, lw, cl, pos_cstar = prop_star
			ll = float(ll)
			lw = float(lw)
			ra_st, dec_st = pos_cstar.split(' ')
			radec_st     = SkyCoord(ra_st, dec_st, frame='icrs')
			ra_stdeg      = radec_st.ra.degree                     # in degree
			dec_stdeg     = radec_st.dec.degree                    # in degree
			if relativecoords:
				pos_cstar = np.array([(ra_stdeg - cc[0])*3600., (dec_stdeg - cc[1])*3600.])
			else:
				pos_cstar = np.array([ra_stdeg, dec_stdeg])
		else:
			print ('ERROR\tIdistmap: prop_star must be size of 3 or 4.')
			return

		ax.hlines(pos_cstar[1], pos_cstar[0]-ll*0.5, pos_cstar[0]+ll*0.5, lw=lw, color=cl,zorder=11)
		ax.vlines(pos_cstar[0], pos_cstar[1]-ll*0.5, pos_cstar[1]+ll*0.5, lw=lw, color=cl,zorder=11)


	# scale bar
	if len(scalebar) == 0:
		pass
	elif len(scalebar) == 8:
		barx, bary, barlength, textx, texty, text, colors, barcsize = scalebar

		barx      = float(barx)
		bary      = float(bary)
		barlength = float(barlength)
		textx     = float(textx)
		texty     = float(texty)

		ax.hlines(bary, barx - barlength*0.5, barx + barlength*0.5, lw=2, color=colors,zorder=10)
		ax.text(textx, texty, text, color=colors, fontsize=barcsize,
		 horizontalalignment='center', verticalalignment='center')
	else:
		print ('The parameter scalebar must have 8 elements but does not.')

	plt.savefig(outname, transparent = True)

	return ax


# Channel maps
def channelmaps(self, grid=None, data=None, outname=None, outformat='pdf', imscale=[], color=False, cbaron=False, cmap='Blues', vmin=None, vmax=None,
	contour=True, clevels=np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]), ccolor='k',
	nrow=5, ncol=5,velmin=None, velmax=None, nskip=1, cw=0.5, color_norm=None,
	xticks=[], yticks=[], relativecoords=True, vsys=None, csize=14, scalebar=np.empty(0),
	cstar=True, prop_star=[], tickcolor='k', axiscolor='k',
	labelcolor='k',cbarlabel=None, txtcolor='k', bcolor='k', figsize=(11.69,8.27),
	cbarticks=None, coord_center=None, noreg=True, arcsec=True, sbar_vertical=False,
	cbaroptions=np.array(['right','5%','0%']), inmode='fits', vlabel_on=True):
	'''
	Make channel maps from a fits file.


	Usage (examples)
	 Draw a map:
	  channelmap('object.co.fits', outname='test', imscale=[-10,10,-10,10],
	   color=True, cmap='Greys', velmin=5.2, velmax=12.5, nrow=5, ncol=8)

	 Overplot:
	  rms = 10. # rms
	  grid = channelmap('object.co.fits', outname='test', imscale=[-10,10,-10,10], color=True)
	  grid = channelmap('object.13co.fits', outname='test', grid=grid, color=False,
	   contour=True, clevels=np.array([3,6,9,12])*rms) # put grid=grid, the same outname

	  # If you want to overplot something more
	  grid[nrow*(ncol-1)].plot(0, 0, marker='*', size=40) # overplot
	  plt.savefig('test.pdf', transparent=True)           # use the same name


	Args
	fitsdata: Input fitsdata. It must be an self cube having 3 or 4 axes.
	outname: Output file name. Do not include file extension.
	outformat: Extension of the output file. Default is eps.
	imscale: scale to be shown (arcsec). It must be given as [xmin, xmax, ymin, ymax].
	color (bool): If True, images will be shown in colorscale. Default is False.
	    cmap: color of the colorscale.
	    vmin: Minimum value of colorscale. Default is None.
	    vmax: Maximum value of colorscale. Default is the maximum value of the self cube.
	    logscale (bool): If True, the color will be shown in logscale.
	contour (bool): If True, images will be shown with contour. Default is True.
	    clevels (ndarray): Contour levels. Input will be treated as absolute values.
	    ccolor: color of contour.
	nrow, ncol: the number of row and column of the channel map.
	relativecoords (bool): If True, the channel map will be produced in relative coordinate. Abusolute coordinate mode is (probably) coming soon.
	velmin, velmax: Minimum and maximum velocity to be shown.
	vsys: Systemic velicity [km/s]. If no input value, velocities will be described in LSRK.
	csize: Caracter size. Default is 9.
	cstar: If True, a central star or the center of an self will be shown as a cross.
	prop_star: Detailed setting for the cross showing stellar position.
	 np.array(['length','width','color']) or np.array(['length','width','color', 'coordinates']).
	logscale (bool): If True, the color scale will be in log scale.
	coord_center (str): Put an coordinate for the map center. The shape must be '00h00m00.00s 00d00m00.00s', or
	 'hh:mm:ss.ss dd:mm:ss.ss'. RA and DEC must be separated by space.
	locsym: Removed. A factor to decide locations of symbols (beam and velocity label). It must be 0 - 1.
	tickcolor, axiscolor, labelcolor, txtcolor: Colors for the maps.
	scalebar (array): If it is given, scalebar will be drawn. It must be given as [barx, bary, bar length, textx, texty, text].
	                   Barx, bary, textx, and texty are locations of a scalebar and a text in arcsec.
	nskip: the number of channel skipped
	'''

	# modules
	import matplotlib.figure as figure
	import matplotlib as mpl
	#from mpl_toolkits.mplot3d import axes3d
	from astropy.coordinates import SkyCoord
	import matplotlib.patches as patches
	from mpl_toolkits.axes_grid1 import ImageGrid

	# format
	formatlist = np.array(['eps','pdf','png','jpeg'])

	# properties of plots
	#mpl.use('Agg')
	plt.rcParams['font.family']     ='Arial' # font (Times New Roman, Helvetica, Arial)
	plt.rcParams['xtick.direction'] = 'in'   # directions of x ticks ('in'), ('out') or ('inout')
	plt.rcParams['ytick.direction'] = 'in'   # directions of y ticks ('in'), ('out') or ('inout')
	plt.rcParams['font.size']       = csize  # fontsize


	# Setting output file name & format
	if (outformat == formatlist).any():
		#outfile = outname + '_nmap{0:02d}'.format(nmap) + '.' + outformat
		outfile = outname + '.' + outformat
	else:
		print ('ERROR\tdraw_channelmaps: Outformat is wrong.')
		return

	if inmode == 'data':
		if data is None:
			print ("inmode ='data' is selected. data must be provided.")
			return

		naxis = len(data.shape)
	else:
		data  = self.data
		header = self.header
		naxis = self.naxis

	# number of axis
	if naxis < 3:
		print ('ERROR\tdraw_channelmaps: NAXIS of fits is < 3 although It must be > 3.')
		return

	if self.beam is None:
		plot_beam = False
	else:
		bmaj, bmin, bpa = self.beam


	# coordinate center
	if coord_center:
		self.shift_coord_center(coord_center)


	# Coordinates
	if relativecoords:
		xx = self.xx
		yy = self.yy
		cc = self.cc
		xlabel = 'RA offset (arcsec)'
		ylabel = 'DEC offset (arcsec)'
	else:
		print ('WARNING: Abusolute coordinates are still in development.')
		xlabel = self.label_i[0]
		ylabel = self.label_i[1]
		xx = self.xx
		yy = self.yy
		cc = self.cc



	# check data axes
	if len(data.shape) == 3:
		pass
	elif len(data.shape) == 4:
		data = data[0,:,:,:]
	else:
		print ('Error\tchannelmaps: Input fits size is not corrected.\
		 It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
		return

	# unit: arcsec or deg
	if arcsec:
		xx     = xx*3600.
		yy     = yy*3600.


	# Get velocity axis
	vaxis = self.vaxis
	delv  = self.delv
	nchan = self.naxis_i[2]

	if delv < 0:
		delv  = - delv
		vaxis = vaxis[::-1]
		data  = data[::-1,:,:]


	# Figure extent
	xmin   = xx[0,0]
	xmax   = xx[-1,-1]
	ymin   = yy[0,0]
	ymax   = yy[-1,-1]
	del_x  = xx[1,1] - xx[0,0]
	del_y  = yy[1,1] - yy[0,0]
	extent = (xmin-0.5*del_x, xmax+0.5*del_x, ymin-0.5*del_y, ymax+0.5*del_y)
	#print (extent)


	# Relative velocity
	if vsys:
		vaxis = vaxis - vsys


	# Set colorscale
	vmax = vmax if vmax is not None else np.nanmax(data)
	vmin = vmin if vmin is not None else np.nanmin(data)


	# color scale
	if type(color_norm) == str:
		if color_norm == 'log':
			norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
	elif type(color_norm) == tuple:
		if hasattr(color_norm[0], '__call__'):
			norm = mpl.colors.FuncNorm(color_norm, vmin=vmin, vmax=vmax)
		else:
			print ('ERROR\tintensitymap: color_norm must be strings or tuple of functions.')
	else:
		norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)


	# !!! Plot !!!
	# Setting colorbar
	cbar_mode = 'single' if color and cbaron else None


	if grid:
		nrow = grid._nrows
		ncol = grid._ncols
	else:
		fig = plt.figure(figsize=figsize)
		 # Setting grid
		cbar_loc, cbar_wd, cbar_pad = cbaroptions
		grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
			axes_pad=0,share_all=True,cbar_mode=cbar_mode,
			cbar_location=cbar_loc,cbar_size=cbar_wd,cbar_pad=cbar_pad,
			label_mode='1')


	# Setting parameters used to plot
	if len(imscale) == 0:
		figxmin, figxmax, figymin, figymax = extent
	elif len(imscale) == 4:
		figxmax, figxmin, figymin, figymax = imscale
	else:
		print ('ERROR\tchannelmaps: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')


	# data for plot
	if velmax:
		d_plt = data[vaxis <= velmax,:,:]
		v_plt = vaxis[vaxis <= velmax]
		nv_plt = len(v_plt)
	else:
		d_plt = data.copy()
		v_plt = vaxis.copy()
		nv_plt = len(v_plt)

	if velmin:
		d_plt  = d_plt[v_plt >= velmin,:,:]
		v_plt  = v_plt[v_plt >= velmin]
		nv_plt = len(v_plt)

	if nskip:
		d_plt  = d_plt[::nskip,:,:]
		v_plt  = v_plt[::nskip]
		nv_plt = len(v_plt)

	# Counter
	i, j, gridi = [0,0,0]
	gridimax    = nrow*ncol-1
	ax    = None

	# Loop
	for ichan in range(nv_plt):
		# maximum grid
		if gridi > gridimax:
			break

		# Select channel
		Sv = d_plt[ichan,:,:]

		# velocity at nchan
		v_i = v_plt[ichan]

		# Axis
		ax = grid[gridi]
		print ('Channel ', '%s'%ichan, ', velocity: ', '%4.2f'%v_i, ' km/s')

		# showing in color scale
		if color:
			imcolor = ax.imshow(Sv, cmap=cmap, origin='lower', extent=extent, norm=norm, rasterized=True)

		if contour:
			imcont  = ax.contour(Sv, colors=ccolor, origin='lower',extent=extent, levels=clevels, linewidths=cw)

		# set axes
		ax.set_xlim(figxmin,figxmax)
		ax.set_ylim(figymin,figymax)
		ax.spines["bottom"].set_color(axiscolor)
		ax.spines["top"].set_color(axiscolor)
		ax.spines["left"].set_color(axiscolor)
		ax.spines["right"].set_color(axiscolor)
		if len(xticks) != 0:
			ax.set_xticks(xticks)

		if len(yticks) != 0:
			ax.set_yticks(yticks)


		ax.set_aspect(1)
		ax.tick_params(which='both', direction='in',bottom=True,
		 top=True, left=True, right=True, color=tickcolor,
		  labelcolor=labelcolor, pad=9, labelsize=csize)

		# Velocity label
		if vlabel_on:
			vlabel = '%3.2f'%v_i
			ax.text(0.1, 0.9,vlabel,color=txtcolor,size=csize,
			 horizontalalignment='left', verticalalignment='top',
			  transform=ax.transAxes)

		# Central star position
		if cstar:
			if len(prop_star) == 0:
				ll = 0.1*np.abs(figxmax - figxmin)
				lw = 1.
				cl = 'k'
				pos_cstar = np.array([0,0]) if relativecoords else cc
			elif len(prop_star) == 3:
				ll,lw, cl = prop_star
				ll = float(ll)
				lw = float(lw)

				pos_cstar = np.array([0,0]) if relativecoords else cc
			else:
				print ('WARNING\tchannelmaps: The parameter prop_star\
					must have 3 elements but does not. The input is ignored.')

			ax.hlines(pos_cstar[1], pos_cstar[0]-ll*0.5, pos_cstar[0]+ll*0.5, lw=lw, color=cl,zorder=11)
			ax.vlines(pos_cstar[0], pos_cstar[1]-ll*0.5, pos_cstar[1]+ll*0.5, lw=lw, color=cl,zorder=11)

		gridi += 1


	# On the bottom-left corner pannel
	# Labels
	grid[(nrow-1)*ncol].set_xlabel(xlabel)
	grid[(nrow-1)*ncol].set_ylabel(ylabel)
	grid[(nrow-1)*ncol].xaxis.label.set_color(labelcolor)
	grid[(nrow-1)*ncol].yaxis.label.set_color(labelcolor)

	# Plot beam
	bmin_plot, bmaj_plot = grid[(nrow-1)*ncol].transLimits.transform((0,bmaj)) - grid[(nrow-1)*ncol].transLimits.transform((bmin,0))   # data --> Axes coordinate
	beam = patches.Ellipse(xy=(0.1, 0.1), width=bmin_plot, height=bmaj_plot, fc=bcolor, angle=bpa, transform=grid[(nrow-1)*ncol].transAxes)
	grid[(nrow-1)*ncol].add_patch(beam)

	# Scale bar
	if len(scalebar) == 0:
		pass
	elif len(scalebar) == 8:
		barx, bary, barlength, textx, texty, text, barcolor, barcsize = scalebar

		barx      = float(barx)
		bary      = float(bary)
		barlength = float(barlength)
		textx     = float(textx)
		texty     = float(texty)

		if sbar_vertical:
			grid[(nrow-1)*ncol].vlines(barx, bary - barlength*0.5,bary + barlength*0.5, color=barcolor, lw=2, zorder=10)
		else:
			grid[(nrow-1)*ncol].hlines(bary, barx - barlength*0.5,barx + barlength*0.5, color=barcolor, lw=2, zorder=10)

		grid[(nrow-1)*ncol].text(textx,texty,text,color=barcolor,fontsize=barcsize,horizontalalignment='center',verticalalignment='center')
	else:
		print ('scalebar must consist of 8 elements. Check scalebar.')



	if color and cbaron and ax:
		# With cbar_mode="single", cax attribute of all axes are identical.
		cax = grid.cbar_axes[0]
		cbar = plt.colorbar(imcolor, ticks=cbarticks, cax=cax)
		#cbar = cax.colorbar(imcolor,ticks=cbarticks)
		cax.toggle_label(True)
		cbar.ax.yaxis.set_tick_params(color=tickcolor) # tick color
		cbar.ax.spines["bottom"].set_color(axiscolor)  # axes color
		cbar.ax.spines["top"].set_color(axiscolor)
		cbar.ax.spines["left"].set_color(axiscolor)
		cbar.ax.spines["right"].set_color(axiscolor)

		if cbarlabel:
			cbar.ax.set_ylabel(cbarlabel,color=labelcolor) # label


	if gridi != gridimax+1 and gridi != 0:
		while gridi != gridimax+1:
			#print gridi
			ax = grid[gridi]
			ax.spines["right"].set_color("none")  # right
			ax.spines["left"].set_color("none")   # left
			ax.spines["top"].set_color("none")    # top
			ax.spines["bottom"].set_color("none") # bottom
			ax.axis('off')
			gridi = gridi+1

	plt.savefig(outfile, transparent = True)

	return grid


# Draw pv diagram
def pvdiagram(self,outname,data=None,header=None,ax=None,outformat='pdf',color=True,cmap='Blues',
	vmin=None,vmax=None,vsys=0,contour=True,clevels=None,ccolor='k', pa=None,
	vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
	lw=1,clip=None,plot_res=True,inmode='fits',xranges=[], yranges=[],
	ln_hor=True, ln_var=True, alpha=None, colorbar=False,
	cbaroptions=('right', '3%', '0%', r'(Jy beam$^{-1}$)')):
	'''
	Draw a PV diagram.

	Args:
	 - outname:
	'''

	# Modules
	import copy
	import matplotlib as mpl
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
		data   = np.squeeze(self.data)
		header = self.header
		naxis  = self.naxis


	# figures
	if ax:
		pass
	else:
		fig = plt.figure(figsize=(11.69,8.27)) # figsize=(11.69,8.27)
		ax  = fig.add_subplot(111)

	# Read
	xaxis = self.xaxis.copy()
	vaxis = self.vaxis.copy()
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


	# set extent of an self
	offmin = xaxis[0] - delx*0.5
	offmax = xaxis[-1] + delx*0.5
	velmin = vaxis[0] - delv*0.5
	velmax = vaxis[-1] + delv*0.5


	# set axes
	if x_offset:
		extent = (offmin,offmax,velmin,velmax)
		xlabel = offlabel
		ylabel = vlabel
		hline_params = [vsys,offmin,offmax]
		vline_params = [0.,velmin,velmax]
		res_x = res_off
		res_y = delv
	else:
		data   = data.T
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

		# color bar
		if colorbar:
			cbar_loc, cbar_wd, cbar_pad, cbar_lbl = cbaroptions
			if cbar_loc != 'right':
				print ('WARNING\tpvdiagram: only right is supported for \
					colorbar location. Your input is ignored.')
			axin_cb = inset_axes(ax,
				width=cbar_wd,
				height='100%',
				loc='lower left',
				bbox_to_anchor=(1.0 + float(cbar_pad.strip('%'))*0.01, 0., 1., 1.),
				bbox_transform=ax.transAxes,
				borderpad=0)
			cbar = plt.colorbar(imcolor, cax=axin_cb)
			cbar.set_label(cbar_lbl)
			#divider = make_axes_locatable(ax)
			#cax     = divider.append_axes(cbar_loc, size=cbar_wd, pad=cbar_pad)
			#cbar    = plt.colorbar(imcolor, cax=cax )#, ax = ax, orientation=cbar_loc, aspect=float(cbar_wd), pad=float(cbar_pad))
			#cbar.set_label(cbar_lbl)

	if contour:
		imcont  = ax.contour(data, colors=ccolor, origin='lower',
			extent=extent, levels=clevels, linewidths=lw, alpha=alpha)


	# axis labels
	ax.set_xlabel(xlabel, fontsize=fontsize)
	ax.set_ylabel(ylabel, fontsize=fontsize)

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

	ax.tick_params(which='both', direction='in',
		bottom=True, top=True, left=True, right=True,
		pad=9, labelsize=fontsize)

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


def generate_grid(nrow, ncol, figsize=(11.69, 8.27),
	cbar_mode=None, axes_pad=(0.2, 0.2), share_all=True,
	cbaroptions=['right', '3%', '0'], label_mode='1'):
	'''
	Generate grid to contain multiple figures. Just using ImageGrid of matplotlib but adjust default parameters for convenience.
	 For more detail of the function, check https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.axes_grid.ImageGrid.html.

	Parameters
	----------
	 nrow(int): Number of rows
	 ncol(int): Number of columns
	 axes_pad (tuple or float): Padding between axes. In cases that tuple is given, it will be treated as (vertical, horizontal) padding.
	 share_all (bool): Whether all axes share their x- and y-axis.
	 cbarmode: If each, colorbar will be shown in each axis. If single, one common colorbar will be prepared. Default None.
	'''

	fig = plt.figure(figsize=figsize)

	# Generate grid
	cbar_loc, cbar_wd, cbar_pad = cbaroptions
	grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
		axes_pad=axes_pad, share_all=share_all, cbar_mode=cbar_mode,
		cbar_location=cbar_loc,cbar_size=cbar_wd, cbar_pad=cbar_pad,
		label_mode=label_mode)

	return grid