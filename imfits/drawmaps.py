# -*- coding: utf-8 -*-
'''
Made and developed by J.Sai.

email: jn.insa.sai@gmail.com
'''



### Modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import make_lupton_rgb



pltconfig_def = {
'xtick.direction': 'in',
'ytick.direction': 'in',
'font.size': '12',
}

pltconfig_darkbg = {
'tickcolor': 'white',
'axiscolor': 'white',
'txtcolor': 'white',
'bcolor': 'white',
'ccolor': 'white',
'labelcolor': 'k'}

figsize_def = (11.69, 8.27)



class AstroCanvas():
    '''
    A Python class to maintain figures from fits images/cubes.

    '''

    def __init__(self, 
        nrow_ncol: tuple = (0, 0),
        axes_pad: tuple = (0.2, 0.2),
        fig = None,
        imagegrid: bool = False,
        cbar_mode: str = None, 
        cbaroptions = ['right', '3%', '0%'],
        label_mode = '1',
        figsize: tuple = figsize_def,
        pltconfig: dict = pltconfig_def) -> None:
        '''
        Initialize and set a canvas.

        Parameters
        ----------
         - nrow_ncol (tuple): Number of rows and columns. Must be given in a format of 
           (n_rows, n_columns).
         - axes_pad (tuple): Padding between each pannel. Must be given in a format of
           (horizontal padding, vertical padding). Each padding can be specified by a number 
           from zero to one.
         - fig (matplotlib object): Figure object of matplotlib if you want to give by yourself.
           AstroCanvas will generate if nothing is given.
         - imagegrid (bool): Use ImageGrid of mpl_toolkits or not. It can be used for 
           better spacing between pannels.
         - cbar_mode (str): Only used when imagegrid=True to specify a colorbar mode. Must be
           chosen from 'each', 'signle', 'edge', or None.
         - cbaroptions (list or tuple): Only used when imagegrid=True to setup a colorbar(s).
           Must be given in a format of [location, width, padding]. The location can be 'right', 'top'
           'left', 'bottom'. The width and padding can be given as a percentage of the axis width
           in str (e.g., '2%').
         - label_mode (str): Only used when imagegrid=True to setup axes labels. Can be either of
           'L', '1', or 'all'.
         - figsize (tuple): Figure size. Default is (11.69, 8.27), which is A4 landscape.
         - pltconfig (dict): Dictionary of matplotlib rcParams to change the plotting style.
        '''
        # style
        self.style(pltconfig)

        # set figure
        nrow, ncol = nrow_ncol
        self.grid = imagegrid
        if fig is not None:
            self.axes = fig.axes
            self.naxes = len(self.axes)
        else:
            fig = plt.figure(figsize=figsize)
            self.fig = fig
            self.naxes = 0

            # axes
            if (np.array([nrow, ncol]) == 1).all():
                self.axes = [fig.add_subplot(111)] # axes
                self.naxes = len(self.axes)
            elif (nrow >= 1) & (ncol >= 1):
                if imagegrid:
                    self.add_grid(nrow, ncol, 
                        cbar_mode = cbar_mode, axes_pad = axes_pad,
                        cbaroptions = cbaroptions, label_mode = label_mode)
                else:
                    self.add_axes(nrow, ncol, 
                        wspace=axes_pad[0], hspace=axes_pad[1])
            else:
                self.axes = []
                self.naxes = 0
                self.nrow = 0
                self.ncol = 0


    def reset_axes(self):
        print('Axes exist in Canvas. Are you sure to reset them?')
        while (ans == 'y') | (ans == 'n'):
            print('Type [y/n].')
            ans = input()

            if ans == 'y':
                self.axes = []
                self.naxes = 0
                self.nrow = 0
                self.ncol = 0
                return 1
            elif ans == 'n':
                return 0


    def add_axis(self):
        if self.naxes == 0:
            self.axes = [fig.add_subplot(111)] # axes
            self.naxes = len(self.axes)
            self.nrow = 1
            self.ncol = 1
            return axes
        else:
            if self.reset_axes:
                self.axes = [fig.add_subplot(111)] # axes
                self.naxes = len(self.axes)
                self.nrow = 1
                self.ncol = 1
                return axes
            return None


    def add_axes(self, nrow, ncol,
        wspace=None, hspace=None):
        import itertools
        if self.naxes == 0:
            gs = GridSpec(nrows=nrow, ncols=ncol, figure=self.fig,
                wspace=wspace, hspace=hspace)
            axes = []
            for irow, icol in itertools.product(range(nrow), range(ncol)):
                ax = self.fig.add_subplot(gs[irow, icol])
                axes.append(ax)
            self.axes = axes
            self.naxes = len(self.axes)
            self.nrow = nrow
            self.ncol = ncol
            return axes
        else:
            if self.reset_axes:
                gs = GridSpec(nrows=nrow, ncols=ncol, figure=self.fig,
                    wspace=wspace, hspace=hspace)
                axes = []
                for irow, icol in itertools.product(range(nrow), range(ncol)):
                    ax = self.fig.add_subplot(gs[irow, icol])
                    axes.append(ax)
                self.axes = axes
                self.naxes = len(self.axes)
                self.nrow = nrow
                self.ncol = ncol
                return axes
            return None



    def style(self, config):
        #for i in config.keys():
        #mpl.rc(i, config[i])
        self._config = config
        plt.rcParams.update(config)


    def add_grid(self, nrow, ncol, 
        cbar_mode=None, axes_pad=(0., 0.), 
        share_all=True, 
        cbaroptions=['right', '3%', '0%'], 
        label_mode='1'):
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

        # Generate grid
        cbar_loc, cbar_wd, cbar_pad = cbaroptions
        grid = ImageGrid(self.fig, rect=111, nrows_ncols=(nrow,ncol),
            axes_pad=axes_pad, share_all=share_all, cbar_mode=cbar_mode,
            cbar_location=cbar_loc,cbar_size=cbar_wd, cbar_pad=cbar_pad,
            label_mode=label_mode)
        self.axes = grid
        self.naxes = len(self.axes)
        self.nrow = nrow
        self.ncol = ncol
        # to retrieve
        self._gridkwargs = {
        'axes_pad': axes_pad,
        'share_all': share_all,
        'cbar_mode': cbar_mode,
        'cbar_location': cbar_loc,
        'cbar_size': cbar_wd,
        'cbar_pad': cbar_pad,
        'label_mode': label_mode,
        }
        self.grid = True

        return grid


    def ax_corner(self, loc='lower left'):
        '''
        Return axis at the corner.

        Parameter
        ---------
        loc (str): the corner location to be extracted. 'upper/lower left/right'.
        '''
        loc = loc.replace('bottom', 'lower')
        loc = loc.replace('top', 'upper')
        if loc == 'lower left':
            indx = self.ncol * (self.nrow - 1)
        elif loc == 'lower right':
            indx = -1
        elif loc == 'upper left':
            indx = 0
        elif loc == 'upper right':
            indx = self.ncol - 1
        else:
            print('ERROR\tax_corner: the input location is wrong.')
            print('ERROR\tax_corner: must be upper left, upper right, \
                lower left, or lower right.')
            return 0

        return self.axes[indx]


    def savefig(self, outname: str, 
        ext : str = 'pdf', transparent: bool = True, 
        dpi = None):
        self.fig.savefig(outname + '.' + ext, transparent=transparent, dpi = dpi)


    def intensitymap(self, image, imscale=[], 
        color=True, cmap='PuBuGn', colorbar=True, cbarlabel=None, 
        cbaroptions=np.array(['right','3%','0%']), 
        vmin=None, vmax=None, color_norm=None,
        contour=False, clevels=None, ccolor='k', clw=1, 
        xticks=[], yticks=[], absolutecoords=False, scalebar=[],
        plot_beam=True, bcolor='k',
        ccross=True, prop_cross=[None, 1., 'k'], 
        coord_center=None, aspect=1,
        interpolation=None, exact_coord=False,
        iaxis=0, saxis=0, vaxis=0, inmode=None, data=None, outname=None,
        transparent = True, color_alpha = 1):
        '''
        Draw the intensity map.


        Parameters
        ----------
        image (Imfits): Input image.
        iaxis (int): Index of the AstroCanvas axes, i.e., index of the panel where you plot.
        imscale (list): Extent of the plot. Must be given as [xmin, xmax, ymin, ymax],
            where each parameter is the minimum or maximum value of the axis.
        color (bool): If plot in color or not.
        cmap (str): Color map.
        colorbar (bool): If plot colorbar or not.
        cbaroptions (list): Setting for colorbar. Must be give as [location, width, pad].
            The location must be 'top', 'right', 'bottom', or 'left'. 
            The width and padding can be specified by parcentage of the extent of the plot.
            E.g., ['top', '3%', '3%'].
        vmin, vmax (float): Minimun and maximun values of the color scale.
        contour (bool): If plot with contours or not.
        clevels (ndarray): Contour levels. Must be give as absolute values.
        ccolor (str): Contour color.
        clw (float): Contour widths
        color_norm (str or tuple): Colorscale normalization. Default is linear.
            Currently-supported formats are 'log', ('asinhstretch', a), or (func, func_inverse), 
            where func1 and func2 are arbitoral functions which will determine 
            the colorscale. For the last option, refer mpl.colors.FuncNorm for more details.
        vaxis (int): Index of the 3rd (typically velocity) axis, if the input image is in three dimension.
        saxis (int): Index of the 4th (typically stokes) axis, if the input image is in four dimension.
        xticks, yticks (list): Ticks for x and y axis.
        absolutecoords (bool): Make plots with absolute coordinates if True. 
            This option is not fully supported, and may result in weired plot axes.
        scalebar (list): Add scale bar. Must be given as [] or [].
        plot_beam (bool): Plot observing beam or not.
        bcolor (k): Color for the beam plot.
        ccross (bool): If plot a cross at the map center or not.
        prop_cross (list): Setting of the central cross. Must be give as [length, width, color].
            E.g., [2., 2., 'k'].
        coord_center (str): New coordinate center. Must be given in
            a format of '00h00m00.00s +00d00m00.00s'.
        aspect (float): Aspect ratio of the figure.
        interpolation (str): Interpolation style for the color plot.
        exact_coord (bool): If True, pcolor is used rather than imshow. It will return
            correct results when grid spacing of the input image is not uniform.
        inmode (str): If 'data', then array given in the 'data' parameter will be used for plot
            while other info like plot extent is from the input image.
        data (ndarray): Data to be plotted instead of the input Imfits image.
        outname (str): If given, the figure will be saved with the output file name.
        transparent (bool): Make the background transparent or not.
        '''
        # axis
        ax = self.axes[iaxis]


        # data
        if inmode == 'data':
            if data is None:
                print ("ERROR\tintensitymap: data cannot be found.")
                print ("ERROR\tintensitymap: data must be provided when inmode ='data'.")
                print ("ERROR\tintensitymap: Use data of the input image.")
                return 0
            naxis = len(data.shape)
        else:
            data  = image.data.copy()
            naxis = image.naxis


        # check data axes
        if len(data.shape) == 2:
            pass
        elif len(data.shape) == 3:
            data = data[vaxis,:,:]
        elif len(data.shape) == 4:
            data = data[saxis, vaxis, :, :]
        else:
            print ('ERROR\tintensitymap: Input fits size is incorrect.\
             Must have 2 to 4 axes. Check the shape of the input image.')
            return


        # unit
        if cbarlabel is None:
            cbarlabel = '(' + image.header['BUNIT'] + ')' if 'BUNIT' in image.header else ''


        # beam
        if image.beam is None:
            plot_beam = False
        else:
            bmaj, bmin, bpa = image.beam


        # center
        if coord_center:
            image.shift_coord_center(coord_center)


        # coordinate style
        if absolutecoords:
            print ('WARNING\tintensitymap: Abusolute coordinate option is still in development.')
            print ('WARNING\tintensitymap: May result in a bad-looking plot.')
            xlabel = image.label_i[0]
            ylabel = image.label_i[1]
            cc    = image.cc
            xaxis = image.xx_wcs[image.ny//2,:] # not exactly though
            yaxis = image.yy_wcs[:, image.nx//2]
        else:
            cc    = image.cc
            xaxis = image.xaxis*3600.
            yaxis = image.yaxis*3600.
            xlabel = 'RA offset (arcsec)'
            ylabel = 'DEC offset (arcsec)'


        # trim data for plot
        if len(imscale) == 0:
            xmin, xmax = xaxis[[0,-1]]
            ymin, ymax = yaxis[[0,-1]]
            extent = (xmin-0.5*image.delx, xmax+0.5*image.delx, 
                ymin-0.5*image.dely, ymax+0.5*image.dely)
            figxmin, figxmax, figymin, figymax = extent
            data, xaxis, yaxis = trim_data(data, xaxis, yaxis, [],
             [figxmax, figxmin], [figymin, figymax], [])
        elif len(imscale) == 4:
            figxmax, figxmin, figymin, figymax = imscale
            data, xaxis, yaxis = trim_data(data, xaxis, yaxis, [],
             [figxmax, figxmin], [figymin, figymax], [])
            xmin, xmax = xaxis[[0,-1]]
            ymin, ymax = yaxis[[0,-1]]
            extent = (xmin-0.5*image.delx, xmax+0.5*image.delx, 
                ymin-0.5*image.dely, ymax+0.5*image.dely)
        else:
            print ('ERROR\tintensitymap: Input imscale is wrong.\
             Must be [xmin, xmax, ymin, ymax]')


        # meshgrid if needed
        if exact_coord:
            if absolutecoords:
                xx, yy = image.xx_wcs.copy(), image.yy_wcs.copy()
            else:
                xx, yy  = np.meshgrid(xaxis, yaxis)


        # set colorscale
        if vmax is None: vmax = np.nanmax(data)


        # color scale
        norm = color_normalization(color_norm, vmin, vmax)


        # color map
        if color:
            if exact_coord:
                imcolor = ax.pcolor(xx, yy, data, cmap=cmap, 
                    norm=norm, shading='auto', rasterized=True, alpha = color_alpha)
            else:
                imcolor = ax.imshow(data, cmap=cmap, origin='lower', 
                    extent=extent, norm=norm, interpolation=interpolation, 
                    rasterized=True, alpha = color_alpha)

            # color bar
            if colorbar: self.add_colorbar(imcolor, iaxis=iaxis,
                cbarlabel=cbarlabel, cbaroptions=cbaroptions)

        # contour map
        if contour:
            if exact_coord:
                imcont = ax.contour(xx, yy, data, colors=ccolor,
                    levels=clevels, linewidths=clw)
            else:
                imcont = ax.contour(data, colors=ccolor, 
                    origin='lower', levels=clevels,
                    linewidths=clw, extent=extent)


        # set axes
        ax.set_xlim(figxmin,figxmax)
        ax.set_ylim(figymin,figymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if len(xticks) != 0:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)
        if  len(yticks) != 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
        ax.set_aspect(aspect)
        ax.tick_params(which='both', direction='in', bottom=True, top=True,
            left=True, right=True, pad=9)


        # plot beam size
        if plot_beam: add_beam(ax, bmaj, bmin, bpa, bcolor)


        # central star position
        if ccross:
            ll, lw, cl = prop_cross
            add_cross(ax, loc=(0,0), length=ll, 
                width=lw, color=cl, zorder=10.)


        # scale bar
        if len(scalebar): add_scalebar(ax, scalebar)


        # save fig
        if outname: self.savefig(outname, transparent = transparent)

        return ax


    def plot_vectors(self, image, iaxis=0, norm=1., pivot='mid', 
        headwidth=1., headlength=0.001, width=0.01, color='red',):
        add_vectors(image, self.axes[iaxis], norm=norm, pivot=pivot, 
            headwidth=headwidth, headlength=headlength, 
            width=width, color=color,)
        return self.axes[iaxis]


    # Channel maps
    def channelmaps(self, image, imscale=[], data=None,
        color=True, cmap='PuBuGn', outname=None,
        vmin=None, vmax=None, contour=True, clevels=[], ccolor='k',
        nrow=5, ncol=5, velmin=None, velmax=None, nskip=1, clw=0.5, color_norm=None,
        xticks=[], yticks=[], vsys=None, scalebar=[],
        ccross=True, prop_cross=[None, 1., 'k'], bcolor='k', 
        cbarticks=None, coord_center=None, sbar_vertical=False,
        vlabel_on=True, colorbar=True, cbarlabel='', alpha = 1.,
        plotall=False, absolutecoords=False, plot_beam=True, txtcolor='k'):
        '''
        Draw channel maps.

        Parameters
        ----------
        image: Input image. Must be Imfits object.
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
        ccross: If True, a central star or the center of an self will be shown as a cross.
        prop_cross: Detailed setting for the cross showing stellar position.
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

        # grid
        if self.naxes == 0:
            self.add_grid(nrow, ncol)

        print('Draw channel maps...')


        # data
        if data is not None:
            naxis = len(data.shape)
        else:
            data  = image.data.copy()
            header = image.header.copy()
            naxis = image.naxis

        # check number of image axes
        if naxis == 3:
            pass
        elif naxis == 4:
            data = data[0, :, :, :]
        else:
            print ('ERROR\tchannelmaps: NAXIS of fits must be 3 or 4.')
            return 0

        # coordinate center
        if coord_center:
            image.shift_coord_center(coord_center)

        # Coordinates
        if absolutecoords:
            print ('WARNING: Abusolute coordinates are still in development.')
            xlabel = image.label_i[0]
            ylabel = image.label_i[1]
            cc = image.cc
            xaxis = image.xx_wcs[image.ny//2,:] # not exactly though
            yaxis = image.yy_wcs[:, image.nx//2]
        else:
            #xx = self.xx*3600.
            #yy = self.yy*3600.
            cc = image.cc
            xaxis = image.xaxis*3600.
            yaxis = image.yaxis*3600.
            xlabel = 'RA offset (arcsec)'
            ylabel = 'DEC offset (arcsec)'

        # velocity axis
        vaxis = image.vaxis.copy()
        delv  = image.delv
        if delv < 0:
            delv  = - delv
            vaxis = vaxis[::-1]
            data  = data[::-1,:,:]

        # Relative velocity
        if vsys:
            vaxis = vaxis - vsys

        # velocity range
        if (velmin is not None) & (velmax is not None):
            vlim = [velmin, velmax]
        elif velmin:
            vlim = [velmin, np.nanmax(vaxis)+1.]
        elif velmax:
            vlim = [np.nanmin(vaxis)-1., velmax]
        else:
            vlim = []

        # trim data for plot
        if len(imscale) == 0:
            xmin, xmax = xaxis[[0,-1]]
            ymin, ymax = yaxis[[0,-1]]
            extent = (xmin-0.5*image.delx, xmax+0.5*image.delx, 
                ymin-0.5*image.dely, ymax+0.5*image.dely)
            figxmin, figxmax, figymin, figymax = extent
            d_plt, x_plt, y_plt, v_plt = trim_data(data, xaxis, yaxis, vaxis,
             [figxmax, figxmin], [figymin, figymax], vlim)
        elif len(imscale) == 4:
            figxmax, figxmin, figymin, figymax = imscale
            d_plt, x_plt, y_plt, v_plt = trim_data(data, xaxis, yaxis, vaxis,
             [figxmax, figxmin], [figymin, figymax], vlim)
            extent = (x_plt[0]-0.5*image.delx, x_plt[-1]+0.5*image.delx,
             y_plt[0]-0.5*image.dely, y_plt[-1]+0.5*image.dely)
        else:
            print ('ERROR\tchannelmaps: Input imscale is wrong.\
             Must be [xmin, xmax, ymin, ymax]')
            d_plt = data
            v_plt = vaxis
        nv_plt = len(v_plt)
        xx, yy = np.meshgrid(x_plt, y_plt)

        if nskip:
            d_plt  = d_plt[::nskip,:,:]
            v_plt  = v_plt[::nskip]
            nv_plt = len(v_plt)

        # Set colorscale
        vmax = vmax if vmax is not None else np.nanmax(data)
        vmin = vmin if vmin is not None else np.nanmin(data)

        # color scale
        norm = color_normalization(color_norm, vmin, vmax)

        # default contour
        if len(clevels) == 0:
            clevels = np.array([0.2, 0.4, 0.6, 0.8])*np.nanmax(d_plt)

        # Counter
        nrow, ncol = self.nrow, self.ncol
        gridimax = nrow * ncol - 1
        gridi    = 0
        imap     = 1
        ax       = None

        # draw channel maps
        for ichan in range(nv_plt):
            # maximum grid
            if gridi > gridimax:
                break

            # Select channel
            Sv = d_plt[ichan,:,:]

            # velocity at nchan
            v_i = v_plt[ichan]

            # Axis
            ax = self.axes[gridi]
            #print ('Channel ', '%s'%ichan, ', velocity: ', '%4.2f'%v_i, ' km/s')

            # color and/or contour plots
            if color:
                imcolor = ax.imshow(Sv, 
                    cmap=cmap, origin='lower', 
                    extent=extent, norm=norm, 
                    rasterized=True, alpha = alpha)
            if contour:
                imcont  = ax.contour(Sv, colors=ccolor, 
                    origin='lower',extent=extent, 
                    levels=clevels, linewidths=clw)

            # set axes
            ax.set_xlim(figxmin,figxmax)
            ax.set_ylim(figymin,figymax)
            if len(xticks) != 0:
                ax.set_xticks(xticks)

            if len(yticks) != 0:
                ax.set_yticks(yticks)

            ax.set_aspect(1)
            ax.tick_params(which='both', direction='in',bottom=True,
             top=True, left=True, right=True, pad=9,)

            # Velocity label
            if vlabel_on:
                vlabel = '%3.2f'%v_i
                ax.text(0.1, 0.9, vlabel, color=txtcolor,
                 horizontalalignment='left', verticalalignment='top',
                  transform=ax.transAxes)

            # Central star position
            if ccross:
                ll, lw, cl = prop_cross
                add_cross(ax, loc=(0,0), length=ll, 
                    width=lw, color=cl, zorder=10.)

            # counter
            gridi += 1

            # break or continue?
            if gridi > gridimax:
                if plotall:
                    # Plot beam
                    if (plot_beam == True) & (image.beam is not None):
                        bmaj, bmin, bpa = image.beam
                        add_beam(self.axes[(nrow-1)*ncol], bmaj, bmin, bpa, bcolor, loc='bottom left')
                    # scalebar
                    if len(scalebar) != 0:
                        add_scalebar(grid[(nrow-1)*ncol], scalebar)
                    # colorbar
                    if color and ax:
                        cax, cbar = self.add_colorbar(imcolor, cbarlabel,)
                    #label
                    self.axes[(nrow-1)*ncol].set_xlabel(xlabel)
                    self.axes[(nrow-1)*ncol].set_ylabel(ylabel)
                    # remove blank pannel
                    if gridi != gridimax+1 and gridi != 0:
                        while gridi != gridimax+1:
                            #print gridi
                            self.axes[gridi].axis('off')
                            gridi = gridi+1
                    # save
                    if outname is None:
                        print('WARNING\tchannelmaps: outname is not provided.')
                        print('WARNING\tchannelmaps: output files are written out when plotall=True.')
                        print('WARNING\tchannelmaps: header name will be channelmaps.')
                        outname = 'channelmaps'
                    self.savefig(outname + '_i%02i'%imap, )
                    # reset
                    self.fig.clf() # clear figure
                    self.style(self._config) # reset style
                    self.axes = ImageGrid(self.fig, rect=111,
                        nrows_ncols=(self.nrow, self.ncol), 
                        **self._gridkwargs) # retrieve grid
                    gridi = 0
                    imap  += 1
                    #break
                else:
                    break

        # On the bottom-left corner pannel
        # Labels
        self.axes[(nrow-1)*ncol].set_xlabel(xlabel)
        self.axes[(nrow-1)*ncol].set_ylabel(ylabel)
        # Plot beam
        if (plot_beam == True) & (image.beam is not None):
            bmaj, bmin, bpa = image.beam
            add_beam(self.axes[(nrow-1)*ncol], bmaj, bmin, bpa, bcolor, loc='bottom left')
        # scalebar
        if len(scalebar) != 0:
            add_scalebar(self.axes[(nrow-1)*ncol], scalebar)
        # colorbar
        if color and ax and colorbar:
            cax, cbar = self.add_colorbar(imcolor, cbarlabel=cbarlabel,)
        # remove blank pannel
        if gridi != gridimax+1 and gridi != 0:
            while gridi != gridimax+1:
                #print gridi
                self.axes[gridi].axis('off')
                gridi = gridi+1

        return self.axes


    def pvdiagram(self, image,
        color=True, cmap='PuBuGn', color_norm=None, vmin=None, vmax=None, 
        contour=True, clevels=None, ccolor='k', lw=1,
        pa=None, vsys=None, vrel=False, x_offset=False, ratio=1.2,
        clip=None, plot_res=True, xlim=[], ylim=[],
        ln_hor=True, ln_var=True, alpha=None, colorbar=False,
        cbaroptions=('right', '3%', '0%'), cbarlabel=r'(Jy beam$^{-1}$)',
        iaxis=0, inmode=None, data=None, outname=None, transparent=True):
        '''
        Draw the PV diagram.

        Parameters
        ----------
         - outname:
        '''

        # Modules
        import copy
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # function
        def change_aspect_ratio(ax, ratio):
            '''
            This function change aspect ratio of figure.
            Parameters:
                ax: ax (matplotlit.pyplot.subplots())
                    Axes object
                ratio: float or int
                    relative x axis width compared to y axis width.
            '''
            aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0])\
             / (ax.get_ylim()[1] - ax.get_ylim()[0])
            aspect = np.abs(aspect)
            aspect = float(aspect)
            ax.set_aspect(aspect)

        # axis
        ax = self.axes[iaxis]

        # Input
        if inmode == 'data':
            if data is None:
                print ("ERROR\tpvdiagram: inmode ='data' is selected. data must be provided.")
                return 0
            naxis = len(data.shape)
        else:
            data   = np.squeeze(image.data)
            header = image.header
            naxis  = image.naxis


        # Read
        xaxis = image.xaxis.copy()
        vaxis = image.vaxis.copy()
        delx  = image.delx
        delv  = image.delv
        nx    = len(xaxis)
        nv    = len(vaxis)

        # Beam
        bmaj, bmin, bpa = image.beam

        if image.res_off:
            res_off = image.res_off
        else:
            # Resolution along offset axis
            if image.pa:
                pa = image.pa

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
            vcenter = 0.
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
            hline_params = [vcenter,offmin,offmax]
            vline_params = [0.,velmin,velmax]
            res_x = res_off
            res_y = delv
        else:
            data   = data.T # np.rot90(data)
            extent = (velmin,velmax,offmin,offmax)
            xlabel = vlabel
            ylabel = offlabel
            hline_params = [0.,velmin,velmax]
            vline_params = [vcenter,offmin,offmax]
            res_x = delv
            res_y = res_off


        # Set colorscale
        vmax = vmax if vmax is not None else np.nanmax(data)
        vmin = vmin if vmin is not None else np.nanmin(data)
        # color scale
        norm = color_normalization(color_norm, vmin, vmax)


        # clip data at some value
        data_color = data.copy()
        if clip: data_color[np.where(data < clip)] = np.nan

        # plot in color
        if color:
            imcolor = ax.imshow(data_color, cmap=cmap, origin='lower',
                extent=extent, norm=norm, alpha=alpha)
        # plot with contours
        if contour:
            if clevels is None:
                clevels = np.arange(0.2, 1., 0.2)*np.nanmax(data)
            imcont  = ax.contour(data, colors=ccolor, origin='lower',
                extent=extent, levels=clevels, linewidths=lw, alpha=alpha)

        # axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # set xlim, ylim
        if len(xlim) == 0:
            ax.set_xlim(extent[0],extent[1])
        elif len(xlim) == 2:
            ax.set_xlim(*xlim)
        else:
            print ('WARRING\tpvdiagram: Input xlim is in a wrong format. Must be [xmin, xmax].')
            ax.set_xlim(extent[0],extent[1])

        if len(ylim) == 0:
            ax.set_ylim(extent[2],extent[3])
        elif len(ylim) == 2:
            ax.set_ylim(*ylim)
        else:
            print ('WARRING\tpvdiagram: Input ylim is in a wrong format. Must be [ymin, ymax].')
            ax.set_ylim(extent[2],extent[3])


        # lines showing offset 0 and relative velocity 0
        if ln_hor:
            if all([i is not None for i in hline_params]):
                xline = ax.hlines(hline_params[0], hline_params[1], hline_params[2], ccolor, linestyles='dashed', linewidths = 1.)
        if ln_var:
            if all([i is not None for i in vline_params]):
                yline = ax.vlines(vline_params[0], vline_params[1], vline_params[2], ccolor, linestyles='dashed', linewidths = 1.)

        ax.tick_params(which='both', direction='in',
            bottom=True, top=True, left=True, right=True, pad=9,)

        # plot resolutions
        if plot_res:
            # x axis
            #print (res_x, res_y)
            res_x_plt, res_y_plt = ax.transLimits.transform((res_x*0.5, res_y*0.5)) -  ax.transLimits.transform((0, 0)) # data --> Axes coordinate
            ax.errorbar(0.1, 0.1, xerr=res_x_plt, yerr=res_y_plt, color=ccolor, capsize=3, capthick=1., elinewidth=1., transform=ax.transAxes)

        # aspect ratio
        if ratio: change_aspect_ratio(ax, ratio)
        # color bar
        if all([color, colorbar]): self.add_colorbar(imcolor, iaxis=iaxis,
            cbarlabel=cbarlabel, cbaroptions=cbaroptions)
        # save figure
        if outname: self.savefig(outname, transparent = transparent)

        return self.axes



    def rgb_plot(self, image_r = None, image_g = None, image_b = None, 
        reference = 'R', normalize = True, ftune_rgb = (1.,1., 1,), iaxis = 0,
        stretch=0.5, scalebar = None
        ):
        # at least one image
        if (image_r is None) * (image_b is None) * (image_g is None):
            print('ERROR\trgb_plot: at least one of R, G, or B images must be provided.')
            return 0

        if (reference == 'R') | (reference == 'red'):
            image_ref = image_r
            image_1 = image_g
            image_2 = image_b
        elif (reference == 'G') | (reference == 'green'):
            image_1 = image_r
            image_ref = image_g
            image_2 = image_b
        elif (reference == 'B') | (reference == 'blue'):
            image_1 = image_r
            image_2 = image_g
            image_ref = image_b
        else:
            print('ERROR\trgb_plot: reference must be R, G, B, red, green or blue.')
            return 0


        # reference image
        data_ref = np.squeeze(image_ref.data)
        ny, nx = data_ref.shape

        # image 1
        if image_1 is None:
            data_1 = np.zeros((ny, nx))
        else:
            _ny, _nx = image_1.ny, image_1.nx
            if (_ny == ny) * (_nx == nx):
                data_1 = np.squeeze(image_1.data)
            else:
                data_1 = np.squeeze(
                    au.match_images(image_1, image_ref))

        # image 2
        if image_2 is None:
            data_2 = np.zeros((ny, nx))
        else:
            _ny, _nx = image_2.ny, image_2.nx
            if (_ny == ny) * (_nx == nx):
                data_2 = np.squeeze(image_2.data)
            else:
                data_2 = np.squeeze(
                    au.match_images(image_2, image_ref))

        # cast back to RGB
        if (reference == 'R') | (reference == 'red'):
            data_r, data_g, data_b = data_ref, data_1, data_2
        elif (reference == 'G') | (reference == 'green'):
            data_r, data_g, data_b = data_1, data_ref, data_2
        elif (reference == 'B') | (reference == 'blue'):
            data_r, data_g, data_b = data_1, data_2, data_ref

        if normalize:
            if image_r is not None: data_r *= ftune_rgb[0] / np.nanmax(data_r)
            if image_g is not None: data_g *= ftune_rgb[1] / np.nanmax(data_g)
            if image_b is not None: data_b *= ftune_rgb[2] / np.nanmax(data_b)

        for d_i in [data_r, data_g, data_b]:
            d_i[d_i < 0.] = 0. # remove minus


        # image
        image = make_lupton_rgb(data_r, data_g, data_b, stretch=stretch)

        # plot
        ax = self.axes[iaxis]
        ax.imshow(image, extent = image_ref.get_mapextent(),
            origin = 'lower', rasterized = True)

        if scalebar is not None:
            dm.add_scalebar(ax, scalebar, 
                orientation='horizontal',)

        return image


    def add_colorbar(self, 
        cim = None, iaxis = 0,
        cbarlabel: str='', 
        cbaroptions: list = ['right', '3%', '0%'],
        ticks: list = None,
        tickcolor: str = 'k', 
        axiscolor: str = 'k', 
        labelcolor: str = 'k'):
        # parameter
        orientations = {
        'right': 'vertical',
        'left': 'vertical',
        'top': 'horizontal',
        'bottom': 'horizontal'}

        # color scale
        if cim is not None:
            pass
        else:
            try:
                cim = self.axes[0].images[0] # assume the first one is a color map.
            except:
                print('ERROR\tadd_colorbar: cannot find a color map.')
                return 0

        # axis or grid
        if (self.grid == False):
            # to axis
            # setting for a color bar
            if len(cbaroptions) == 3:
                cbar_loc, cbar_wd, cbar_pad = cbaroptions
            elif len(cbaroptions) == 4:
                cbar_loc, cbar_wd, cbar_pad, cbarlabel = cbaroptions
            else:
                print('WARNING\tadd_colorbar: cbaroptions must have three or four elements. \
                Input is ignored.')
            cbar_loc, cbar_wd, cbar_pad = cbaroptions
            # divide axis
            #divider = make_axes_locatable(self.axes[iaxis])
            #cax     = divider.append_axes(cbar_loc, size=cbar_wd, pad=cbar_pad)
            # use inset axes
            if cbar_loc == 'right':
                width = cbar_wd
                height = '100%'
                bbox_to_anchor = (1. + float(cbar_pad.strip('%'))*0.01, 0., 1., 1.)
            elif cbar_loc == 'top':
                width = '100%'
                height = cbar_wd
                bbox_to_anchor = (0., 1. + float(cbar_pad.strip('%'))*0.01, 1., 1.)
            else:
                print("ERROR\tadd_colorbar: cbar_loc must be 'right' or 'top'.")
                return 0
            cax = inset_axes(self.axes[iaxis],
                width = width,
                height = height,
                loc = 'lower left',
                bbox_to_anchor = bbox_to_anchor,
                bbox_transform = self.axes[iaxis].transAxes,
                borderpad = 0.)
            # add a color bar
            cbar = plt.colorbar(cim, cax=cax, ticks=ticks, 
                orientation=orientations[cbar_loc], ticklocation=cbar_loc)
            cbar.set_label(cbarlabel)
        else:
            # to grid
            cax  = self.axes.cbar_axes[iaxis]
            cbar = plt.colorbar(cim, cax=cax, 
            orientation = orientations[self._gridkwargs['cbar_location']],
            ticklocation = self._gridkwargs['cbar_location']) # ticks=cbarticks
            cax.toggle_label(True)

            if cbarlabel:
                cbar.set_label(cbarlabel, color=labelcolor,)
                #cbar.ax.set_ylabel(cbarlabel, color=labelcolor,) # label

        #cbar.ax.tick_params(labelsize=fontsize, labelcolor=labelcolor, color=tickcolor,)
        return cax, cbar



# intensity map
def intensitymap(self, ax=None, outname=None, imscale=[], outformat='pdf',
    color=True, cmap='Blues', colorbar=True, cbarlabel='', cbaroptions=np.array(['right','3%','0%']), vmin=None,vmax=None,
    contour=True, clevels=None, ccolor='k', clw=1,
    data=None, axis=0, xticks=[], yticks=[], relativecoords=True, csize=18, scalebar=[],
    cstar=True, prop_star=[], color_norm=None, bcolor='k',figsize=(11.69,8.27),
    coord_center=None, plot_beam = True, aspect=1,
    interpolation=None, noreg=True, inmode=None, exact_coord=False,
    tickcolor='k',axiscolor='k',labelcolor='k',darkbg=False):
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

    if darkbg:
        tickcolor   = 'white'
        axiscolor   = 'white'
        ccolor      = 'white'
        bcolor      = 'white'
        labelcolor  = 'k'
        transparent = False
    else:
        transparent=True

    # setting output file name & format
    #outname = outname if outname else self.file.replace('.fits', '_intensitymap')

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

    # beam
    if self.beam is None:
        plot_beam = False
    else:
        bmaj, bmin, bpa = self.beam

    # center
    if coord_center:
        self.shift_coord_center(coord_center)

    # coordinate style
    if relativecoords:
        cc    = self.cc
        xaxis = self.xaxis*3600.
        yaxis = self.yaxis*3600.
        xlabel = 'RA offset (arcsec)'
        ylabel = 'DEC offset (arcsec)'
    else:
        print ('WARNING: Abusolute coordinates are still in development.')
        xlabel = self.label_i[0]
        ylabel = self.label_i[1]
        cc    = self.cc
        xaxis = xx_wcs[self.nx//2,:] # not exactly though
        yaxis = yy_wcs[:,self.ny//2]


    # trim data for plot
    if len(imscale) == 0:
        xmin, xmax = xaxis[[0,-1]]
        ymin, ymax = yaxis[[0,-1]]
        extent = (xmin-0.5*self.delx, xmax+0.5*self.delx, ymin-0.5*self.dely, ymax+0.5*self.dely)
        figxmin, figxmax, figymin, figymax = extent
        data, xaxis, yaxis = trim_data(data, xaxis, yaxis, [],
         [figxmax, figxmin], [figymin, figymax], [])
    elif len(imscale) == 4:
        figxmax, figxmin, figymin, figymax = imscale
        data, xaxis, yaxis = trim_data(data, xaxis, yaxis, [],
         [figxmax, figxmin], [figymin, figymax], [])
        xmin, xmax = xaxis[[0,-1]]
        ymin, ymax = yaxis[[0,-1]]
        extent = (xmin-0.5*self.delx, xmax+0.5*self.delx, ymin-0.5*self.dely, ymax+0.5*self.dely)
    else:
        print ('ERROR\tchannelmaps: Input imscale is wrong.\
         Must be [xmin, xmax, ymin, ymax]')

    # meshgrid if needed
    if exact_coord: xx, yy  = np.meshgrid(xaxis, yaxis)

    # setting figure
    if ax is not None:
        pass
    else:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111)

    if darkbg: ax.set_facecolor('k')

    # set colorscale
    if vmax is None: vmax = np.nanmax(data)

    # color scale
    if type(color_norm) == str:
        if color_norm == 'log':
            norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    elif type(color_norm) == tuple:
        if hasattr(color_norm[0], '__call__'):
            norm = mpl.colors.FuncNorm(color_norm, vmin=vmin, vmax=vmax)
        elif color_norm[0].replace(' ','').lower() == 'asinhstretch':
            a = float(color_norm[1])
            def _forward(x,):
                return np.arcsinh(x/a)/np.arcsinh(1./a)
            def _inverse(x,):
                return a*np.sinh(x*(np.arcsinh(1/a)))
            norm = mpl.colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)
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
        if colorbar: add_colorbar_toaxis(ax, imcolor, cbaroptions, cbarlabel=cbarlabel,
            fontsize=csize, tickcolor=tickcolor, labelcolor=labelcolor)

    # contour map
    if contour:
        if exact_coord:
            imcont = ax.contour(xx, yy, data, colors=ccolor,
                levels=clevels, linewidths=clw)
        else:
            imcont = ax.contour(data, colors=ccolor, origin='lower', levels=clevels,
                linewidths=clw, extent=extent)


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

    ax.set_aspect(aspect)
    ax.tick_params(which='both', direction='in',bottom=True, top=True,
        left=True, right=True, labelsize=csize, color=tickcolor,
        labelcolor=labelcolor, pad=9)

    # plot beam size
    if plot_beam: add_beam(ax, bmaj, bmin, bpa, bcolor)

    # central star position
    if cstar:
        if len(prop_star) == 0:
            ll = 0.1*np.abs(figxmax - figxmin)
            lw = 1.
            cl = 'white' if darkbg else 'k'
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
    if len(scalebar): add_scalebar(ax, scalebar)

    # save fig
    plt.savefig(outname, transparent = transparent)

    return ax


# Channel maps
def channelmaps(self, grid=None, data=None, outname=None, outformat='pdf',
    imscale=[], color=False, cbaron=False, cmap='Blues', vmin=None, vmax=None,
    contour=True, clevels=[], ccolor='k',
    nrow=5, ncol=5, velmin=None, velmax=None, nskip=1, clw=0.5, color_norm=None,
    xticks=[], yticks=[], relativecoords=True, vsys=None, csize=14, scalebar=np.empty(0),
    cstar=True, prop_star=[], tickcolor='k', axiscolor='k', darkbg=False,
    labelcolor='k',cbarlabel=None, txtcolor='k', bcolor='k', figsize=(11.69,8.27),
    cbarticks=None, coord_center=None, noreg=True, arcsec=True, sbar_vertical=False,
    cbaroptions=np.array(['right','3%','0%']), inmode='fits', vlabel_on=True,
    plotall=False,):
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

    if darkbg:
        tickcolor  = 'white'
        axiscolor  = 'white'
        txtcolor   = 'white'
        bcolor     = 'white'
        ccolor     = 'white'
        labelcolor = 'k'

    print('Draw channel maps...')


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
        pass
        #plot_beam = False
    else:
        bmaj, bmin, bpa = self.beam


    # coordinate center
    if coord_center:
        self.shift_coord_center(coord_center)


    # Coordinates
    if relativecoords:
        #xx = self.xx*3600.
        #yy = self.yy*3600.
        cc = self.cc
        xaxis = self.xaxis*3600.
        yaxis = self.yaxis*3600.
        xlabel = 'RA offset (arcsec)'
        ylabel = 'DEC offset (arcsec)'
    else:
        print ('WARNING: Abusolute coordinates are still in development.')
        xlabel = self.label_i[0]
        ylabel = self.label_i[1]
        #xx = self.xx_wcs
        #yy = self.yy_wcs
        cc = self.cc
        xaxis = xx_wcs[self.nx//2,:] # not exactly though
        yaxis = yy_wcs[:,self.ny//2]

    # vaxis
    # Get velocity axis
    vaxis = self.vaxis.copy()
    delv  = self.delv

    if delv < 0:
        delv  = - delv
        vaxis = vaxis[::-1]
        data  = data[::-1,:,:]

    # Relative velocity
    if vsys:
        vaxis = vaxis - vsys

    if (velmin is not None) & (velmax is not None):
        vlim = [velmin, velmax]
    elif velmin is not None:
        vlim = [velmin, np.nanmax(vaxis)+1.]
    elif velmax is not None:
        vlim = [np.nanmin(vaxis)-1., velmax]
    else:
        vlim = []

    # trim data for plot
    if len(imscale) == 0:
        xmin, xmax = xaxis[[0,-1]]
        ymin, ymax = yaxis[[0,-1]]
        extent = (xmin-0.5*self.delx, xmax+0.5*self.delx, ymin-0.5*self.dely, ymax+0.5*self.dely)
        figxmin, figxmax, figymin, figymax = extent
        d_plt, x_plt, y_plt, v_plt = trim_data(data, xaxis, yaxis, vaxis,
         [figxmax, figxmin], [figymin, figymax], vlim)
    elif len(imscale) == 4:
        figxmax, figxmin, figymin, figymax = imscale
        d_plt, x_plt, y_plt, v_plt = trim_data(data, xaxis, yaxis, vaxis,
         [figxmax, figxmin], [figymin, figymax], vlim)
        extent = (x_plt[0]-0.5*self.delx, x_plt[-1]+0.5*self.delx,
         y_plt[0]-0.5*self.dely, y_plt[-1]+0.5*self.dely)
    else:
        print ('ERROR\tchannelmaps: Input imscale is wrong.\
         Must be [xmin, xmax, ymin, ymax]')
        d_plt = data
        v_plt = vaxis
    nv_plt = len(v_plt)
    xx, yy = np.meshgrid(x_plt, y_plt)

    if nskip:
        d_plt  = d_plt[::nskip,:,:]
        v_plt  = v_plt[::nskip]
        nv_plt = len(v_plt)

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


    if len(clevels) == 0:
        clevels = np.array([0.2, 0.4, 0.6, 0.8])*np.nanmax(d_plt)

    # Counter
    gridimax = nrow*ncol-1
    gridi    = 0
    imap     = 0
    ax       = None

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
        #print ('Channel ', '%s'%ichan, ', velocity: ', '%4.2f'%v_i, ' km/s')

        # showing in color scale
        if color:
            imcolor = ax.imshow(Sv, cmap=cmap, origin='lower', extent=extent, norm=norm, rasterized=True)

        if contour:
            imcont  = ax.contour(Sv, colors=ccolor, origin='lower',extent=extent, levels=clevels, linewidths=clw)

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
                cl = 'white' if darkbg else 'k'
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

        # break or continue?
        if gridi > gridimax:
            if plotall:
                # Plot beam
                add_beam(grid[(nrow-1)*ncol], bmaj, bmin, bpa, bcolor, loc='bottom left')
                # scalebar
                if len(scalebar) != 0:
                    add_scalebar(grid[(nrow-1)*ncol], scalebar)
                # colorbar
                if color and cbaron and ax:
                    cbar = add_colorbar_togrid(imcolor, grid, cbarlabel,
                        tickcolor, axiscolor, labelcolor)
                #label
                grid[(nrow-1)*ncol].set_xlabel(xlabel)
                grid[(nrow-1)*ncol].set_ylabel(ylabel)
                grid[(nrow-1)*ncol].xaxis.label.set_color(labelcolor)
                grid[(nrow-1)*ncol].yaxis.label.set_color(labelcolor)
                # save
                plt.savefig(outfile.replace('.'+outformat, '_%02i.'%imap + outformat),
                    transparent = True)
                # reset
                fig = plt.figure(figsize=figsize)
                grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
                    axes_pad=0,share_all=True,cbar_mode=cbar_mode,
                    cbar_location=cbar_loc,cbar_size=cbar_wd,cbar_pad=cbar_pad,
                    label_mode='1')
                gridi = 0
                ax    = None
                imap  += 1
            else:
                break

    # On the bottom-left corner pannel
    # Labels
    grid[(nrow-1)*ncol].set_xlabel(xlabel)
    grid[(nrow-1)*ncol].set_ylabel(ylabel)
    grid[(nrow-1)*ncol].xaxis.label.set_color(labelcolor)
    grid[(nrow-1)*ncol].yaxis.label.set_color(labelcolor)

    # Plot beam
    add_beam(grid[(nrow-1)*ncol], bmaj, bmin, bpa, bcolor, loc='bottom left')
    # Scale bar
    if len(scalebar) != 0:
        add_scalebar(grid[(nrow-1)*ncol], scalebar)
    # colorbar
    if color and cbaron and ax:
        cbar = add_colorbar_togrid(imcolor, grid, cbarlabel,
            tickcolor, axiscolor, labelcolor)

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

    if plotall:
        plt.savefig(outfile.replace('.'+outformat, '_%02i.'%imap + outformat),
            transparent = True)
    else:
        plt.savefig(outfile, transparent = True)

    return grid



# Draw pv diagram
def pvdiagram(self,outname,data=None,header=None,ax=None,outformat='pdf',color=True,cmap='Blues',
    vmin=None,vmax=None,vsys=None,contour=True,clevels=None,ccolor='k', pa=None,
    vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
    lw=1,clip=None,plot_res=True,inmode='fits',xlim=[], ylim=[],
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
        vcenter = 0.
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
        hline_params = [vcenter,offmin,offmax]
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
        if clevels is None:
            clevels = np.arange(0.2, 1., 0.2)*np.nanmax(data)
        imcont  = ax.contour(data, colors=ccolor, origin='lower',
            extent=extent, levels=clevels, linewidths=lw, alpha=alpha)


    # axis labels
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    # set xlim, ylim
    if len(xlim) == 0:
        ax.set_xlim(extent[0],extent[1])
    elif len(xlim) == 2:
        ax.set_xlim(*xlim)
    else:
        print ('WARRING: Input xranges is wrong. Must be [xmin, xmax].')
        ax.set_xlim(extent[0],extent[1])

    if len(ylim) == 0:
        ax.set_ylim(extent[2],extent[3])
    elif len(ylim) == 2:
        ax.set_ylim(*ylim)
    else:
        print ('WARRING: Input yranges is wrong. Must be [ymin, ymax].')
        ax.set_ylim(extent[2],extent[3])


    # lines showing offset 0 and relative velocity 0
    if ln_hor:
        if all([i is not None for i in hline_params]):
            xline = ax.hlines(hline_params[0], hline_params[1], hline_params[2], ccolor, linestyles='dashed', linewidths = 1.)
    if ln_var:
        if all([i is not None for i in vline_params]):
            yline = ax.vlines(vline_params[0], vline_params[1], vline_params[2], ccolor, linestyles='dashed', linewidths = 1.)

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
    cbaroptions=['right', '3%', '0%'], label_mode='1'):
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
    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(figsize=figsize)

    # Generate grid
    cbar_loc, cbar_wd, cbar_pad = cbaroptions
    grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
        axes_pad=axes_pad, share_all=share_all, cbar_mode=cbar_mode,
        cbar_location=cbar_loc,cbar_size=cbar_wd, cbar_pad=cbar_pad,
        label_mode=label_mode)

    return grid


def trim_data(data, x, y, v,
    xlim: list = [], ylim: list = [], vlim: list = []):
    '''
    Clip a cube image with given axis ranges.

    Parameters
    ----------
        data (array): Data
        x (array): x axis
        y (array): y axis
        v (array): v axis
        xlim (list): x range. Must be given as [xmin, xmax].
        ylim (list): y range. Must be given as [ymin, ymax].
        vlim (list): v range. Must be given as [vmin, vmax].
    '''
    data = np.squeeze(data)
    if len(data.shape) == 2:
        ny, nx = data.shape
        yimin, yimax = index_between(y, ylim, mode='edge')[0]
        ximin, ximax = index_between(x, xlim, mode='edge')[0]
        # for beautiful map edge
        yimin = yimin - 1 if yimin >= 1 else yimin
        yimax = yimax + 1 if yimax < ny-1 else yimax
        ximin = ximin - 1 if ximin >= 1 else ximin
        ximax = ximax + 1 if ximax < nx-1 else ximax
        # triming
        data = data[yimin:yimax+1, ximin:ximax+1]
        y = y[yimin:yimax+1]
        x = x[ximin:ximax+1]
        return data, x, y
    elif len(data.shape) == 3:
        nv, ny, nx = data.shape
        vimin, vimax = index_between(v, vlim, mode='edge')[0]
        yimin, yimax = index_between(y, ylim, mode='edge')[0]
        ximin, ximax = index_between(x, xlim, mode='edge')[0]
        # for beautiful map edges
        yimin = yimin - 1 if yimin >= 1 else yimin
        yimax = yimax + 1 if yimax < ny-1 else yimax
        ximin = ximin - 1 if ximin >= 1 else ximin
        ximax = ximax + 1 if ximax < nx-1 else ximax
        # triming
        data = data[vimin:vimax+1, yimin:yimax+1, ximin:ximax+1]
        v    = v[index_between(v, vlim)]
        y = y[yimin:yimax+1]
        x = x[ximin:ximax+1]
        return data, x, y, v
    else:
        print('trim_data: Invalid data shape.')
        return -1


def color_normalization(color_norm, vmin, vmax):
    # color scale
    if type(color_norm) == str:
        if color_norm == 'log':
            norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    elif type(color_norm) == tuple:
        if hasattr(color_norm[0], '__call__'):
            norm = mpl.colors.FuncNorm(color_norm, vmin=vmin, vmax=vmax)
        elif color_norm[0].replace(' ','').lower() == 'asinhstretch':
            a = float(color_norm[1])
            def _forward(x,):
                return np.arcsinh(x/a)/np.arcsinh(1./a)
            def _inverse(x,):
                return a*np.sinh(x*(np.arcsinh(1/a)))
            norm = mpl.colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)
        else:
            print ('ERROR\tcolor_normalization: color_norm must be strings or tuple of functions.')
    else:
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)

    return norm


def index_between(t, tlim, mode='all'):
    if not (len(tlim) == 2):
        if mode=='all':
            return np.full(np.shape(t), True)
        elif mode == 'edge':
            if len(t.shape) == 1:
                return ([0, len(t)-1], )
            else:
                return [[0, t.shape[i]] for i in range(len(t.shape))]
        else:
            print('index_between: mode parameter is not right.')
            return np.full(np.shape(t), True)
    else:
        if mode=='all':
            return (tlim[0] <= t) * (t <= tlim[1])
        elif mode == 'edge':
            nonzero = np.nonzero((tlim[0] <= t) * (t <= tlim[1]))
            return tuple([[np.min(i), np.max(i)] for i in nonzero])
        else:
            print('index_between: mode parameter is not right.')
            return (tlim[0] <= t) * (t <= tlim[1])


def add_cross(ax, loc=(0,0), length=None, width=1., color='k',
    zorder=10.):
    '''
    Add a cross in a map to indicate a location of, for example, a stellar object.

    Args:
     - ax: axis
     - loc (tuple or str): coordinates for plot.
     - length: line length for the cross
     - width: line width
     - color: line color
    '''
    if length is None:
        length = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0])*0.1

    if type(loc) == tuple or type(loc) == list:
        loc_x, loc_y = loc
    else:
        print('ERROR\tadd_cross: loc must be tuple or list.')
        return 0

    ax.hlines(loc_y, loc_x-length*0.5, loc_x+length*0.5, lw=width, color=color, zorder=zorder)
    ax.vlines(loc_x, loc_y-length*0.5, loc_y+length*0.5, lw=width, color=color, zorder=zorder)


def add_sources(ax, image, coords, marker = 'cross', 
    length=None, width=1., color='k', zorder=10., alpha = 1., 
    frame = 'icrs', unit=(u.hour, u.deg), equinox='J2000'):
    # check input
    if type(coords) == str:
        coords = [coords]
    elif type(coords) == list:
        pass
    else:
        print('ERROR\tadd_sources: coords must be str or list object.')
        return 0

    # image center
    cc = SkyCoord(image.cc[0], image.cc[1], frame=image.frame.lower(), 
            unit=(u.deg, u.deg), equinox=image.equinox)
    # plot
    for coord in coords:
        xc, yc = coord.split(' ')
        c_plt  = SkyCoord(xc, yc, frame=frame.lower(), 
            unit=unit, equinox=equinox)
        x_plt, y_plt = cc.spherical_offsets_to(c_plt)
        x_plt = x_plt.arcsec # deg to arcsec
        y_plt = y_plt.arcsec # deg to arcsec

        add_cross(ax, loc=(x_plt, y_plt), 
            length=length, width=width, color=color,
            zorder=zorder)


def add_line(ax, length=None, pa=0., cent=(0,0), width=1., color='k',
    ls='-', alpha=1., zorder=10.):
    if length is None:
        length = np.sqrt(
            (ax.get_xlim()[1] - ax.get_xlim()[0])**2 + (ax.get_ylim()[1] - ax.get_ylim()[0])**2
            )
    x0, y0, x1, y1 = length*0.5 * np.array([
        np.sin(np.radians(pa)), np.cos(np.radians(pa)),
        np.sin(np.radians(pa+180.)), np.cos(np.radians(pa+180.))
        ])
    ax.plot([x0 - cent[0], x1  - cent[0]],
        [y0 - cent[1], y1 - cent[1]], color=color, lw=width, 
        ls=ls, alpha=alpha, zorder=zorder)


def add_box(ax, xc, yc, xl, yl, width=1., color='k', 
    zorder=10., ls='--', angle=0.0):
    import matplotlib.patches as patches

    # box
    rect = patches.Rectangle((xc - xl*0.5, yc - yl*0.5), xl, yl, 
        ls=ls, color=color, fill=False, linewidth=width, zorder=zorder, angle=angle)
    ax.add_patch(rect)


def add_vectors(image, ax, norm=1., pivot='mid', 
    headwidth=1., headlength=0.001, width=0.01, color='red',):
    if 'vectors' not in image.__dict__.keys():
        print('ERROR\tadd_vectors: Vectors are not attached.')
        return 0

    f_norm = image.vectors[norm].values if type(norm) == str else norm
    # u: minus sign is necessary cuz pa is defined as an angle from top to left.
    u = - f_norm * np.sin(image.vectors.ANG*np.pi/180.)
    v = f_norm * np.cos(image.vectors.ANG*np.pi/180.)
    ax.quiver(image.vectors.delRA*3600., image.vectors.delDEC*3600., # x, y
              u, v, pivot=pivot, headwidth=headwidth, headlength=headlength,
              width = width, color=color)
    return ax


def add_beam(ax, bmaj: float, bmin: float, bpa: float,
 bcolor: str = 'k', loc: str = 'bottom left', alpha: float = 1.,
 zorder: float = 10., fill = True, ls = '-'):
    import matplotlib.patches as patches

    coords = {'bottom left': (0.1, 0.1),
    'bottom right': (0.9, 0.1),
    'top left': (0.1, 0.9),
    'top right': (0.9, 0.9),
    }
    if type(loc) == str:
        if loc in coords.keys():
            xy = coords[loc]
        else:
            print('ERROR\tplot_beam: loc keyword is not correct.')
            print('ERROR\tplot_beam: must be bottom left, bottom right, \
            top left, or top right.')
            return 0
    elif (type(loc) == tuple) & (len(loc) == 2):
        xy = loc
    else:
        print('ERROR\tplot_beam: loc keyword is not correct.')
        print('ERROR\tplot_beam: must be strings or tuple with two elements.')
        return 0

    # fill beam?
    if fill:
        fc = bcolor
        ec = None
    else:
        fc = 'none'
        ec = bcolor

    # plot
    bmin_plot, bmaj_plot = ax.transLimits.transform((0,bmaj)) - ax.transLimits.transform((bmin,0)) # data --> Axes coordinate
    beam = patches.Ellipse(xy=xy,
        width=bmin_plot, height=bmaj_plot, fc=fc, ec=ec, ls = ls,
        angle=bpa, transform=ax.transAxes, alpha=alpha, zorder=zorder)
    ax.add_patch(beam)


def add_colorbar_togrid(cim, grid, cbarlabel: str='',
    tickcolor: str = 'k', axiscolor: str = 'k', labelcolor: str = 'k',
    cbar_loc = 'right'):
    # parameter
    orientations = {
    'right': 'vertical',
    'left': 'vertical',
    'top': 'horizontal',
    'bottom': 'horizontal'}

    # plot
    cax  = grid.cbar_axes[0]
    cbar = plt.colorbar(cim, cax=cax, orientation=orientations[cbar_loc]) # ticks=cbarticks
    cax.toggle_label(True)
    cbar.ax.yaxis.set_tick_params(color=tickcolor) # tick color
    cbar.ax.spines["bottom"].set_color(axiscolor)  # axes color
    cbar.ax.spines["top"].set_color(axiscolor)
    cbar.ax.spines["left"].set_color(axiscolor)
    cbar.ax.spines["right"].set_color(axiscolor)

    # label
    if cbarlabel:
        cbar.ax.set_ylabel(cbarlabel,color=labelcolor) # label
    return cbar


def add_colorbar_toaxis_old(ax, cim=None, 
    cbaroptions: list = [], cbarlabel = '',
    ticks: list = None, fontsize: float = None, 
    tickcolor: str = 'k', labelcolor: str = 'k'):
    '''
    An oloder version of add_colorbar_toaxis.
    This old version uses divider.append_axes, which divides an axis into the main axis 
    and a sub axis for the colorbar. This changes the aspect ratio of the main axis.

    Parameters:
     - ax (mpl axes object): axes.
     - cim (color image):
     - cbaroptions (list): [loc, width, pad]
     - fontsize
     - ticks
    '''
    # parameter
    orientations = {
    'right': 'vertical',
    'left': 'vertical',
    'top': 'horizontal',
    'bottom': 'horizontal'}

    # color image
    if cim is not None:
        pass
    else:
        try:
            cim = ax.images[0] # assume the first one is a color map.
        except:
            print('ERROR\tadd_colorbar_toaxis: cannot find a color map.')
            return 0

    # setting for a color bar
    if len(cbaroptions) == 0:
        cbar_loc, cbar_wd, cbar_pad, cbar_lbl = ['right', '3%', '0%', cbarlabel]
    elif len(cbaroptions) == 3: 
        cbar_loc, cbar_wd, cbar_pad = cbaroptions
        cbar_lbl = cbarlabel
    elif len(cbaroptions) == 4:
        cbar_loc, cbar_wd, cbar_pad, cbar_lbl = cbaroptions
    else:
        print('WARNING\tadd_colorbar_toaxis: cbaroptions must have 4 elements. \
            Input is ignored.')
        cbar_loc, cbar_wd, cbar_pad, cbar_lbl = ['right', '3%', '0%', cbarlabel]
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes(cbar_loc, size=cbar_wd, pad=cbar_pad)

    # add a color bar
    cbar = plt.colorbar(cim, cax=cax, ticks=ticks, 
        orientation=orientations[cbar_loc], ticklocation=cbar_loc)
    cbar.set_label(cbar_lbl, fontsize=fontsize, color=labelcolor,)
    cbar.ax.tick_params(labelsize=fontsize, labelcolor=labelcolor, 
        color=tickcolor,)
    return cax, cbar


def add_colorbar_toaxis(ax, cim=None, 
    loc = 'right', pad = '3%', width = '3%',
    length = '100%', cbaroptions: list = None, 
    cbarlabel = '', ticks: list = None, fontsize: float = None, 
    tickcolor: str = 'k', labelcolor: str = 'k'):
    '''
    Add a colorbar to axis.

    Parameters
    ----------
    loc (str): Location of the colorbar. Must be choosen from right, left, top or bottom.
    pad (str or float): Pad between the image and colorbar. Must be given as percentage (e.g., '3%') or
        fraction (e.g, 0.03) of the full plot width.
    width (str or float): Width of the colorbar. Must be given as percentage or fraction of 
        the full plot width.
    length (str or float): Length of the colorbar. Must be given as percentage or fraction of 
        the full plot width.
    cbaroptions (list): Colorbar options which set location, pad, width all at once.
    ticks (list): Ticks of colorbar. Optional parameter.
    fontsize (float): Fontsize of the colorbar label and tick labels. Optional parameter.
    tickcolor, labelcolor (str): Set tick and label colors.
    '''
    # parameters to set orientation
    orientations = {
    'right': 'vertical',
    'left': 'vertical',
    'top': 'horizontal',
    'bottom': 'horizontal'}

    # colorbar options
    if cbaroptions is not None:
        if len(cbaroptions) == 3:
            cbar_loc, cbar_wd, cbar_pad = cbaroptions
        elif len(cbaroptions) == 4:
            cbar_loc, cbar_wd, cbar_pad, cbar_lbl = cbaroptions
        else:
            print('WARNING\tadd_colorbar_toaxis: cbaroptions must be a list object with three or four elements.')
            print('WARNING\tadd_colorbar_toaxis: Input cbaroptions are ignored.')
    else:
        cbar_loc = loc
        cbar_wd = width
        cbar_pad = pad
    # str to float
    if type(cbar_wd) == str: cbar_wd = float(cbar_wd.strip('%')) * 0.01
    if type(cbar_pad) == str: cbar_pad = float(cbar_pad.strip('%')) * 0.01
    if type(length) == str: length = float(length.strip('%')) * 0.01

    # check loc keyword
    if cbar_loc not in orientations.keys():
        print('ERROR\tadd_colorbar_toaxis: location keyword is wrong.')
        print('ERROR\tadd_colorbar_toaxis: it must be choosen from right, left, top or bottom.')
        return 0

    # color image
    if cim is not None:
        pass
    else:
        try:
            cim = ax.images[0] # assume the first one is a color map.
        except:
            print('ERROR\tadd_colorbar_toaxis: cannot find a color map.')
            return 0

    # set an inset axis
    # x0 and y0 of bounds are lower-left corner
    if cbar_loc == 'right':
        bounds = [1.0 + cbar_pad, 0., cbar_wd, length] # x0, y0, dx, dy
    elif cbar_loc == 'left':
        bounds = [0. - cbar_pad - cbar_wd, 0., cbar_wd, length]
    elif cbar_loc == 'top':
        bounds = [0., 1. + cbar_pad, length, cbar_wd]
    elif cbar_loc == 'bottom':
        bounds = [0., 0. - cbar_pad - cbar_wd, length, cbar_wd]

    # set a colorbar axis
    cax = ax.inset_axes(bounds, transform=ax.transAxes)
    cbar = plt.colorbar(cim, cax=cax, ticks=ticks, 
        orientation=orientations[cbar_loc], ticklocation=cbar_loc)
    cbar.set_label(cbarlabel)
    cbar.ax.tick_params(labelsize=fontsize, labelcolor=labelcolor, 
        color=tickcolor,)
    return cax, cbar


def add_scalebar(ax, scalebar: list, orientation='horizontal',
    loc: str = 'bottom right', barcolor: str = 'k', fontsize: float = 11.,
    lw: float = 2., zorder: float = 10., alpha: float = 1.,
    coordinate_mode = 'relative'):

    coords = {'bottom left': (0.1, 0.1),
            'bottom right': (0.9, 0.1),
            'top left': (0.1, 0.9),
            'top right': (0.9, 0.9),
            }
    offsets = {'bottom left': (0.05, -0.02),
            'bottom right': (-0.05, -0.02),
            'top left': (0.05, -0.02),
            'top right': (-0.05, -0.02),
            }

    if len(scalebar) == 5:
        barlength, bartext, loc, barcolor, fontsize = scalebar
        barlength = float(barlength)
        fontsize  = float(fontsize)

        if type(loc) == str:
            if loc in coords.keys():
                barx, bary = coords[loc]
                offx, offy = offsets[loc]
            else:
                print('CAUTION\tplot_beam: loc keyword is not correct.')
                return 0
        elif type(loc) == list:
            barx, bary = loc[0]
            txtx, txty = loc[1]
            offx = txtx - barx
            offy = txty - bary

        inv = ax.transLimits.inverted()
        if orientation == 'vertical':
            offy = 0.
            _, bary_l = ax.transLimits.transform(
                inv.transform((barx, bary)) - np.array([0., barlength*0.5]))
            _, bary_u = ax.transLimits.transform(
                inv.transform((barx, bary)) + np.array([0., barlength*0.5,]))
            ax.vlines(barx, bary_l, bary_u, 
                color=barcolor, lw=lw, zorder=zorder,
                transform=ax.transAxes, alpha=alpha)
            ax.text(barx + offx, bary + offy, bartext, fontsize=fontsize,
                color=barcolor, transform=ax.transAxes, 
                verticalalignment='center', horizontalalignment=loc.split(' ')[1])
        elif orientation == 'horizontal':
            offx = 0.
            barx_l, _ = ax.transLimits.transform(
                inv.transform((barx, bary)) - np.array([barlength*0.5, 0]))
            barx_u, _ = ax.transLimits.transform(
                inv.transform((barx, bary)) + np.array([barlength*0.5, 0]))
            ax.hlines(bary, barx_l, barx_u, 
                color=barcolor, lw=lw, zorder=zorder,
                transform=ax.transAxes, alpha=alpha)
            ax.text(barx + offx, bary + offy, bartext, fontsize=fontsize,
                color=barcolor, transform=ax.transAxes, 
                horizontalalignment='center', verticalalignment='top')
        else:
            print('ERROR\tadd_scalebar: orientation must be vertical or horizontal.')
            return 0
    elif len(scalebar) == 8:
        # read
        barx, bary, barlength, textx, texty, text, barcolor, barcsize = scalebar

        # to float
        barx      = float(barx)
        bary      = float(bary)
        barlength = float(barlength)
        textx     = float(textx)
        texty     = float(texty)

        if orientation == 'vertical':
            ax.vlines(barx, bary - barlength*0.5,bary + barlength*0.5, 
                color=barcolor, lw=lw, zorder=zorder, alpha=alpha)
        elif orientation == 'horizontal':
            ax.hlines(bary, barx - barlength*0.5,barx + barlength*0.5, 
                color=barcolor, lw=lw, zorder=zorder, alpha=alpha)
        else:
            print('ERROR\tadd_scalebar: orientation must be vertical or horizontal.')
            return 0

        ax.text(textx,texty,text,color=barcolor,fontsize=barcsize,horizontalalignment='center',verticalalignment='center')
    else:
        print ('ERROR\tadd_scalebar: scalebar must consist of 5 or 8 elements. Check scalebar.')



def change_aspect_ratio(ax, ratio, plottype='linear'):
    '''
    This function change aspect ratio of figure.
    Parameters:
        ax: ax (matplotlit.pyplot.subplots())
            Axes object
        ratio: float or int
            relative x axis width compared to y axis width.
    '''
    if plottype == 'linear':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    elif plottype == 'loglog':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    elif plottype == 'linearlog':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / np.log10(ax.get_ylim()[1]/ax.get_ylim()[0])
    elif plottype == 'loglinear':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    else:
        print('ERROR\tchange_aspect_ratio: plottype must be choosen from the types below.')
        print('   plottype can be linear or loglog.')
        print('   plottype=loglinear and linearlog is being developed.')
        return

    aspect = np.abs(aspect)
    aspect = float(aspect)
    ax.set_aspect(aspect)