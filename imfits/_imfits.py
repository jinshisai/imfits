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
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt


### Constants (in cgs)
clight     = 2.99792458e10 # light speed [cm s^-1]




### Imfits
class Imfits():
    '''
    Read a fits file, store the information, and draw maps.
    '''

    def __init__(self, infile, pv=False, frame=None, equinox='J2000'):
        self.file = infile
        self.data, self.header = fits.getdata(infile, header=True)

        self.frame = frame if frame else None

        if equinox:
            self.equinox = equinox

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


        # Coordinates
        # Coordinate frame
        if self.frame:
            pass
        else:
            if 'RADESYS' in header:
                self.frame = header['RADESYS'].strip()
                if 'EQUINOX' in header:
                    self.equinox = 'J' + str(header['EQUINOX'])
                else:
                    self.equinox = None
            else:
                print('WARRING\tread_header: Cannot find info. of the coordinate system.')
                print('WARRING\tread_header: Input frame by hand to get coordinates.')
                self.frame = None

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
            print ('CAUTION\tread_header: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
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
            self.nv = naxis_i[2]

            if naxis == 4:
                # stokes
                saxis = axes[3]
                saxis = saxis[:naxis_i[3]]
                self.ns = naxis_i[3]
            else:
                saxis = np.array([0.])
        else:
            vaxis = np.array([0.])
            saxis = np.array([0.])


        # frequency --> velocity
        keys_velocity = ['VRAD', 'VELO', 'VELO-LSR']
        if len(vaxis) > 1:
            # into Hz
            if 'CUNIT3' in self.header:
                if self.header['CUNIT3'] == 'GHz':
                    vaxis *= 1.e9
                elif self.header['CUNIT3'] == 'MHz':
                    vaxis *= 1.e6

            # to velocity
            if velocity:
                if label_i[2] in keys_velocity:
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
            self.res_off = self.beam[0] # bmaj


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

        # if stokes axis
        axes_out = np.array([xaxis, vaxis], dtype=object)
        if naxis >= 3:
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
        from astropy.coordinates import SkyCoord
        import astropy.units as u

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
        #xx = xx - ref_x
        #xx = xx*np.cos(np.radians(yy))
        # dec
        #yy = yy - ref_y
        cc   = SkyCoord(ref_x, ref_y, frame=self.frame.lower(), 
            unit=(u.deg, u.deg), equinox=self.equinox)
        grid = SkyCoord(self.xx_wcs, self.yy_wcs, frame=self.frame.lower(), 
            unit=(u.deg,u.deg), equinox=self.equinox)
        xx, yy = cc.spherical_offsets_to(grid)

        self.xx = xx.deg
        self.yy = yy.deg
        #self.xaxis = xx[ny//2,:]
        #self.yaxis = yy[:,nx//2]
        self.cc = np.array([ref_x, ref_y]) # coordinate center


    def shift_coord_center(self, coord_center, 
        interpolate=False):
        '''
        Shift the coordinate center.

        Args:
            coord_center: Put an coordinate for the map center.
               The shape must be '00h00m00.00s 00d00m00.00s', or
               'hh:mm:ss.ss dd:mm:ss.ss'. RA and DEC must be separated
               by space.
            regrid (bool): Interpolate data to make the new grid fit to the old grid
        '''
        # ra, dec
        cc = SkyCoord(self.cc[0], self.cc[1], frame=self.frame.lower(), 
            unit=(u.deg, u.deg), equinox=self.equinox)
        c_ra, c_dec = coord_center.split(' ')
        cc_new      = SkyCoord(c_ra, c_dec, frame=self.frame.lower(), 
            unit=(u.hour, u.deg), equinox=self.equinox)
        cra_deg     = cc_new.ra.degree               # in degree
        cdec_deg    = cc_new.dec.degree              # in degree
        new_cent    = np.array([cra_deg, cdec_deg])  # absolute coordinate of the new image center
        # offset of the center
        #x_offset, y_offset = cc.spherical_offsets_to(cc_new)

        # current coordinates
        #alpha = self.xx_wcs
        #delta = self.yy_wcs

        # shift of the center
        # approximately
        #alpha = (alpha - cra_deg)*np.cos(np.radians(delta))
        #delta = delta - cdec_deg
        # more exactly
        grid = SkyCoord(self.xx_wcs, self.yy_wcs, frame=self.frame.lower(), 
            unit=(u.deg, u.deg), equinox=self.equinox)
        # in offset
        grid_new = cc_new.spherical_offsets_to(grid)
        xx_new, yy_new = grid_new

        if interpolate:
            from scipy.interpolate import griddata
            # 2D --> 1D
            xinp     = xx_new.deg.reshape(xx_new.deg.size)
            yinp     = yy_new.deg.reshape(yy_new.deg.size)
            print('interpolating... May take time.')
            if self.naxis == 2:
                data_reg = griddata((xinp, yinp), self.data.reshape(self.data.size), 
                (self.xx, self.yy), method='cubic',rescale=True)
            elif self.naxis == 3:
                data_reg = np.array([ griddata((xinp, yinp), self.data[i,:,:].reshape(self.data[i,:,:].size), 
                    (self.xx, self.yy), method='cubic',rescale=True) for i in range(self.nv) ])
            elif self.naxis == 4:
                data_reg = np.array([[ griddata((xinp, yinp), self.data[i, j,:,:].reshape(self.data[i, j,:,:].size), 
                    (self.xx, self.yy), method='cubic',rescale=True) 
                for j in range(self.nv) ] for i in range(self.ns) ])
                #print(data_reg.shape, self.data.shape)
            else:
                print('ERROR\tImfits: NAXIS must be <= 4.')
                return 0
            # Interpolated data
            self.cc = new_cent
            self.data = data_reg
            # World coordinate
            self.xx_wcs += self.xx - xx_new.deg
            self.yy_wcs += self.yy - yy_new.deg
        else:
            xcent_indx = np.argmin(np.abs(yy_new), axis=0)[self.nx//2]
            ycent_indx = np.argmin(np.abs(xx_new), axis=1)[self.ny//2]

            # update
            self.xx = xx_new.deg #alpha
            self.yy = yy_new.deg #delta
            self.cc = new_cent
            self.xaxis = xx_new[ycent_indx, :].deg
            self.yaxis = yy_new[:, xcent_indx].deg
            #self.xaxis = xx_new[self.nx//2,:].deg # or self.xaxis -= x_offset.deg
            #self.yaxis = yy_new[:, self.ny//2].deg # or -= y_offset.deg


    def convert_units(self, conversion='IvtoTb'):
        '''
        Unit conversion

        Parameters
        ----------
            coversion (str):
             IvtoTb -- Convert Iv to Tb
             TbtoIv -- Convert Tb to Iv
             pas2topbm --  Convert Jy/arcsec^2 to Jy/beam
        '''
        from .mapunit import IvTOJT, TbTOIv, pas2TOpbm

        if conversion == 'IvtoTb':
            self.data = IvTOJT(self.data, self.restfreq, 
                self.beam[0]/3600, self.beam[1]/3600.)
        elif conversion == 'TbtoIv':
            self.data = TbTOIv(self.data, self.restfreq, 
                self.beam[0]/3600, self.beam[1]/3600.)
        elif conversion == 'pas2topbm':
            self.data = pas2TOpbm(self.data, self.delx, 
                self.dely, self.beam[0]/3600., self.beam[1]/3600.)
        else:
            print ('ERROR\tconvert_units: \
                Currently supported conversions are\
                Iv <--> Tb and Jy/arcsec^2 to Jy/beam.')
            return


    def attach_vectors(self, infile, sep='\s+', rotate=False,
        comment='=', pandas_args=[]):
        '''
        Read and contain polarization/B-field vectors.

        Parameters
        ----------

        '''
        import pandas as pd

        # read
        if len(pandas_args):
            self.vectors = pd.read_csv(infile, *pandas_args)
        else:
            self.vectors = pd.read_csv(infile, sep=sep, comment=comment)

        if rotate:
            self.vectors.ANG += 90. # rotate by 90 degree

        # offset from center
        cc = SkyCoord(*self.cc, frame=self.frame.lower(), 
            unit=(u.deg, u.deg), equinox=self.equinox)
        pcoord = SkyCoord(self.vectors.RA.values, self.vectors.DEC.values, 
            frame=self.frame.lower(), unit=(u.hour, u.degree), equinox=self.equinox)
        pa_offset  = cc.position_angle(pcoord)
        sep_offset = cc.separation(pcoord)
        #cc.directional_offset_by(pa_offset, sep_offset) # return absolute ra, dec
        dra, ddec = cc.spherical_offsets_to(pcoord)

        # add offset from cetner
        self.vectors.insert(self.vectors.columns.get_loc('DEC')+1, 'delDEC', ddec.value)
        self.vectors.insert(self.vectors.columns.get_loc('DEC')+1, 'delRA', dra.value)


    def getmoments(self, moment=[0], vrange=[], threshold=[],
        outfits=True, outname=None, overwrite=False, i_stokes=0):
        '''
        Calculate moment maps.

        moment (list): Index of moments that you want to calculate.
        vrange (list): Velocity range to calculate moment maps.
        threshold (list): Threshold to clip data. Shoul be given as [minimum intensity, maximum intensity]
        outfits (bool): Output as a fits file?
        outname (str): Output file name if outfits=True.
        overwrite (bool): If overwrite an existing fits file or not.
        '''

        # data check
        data  = self.data
        if len(data.shape) <= 2:
            print ('ERROR\tgetmoments: Data must have more than three axes to calculate moments.')
            return
        elif len(data.shape) == 3:
            pass
        elif len(data.shape) == 4:
            data = data[i_stokes,:,:,:]
            xaxis, yaxis, vaxis, saxis = self.axes
        else:
            print ('ERROR\tgetmoments: Data have more than five axes. Cannot recognize.')
            return 0

        # axes
        if len(self.axes) == 3:
            xaxis, yaxis, vaxis = self.axes
        elif len(self.axes) == 4:
            xaxis, yaxis, vaxis, saxis = self.axes
        else:
            print ('ERROR\tgetmoments: Data shape is unexpected.')
            print ('ERROR\tgetmoments: Data have more than five axes.')
            return 0
        nx = self.nx
        ny = self.ny
        delv = self.delv

        if len(vrange) == 2:
            index = np.where( (vaxis >= vrange[0]) & (vaxis <= vrange[1]))
            data  = data[index[0],:,:]
            vaxis = vaxis[index[0]]

        if len(threshold) == 2:
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
        if delv < 0.:
            delv *= -1
        mom0 = np.array([[np.sum(delv*data[:,j,i]) for i in range(nx)] for j in range(ny)])
        w2   = np.array([[np.sum(delv*data[:,j,i]*delv*data[:,j,i]) for i in range(nx)]
            for j in range(ny)]) # Sum_i w_i^2
        ndata = np.array([
            [len(np.nonzero(data[:,j,i])[0]) for i in range(nx)] 
            for j in range(ny)]) # number of data points used for calculations


        if 0 in moment:
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
            #       np.sum(vaxis[np.where(data[:,j,i] > 0)]*delv*vaxis[np.where(data[:,j,i] > 0)]*delv)
            #       for i in range(nx)] for j in range(ny)])
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

        if 8 in moment:
            mom8 = np.nanmax(data, axis=0)
            moments.append(mom8)

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


    # trim data to make it light
    def trim_data(self,
        xlim: list = [],
        ylim: list = [],
        vlim: list = [],
        slim: list = []):
        '''
        Trim a cube image without interpolation to fit to given axis ranges.

        Parameters
        ----------
            xlim (list): x range. Must be given as [xmin, xmax] in arcsec.
            ylim (list): y range. Must be given as [ymin, ymax] in arcsec.
            vlim (list): v range. Must be given as [vmin, vmax] in km s^-1.
        '''
        def index_between(t, tlim, mode='all'):
            if not (len(tlim) == 2):
                if mode=='all':
                    return np.full(np.shape(t), True)
                elif mode == 'edge':
                    if len(t.shape) == 1:
                        return tuple([[0, len(t)-1]])
                    else:
                        return tuple([[0, t.shape[i]] for i in range(len(t.shape))])
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

        xlim = np.array(xlim)/3600. # arcsec --> deg
        ylim = np.array(ylim)/3600. # arcsec --> deg
        if self.naxis == 2:
            yimin, yimax = index_between(self.yaxis, ylim, mode='edge')[0]
            ximin, ximax = index_between(self.xaxis, xlim, mode='edge')[0]
            self.data    = self.data[yimin:yimax+1, ximin:ximax+1]
            self.xx      = self.xx[yimin:yimax+1, ximin:ximax+1]
            self.yy      = self.yy[yimin:yimax+1, ximin:ximax+1]
            self.xx_wcs  = self.xx_wcs[yimin:yimax+1, ximin:ximax+1]
            self.yy_wcs  = self.yy_wcs[yimin:yimax+1, ximin:ximax+1]
            self.yaxis   = self.yaxis[index_between(self.yaxis, ylim)]
            self.xaxis   = self.xaxis[index_between(self.xaxis, xlim)]
            self.nx = len(self.xaxis)
            self.ny = len(self.yaxis)
        elif self.naxis == 3:
            vimin, vimax = index_between(self.vaxis, vlim, mode='edge')[0]
            yimin, yimax = index_between(self.yaxis, ylim, mode='edge')[0]
            ximin, ximax = index_between(self.xaxis, xlim, mode='edge')[0]
            self.data    = self.data[vimin:vimax+1, yimin:yimax+1, ximin:ximax+1]
            self.xx      = self.xx[yimin:yimax+1, ximin:ximax+1]
            self.yy      = self.yy[yimin:yimax+1, ximin:ximax+1]
            self.xx_wcs  = self.xx_wcs[yimin:yimax+1, ximin:ximax+1]
            self.yy_wcs  = self.yy_wcs[yimin:yimax+1, ximin:ximax+1]
            self.vaxis   = self.vaxis[index_between(self.vaxis, vlim)]
            self.yaxis   = self.yaxis[index_between(self.yaxis, ylim)]
            self.xaxis   = self.xaxis[index_between(self.xaxis, xlim)]
            self.nx = len(self.xaxis)
            self.ny = len(self.yaxis)
            self.nv = len(self.vaxis)
        elif self.naxis == 4:
            simin, simax = index_between(self.saxis, slim, mode='edge')[0]
            vimin, vimax = index_between(self.vaxis, vlim, mode='edge')[0]
            yimin, yimax = index_between(self.yaxis, ylim, mode='edge')[0]
            ximin, ximax = index_between(self.xaxis, xlim, mode='edge')[0]
            self.data    = self.data[simin:simax+1, vimin:vimax+1, yimin:yimax+1, ximin:ximax+1]
            self.xx      = self.xx[yimin:yimax+1, ximin:ximax+1]
            self.yy      = self.yy[yimin:yimax+1, ximin:ximax+1]
            self.xx_wcs  = self.xx_wcs[yimin:yimax+1, ximin:ximax+1]
            self.yy_wcs  = self.yy_wcs[yimin:yimax+1, ximin:ximax+1]
            self.saxis   = self.saxis[index_between(self.saxis, slim)]
            self.vaxis   = self.vaxis[index_between(self.vaxis, vlim)]
            self.yaxis   = self.yaxis[index_between(self.yaxis, ylim)]
            self.xaxis   = self.xaxis[index_between(self.xaxis, xlim)]
            self.nx = len(self.xaxis)
            self.ny = len(self.yaxis)
            self.nv = len(self.vaxis)
            self.ns = len(self.saxis)
        else:
            print('trim_data: Invalid data shape.')
            return -1

    def get_mapextent(self, unit='arcsec'):
        xmin, xmax = self.xaxis[[0,-1]]
        ymin, ymax = self.yaxis[[0,-1]]
        extent = (xmin-0.5*self.delx, 
                xmax+0.5*self.delx, 
                ymin-0.5*self.dely, 
                ymax+0.5*self.dely)
        if unit == 'arcsec':
            extent = tuple(np.array(list(extent))*3600.) # deg --> arcsec
        self.extent = extent
        return extent


    def get_1dresolution(self, pa):
        '''
        Calculate beam size along a certain direction given by a position angle.

        Parameters
        ----------
         - self: Imfits object
         - pa: position angle of a direction in which you want to calculate a resolution.

        Equations
        ---------
         Describe a beam with an ellipse:
          (x/bmaj)**2 + (y/bmin)**2 = 1
          taking the beam major axis to be x-axis.

         Line toward an angle theta measured from x-axis to y-axis:
          y = x*tan(theta)

         The crossing point is the resolution in a one-dimensional slice.
          --> Solve these equations to get r, which is the resolution.
        '''
        bmaj, bmin, bpa = self.beam
        del_pa = pa - bpa          # angle from the major axis of the beam (i.e., x-axis) to the 1D-slice direction.
        del_pa = del_pa*np.pi/180. # radian
        term_sin = (np.sin(del_pa)/bmin)**2.
        term_cos = (np.cos(del_pa)/bmaj)**2.
        res_off  = np.sqrt(1./(term_sin + term_cos))
        return res_off


    def get_relative_coordinates(self, coord, frame=None):
        '''
        Convert a coordinate to a relative coordinate.

        Parameters
        ----------
            coord (str): Absolute coordinate.
        '''
        if frame is not None:
            pass
        else:
            frame = self.frame.lower()
        cc = SkyCoord(self.cc[0], self.cc[1], frame=self.frame.lower(), 
            unit=(u.deg, u.deg), equinox=self.equinox)
        ci_ra, ci_dec = coord.split(' ')
        if (':' in ci_ra) or ('h' in ci_ra):
            unit=(u.hour, u.deg)
        else:
            unit=(u.deg, u.deg)
        ci = SkyCoord(ci_ra, ci_dec, frame=frame, 
            unit=unit, equinox=self.equinox)

        # in offset
        ci_rel = cc.spherical_offsets_to(ci)
        ra_out, dec_out = ci_rel[0].deg, ci_rel[1].deg
        return ra_out, dec_out