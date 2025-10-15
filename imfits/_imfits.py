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
import astropy.coordinates
import astropy.units as u
#import matplotlib.pyplot as plt
from scipy.interpolate import griddata


### Constants (in cgs)
clight     = 2.99792458e10 # light speed [cm s^-1]


from .mapunit import IvTOJT, TbTOIv, pas2TOpbm, IvTOTex
from . import iu

### Imfits
class Imfits():
    '''
    Read a fits file, store the information, and draw maps.
    '''

    def __init__(self, infile, pv=False, 
        frame=None, equinox='J2000', axesorder=(), 
        velocity=True, flip_vaxis = True, 
        generate_empty_object = False,
        beam_savetype = 'full', beam_index = 0):
        '''
        Read and save fits data and header information.


        Parameters
        ----------
         infile (str): Path to the fits file.
         pv (bool): If the input fits file is for the position-velocity image or not.
         frame (str or None): Optional. Coordinate frame, 
            which will be required if it is missing in the fits header.
         equinox (str): Optional. Equinox of the coordinate frame. It will be used 
            when the coordinate frame is Jxxxx and the equinox is missing in the header.
         axesorder (tuple): An order of axes, if they need to be reordered.
        '''
        self.file = infile

        # Generate empty Imfits object
        if generate_empty_object:
            return

        self.data, self.header = fits.getdata(infile, header=True)

        self.ifpv = pv
        if pv:
            self.read_pvfits(axesorder=axesorder)
            self.frame = None
            self.equinox = None
        else:
            self.read_header(frame=frame, 
                axesorder=axesorder, 
                velocity=velocity, equinox=equinox,
                beam_savetype = beam_savetype, beam_index = beam_index)
            self.get_coordinates()

        if self.naxis >= 3:
            if (flip_vaxis) * (self.delv < 0):
                self.flip_vaxis()

        self.get_mapextent()


    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj


    def copy(self):
        return self.__copy__()


    def read_coordinate_frame(self):
        header = self.header
        if 'RADESYS' in header:
            self.frame = header['RADESYS'].strip()
            self.equinox = 'J' + str(header['EQUINOX']) if 'EQUINOX' in header else None
        elif any([i in header for i in ('EQUINOX', 'EPOCH')]):
            key = 'EQUINOX' if 'EQUINOX' in header else 'EPOCH'
            self.equinox = 'J' + str(header[key])
            self.frame = 'fk5' if header[key] >= 1984.0 else 'fk4'
        else:
            print('WARRING\tread_coordinate_frame: Cannot find the coordinate frame in the header.')
            print('WARRING\tread_coordinate_frame: ICRS is assumed. Input frame by hand to use another frame.')
            self.frame = 'icrs'
            self.equinox = None


    def read_header(self, frame=None, 
        velocity=True, axesorder=(), equinox='J2000',
        beam_savetype = 'full', beam_index = 0):
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
        self.naxis_i  = naxis_i
        self.label_i  = label_i
        self.refpix_i = refpix_i
        self.refval_i = refval_i
        if 'CDELT1' in header:
            del_i = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree
            self.del_i = del_i
        else:
            self.del_i = []


        # beam size (degree)
        if 'BMAJ' in header:
            bmaj     = header['BMAJ'] # degree
            bmin     = header['BMIN'] # degree
            bpa      = header['BPA']  # degree
            self.beam = np.array([bmaj*3600., bmin*3600., bpa]) # Get in arcsec
            self.multibeam = False
        elif 'CASAMBM' in header:
            self.multibeam = True
            self.beam_savetype = beam_savetype
            self.read_casa_multibeam(
                savetype = beam_savetype, index = beam_index)
        else:
            self.beam = None
            self.multibeam = False


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
        if frame is not None:
            self.frame = frame
            self.equinox = equinox
        else:
            self.read_coordinate_frame()

        # read projection type
        try:
            ra_indx = [i for i in range(self.naxis) if 'RA' in self.label_i[i]][0]
            projection = label_i[ra_indx].replace('RA---','')
        except:
            print ('WARNING\tread_header: Cannot read information about projection from header.')
            print ('WARNING\tread_header: Projection SIN is used assuming radio interferometric data.')
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
        self._axes = axes
        # reorder if neccessary
        if len(axesorder) == 0:
            # auto re-ordering
            axes_keys = {
            'RA---'+self.projection: 0, 
            'DEC--'+self.projection: 1, 
            'FREQ': 2, 
            'STOKES': 3,}
            # if OFFSET, FREQ, STOKES axes are found
            if all([i in axes_keys.keys() for i in self.label_i]):
                self.reorder_axes(tuple([axes_keys[i] for i in self.label_i]))
        elif len(axesorder): self.reorder_axes(axesorder)


        # x & y (RA & DEC)
        xaxis = axes[0]
        xaxis = xaxis[:self.naxis_i[0]] # offset
        yaxis = axes[1]
        yaxis = yaxis[:self.naxis_i[1]] # offset
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.delx = xaxis[1] - xaxis[0]
        self.dely = yaxis[1] - yaxis[0]
        self.nx = self.naxis_i[0]
        self.ny = self.naxis_i[1]

        # frequency & stokes
        if naxis >= 3:
            # frequency
            vaxis = axes[2]
            vaxis = vaxis[:self.naxis_i[2]] + self.refval_i[2]  # frequency, absolute
            self.nv = self.naxis_i[2]

            if naxis == 4:
                # stokes
                saxis = axes[3]
                saxis = saxis[:self.naxis_i[3]]
                self.ns = self.naxis_i[3]
            else:
                saxis = np.array([0.])
        else:
            vaxis = np.array([0.])
            saxis = np.array([0.])


        # frequency --> velocity
        keys_velocity = ['VRAD', 'VELO',]
        if len(vaxis) > 1:
            # into Hz
            if 'CUNIT3' in self.header:
                if self.header['CUNIT3'] == 'GHz':
                    vaxis *= 1.e9
                elif self.header['CUNIT3'] == 'MHz':
                    vaxis *= 1.e6

            # to velocity
            if velocity:
                print ('The third axis is ', self.label_i[2])
                if any([i in self.label_i[2] for i in keys_velocity]):
                    #print ('The third axis is ', self.label_i[2])
                    # m/s --> km/s
                    vaxis    = vaxis*1.e-3 # m/s --> km/s
                    #del_v    = del_v*1.e-3
                    #refval_v = refval_v*1.e-3
                    #vaxis    = vaxis*1.e-3
                elif self.label_i[2] == 'FREQ':
                    #print ('The third axis is ', self.label_i[2])
                    print ('Convert frequency to velocity')
                    if self.restfreq is None:
                        vaxis = [1] * self.nv
                    else:
                        # frequency (Hz) --> radio velocity (km/s)
                        vaxis = clight*(1.-vaxis/restfreq)*1.e-5
                else:
                    print ('No conversion applied.')

                if len(vaxis) >= 2:
                    self.delv = vaxis[1] - vaxis[0]
                else:
                    self.delv = 1.
        else:
            self.delv = 1.

        self.vaxis = vaxis
        self.saxis = saxis

        axes = np.array([xaxis, yaxis, vaxis, saxis], dtype=object)
        self.axes  = axes


    def flip_vaxis(self):
        '''
        Flip the velocity axis. The order of axes is assumed to be (s, v, y, x).
        Re-order axis first if this is not the case.
        '''
        self.vaxis = self.vaxis[::-1]
        self.delv = self.vaxis[1] - self.vaxis[0]
        if self.ifpv:
            self.data = self.data[:,::-1]
        else:
            if self.naxis == 3:
                self.data = self.data[::-1,:,:]
            elif self.naxis == 4:
                self.data = self.data[:, ::-1, :, :]
            else:
                print('ERROR\tflip_vaxis: NAXIS must be 3 or 4.')
                print('ERROR\tflip_vaxis: Check axes of the input fits file.')
                return 0


    def read_casa_multibeam(self, 
        savetype = 'full', index = 0):
        '''
        Read CASA Multibeam.
        '''
        hdul = fits.open(self.file)
        print(hdul.info())
        bmaj = hdul['BEAMS'].data['BMAJ']
        bmin = hdul['BEAMS'].data['BMIN']
        bpa = hdul['BEAMS'].data['BPA']

        if savetype == 'full':
            self.beam = np.array([bmaj, bmin, bpa])
        elif savetype == 'average':
            bmaj = np.mean(bmaj)
            bmin = np.mean(bmin)
            bpa = np.mean(bpa)
            self.beam = np.array([bmaj, bmin, bpa])
        elif savetype == 'single':
            bmaj = bmaj[index]
            bmin = bmin[index]
            bpa = bpa[index]
            self.beam = np.array([bmaj, bmin, bpa])
        else:
            print('WARNING\tread_casa_multibeam: savetype must be full, average or single.')
            print('WARNING\tread_casa_multibeam: save full beam sizes.')
            self.beam = np.array([bmaj, bmin, bpa])

        hdul.close()
        self.beam_savetype = savetype




    # Read fits file of Poistion-velocity (PV) diagram
    def read_pvfits(self, axesorder=()):
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
        self.naxis_i  = naxis_i
        self.label_i  = label_i
        self.refpix_i = refpix_i
        self.refval_i = refval_i
        if 'CDELT1' in header:
            del_i = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree
            self.del_i = del_i
        else:
            self.del_i = []


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
        self._axes = axes


        # reorder if neccessary
        if len(axesorder) == 0:
            # auto re-ordering
            axes_keys = {'OFFSET': 0, 'FREQ': 1, 'STOKES': 2,}
            # if OFFSET, FREQ, STOKES axes are found
            if all([i in axes_keys.keys() for i in self.label_i]):
                self.reorder_axes(tuple([axes_keys[i] for i in self.label_i]))
        elif len(axesorder): self.reorder_axes(axesorder)


        # x & v axes
        xaxis = self._axes[0]
        vaxis = self._axes[1]
        xaxis = xaxis[:self.naxis_i[0]]               # offset
        vaxis = vaxis[:self.naxis_i[1]] + self.refval_i[1] # frequency, absolute


        # check unit of offest
        if 'CUNIT1' in header:
            unit_i = np.array([header['CUNIT'+str(i+1)] for i in range(naxis)]) # degree
            if unit_i[0] == 'degree' or unit_i[0] == 'deg':
                # degree --> arcsec
                xaxis    = xaxis*3600.
                self.del_i[0] = self.del_i[0]*3600.
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
            saxis = self._axes[2]
            saxis = saxis[:self.naxis_i[2]]
            axes_out = np.array([xaxis, vaxis, saxis], dtype=object)


        # get delta
        delx = xaxis[1] - xaxis[0]
        delv = vaxis[1] - vaxis[0]

        self.axes  = axes_out
        self.xaxis = xaxis
        self.vaxis = vaxis
        self.delx  = delx
        self.delv  = delv
        self.nx = len(xaxis)
        self.nv = len(vaxis)


    def reorder_axes(self, order):
        '''
        Reorder fits axes.

        Parameter
        ---------
         order (tuple): New order of axes. Must be given as tuple within which
            new order is specified by integers. E.g., in order to swich the 2nd and 4th axes,
            the input will be (0, 3, 2, 1). Note that the 1st axis index is zero following the python rule.
        '''
        # check
        if type(order) != tuple:
            print('ERROR\reorder_axes\t: input new order must be tuple')
            return 0
        elif all([type(i) == int for i in order]) == False:
            print('ERROR\reorder_axes\t: indices in the tuple must be intergers')
            return 0
        elif len(order) != self.naxis:
            print('ERROR\reorder_axes\t: The input order and fits axes must have the same dimension.')
            return 0

        outbox = []
        for i in [self.naxis_i, self.label_i, self.refpix_i, self.refval_i, self.del_i, self._axes]:
            if len(i) != 0:
                i = [i[j] for j in order]
            outbox.append(i)
        # replace with reordered ones
        self.naxis_i, self.label_i, self.refpix_i, self.refval_i, self.del_i, self._axes = outbox


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
        interpolate=False, zero_center = True):
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
            # 2D --> 1D
            xinp     = xx_new.deg.reshape(xx_new.deg.size)
            yinp     = yy_new.deg.reshape(yy_new.deg.size)
            # align the new coordinate center with the map center
            x_org = np.array(range(self.nx), dtype = np.float64) - self.nx//2
            x_org *= self.delx
            y_org = np.array(range(self.ny), dtype = np.float64) - self.ny//2
            y_org *= self.dely
            if (self.nx%2 == 0) & (zero_center is False):
                x_org += self.delx * 0.5
            if (self.ny%2 == 0) & (zero_center is False):
                y_org += self.dely * 0.5
            xx_org, yy_org = np.meshgrid(x_org, y_org)
            print('interpolating... May take time.')
            if self.naxis == 2:
                data_reg = griddata((xinp, yinp), self.data.reshape(self.data.size), 
                (xx_org, yy_org), method='cubic',rescale=True)
            elif self.naxis == 3:
                data_reg = np.array([ griddata((xinp, yinp), self.data[i,:,:].reshape(self.data[i,:,:].size), 
                    (xx_org, yy_org), method='cubic',rescale=True) for i in range(self.nv) ])
            elif self.naxis == 4:
                data_reg = np.array([[ griddata((xinp, yinp), self.data[i, j,:,:].reshape(self.data[i, j,:,:].size), 
                    (xx_org, yy_org), method='cubic',rescale=True) 
                for j in range(self.nv) ] for i in range(self.ns) ])
                #print(data_reg.shape, self.data.shape)
            else:
                print('ERROR\tImfits: NAXIS must be <= 4.')
                return 0
            # Interpolated data
            self.cc = new_cent
            self.data = data_reg
            # World coordinate
            wcs_new = cc_new.spherical_offsets_by(
                astropy.coordinates.Angle(xx_org, unit = u.deg), 
                astropy.coordinates.Angle(yy_org, unit = u.deg))
            self.xx_wcs, self.yy_wcs = wcs_new.ra.degree, wcs_new.dec.degree
            #self.xx_wcs  += self.xx - xx_new.deg
            #self.yy_wcs += self.yy - yy_new.deg
            # updated coordinates relative to new center
            self.xx = xx_org #alpha
            self.yy = yy_org #delta
            self.xaxis = x_org
            self.yaxis = y_org
        else:
            #print(xx_new.shape)
            #print(self.nx, len(np.argmin(np.abs(yy_new.deg), axis=0)))
            #print(self.ny, len(np.argmin(np.abs(xx_new.deg), axis=1)))
            ycent_indx = np.argmin(np.abs(yy_new), axis=0)[self.nx//2]
            xcent_indx = np.argmin(np.abs(xx_new), axis=1)[self.ny//2]
            #print(xcent_indx, ycent_indx)

            # update
            self.xx = xx_new.deg #alpha
            self.yy = yy_new.deg #delta
            self.cc = new_cent
            self.xaxis = xx_new[ycent_indx, :].deg
            self.yaxis = yy_new[:, xcent_indx].deg
            #self.xaxis = xx_new[self.nx//2,:].deg # or self.xaxis -= x_offset.deg
            #self.yaxis = yy_new[:, self.ny//2].deg # or -= y_offset.deg


    def estimate_noise(self, nitr=1000, thr=2.3):
        '''
        Estimate map noise by calculating rms iteratively.
        For more precise measurements of the noise level,
        use getrms_cube method.

        Parameters
        ----------
        nitr (int): Number of the maximum iteration.
        thr (float): Threshold where iteration stops.
        '''

        d = self.data.copy()
        rms = np.sqrt(np.nanmean(d*d))
        for i in range(nitr):
            rms_p = rms
            d[d >= thr*rms] = np.nan
            rms = np.sqrt(np.nanmean(d*d))

            if (rms - rms_p)*(rms - rms_p) < 1e-20:
                return rms

        print('Reach maximum number of iteration.')
        return rms


    def getrms_cube(self, vwindows=[[]], 
        radius=None, saxis=0, mask = None):
        '''
        Calculate rms based on line free channels.

        Parameters:
        '''
        naxis = self.naxis
        if naxis == 3:
            nv, ny, nx = self.data.shape
            if nv <= 1:
                print ('ERROR\tgetrms_cube: No proper frequency axis is found.')
                print ('ERROR\tgetrms_cube: Must have the frequency axis.')
                return 0
            data = self.data.copy()
        elif naxis == 4:
            ns, nv, ny, nx = self.data.shape
            if nv <= 1:
                print ('ERROR\tgetrms_cube: No proper frequency axis is found.')
                print ('ERROR\tgetrms_cube: Must have the frequency axis.')
                return 0
            data = self.data[saxis,:,:,:].copy()
        else:
            print ('ERROR\tgetrms_cube: NAXIS of fits is neither 3 nor 4.')
            print ('ERROR\tgetrms_cube: The input fits file must have 3 or 4 axes.')
            return 0


        vaxis = self.vaxis.copy()
        xx = self.xx.copy() * 3600.
        yy = self.yy.copy() * 3600.
        rr = np.sqrt(xx * xx + yy * yy)

        if type(vwindows) != list:
            print('ERROR\trmsmap: Input vwindows format is wrong.')
            print('ERROR\trmsmap: Must be a list object.')
            return 0

        if (type(vwindows[0]) == float) | (type(vwindows[0]) == np.float64):
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

        if radius is not None:
            for i in range(d_masked.shape[0]):
                _d = d_masked[i,:,:]
                _d[rr > radius] = np.nan
                d_masked[i,:,:] = _d

        if mask is not None:
            for i in range(d_masked.shape[0]):
                _d = d_masked[i,:,:]
                _d[mask] = np.nan
                d_masked[i,:,:] = _d

        # rms for each pannel
        _rms = np.sqrt(np.nanmean(d_masked * d_masked, axis=(1,2)))
        return np.nanmean(_rms)


    def binning(self, nbin, axis = 'velocity'):
        '''
        Rebin data.

        Parameters
        ----------
        nbin (int): Binning factor. E.g., if nbin = 3, original three pixels will be
         binned into a single pixel.
        axis (str): 'velocity' or 'xy'.
        '''
        if (axis == 'velocity') | (axis == 'v'):
            self.binning_v(nbin)
        elif axis == 'xy':
            self.binning_xy(nbin)
        else:
            print('ERROR\tbinning: axis must be xy or v.')
            return 0


    def binning_v(self, nbin,):
        '''
        Rebin data along velocity or frequency axis.
        '''
        nv = self.nv
        ncut = nv%nbin
        if self.naxis == 3:
            d = self.data[:nv-ncut,:,:]
            _d = np.array([
                d[i::nbin, :, :] for i in range(nbin)])
            self.data = np.nanmean(_d, axis = 0)
        elif self.naxis == 4:
            d = self.data[:,:nv-ncut,:,:]
            _d = np.array([
                d[:, i::nbin, :, :] for i in range(nbin)])
            self.data = np.nanmean(_d, axis = 0)
        else:
            print('ERROR\tbinning_v: naxis must be 3 or 4.')
            return 0
        v = self.vaxis[:nv-ncut]
        _v = np.array([v[i::nbin] for i in range(nbin)])
        self.vaxis = np.nanmean(_v, axis = 0)
        self.delv = self.vaxis[1] - self.vaxis[0]
        self.nv = len(self.vaxis)


    def binning_xy(self, nbin,):
        '''
        Rebin data in the xy spatial plane.
        '''
        nx, ny = self.nx, self.ny
        if type(nbin) == int:
            nxbin = nbin
            nybin = nbin
            nxcut = nx%nbin
            nycut = ny%nbin
        elif (type(nbin) == list) | (type(nbin) == tuple):
            nxbin, nybin = nbin
            nxcut = nx%nxbin
            nycut = ny%nybin
        else:
            print('ERROR\tbinning_xy: input nbin type is wrong.')
            print('ERROR\tbinning_xy: nbin must be int, or list or tuple having two elements.')
            return 0


        if (nxcut != 0):
            print('CAUTION\tbinning_xy: nx is not a multiple of nbin.')
            print('CAUTION\tbinning_xy: remaining pixels are cut off.')
            print('CAUTION\tbinning_xy: image center could shift.')
        if (nycut != 0):
            print('CAUTION\tbinning_xy: ny is not a multiple of nbin.')
            print('CAUTION\tbinning_xy: remaining pixels are cut off.')
            print('CAUTION\tbinning_xy: image center could shift.')

        if self.naxis == 2:
            d = self.data[:ny-nycut,:nx-nxcut]
            _d = np.array([
                d[j::nbin, i::nbin] for j in range(nybin) for i in range(nxbin)
                ])
        elif self.naxis == 3:
            d = self.data[:,:ny-nycut,:nx-nxcut]
            _d = np.array([
                d[:, j::nbin, i::nbin] for j in range(nybin) for i in range(nxbin)
                ])
        elif self.naxis == 4:
            d = self.data[:,:,:ny-nycut,:nx-nxcut]
            _d = np.array([
                d[:,:,j::nbin, i::nbin] for j in range(nybin) for i in range(nxbin)
                ])
        else:
            print('ERROR\tbinning_xy: naxis must be 2 to 4.')
            return 0
        # binning data
        self.data = np.nanmean(_d, axis = 0)

        # update axes
        x = self.xaxis[:nx-nxcut]
        y = self.yaxis[:ny-nycut]
        _x = np.array([x[i::nxbin] for i in range(nxbin)])
        _y = np.array([y[i::nybin] for i in range(nybin)])
        self.xaxis = np.nanmean(_x, axis = 0)
        self.yaxis = np.nanmean(_y, axis = 0)
        self.delx = self.xaxis[1] - self.xaxis[0]
        self.dely = self.yaxis[1] - self.yaxis[0]
        self.nx = len(self.xaxis)
        self.ny = len(self.yaxis)
        xx      = self.xx[:ny-nycut,:nx-nxcut]
        yy      = self.yy[:ny-nycut,:nx-nxcut]
        xx_wcs  = self.xx_wcs[:ny-nycut,:nx-nxcut]
        yy_wcs  = self.yy_wcs[:ny-nycut,:nx-nxcut]
        self.xx = np.nanmean(
            np.array([
                xx[j::nbin, i::nbin] for j in range(nybin) for i in range(nxbin)
                ]), axis = 0
            )
        self.yy = np.nanmean(
            np.array([
                yy[j::nbin, i::nbin] for j in range(nybin) for i in range(nxbin)
                ]), axis = 0
            )
        self.xx_wcs = np.nanmean(
            np.array([
                xx_wcs[j::nbin, i::nbin] for j in range(nybin) for i in range(nxbin)
                ]), axis = 0
            )
        self.yy_wcs = np.nanmean(
            np.array([
                yy_wcs[j::nbin, i::nbin] for j in range(nybin) for i in range(nxbin)
                ]), axis = 0
            )



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

        if conversion == 'IvtoTb':
            self.data = IvTOJT(self.data, self.restfreq, 
                self.beam[0]/3600, self.beam[1]/3600.)
        elif conversion == 'TbtoIv':
            self.data = TbTOIv(self.data, self.restfreq, 
                self.beam[0]/3600, self.beam[1]/3600.)
        elif conversion == 'pas2topbm':
            self.data = pas2TOpbm(self.data, self.delx, 
                self.dely, self.beam[0]/3600., self.beam[1]/3600.)
        elif conversion == 'IvtoTex':
            self.data = IvTOTex(self.data, self.restfreq, 
                self.beam[0]/3600, self.beam[1]/3600.)
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


    def getmoments(self, moments=[0], vrange=[], threshold=[],
        outfits=True, outname=None, overwrite=False, i_stokes=0):
        '''
        Calculate moment maps.

        moments (list): Index of moments that you want to calculate.
        vrange (list): Velocity range to calculate moment maps.
        threshold (list): Threshold to clip data. Shoul be given as [minimum intensity, maximum intensity]
        outfits (bool): Output as a fits file?
        outname (str): Output file name if outfits=True.
        overwrite (bool): If overwrite an existing fits file or not.
        '''

        # data check
        data  = self.data.copy()
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
        mom0 = np.sum(delv*data, axis=0)
        w2 = np.sum(delv * data * delv * data, axis=0) # Sum_i w_i^2
        ndata = np.count_nonzero(data, axis=0)

        if 0 in moments:
            out_moments = [mom0]
        else:
            out_moments = []

        # moment 1
        if any([i in [1,2] for i in moments]):
            vtile = np.tile(vaxis, (nx, ny, 1))
            vtile = np.transpose(vtile, (2, 1, 0))
            mom1 = np.sum(data * vtile * delv, axis=0)/mom0


            # calculate error
            vmean = np.tile(mom1, (nchan, 1, 1))
            vcount = np.where(data > 0., vtile, np.nan)
            sig_v2 = np.nansum(delv * (vmean - vcount)**2., axis=0)

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

            if any([i == 1 for i in moments]):
                out_moments.append(mom1)
                out_moments.append(sig_mom1)


        if any([i == 2 for i in moments]):
            vtile = np.tile(vaxis, (nx, ny, 1))
            vtile = np.transpose(vtile, (2, 1, 0))
            vmean = np.tile(mom1, (nchan, 1, 1))
            vcount = np.where(data > 0., vtile, np.nan)
            mom2 = np.sqrt(np.nansum(data*delv*(vcount - vmean)**2., axis=0)/mom0)

            sig_mom2_sq = 2./(np.count_nonzero(data, axis=0) - 1.) \
                * (mom0 * mom2 * mom2/(mom0 - (w2/mom0)))**2.
            sig_mom2_sq[sig_mom2_sq < 0.] = 0.
            sig_mom2 = np.sqrt(sig_mom2_sq)

            #moments_err.append(sig_mom2)
            out_moments.append(mom2)
            out_moments.append(sig_mom2)

        if 8 in moments:
            mom8 = np.nanmax(data, axis=0)
            out_moments.append(mom8)

        print ('Done.')


        # output
        #print (np.array(moments).shape)
        print ('Output: Moments '+' '.join([str(moments[i]) for i in range(len(moments))])
            +' and thier error maps except for moment 0.')

        if outfits:
            hdout = self.header
            naxis = self.naxis

            if naxis == 3:
                outmaps = np.array(out_moments)
            elif naxis == 4:
                outmaps = np.array([out_moments])
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

            hdout['HISTORY'] = 'Produced by getmoments of Imfits.'
            hdout['HISTORY'] = 'Moments: '+' '.join([str(moments[i]) for i in range(len(moments))])

            if outname:
                pass
            else:
                outname = self.file.replace('.fits','_moments.fits')

            #print (outmaps.shape)
            fits.writeto(outname, outmaps, header=hdout, overwrite=overwrite)

        return out_moments


    def get_momentzero_rms(self, rms_ch, vrange = None):
        '''
        Calculate theoretically expected rms of moment 0 map.
        '''
        if vrange is not None:
            index = np.where( (self.vaxis >= vrange[0]) & (self.vaxis <= vrange[1]))
            nv = len(self.vaxis[index])
        else:
            nv = self.nv

        return rms_ch * self.delv * np.sqrt(nv)



    def search_for_line_detection(self, 
        threshold, cond_area = 0.7, 
        saxis = 0, radius = None, output_mode = 'minmax'):
        if self.naxis <= 2:
            print('ERROR\tsearch_for_line_detection: No frequency axis is found.')
            print('ERROR\tsearch_for_line_detection: This function is for cube images.')
            return 0
        elif self.naxis == 3:
            data = self.data.copy()
        elif self.naxis == 4:
            data = self.data.copy()[saxis,:,:,:,]
        else:
            print('ERROR\tsearch_for_line_detection: naxis must be 3 or 4.')
            return 0

        x = self.xx * 3600.
        y = self.yy * 3600.

        if radius is not None:
            r = np.sqrt(x*x + y*y)
            where = np.where(r > radius)
            data[:, where[0], where[1]] = -100. # mask

        # beam size
        beam = 0.5 * (self.beam[0] + self.beam[1])

        # search for detection
        detection = []
        for d in data:
            cont_size = iu.get_contour_size(x, y, d, threshold)
            if np.isnan(cont_size) | (cont_size < beam * cond_area):
                detection.append(False)
            else:
                detection.append(True)

        indices = [i for i, x in enumerate(detection) if x]

        if output_mode == 'full':
            return indices
        elif output_mode == 'minmax':
            return min(indices), max(indices)
        else:
            print('ERROR\tsearch_for_line_detection: input output_mode is wrong.')
            print('ERROR\tsearch_for_line_detection: output_mode must be minmax or full.')
            return 0


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

        # if pvd
        if self.ifpv == True:
            self.data = np.squeeze(self.data)
            yimin, yimax = index_between(self.vaxis, ylim, mode='edge')[0]
            ximin, ximax = index_between(self.xaxis, xlim, mode='edge')[0]
            self.data    = self.data[yimin:yimax+1, ximin:ximax+1]
            #self.xx      = self.xx[yimin:yimax+1, ximin:ximax+1]
            #self.yy      = self.yy[yimin:yimax+1, ximin:ximax+1]
            self.vaxis   = self.vaxis[index_between(self.vaxis, ylim)]
            self.xaxis   = self.xaxis[index_between(self.xaxis, xlim)]
            self.nx = len(self.xaxis)
            self.nv = len(self.vaxis)
            return 1

        # cube
        # spatially trim
        xlim = np.array(xlim)/3600. # arcsec --> deg
        ylim = np.array(ylim)/3600. # arcsec --> deg
        yimin, yimax = index_between(self.yaxis, ylim, mode='edge')[0]
        ximin, ximax = index_between(self.xaxis, xlim, mode='edge')[0]
        self.xx      = self.xx[yimin:yimax+1, ximin:ximax+1]
        self.yy      = self.yy[yimin:yimax+1, ximin:ximax+1]
        self.xx_wcs  = self.xx_wcs[yimin:yimax+1, ximin:ximax+1]
        self.yy_wcs  = self.yy_wcs[yimin:yimax+1, ximin:ximax+1]
        self.xaxis   = self.xaxis[index_between(self.xaxis, xlim)]
        self.yaxis   = self.yaxis[index_between(self.yaxis, ylim)]
        self.nx = len(self.xaxis)
        self.ny = len(self.yaxis)
        if self.naxis == 2:
            self.data    = self.data[yimin:yimax+1, ximin:ximax+1]
            self.axes = np.array([self.xaxis, self.yaxis], dtype=object)
        elif self.naxis == 3:
            vimin, vimax = index_between(self.vaxis, vlim, mode='edge')[0]
            self.data    = self.data[vimin:vimax+1, yimin:yimax+1, ximin:ximax+1]
            self.vaxis   = self.vaxis[index_between(self.vaxis, vlim)]
            self.nv = len(self.vaxis)
            self.axes = np.array([self.xaxis, self.yaxis, self.vaxis], dtype=object)
        elif self.naxis == 4:
            simin, simax = index_between(self.saxis, slim, mode='edge')[0]
            vimin, vimax = index_between(self.vaxis, vlim, mode='edge')[0]
            self.data    = self.data[simin:simax+1, vimin:vimax+1, yimin:yimax+1, ximin:ximax+1]
            self.saxis   = self.saxis[index_between(self.saxis, slim)]
            self.vaxis   = self.vaxis[index_between(self.vaxis, vlim)]
            self.nv = len(self.vaxis)
            self.ns = len(self.saxis)
            self.axes = np.array([self.xaxis, self.yaxis, self.vaxis, self.saxis], dtype=object)
        else:
            print('trim_data: Invalid data shape.')
            return 0

        return 1


    def get_mapextent(self, unit='arcsec'):
        xaxis = self.xaxis
        delx = self.delx
        yaxis = self.vaxis if self.ifpv else self.yaxis
        dely = self.delv if self.ifpv else self.dely
        xmin, xmax = xaxis[[0,-1]]
        ymin, ymax = yaxis[[0,-1]]
        extent = (xmin-0.5 * delx, 
                xmax+0.5 * delx, 
                ymin-0.5 * dely, 
                ymax+0.5 * dely)
        if (self.ifpv == False) & (unit == 'arcsec'):
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


    def sampling(self, steps: list, 
        units = 'resolution', keep_center = True):
        d = self.data.copy()
        nd = len(d.shape)
        nsteps = len(steps)

        # input check
        if nsteps > nd:
            print('WARRING\tsampling: Dimension of steps is larger than \
                the demension of the data.')
            print('WARRING\tsampling: Only use first %i step values.'%nd)
            steps = steps[:nd]
        elif nsteps < 2:
            print('ERROR\tsampling: steps must have at least two elements.')
            return 0

        # dimension
        nsteps = len(steps) # renew
        if nsteps == 2:
            x_smpl, y_smpl = steps
            v_smpl = 1
            s_smpl = 1
        elif nsteps == 3:
            x_smpl, y_smpl, v_smpl = steps
            s_smpl = 1
        else:
            x_smpl, y_smpl, v_smpl, s_smpl = steps

        # if pvd
        if self.ifpv == True:
            if units == 'resolution':
                x_smpl = int(self.res_off / x_smpl / self.delx )
                y_smpl = int(1. / y_smpl)
            elif units == 'pixel':
                pass
            elif units == 'absolute':
                x_smpl = int(x_smpl / self.delx )
                y_smpl = int(y_smpl /  self.delv)
            else:
                print('ERROR\tsampling: Input units key word is wrong.')
                print('ERROR\tsampling: Must be resolution, pixel or absolute.')
                return 0
            self.data = np.squeeze(d)[y_smpl//2::y_smpl, x_smpl//2::x_smpl]
            self.vaxis = self.vaxis[y_smpl//2::y_smpl]
            self.xaxis = self.xaxis[x_smpl//2::x_smpl]
            self.delx = self.xaxis[1] - self.xaxis[0]
            self.delv = self.vaxis[1] - self.vaxis[0]
            self.nx = len(self.xaxis)
            self.nv = len(self.vaxis)
            return 1

        # cube
        if units == 'resolution':
            res_off = self.beam[1] / 3600. # in deg
            x_smpl = int(- res_off / x_smpl / self.delx ) if self.delx < 0. \
            else int(res_off / x_smpl / self.delx )
            y_smpl = int(res_off / y_smpl / self.dely )
            v_smpl = int(1. / v_smpl)
        elif units == 'pixel':
            pass
        elif units == 'absolute':
            x_smpl = int(- x_smpl / self.delx ) if self.delx < 0. \
            else int(x_smpl / self.delx )
            y_smpl = int(y_smpl /  self.delv)
            v_smpl = int(v_smpl / self.delv)
        else:
            print('ERROR\tsampling: Input units key word is wrong.')
            print('ERROR\tsampling: Must be resolution, pixel or absolute.')
            return 0

        # sampling
        if keep_center:
            x0, x1, y0, y1 = x_smpl//2, self.nx, y_smpl//2, self.ny
        else:
            x0, x1, y0, y1 = x_smpl//2 + 1, -1, y_smpl//2 + 1, -1
        self.xx      = self.xx[y0:y1:y_smpl, x0:x1:x_smpl]
        self.yy      = self.yy[y0:y1:y_smpl, x0:x1:x_smpl]
        self.xx_wcs  = self.xx_wcs[y0:y1:y_smpl, x0:x1:x_smpl]
        self.yy_wcs  = self.yy_wcs[y0:y1:y_smpl, x0:x1:x_smpl]
        self.yaxis   = self.yaxis[y0:y1:y_smpl]
        self.xaxis   = self.xaxis[x0:x1:x_smpl]
        self.nx = len(self.xaxis)
        self.ny = len(self.yaxis)
        self.delx = self.xaxis[1] - self.xaxis[0]
        self.dely = self.yaxis[1] - self.yaxis[0]

        if nd == 2:
            self.data = d[y0:y1:y_smpl, x0:x1:x_smpl]
            self.axes = np.array([self.xaxis, self.yaxis], dtype=object)
        elif nd == 3:
            self.data    = d[v_smpl//2::v_smpl, y0:y1:y_smpl, x0:x1:x_smpl]
            self.vaxis   = self.vaxis[v_smpl//2::v_smpl]
            self.nv = len(self.vaxis)
            self.delv = self.vaxis[1] - self.vaxis[0] if self.nv > 1 else 1.
            self.axes = np.array([self.xaxis, self.yaxis, self.vaxis], dtype=object)
        elif self.naxis == 4:
            self.data    = d[s_smpl//2:s_smpl, v_smpl//2::v_smpl, y0:y1:y_smpl, x0:x1:x_smpl]
            self.vaxis   = self.vaxis[v_smpl//2::v_smpl]
            self.nv = len(self.vaxis)
            self.delv = self.vaxis[1] - self.vaxis[0] if self.nv > 1 else 1.
            self.axes = np.array([self.xaxis, self.yaxis, self.vaxis, self.saxis], dtype=object)
        else:
            print('ERROR\tsampling: Invalid data shape.')
            return 0

        return 1


    def update_hdinfo(self, vunit='freq'):
        # dimension of data
        self.naxis = len(self.data.shape)

        # read data
        if self.ifpv:
            if vunit == 'velocity':
                self.label_i[1] = 'VRAD'
            else:
                self.label_i[1] = 'FREQ'
                if self.restfreq is None:
                    freq = [1] * self.nv
                else:
                    freq = ( 1. - self.vaxis * 1.e5 / clight ) * self.restfreq # km/s --> Hz
                delv = freq[1] - freq[0]
            refpix_i = [self.nx//2 + 1., 1.]
            refval_i = [self.xaxis[self.nx//2], freq[0]]
            naxis_i  = [self.nx, self.nv]
            del_i = [self.delx, delv]

            # stokes axis
            if self.naxis == 3:
                refpix_i.append(1.)
                refval_i.append(1.)
                naxis_i.append(1)
                del_i.append(1.)
        else:
            # spatial axis
            refpix_i = [self.nx//2 + 1., self.ny//2 + 1.]
            refval_i = [
            self.xx_wcs[self.ny//2, self.nx//2],
            self.yy_wcs[self.ny//2, self.nx//2]
            ]
            naxis_i  = [self.nx, self.ny]
            del_i = [self.delx, self.dely]

            # velocity
            if self.naxis >= 3:
                if vunit == 'velocity':
                    self.label_i[2] = 'VELO'
                    vaxis = self.vaxis * 1.e3 # in cm/s
                    refpix_i.append(1.)
                    refval_i.append(vaxis[0])
                    naxis_i.append(self.nv)
                    del_i.append(vaxis[1] - vaxis[0])
                else:
                    self.label_i[2] = 'FREQ'
                    freq = ( 1. - self.vaxis * 1.e5 / clight ) * self.restfreq # km/s --> Hz
                    delv = freq[1] - freq[0] if self.naxis_i[2] > 1 else 1.
                    refpix_i.append(1.)
                    refval_i.append(freq[0])
                    naxis_i.append(len(freq))
                    del_i.append(delv)

            # stokes
            if self.naxis == 4:
                refpix_i.append(1.)
                refval_i.append(1.)
                naxis_i.append(1)
                del_i.append(1.)

        # update
        self.naxis_i  = naxis_i
        self.refpix_i = refpix_i
        self.refval_i = refval_i
        self.del_i = del_i
        return 1


    def writeout(self, outname, 
        comment = None, hdkeys = None, overwrite = False,
        vunit = 'freq'):
        print('This function is under development. Might return a wrong result.')
        print('No multibeam output is currently supported.')
        print('Please use it with causion')

        # import
        from datetime import datetime

        # header
        hdout = self.header.copy()
        self.update_hdinfo(vunit)

        # write header
        hdout['NAXIS'] = self.naxis
        for i in range(self.naxis):
            hdout['NAXIS%i'%(i+1)] = self.naxis_i[i]
            hdout['CRVAL%i'%(i+1)] = self.refval_i[i]
            hdout['CRPIX%i'%(i+1)] = self.refpix_i[i]
            hdout['CDELT%i'%(i+1)] = self.del_i[i]

        if comment is not None: hdout['COMMENT'] = comment

        if hdkeys is not None:
            for i in hdkeys.keys():
                hdout[i] = hdkeys[i]

        today = datetime.today().strftime('%Y-%m-%d')
        hdout['HISTORY'] = today + ': written by imfits.'

        if self.multibeam:
            if self.beam_savetype in ['average', 'single']:
                hdout['BMAJ'] = self.beam[0] / 3600. # arcsec -> deg
                hdout['BMIN'] = self.beam[1] / 3600. # arcsec -> deg
                hdout['BPA'] = self.beam[2]
                del hdout["CASAMBM"]
            else:
                print('WARNING\twriteout: Found multibeam.')
                print('WARNING\twriteout: No beam info is included in the output fits file.')

        fits.writeto(outname, self.data, header = hdout, overwrite = overwrite)

        return 1
