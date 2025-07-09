### modules
import numpy as np
from astropy import constants, units

# constants (in cgs)

Ggrav  = constants.G.cgs.value        # Gravitational constant
ms     = constants.M_sun.cgs.value    # Solar mass (g)
ls     = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
rs     = constants.R_sun.cgs.value    # Solar radius (cm)
au     = units.au.to('cm')            # 1 au (cm)
pc     = units.pc.to('cm')            # 1 pc (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
hp     = constants.h.cgs.value        # Planck constant
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mp     = constants.m_p.cgs.value      # Proton mass (g)



# Jy/beam --> K (when optically thick)
def IvTOTex(Iv, nu0, bmaj, bmin, sigI=None):
    '''
    Convert Iv to Tb WITHOUT Rayleigh-Jeans approximation.

    Parameters
    ----------
        nu0 (float): Rest frequency [Hz]
        bmaj (float): Major beam size [deg]
        bmin (float): Minor beam size [deg]
        Iv (array or float): Intensity [Jy/beam]
        sigI (array or float): Sigma of intensity [Jy/beam]

    Others
    ------
        C1: coefficient to convert Iv to Tex
        C2: coefficient to convert beam to str
    '''

    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    C1=2.*hp*(nu0*nu0*nu0)/(clight*clight) # in cgs

    # Jy/beam -> Jy/str
    # Omg_beam (str) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/str]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str
    Istr = Iv/bTOstr # Jy/beam --> Jy/str
    Istr = Istr*1.0e-26 # Jy --> MKS (Jy = 10^-26 Wm-2Hz-1)
    Istr = Istr*1.e7*1.e-4 # MKS --> cgs

    Tex = (hp*nu0/kb)/(np.log(C1/Istr + 1.)) # no approximation [K]

    if sigI is not None:
        # Error propagation
        dT_dI = (hp*nu0/kb) * C1 * (Istr**(-2.)) * (C1/Istr + 1.)**(-1.) \
        * (np.log(C1/Istr + 1.))**(-2) # derivative
        sigI_str = sigI / bTOstr * 1.0e-26 * 1.e7 * 1.e-4
        sigTex = np.sqrt(dT_dI*dT_dI*sigI_str*sigI_str)

        return Tex, sigTex
    else:
        return Tex


# Calculate equivalent brightness temperature
def IvTOJT(Iv, nu0, bmaj, bmin):
    '''
    Calculate the equivalent brightness temperature from intensity in a unit of Jy/beam.

    Input
    Iv: intensity [Jy/beam]
     nu0: rest frequency [Hz]
     bmaj: major beam size [deg]
     bmin: minor beam size [deg]

    Others
     C2: coefficient to convert beam to str
     JT: equivalent brightness temperature [K]
    '''
    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    # Jy/beam -> Jy/sr
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str
    Istr = Iv/bTOstr # Jy/beam --> Jy/str
    Istr = Istr*1.0e-26 # Jy --> MKS (Jy = 10^-26 Wm-2Hz-1)
    Istr = Istr*1.e7*1.e-4 # MKS --> cgs

    JT = (clight*clight/(2.*nu0*nu0*kb))*Istr # equivalent brightness temperature
    return JT


# convert Icgs --> Iv Jy/pixel
def IcgsTOjpp(Icgs, px, py ,dist):
    '''
    Convert Intensity in cgs to in Jy/beam

    Icgs: intensity in cgs unit [erg s-1 cm-2 Hz-1 str-1]
    psize: pixel size (au)
    dist: distance to the object (pc)
    '''

    # cgs --> Jy/str
    Imks = Icgs*1.e-7*1.e4   # cgs --> MKS
    Istr = Imks*1.0e26       # MKS --> Jy/str, 1 Jy = 10^-26 Wm-2Hz-1

    # Jy/sr -> Jy/pixel
    px = np.radians(px/dist/3600.) # au --> radian
    py = np.radians(py/dist/3600.) # au --> radian
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area  = px*py
    Ijpp            = Istr*one_pixel_area # Iv (Jy per pixel)
    return Ijpp


def IcgsTObeam(Icgs,bmaj,bmin):
    '''
    Convert Intensity in cgs to in Jy/beam

    Icgs: intensity in cgs unit [erg s-1 cm-2 Hz-1 str-1]
    bmaj, bmin: FWHMs of beam major and minor axes [deg]
    Ibeam: intensity in Jy/beam
    '''
    # beam unit
    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    # cgs --> Jy/str
    Imks = Icgs*1.e-7*1.e4   # cgs --> MKS
    Istr = Imks*1.0e26       # MKS --> Jy/str, 1 Jy = 10^-26 Wm-2Hz-1

    # Jy/sr -> Jy/beam(arcsec)
    # beam          = thmaj*thmin (arcsec^2) = (a_to_rad)^2*thmaj*thmin (rad^2)
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2) = (pi/4ln2)*(a_to_rad)^2*thmaj*thmin
    # I [Jy/beam]   = I [Jy/sr] * Omg_beam
    C2              = np.pi/(4.*np.log(2.))
    #radTOarcsec     = (60.0*60.0*180.0)/np.pi
    beam_th         = 1. / (C2*bmaj*bmin) # beam(sr) -> beam(arcsec), 1/beam_sr
    Ibeam           = Istr/beam_th
    return Ibeam


# convert Jy/arcsec^2 --> Jy/beam
def pas2TOpbm(Iin, px, py, bmaj, bmin):
    '''
    Convert Intensity in Jy/arcsec^2 to in Jy/beam

    Parameters
    ----------
        Iin (float, array): Intensity (Jy/arcsec^2)
        px, py (float): Pixel size (deg)
        bmaj, bmin (float): Beam size (deg)
    '''

    # Jy/arcsec^2 -> Jy/str
    #  one_pixel_area = pixel*pixel (rad^2)
    #  More exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel when pixel area is small.
    #  (When psize ~ 0.14, S_apprx/S_acc = 1.00000000000004)
    as2tostr = np.radians(1./3600.)*np.radians(1./3600.) # 1 arcsec^2 in str
    Ijypstr  = Iin/as2tostr # Iv (Jy per str)

    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    C2     = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2           # beam --> str

    # Jy/str --> Jy/beam
    Ijypbm = Ijypstr*bTOstr
    return Ijypbm


# convert Iv Jy/beam  --> Iv Jy/pixel
def IbeamTOjpp(Ibeam, bmaj, bmin, px, py , au=False, dist=140.):
    '''
    Convert Intensity in cgs to in Jy/beam

    Ibeam: intensity in Jy/beam
    bmaj, bmin: beam size (degree)
    psize: pixel size (default in degree). If au=True, they will be treated in units of au.
    dist: distance to the object (pc)
    '''

    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    C2     = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str

    # Jy/beam --> Jy/str
    Istr = Ibeam/bTOstr # Jy/beam --> Jy/str


    # Jy/sr -> Jy/pixel
    if au:
        px = np.radians(px/dist/3600.) # au --> radian
        py = np.radians(py/dist/3600.) # au --> radian
    else:
        px = np.radians(px) # deg --> rad
        py = np.radians(py) # deg --> rad
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area  = np.abs(px*py)
    Ijpp            = Istr*one_pixel_area # Iv (Jy per pixel)
    return Ijpp


# Convert Tb to Iv
def TbTOIv(Tb, nu0, bmaj, bmin):
    '''
    Convert Tb to Iv

    Parameters
    ----------
     Tb: Brightness temperature [K]
     nu0: Rest frequency [Hz]
     bmaj: Major beam size [deg]
     bmin: Minor beam size [deg]

    Return
    ------
     Iv: Intensity [Jy/beam]

    Others
    ------
     C2: coefficient to convert beam to str
    '''

    # a conversion factor
    bmaj   = bmaj*np.pi/180.       # deg --> radian
    bmin   = bmin*np.pi/180.       # deg --> radian
    C2     = np.pi/(4.*np.log(2.)) # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2          # beam --> str

    # K --> Iv(in cgs, /str)
    Istr = ((2.*nu0*nu0*kb)/(clight*clight))*Tb

    # Jy/str --> Jy/beam
    Istr = Istr*1.e-7*1.e4 # cgs --> MKS
    Istr = Istr*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Iv   = Istr*bTOstr     # Jy/str --> Jy/beam

    return Iv


def TTOIv(T, nu0, bmaj, bmin):
    '''
    Convert T to Iv with the full Plank function.

    Parameters
    ----------
     T (float): Temperature [K]
     nu0 (float): Rest frequency [Hz]
     bmaj (float): Major beam size [deg]
     bmin (float): Minor beam size [deg]

    Return
    ------
     Iv: Intensity [Jy/beam]
    '''

    return Bv_Jybeam(T, nu0, bmaj, bmin)



# partition function
def Pfunc(EJ, gJ, J, Tk):
    # EJ: energy at energy level J
    # gJ: statistical weight
    # J: energy level
    # Tk: kinetic energy
    Z = 0.0
    for j in J:
        Z = Z + gJ[j]*np.exp(-EJ[j]/Tk)
        #Z = Z + (2.*j+1.)*np.exp(-EJ[j]/Tk)
    return Z


# Planck function
def Bv(T,v):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    v = v * 1.e9 # GHz --> Hz
    exp=np.exp((hp*v)/(kb*T))-1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv=fterm/exp
    return Bv


# Jy/beam
def Bv_Jybeam(T,v,bmaj,bmin):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [Hz]
    bmaj, bmin: beamsize [deg]
    '''

    # units
    bmaj = np.radians(bmaj) # deg --> radian
    bmin = np.radians(bmin) # deg --> radian

    # coefficient for unit convertion
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str


    exp=np.exp((hp*v)/(kb*T))-1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv=fterm/exp

    # cgs --> Jy/beam
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Bv = Bv*bTOstr     # Jy/str --> Jy/beam
    return Bv



# Rayleigh-Jeans approx.
def BvRJ(T,v):
    # T: temprature [K]
    # v: frequency [GHz]
    v = v * 1.e9 # GHz --> Hz
    Bv = 2.*v*v*kb*T/(clight*clight)
    return Bv