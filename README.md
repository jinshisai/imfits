imfits
-----------------------
imfits is a python module to read, handle, analyze and draw maps from fits files for astronomy easily. Imfits, the python class contains header and image information in variables, making it easy to call them. This is made for fits images especially at (sub)millimeter wavelengths obtained with telescopes like ALMA, ACA, SMA, JCMT, IRAM-30m and so on. Not guaranteed but could be applied to other fits data like at IR wavelengths.

Imfits can also read fits files of position-velocity (PV) diagrams. The script is well tested for fits files exported from [CASA](https://casa.nrao.edu).

**Requirements**

- numpy
- astropy
- matplotlib


**Contanct**  
E-mail: jn.insa.sai@gmail.com  
Jinshi Sai (Insa Choi)  
Department of Astronomy, the University of Tokyo


Install Imfits
----
You can get it easily using pip install.

```bash
pip install git+https://github.com/jinshisai/Imfits
```

Or at a directory you want to put this,

```bash
git clone https://github.com/jinshisai/Imfits
```

In the case of using git clone, to update it,
```bash
git pull
````


Use Imfits
---------------

Imfits read a fits file.

```python
from imfits import Imfits

infile   = 'fitsname.fits' # fits file
fitsdata = Imfits(infile)  # Read information
```

Then, you can call the data and the header information easily.

```python
data  = fitsdata.data  # Call data array
xaxis = fitsdata.xaxis # Call x axis
nx    = fitsdata.nx    # Size of xaxis
```

You can also draw maps calling drawmaps. Moment maps, channel maps and PV diagrams are now supported.

```python
from imfits import drawmaps

# Moment map
drawmaps.intensitymap(fitsdata, outname='test', outformat='pdf')
# That's it! test.pdf is the drawn map.
# You can put more options to draw beautiful maps.

# Channel maps
drawmaps.channelmaps(fitsdata, outname='test', outformat='pdf')
```

Add an option ```pv=True``` to read a fits file of a position-velocity (PV) diagram.

```python
from imfits import Imfits

infile   = 'pvfits.fits'           # fits file
fitsdata = Imfits(infile, pv=True) # Read information
```