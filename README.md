imfits
-----------------------
imfits is a python module to read, handle and analyze fits files for astronomy easily. The main function, Imfits, is a python class to contain header and image information in variables, making it easy to call them. This is made for fits images especially at (sub)millimeter wavelengths obtained with like ALMA, ACA, SMA, JCMT, IRAM-30m and so on. Not guaranteed but could be applied to other fits data like at IR wavelengths.

Imfits can read fits files of position-velocity (PV) diagrams. The script is well tested for fits files exported from [CASA](https://casa.nrao.edu).


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

You can also draw maps.

```python
# Moment maps
fitsdata.draw_Idistmap(outname='map', outformat='pdf')
# Just an example.
# Put more options for the function to work correctly.
```

Add ```pv=True``` as an option to read a fits file of a position-velocity (PV) diagram.

```python
from imfits import Imfits

infile   = 'pvfits.fits'           # fits file
fitsdata = Imfits(infile, pv=True) # Read information
```