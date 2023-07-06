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
Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),  
Taipei, Taiwan  
Website: [jinshisai.github.io](https://jinshisai.github.io)


Install Imfits
----
You can get it using `git clone`.

```bash
git clone https://github.com/jinshisai/Imfits
```
run `git pull` in the Imfits directory to make it up-to-date. Adding path in .bashrc (or .zshrc) is useful to call the module.


`pip` install is also available but `git clone` & `git pull` is recommended to catch up all updates. If you wish to use `pip`, type


```bash
pip install git+https://github.com/jinshisai/imfits
```



Use Imfits
---------------

Imfits read a fits file.

```python
from imfits import Imfits

infile   = 'fitsname.fits' # fits file
fitsdata = Imfits(infile)  # Read information
```

Then, you can call the data and header information easily.

```python
data  = fitsdata.data  # Call data array
xaxis = fitsdata.xaxis # Call x axis
nx    = fitsdata.nx    # Size of xaxis
```

You can also draw maps calling drawmaps. Moment maps, channel maps and position-velocity (PV) diagrams are now supported.

```python
from imfits import drawmaps as dm

# Moment map
drawmaps.intensitymap(fitsdata, outname='test', outformat='pdf')
# That's it! test.pdf is the drawn map.
# You can put more options to draw beautiful maps.

# Channel maps
drawmaps.channelmaps(fitsdata, outname='test', outformat='pdf')
```

Add an option `pv=True` to read a fits file of a PV diagram.

```python
from imfits import Imfits

infile   = 'pvdiagram.fits'           # fits file
fitsdata = Imfits(infile, pv=True) # Read information
```