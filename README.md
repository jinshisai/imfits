imfits
-----------------------

[![Documentation Status](https://readthedocs.org/projects/imfits/badge/?version=latest)](https://imfits.readthedocs.io/en/latest/?badge=latest)

imfits is a python module to read, handle, analyze and draw maps from fits files for astronomy easily. The python class `Imfits` contains header and image information in variables, making it easy to call them. This has been developed for fits images/cubes at (sub)millimeter wavelengths (and particularly for ALMA data). Not guaranteed but could be applied to other fits data at other wavelengths (optical, infrared, and so on).

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


Install imfits
----
You can get it using `git clone`.

```bash
git clone https://github.com/jinshisai/imfits
```
run `git pull` in the Imfits directory to make it up-to-date. Adding path in .bashrc (or .zshrc) is useful to call the module.


`pip` install is also available but `git clone` & `git pull` is recommended to catch up all updates. If you wish to use `pip`, type


```bash
pip install git+https://github.com/jinshisai/imfits
```



Use imfits
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

You can also draw maps calling `AstroCanvas` from `imfits.drawmaps`.

```python
from imfits.drawmaps import AstroCanvas

canvas = dm.AstroCanvas((1,1))
canvas.intensitymap(im)
canvas.savefig('outputname')
plt.show()
```

Continuum maps, moment maps, channel maps and position-velocity (PV) diagrams are now supported. See [![Documentation Status](https://readthedocs.org/projects/imfits/badge/?version=latest)](https://imfits.readthedocs.io/en/latest/?badge=latest) for more details.