imfits
-----------------------

[![Documentation Status](https://readthedocs.org/projects/imfits/badge/?version=latest)](https://imfits.readthedocs.io/en/latest/?badge=latest)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18918662.svg)](https://doi.org/10.5281/zenodo.18918662)

`imfits` is a python module to read, handle, analyze and draw maps from fits files for astronomy easily. The python class `Imfits` contains header and image information in variables, making it easy to call them. This has been developed for fits images/cubes at (sub)millimeter wavelengths (and particularly for ALMA data). Not guaranteed but could be applied to other fits data at other wavelengths (optical, infrared, and so on).

`Imfits` can also read fits files of position-velocity (PV) diagrams. The script is well tested for fits files exported from [CASA](https://casa.nrao.edu).

**Requirements**

- numpy
- scipy
- astropy
- matplotlib


**Contanct**  
E-mail: jn.insa.sai@gmail.com  
Jinshi Sai  
Kagoshima Univ.,  
Kagoshima, Japan  
Website: [jinshisai.github.io](https://jinshisai.github.io)


Install imfits
----

`pip` install is available now!

```bash
pip install imfits
```

To update it,

```bash
pip install -U imfits
```

You can also get it using `git clone`.

```bash
git clone https://github.com/jinshisai/imfits
```

Run `git pull` in the imfits directory to make it up-to-date. Adding path in .bashrc (or .zshrc) is useful to call the module in this case.



Use imfits
---------------

The python class, `Imfits`, makes it easy to read a fits file.

```python
from imfits import Imfits

f   = 'fitsname.fits' # fits file
im = Imfits(f)  # Read information
```

You can call the data and header information easily.

```python
data  = im.data  # Call data array
xaxis = im.xaxis # Call x axis
nx    = im.nx    # Size of xaxis
```

You can also draw maps calling `AstroCanvas` from `imfits.drawmaps`.

```python
from imfits.drawmaps import AstroCanvas

canvas = AstroCanvas((1,1))
canvas.intensitymap(im)
canvas.savefig('outputname')
plt.show()
```

Continuum maps, moment maps, channel maps and position-velocity (PV) diagrams are now supported. See [![Documentation Status](https://readthedocs.org/projects/imfits/badge/?version=latest)](https://imfits.readthedocs.io/en/latest/?badge=latest) for more details.



Citation
--------

If you use `imfits` for your publications, please cite it via [zenodo](https://zenodo.org/records/18918665) in the software or acknowledgment section:

```
@software{imfits:2026,
  author       = {Jinshi Sai},
  title        = {imfits},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v2.3.5},
  doi          = {10.5281/zenodo.18918665},
  url          = {https://doi.org/10.5281/zenodo.18918665},
}
```