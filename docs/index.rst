.. imfits documentation master file, created by
   sphinx-quickstart on Thu Jul  6 16:01:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

imfits
==================================

imfits is a python module to read, handle, analyze and draw maps from fits files for astronomy easily. The python class Imfits contains header and image information in variables, making it easy to call. The module has been developed for fits images/cubes at (sub)millimeter wavelengths (and particularly for ALMA data). Not guaranteed but could be applied to other fits data at other wavelengths (optical, infrared, and so on).

Imfits can also read fits files of position-velocity (PV) diagrams. The script is well tested for fits files exported from `CASA <https://casa.nrao.edu>`_.



Installation
=============

The pip install is now available! For an easy installation, just type

::
   
   pip install imfits

To update it, type

::
   
   pip install -U imfits


You can also get it using 'git clone'.

::

   git clone https://github.com/jinshisai/imfits


Run 'git pull' in the imfits directory to make it up-to-date. Adding path in .bashrc (or .zshrc) is useful to call the module.



Tutorials
==================

.. toctree::
   :maxdepth: 2

   tutorials/demo_basic
   tutorials/demo_channelmaps
   tutorials/demo_pvdiagram


Known issues
====================
.. toctree::
   :maxdepth: 2

   notes/known_issues
