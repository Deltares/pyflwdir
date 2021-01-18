===============================================================================
pyflwdir: Fast methods to work with hydro- and topography data in pure Python.
===============================================================================

Intro
-----

This package contains a series of methods to work with gridded DEM and flow direction 
datasets, which are key to many workflows in many earth siences. Compared to other
flow direction packages pyflwdir supports several flow direction data conventions and 
can easily be extended to include more. The package contains some unique methods such as 
Iterative Hydrography Upscaling (IHU) method to upscale flow directions from 
high resolution data to coarser model resolution. 
Pyflwdir is in pure python and powered by numba to keep it fast.

Featured methods:

- Flow direction upscaling
- (sub)basin delineation
- pfafstetter coded subbasins delineation
- up- and downstream tracing and arithmetics
- height above nearest drainage (HAND) and floodplain delineation
- upstream accumulation
- strahler stream order
- hydrologically adjusting elevation
- vectorizing stream features
- many more!


.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  Installation <installation>

.. toctree::
  :maxdepth: 1
  :caption: User Guide
  
  Flow direction data <flwdir>

.. toctree::
  :maxdepth: 1
  :caption: Reference Guide

  FlwdirRaster reference <reference>
  Changelog <changelog>


.. toctree::
  :maxdepth: 1
  :caption: Developer

  Contributing to PyFlwDir <contributing>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
