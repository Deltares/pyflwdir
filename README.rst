################################################################################
PyFlwDir
################################################################################

Intro
-----

This package contains a series of methods to work with gridded DEM and flow direction 
datasets, which are key to many workflows in many earth siences. Compared to other
flow direction packages pyflwdir supports several flow direction data conventions and 
can easily be extended to include more. The package contains some unique methods such as 
Iterative Hydrography Upscaling (IHU) method to upscale flow directions from 
high resolution data to coarser model resolution. 

Pyflwdir is in pure python and powered by numba to keep it fast.


Featured methods
----------------

.. image:: docs/_static/pyflwdir.png
  :width: 100%

- flow directions from elevation data using a steepest gradient algorithm
- strahler stream order
- flow direction upscaling
- (sub)basin delineation
- pfafstetter subbasins delineation
- classic stream order
- height above nearest drainage (HAND) 
- geomorphic floodplain delineation
- up- and downstream tracing and arithmetics
- hydrologically adjusting elevation
- upstream accumulation
- vectorizing streams
- many more!


Getting started
===============

Install the package from pip using

.. code-block:: console

    $ pip install pyflwdir

Install the package from conda using

.. code-block:: console

    $ conda install pyflwdir -c conda-forge


Development and Testing
=======================

See `CONTRIBUTING.rst <CONTRIBUTING.rst/>`__

Documentation
=============

See `docs <https://deltares.github.io/pyflwdir/latest/>`__

License
=======

See `LICENSE <LICENSE>`__

Authors
=======

See `AUTHORS.txt <AUTHORS.txt>`__

Changes
=======

See `CHANGESLOG.rst <CHANGELOG.rst>`__
