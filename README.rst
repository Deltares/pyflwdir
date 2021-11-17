
.. image:: https://codecov.io/gh/Deltares/pyflwdir/branch/main/graph/badge.svg?token=N4VMHJJAV3
    :target: https://codecov.io/gh/Deltares/pyflwdir

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: https://deltares.github.io/pyflwdir/latest
    :alt: Latest developers docs

.. image:: https://badge.fury.io/py/pyflwdir.svg
    :target: https://pypi.org/project/pyflwdir/
    :alt: Latest PyPI version

.. image:: https://anaconda.org/conda-forge/pyflwdir/badges/version.svg
    :target: https://anaconda.org/conda-forge/pyflwdir

################################################################################
PyFlwDir
################################################################################

Intro
-----

PyFlwDir contains a series of methods to work with gridded DEM and flow direction 
datasets, which are key to many workflows in many earth siences. Pyflwdir supports several 
flow direction data conventions and can easily be extended to include more. 
The package contains some unique methods such as Iterative Hydrography Upscaling (IHU) 
method to upscale flow directions from high resolution data to coarser model resolution. 

Pyflwdir is in pure python and powered by numba to keep it fast.


Featured methods
----------------

.. image:: https://raw.githubusercontent.com/Deltares/pyflwdir/master/docs/_static/pyflwdir.png
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


Installation
============

We recommend installing PyFlwdir using conda or pip. 

Install the package from conda using:

.. code-block:: console

    $ conda install pyflwdir -c conda-forge


Install the package from pip using:

.. code-block:: console

    $ pip install pyflwdir

In order to run the examples in the notebook folder some aditional packages to read 
and write raster and vector data, as well as to plot these data are required. 
A complete environment can be installed from the environment.yml file using:

.. code-block:: console

    $ conda env create -f environment.yml
    $ pip install pyflwdir

Quickstart
==========

The most common workflow to derive flow direction from digital elevation data and 
subsequent delineate basins or vectorize a stream network can be done in just a few
lines of code. 

To read elevation data from a geotiff raster file *elevation.tif* do:

.. code-block:: python

    import rasterio
    with rasterio.open("elevation.tif", "r") as src:
        elevtn = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        

Derive a FlwdirRaster object from this data:

.. code-block:: python

    import pyflwdir
    flw = pyflwdir.from_dem(
        data=elevtn,
        nodata=src.nodata,
        transform=transform,
        latlon=crs.is_geographic,
    )

Delineate basins and retrieve a raster with unique IDs per basin:
Tip: This raster can directly be written to geotiff and/or vectorized to save as 
vector file with `rasterio <https://rasterio.readthedocs.io/>`_

.. code-block:: python

    basins = flw.basins()

Vectorize the stream network and save to a geojson file:

.. code-block:: python

    import geopandas as gpd
    feat = flw.streams()
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf.to_file('streams.geojson', driver='GeoJSON')


Documentation
=============

See `docs <https://deltares.github.io/pyflwdir/latest/>`__ for a many examples and a 
full reference API.


Development and Testing
=======================

Welcome to the pyflwdir project. All contributions, bug reports, bug fixes, 
documentation improvements, enhancements, and ideas are welcome. 
See `CONTRIBUTING.rst <CONTRIBUTING.rst/>`__ for how we work.

Changes
=======

See `CHANGELOG.rst <CHANGELOG.rst>`__

Authors
=======

See `AUTHORS.txt <AUTHORS.txt>`__

License
=======

This is free software: you can redistribute it and/or modify it under the terms of the
MIT License. A copy of this license is provided in `LICENSE <LICENSE>`__