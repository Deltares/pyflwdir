.. image:: https://gitlab.com/deltares/wflow/pyflwdir/badges/master/coverage.svg
   :target: https://gitlab.com/deltares/wflow/pyflwdir/commits/master

################################################################################
PyFlwDir
################################################################################

Fast methods to work with hydro- and topography data in pure Python. 

Methods include:

- flow direction upscaling
- (sub)basin delineation
- pfafstetter subbasins delineation
- upstream accumulation
- up- and downstream tracing and arithmetics
- strahler stream order
- hydrologically adjusting elevation
- height above nearest drainage
- vectorizing stream features
- *many more!*


Example
=======

Here we show an example of a 30 arcsec D8 map of the Rhine basin which is saved in 
as 8-bit GeoTiff. We read the flow direction, including meta-data using `rasterio <https://rasterio.readthedocs.io/en/latest/>`_ 
to begin with.

.. code-block:: python

    import rasterio
    with rasterio.open('data/rhine_d8.tif', 'r') as src:
        flwdir = src.read(1)
        transform = src.transform
        latlon = src.crs.to_epsg() == 4326

Next, we parse this data to a **FlwdirRaster** object, the core object 
to work with flow direction data. The most common way to initialize a `FlwdirRaster` object 
is based on gridded flow direction data in D8, LDD or NEXTXY format using 
the **pyflwdir.from_array** method. Optional arguments describe the geospatial
location of the gridded data. In this step the D8 data is parsed to an actionable format.

.. code-block:: python

    import pyflwdir
    flw = pyflwdir.from_array(flwdir, ftype='d8', transform=transform, latlon=latlon)
    # When printing the FlwdirRaster instance we see its attributes. 
    print(flw)

Now all pyflwdir FlwdirRaster methods are available, for instance the subbasins method
which creates a map with unique IDs for subbasin with a minumum stream_order. 

Browse the `docs API <https://deltares.gitlab.io/wflow/pyflwdir/reference.html>`_ for all methods

.. code-block:: python

    subbasins = flw.subbasins()

Getting started
===============

Install the package from pip using

.. code-block:: console

    $ pip install pyflwdir


Development and Testing
=======================

See `CONTRIBUTING.rst <CONTRIBUTING.rst/>`__

Documentation
=============

See `docs <https://deltares.gitlab.io/wflow/pyflwdir/>`__

License
=======

See `LICENSE <LICENSE>`__

Authors
=======

See `AUTHORS.txt <AUTHORS.txt>`__

Changes
=======

See `CHANGESLOG.rst <CHANGELOG.rst>`__
