.. image:: https://gitlab.com/deltares/wflow/pyflwdir/badges/master/coverage.svg
   :target: https://gitlab.com/deltares/wflow/pyflwdir/commits/master

################################################################################
PyFlwDir
################################################################################

Fast methods to work with hydro- and topography data in pure Python.


Example
=======

Here we show an example of a 30 arcsec D8 map of the Rhine basin which is saved in 
as 8-bit GeoTiff. We read the flow direction and eleveation data, including meta-data 
using rasterio to begin with.

.. ipython:: python

    # import rasterio and read D8 flow direction data
    # NOTE rasterio is not part of the package dependencies
    import rasterio
    with rasterio.open('data/rhine_d8.tif', 'r') as src:
        flwdir = src.read(1)
        transform = src.transform
        latlon = src.crs.to_epsg() == 4326

Next, we parse this data to a :py:class:`~pyflwdir.FlwdirRaster` object, the core object 
to work with flow direction data. The most common way to initialize a `FlwdirRaster` object 
is based on gridded flow direction data in D8, LDD or NEXTXY format using 
the :py:func:`pyflwdir.from_array` method. Optional arguments describe the geospatial
location of the gridded data. In this step the D8 data is parsed to an actionable format.

.. ipython:: python

    import pyflwdir
    flw = pyflwdir.from_array(flwdir, ftype='d8', transform=transform, latlon=latlon)
    # When printing the FlwdirRaster instance we see its attributes. 
    print(flw)

Now all pyflwdir methods are available, for instance the stream_order method which
returns a map with the  Strahler order of each cell. 
Browse the `docs API <https://deltares.gitlab.io/wflow/pyflwdir/reference.html>`_ for all methods

.. ipython:: python

    stream_order = flw.stream_order()


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
