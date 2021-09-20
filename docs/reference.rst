.. currentmodule:: pyflwdir

Reference
=========

FlwdirRaster
------------

.. autoclass:: pyflwdir.FlwdirRaster


Input/Output
^^^^^^^^^^^^

.. automethod:: pyflwdir.from_array()

.. automethod:: pyflwdir.from_dem()

.. automethod:: pyflwdir.FlwdirRaster.to_array()

.. automethod:: pyflwdir.FlwdirRaster.load()

.. automethod:: pyflwdir.FlwdirRaster.dump()



Flow direction attributes
^^^^^^^^^^^^^^^^^^^^^^^^^

The following attributes describe the flow direction and are at the core to the object.

.. autoattribute:: pyflwdir.FlwdirRaster.idxs_ds

.. autoattribute:: pyflwdir.FlwdirRaster.idxs_us_main

.. autoattribute:: pyflwdir.FlwdirRaster.idxs_seq

.. autoattribute:: pyflwdir.FlwdirRaster.idxs_pit

.. autoattribute:: pyflwdir.FlwdirRaster.ncells

.. autoattribute:: pyflwdir.FlwdirRaster.rank

.. autoattribute:: pyflwdir.FlwdirRaster.isvalid

.. autoattribute:: pyflwdir.FlwdirRaster.mask


Flow direction methods
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyflwdir.FlwdirRaster.order_cells()

.. automethod:: pyflwdir.FlwdirRaster.main_upstream()

.. automethod:: pyflwdir.FlwdirRaster.add_pits()

.. automethod:: pyflwdir.FlwdirRaster.repair_loops()

.. automethod:: pyflwdir.FlwdirRaster.vectorize()


Raster & geospatial attributes and methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FlwdirRaster object contains :attr:`FlwdirRaster.shape`, :attr:`FlwdirRaster.transform`
and :attr:`FlwdirRaster.latlon` attributes describing its geospatial location. The first 
attribute is required at initializiation, while the others can be set later.

.. automethod:: pyflwdir.FlwdirRaster.set_transform()

.. automethod:: pyflwdir.FlwdirRaster.index()

.. automethod:: pyflwdir.FlwdirRaster.xy()

.. autoproperty:: pyflwdir.FlwdirRaster.bounds()

.. autoproperty:: pyflwdir.FlwdirRaster.extent()


Streams and flow paths
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyflwdir.FlwdirRaster.stream_order()

.. automethod:: pyflwdir.FlwdirRaster.path()

.. automethod:: pyflwdir.FlwdirRaster.snap()

.. automethod:: pyflwdir.FlwdirRaster.stream_distance()

.. automethod:: pyflwdir.FlwdirRaster.streams()

.. automethod:: pyflwdir.FlwdirRaster.geofeatures()


(Sub)basins
^^^^^^^^^^^

.. automethod:: pyflwdir.FlwdirRaster.basins()

.. automethod:: pyflwdir.FlwdirRaster.basin_bounds()

.. automethod:: pyflwdir.FlwdirRaster.subbasins_streamorder()

.. automethod:: pyflwdir.FlwdirRaster.subbasins_pfafstetter()


Up- and downstream values
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyflwdir.FlwdirRaster.downstream()

.. automethod:: pyflwdir.FlwdirRaster.upstream_sum()


Up- and downstream arithmetics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyflwdir.FlwdirRaster.moving_average()

.. automethod:: pyflwdir.FlwdirRaster.accuflux()

.. automethod:: pyflwdir.FlwdirRaster.upstream_area()


Upscale and subgrid methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyflwdir.FlwdirRaster.upscale()

.. automethod:: pyflwdir.FlwdirRaster.upscale_error()

.. automethod:: pyflwdir.FlwdirRaster.subgrid_rivlen()

.. automethod:: pyflwdir.FlwdirRaster.subgrid_rivslp()

.. automethod:: pyflwdir.FlwdirRaster.subgrid_rivavg()

.. automethod:: pyflwdir.FlwdirRaster.ucat_area()

.. automethod:: pyflwdir.FlwdirRaster.ucat_outlets()


Elevation
^^^^^^^^^

.. automethod:: pyflwdir.FlwdirRaster.dem_adjust()

.. automethod:: pyflwdir.FlwdirRaster.hand()

.. automethod:: pyflwdir.FlwdirRaster.floodplains()

.. automethod:: pyflwdir.dem.fill_depressions()

.. automethod:: pyflwdir.dem.slope()
