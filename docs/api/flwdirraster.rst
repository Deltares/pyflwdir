.. currentmodule:: pyflwdir

FlwdirRaster
------------

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster


Input/Output
^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_generated

   from_array
   from_dem
   FlwdirRaster.to_array
   FlwdirRaster.load
   FlwdirRaster.dump


Flow direction attributes
^^^^^^^^^^^^^^^^^^^^^^^^^

The following attributes describe the flow direction and are at the core to the object.

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.idxs_ds
   FlwdirRaster.idxs_us_main
   FlwdirRaster.idxs_seq
   FlwdirRaster.idxs_pit
   FlwdirRaster.ncells
   FlwdirRaster.rank
   FlwdirRaster.isvalid
   FlwdirRaster.mask
   FlwdirRaster.area
   FlwdirRaster.distnc
   FlwdirRaster.n_upstream


Flow direction methods
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.order_cells
   FlwdirRaster.main_upstream
   FlwdirRaster.add_pits
   FlwdirRaster.repair_loops
   FlwdirRaster.vectorize


Raster & geospatial attributes and methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FlwdirRaster object contains :py:attr:`FlwdirRaster.shape`, :py:attr:`FlwdirRaster.transform`
and :py:attr:`FlwdirRaster.latlon` attributes describing its geospatial location. The first
attribute is required at initializiation, while the others can be set later.

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.set_transform
   FlwdirRaster.index
   FlwdirRaster.xy
   FlwdirRaster.bounds
   FlwdirRaster.extent


Streams and flow paths
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.stream_order
   FlwdirRaster.path
   FlwdirRaster.snap
   FlwdirRaster.outflow_idxs
   FlwdirRaster.inflow_idxs
   FlwdirRaster.stream_distance
   FlwdirRaster.streams
   FlwdirRaster.geofeatures


(Sub)basins
^^^^^^^^^^^

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.basins
   FlwdirRaster.subbasins_streamorder
   FlwdirRaster.subbasins_pfafstetter
   FlwdirRaster.subbasins_area
   FlwdirRaster.basin_outlets
   FlwdirRaster.basin_bounds


Up- and downstream values
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.downstream
   FlwdirRaster.upstream_sum


Up- and downstream arithmetics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.accuflux
   FlwdirRaster.upstream_area
   FlwdirRaster.moving_average
   FlwdirRaster.moving_median
   FlwdirRaster.smooth_rivlen
   FlwdirRaster.fillnodata


Upscale and subgrid methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.upscale
   FlwdirRaster.upscale_error
   FlwdirRaster.subgrid_rivlen
   FlwdirRaster.subgrid_rivslp
   FlwdirRaster.subgrid_rivavg
   FlwdirRaster.subgrid_rivmed
   FlwdirRaster.ucat_area
   FlwdirRaster.ucat_outlets


Elevation
^^^^^^^^^

.. autosummary::
   :toctree: ../_generated

   FlwdirRaster.dem_adjust
   FlwdirRaster.dem_dig_d4
   FlwdirRaster.hand
   FlwdirRaster.floodplains
