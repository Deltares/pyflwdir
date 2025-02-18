###########
Change Log
###########

in development
**************

0.5.10  (18-02-2025)
********************
* Add support for py 3.13
* bugfix in `from_dataframe` method when downstream index is not present in the dataframe

0.5.9  (19-12-2024)
********************
* Fixed numpy 2.0 compatibility issues
* Fixed bug in strahler stream order method when more than two streams meet.
* Fixed support for interger type DEMs in `from_dem` and `dem.fill_depressions` methods
* Add support for py 3.12
* Add support for pixi
* Add support for large rasters in the flwdir object (Contributed by @robgpita)

0.5.8  (06-Oct-2023)
********************

* support py 3.11 and drop support for py 3.8

0.5.7  (22-Mar-2023)
********************

New
---
* add FlwdirRaster.ucat_volume & subgrid.ucat_volume methods


0.5.6  (15-Nov-2022)
********************

New
---
* `FlwdirRaster.smooth_rivlen` method to smooth river length with a moving window operation over a river network.

Changed
-------
* Move to flit and pyproject.toml for installation and publication
* drop support for python 3.7
* update docs to Sphinx pydata style

Bugfix
------
* use np.uint64 as dtype for large arrays

0.5.5  (16-Feb-2022)
********************

New
---
* read_nextxy method to read binary nextxy data

Bugfix
------
* Support -9 (river outlet at ocean) and -10 (inland river pit) pit values for nextxy data
* Fix 'argmin of an empty sequence' error in dem_dig_d4

Improved
--------
* improve gvf and manning estimates in river_depth method


0.5.4  (18-Jan-2022)
********************

Improved
---------
* prioritize non-boundary cells with same elevation over boundary cells in dem.fill_depressions #17

Bugfix
------
* fix dem_adjust method #16


0.5.3  (18-Nov-2021)
********************

Improved
---------
* add new idxs_pit argument to dem.fill_depressions

Bugfix
------
* min_rivdph argument was not always applied in FlwdirRaster.river_depth


0.5.2 (17-Nov-2021)
*******************

New
---
* Flwdir.river_depth for gradually varying flow (gvf) and manning river depth estimation
* Flwdir.path method to get the indices of flow paths for vector flow directions

Improved
--------
* FlwdirRaster.streams method includes a (zero-length) line at pits and adds option for trace direction in combination with segment end cells.
* Moved accuflux method from FlwdirRaster to parent Flwdir class.
* Additional `how` argument in fillnodata to indicate how to combine values at confluences. Min, max, sum and mean are supported.


0.5.1 (3-Oct-2021)
******************

New
---
* Restore FlwdirRaster.inflow_idxs and FlwdirRaster.outflow_idxs methods

0.5 (28-Sept-2021)
******************
New
---
* General Flwdir object for 1D vector based (instead of raster based) flow directions
* flwdir.from_dataframe methods to derive a Flwdir object from a (Geo)DataFrame based on the row index and a column with downstream row indices.
* dem.fill_depressions and pyflwdir.from_dem methods to derive flow directions from DEMs based on Wang & Lui (2015)
* gis_utils.get_edge method to get a boolean mask of valid cells at the interface with nodata cells or the array edge.
* gis_utils.spread2d method to spread valid values on a 2D raster with optional friction and mask rasters
* FlwdirRaster.dem_dig_d4 method to adjust a DEM such that each cell has a 4D neighbor with equal or lower elevation.
* FlwdirRaster.fillnodata method fill nodata gaps by propagating valid values up or downstream.
* region.region_outlets method; which is also wrapped in the new FlwdirRaster.basin_outlets method
* region.region_dissolve method to dissovle regions into their nearest neighboring region
* FlwdirRaster.subbasins_areas method to derive subbasins based on a minimal area threshold

Improved
--------
* added type="classis" for bottum-up stream order to FlwdirRaster.stream_order, default is type="strahler"
* return subbasin outlet indices for all FlwdirRaster.subbasin* methods
* improved subgrid slope method with optional lstsq regression based slope
* FlwdirRaster.streams takes an optional `idxs_out` argument to derive stream vectors for unit catchments
* FlwdirRaster.streams takes an optional `max_len` argument to split large segments into multiple smaller ones.
* Using the new Flwdir object as common base of FlwdirRaster to share methods and properties
* gis_utils.IDENTITY transform has North -> South orientation (yres < 0) instead of S->N orientation which is in line with flow direction rasters.
* new `restrict_strord` argument in FlwdirRaster.moving_average and FlwdirRaster.moving_median methods to restrict the moving window to cells with same or larger stream order.

Bugfix
------
* strahler stream_order method gave incorrect results
* basins.subbasins_pfafstetter reimplementation to fix mall functioning when jitted
* FlwdirRaster.streams fix when called with optional `min_sto` argument

Deprecated
----------
* FlwdirRaster.main_tributaries method is deprecated due to mallfunctioning when jitted
* FlwdirRaster.inflow_idxs and FlwdirRaster.outflow_idxs

0.4.6
*****
Improved
--------
* vectorizing of local flow directions and streams in seperate methods
* fixed subbasins method
* documentation using nbsphinx

0.4.5
*****
New
---
* subbasin_mask_within_region
* contiguous_area_within_region


0.4.4
*****
Improved
--------
* IHU upscaling (HESS preprint)

0.4.3
*****
Improved
--------
* vectorizing of streams
* pfafstetter method improved
* remove use of pandas and geopandas to limit dependencies

New
---
* new subbasins method
* features method in favor vectorize

0.4.2
*****
Improved
--------
* improved test coverage
* prepared release for pip

New
---

0.4.1
*****
Improved
--------
* code reformatted using black
* improved subgrid river methods

New
---
* subgrid_rivlen, subgrid_rivslp methods in favor of ucat_channel (will be deprecated)

0.4.0
*****
Improved
--------
* improved COM upscaling

New
---

0.3.0
*****
Improved
--------
* simplified data layout based on linear downstream cell indices and a ordered sequence or down- to upstream cell indices.

New
---
* hand - height above neares drain based on Nobre et al. (2016)
* floodplains - flood plain delineation based on Nardi et al. (2019)
* snap/path - methods to follow a streamline in up-  or downstream direction

0.2.0
*****

New
---
* suport for multiple flow direction types

Improved
--------

* upscale - Connecting outlets method is born


0.1.0
*****

New
-----

* setup_network - Setup all upstream - downstream connections based on the flow direcion map.
* get_pits - Return the indices of the pits/outlets in the flow direction map.
* upstream_area - Returns the upstream area [km] based on the flow direction map.
* stream_order - Returns the Strahler Order map
* delineate_basins - Returns a map with basin ids and corresponding bounding boxes.
* basin_map - Returns a map with (sub)basins based on the up- downstream network.
* ucat_map - Returns the unit-subcatchment and outlets map.
* basin_shape - Returns the vectorized basin boundary.
* stream_shape - Returns a GeoDataFrame with vectorized river segments.
* upscale - Returns upscaled flow direction map using the extended effective area method.
* propagate_downstream - Returns a map with accumulated material from all upstream cells.
* propagate_upstream - Returns a map with accumulated material from all downstream cells.
* adjust_elevation - Returns hydrologically adjusted elevation map.
