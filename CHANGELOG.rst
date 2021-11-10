###########
Change Log
###########

0.5.2 (unreleased)
******************

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

