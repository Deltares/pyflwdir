###########
Change Log
###########

Unreleased
***********
New
---
* classis "bottum up" stream order 
* General Flwdir object for 1D vector based (instead of raster based) flow directions
* flwdir.from_dataframe methods to derive a Flwdir object from a (Geo)DataFrame based on the row index and a column with downstream row indices.
* dem.fill_depressions and pyflwdir.from_dem methods to derive flow directions from DEMs based on Wang & Lui (2015) 
* gis_utils.get_edge method to get a boolean mask of valid cells at the interface with nodata cells or the array edge.
* FlwdirRaster.adjust_dem_d4 method to adjust a DEM such that each cell has a 4D neighbor with equal or lower elevation.
* new `fillnodata` method fill nodata gaps by propagating valid values up or downstream.

Improved
--------
* streams takes a `idxs_out` argument to derive stream vectors for unit catchments
* streams takes a `max_len` argument to split large segments into multiple smaller ones.
* Use of Flwdir as common base of FlwdirRaster to share methods and properties 
* changed IDENTITY transform to have North -> South orientation (yres < 0) which is in line with flow direction rasters.
* new `restrict_strord` argument in `moving_average` and `moving_median` methods to restrict the moving window to cells with same or larger stream order.

Bugfix
------
* pfafstetter subbasins reimplementation to fix mall functioning when jitted
* stream_order method gave incorrect results
* streams method gave incorrect segments with the min_sto argument

Deprecated
----------
* main_tributaries method is deprecated due to mallfunctioning when jitted

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

