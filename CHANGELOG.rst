###########
Change Log
###########

0.4.3
*****
Improved
--------
* vectorizing of streams
* pfafstetter method improved
* remove use of pandas and geopandas to limit dependencies

Added
-----
* new subbasins method
* features method in favor vectorize

0.4.2
*****
Improved
--------
* improved test coverage
* prepared release for pip
* version for HESS preprint

Added
-----

0.4.1
*****
Improved
--------
* code reformatted using black
* improved subgrid river methods

Added
-----
* subgrid_rivlen, subgrid_rivslp methods in favor of ucat_channel (will be deprecated)

0.4.0
*****
Improved
--------
* improved COM upscaling

Added
-----

0.3.0
*****
Improved
--------
* simplified data layout based on linear downstream cell indices and a ordered sequence or down- to upstream cell indices.

Added
-----
* hand - height above neares drain based on Nobre et al. (2016)
* floodplains - flood plain delineation based on Nardi et al. (2019)
* snap/path - methods to follow a streamline in up-  or downstream direction

0.2.0
*****

Added
-----
* suport for multiple flow direction types

Improved
--------

* upscale - Connecting outlets method is born


0.1.0
*****

Added
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

