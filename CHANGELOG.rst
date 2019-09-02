###########
Change Log
###########


0.2.0
************

Improved
-----

* upscale - The upscaling method now reduces the errors by forcing more rivers to connect on highres map.


0.1.0
************
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

