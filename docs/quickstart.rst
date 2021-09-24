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
vector file with `rasterio <https://rasterio.readthedocs.io/>`__

.. code-block:: python

    basins = flw.basins()

Vectorize the stream network and save to a geojson file:

.. code-block:: python

    import geopandas as gpd
    feat = flw.streams()
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf.to_file('streams.geojson', driver='GeoJSON')
