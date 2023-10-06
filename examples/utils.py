import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import cartopy.crs as ccrs
import descartes
import numpy as np
import os
import rasterio
from rasterio import features
import geopandas as gpd

np.random.seed(seed=101)
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.dpi"] = 256
plt.style.use("seaborn-v0_8-whitegrid")

# read example elevation data and derive background hillslope
fn = os.path.join(os.path.dirname(__file__), "rhine_elv0.tif")
with rasterio.open(fn, "r") as src:
    elevtn = src.read(1)
    extent = np.array(src.bounds)[[0, 2, 1, 3]]
    crs = src.crs
ls = matplotlib.colors.LightSource(azdeg=115, altdeg=45)
hs = ls.hillshade(np.ma.masked_equal(elevtn, -9999), vert_exag=1e3)


# convenience method for plotting
def quickplot(
    gdfs=[], raster=None, hillshade=True, extent=extent, hs=hs, title="", filename=""
):
    fig = plt.figure(figsize=(8, 15))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    # plot hillshade background
    if hillshade:
        ax.imshow(
            hs,
            origin="upper",
            extent=extent,
            cmap="Greys",
            alpha=0.3,
            zorder=0,
        )
    # plot geopandas GeoDataFrame
    for gdf, kwargs in gdfs:
        gdf.plot(ax=ax, **kwargs)
    if raster is not None:
        data, nodata, kwargs = raster
        ax.imshow(
            np.ma.masked_equal(data, nodata),
            origin="upper",
            extent=extent,
            **kwargs,
        )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize="large")
    ax.text(
        0.01, 0.01, "created with pyflwdir", transform=ax.transAxes, fontsize="large"
    )
    if filename:
        plt.savefig(f"{filename}.png")
    return ax


# convenience method for vectorizing a raster
def vectorize(data, nodata, transform, crs=crs, name="value"):
    feats_gen = features.shapes(
        data,
        mask=data != nodata,
        transform=transform,
        connectivity=8,
    )
    feats = [
        {"geometry": geom, "properties": {name: val}} for geom, val in list(feats_gen)
    ]

    # parse to geopandas for plotting / writing to file
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf[name] = gdf[name].astype(data.dtype)
    return gdf
