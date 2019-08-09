# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from os.path import join
import numba
import math
import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds
from rasterio import features
import geopandas as gp
from shapely.geometry import LineString
import xarray as xr


def nodes_to_ls(nodes, lats, lons, shape):
    return LineString([(x,y) for x, y in zip(*idx_to_xy(nodes, lons, lats, shape[1]))])

# convert raster to GeoDataFrame
def vectorize(data, nodata, transform, crs=None, connectivity=8):
    feats_gen = features.shapes(data, mask=data!=nodata, transform=transform, connectivity=connectivity)
    feats = [{'geometry': geom, 'properties': {'index': idx}} for geom, idx in list(feats_gen)]
    gdf = gp.GeoDataFrame.from_features(feats).set_index('index')
    gdf.index = gdf.index.astype(data.dtype)
    return gdf

# convert GeoDataFrame to raster
def rasterize(gdf, fn=None, col_name='index', # rasterize
              transform=None, out_size=None,
              fill=-9999, dtype=None, fapply=None, **kwargs):
    """Rasterize the index value geopandas dataframe onto a raster defined
    by either its lat/lon coordinates (1d arrays) or
    transform (rasterio transform) and out_size (tuple).
    """
    sindex = gdf.sindex
    nrow, ncol = out_size
    bbox = array_bounds(nrow, ncol, transform)
    idx = list(sindex.intersection(bbox))
    geoms = gdf.iloc[idx,].geometry.values
    values = gdf.iloc[idx,].reset_index()[col_name].values
    dtype = values.dtype if dtype is None else dtype
    if geoms.size > 0:
        shapes = list(zip(geoms, values))
        raster = features.rasterize(
            shapes, out_shape=out_size, fill=fill, transform=transform, **kwargs
        )
    else:
        return
    if dtype is not None:
        raster = np.array(raster).astype(dtype)
    if fapply is not None:
        raster = fapply(raster)
    if fn is not None:
        kwargs = dict(
            driver='GTiff', 
            height=raster.shape[0], 
            width=raster.shape[1], 
            count=1, 
            dtype=raster.dtype, 
            crs=getattr(gdf, 'crs', None), 
            transform=transform,
            nodata=fill
        )
        with rasterio.open(fn, 'w', **kwargs) as dst:
            dst.write(raster, 1)
    else:
        return raster

#NOTE: this assumes python Affine transfrom. should include check
@numba.njit
def calc_slope(dem, nodata, sizeinmetres, transform):
    
    slope = np.zeros(dem.shape,dtype=np.float64)    
    nrow, ncol = dem.shape
    cellsize = transform[0]
    
    elev = np.zeros((3,3), dtype=dem.dtype)

    for r in range(0,nrow):
        for c in range(0,ncol):
            
            
            if dem[r,c] != nodata:
                # EDIT: start with matrix based on central value (inside loop)
                elev[:,:] = dem[r,c]
            
                for dr in range(-1, 2):
                    row = r + dr
                    i = dr + 1
                    if row >= 0 and row < nrow:
                        for dc in range(-1, 2):
                            col = c + dc
                            j = dc + 1
                            if col >= 0 and col < ncol:
                                # EDIT: fill matrix with elevation, except when nodata
                                if dem[row,col] != nodata:
                                    elev[i,j] = dem[row,col]

                dzdx = ((elev[0,0]+2*elev[1,0]+elev[2,0]) - (elev[0,2]+2*elev[1,2]+elev[2,2]))/ (8 * cellsize) 
                dzdy = ((elev[0,0]+2*elev[0,1]+elev[0,2]) - (elev[2,0]+2*elev[2,1]+elev[2,2]))/ (8 * cellsize)

                if sizeinmetres:
                    slp = math.hypot(dzdx, dzdy)
                else:
                    # EDIT: convert lat/lon to dy/dx in meters to calculate hypot
                    radlat = np.radians(transform[5] - 0.5*cellsize - r*cellsize)
                    dy, dx = lat_to_dy(radlat)[0], lat_to_dx(radlat)[0]
                    slp = math.hypot(dzdx*dx, dzdy*dy)
            else:
                slp = nodata
            
            slope[r,c] = slp
        
    return slope

@numba.vectorize(["float64(float64)", "float32(float32)"])
def lat_to_dy(radlat):
    """"
    Determines the length of one degree lat at a given latitude (in meter).
    Input: array of lattitude values for each cell
    Returns: length of a cell lat
    """

    m1 = 111132.92  # latitude calculation term 1
    m2 = -559.82  # latitude calculation term 2
    m3 = 1.175  # latitude calculation term 3
    m4 = -0.0023  # latitude calculation term 4
    # # Calculate the length of a degree of latitude and longitude in meters

    latlen = (
        m1
        + (m2 * math.cos(2.0 * radlat))
        + (m3 * math.cos(4.0 * radlat))
        + (m4 * math.cos(6.0 * radlat))
    )

    return latlen

@numba.vectorize(["float64(float64)", "float32(float32)"])
def lat_to_dx(radlat):
    """"
    Determines the length of one degree long at a given latitude (in meter).
    Input: array of lattitude values for each cell
    Returns: length of a cell long
    """

    p1 = 111412.84  # longitude calculation term 1
    p2 = -93.5  # longitude calculation term 2
    p3 = 0.118  # longitude calculation term 3
    # # Calculate the length of a degree of latitude and longitude in meters

    longlen = (
        (p1 * math.cos(radlat))
        + (p2 * math.cos(3.0 * radlat))
        + (p3 * math.cos(5.0 * radlat))
    )

    return longlen

# cell resolution
def cellare_metres(transform, shape):
    height, width = shape
    resx, resy = transform[0], transform[4] 
    dx = np.ones((1, width), dtype=np.float32)*np.abs(resx)
    dy = np.ones((height, 1), dtype=np.float32)*np.abs(resy)
    return dy * dx

def latlon_cellare_metres(transform, shape):
    dy_lat, dx_lat = latlon_cellres_metres(transform, shape)
    are = dy_lat * dx_lat
    return np.tile(are[:, None], (1, shape[1]))

def latlon_cellres_metres(transform, shape):
    """"""
    lat, _ = transform_to_latlon(transform, shape)
    radlat = np.radians(lat)
    resx, resy = np.abs(transform[0]), np.abs(transform[4])
    dy_lat, dx_lat = lat_to_dy(radlat)*resy, lat_to_dx(radlat)*resx
    return  dy_lat, dx_lat

# conversion
def transform_to_latlon(transform, shape):
    height, width = shape
    xmin, ymin, xmax, ymax = array_bounds(height, width, transform)
    resx, resy = transform[0], transform[4] 
    if np.sign(resy) == -1:
        lats = np.linspace(ymax+resy/2., ymin-resy/2., shape[0])
    else:
        lats = np.linspace(ymin+resy/2., ymax-resy/2., shape[0])
    lons = np.linspace(xmin+resx/2., xmax-resx/2., shape[1])
    return lats, lons

def latlon_to_transform(lat, lon):
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    resX = (lon[-1] - lon[0]) / float(lon.size - 1)
    resY = (lat[-1] - lat[0]) / float(lat.size - 1)
    trans = Affine.translation(lon[0] - resX/2., lat[0] - resY/2.)
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

# @numba.guvectorize(
#     ["void(int32, float64[:], float64[:], int64, float64[:])"],
#     "(n),(m),(),()->(n)"
#     )
@numba.njit
def idx_to_xy(idx, xs, ys, ncol):
    shape = idx.shape
    idx = idx.ravel()
    y = ys[idx // ncol]
    x = xs[idx %  ncol]
    return x.reshape(shape), y.reshape(shape)

if __name__ == "__main__":
    pass