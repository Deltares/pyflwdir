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
from shapely.geometry import LineString, Point
import xarray as xr

# vectorize
def nodes_to_ls(nodes, lats, lons):
    ncol = lons.size
    return LineString([(x,y) for x, y in zip(*idx_to_xy(nodes, lons, lats, ncol))])

def nodes_to_pnts(nodes, lats, lons):
    ncol = lons.size
    return [Point(x,y) for x, y in zip(*idx_to_xy(nodes, lons, lats, ncol))]

def vectorize(data, nodata, transform, crs=None, connectivity=8):
    feats_gen = features.shapes(data, mask=data!=nodata, transform=transform, connectivity=connectivity)
    feats = [{'geometry': geom, 'properties': {'index': idx}} for geom, idx in list(feats_gen)]
    gdf = gp.GeoDataFrame.from_features(feats, crs=crs).set_index('index')
    gdf.index = gdf.index.astype(data.dtype)
    return gdf

# rasterize
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

# latlon to length conversion
@numba.vectorize(["float64(float64)", "float32(float32)"])
def lat_to_dy(lat):
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
    radlat = math.radians(lat)
    latlen = (
        m1
        + (m2 * math.cos(2.0 * radlat))
        + (m3 * math.cos(4.0 * radlat))
        + (m4 * math.cos(6.0 * radlat))
    )

    return latlen

@numba.vectorize(["float64(float64)", "float32(float32)"])
def lat_to_dx(lat):
    """"
    Determines the length of one degree long at a given latitude (in meter).
    Input: array of lattitude values for each cell
    Returns: length of a cell long
    """

    p1 = 111412.84  # longitude calculation term 1
    p2 = -93.5  # longitude calculation term 2
    p3 = 0.118  # longitude calculation term 3
    # # Calculate the length of a degree of latitude and longitude in meters
    radlat = math.radians(lat)
    longlen = (
        (p1 * math.cos(radlat))
        + (p2 * math.cos(3.0 * radlat))
        + (p3 * math.cos(5.0 * radlat))
    )

    return longlen

@numba.vectorize(["float64(float64)", "float32(float32)"])
def lat_to_area(lat):
    dx = lat_to_dx(lat)
    dy = lat_to_dy(lat)
    return dx*dy

def latlon_cellare_metres(transform, shape):
    lat, _ = transform_to_latlon(transform, shape)
    resx, resy = np.abs(transform[0]), np.abs(transform[4])
    are = lat_to_area(lat)*resx*resy
    return are

def latlon_cellres_metres(transform, shape):
    """"""
    lat, _ = transform_to_latlon(transform, shape)
    resx, resy = np.abs(transform[0]), np.abs(transform[4])
    dy_lat, dx_lat = lat_to_dy(lat)*resy, lat_to_dx(lat)*resx
    return  dy_lat, dx_lat

# transform to latlon and vice versa
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

@numba.njit
def idx_to_xy(idx, xcoords, ycoords, ncol):
    shape = idx.shape
    idx = idx.ravel()
    y = ycoords[idx // ncol]
    x = xcoords[idx %  ncol]
    return x.reshape(shape), y.reshape(shape)

def xy_to_idx(xs, ys, transform, shape):
    r, c = rasterio.transform.rowcol(transform, xs, ys)
    r, c = np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64)
    if not np.all(np.logical_and(
        np.logical_and(r>=0, r<shape[0]),
        np.logical_and(c>=0, c<shape[1])
    )):
        raise ValueError('xy outside domain')
    idx = r * shape[1] + c
    return idx