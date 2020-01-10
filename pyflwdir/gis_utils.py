# -- coding: utf-8 --
from numba import vectorize, njit
import numpy as np
import math
from affine import identity as IDENTITY

def idxs_to_coords(idxs_valid, affine, shape):
    """Returs centered coordinates of idxs raster indices based affine.

    Parameters
    ----------
    affine : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.

    Returns
    -------
    x, y coordinate arrays : tuple of ndarray of float
    """
    ncol = shape[1]
    r = idxs_valid // ncol
    c = idxs_valid % ncol 
    x, y = affine * (c+0.5, r+0.5)
    return x, y

def affine_to_coords(affine, shape):
    """Returs a raster axis with pixel center coordinates based on the affine.

    Parameters
    ----------
    affine : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.

    Returns
    -------
    x, y coordinate arrays : tuple of ndarray of float
    """
    height, width = shape
    x_coords, _ = affine * (np.arange(width) + 0.5, np.zeros(width) + 0.5)
    _, y_coords = affine * (np.zeros(height) + 0.5, np.arange(height) + 0.5)
    return x_coords, y_coords

def reggrid_dx(lats, lons):
    """returns a the cell widths (dx) for a regular grid with cell centers lats & lons [m]"""
    xres = np.abs(np.mean(np.diff(lons)))
    dx = degree_metres_x(lats)*xres
    return dx[None,:]*np.ones((lats.size,lons.size), dtype=lats.dtype)

def reggrid_dy(lats, lons):
    """returns a the cell heights (dy) for a regular grid with cell centers lats & lons [m]"""
    yres = np.abs(np.mean(np.diff(lats)))
    dy = degree_metres_y(lats)*yres
    return dy[:,None]*np.ones((lats.size,lons.size), dtype=lats.dtype)

def reggrid_area(lats, lons):
    """returns a the cell area for a regular grid with cell centres lats & lons [m2]"""
    xres = np.abs(np.mean(np.diff(lons)))
    yres = np.abs(np.mean(np.diff(lats)))
    return cellarea(lats, xres, yres)[:,None]*np.ones((lats.size,lons.size), dtype=lats.dtype)

@vectorize([
    "float64(float64,float64,float64)", 
    "float32(float32,float32,float32)"
    ])
def cellarea(lat, xres, yres):
    """returns the area of cell with a given resolution (resx,resy) at a given cell center latitude [2]"""
    _R = 6371e3 # Radius of earth in m. Use 3956e3 for miles
    l1 = math.radians(lat-abs(yres)/2.)
    l2 = math.radians(lat+abs(yres)/2.)
    dx = math.radians(xres)
    return _R**2*dx*(math.sin(l2) - math.sin(l1))

@vectorize([
    "float64(float64,float64,float64,float64)", 
    "float32(float32,float32,float32,float32)"
    ])
def distance(lon1, lat1, lon2, lat2):
    """returns the great circle distance between two points on the earth [m]"""
    # haversine formula 
    _R = 6371e3 # Radius of earth in m. Use 3956e3 for miles
    dlon = math.radians(lon2 - lon1) 
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    return c * _R

# latlon to length conversion
@vectorize(["float64(float64)", "float32(float32)"])
def degree_metres_y(lat):
    """"returns the verical length of a degree in metres at a given latitude"""

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

@vectorize(["float64(float64)", "float32(float32)"])
def degree_metres_x(lat):
    """"returns the horizontal length of a degree in metres at a given latitude"""
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
