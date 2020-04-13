# -*- coding: utf-8 -*-
"""Methods to derive maps of basin/stream characteristics. These methods require
the basin indices to be ordered from down- to upstream."""

from numba import njit
from numba.typed import List
import numpy as np

# import flow direction definition
from pyflwdir import core
from pyflwdir import gis_utils

_mv = core._mv
__all__ = []

# general methods
@njit
def accuflux(idxs_ds, seq, material, nodata):
    """Returns maps of accumulate upstream <material>
    
    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    material : 1D array
        local values to be accumulated
    nodata : float, integer
        nodata value 

    Returns
    -------
    1D array of material.dtype
        accumulated upstream material
    """
    if idxs_ds.shape != material.shape:
        raise ValueError("Invalid data shape")
    # intialize output with correct dtype
    accu = material.copy()
    for idx0 in seq[::-1]:
        idx_ds = idxs_ds[idx0]
        if idx0 == idx_ds: # pit
            continue
        if accu[idx_ds] != nodata and accu[idx0] != nodata:
            accu[idx_ds] += accu[idx0]
    return accu

@njit
def fillnodata_upstream(idxs_ds, seq, data, nodata):
    """Retuns a a copy of <data> where upstream cell with <nodata> values are filled 
    based on the first downstream valid cell value. 
    
    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    data : 1D array
        original data with missing values
    nodata : float, integer
        nodata value
    
    Returns
    -------
    1D array of data.dtype
        infilled data
    """
    if idxs_ds.shape != data.shape:
        raise ValueError("Invalid data shape")
    res = data.copy()
    for idx0 in seq:  # down- to upstream
        if res[idx0] == nodata:
            res[idx0] = res[idxs_ds[idx0]]
    return res

@njit()
def upstream_area(
    idxs_ds,
    seq,
    ncol,
    latlon=False,
    transform=gis_utils.IDENTITY,
    area_factor=1,
    nodata=-9999.0,
    dtype=np.float64,
):
    """Returns the accumulated upstream area, invalid cells are assinged a the nodata 
    value. The arae is calculated using the transform. If latlon is True, the resolution
    is interpreted in degree and transformed to m2.
    
    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    ncol : int
        number of columns in raster
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY
    area_factor : float, optional
        multiplication factor for unit conversion, by default 1
    nodata : float, optional 
        nodata value, by default -9999.0
    dtype : numpy.dtype, optional
        data output type, by default numpy.float64

    Returns
    -------
    1D array of <dtype>
        accumulated upstream area
    """
    xres, yres, north = transform[0], transform[4], transform[5]
    area0 = abs(xres * yres) / area_factor
    # intialize uparea with correct dtype
    uparea = np.full(idxs_ds.size, nodata, dtype=dtype)
    for idx0 in seq[::-1]: # up- to downstream
        idx_ds = idxs_ds[idx0]
        if idx0 == idx_ds: # pit
            continue
        # local area
        for idx in [idx0, idx_ds]:
            if uparea[idx] == nodata:
                if latlon:
                    lat = north + (idx // ncol + 0.5) * yres
                    area0 = gis_utils.cellarea(lat, xres, yres) / area_factor
                uparea[idx] = area0
        # accumulate upstream area
        uparea[idx_ds] += uparea[idx0]
    return uparea

@njit
def stream_order(idxs_ds, seq):
    """"Returns the cell stream order, invalid cells are assinged a nodata value of -1
        
    The smallest streams, which are the cells with no upstream cells, get 
    order 1. Where two channels of order 1 join, a channel of order 2 
    results downstream. In general, where two channels of order i join, 
    a channel of order i+1 results

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream

    Returns
    -------
    1D array of uint8
        stream order
    """
    nodata = np.uint8(-1)
    strord = np.full(idxs_ds.size, nodata, dtype=np.int8)
    for idx0 in seq[::-1]:  # up- to downstream
        idx_ds = idxs_ds[idx0]
        if idx0 == idx_ds: # pit
            continue
        sto = strord[idx0]
        # set sto = 1 if headwater cell
        if sto < 1:  
            sto = np.uint8(1)
            strord[idx0] = sto
        # update next downstream cell
        sto_ds = strord[idx_ds]
        if sto > sto_ds:
            strord[idx_ds] = sto
        elif sto == sto_ds:
            strord[idx_ds] += 1
    return strord

def downstream_distance(
    idxs_ds,
    seq,
    ncol,
    mask=None,
    latlon=False,
    transform=gis_utils.IDENTITY,
):
    """Returns distance to outlet or next downstream True cell in mask
    
    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    ncol : int
        number of columns in raster
    mask : 1D-array of bool, optional
        True if stream cell
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY
    
    Returns
    -------
    1D array of float
        distance to outlet or next downstream True cell
    """
    dist = np.full(idxs_ds.size, -9999.0, dtype=np.float64)
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        if idx0 == idx_ds or (mask is not None and mask[idx0] == True):
            dist[idx0] == 0
            continue
        d = gis_utils.distance(idx0, idx_ds, ncol, latlon, transform)
        dist[idx0] = dist[idx_ds] + d
    return dist