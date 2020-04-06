# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Core upstream / downstream functionality based on 
parsed indices from core_xxx.py submules"""

from numba import njit, prange
from numba.typed import List
import numpy as np
import math
import heapq

from pyflwdir import gis_utils

__all__ = []
_mv = np.uint32(-1)


@njit
def _idxs_us(idxs_ds):
    """Returns a 2D array with upstream cell indices for each cell 
    
    Parameters
    ----------
    idxs_ds : 1D-array of uint32
        sparse indices of downstream cells
    
    Returns
    -------
    2D-array of uint32
        indices of upstream cells
    """
    n_up = n_upstream(idxs_ds)
    d = np.int64(np.max(n_up))
    n = idxs_ds.size
    # 2D arrays of upstream index
    idxs_us = np.full((n, d), _mv, dtype=np.uint32)
    n_up[:] = np.uint8(0)
    for i in range(n):
        i_ds = idxs_ds[i]
        if i_ds == i:
            continue
        # valid ds cell
        ii = n_up[i_ds]
        idxs_us[i_ds][ii] = i
        n_up[i_ds] += 1
    return idxs_us


#### UPSTREAM  functions ####
@njit
def upstream_sum(idxs_ds, arr, mv):
    """Returns sum of first upstream values 
    
    Parameters
    ----------
    idxs_ds : 1D-array of uint32
        sparse indices of downstream cells
    
    Returns
    -------
    2D-array of uint32
        indices of upstream cells
    """
    # 2D arrays of upstream index
    arr_sum = np.full(arr.size, 0, dtype=arr.dtype)
    for i in range(n):
        i_ds = idxs_ds[i]
        if i_ds == i:
            continue
        # valid ds cell
        if arr[i] == mv or arr_sum[i_ds] == mv:
            arr_sum = mv
        else:
            arr_sum[i_ds] += arr[i]
    return arr_sum


@njit
def upstream(idx0, idxs_us):
    """Returns the sparse upstream indices.
    
    Parameters
    ----------
    idx0 : array_like of uint32
        sparse index of start cell
    idxs_us : 1D-array of uint32
        sparse indices of upstream cells

    Returns
    -------
    1D-array of uint32  
        upstream indices 
    """
    idxs_us0 = idxs_us[idx0, :].ravel()
    return idxs_us0[idxs_us0 != _mv]


@njit
def _main_upstream(idx0, idxs_us, uparea_sparse, upa_min=0.0):
    """Returns the index of the upstream cell with 
    the largest uparea."""
    idx_us0 = _mv
    upa0 = upa_min
    for idx_us in idxs_us[idx0, :]:
        if idx_us == _mv:
            break
        if uparea_sparse[idx_us] >= upa0:
            idx_us0 = idx_us
            upa0 = uparea_sparse[idx_us]
    return idx_us0


@njit
def main_upstream(idxs, idxs_us, uparea_sparse, upa_min=0.0):
    """Returns the index of the upstream cell with 
    the largest uparea.
    
    Parameters
    ----------
    idxs : 1D-array of uint32
        sparse indices of local cells
    idxs_us : 1D-array of uint32
        sparse indices of upstream cells
    uparea_sparse : 1D-array
        sparse array with upstream area values
    upa_min : float, optional 
        minimum upstream area for cell to be considered
    
    Returns
    -------
    1D-array of uint32  
        sparse upstream indices 
    """
    if idxs_us.shape[0] != uparea_sparse.size:
        raise ValueError("uparea_sparse has invalid size")
    idxs_us_main = np.ones(idxs.size, dtype=idxs_us.dtype) * _mv
    for i in range(idxs.size):
        idxs_us_main[i] = _main_upstream(
            idxs[i], idxs_us, uparea_sparse, upa_min=upa_min
        )
    return idxs_us_main


@njit
def n_upstream(idxs_ds):
    """Returns array with number of upstream cells per cell.
    
    Parameters
    ----------
    idxs_ds : 1D-array of uint32
        sparse indices of downstream cells
    
    Returns
    -------
    1D-array of uint8  
        number of upstream cells 
    """
    n = idxs_ds.size
    n_up = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        i_ds = idxs_ds[i]
        if i != i_ds:  # pit
            n_up[i_ds] += 1
    return n_up


@njit
def idxs_headwater(idxs_us):
    """Returns indices of cells without upstream neighbors"""
    return np.where(idxs_us[:, 0] == _mv)[0]


@njit
def main_tibutaries(idx0, idxs_us, uparea_sparse, idx_end=_mv, min_upa=0, n=4):
    """Return indices of n largest tributaries upstream from idx0 and downstream
    from idx_end
    
    Parameters
    ----------
    idx0 : uint32
        sparse index of start cell
    idxs_us : 1D-array of uint32
        sparse indices of upstream cells
    uparea_sparse : 1D-array
        sparse array with upstream area values
    idx_end : uint32, optional
        most upstream index, by default set to missing value (no fixed most upstream cell)
    upa_min : float, optional 
        minimum upstream area for subbasin
    n : int, optional
        number of tributaries, by default 4

    Returns
    -------
    1D array of uint32 with size n
        sparse indices of largest tributaries
    1D array of uint32 with size n*2+1
        sparse indices of inter- and subbasins   
    """
    # use heapq to keep list of n largest tributaries
    # initialize with n (zero, mv, mv, i) tuples
    upa_ini = np.zeros(n, dtype=uparea_sparse.dtype)
    ntrib = [(upa_ini[i], _mv, _mv, _mv) for i in range(n)]
    heapq.heapify(ntrib)
    upa0 = max(min_upa, heapq.nsmallest(1, ntrib)[0][0])
    upa_main = upa0
    upa_main0 = uparea_sparse[idx0]
    i = np.uint32(0)  # sequence of basins
    # ordered indices of sub- and interbasins from down to upstream
    idxs = np.zeros(n * 2 + 1, dtype=idxs_us.dtype)
    idxs[0] = idx0
    # stop when no more upstream cells or upstream area on main stream
    # smaller than nth largest tributary
    while idx0 != _mv and upa_main >= upa0 and idx0 != idx_end:
        idx_us_main = _mv
        upa_main = 0.0
        # find main upstream
        for idx_us in idxs_us[idx0, :]:
            if idx_us == _mv:
                break
            upa = uparea_sparse[idx_us]
            if upa > upa_main:
                idx_us_main = idx_us
                upa_main = upa
        # check min size interbasins
        # if upa_main0 - uparea_sparse[idx0] > upa0:
        # add tributaries
        for idx_us in idxs_us[idx0, :]:
            if idx_us == _mv:
                break
            upa = uparea_sparse[idx_us]
            if upa < upa_main and upa > upa0:
                i += 1
                heapq.heappushpop(ntrib, (upa, idx_us, idx_us_main, np.uint32(i)))
                upa0 = max(min_upa, heapq.nsmallest(1, ntrib)[0][0])
                upa_main0 = uparea_sparse[idx_us_main]
        # next iter
        idx0 = idx_us_main
    seq = np.argsort(np.array([ntrib[i][-1] for i in range(n)]))
    idxs_subbasin = np.array([ntrib[i][1] for i in seq])
    idxs_interbasin = np.array([ntrib[i][2] for i in seq])
    for i in range(1, n * 2 + 1):
        pfaf_id = i + 1
        if pfaf_id % 2 == 0:  # even pfafstetter code -> subbasins
            idx0 = idxs_subbasin[pfaf_id // 2 - 1]
        else:  # odd pfafstetter code -> inter-subbasin
            idx0 = idxs_interbasin[pfaf_id // 2 - 1]
        idxs[i] = idx0
    return idxs_subbasin, idxs


####  DOWNSTREAM functions ####
@njit
def downstream(idx0, idxs_ds):
    """Returns next downstream indices.
    
    Parameters
    ----------
    idx0 : array_like of uint32
        sparse index of start cell
    idxs_ds : 1D-array of uint32
        sparse indices of downstream cells

    Returns
    -------
    array_like of uint32
        sparse downstream index
    """
    return idxs_ds[idx0]


@njit
def downstream_all(
    idx0,
    idxs_ds,
    mask_sparse=None,
    max_length=None,
    real_length=False,
    idxs_dense=None,
    ncol=None,
    latlon=False,
    transform=gis_utils.IDENTITY,
):
    """Returns sparse indices of all downstream cells, including the start cell, until:
    - a pit is found OR
    - a True cell is found in mask OR
    - the distance from the start point is larger than max_length. 

    Parameters
    ----------
    idx0 : uint32
        sparse index of start cells
    idxs_ds : 1D-array of uint32
        sparse indices of downstream cells
    mask_sparse : 1D-array of bool, optional
        True if stream cell
    max_length : float, optional
        maximum distance to move downstream, by default None
    real_length : bool, optional
        unit of length in meters if True, cells if False, by default False
    idxs_dense : 1D-array of uint32
        linear indices of dense raster
    ncol : int
        number of columns in raster
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY
    
    Returns
    -------
    1D-array of uint32  
        sparse indices of all downstream cells
    float
        distance between start and end cell
    """
    idxs = []
    idxs.append(np.uint32(idx0))
    at_stream = mask_sparse is not None and mask_sparse[idx0]
    ltot = 0.0
    l = 1.0
    while not at_stream:
        idx_ds = idxs_ds[idx0]
        if idx_ds == idx0:  # pit
            break
        if real_length and idxs_dense is not None:
            l = _downstream_dist(
                idx0, idxs_ds, idxs_dense, ncol, latlon=latlon, transform=transform
            )[1]
        if max_length is not None and ltot + l > max_length:
            break
        ltot += l
        idx0 = idx_ds
        idxs.append(np.uint32(idx0))
        at_stream = mask_sparse is not None and mask_sparse[idx0]
    return np.array(idxs, dtype=np.uint32), ltot


@njit
def downstream_path(
    idxs0,
    idxs_ds,
    mask_sparse=None,
    max_length=None,
    real_length=False,
    idxs_dense=None,
    ncol=None,
    latlon=False,
    transform=gis_utils.IDENTITY,
):
    """See downstream_all method, except this function works for a 1D-array of sparse 
    start indices.

    Returns
    -------
    list of 1D-array of uint32  
        sparse indices of all downstream cells
    1D-array of float
        distance between start and end cell
    """
    paths = List()
    dists = np.zeros(idxs0.size, dtype=np.float64)
    for i in range(idxs0.size):
        path, d = downstream_all(
            idxs0[i],
            idxs_ds,
            mask_sparse=mask_sparse,
            max_length=max_length,
            real_length=real_length,
            idxs_dense=idxs_dense,
            ncol=ncol,
            latlon=latlon,
            transform=transform,
        )
        paths.append(path)
        dists[i] = d
    return paths, dists


@njit
def downstream_snap(
    idxs0,
    idxs_ds,
    mask_sparse=None,
    real_length=False,
    idxs_dense=None,
    ncol=None,
    latlon=False,
    transform=gis_utils.IDENTITY,
):
    """Returns indices the most downstream cell where mask is True or is pit.
    
    See downstream_all method for paramters, except this function works for a 1D-array 
    of sparse start indices.
    
    Returns
    -------
    1D-array of uint32  
        sparse indices of most downstream cells
    1D-array of float
        distance between start and end cell
    """
    idxs = idxs0.copy()
    dists = np.zeros(idxs0.size, dtype=np.float64)
    for i in range(idxs0.size):
        path, d = downstream_all(
            idxs0[i],
            idxs_ds,
            mask_sparse=mask_sparse,
            real_length=real_length,
            idxs_dense=idxs_dense,
            ncol=ncol,
            latlon=latlon,
            transform=transform,
        )
        idxs[i] = path[-1]
        dists[i] = d
    return idxs, dists


@njit
def downstream_dist(
    idxs_ds, idxs_dense, ncol, latlon=False, transform=gis_utils.IDENTITY
):
    """Return the distance to the next downstream cell"""
    dists = np.zeros(idxs_ds.size, dtype=np.float64)
    for idx0 in range(idxs_ds.size):
        dists[idx0] = _downstream_dist(
            idx0, idxs_ds, idxs_dense, ncol, latlon=latlon, transform=transform
        )[1]

    return dists


@njit
def _downstream_dist(
    idx0, idxs_ds, idxs_dense, ncol, latlon=False, transform=gis_utils.IDENTITY
):
    """Return the next downstream cell index as well 
    as the length in downstream direction assuming a 
    regular raster defined by the affine transform.
    
    Parameters
    ----------
    idx0 : uint32
        sparse index of start cell
    idxs_ds : 1D-array of uint32
        sparse indices of downstream cells
    idxs_dense : 1D-array of uint32
        linear indices of dense raster
    ncol : int
        number of columns in raster
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping

    Returns
    -------
    uint32
        downstream index
    float
        length
    """
    xres, yres, north = transform[0], transform[4], transform[5]
    idx1 = downstream(idx0, idxs_ds)  # next downstream
    # convert to flattened raster indices
    idx = idxs_dense[idx0]
    idx_ds = idxs_dense[idx1]
    # compute delta row, col
    r = idx_ds // ncol
    dr = (idx // ncol) - r
    dc = (idx % ncol) - (idx_ds % ncol)
    if latlon:  # calculate cell size in metres
        lat = north + (r + 0.5) * yres
        dy = 0.0 if dr == 0 else gis_utils.degree_metres_y(lat) * yres
        dx = 0.0 if dc == 0 else gis_utils.degree_metres_x(lat) * xres
    else:
        dy = xres
        dx = yres
    return idx1, math.hypot(dy * dr, dx * dc)  # length


##### UP/DOWNSTREAM window ####
@njit
def flwdir_window(idx0, n, idxs_ds, idxs_us, uparea_sparse, upa_min=0.0):
    """Returns the indices of between the nth upstream to nth downstream cell from 
    the current cell. Upstream cells are with based on the  _main_upstream method."""
    idxs = np.full(n * 2 + 1, _mv, idxs_ds.dtype)
    idxs[n] = idx0
    # get n downstream cells
    for i in range(n):
        idx_ds = idxs_ds[idx0]
        if idx_ds == idx0:  # pit
            break
        idx0 = idx_ds
        idxs[n + i + 1] = idx0
    # get n upstreams cells
    idx0 = idxs[n]
    for i in range(n):
        idx_us = _main_upstream(idx0, idxs_us, uparea_sparse, upa_min=upa_min)
        if idx_us == _mv:  # at headwater / no upstream cells
            break
        idx0 = idx_us
        idxs[n - i - 1] = idx0
    return idxs


#### PIT / LOOP INDICES ####
@njit
def pit_indices(idxs_ds):
    """Returns the sparse pit indices, i.e. where the 
    downstream cell is equal to the local cell.
    
    Parameters
    ----------
    idxs_ds : 1D-array of uint32
        sparse indices of downstream cells

    Returns
    -------
    1D-array of uint32  
        sparse downstream indices
    """
    idx_lst = []
    for idx0 in range(idxs_ds.size):
        if idx0 == idxs_ds[idx0]:
            idx_lst.append(idx0)
    return np.array(idx_lst, dtype=idxs_ds.dtype)


@njit
def loop_indices(idxs_ds, idxs_us):
    """Returns the sparse loop indices, i.e. for cells
    which do not have a pit at its most downstream end.
    
    Parameters
    ----------
    idxs_ds, idxs_us : 1D-array of uint32
        sparse indices of pit, downstream, upstream cells

    Returns
    -------
    1D-array of uint32  
        sparse loop indices 
    """
    # sparse lists
    no_loop = idxs_ds == _mv
    idxs_ds0 = pit_indices(idxs_ds)
    if idxs_ds0.size > 0:  # no valid cells!
        no_loop[idxs_ds0] = True
    # loop over pits and mark upstream area
    while True:
        idxs_us0 = upstream(idxs_ds0, idxs_us)
        if idxs_us0.size == 0:
            break
        no_loop[idxs_us0] = ~no_loop[idxs_us0]
        idxs_ds0 = idxs_us0  # next iter
    return np.where(~no_loop)[0]


## VECTORIZE
def to_linestring(idxs_ds, xs, ys, mask=None):
    """Returns a list of LineString for each up- downstream connection"""
    try:
        from shapely.geometry import LineString
    except ImportError:
        msg = "The `to_linestring` method requires the additional shapely package."
        raise ImportError(msg)

    geoms = list()
    for idx0 in range(idxs_ds.size):
        if mask is not None and mask[idx0] != 1:
            continue
        idx_ds = idxs_ds[idx0]
        geoms.append(LineString([(xs[idx0], ys[idx0]), (xs[idx_ds], ys[idx_ds]),]))
    return geoms


#### sparse data indexing and reordering functions ####
@njit
def _sparse_idx(idxs, idxs_dense, size):
    """Convert linear indices of dense raster to sparse indices.
    
    Parameters
    ----------
    idxs : 1D-array of uint32
        linear indices of dense raster to be converted
    idxs_dense : 1D-array of uint32
        linear indices of dense raster of valid cells
    size : int
        size of flwdir raster

    Returns
    -------
    1D-array of uint32    
        sparse indices 
    """
    # NOTE dense idxs in intp data type, sparse idxs in uint32
    # TODO: test if this can be done faster with sorted idxs array
    if np.any(idxs < 0) or np.any(idxs >= size):
        raise ValueError("Index out of bounds")
    idxs_sparse = np.full(size, _mv, np.uint32)
    idxs_sparse[idxs_dense] = np.array(
        [i for i in range(idxs_dense.size)], dtype=np.uint32
    )
    return idxs_sparse[idxs]


@njit
def _densify(data, idxs_dense, shape, nodata=-9999):
    """Densify sparse 1D array.
    
    Parameters
    ----------
    data : 1D-array 
        1D sparse data
    idxs_dense : 1D-array of uint32
        linear indices of dense raster
    shape : tuple of int
        shape of output raster

    Returns
    -------
    2D-array of data dtype
        2D raster data 
    """
    if idxs_dense.size != data.size:
        raise ValueError("data has invalid size")
    data_out = np.full(shape[0] * shape[1], nodata, dtype=data.dtype)
    data_out[idxs_dense] = data
    return data_out.reshape(shape)
