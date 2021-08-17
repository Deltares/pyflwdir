# -*- coding: utf-8 -*-
"""Methods to derive maps of basin/stream characteristics. These methods require
the basin indices to be ordered from down- to upstream."""

from numba import njit
import numpy as np

# import local libraries
from pyflwdir import gis_utils, arithmetics, core

__all__ = []


# general methods
@njit
def accuflux(idxs_ds, seq, data, nodata):
    """Returns maps of accumulate upstream <data>

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    data : 1D array
        local values to be accumulated
    nodata : float, integer
        nodata value

    Returns
    -------
    1D array of data.dtype
        accumulated upstream data
    """
    # intialize output with correct dtype
    accu = data.copy()
    for idx0 in seq[::-1]:  # up- to downstream
        idx_ds = idxs_ds[idx0]
        if idx0 != idx_ds and accu[idx_ds] != nodata and accu[idx0] != nodata:
            accu[idx_ds] += accu[idx0]
    return accu


@njit
def accuflux_ds(idxs_ds, seq, data, nodata):
    """Returns maps of accumulate downstream <data>

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    data : 1D array
        local values to be accumulated
    nodata : float, integer
        nodata value

    Returns
    -------
    1D array of data.dtype
        accumulated upstream data
    """
    # intialize output with correct dtype
    accu = data.copy()
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        if idx0 != idx_ds and accu[idx_ds] != nodata and accu[idx0] != nodata:
            accu[idx0] += accu[idx_ds]
    return accu


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

    NOTE: does not require area grid in memory

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
    # intialize uparea with correct dtype
    uparea = np.full(idxs_ds.size, nodata, dtype=dtype)
    # local area
    xres, yres, north = transform[0], transform[4], transform[5]
    if latlon:
        for idx in seq:
            lat = north + (idx // ncol + 0.5) * yres
            uparea[idx] = gis_utils.cellarea(lat, xres, yres) / area_factor
    else:
        uparea[seq] = abs(xres * yres) / area_factor
    # accumulate upstream area
    for idx0 in seq[::-1]:  # up- to downstream
        idx_ds = idxs_ds[idx0]
        if idx0 != idx_ds:
            uparea[idx_ds] += uparea[idx0]
    return uparea


@njit
def streams(idxs_ds, seq, mask=None, max_len=0, mv=core._mv):
    """Returns list of linear indices per stream of equal stream order.

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    mask : 1D-array of bool, optional
        Mask of stream cells
    max_len: int, optional
        Maximum length of a single stream segment measured in cells
        Longer streams segments are divided into smaller segments of equal length
        as close as possible to max_len.

    Returns
    -------
    streams : list of 1D-arrays of intp
        linear indices of streams
    """
    if mask is None:
        nup = core.upstream_count(idxs_ds=idxs_ds, mv=mv)
    else:
        nup = arithmetics.upstream_sum(
            idxs_ds=idxs_ds, data=mask.astype(np.int8), mv=mv
        )
    # get list of indices arrays of segments
    streams = []
    done = np.array([bool(0) for _ in range(idxs_ds.size)])  # all False
    for idx0 in seq[::-1]:
        if done[idx0] or (mask is not None and ~mask[idx0]):
            continue
        idxs = [idx0]  # initiate with correct dtype
        while True:
            done[idx0] = True
            idx_ds = idxs_ds[idx0]
            idxs.append(idx_ds)
            if nup[idx_ds] > 1 or idx_ds == idx0:
                l = len(idxs)
                if l > max_len > 0:
                    n, k = l, 1
                    if (l / max_len) > 1.5:
                        k = round(l / max_len)
                        n = round(l / k)
                    for i in range(k):  # split into k segments with overlapping point
                        if i + 1 == k:
                            streams.append(np.array(idxs[i * n :], dtype=idxs_ds.dtype))
                        else:
                            _idxs = idxs[i * n : n * (i + 1) + 1]
                            streams.append(np.array(_idxs, dtype=idxs_ds.dtype))
                else:
                    streams.append(np.array(idxs, dtype=idxs_ds.dtype))
                break
            idx0 = idx_ds
    return streams


@njit
def stream_order(idxs_ds, seq, mask=None):
    """Returns the cell stream order, invalid cells are assinged a nodata value of -1

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
    mask : 1D-array of bool, optional
        True if stream cell

    Returns
    -------
    1D array of uint8
        stream order
    """
    # initialize valid cells with stream order 1
    if mask is None:
        strord = np.full(idxs_ds.size, 0, dtype=np.uint8)
        strord[seq] = 1
    else:
        strord = np.where(mask, np.uint8(1), np.uint8(0))
    count = np.zeros(idxs_ds.size, dtype=np.uint8)
    for idx0 in seq[::-1]:  # up- to downstream
        # set strean order of current cell
        if strord[idx0] == 0:  # invalid cell
            continue
        if count[idx0] > 1:  # bump order if count > 1
            strord[idx0] += 1
        sto = strord[idx0]
        # set maximum upstream order at dowsntream neighbor
        # and count number of upstream cells with maximum order
        idx_ds = idxs_ds[idx0]
        if idx0 != idx_ds and strord[idx_ds] != 0:  # pit
            sto_ds = strord[idx_ds]
            if sto > sto_ds:
                count[idx_ds] = 1
                strord[idx_ds] = sto
            elif sto == sto_ds:
                count[idx_ds] += 1
    return strord


def stream_distance(
    idxs_ds,
    seq,
    ncol,
    mask=None,
    real_length=True,
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
    mv = -9999.0
    dist = np.full(idxs_ds.size, mv, dtype=np.float32 if real_length else np.int32)
    dist[seq] = 0  # initialize valid cells with zero length
    d = 1
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        # sum distances; skip if at pit or mask is True
        if idx0 == idx_ds or (mask is not None and mask[idx0] == True):
            continue
        if real_length:
            d = gis_utils.distance(idx0, idx_ds, ncol, latlon, transform)
        dist[idx0] = dist[idx_ds] + d
    return dist
