# -*- coding: utf-8 -*-
"""Methods to derive maps of basin/stream characteristics. These methods require
the basin indices to be ordered from down- to upstream."""

from numba import njit
import numpy as np

# import local libraries
from . import gis_utils, core

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
    nup = core.upstream_count(idxs_ds=idxs_ds, mask=mask, mv=mv)
    # get list of indices arrays of segments
    streams = []
    done = np.array([bool(0) for _ in range(idxs_ds.size)])  # all False
    for idx0 in seq[::-1]:  # up- to downstream
        if done[idx0] or (mask is not None and ~mask[idx0]):
            continue
        idxs = [idx0]  # initiate with correct dtype
        while True:
            done[idx0] = True
            idx_ds = idxs_ds[idx0]
            pit = idx_ds == idx0
            if not pit:
                idxs.append(idx_ds)
            if nup[idx_ds] > 1 or pit:
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
                # CHANGED in v0.5.2: add zero length LineString at pits
                if pit:
                    streams.append(np.array([idx_ds, idx_ds], dtype=idxs_ds.dtype))
                break
            idx0 = idx_ds
    return streams


@njit
def stream_order(idxs_ds, seq, idxs_us_main, mask=None, mv=core._mv):
    """Returns the classic or Hack's "bottum up" stream order.

    The main stem, based on upstream area has order 1.
    Each tributary is given a number one greater than that of the
    river or stream into which they discharge.

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
    nup = core.upstream_count(idxs_ds=idxs_ds, mask=mask, mv=mv)
    strord = np.full(idxs_ds.size, 0, dtype=np.uint8)
    for idx0 in seq:  # down- to upstream
        if mask is not None and not mask[idx0]:  # invalid cell
            continue
        idx_ds = idxs_ds[idx0]
        if idx_ds == idx0:  # pit
            strord[idx0] = 1
        elif nup[idx_ds] > 1 and idxs_us_main[idx_ds] != idx0:
            strord[idx0] = strord[idx_ds] + 1
        else:
            strord[idx0] = strord[idx_ds]
    return strord


@njit
def strahler_order(idxs_ds, seq, mask=None):
    """Returns the strahler "top down" stream order.

    Rivers of the first order are the most upstream tributaries or head water cells.
    If two streams of the same order merge, the resulting stream has an order of one higher.
    If two rivers with different stream orders merge, the resulting stream is given the maximum of the two order.

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
    strord = np.full(idxs_ds.size, 0, dtype=np.uint8)
    strmax = np.full(idxs_ds.size, 0, dtype=np.uint8)
    for idx0 in seq[::-1]:  # up- to downstream
        if mask is not None and not mask[idx0]:  # invalid
            continue
        if strord[idx0] == 0:  # headwater cell
            strord[idx0] = 1
        idx_ds = idxs_ds[idx0]
        if idx_ds == idx0:
            continue
        sto, sto_ds = strord[idx0], strord[idx_ds]
        sto_up = strmax[idx_ds]  # max order of other upstream tributaries
        if sto_ds < sto:
            strord[idx_ds] = sto
        # increment order if same order joins
        elif sto == sto_ds and sto_up == sto:
            strord[idx_ds] += 1
        if sto_up < sto:
            strmax[idx_ds] = sto
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
        if idx0 == idx_ds or (mask is not None and mask[idx0]):
            continue
        if real_length:
            d = gis_utils.distance(idx0, idx_ds, ncol, latlon, transform)
        dist[idx0] = dist[idx_ds] + d
    return dist


@njit
def smooth_rivlen(
    idxs_ds,
    idxs_us_main,
    rivlen,
    min_rivlen,
    max_window=10,
    nodata=-9999.0,
    mv=core._mv,
):
    """Return smoothed river length, by taking the window average of river length.
    The window size is increased until the average exceeds the `min_rivlen` threshold
    or the max_window size is reached.

    Parameters
    ----------
    rivlen : 1D array
        River length values.
    min_rivlen : float
        Minimum river length.
    max_window : int
        maximum window size

    Returns
    -------
    1D array of float
        River length values.
    """
    rivlen_out = rivlen.copy()
    n = max_window // 2
    for idx0 in range(rivlen.size):
        len0 = rivlen_out[idx0]
        if len0 != nodata and len0 < min_rivlen:
            len_avg1 = len0
            idxs = core._window(idx0, n, idxs_ds, idxs_us_main, mv=mv)
            # smooth over increasing window until min river length is reached
            for i in range(1, n):
                idxs0 = idxs[n - i : n + i + 1]
                idxs0 = idxs0[idxs0 != mv]
                idxs0 = idxs0[rivlen_out[idxs0] != nodata]
                len_avg0 = np.mean(rivlen_out[idxs0])
                if len_avg0 > len_avg1:
                    idxs1 = idxs0
                    len_avg1 = len_avg0
                # break at smallest window if average > min_rivlen
                if len_avg1 > min_rivlen:
                    break
            # replace lengths in window with average
            if len_avg1 > len0:
                rivlen_out[idxs1] = len_avg1

    return rivlen_out
