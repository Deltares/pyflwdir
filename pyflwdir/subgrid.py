# -*- coding: utf-8 -*-
"""Methods to derive unit catchments from high resolution flow direction data."""

from numba import njit
import numpy as np

from . import core, upscale, arithmetics

_mv = core._mv
__all__ = []


def outlets(idxs_ds, uparea, cellsize, shape, method="eam_plus", mv=_mv):
    """Returns linear indices of unit catchment outlet cells.

    For more information about the methods see upscale script.

    Parameters
    ----------
    idxs_ds : ndarray of int
        linear indices of downstream cells
    uparea : ndarray of float
        flattened upstream area
    cellsize : int
        size of unit catchment measured in no. of high resolution cells
    shape : tuple of int
        raster shape
    method : {"eam_plus", "dmm"}, optional
        method to derive outlet cell indices, by default 'eam_plus'

    Returns
    -------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    """
    # calculate new size
    nrow, ncol = shape
    shape_out = (int(np.ceil(nrow / cellsize)), int(np.ceil(ncol / cellsize)))
    # get outlets cells
    args = (idxs_ds, uparea, shape, shape_out, cellsize)
    if method.lower() == "dmm":
        idxs_out = upscale.dmm_exitcell(*args, mv=mv)
    elif method.lower() == "eam_plus":
        idxs_rep = upscale.eam_repcell(*args, mv=mv)
        idxs_out = upscale.ihu_outlets(idxs_rep, *args, mv=mv)
    else:
        raise ValueError(f'Method {method} unknown, choose from ["eam_plus", "dmm"]')
    return idxs_out, shape_out


@njit
def ucat_area(
    idxs_out,
    idxs_ds,
    seq,
    area,
    mv=_mv,
):
    """Returns the segment catchment map and contributing area.

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_ds : ndarray of int
        linear indices of downstream cells
    seq : 1D array of int
        ordered cell indices from down- to upstream
    area : ndarray of float
        area of each node/cell in a flattened array

    Returns
    -------
    1D array of float of size idxs_ds
        unit catchment map
    1D array of float of size idxs_out
        unit catchment area
    """
    # initialize outputs
    ucatch_map = np.full(idxs_ds.size, 0, dtype=idxs_ds.dtype)
    ucatch_are = np.full(idxs_out.size, -9999, dtype=area.dtype)
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 != mv:
            ucatch_map[idx0] = i + 1
            ucatch_are[i] = area[idx0]
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        ucat_ds = ucatch_map[idx_ds]
        if ucatch_map[idx0] == 0 and ucat_ds != 0:
            ucatch_map[idx0] = ucat_ds
            ucatch_are[ucat_ds - 1] += area[idx0]
    return ucatch_map, ucatch_are


@njit
def ucat_volume(
    idxs_out,
    idxs_ds,
    seq,
    hand,
    area,
    depths=np.arange(0.5, 3.0, 0.5, dtype=np.float32),
    mv=_mv,
):
    """Returns the floodplain volume as function of the depth.

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_ds : ndarray of int
        linear indices of downstream cells
    seq : 1D array of int
        ordered cell indices from down- to upstream
    hand, area : ndarray of float
        height above nearest drain, area of each node/cell in a flattened array

    Returns
    -------
    1D array of float of size idxs_ds
        unit catchment map
    1D array of float of size idxs_out
        unit catchment floodplain profile
    """
    # initialize outputs
    ucatch_map = np.full(idxs_ds.size, 0, dtype=idxs_ds.dtype)
    fldpln_vol = np.full((depths.size, idxs_out.size), -9999, dtype=depths.dtype)
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 != mv:
            ucatch_map[idx0] = i + 1
            dh = np.maximum(0, depths - hand[idx0])
            fldpln_vol[:, i] = area[idx0] * dh
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        ucat_ds = ucatch_map[idx_ds]
        if ucatch_map[idx0] == 0 and ucat_ds != 0:
            ucatch_map[idx0] = ucat_ds
            dh = np.maximum(0, depths - hand[idx0])
            fldpln_vol[:, ucat_ds - 1] += area[idx0] * dh
    return ucatch_map, fldpln_vol


@njit
def segment_length(
    idxs_out,
    idxs_nxt,
    distnc,
    mask=None,
    nodata=-9999.0,
    mv=_mv,
):
    """Returns the channel length which is defined by the path starting at the outlet
    pixel of each cell moving up- or downstream until it reaches the next upstream outlet
    pixel. If moving upstream and a pixel has multiple upstream neighbors, the pixel with
    the largest upstream area is selected.

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_nxt : ndarray of int
        linear indices the next main upstream or downstream cell.
    ncol : int
        number of columns in raster
    mask : ndarray of boolean, optional
        only consider True cells to calculate channel length
    nodata : float, optional
        nodata value, by default -9999.0
    dtype : numpy.dtype, optional
        data output type, by default numpy.float32

    Returns
    -------
    rivlen : 1D array of float
        channel section length [m]
    """
    # temp binary array with outlets
    outlets = np.array([bool(0) for _ in range(idxs_nxt.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = bool(1)
    # allocate output
    rivlen = np.full(idxs_out.size, nodata, dtype=distnc.dtype)
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        x0 = distnc[idx0]
        idx = idx0
        while True:
            idx1 = idxs_nxt[idx]
            if idx1 == mv or idx1 == idx or (mask is not None and mask[idx1] == False):
                idx1 = idx
                break
            # next iter
            idx = idx1
            # break if at up/downstream stream outlet (include!)
            if outlets[idx1]:
                break
        # write channel length
        rivlen[i] = abs(distnc[idx] - x0)
    return rivlen


@njit
def segment_average(
    idxs_out,
    idxs_nxt,
    data,
    weights,
    mask=None,
    nodata=-9999.0,
    mv=_mv,
):
    """Returns the mean value over a river segment. The segment is defined by the flow path starting
    at the outlet pixel of each cell moving up- or downstream until it reaches the next
    upstream outlet pixel.

    Parameters
    ----------
    data : 1D (sparse) array
        values to be averaged
    weights : 1D (sparse) array
        weights
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_nxt : ndarray of int
        linear indices the next main upstream or downstream cell.
    mask : ndarray of boolean, optional
        only consider True cells to calculate channel average value
    nodata : float, optional
        Nodata value which is ignored when calculating the average, by default -9999.0

    Returns
    -------
    data_out : 1D array of float
        segment mean value [m]
    """
    # temp binary array with outlets
    outlets = np.array([bool(0) for _ in range(idxs_nxt.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = bool(1)
    # allocate output
    data_out = np.full(idxs_out.size, nodata, dtype=data.dtype)
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        idxs = [idx0]
        idx = idx0
        while True:
            idx1 = idxs_nxt[idx]
            if (
                idx1 == mv
                or idx1 == idx
                or outlets[idx1]
                or (mask is not None and mask[idx1] == False)
            ):
                break
            idxs.append(idx1)
            # next iter
            idx = idx1
        # get average value
        if len(idxs) > 0:
            idxs_np = np.asarray(idxs, dtype=np.intp)
            data_out[i] = arithmetics._average(data[idxs_np], weights[idxs_np], nodata)
    return data_out


## NOTE: not unit tested
@njit
def segment_median(
    idxs_out,
    idxs_nxt,
    data,
    mask=None,
    nodata=-9999.0,
    mv=_mv,
):
    """Returns the median value along a river segment. The segment is defined by the flow path starting
    at the segment outlet pixel of each cell moving up- or downstream until it reaches the next
    segment outlet pixel.

    Parameters
    ----------
    data : 1D (sparse) array
        values to be averaged
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_nxt : ndarray of int
        linear indices the next main upstream or downstream cell.
    mask : ndarray of boolean, optional
        only consider True cells to calculate channel average value
    nodata : float, optional
        Nodata value which is ignored when calculating the average, by default -9999.0

    Returns
    -------
    data_out : 1D array of float
        segment median value [m]
    """
    # temp binary array with outlets
    outlets = np.array([bool(0) for _ in range(idxs_nxt.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = bool(1)
    # allocate output
    data_out = np.full(idxs_out.size, nodata, dtype=data.dtype)
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        idxs = [idx0]
        idx = idx0
        while True:
            idx1 = idxs_nxt[idx]
            if (
                idx1 == mv
                or idx1 == idx
                or outlets[idx1]
                or (mask is not None and mask[idx1] == False)
            ):
                break
            idxs.append(idx1)
            # next iter
            idx = idx1
        # get median value
        if len(idxs) > 0:
            data_seg = data[np.asarray(idxs)]
            data_out[i] = np.nanmedian(np.where(data_seg == nodata, np.nan, data_seg))
    return data_out


## NOTE: not unit tested
@njit
def segment_indices(
    idxs_out,
    idxs_nxt,
    mask=None,
    max_len=0,
    mv=_mv,
):
    """Returns the linear indices of river segments. The segment is defined by the flow path starting
    at the segment outlet pixel of each cell moving up- or downstream until it reaches the next
    segment outlet pixel.

    Parameters
    ----------
    elevtn, distnc : 1D (sparse) array
        elevation [m], downstream distance to outlet [m]
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_nxt : ndarray of int
        linear indices the next main upstream or downstream cell.
    mask : ndarray of boolean, optional
        only consider True cells to calculate channel average value
    nodata : float, optional
        Nodata value which is ignored when calculating the average, by default -9999.0
    max_len: int, optional
        Maximum length of a single stream segment measured in cells
        Longer streams segments are divided into smaller segments of equal length
        as close as possible to max_len.

    Returns
    -------
    streams : list of 1D-arrays of int
        linear indices of streams
    """
    # temp binary array with outlets
    outlets = np.array([bool(0) for _ in range(idxs_nxt.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = bool(1)
    # allocate output
    streams = []
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        idxs = [idx0]
        idx = idx0
        while True:
            idx1 = idxs_nxt[idx]
            pit = idx1 == idx
            if (
                idx1 == mv
                or pit
                or (mask is not None and mask[idx1] == False)
                or (max_len > 0 and len(idxs) == max_len)
            ):
                break
            idxs.append(idx1)
            if outlets[idx1]:  # include next outlet in stream
                break
            # next iter
            idx = idx1
        # append indices to list of stream segments
        if len(idxs) > 1:
            streams.append(np.array(idxs, dtype=idxs_nxt.dtype))
        # changed in v0.5.2: add zero-length line at pits
        if pit:
            streams.append(np.array([idx1, idx1], dtype=idxs_nxt.dtype))
    return streams


## NOTE: not unit tested
@njit
def segment_slope(
    idxs_out,
    idxs_nxt,
    elevtn,
    distnc,
    mask=None,
    nodata=-9999.0,
    lstsq=True,
    mv=_mv,
):
    """Returns the slope of the river segment segment slope. The segment is defined by the flow path starting
    at the segment outlet pixel of each cell moving up- or downstream until it reaches the next
    segment outlet pixel. The slope is calculated based on a linear fit if `lstsq` equals True,
    else it based on the average slope.

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_nxt : ndarray of int
        linear indices the next main upstream or downstream cell.
    elevtn, distnc : 1D (sparse) array
        elevation [m], downstream distance to outlet [m]
    mask : ndarray of boolean, optional
        only consider True cells to calculate channel average value
    nodata : float, optional
        Nodata value which is ignored when calculating the average, by default -9999.0

    Returns
    -------
    rivslp : 1D array of float
        channel section slope [m]
    """
    # temp binary array with outlets
    outlets = np.array([bool(0) for _ in range(idxs_nxt.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = bool(1)
    # allocate output
    rivslp = np.full(idxs_out.size, nodata, dtype=elevtn.dtype)
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        idxs = [idx0]
        idx = idx0
        while True:
            idx1 = idxs_nxt[idx]
            if (
                idx1 == mv
                or idx1 == idx
                or outlets[idx1]
                or (mask is not None and mask[idx1] is False)
            ):
                break
            idxs.append(idx1)
            # next iter
            idx = idx1
        # get median value
        if len(idxs) > 1:
            if lstsq:
                idxs_np = np.asarray(idxs)
                rivslp[i] = abs(arithmetics.lstsq(distnc[idxs_np], elevtn[idxs_np])[0])
            else:
                dz = elevtn[idxs[0]] - elevtn[idxs[-1]]
                dx = distnc[idxs[0]] - distnc[idxs[-1]]
                rivslp[i] = abs(dz / dx)
        else:
            rivslp[i] = 0.0
    return rivslp


@njit
def fixed_length_slope(
    idxs_out,
    idxs_ds,
    idxs_us_main,
    elevtn,
    distnc,
    length=1e3,
    mask=None,
    lstsq=True,
    mv=_mv,
):
    """Returns the channel slope at the outlet pixel. The slope is based on the elevation values
    within half length distance around from the segment outlet pixel based on least squared error fit.
    The slope is calculated based on a linear fit if `lstsq` equals True, else it based on the average slope.

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_ds, idxs_us_main : array of int
        indices of downstream, main upstream cells
    elevtn, distance : 1D array of float
        flattened 1D elevation [m], downstream distance to outlet [m]
    length : float, optional [m]
        River length over which to calculate the slope. Note that at the up- and downstream
        end of rivers (or where rivers are masked) the slope is calculated over shorter
        lengths. By default set to 1 km.
    ncol : int
        number of columns in raster
    mask : ndarray of boolean, optional
        only consider True cells to calculate channel slope.

    Returns
    -------
    rivslp : 1D array of float
        channel section slope [m/m]
    """
    # allocate output
    rivslp = np.full(idxs_out.size, -9999.0, dtype=np.float32)
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        # move downstream until half length distance
        x0 = distnc[idx0] - length / 2
        x1 = distnc[idx0] + length / 2
        while distnc[idx0] > x0:
            idx_ds = idxs_ds[idx0]
            if idx_ds == idx0 or (mask is not None and mask[idx0] is False):
                break
            idx0 = idx_ds
        # move upstream and collect x & z
        xs = [distnc[idx0]]
        zs = [elevtn[idx0]]
        while distnc[idx0] < x1:
            idx_us = idxs_us_main[idx0]
            if idx_us == mv or (mask is not None and mask[idx_us] is False):
                break
            xs.append(distnc[idx_us])
            zs.append(elevtn[idx_us])
            idx0 = idx_us
        # write lstsq channel slope
        if len(xs) >= 2:
            if lstsq:
                rivslp[i] = abs(arithmetics.lstsq(np.array(xs), np.array(zs))[0])
            else:
                rivslp[i] = abs((zs[0] - zs[-1]) / (xs[0] - xs[-1]))
        else:
            rivslp[i] = 0.0
    return rivslp
