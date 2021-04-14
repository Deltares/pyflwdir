# -*- coding: utf-8 -*-
"""Methods to derive unit catchments from high resolution flow direction data."""

from numba import njit
import numpy as np
import math

from pyflwdir import core, gis_utils, upscale
from pyflwdir.arithmetics import _average

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
        size of unit catchment measured in no. of higres cells
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
    ncol,
    latlon=False,
    transform=gis_utils.IDENTITY,
    area_factor=1,
    nodata=-9999.0,
    dtype=np.float64,
    mv=_mv,
):
    """Returns the unit catchment map (highres) and area (lowres) [m2].

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_ds : ndarray of int
        linear indices of downstream cells
    seq : 1D array of int
        ordered cell indices from down- to upstream
    uparea, elevtn : ndarray of float
        flattened upstream area, elevation
    ncol : int
        number of columns in raster
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY
    area_factor : float, optional
        multiplication factor for unit conversion, by default 1
    nodata : float, optional
        nodata value, by default -9999.0
    dtype : numpy.dtype, optional
        data output type, by default numpy.float64

    Returns
    -------
    1D array of float of size idxs_ds
        unit catchment map
    1D array of float of size idxs_out
        unit catchment area
    """
    xres, yres, north = transform[0], transform[4], transform[5]
    area0 = abs(xres * yres)
    # initialize outputs
    ucatch_map = np.full(idxs_ds.size, 0, dtype=idxs_ds.dtype)
    ucatch_are = np.full(idxs_out.size, nodata, dtype=dtype)
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 != mv:
            ucatch_map[idx0] = i + 1
            if latlon:
                lat = north + (idx0 // ncol + 0.5) * yres
                area0 = gis_utils.cellarea(lat, xres, yres) / area_factor
            ucatch_are[i] = area0
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        ucat0 = ucatch_map[idx0]
        ucat_ds = ucatch_map[idx_ds]
        if ucat0 == 0 and ucat_ds != 0:
            if latlon:
                lat = north + (idx0 // ncol + 0.5) * yres
                area0 = gis_utils.cellarea(lat, xres, yres) / area_factor
            ucatch_map[idx0] = ucat_ds
            ucatch_are[ucat_ds - 1] += area0
    return ucatch_map, ucatch_are


@njit
def channel_length(
    idxs_out,
    idxs_nxt,
    ncol,
    mask=None,
    latlon=False,
    transform=gis_utils.IDENTITY,
    nodata=-9999.0,
    dtype=np.float32,
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
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY
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
    outlets = np.array([np.bool(0) for _ in range(idxs_nxt.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = np.bool(1)
    # allocate output
    rivlen = np.full(idxs_out.size, nodata, dtype=dtype)
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        l = np.float64(0.0)
        idx = idx0
        while True:
            idx1 = idxs_nxt[idx]
            if idx1 == mv or idx1 == idx or (mask is not None and mask[idx1] == False):
                idx1 = idx
                break
            # update length
            l += gis_utils.distance(idx, idx1, ncol, latlon, transform)
            # break if at up/downstream stream outlet
            if outlets[idx1]:
                break
            # next iter
            idx = idx1
        # write channel length
        rivlen[i] = l
    return rivlen


@njit
def channel_average(
    idxs_out,
    idxs_nxt,
    data,
    weights,
    mask=None,
    nodata=-9999.0,
    mv=_mv,
):
    """Returns the mean channel value. The channel is defined by the flow path starting
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
    rivlen : 1D array of float
        channel section length [m]
    """
    # temp binary array with outlets
    outlets = np.array([np.bool(0) for _ in range(idxs_nxt.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = np.bool(1)
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
            data_out[i] = _average(data[idxs_np], weights[idxs_np], nodata)
    return data_out


@njit
def channel_slope(
    idxs_out,
    idxs_ds,
    idxs_us_main,
    elevtn,
    ncol,
    length=1e3,
    mask=None,
    latlon=False,
    transform=gis_utils.IDENTITY,
    mv=_mv,
):
    """Returns the channel slope at the outlet pixel. The slope is estimated
    from the elevation difference between length/2 downstream and lenght/2 upstream
    of the outlet pixel.

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_ds, idxs_us_main : array of int
        indices of downstream, main upstream cells
    elevtn : ndarray of float, optional
        flattened 1D elevation [m]
    length : float, optional [m]
        River length over which to calculate the slope. Note that at the up- and downstream
        end of rivers (or where rivers are masked) the slope is calculated over shorter
        lengths. By default set to 1 km.
    ncol : int
        number of columns in raster
    mask : ndarray of boolean, optional
        only consider True cells to calculate channel slope.
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY

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
        l0, l1 = np.float64(0.0), np.float64(0.0)
        idx1 = idx0
        # move upstream on subgrid to set idx1 (upstream end)
        while l1 < length / 2:
            idx_us = idxs_us_main[idx1]
            if idx_us == mv or (mask is not None and mask[idx_us] == False):
                break
            l1 += gis_utils.distance(idx1, idx_us, ncol, latlon, transform)
            # next iter
            idx1 = idx_us
        # move downstream on subgrid to set idx0 (downsteam end)
        while l0 < length / 2 and (l0 + l1) < length:
            idx_ds = idxs_ds[idx0]
            if idx_ds == idx0 or (mask is not None and mask[idx0] == False):
                break
            l0 += gis_utils.distance(idx0, idx_ds, ncol, latlon, transform)
            # next iter
            idx0 = idx_ds
        # write mean channel slope
        l = l0 + l1
        z0 = elevtn[idx0]
        z1 = elevtn[idx1]
        rivslp[i] = 0.0 if l == 0 else abs(z1 - z0) / l
    return rivslp


# TODO remove in v0.5
@njit
def channel(
    idxs_out,
    idxs_nxt,
    idxs_prev,
    elevtn,
    rivwth,
    uparea,
    ncol,
    upa_min=0.0,
    len_min=0.0,
    latlon=False,
    transform=gis_utils.IDENTITY,
    mv=_mv,
):
    """Returns the channel length and slope per channel segment which is defined by the
    path starting at the outlet cell moving upstream following the upstream cells with
    the largest upstream area until it reaches the next upstream outlet cell.

    A mimumum upstream area threshold <upa_min> can be set to define subgrid channel
    cells.

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_nxt, idxs_prev : ndarray of int
        linear indices of next and previous cells, if moving upstream next is the main
        upstream cell index, else the next downstream cell index and vice versa.
    uparea, elevtn, rivwth : ndarray of float, optional
        flattened upstream area [km2], elevation [m], river width [m]
    ncol : int
        number of columns in raster
    upa_min : float, optional
        minimum upstream area threshold [km2], requires uparea
    len_min : float, optional
        minimum river length threshold [m] to caculate a slope, if the river is shorter
        it is extended in both directions until this requirement is met.
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY

    Returns
    -------
    rivlen1 : 1D array of float
        channel section length [m]
    rivslp1 : 1D array of float
        channel section slope [m/m]
    rivwth1 : 1D array of float
        mean channel section width [m]
    """
    # temp binary array with outlets
    outlets = np.array([np.bool(0) for _ in range(idxs_nxt.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = np.bool(1)
    # allocate output
    rivlen1 = np.full(idxs_out.size, -9999.0, dtype=np.float32)
    rivslp1 = np.full(idxs_out.size, -9999.0, dtype=np.float32)
    rivwth1 = np.full(idxs_out.size, -9999.0, dtype=np.float32)
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        l = np.float64(0.0)
        # mean width; including starting outlet; excluding final outlet
        w = np.float64(0.0)
        n = 0
        if rivwth is not None and rivwth[idx0] > 0:
            w += np.float64(rivwth[idx0])
            n += 1
        idx = idx0
        while True:
            idx1 = idxs_nxt[idx]
            if (
                idx1 == mv
                or idx1 == idx
                or (uparea is not None and uparea[idx1] < upa_min)
            ):
                idx1 = idx
                break
            # update length
            l += gis_utils.distance(idx, idx1, ncol, latlon, transform)
            # break if at up/downstream stream outlet
            if outlets[idx1]:
                break
            if rivwth is not None and rivwth[idx1] > 0:  # use only valid values
                w += rivwth[idx1]
                n += 1
            # next iter
            idx = idx1
        # write channel length
        rivlen1[i] = l
        # arithmetic mean channel width
        if rivwth is not None:
            rivwth1[i] = 0 if n == 0 else w / n
        # channel slope
        if elevtn is not None:
            # extend reach if shorter than len_min to caculate slope
            while l < len_min:
                # extend in nxt direction
                idx = idx1
                idx1 = idxs_nxt[idx]
                if (
                    idx1 == mv
                    or idx1 == idx
                    or (uparea is not None and uparea[idx1] < upa_min)
                ):
                    idx1 = idx
                if idx1 != idx:
                    l += gis_utils.distance(idx, idx1, ncol, latlon, transform)
                # break if min length reached
                if l >= len_min:
                    break
                # extend in prev direction
                _idx = idx0
                idx0 = idxs_prev[_idx]
                if (
                    idx0 == mv
                    or idx0 == _idx
                    or (uparea is not None and uparea[idx0] < upa_min)
                ):
                    idx0 = _idx
                if idx0 != _idx:
                    l += gis_utils.distance(_idx, idx0, ncol, latlon, transform)
                # break if no more up or downstream cells
                if idx == idx1 and _idx == idx0:
                    break
            # write absolute mean channel slope
            z0 = elevtn[idx0]
            z1 = elevtn[idx1]
            rivslp1[i] = 0.0 if l == 0 else abs(z1 - z0) / l
    return rivlen1, rivslp1, rivwth1
