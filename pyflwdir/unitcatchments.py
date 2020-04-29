# -*- coding: utf-8 -*-
"""Methods to derive unit catchments from high resolution flow direction data."""

from numba import njit
import numpy as np
import math

from pyflwdir import core, gis_utils, upscale

_mv = core._mv
__all__ = []


def outlets(idxs_ds, uparea, cellsize, shape, method="eam", mv=_mv):
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
    method : {"eam", "dmm"}, optional
        method to derive outlet cell indices, by default 'eam'
    
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
    elif method.lower() == "eam":
        idxs_rep = upscale.eam_repcell(*args, mv=mv)
        idxs_out = upscale.com_outlets(idxs_rep, *args, mv=mv)
    else:
        raise ValueError(f'Method {method} unknown, choose from ["eam", "dmm"]')
    return idxs_out, shape_out


@njit
def area(
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
def channel(
    idxs_out,
    idxs_ds,
    uparea,
    elevtn,
    ncol,
    upa_min=0.0,
    latlon=False,
    transform=gis_utils.IDENTITY,
    mv=_mv,
):
    """Returns the channel length and slope per channel segment which is defined by the 
    path starting at the outlet cell moving upstream following the upstream cells with 
    the largest upstream area until it reaches the next upstream outlet cell. 
    
    A mimumum upstream area can be set to discriminate channel cells.

    Parameters
    ----------
    idxs_out : ndarray of int
        linear indices of unit catchment outlet cells
    idxs_ds : ndarray of int
        linear indices of downstream cells
    uparea, elevtn : ndarray of float
        flattened upstream area, elevation
    ncol : int
        number of columns in raster
    upa_min : float, optional 
        minimum upstream area threshold
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY

    Returns
    -------
    1D array of float
        channel section length [m]
    1D array of float
        channel section slope [m/m] 
    """
    # get indices of main upstream cells
    idxs_us_main = core.main_upstream(idxs_ds, uparea, upa_min=upa_min, mv=mv)
    # temp binary array with outlets
    outlets = np.array([np.bool(0) for _ in range(idxs_ds.size)])
    for idx0 in idxs_out:
        if idx0 != mv:
            outlets[idx0] = np.bool(1)
    # allocate output
    rivlen = np.full(idxs_out.size, -9999.0, dtype=np.float64)
    rivslp = np.full(idxs_out.size, -9999.0, dtype=np.float64)
    # loop over outlet cell indices
    for i in range(idxs_out.size):
        idx0 = idxs_out[i]
        if idx0 == mv:
            continue
        l = np.float64(0.0)
        idx = idx0
        while True:
            idx1 = idxs_us_main[idx]
            if idx1 == mv:
                idx1 = idx
                break
            # update length
            l += gis_utils.distance(idx, idx1, ncol, latlon, transform)
            # break if at upstream outlet
            if outlets[idx1]:
                break
            # next iter
            idx = idx1
        # write channel length
        rivlen[i] = l
        # write channel slope
        z0 = elevtn[idx0]
        z1 = elevtn[idx1]
        rivslp[i] = 0.0 if l == 0 else (z1 - z0) / l
    return rivlen, rivslp
