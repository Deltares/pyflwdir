# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# January 2020
""""""

from numba import njit
import numpy as np
import math

from pyflwdir import core, core_nextidx, gis_utils, upscale
_mv = core._mv


@njit
def river_params(subidxs_out,
                 subidxs_valid,
                 subidxs_ds,
                 subidxs_us,
                 subuparea,
                 subelevtn,
                 subshape,
                 min_uparea=1.,
                 latlon=False,
                 affine=gis_utils.IDENTITY):
    """Returns the subgrid river length and slope per lowres cell. The 
    subgrid river is defined by the path starting at the subgrid outlet 
    cell moving upstream following the upstream subgrid cells with the 
    largest upstream area until it reaches the next upstream outlet cell. 
    
    A mimumum upstream area can be set to discriminate river cells.

    Parameters
    ----------
    subidxs_out : ndarray of int
        internal highres indices of subgrid outlet cells
    subidxs_valid : ndarray of int
        highres raster indices of vaild cells
    subidxs_ds, subidxs_us : ndarray of int
        internal highres indices of downstream, upstream cells
    subuparea : ndarray of float
        highres flattened upstream area
    subelevtn : ndarray of float
        highres flattened elevation
    subshape : tuple of int
        highres raster shape
    min_uparea : float
        minimum upstream area of a river cell
    latlon : bool, optional
        True if WGS84 coordinates
        (the default is False)
    affine : affine transform
        Two dimensional transform for 2D linear mapping
        (the default is an identity transform which results 
        in an area of 1 for every cell)

    Returns
    -------
    1D array of float
        subgrid river length [m]
    1D array of float
        subgrid river slope [m/m] 
    """
    subncol = subshape[1]
    # binary array with outlets
    subn = subidxs_ds.size
    outlets = np.array([np.bool(0) for _ in range(subn)])
    outlets[subidxs_out] = np.bool(1)
    # allocate output
    n = subidxs_out.size
    rivlen = np.ones(n, dtype=np.float64) * -9999.
    rivslp = np.ones(n, dtype=np.float64) * -9999.
    # loop over outlet cell indices
    for idx0 in range(n):
        subidx = subidxs_out[idx0]
        z0 = subelevtn[subidx]
        l = np.float64(0.)
        while True:
            subidx1 = core._main_upstream(subidx, subidxs_us, subuparea,
                                          min_uparea)
            # break if no more upstream cells
            if subidx1 == _mv:
                z1 = subelevtn[subidx]
                break
            # update length
            l += core.downstream_length(subidx1, subidxs_ds, subidxs_valid,
                                        subncol, latlon, affine)[1]
            # break if at upstream subgrid outlet
            if outlets[subidx1]:
                z1 = subelevtn[subidx1]
                break
            # next iter
            subidx = subidx1
        # write riv length
        rivlen[idx0] = l
        rivslp[idx0] = 0. if l == 0 else (z1 - z0) / l
    return rivlen, rivslp


def cell_area(subidxs_out,
              subidxs_valid,
              subidxs_us,
              subshape,
              latlon=False,
              affine=gis_utils.IDENTITY):
    """Returns the subgrid cell area. 
    
    Parameters
    ----------
    subidxs_out : ndarray of int
        internal highres indices of subgrid outlet cells
    subidxs_valid : ndarray of int
        highres raster indices of vaild cells
    subidxs_us : ndarray of int
        internal highres indices of upstream cells
    subshape : tuple of int
        highres raster shape
    latlon : bool, optional
        True if WGS84 coordinates
        (the default is False)
    affine : affine transform
        Two dimensional transform for 2D linear mapping
        (the default is an identity transform which results 
        in an area of 1 for every cell)

    Returns
    -------
    1D array of float
        subgrid cell area
    """
    subncol = subshape[1]
    xres, yres, north = affine[0], affine[4], affine[5]
    area0 = abs(xres * yres)
    # binary array with outlets
    subn = subidxs_valid.size
    outlets = np.array([np.bool(0) for _ in range(subn)])
    outlets[subidxs_out] = np.bool(1)
    # allocate output
    n = subidxs_out.size
    subare = np.ones(n, dtype=np.float64) * -9999.
    # loop over outlet cell indices
    for idx0 in range(n):
        area = np.float64(0)
        subidxs = np.array([subidxs_out[idx0]])
        while True:
            next_lst = []
            for subidx in subidxs_us[subidxs, :].ravel():
                if subidx == _mv or outlets[subidx]:
                    continue
                next_lst.append(subidx)
                if latlon:
                    r = subidxs_valid[subidx] // subncol
                    lat = north + (r + 0.5) * yres
                    area0 = gis_utils.cellarea(lat, xres, yres)
                area += area0
            # next iter
            if len(next_lst) == 0:
                break
            subidxs = np.array(next_lst)
        subare[idx0] = area
    return subare


@njit
def connected(subidxs_out, idxs_ds, subidxs_ds):
    """Returns binary array with ones if sugrid outlet/representative 
    cells are connected in d8.
    
    Parameters
    ----------
    idxs_ds : ndarray of int
        internal lowres indices of next downstream cell
    subidxs_out, subidxs_ds : ndarray of int
        internal highres indices of outlet, next downstream cells

    Returns
    -------
    array with ones where connected : ndarray of int
    """
    assert subidxs_out.size == idxs_ds.size
    # binary array with outlets
    subn = subidxs_ds.size
    outlets = np.array([np.bool(0) for _ in range(subn)])
    outlets[subidxs_out] = True
    # allocate output. intialize 'True' map
    n = idxs_ds.size
    connect_map = np.array([np.bool(1) for _ in range(n)])
    # loop over outlet cell indices
    for idx0 in range(n):
        subidx = subidxs_out[idx0]
        idx_ds = idxs_ds[idx0]
        while True:
            subidx1 = subidxs_ds[subidx]  # next downstream subgrid cell index
            if outlets[subidx1] or subidx1 == subidx:  # at outlet or at pit
                if subidx1 != subidxs_out[idx_ds]:
                    connect_map[idx0] = False
                break
            # next iter
            subidx = subidx1
    return connect_map
