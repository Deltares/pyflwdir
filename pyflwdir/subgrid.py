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
def river_length(subidxs_out, 
        idxs_valid, idxs_ds, idxs_us,
        subidxs_valid, subidxs_ds, subidxs_us, subshape,
        subuparea, min_uparea=1., 
        latlon=False, affine=gis_utils.IDENTITY):
    """Returns subgrid river length, calculated by following the subgrid flow direction network from
    one subgrid outlet cell to the next downstream outlet cell. A
    
    Parameters
    ----------
    subidxs_out : ndarray of int
        internal highres indices of subgrid outlet cells
    idxs_valid : ndarray of int
        lowres indices of valid cells
    idxs_ds, idxs_us : ndarray of int
        internal lowres indices of downstream, upstream cells
    subidxs_valid : ndarray of int
        highres raster indices of vaild cells
    subidxs_ds, subidxs_us : ndarray of int
        internal highres indices of downstream, upstream cells
    subshape : tuple of int
        highres raster shape
    subuparea : ndarray
        highres flattened upstream area array
    min_uparea : float
        minimum upstream area of a river cell
    latlon : bool, optional
        True if WGS84 coordinates
        (the default is False)
    affine : affine transform
        Two dimensional affine transform for 2D linear mapping
        (the default is an identity transform which results in an area of 1 for every cell)

    Returns
    -------
    array with subgrid river lengths : ndarray of float
    """
    assert subidxs_out.size == idxs_ds.size # size should match
    subncol = subshape[1]
    # binary array with outlets
    outlets = np.zeros(subidxs_ds.size, dtype=np.int8)
    outlets[subidxs_out] = np.int8(1)
    # uparea at outlets
    uparea = subuparea[subidxs_out]
    # allocate output 
    rivlen = np.ones(idxs_ds.size, dtype=np.float64)*-9999.
    # loop over outlet cell indices
    for idx0 in range(idxs_ds.size):
        # STEP 1 get subgrid index at upstream end of river reach
        idx_us = core._main_upstream(idx0, idxs_us, uparea, upa_min = min_uparea) # internal index
        if idx_us == _mv: # no upstream neighbor with river cells
            # find most upstream river subgrid cell
            subidx1 = subidxs_out[idx0]
            while subidx1 != _mv:
                subidx = subidx1
                subidx1 = core._main_upstream(subidx, subidxs_us, subuparea, upa_min = min_uparea)
        else:
            subidx = subidxs_out[idx_us]
        assert subidx != _mv
        # STEP 2 follow river downstream to get subgrid river length
        l = np.float64(0.)
        while True:
            subidx1, dist = core.downstream_length(subidx, subidxs_ds, subidxs_valid, subncol, latlon=latlon, affine=affine)
            l += dist
            # break if at subgrid outlet or 
            if outlets[subidx1] == np.int8(1) or subidx1 == subidx:
                break
            # next iter
            subidx = subidx1
        # write riv length
        rivlen[idx0] = l

    return rivlen

@njit
def connected(idxs_ds, subidxs_out, subidxs_ds):
    """Returns binary array with ones if sugrid outlet/representative cells are connected in d8.
    
    NOTE all indices are internal!

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
    outlets = np.zeros(subidxs_ds.size, dtype=np.int8)
    outlets[subidxs_out] = np.int8(1)
    # allocate output 
    connect_map = np.ones(idxs_ds.size, dtype=np.int8)*np.int8(-1)
    # loop over outlet cell indices
    for idx0 in range(idxs_ds.size):
        subidx = subidxs_out[idx0]
        idx_ds = idxs_ds[idx0]
        while True:
            subidx1 = subidxs_ds[subidx] # next downstream subgrid cell index
            if outlets[subidx1] == np.int8(1) or subidx1 == subidx: # at outlet or at pit 
                if subidx1 == subidxs_out[idx_ds]:
                    connect_map[idx0] = np.int8(1)
                else: # not connected
                    connect_map[idx0] = np.int8(0)
                break
            # next iter
            subidx = subidx1
    return connect_map