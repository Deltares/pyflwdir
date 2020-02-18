# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
import numpy as np
from pyflwdir import core

# D8 type
_ftype = 'd8'
_ds = np.array([[32, 64, 128], [16, 0, 1], [8, 4, 2]], dtype=np.uint8)
_us = np.array([[2, 4, 8], [1, 0, 16], [128, 64, 32]], dtype=np.uint8)
_mv = np.uint8(247)
_pv = np.array([0, 255], dtype=np.uint8)
_d8_ = np.unique(np.concatenate([_pv, np.array([_mv]), _ds.flatten()]))
_max_depth = 8


@njit("Tuple((int8, int8))(uint8)")
def drdc(dd):
    """convert d8 value to delta row/col"""
    dr, dc = np.int8(0), np.int8(0)
    if dd <= np.uint8(8):  # PIT / E / SW / S / SE
        if dd >= np.uint8(2):  # SW / S / SE
            dr = np.int8(1)
            dc = np.int8(2 - np.log2(dd))
        else:  # PIT / E
            dr = np.int8(0)
            dc = np.int8(dd)
    elif dd <= np.uint8(128):  # W / NW / N / NE
        if dd == np.uint8(16):  # W
            dr, dc = np.int8(0), np.int8(-1)
        else:  # NW / N / NE
            dr = np.int8(-1)
            dc = np.int8(np.log2(dd) - 6)
    return dr, dc


@njit("u1(u4, u4, Tuple((u8,u8)))")
def idx_to_dd(idx0, idx_ds, shape):
    """returns local D8 value based on current and downstream index"""
    ncol = shape[1]
    r = (idx_ds // ncol) - (idx0 // ncol) + 1
    c = (idx_ds % ncol) - (idx0 % ncol) + 1
    dd = _mv
    if r >= 0 and r < 3 and c >= 0 and c < 3:
        dd = _ds[r, c]
    return dd


@njit("Tuple((u4[:], u4[:], u4[:,:], u4[:]))(u1[:,:])")
def from_array(flwdir):
    """convert 2D D8 network to 1D indices"""
    size = flwdir.size
    nrow, ncol = flwdir.shape[0], flwdir.shape[-1]
    flwdir_flat = flwdir.ravel()
    # keep valid indices only
    idxs_valid = []
    idxs_internal = np.ones(size, np.uint32)
    idxs_internal[:] = core._mv
    i = np.uint32(0)
    for idx0 in range(size):
        if flwdir_flat[idx0] != _mv:
            idxs_valid.append(np.uint32(idx0))
            idxs_internal[idx0] = i
            i += 1
    n = np.int64(i)
    # calculate depth per cell & downsteam index
    pits_lst = []
    idxs_ds = np.ones(n, dtype=np.uint32)
    idxs_ds[:] = core._mv
    n_up = np.zeros(n, dtype=np.uint8)
    for idx0 in idxs_valid:
        i = idxs_internal[idx0]
        dd = flwdir_flat[idx0]
        dr, dc = drdc(dd)
        r = int(idx0 // ncol + dr) 
        c = int(idx0  % ncol + dc)
        pit = dr == 0 and dc == 0
        outside = r >= nrow or c >= ncol or r < 0 or c < 0
        # pit or outside
        if pit or outside:
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:        
            idx_ds = idx0 + dc + dr * ncol
            i_ds = idxs_internal[idx_ds]
            if i_ds == core._mv:
                raise ValueError('invalid flwdir data')
            idxs_ds[i] = i_ds
            n_up[i_ds] += 1
    # list with arrays of upstream index
    _max_depth = np.int64(np.max(n_up))
    idxs_us = np.ones((n, _max_depth), dtype=np.uint32)
    idxs_us[:] = core._mv
    n_up[:] = np.uint8(0)
    for i in range(n):
        i_ds = idxs_ds[i]
        if i_ds == i:
            continue
        # valid ds cell
        ii = n_up[i_ds]
        idxs_us[i_ds][ii] = i
        n_up[i_ds] += 1
    return np.array(idxs_valid), idxs_ds, idxs_us, np.array(pits_lst)


@njit("u1[:,:](u4[:], u4[:], Tuple((u8, u8)))")
def to_array(idxs_valid, idxs_ds, shape):
    """convert 1D index to 2D D8 raster"""
    n = idxs_valid.size
    flwdir = np.ones(shape, dtype=np.uint8).ravel() * _mv
    for i in range(n):
        idx0 = idxs_valid[i]
        idx_ds = idxs_valid[idxs_ds[i]]
        dd = idx_to_dd(idx0, idx_ds, shape)
        if dd == _mv:
            msg = 'Invalid data. Downstream cell outside 8 neighbors.'
            raise ValueError(msg)
        flwdir[idx0] = dd
    return flwdir.reshape(shape)


# core d8 functions
def isvalid(flwdir):
    """True if 2D D8 raster is valid"""
    return (isinstance(flwdir, np.ndarray) and 
            flwdir.dtype == 'uint8' and 
            flwdir.ndim == 2 and
            np.all([v in _d8_ for v in np.unique(flwdir)]))


# @vectorize(["b1(u1)"])
@njit
def ispit(dd):
    """True if D8 pit"""
    return np.any(dd == _pv)


# @vectorize(["b1(u1)"])
@njit
def isnodata(dd):
    """True if D8 nodata"""
    return dd == _mv


####################################################
# # NOTE functions below are used in upscale d8 only


@njit
def downstream(idx0, flwdir_flat, shape, dd=_mv):
    """returns numpy array (int64) with indices of donwstream neighbors on a D8 grid.
    At a pit the current index is returned
    
    D8 format
    1:E 2:SE, 4:S, 8:SW, 16:W, 32:NW, 64:N, 128:NE, 0:mouth, -1/255: inland pit, -9/247: undefined (ocean)
    """
    nrow, ncol = shape
    # FIXME: i don't like this extra if statement. can we do this differently?
    if dd == _mv:
        dd = flwdir_flat[idx0]
    r0 = idx0 // ncol
    c0 = idx0 % ncol
    dr, dc = drdc(dd)
    idx = np.int64(-1)
    if not (r0 == 0 and dr == -1) and not (c0 == 0 and dc == -1)\
        and not (r0 == nrow-1 and dr == 1) and not (c0 == ncol-1 and dc == 1):
        idx = idx0 + dc + dr * ncol
    return idx


@njit
def upstream(idx0, flwdir_flat, shape):
    """returns a numpy array (int64) with indices of upstream neighbors on a D8 grid
    if it leaves the domain a negative D8 value indicating the side where it leaves the domain is returned"""
    nrow, ncol = shape
    # assume c-style row-major
    r = idx0 // ncol
    c = idx0 % ncol
    idx = np.int64(-1)
    us_idxs = list()
    for dr in range(-1, 2):
        row = r + dr
        for dc in range(-1, 2):
            col = c + dc
            if dr == 0 and dc == 0:  # skip pit -> return empty array
                continue
            elif row >= 0 and row < nrow and col >= 0 and col < ncol:  # check bounds
                idx = np.int64(row * ncol + col)
                if flwdir_flat[idx] == _us[dr + 1, dc + 1]:
                    us_idxs.append(idx)
    return np.array(us_idxs, dtype=np.int64)
