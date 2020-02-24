# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
import numpy as np
from pyflwdir import core

# LDD type
_ftype = 'ldd'
_ds = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=np.uint8)
_us = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]], dtype=np.uint8)
_mv = np.uint8(255)
_pv = np.uint8(5)
_ldd_ = np.unique(np.concatenate([[_pv], [_mv], _ds.flatten()]))
_max_depth = 8


@njit("Tuple((int8, int8))(uint8)")
def drdc(dd):
    """convert ldd value to delta row/col"""
    dr, dc = np.int8(0), np.int8(0)
    if dd >= np.uint8(4):  # W / PIT / E / NW / N / NE
        if dd >= np.uint(7):  # NW / N / NE
            dr = np.int8(-1)
            dc = np.int8(dd - 8)
        else:  # W / PIT / E
            dr = np.int8(0)
            dc = np.int8(dd - 5)
    else:  # SW / S / SE
        dr = np.int8(1)
        dc = np.int8(dd - 2)
    return dr, dc


@njit("u1(u4, u4, Tuple((u8,u8)))")
def idx_to_dd(idx0, idx_ds, shape):
    """returns local LDD value based on current and downstream index"""
    ncol = shape[1]
    r = (idx_ds // ncol) - (idx0 // ncol) + 1
    c = (idx_ds % ncol) - (idx0 % ncol) + 1
    dd = _mv
    if r >= 0 and r < 3 and c >= 0 and c < 3:
        dd = _ds[r, c]
    return dd


@njit("Tuple((intp[:], u4[:], u4[:]))(u1[:,:])")
def from_array(flwdir):
    """convert 2D LDD network to 1D indices"""
    size = flwdir.size
    nrow, ncol = flwdir.shape[0], flwdir.shape[-1]
    flwdir_flat = flwdir.ravel()
    # find valid dense indices
    idxs_dense = []
    idxs_sparse = np.full(size, core._mv, dtype=np.uint32)
    i = np.uint32(0)
    for idx0 in range(size):
        if flwdir_flat[idx0] != _mv:
            idxs_dense.append(np.intp(idx0))
            idxs_sparse[idx0] = i
            i += 1
    n = i
    # get downsteam sparse index
    pits_lst = []
    idxs_ds = np.full(n, core._mv, dtype=np.uint32)
    # loop over sparse indices
    i = np.uint32(0)
    for idx0 in idxs_dense:
        i = idxs_sparse[idx0]
        dr, dc = drdc(flwdir_flat[idx0]) # NOTE only difference to D8
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
            i_ds = idxs_sparse[idx_ds]
            if i_ds == core._mv:
                raise ValueError('invalid LDD data')
            idxs_ds[i] = i_ds
    return np.array(idxs_dense), idxs_ds, np.array(pits_lst)


@njit#("u1[:,:](intp[:], u4[:], Tuple((u8, u8)))")
def to_array(idxs_dense, idxs_ds, shape):
    """convert sparse downstream indices to dense LDD raster"""
    n = idxs_dense.size
    flwdir = np.full(shape, _mv, dtype=np.uint8).ravel()
    for i in range(n):
        idx0 = idxs_dense[i]
        idx_ds = idxs_dense[idxs_ds[i]]
        dd = idx_to_dd(idx0, idx_ds, shape) # NOTE only difference to D8
        if dd == _mv:
            msg = 'Invalid data. Downstream cell outside 8 neighbors.'
            raise ValueError(msg)
        flwdir[idx0] = dd
    return flwdir.reshape(shape)


def isvalid(flwdir):
    """True if 2D LDD raster is valid"""
    return (isinstance(flwdir, np.ndarray) and flwdir.dtype == 'uint8'
            and flwdir.ndim == 2
            and np.all([v in _ldd_ for v in np.unique(flwdir)]))


# @vectorize(["b1(u1)"])
@njit
def ispit(dd):
    """True if LDD pit"""
    return dd == _pv


# @vectorize(["b1(u1)"])
@njit
def isnodata(dd):
    """True if LDD nodata"""
    return dd == _mv
