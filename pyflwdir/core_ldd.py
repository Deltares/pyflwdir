# -*- coding: utf-8 -*-
"""Description of LDD flow direction type and methods to convert to/from general
nextidx."""

from numba import njit, vectorize
import numpy as np
from . import core, core_d8

__all__ = []

# LDD type
_ftype = "ldd"
_ds = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=np.uint8)
_us = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]], dtype=np.uint8)
_mv = np.uint8(255)
_pv = np.uint8(5)
_all = np.array([7, 8, 9, 4, 5, 6, 1, 2, 3, 255], dtype=np.uint8)


from numba import njit
import numpy as np


@njit("Tuple((int8, int8))(uint8)")
def drdc(dd):
    """convert ldd value to delta row/col"""
    dr, dc = np.int8(0), np.int8(0)
    if dd >= np.uint8(4):  # W / PIT / E / NW / N / NE
        if dd >= np.uint8(7):  # NW / N / NE
            dr = np.int8(-1)
            dc = np.int8(dd) - np.int8(8)
        else:  # W / PIT / E
            dr = np.int8(0)
            dc = np.int8(dd) - np.int8(5)
    else:  # SW / S / SE
        dr = np.int8(1)
        dc = np.int8(dd) - np.int8(2)
    return dr, dc


@njit
def from_array(flwdir, _mv=_mv, dtype=np.intp):
    """convert 2D LDD data to 1D next downstream indices"""
    nrow, ncol = flwdir.shape
    flwdir_flat = flwdir.ravel()
    # get downsteam indices
    pits_lst = []
    idxs_ds = np.full(flwdir.size, core._mv, dtype=dtype)
    n = 0
    for idx0 in range(flwdir.size):
        if flwdir_flat[idx0] == _mv:
            continue
        dr, dc = drdc(flwdir_flat[idx0])
        r_ds = int(idx0 // ncol + dr)
        c_ds = int(idx0 % ncol + dc)
        pit = dr == 0 and dc == 0
        outside = r_ds >= nrow or c_ds >= ncol or r_ds < 0 or c_ds < 0
        idx_ds = c_ds + r_ds * ncol
        # pit or outside or ds cell has mv
        if pit or outside or flwdir_flat[idx_ds] == _mv:
            pits_lst.append(idx0)
            idxs_ds[idx0] = idx0
        else:
            idxs_ds[idx0] = idx_ds
        n += 1
    return idxs_ds, np.array(pits_lst, dtype=dtype), n


@njit
def _downstream_idx(idx0, flwdir_flat, shape, mv=core._mv):
    """Returns linear index of the donwstream neighbor; idx0 if at pit"""
    nrow, ncol = shape
    r0 = idx0 // ncol
    c0 = idx0 % ncol
    dr, dc = drdc(flwdir_flat[idx0])
    r_ds, c_ds = r0 + dr, c0 + dc
    if r_ds >= 0 and r_ds < nrow and c_ds >= 0 and c_ds < ncol:  # check bounds
        idx_ds = c_ds + r_ds * ncol
    else:
        idx_ds = mv
    return idx_ds


# general
@njit
def to_array(idxs_ds, shape, mv=core._mv):
    """convert downstream linear indices to dense D8 raster"""
    ncol = shape[1]
    flwdir = np.full(idxs_ds.size, _mv, dtype=np.uint8)
    for idx0 in range(idxs_ds.size):
        idx_ds = idxs_ds[idx0]
        if idx_ds == mv:
            continue
        dr: np.int32 = np.int32(idx_ds // ncol) - np.int32(idx0 // ncol)
        dc: np.int32 = np.int32(idx_ds % ncol) - np.int32(idx0 % ncol)
        if dr >= -1 and dr <= 1 and dc >= -1 and dc <= 1:
            dd = _ds[dr + 1, dc + 1]
        else:
            raise ValueError("Invalid data downstream index outside 8 neighbors.")
        flwdir[idx0] = dd
    return flwdir.reshape(shape)


def isvalid(flwdir, _all=_all):
    """True if 2D LDD raster is valid"""
    return core_d8.isvalid(flwdir, _all)


@njit
def ispit(dd, _pv=_pv):
    """True if LDD pit"""
    return dd == _pv


@njit
def isnodata(dd, _mv=_mv):
    """True if LDD nodata"""
    return core_d8.isnodata(dd, _mv)


@njit
def _upstream_idx(idx0, flwdir_flat, shape, _us=_us, dtype=np.intp):
    """Returns a numpy array (int64) with linear indices of upstream neighbors"""
    return core_d8._upstream_idx(idx0, flwdir_flat, shape, _us, dtype=dtype)
