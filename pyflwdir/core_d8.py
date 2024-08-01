# -*- coding: utf-8 -*-
"""Description of D8 flow direction type and methods to convert to/from general
nextidx."""

from typing import Tuple
import numpy as np
from numba import njit

from . import core

__all__ = []

# D8 type
_ftype = "d8"
_ds = np.array([[32, 64, 128], [16, 0, 1], [8, 4, 2]], dtype=np.uint8)
_us = np.array([[2, 4, 8], [1, 0, 16], [128, 64, 32]], dtype=np.uint8)
_mv = np.uint8(247)
_pv = np.array([0, 255], dtype=np.uint8)
_all = np.array([32, 64, 128, 16, 0, 1, 8, 4, 2, 247, 255], dtype=np.uint8)


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


@njit
def from_array(flwdir, _mv=_mv, dtype=np.intp):
    """convert 2D D8 data to 1D next downstream indices"""
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
        r_ds = int(idx0 // ncol) + int(dr)
        c_ds = int(idx0 % ncol) + int(dc)
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
def to_array(idxs_ds: np.ndarray[np.uint64], shape: Tuple[int, int], mv=core._mv):
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
            dd: np.uint8 = _ds[dr + 1, dc + 1]
        else:
            raise ValueError("Invalid data downstream index outside 8 neighbors.")
        flwdir[idx0] = dd
    return flwdir.reshape(shape)


def isvalid(flwdir: np.uint8, _all: np.ndarray[np.uint8] = _all) -> bool:
    """True if 2D D8 raster is valid"""
    return (
        isinstance(flwdir, np.ndarray)
        and flwdir.dtype == "uint8"
        and flwdir.ndim == 2
        and check_values(flwdir, _all)
    )


@njit
def check_values(flwdir, _all):
    check = True
    for dd in flwdir.ravel():
        if np.all(_all != dd):
            check = False
            break
    return check


@njit
def ispit(dd, _pv=_pv):
    """True if D8 pit"""
    return np.any(dd == _pv)


@njit
def isnodata(dd, _mv=_mv):
    """True if D8 nodata"""
    return dd == _mv


@njit
def _upstream_idx(idx0, flwdir_flat, shape, _us=_us, dtype=np.intp):
    """Returns a numpy array (int64) with linear indices of upstream neighbors"""
    nrow, ncol = shape
    # assume c-style row-major
    r = idx0 // ncol
    c = idx0 % ncol
    idxs_lst = list()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:  # skip pit -> return empty array
                continue
            r_us, c_us = r + dr, c + dc
            if r_us >= 0 and r_us < nrow and c_us >= 0 and c_us < ncol:  # check bounds
                idx = r_us * ncol + c_us
                if flwdir_flat[idx] == _us[dr + 1, dc + 1]:
                    idxs_lst.append(idx)
    return np.array(idxs_lst, dtype=dtype)
