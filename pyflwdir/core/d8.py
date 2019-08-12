# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, int64, uint32, uint8, int8, boolean, float32
from numba.types import Tuple
import numpy as np
from math import hypot

_ds = np.array([
    [32, 64, 128], 
    [16, 0,  1], 
    [8,  4,  2]], dtype=np.uint8)
_us = np.array([
    [2,   4,  8], 
    [1,   0,  16], 
    [128, 64, 32]], dtype=np.uint8)
_nodata = np.uint8(247)
_pits = np.array([0, 255], dtype=np.uint8)

@njit(uint32[:](uint8[:]))
def pit_indices(flwdir_flat):
    idxs = []
    for idx0 in range(flwdir_flat.size):
        if np.any(flwdir_flat[idx0] == _pits):
            idxs.append(idx0)
    return np.array(idxs, dtype=np.uint32)

@njit(boolean(uint8))
def ispit(dd):
    return np.any(dd == _pits)

@njit(boolean(uint8))
def isdir(dd):
    return np.any(dd == _ds)

@njit(boolean(uint8))
def isnodata(dd):
    return dd == _nodata

@njit(Tuple((int8, int8))(uint8))
def dd_2_drdc(dd):
    dr, dc = np.int8(0), np.int8(0)
    if dd >= np.uint8(16) and dd <= np.uint8(128): 
        if dd == np.uint8(16): #west
            dr, dc = np.int8(0), np.int8(-1)
        else: # north
            dr = np.int8(-1)
            dc = np.int8(np.log2(dd) - 6)
    elif dd < np.uint8(16):
        if dd >= np.uint8(2): # south
            dr = np.int8(1)
            dc = np.int8(-1 * (np.log2(dd) - 2))
        else: # pit / #east
            dr = np.int8(0)
            dc = np.int8(dd)
    return dr, dc

@njit(uint32(uint32, uint8[:], Tuple((int64, int64))))
def ds_index(idx0, flwdir_flat, shape):
    """returns numpy array (uint32) with indices of donwstream neighbors on a D8 grid.
    At a pit the current index is returned
    
    D8 format
    1:E 2:SE, 4:S, 8:SW, 16:W, 32:NW, 64:N, 128:NE, 0:mouth, -1/255: inland pit, -9/247: undefined (ocean)
    """
    nrow, ncol = shape
    dd = flwdir_flat[idx0]
    r0 = idx0 // ncol
    c0 = idx0 %  ncol
    dr, dc = dd_2_drdc(dd)
    if (r0 == 0 and dr == -1) or (c0 == 0 and dc == -1) or (r0 == nrow-1 and dr == 1) or (c0 == ncol-1 and dc == 1):
        idx = np.uint32(-1) # outside domain
    else:
        idx = np.uint32(idx0 + dc + dr*ncol)
    return idx

@njit(uint32[:](uint32, uint8[:], Tuple((int64, int64))))
def us_indices(idx0, flwdir_flat, shape):
    """returns a numpy array (uint32) with indices of upstream neighbors on a D8 grid
    if it leaves the domain a negative D8 value indicating the side where it leaves the domain is returned"""
    nrow, ncol = shape
    # assume c-style row-major
    r = np.uint32(idx0 // ncol)
    c = np.uint32(idx0 %  ncol)
    us_idxs = list()
    for dr in range(-1, 2):
        row = np.uint32(r + dr)
        for dc in range(-1, 2):
            col = np.uint32(c + dc)
            if dr == 0 and dc == 0: # skip pit -> return empty array
                continue
            elif row >= nrow or col >= ncol: # out of bounds
                pass
            else:
                idx = np.uint32(row*ncol + col)
                if flwdir_flat[idx] == _us[dr+1, dc+1]:
                    us_idxs.append(np.uint32(idx))
    return np.array(us_idxs, dtype=np.uint32)

@njit(Tuple((uint32, float32))(uint32, uint8[:], uint8[:], Tuple((int64, int64)), float32))
def us_main_index(idx0, flwdir_flat, uparea_flat, shape, upa_min):
    """returns the index (uint32) of the upstream cell with the largest uparea"""
    nrow, ncol = shape
    # assume c-style row-major
    r = np.uint32(idx0 // ncol)
    c = np.uint32(idx0 % ncol)
    us_idx = np.uint32(-1)
    us_upa = uparea_flat[idx0]
    if us_upa > upa_min:
        us_upa = upa_min
        for dr in range(-1, 2):
            row = np.uint32(r + dr)
            for dc in range(-1, 2):
                col = np.uint32(c + dc)
                if dr == 0 and dc == 0:
                    continue
                elif row >= nrow or col >= ncol: # out of bounds
                    pass
                idx = np.uint32(row*ncol + col)
                if flwdir_flat[idx] == _us[dr+1, dc+1]:
                    upa = uparea_flat[idx]
                    if upa > us_upa:
                        us_idx = idx
                        us_upa = upa
    return us_idx, us_upa

@njit(uint8(uint32, uint32, Tuple((int64, int64)))) 
def idx_to_dd(idx0, idx_ds, shape):
    """returns local D8 flow direction based on current and downstream index"""
    nrow, ncol = shape
    size = nrow * ncol
    assert idx0 < size and idx_ds < size
    r = (idx_ds // ncol) - (idx0 // ncol) + 1
    c = (idx_ds %  ncol) - (idx0 %  ncol) + 1
    if r < 0 or r >= 3 or c < 0 or c >= 3:
        dd = _nodata
    else:
        dd = _ds[r, c]
    return dd

@njit(boolean(uint8[:]))
def _check_format(flwdir_flat):
    dds = np.unique(flwdir_flat)
    check = True
    for i in range(dds.size):
        dd = np.array([dds[i]])
        if np.all(dd != _ds) and np.all(dd != _nodata) and np.all(dd != _pits):
            check = False
            break
    return check