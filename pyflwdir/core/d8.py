# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, int64, uint32, uint8, int8, boolean, float32, float64
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
_format = 'd8'

@njit
def pit_indices(flwdir_flat):
    idxs = []
    for idx0 in range(flwdir_flat.size):
        if np.any(flwdir_flat[idx0] == _pits):
            idxs.append(idx0)
    return np.array(idxs, dtype=np.int64)

@njit
def ispit(dd):
    return np.any(dd == _pits)

@njit
def isdir(dd):
    return np.any(dd == _ds)

@njit 
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

@njit
def ds_index(idx0, flwdir_flat, shape):
    """returns numpy array (int64) with indices of donwstream neighbors on a D8 grid.
    At a pit the current index is returned
    
    D8 format
    1:E 2:SE, 4:S, 8:SW, 16:W, 32:NW, 64:N, 128:NE, 0:mouth, -1/255: inland pit, -9/247: undefined (ocean)
    """
    nrow, ncol = shape
    dd = flwdir_flat[idx0]
    r0 = idx0 // ncol
    c0 = idx0 %  ncol
    dr, dc = dd_2_drdc(dd)
    idx = np.int64(-1)
    if not (r0 == 0 and dr == -1) and not (c0 == 0 and dc == -1)\
        and not (r0 == nrow-1 and dr == 1) and not (c0 == ncol-1 and dc == 1):
        idx = idx0 + dc + dr*ncol
    return idx

@njit
def us_indices(idx0, flwdir_flat, shape):
    """returns a numpy array (uint32) with indices of upstream neighbors on a D8 grid
    if it leaves the domain a negative D8 value indicating the side where it leaves the domain is returned"""
    nrow, ncol = shape
    # assume c-style row-major
    r = idx0 // ncol
    c = idx0 %  ncol
    idx = np.int64(-1)
    us_idxs = list()
    for dr in range(-1, 2):
        row = r + dr
        for dc in range(-1, 2):
            col = c + dc
            if dr == 0 and dc == 0: # skip pit -> return empty array
                continue
            elif row >= 0 and row < nrow and col >= 0 and col < ncol: # check bounds
                idx = row*ncol + col
                if flwdir_flat[idx] == _us[dr+1, dc+1]:
                    us_idxs.append(idx)
    return np.array(us_idxs)

@njit
def us_main_indices(idx0, flwdir_flat, uparea_flat, shape, upa_min):
    """returns the index (uint32) of the upstream cell with the largest uparea"""
    nrow, ncol = shape
    # assume c-style row-major
    r = idx0 // ncol
    c = idx0 % ncol
    idx = np.int64(-1)
    us_upa = uparea_flat[idx0]
    idx_lst = list()
    upa_lst = list()
    if us_upa >= upa_min:
        us_upa = upa_min
        for dr in range(-1, 2):
            row = r + dr
            for dc in range(-1, 2):
                col = c + dc
                if dr == 0 and dc == 0: # skip pit -> return empty array
                    continue
                elif row >= 0 and row < nrow and col >= 0 and col < ncol: # check bounds
                    idx = row*ncol + col
                    if flwdir_flat[idx] == _us[dr+1, dc+1]:
                        upa = uparea_flat[idx]
                        if upa >= us_upa:
                            us_upa = upa
                            idx_lst.append(idx)
                            upa_lst.append(upa)
    us_idxs = np.array(idx_lst)
    if us_idxs.size > 1:
        upas = np.array(upa_lst)
        us_idxs = us_idxs[upas>=us_upa]
    return us_idxs, us_upa

@njit
def idx_to_dd(idx0, idx_ds, shape):
    """returns local D8 flow direction based on current and downstream index"""
    ncol = shape[1]
    # size = nrow * ncol
    # assert idx0 < size and idx_ds < size
    r = (idx_ds // ncol) - (idx0 // ncol) + 1
    c = (idx_ds %  ncol) - (idx0 %  ncol) + 1
    dd = _nodata
    if r >= 0 and r < 3 and c >= 0 and c < 3:
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