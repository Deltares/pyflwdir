# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, int8, uint8, int64
from numba.types import Tuple
import numpy as np
from math import hypot

_ds = np.array([
    [8,  1,  2], 
    [7,  0,  3], 
    [6,  5,  4]], dtype=np.uint8)
_nodata = np.uint8(247)
_pits = np.array([0, 255], dtype=np.uint8)

@njit(Tuple((int8, int8))(uint8))
def dd_2_drdc(dd):
    if dd == _nodata or np.any(dd==_pits): # inland pit / nodata
        dr, dc = np.int8(0), np.int8(0)
    elif dd >= np.uint8(5): 
        if dd == np.uint8(5): #south
            dr, dc = np.int8(1), np.int8(0)
        else: # west
            dr = np.int8(7-dd)
            dc = np.int8(-1)
    else:
        if dd == np.uint8(1): # north
            dr = np.int8(-1)
            dc = np.int8(0)
        else: #east
            dr = np.int(dd-3)
            dc = np.int8(1)
    return dr, dc

@njit(int64(int64, uint8[:], Tuple((int64, int64))))
def ds_d8(idx0, flwdir_flat, shape):
    """returns numpy array (int32) with indices of donwstream neighbors on a D8trip grid.
    At a pit the current index is returned
    
    D8 TRIP format
    1:N, 2:NE, 3:E, 4:SE, 5:S, 6:SW, 7:W, 8:NW, 0:mouth, -1/255:inland pit, -9/247: undefined (ocean)
    """
    nrow, ncol = shape
    dd = flwdir_flat[idx0]
    r0 = idx0 // ncol
    c0 = idx0 %  ncol
    dr, dc = dd_2_drdc(dd)
    if (r0 == 0 and dr == -1) or (c0 == 0 and dc == -1) or (r0 == nrow-1 and dr == 1) or (c0 == ncol-1 and dc == 1):
        idx = np.int64(-1) # outside domain
    else:
        idx = np.int64(idx0 + dc + dr*ncol)
    return idx