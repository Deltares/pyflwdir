# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
import numpy as np
from pyflwdir import core

# FLOW type
_ftype = 'flow'
_mv = np.int32(-9999)
_pv = np.int32(-9)
# NOTE: data below for consistency with LDD / D8 types and testing
_ds = np.array([
    [ 
        [0, 2, 4],
        [0, _pv, 4],
        [0, 2, 4]
    ],[ 
        [0, 0, 0],
        [2, _pv, 2],
        [4, 4, 4]
    ]
], dtype=np.int32)
_us = np.ones((2,3,3), dtype=np.int32)*2
_us[:,1,1] = _pv
_max_depth = 35

def from_flwdir(flwdir):
    assert isvalid(flwdir) # required to throw meaningfull errors is data is not valid
    nextx, nexty = flwdir # convert [2,:,:] OR ([:,:], [:,:]) to [:,:], [:,:]
    return _from_flwdir(nextx, nexty)

def to_flwdir(idxs_valid, idxs_ds, shape):
    nextx, nexty = _to_flwdir(idxs_valid, idxs_ds, shape)
    return np.stack([nextx, nexty])

@njit("Tuple((u4[:], u4[:], u4[:,:], u4[:]))(i4[:,:], i4[:,:])")
def _from_flwdir(nextx, nexty):
    size = nextx.size
    nrow, ncol = nextx.shape[0], nextx.shape[-1]
    nextx_flat = nextx.ravel()
    nexty_flat = nexty.ravel()
    # keep valid indices only
    idxs_valid = np.where(nextx.ravel()!=_mv)[0].astype(np.uint32)
    n = idxs_valid.size
    idxs_inv = np.ones(size, np.uint32)*core._mv
    idxs_inv[idxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output arrays
    pits_lst = []
    idxs_ds = np.ones(n, dtype=np.uint32)*core._mv
    idxs_us = np.ones((n, _max_depth), dtype=np.uint32)*core._mv
    _max_us = 0
    i = np.uint32(0)
    for i in range(n):
        idx0 = idxs_valid[i]
        c = nextx_flat[idx0]
        r = nexty_flat[idx0]
        if r == _pv or r == _mv:
            # pit // mv
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:
            r, c = r-1, c-1 # convert from to zero-based index
            idx_ds = c + r*ncol
            ids = idxs_inv[idx_ds]
            if r >= nrow or c >= ncol or r<0 or c<0 or ids == core._mv: 
                # ds cell is out of bounds / invalid -> set pit - flag somehow ??
                idxs_ds[i] = i
                pits_lst.append(np.uint32(i))
            else:
                idxs_ds[i] = ids
                for ii in range(_max_depth):
                    if idxs_us[ids,ii] == core._mv:
                        idxs_us[ids,ii] = i
                        break
                if ii >  _max_us:
                    _max_us = ii
                if ii == _max_depth-1:
                    raise ValueError('increase max depth')
    idxs_us = idxs_us[:, :_max_us+1]
    return idxs_valid, idxs_ds, idxs_us, np.array(pits_lst)

@njit("Tuple((i4[:,:], i4[:,:]))(u4[:], u4[:], Tuple((u8, u8)))")
def _to_flwdir(idxs_valid, idxs_ds, shape):
    """convert 1D index to 3D FLOW raster"""
    n = idxs_valid.size
    ncol = shape[1]
    nextx = np.ones(shape, dtype=np.int32).ravel()*_mv
    nexty = np.ones(shape, dtype=np.int32).ravel()*_mv
    for i in range(n):
        idx0 = idxs_valid[i]
        idx_ds = idxs_valid[idxs_ds[i]]
        if i != idx_ds:
            # convert idx_ds to one-based row / col indices
            nextx[idx0] = idx_ds %  ncol + 1
            nexty[idx0] = idx_ds // ncol + 1
        else:
            # pit
            nextx[idx0] = _pv
            nexty[idx0] = _pv
    return nextx.reshape(shape), nexty.reshape(shape)

def isvalid(flwdir):
    """True if NEXTXY raster is valid"""
    if not (
        (isinstance(flwdir, tuple) and len(flwdir) == 2) or
        (isinstance(flwdir, np.ndarray) and flwdir.ndim==3 and flwdir.shape[0]==2)
        ):
        raise TypeError("Flow data type and/or dimensions not understood")
    nextx, nexty = flwdir # should work for [2,:,:] and ([:,:], [:,:])
    mask = np.logical_or(nextx==_mv, nextx==_pv)
    return (nexty.dtype == 'int32' and nextx.dtype == 'int32' and
            np.all(nexty.shape == nextx.shape) and 
            np.all(nextx[~mask]>=0) and
            np.all(nextx[mask] == nexty[mask]))

@vectorize(["b1(i4)", "b1(i8)"])
def ispit(dd):
    """True if FLOW pit"""
    return dd == _pv

@vectorize(["b1(i4)", "b1(i8)"])
def isnodata(dd):
    """True if FLOW nodata"""
    return dd == _mv