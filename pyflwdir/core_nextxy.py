# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
import numpy as np
from pyflwdir import core

# NEXTXY type
_ftype = 'nextxy'
_mv = np.int32(-9999)
_pv = np.int32(-9)
# NOTE: data below for consistency with LDD / D8 types and testing
_us = np.ones((2,3,3), dtype=np.int32)*2
_us[:,1,1] = _pv

def from_flwdir(flwdir):
    if not (
        (isinstance(flwdir, tuple) and len(flwdir) == 2) or
        (isinstance(flwdir, np.ndarray) and flwdir.ndim==3 and flwdir.shape[0]==2)
        ):
        raise TypeError('NEXTXT flwdir data not understood')
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
    # max number of upstream cells unkonwn -> calculate max depth
    n_up = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        idx0 = idxs_valid[i]    
        c = nextx_flat[idx0]
        r = nexty_flat[idx0]
        if r != _pv and r <= nrow and c <= ncol and r>=1 and c>=1:
            r, c = r-1, c-1 # convert from to zero-based index
            idx_ds = c + r*ncol
            ids = idxs_inv[idx_ds] # internal idx_ds
            if ids == core._mv or ids == i:
                raise ValueError('invalid flwdir data')
            n_up[ids] += 1
    _max_depth = np.int64(np.max(n_up))
    # allocate output arrays
    pits_lst = []
    idxs_ds = np.ones(n, dtype=np.uint32)*core._mv
    idxs_us = np.ones((n, _max_depth), dtype=np.uint32)*core._mv
    i = np.uint32(0)
    for i in range(n):
        idx0 = idxs_valid[i]
        c = nextx_flat[idx0]
        r = nexty_flat[idx0]
        if r == _pv or r > nrow or c > ncol or r<1 or c<1:
            # pit or ds cell is out of bounds / invalid -> set pit
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:
            r, c = r-1, c-1 # convert from to zero-based index
            idx_ds = c + r*ncol
            ids = idxs_inv[idx_ds]
            if ids == core._mv or ids == i:
                raise ValueError('invalid flwdir data')
            idxs_ds[i] = ids
            for ii in range(_max_depth):
                if idxs_us[ids,ii] == core._mv:
                    idxs_us[ids,ii] = i
                    break
    return idxs_valid, idxs_ds, idxs_us, np.array(pits_lst)

@njit("Tuple((i4[:,:], i4[:,:]))(u4[:], u4[:], Tuple((u8, u8)))")
def _to_flwdir(idxs_valid, idxs_ds, shape):
    """convert 1D index to 3D NEXTXY raster"""
    n = idxs_valid.size
    ncol = shape[1]
    nextx = np.ones(shape, dtype=np.int32).ravel()*_mv
    nexty = np.ones(shape, dtype=np.int32).ravel()*_mv
    for i in range(n):
        idx0 = idxs_valid[i]
        idx_ds = idxs_valid[idxs_ds[i]]
        if idx0 != idx_ds:
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
        return False
    nextx, nexty = flwdir # should work for [2,:,:] and ([:,:], [:,:])
    mask = np.logical_or(nextx==_mv, nextx==_pv)
    return (nexty.dtype == 'int32' and nextx.dtype == 'int32' and
            np.all(nexty.shape == nextx.shape) and 
            np.all(nextx[~mask]>=0) and
            np.all(nextx[mask] == nexty[mask]))

@vectorize(["b1(i4)", "b1(i8)"])
def ispit(dd):
    """True if NEXTXY pit"""
    return dd == _pv

@vectorize(["b1(i4)", "b1(i8)"])
def isnodata(dd):
    """True if NEXTXY nodata"""
    return dd == _mv