# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
import numpy as np
from pyflwdir import core

# NEXTIDX type
#
_ftype = 'nextidx'
_mv = np.uint32(-1)  # NOTE same as core _mv
_pv = None  # NOTE a pit is defined by a ds index refering to itself
# data below for consistency with LDD / D8 types and testing
_us = np.ones((3, 3), dtype=np.uint32) * np.uint32(4)


@njit([
    "Tuple((intp[:], u4[:], u4[:]))(u4[:,:])",
    "Tuple((intp[:], u4[:], u4[:]))(u4[:])",
])
def from_array(nextidx):
    size = nextidx.size
    nextidx_flat = nextidx.ravel()
    # find valid dense indices
    idxs_dense = []
    idxs_sparse = np.full(size, core._mv, dtype=np.uint32)
    i = np.uint32(0)
    for idx0 in range(size):
        if nextidx_flat[idx0] != _mv:
            idxs_dense.append(np.intp(idx0))
            idxs_sparse[idx0] = i
            i += 1
    n = i    
    # allocate output arrays
    pits_lst = []
    idxs_ds = np.full(n, core._mv, dtype=np.uint32)
    i = np.uint32(0)
    for i in range(n):
        idx0 = idxs_dense[i]
        idx_ds = nextidx_flat[idx0]
        if idx_ds < 0 or idx_ds >= size or idx_ds == idx0:
            # ds cell is out of bounds / invalid or pit -> set pit
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:
            i_ds = idxs_sparse[idx_ds] 
            if i_ds == core._mv or i_ds == i:
                raise ValueError('invalid NEXTIDX data')
            idxs_ds[i] = i_ds
    return np.array(idxs_dense), idxs_ds, np.array(pits_lst)

@njit  #("u4[:,:](intp[:], u4[:], Tuple(i8, i8))")
def to_array(idxs_dense, idxs_ds, shape):
    """convert 1D index to 2D NEXTIDX raster"""
    size = shape[0]*shape[1]
    nextidx = np.full(size, _mv, dtype=np.uint32)
    nextidx[idxs_dense] = idxs_dense[idxs_ds]
    return nextidx.reshape(shape)


def isvalid(flwdir):
    """True if NEXTIDX raster is valid"""
    # TODO more checks
    return (flwdir.dtype == 'uint32')
