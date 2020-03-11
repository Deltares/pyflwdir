# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
import numpy as np
from pyflwdir import core

__all__ = []

# NEXTIDX type
#
_ftype = "nextidx"
_mv = np.uint32(-1)  # NOTE same as core _mv
_pv = None  # NOTE a pit is defined by a ds index refering to itself
# data below for consistency with LDD / D8 types and testing
_us = np.ones((3, 3), dtype=np.uint32) * np.uint32(4)


@njit(
    [
        "Tuple((intp[:], u4[:], u4[:]))(u4[:,:])",
        "Tuple((intp[:], u4[:], u4[:]))(u4[:])",
    ]
)
def from_array(nextidx):
    size = nextidx.size
    nrow, ncol = nextidx.shape[0], nextidx.shape[-1]
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
        idx0 = np.intp(idxs_dense[i])
        idx_ds = np.uint32(nextidx_flat[idx0])
        r_ds = idx_ds // ncol
        c_ds = idx_ds % ncol
        pit = idx_ds == idx0
        outside = r_ds >= nrow or c_ds >= ncol or r_ds < 0 or c_ds < 0
        # pit or outside or ds cell has mv
        if pit or outside or idxs_sparse[idx_ds] == core._mv:
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:
            idxs_ds[i] = np.uint32(idxs_sparse[idx_ds])
    return np.array(idxs_dense), idxs_ds, np.array(pits_lst)


@njit  # ("u4[:,:](intp[:], u4[:], Tuple(i8, i8))")
def to_array(idxs_dense, idxs_ds, shape):
    """convert 1D index to 2D NEXTIDX raster"""
    size = shape[0] * shape[1]
    nextidx = np.full(size, _mv, dtype=np.uint32)
    nextidx[idxs_dense] = idxs_dense[idxs_ds]
    return nextidx.reshape(shape)


def isvalid(flwdir):
    """True if NEXTIDX raster is valid"""
    # TODO more checks
    return flwdir.dtype == "uint32"
