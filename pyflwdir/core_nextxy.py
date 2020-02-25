# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
import numpy as np
from pyflwdir import core

# NEXTXY type
_ftype = "nextxy"
_mv = np.int32(-9999)
_pv = np.int32(-9)
# NOTE: data below for consistency with LDD / D8 types and testing
_us = np.ones((2, 3, 3), dtype=np.int32) * 2
_us[:, 1, 1] = _pv


def from_array(flwdir):
    if not (
        (isinstance(flwdir, tuple) and len(flwdir) == 2)
        or (
            isinstance(flwdir, np.ndarray) and flwdir.ndim == 3 and flwdir.shape[0] == 2
        )
    ):
        raise TypeError("NEXTXY flwdir data not understood")
    nextx, nexty = flwdir  # convert [2,:,:] OR ([:,:], [:,:]) to [:,:], [:,:]
    return _from_array(nextx, nexty)


def to_array(idxs_dense, idxs_ds, shape):
    nextx, nexty = _to_array(idxs_dense, idxs_ds, shape)
    return np.stack([nextx, nexty])


@njit("Tuple((intp[:], u4[:], u4[:]))(i4[:,:], i4[:,:])")
def _from_array(nextx, nexty):
    size = nextx.size
    nrow, ncol = nextx.shape[0], nextx.shape[-1]
    nextx_flat = nextx.ravel()
    nexty_flat = nexty.ravel()
    # find valid dense indices
    idxs_dense = []
    idxs_sparse = np.full(size, core._mv, dtype=np.uint32)
    i = np.uint32(0)
    for idx0 in range(size):
        if nextx_flat[idx0] != _mv and nexty_flat[idx0] != _mv:
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
        c = np.uint32(nextx_flat[idx0])
        r = np.uint32(nexty_flat[idx0])
        if r == _pv or r > nrow or c > ncol or r < 1 or c < 1:
            # pit or ds cell is out of bounds / invalid -> set pit
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:
            r, c = r - 1, c - 1  # convert from to zero-based index
            idx_ds = c + r * ncol
            i_ds = idxs_sparse[idx_ds]
            if i_ds == core._mv or i_ds == i:
                raise ValueError("invalid NEXTXY data")
            idxs_ds[i] = i_ds
    return np.array(idxs_dense), idxs_ds, np.array(pits_lst)


@njit  # ("Tuple((i4[:,:], i4[:,:]))(intp[:], u4[:], Tuple((u8, u8)))")
def _to_array(idxs_dense, idxs_ds, shape):
    """convert 1D index to 3D NEXTXY raster"""
    n = idxs_dense.size
    ncol = shape[1]
    size = shape[0] * shape[1]
    nextx = np.full(size, _mv, dtype=np.int32)
    nexty = np.full(size, _mv, dtype=np.int32)
    for i in range(n):
        idx0 = idxs_dense[i]
        idx_ds = idxs_dense[idxs_ds[i]]
        if idx0 != idx_ds:
            # convert idx_ds to one-based row / col indices
            nextx[idx0] = idx_ds % ncol + 1
            nexty[idx0] = idx_ds // ncol + 1
        else:
            # pit
            nextx[idx0] = _pv
            nexty[idx0] = _pv
    return nextx.reshape(shape), nexty.reshape(shape)


def isvalid(flwdir):
    """True if NEXTXY raster is valid"""
    isfmt1 = isinstance(flwdir, tuple) and len(flwdir) == 2
    isfmt2 = (
        isinstance(flwdir, np.ndarray) and flwdir.ndim == 3 and flwdir.shape[0] == 2
    )
    if not (isfmt1 or isfmt2):
        return False
    nextx, nexty = flwdir  # should work for [2,:,:] and ([:,:], [:,:])
    mask = np.logical_or(nextx == _mv, nextx == _pv)
    return (
        nexty.dtype == "int32"
        and nextx.dtype == "int32"
        and np.all(nexty.shape == nextx.shape)
        and np.all(nextx[~mask] >= 0)
        and np.all(nextx[mask] == nexty[mask])
    )


# @vectorize(["b1(i4)", "b1(i8)"])
@njit
def ispit(dd):
    """True if NEXTXY pit"""
    return dd == _pv


# @vectorize(["b1(i4)", "b1(i8)"])
@njit
def isnodata(dd):
    """True if NEXTXY nodata"""
    return dd == _mv
