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
    "Tuple((u4[:], u4[:], u4[:,:], u4[:]))(u4[:,:])",
    "Tuple((u4[:], u4[:], u4[:,:], u4[:]))(u4[:])",
])
def from_array(nextidx):
    size = nextidx.size
    nextidx_flat = nextidx.ravel()
    # keep valid indices only
    idxs_valid = np.where(nextidx_flat != _mv)[0].astype(np.uint32)
    n = idxs_valid.size
    idxs_inv = np.ones(size, np.uint32) * core._mv
    idxs_inv[idxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # max number of upstream cells unkonwn -> calculate max depth
    n_up = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        idx0 = idxs_valid[i]
        idx_ds = nextidx_flat[idx0]
        if idx_ds >= 0 and idx_ds < size and idx_ds != idx0:
            ids = idxs_inv[idx_ds]  # internal idx_ds
            if ids == core._mv or ids == i:
                raise ValueError('invalid flwdir data')
            n_up[ids] += 1
    _max_depth = np.int64(np.max(n_up))
    # allocate output arrays
    pits_lst = []
    idxs_ds = np.ones(n, dtype=np.uint32) * core._mv
    idxs_us = np.ones((n, _max_depth), dtype=np.uint32) * core._mv
    i = np.uint32(0)
    for i in range(n):
        idx0 = idxs_valid[i]
        idx_ds = nextidx_flat[idx0]
        if idx_ds < 0 or idx_ds >= size or idx_ds == idx0:
            # ds cell is out of bounds / invalid or pit -> set pit
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:
            ids = idxs_inv[idx_ds]  # internal idx_ds
            idxs_ds[i] = ids
            for ii in range(_max_depth):
                if idxs_us[ids, ii] == core._mv:
                    idxs_us[ids, ii] = i
                    break
    return idxs_valid, idxs_ds, idxs_us, np.array(pits_lst)


@njit  #("u4[:,:]( u4[:], u4[:], Tuple(i8, i8) )")
def to_array(idxs_valid, idxs_ds, shape):
    """convert 1D index to 2D NEXTIDX raster"""
    nextidx = np.ones(shape, dtype=np.uint32).ravel() * _mv
    nextidx[idxs_valid] = idxs_valid[idxs_ds]
    return nextidx.reshape(shape)


def isvalid(flwdir):
    """True if NEXTIDX raster is valid"""
    # TODO more checks
    return (flwdir.dtype == 'uint32')
