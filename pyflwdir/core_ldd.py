# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
import numpy as np
from pyflwdir import core

# LDD type
_ftype = 'ldd'
_ds = np.array([
    [7, 8, 9],
    [4, 5, 6],
    [1, 2, 3]], dtype=np.uint8)
_us = np.array([
    [3, 2, 1],
    [6, 5, 4],
    [9, 8, 7]], dtype=np.uint8)
_mv = np.uint8(255)
_pv = np.uint8(5)
_ldd_ = np.unique(np.concatenate([[_pv], [_mv], _ds.flatten()])) 
_max_depth = 8

@njit("Tuple((int8, int8))(uint8)")
def drdc(dd):
    """convert ldd value to delta row/col"""
    dr, dc = np.int8(0), np.int8(0)
    if dd >= np.uint8(4): # W / PIT / E / NW / N / NE
        if dd >= np.uint(7): # NW / N / NE
            dr = np.int8(-1)
            dc = np.int8(dd-8)
        else: # W / PIT / E
            dr = np.int8(0)
            dc = np.int8(dd-5)
    else: # SW / S / SE
        dr = np.int8(1)
        dc = np.int8(dd-2)
    return dr, dc

@njit("u1(u4, u4, Tuple((u8,u8)))")
def idx_to_dd(idx0, idx_ds, shape):
    """returns local LDD value based on current and downstream index"""
    ncol = shape[1]
    r = (idx_ds // ncol) - (idx0 // ncol) + 1
    c = (idx_ds %  ncol) - (idx0 %  ncol) + 1
    dd = _mv
    if r >= 0 and r < 3 and c >= 0 and c < 3:
        dd = _ds[r, c]
    return dd

@njit("Tuple((u4[:], u4[:], u4[:,:], u4[:]))(u1[:,:])")
def from_flwdir(flwdir):
    """convert 2D LDD network to 1D indices"""
    size = flwdir.size
    nrow, ncol = flwdir.shape[0], flwdir.shape[-1]
    flwdir_flat = flwdir.ravel()
    # keep valid indices only
    idxs_valid = np.where(flwdir.ravel()!=_mv)[0].astype(np.uint32)
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
        dr, dc = drdc(flwdir_flat[idx0])
        r, c = idx0//ncol+dr, idx0%ncol+dc
        if (dr==0 and dc==0) or r >= nrow or c >= ncol or r<0 or c<0 or flwdir_flat[idx0] == _mv: 
            # pit or ds cell is out of bounds / invalid -> set pit
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:
            # valid ds cell
            idx_ds = idx0 + dc + dr*ncol
            ids = idxs_inv[idx_ds]
            if ids == core._mv or ids == i:
                raise ValueError('invalid flwdir data')
            idxs_ds[i] = ids
            for ii in range(_max_depth):
                if idxs_us[ids,ii] == core._mv:
                    idxs_us[ids,ii] = i
                    break
            if ii >  _max_us:
                _max_us = ii
    idxs_us = idxs_us[:, :_max_us+1]
    return idxs_valid, idxs_ds, idxs_us, np.array(pits_lst)

@njit("u1[:,:](u4[:], u4[:], Tuple((u8, u8)))")
def to_flwdir(idxs_valid, idxs_ds, shape):
    """convert 1D index to 2D LDD raster"""
    n = idxs_valid.size
    flwdir = np.ones(shape, dtype=np.uint8).ravel()*_mv
    for i in range(n):
        idx0 = idxs_valid[i]
        idx_ds = idxs_valid[idxs_ds[i]]
        flwdir[idx0] = idx_to_dd(idx0, idx_ds, shape)
    return flwdir.reshape(shape)

def isvalid(flwdir):
    """True if 2D LDD raster is valid"""
    return (flwdir.dtype == 'uint8' and 
            flwdir.ndim == 2 and
            np.all([v in _ldd_ for v in np.unique(flwdir)]))

@vectorize(["b1(u1)"])
def ispit(dd):
    """True if LDD pit"""
    return dd == _pv

@vectorize(["b1(u1)"])
def isnodata(dd):
    """True if LDD nodata"""
    return dd == _mv