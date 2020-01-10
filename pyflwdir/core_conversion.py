# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

"""Convert between different flwdir types"""

from numba import njit
import numpy as np
from pyflwdir import core_d8, core_ldd

def d8_to_ldd(flwdir):
    # create conversion dict
    remap = {k:v for (k,v) in zip(core_d8._ds.flatten(), core_ldd._ds.flatten())}
    # add addional land pit code to pcr pit
    remap.update({
        core_d8._pv[1]: core_ldd._pv, 
        core_d8._mv: core_ldd._mv
        })
    # remap values
    return np.vectorize(lambda x: remap.get(x, core_ldd._mv))(flwdir)

def ldd_to_d8(flwdir):
    # create conversion dict
    remap = {k:v for (k,v) in zip(core_ldd._ds.flatten(), core_d8._ds.flatten())}
    # add addional land pit code to pcr pit
    remap.update({
        core_ldd._pv: core_d8._pv[0], 
        core_ldd._mv: core_d8._mv
        })
    # remap values
    return np.vectorize(lambda x: remap.get(x, core_d8._mv))(flwdir)

@njit("u1[:,:](u4[:], Tuple((u8, u8)))")
def nextidx_to_d8(nextidx, shape):
    """convert 1D index to 2D D8 raster"""
    flwdir = np.ones(shape, dtype=np.uint8).ravel()*core_d8._mv
    for idx0 in range(nextidx.size):
        if idx0 == core_d8._mv: continue
        idx_ds = nextidx[idx0]
        flwdir[idx0] = core_d8.idx_to_dd(idx0, idx_ds, shape)
    return flwdir.reshape(shape)

@njit("u1[:,:](u4[:], Tuple((u8, u8)))")
def nextidx_to_ldd(nextidx, shape):
    """convert 1D index to 2D D8 raster"""
    flwdir = np.ones(shape, dtype=np.uint8).ravel()*core_ldd._mv
    for idx0 in range(nextidx.size):
        if idx0 == core_ldd._mv: continue
        idx_ds = nextidx[idx0]
        flwdir[idx0] = core_ldd.idx_to_dd(idx0, idx_ds, shape)
    return flwdir.reshape(shape)