# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, vectorize
from numba import types
import numpy as np

from pyflwdir import core

# FLOW type
_mv = np.int32(-9999)
_pv = np.int32(-9)

def _is_flow(flwdir):
    nextx, nexty = flwdir
    if not nexty.shape == nextx.shape:
        raise ValueError("nextx and nexty should have the shape")
    size = nextx.size
    mask = np.logical_and(nextx!=_mv, nextx!=_pv)
    nxs = nextx[mask]
    nys = nexty[mask]
    return (np.all(np.logical_and(nxs>0, nxs<=size)) and
            np.all(np.logical_and(nys>0, nys<=size)))

@njit
def flow_to_idxs(nextx, nexty):
    idxs = np.ones(nextx.size, dtype=np.uint32)*core._mv
    nrow, ncol = nextx.shape
    size = nextx.size
    idx0 = np.uint32(0)
    for r0 in range(nrow):
        for c0 in range(ncol):
            c = nextx[r0,c0]
            r = nexty[r0,c0]
            if r != _mv:
                if r == _pv:
                    r, c = r0, c0
                else:
                    r, c = r-1, c-1
                idx_ds = c + r*ncol
                if idx_ds >=0 and idx_ds < size:
                    idxs[idx0] = idx_ds
            idx0 += 1
    return idxs

@njit
def idxs_to_flow(idxs, shape):
    nextx = np.ones(shape, dtype=np.int32)*_mv
    nexty = np.ones(shape, dtype=np.int32)*_mv
    nrow, ncol = shape
    idx0 = np.uint32(0)
    for r0 in range(nrow):
        for c0 in range(ncol):
            idx_ds = idxs[idx0]
            if idx_ds == idx0:
                nextx[r0, c0] = _pv
                nexty[r0, c0] = _pv
            elif idx_ds != core._mv:
                nextx[r0, c0] = idx_ds %  ncol + 1
                nexty[r0, c0] = idx_ds // ncol + 1
            idx0 += 1
    return nextx, nexty