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
def parse_flow(nextx, nexty, _max_depth = 35):
    size = nextx.size
    ncol = nextx.shape[1]
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
        if r == _pv:
            idxs_ds[i] = i
            pits_lst.append(np.uint32(i))
        else:
            r, c = r-1, c-1
            idx_ds = c + r*ncol
            ids = idxs_inv[idx_ds]
            idxs_ds[i] = ids
            for ii in range(_max_depth):
                if idxs_us[ids,ii] == core._mv:
                    idxs_us[ids,ii] = i
                    break
            if ii >  _max_us:
                _max_us = ii
            if ii == _max_depth-1:
                raise ValueError('increase max depth')
    idxs_us = idxs_us[:, :_max_depth]
    return idxs_valid, idxs_ds, idxs_us, np.array(pits_lst)