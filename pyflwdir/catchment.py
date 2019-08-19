# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np

# import flow direction definition
from .core import fd
from .network import _nbs_us

@njit
def delineate_basins(idxs_ds, flwdir_flat, shape, lats, lons, resy, resx):
    nrow, ncol = shape
    size = nrow*ncol
    assert size < 2**32-2       # maximum size we can have with uint32 indices
    # initialize arrays
    basidx_flat = np.zeros(size, dtype=np.uint32)
    rcbboxs = np.zeros((idxs_ds.size, 4), dtype=np.int32)*-1
    bboxs = np.ones((idxs_ds.size, 4), dtype=lats.dtype)*-1
    
    # get bbox in row/col integers
    for ibas in range(idxs_ds.size):
        rcbboxs[ibas,:] = _update_bbox(idxs_ds[ibas], ncol, nrow, 0, 0, ncol)
        basidx_flat[idxs_ds[ibas]] = np.uint32(ibas+1)
    
    # loop through flwdir map
    while True:
        nbs_us, valid = _nbs_us(idxs_ds, flwdir_flat, shape) # NOTE nbs_us has dtype uint32
        if np.all(valid==np.int8(0)):
            break
        for i in range(idxs_ds.size):
            idx_ds = idxs_ds[i]
            idxs_us = nbs_us[i,]
            ibas = basidx_flat[idx_ds]
            if ibas == 0: continue
            ibas -= 1 # convert to zero based count
            for idx_us in idxs_us:
                #NOTE: only flowwing block is different from flux.propagate_upstream
                if idx_us >= size: break
                if basidx_flat[idx_us] == 0: 
                    basidx_flat[idx_us] = np.uint32(ibas+1)
                    xmin, ymin, xmax, ymax = rcbboxs[ibas, :]
                    rcbboxs[ibas,:] = _update_bbox(idx_us, xmin, ymin, xmax, ymax, ncol)
        # next iter
        idxs_ds = nbs_us.ravel()
        idxs_ds = idxs_ds[idxs_ds < size]

    # convert to lat/lon bbox assuming lat/lon on ceter pixel
    for ibas in range(bboxs.shape[0]):
        xmin, ymin, xmax, ymax = rcbboxs[ibas, :]
        if xmin == -1: continue
        assert ymax < nrow and xmax < ncol
        west, east = lons[xmin]-resx/2., lons[xmax]+resx/2.
        if resy<0: # N -> S
            south, north = lats[ymax]+resy/2., lats[ymin]-resy/2.
        else:
            south, north = lats[ymax]-resy/2., lats[ymin]+resy/2.
        bboxs[ibas,:] = west, south, east, north

    return basidx_flat.reshape(shape), bboxs
    
@njit
def _update_bbox(idx_ds, xmin, ymin, xmax, ymax, ncol):
    y = idx_ds // ncol
    x = idx_ds %  ncol
    ymax, ymin = np.maximum(y, ymax), np.minimum(y, ymin)
    xmax, xmin = np.maximum(x, xmax), np.minimum(x, xmin)
    return xmin, ymin, xmax, ymax