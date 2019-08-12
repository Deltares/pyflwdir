# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np
from math import hypot, pi, atan2

# import flow direction definition
from .core import fd
_nodata = fd._nodata
_pits = fd._pits 
_ds = fd._ds

# @njit
# def _vector_d8(flwdir, dx_matrix, dy_matrix):
#     mag = np.zeros(flwdir.size, np.float32)
#     angle = np.zeros(flwdir.size, np.float32)
#     flwdir_flat = flwdir.flatten()
#     dx_flat = dx_matrix.flatten()
#     dy_flat = dy_matrix.flatten()
#     for idx in range(flwdir.size):
#         dd = flwdir_flat[idx]
#         dr, dc = fd.dd_2_drdc(dd)
#         dy = dy_flat[idx] * float(dr) * -1
#         dx = dx_flat[idx] * float(dc)
#         mag0 = hypot(dy, dx)
#         mag[idx] = mag0
#         if mag0 != 0:
#             ang0 = pi/2. - atan2(dx/mag0, dy/mag0)
#             angle[idx] = ang0
#     return mag.reshape(flwdir.shape), angle.reshape(flwdir.shape)

# def vector_d8(flwdir, dx=None, dy=None):
#     """returns the magnitude and angle for a d8 field 
#     which can be used as input to plot a vector field"""
#     if dx is None or dy is None:
#         dx = np.ones(flwdir.shape, np.float32)
#         dy = np.ones(flwdir.shape, np.float32)
#     return _vector_d8(flwdir, dx, dy)

@njit
def trace_riv_reach(idx_ds, stro_ds, flwdir_flat, stream_order_flat, shape):
    idx0 = fd.ds_index(idx_ds, flwdir_flat, shape)
    if idx0 != np.uint32(-1) and idx0 != idx_ds:
        nodes = list([idx0, idx_ds])
    else:
        nodes = list([idx_ds])
    tribs = list()
    while True:
        idxs_us = fd.us_indices(idx_ds, flwdir_flat, shape)
        for idx_us in idxs_us:
            if stream_order_flat[idx_us] == stro_ds: #main stream
                nodes.append(idx_us)
            elif stream_order_flat[idx_us] > 0:
                tribs.append(idx_us)
        if nodes[-1] == idx_ds:
            break
        idx_ds = nodes[-1]
    return np.array(nodes, np.uint32), np.array(tribs, np.uint32)

@njit
def river_nodes(idx_ds, flwdir_flat, stream_order_flat, shape):
    idx_ds = np.asarray(idx_ds, np.uint32)
    rivs = list()
    stro = list()
    while True:
        idx_next = list()
        for idx in idx_ds:
            stro0 = stream_order_flat[idx]
            riv, tribs = trace_riv_reach(idx, stro0, flwdir_flat, stream_order_flat, shape)
            rivs.append(riv)
            stro.append(stro0)
            idx_next.extend(tribs)
        if len(idx_next) == 0:
            break
        idx_ds = np.array(idx_next, dtype=np.uint32)
    return rivs, stro

@njit
def _update_bbox(idx_ds, xmin, ymin, xmax, ymax, ncol):
    y = idx_ds // ncol
    x = idx_ds %  ncol
    ymax, ymin = np.maximum(y, ymax), np.minimum(y, ymin)
    xmax, xmin = np.maximum(x, xmax), np.minimum(x, xmin)
    return xmin, ymin, xmax, ymax

@njit
def basin_bbox(rnodes, rnodes_up, idx, lats, lons, resy, resx):
    nrow, ncol = lats.size, lons.size
    # initialize arrays
    basidx_flat = np.zeros(nrow*ncol, dtype=np.uint32)
    rcbboxs = np.ones((idx.size, 4), dtype=np.uint32)*np.uint32(-1)
    bboxs = np.ones((idx.size, 4), dtype=lats.dtype)*-1
    
    # get bbox in row/col integers
    for ibas in range(idx.size):
        rcbboxs[ibas,:] = _update_bbox(idx[ibas], ncol, nrow, 0, 0, ncol)
        basidx_flat[idx[ibas]] = np.uint32(ibas+1)
    for i in range(len(rnodes)):
        k = -i-1
        for j in range(len(rnodes[k])):
            idx_ds = rnodes[k][j]
            idxs_us = rnodes_up[k][j] # NOTE: has nodata (-1) values
            ibas = basidx_flat[idx_ds]
            if ibas == 0: continue
            ibas -= 1 # convert to zero based count
            for idx_us in idxs_us:
                #NOTE: only flowwing block is different from flux.propagate_upstream
                if idx_us == np.uint32(-1): break
                if basidx_flat[idx_us] == 0: 
                    basidx_flat[idx_us] = np.uint32(ibas+1)
                    xmin, ymin, xmax, ymax = rcbboxs[ibas, :]
                    rcbboxs[ibas,:] = _update_bbox(idx_us, xmin, ymin, xmax, ymax, ncol)

    # convert to lat/lon bbox assuming lat/lon on ceter pixel
    for ibas in range(idx.size):
        xmin, ymin, xmax, ymax = rcbboxs[ibas, :]
        if xmin == np.uint32(-1): continue
        assert ymax < nrow
        west, east = lons[xmin]-resx/2., lons[xmax]+resx/2.
        if resy<0: # N -> S
            south, north = lats[ymax]+resy/2., lats[ymin]-resy/2.
        else:
            south, north = lats[ymax]-resy/2., lats[ymin]+resy/2.
        bboxs[ibas,:] = west, south, east, north

    return bboxs