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
#     """returns a numpy array (int32) with indices of upstream neighbors on a d8 grid"""
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
def _update_bbox(idx_ds, xmin, ymin, xmax, ymax, ncol):
    y = idx_ds // ncol
    x = idx_ds %  ncol
    ymax, ymin = np.maximum(y, ymax), np.minimum(y, ymin)
    xmax, xmin = np.maximum(x, xmax), np.minimum(x, xmin)
    return xmin, ymin, xmax, ymax

@njit
def basin_bbox(rnodes, rnodes_up, rbasins, lats, lons, res):
    nrow, ncol = lats.size, lons.size
    idxs_ds = rnodes[-1]
    basins_ds = rbasins[-1]
    rcbboxs = np.ones((basins_ds.max()+1, 4), dtype=np.int64)*-1
    bboxs = np.ones((basins_ds.max()+1, 4), dtype=np.float32)*-9999
    # initialize based on network starting points
    for j in range(idxs_ds.size):
        idx = idxs_ds[j]
        ibas = basins_ds[j]
        xmin, ymin, xmax, ymax = ncol, nrow, 0, 0
        rcbboxs[ibas,:] = _update_bbox(idx, xmin, ymin, xmax, ymax, ncol)
    # loop through network
    for i in range(len(rnodes)):
        nn_us = rnodes_up[-i-1]
        basins_ds = rbasins[-i-1]
        for j in range(nn_us.shape[0]):
            ibas = basins_ds[j]
            idxs_us = nn_us[j,:] # NOTE: has nodata (-1) values
            for idx_us in idxs_us:
                if idx_us < 0: continue
                xmin, ymin, xmax, ymax = rcbboxs[ibas, :]
                rcbboxs[ibas,:] = _update_bbox(idx_us, xmin, ymin, xmax, ymax, ncol)
    # convert to lat/lon bbox assuming lat/lon on ceter pixel
    for ibas in range(rcbboxs.shape[0]):
        xmin, ymin, xmax, ymax = rcbboxs[ibas, :]
        if xmin < 0: continue
        assert ymax < nrow
        west, east = lons[xmin]-res/2., lons[xmax]+res/2.
        if lats[0]>lats[1]: # N -> S
            south, north = lats[ymax]-res/2., lats[ymin]+res/2.
        else:
            south, north = lats[ymax]-res/2., lats[ymin]+res/2.
        bboxs[ibas,:] = west, south, east, north
    valid = np.where(bboxs[:,0]!=-9999)[0]
    return bboxs[valid, :]