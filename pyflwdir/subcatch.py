# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np
import math

# import flow direction definition
from .core import fd
from .d8_scaling import _outlet

@njit
def unitcatchment_outlets(scale_ratio, flwdir_flat, uparea_flat, shape):
    sub_nrow, sub_ncol = shape
    lr_nrow, lr_ncol = int(sub_nrow/scale_ratio), int(sub_ncol/scale_ratio)
    shape_lr = lr_nrow, lr_ncol
    size_lr = lr_nrow * lr_ncol
    # output cells
    outlet_idx = np.ones(size_lr, dtype=np.int64)*-1

    # get representative cells (largest uparea in effective area) on highres maps
    # trace to lowres cell boundary to find outlet cells
    for idx0 in range(outlet_idx.size):
        subidx_out = _outlet(idx0, flwdir_flat, uparea_flat, shape, scale_ratio)[0]
        outlet_idx[idx0] = subidx_out   # subgrid index of outlet point lowres grid cell

    return outlet_idx.reshape(shape_lr)

@njit
def subcatch_indices(outlet_idx, flwdir_flat, uparea_flat, upa_min, shape):
    """returns lists of indices of subcatchment indices and main river cell 
    defined as path upstream from the outlet connecting the largest upstream 
    cells with minimum upstream area of <upa_min>"""
    size = shape[0]*shape[1]
    # prepare output
    groups = []
    catidx_flat = np.zeros(size, dtype=np.int8)
    for idx0 in range(outlet_idx.size):
        subidx = outlet_idx[idx0]
        if subidx >= 0 and subidx < size:
            catidx_flat[subidx] = np.int8(1)
            groups.append(idx0)
    groups = np.array(groups)
    # delineate unit catchments and get main upstream cells with uparea > upa_min
    cat_lst = []
    riv_lst = []
    for idx0 in groups:
        subidx = outlet_idx[idx0]
        catidxs, rividxs = _subgrid_idx(subidx, flwdir_flat, uparea_flat, catidx_flat, shape, upa_min=upa_min)
        cat_lst.append(catidxs)
        riv_lst.append(rividxs)
    return cat_lst, riv_lst

@njit
def _subgrid_idx(idx0, flwdir_flat, uparea_flat, catidx_flat, shape, upa_min=0.5):
    """returns list of indices for single subcatch and main river cells"""
    idx_lst = [idx0] # all unit catchment subgrid indice
    riv_lst = []
    riv = False
    if uparea_flat[idx0] > upa_min:
        riv_lst.append(idx0)
        riv = True
    i = 0    
    while len(idx_lst) > i:
        idxs_up = fd.us_indices(idx_lst[i], flwdir_flat, shape)
        upa1 = upa_min
        idx1 = -1
        for idx in idxs_up:
            if idx >= 0 and catidx_flat[idx] == np.int8(0):
                idx_lst.append(idx)
                if riv:
                    upa = uparea_flat[idx]
                    if upa > upa1:
                        idx1 = idx
                        upa1 = upa
        if riv and idx1 != -1:
            riv_lst.append(idx1)
        else:
            riv = False
        i += 1 # next iter
    return np.array(idx_lst, dtype=np.int64), np.array(riv_lst, dtype=np.int64)

@njit
def river_length_slope(indices, flwdir_flat, elevtn_flat, dx, dy, shape):
    """Calculate the length and slope of a river segment insice a subcatchment
    note dx, dy should be in [m]
    """
    ncol = shape[1]
    nsegments = len(indices)
    rivlen = np.zeros(nsegments, dtype=dx.dtype) #m
    rivslp = np.zeros(nsegments, dtype=dx.dtype) #m/m
    for idx in range(nsegments):
        l = rivlen[idx]
        idxs = indices[idx]
        for subidx in idxs:
            r = idx // ncol
            dr, dc = fd.dd_2_drdc(flwdir_flat[subidx])
            l += math.hypot(dr*dy[r], dc*dx[r])
        if l > 0:
            z0 = elevtn_flat[idxs[0]]
            z1 = elevtn_flat[idxs[-1]]
            rivslp[idx] = (z1-z0) / l # m/m
            rivlen[idx] = l
    return rivlen, rivslp

@njit
def subcatch_map(groups, indices, values, shape):
    nrow, ncol = shape
    catidx_flat = np.zeros(nrow*ncol, dtype=values.dtype)
    for idx in range(groups.size):
        idxs = indices[idx]
        catidx_flat[idxs] = groups[idx]
    return catidx_flat.reshape(shape)


