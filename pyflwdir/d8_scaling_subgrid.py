# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np
# local
from d8_func import d8


@njit
def group_count(groups, indices):
    grpcnt = np.ones(groups.size, dtype=np.int32)
    for idx in range(groups.size):
        idxs = indices[idx]
        grpcnt[idx] = idxs.size
    return grpcnt

@njit
def group_sum(groups, indices, values):
    values = values.flatten()
    grpsum = np.ones(groups.size, dtype=values.dtype)*-9999
    for idx in range(groups.size):
        idxs = indices[idx]
        grpsum[idx] = np.sum(values[idxs])
    return grpsum

@njit
def group_mean(groups, indices, values):
    values = values.flatten()    
    grpmean = np.ones(groups.size, dtype=values.dtype)*-9999
    for idx in range(groups.size):
        idxs = indices[idx]
        grpmean[idx] = np.mean(values[idxs])
    return grpmean

@njit
def group_percentile(groups, indices, values, q):
    grpqnt = np.ones((groups.size, q.size), dtype=values.dtype)*-9999
    for idx in range(groups.size):
        idxs = indices[idx]
        vals = values[idxs]
        grpqnt[idx,:] = np.percentile(vals, q)
    return grpqnt

@njit
def river_length_slope(groups, indices, flwdir, elevtn, dy, dx):
    flwdir_flat = flwdir.flatten()
    elevtn_flat = elevtn.flatten()
    dy_flat = dy.flatten()
    dx_flat = dx.flatten()
    rivlen = np.zeros(groups.size, dtype=np.float32) #km
    rivslp = np.zeros(groups.size, dtype=np.float32) #m/m
    for idx in range(groups.size):
        l = np.float32(0)
        idxs = indices[idx]
        for subidx in idxs:
            l += d8.dist_d8(subidx, flwdir_flat, dy_flat, dx_flat) #km
        if l > 0:
            z0 = elevtn_flat[idxs[0]]
            z1 = elevtn_flat[idxs[-1]]
            rivslp[idx] = (z1-z0) / (l*1e3) # m/m
            rivlen[idx] = l
    return rivlen, rivslp

@njit
def ucat_map(groups, indices, shape):
    nrow, ncol = shape
    catidx_flat = np.ones(nrow*ncol, dtype=np.int64)*-9999
    for idx in range(groups.size):
        idxs = indices[idx]
        assert np.all(catidx_flat[idxs] == -9999)
        catidx_flat[idxs] = groups[idx]
    return catidx_flat.reshape(shape)

@njit
def subgrid_indices(outlet_lr, flwdir, uparea, upa_min):
    """return lists of indices of subgrid unitcatchment and subgrid river cell"""
    shape = flwdir.shape
    flwdir_flat = flwdir.flatten()
    uparea_flat = uparea.flatten()
    outlet_lr_flat = outlet_lr.flatten()
    # prepare output
    groups = []
    catidx_flat = np.zeros(flwdir.size, dtype=np.uint8)
    for idx0 in range(outlet_lr.size):
        subidx = outlet_lr_flat[idx0]
        if subidx < 0: continue
        catidx_flat[subidx] = np.uint8(1)
        groups.append(idx0)
    groups = np.array(groups)
    # delineate unit catchments and get main upstream cells with uparea > upa_min
    cat_lst = []
    riv_lst = []
    for idx0 in groups:
        subidx = outlet_lr_flat[idx0]
        catidxs, rividxs = _subgrid_idx(subidx, flwdir_flat, uparea_flat, catidx_flat, shape, upa_min=upa_min)
        cat_lst.append(catidxs)
        riv_lst.append(rividxs)
    return groups, cat_lst, riv_lst

@njit
def _subgrid_idx(idx0, flwdir_flat, uparea_flat, catidx_flat, shape, upa_min=0.5):
    idx_lst = [idx0] # all unit catchment subgrid indice
    riv_lst = []
    riv = False
    if uparea_flat[idx0] > upa_min:
        riv_lst.append(idx0)
        riv = True
    i = 0    
    while len(idx_lst) > i:
        idxs_up = d8.us_d8(idx_lst[i], flwdir_flat, shape)
        upa1 = upa_min
        idx1 = -1
        for idx in idxs_up:
            if idx >= 0 and catidx_flat[idx] == 0:
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
