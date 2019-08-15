# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np

# import flow direction definition
from .core import fd
_nodata = fd._nodata
_pits = fd._pits 
_ds = fd._ds

# TODO: check that all cells drain to an outlet.
# On a tile level (with buffer) this could be done by checking 
# that cells reach either the tile bounds or an outlet
@njit()
def flwdir_check(flwdir_flat, shape):
    # internal lists
    idx_check = []
    idx_repair = []
    data_check = np.zeros(flwdir_flat.size, dtype=np.uint8)

    # check flwdir. save indices where 
    # - local flwdir is pit
    # - ds neighbor is nodata / outside domain -> set pit
    for idx0 in range(flwdir_flat.size):
        dd = flwdir_flat[idx0]
        if dd == _nodata:
            data_check[idx0] = np.uint8(1) # mark nodata cells as checked
        elif np.any(dd == _pits):
            idx_check.append(idx0)
        elif np.any(dd == _ds):
            # check downstream neighbor
            idx_ds = fd.ds_index(idx0, flwdir_flat, shape)
            if idx_ds == -1: # flows outside
                idx_check.append(idx0)
                idx_repair.append(idx0) # add to repair list
        else:
            raise ValueError('unknown flow direction value found')

    # loop over pits and mark upstream area
    idxs_ds = np.array(idx_check) # this list should always contain indices
    data_check[idxs_ds] = np.uint8(1) # mark pit as checked
    while True:
        idxs_next = []
        for idx_ds in idxs_ds:
            idxs_us = fd.us_indices(idx_ds, flwdir_flat, shape)
            # for i in range(idxs_us.size):
            for idx0 in idxs_us:
                if idx0 < flwdir_flat.size:
                    data_check[idx0] = np.uint8(1) # mark cells connected to pit as checked
                    idxs_next.append(idx0)
        if len(idxs_next) == 0:
            break
        idxs_ds = np.array(idxs_next) # next iter
    
    # check if all cells marked
    # hascyles = np.any(data_check == np.uint8(0))
    hasloops = False
    for idx0 in range(data_check.size):
        if data_check[idx0] == np.uint8(0): # not marked
            hasloops = True
            # mark all cells in loop
            idx_ds = idx0
            while data_check[idx_ds] == np.uint8(0):
                data_check[idx_ds] = np.uint8(1)
                idx_us = idx_ds 
                idx_ds = fd.ds_index(idx_us, flwdir_flat, shape)
            idx_repair.append(idx_us) # add to repair list

    return np.array(idx_repair, dtype=np.uint32), hasloops

# general functions to apply stats of multiple groups (e.g. subbasins)
@njit
def group_count(groups, indices):
    grpcnt = np.ones(groups.size, dtype=np.int32)
    for idx in range(groups.size):
        idxs = indices[idx]
        grpcnt[idx] = idxs.size
    return grpcnt

@njit
def group_sum(groups, indices, values):
    values = values.ravel()
    grpsum = np.ones(groups.size, dtype=values.dtype)*-9999
    for idx in range(groups.size):
        idxs = indices[idx]
        grpsum[idx] = np.sum(values[idxs])
    return grpsum

@njit
def group_mean(groups, indices, values):
    values = values.ravel()    
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