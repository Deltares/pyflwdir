# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, prange
from numba.errors import NumbaPendingDeprecationWarning
import numpy as np
import warnings
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# import flow direction definition
from .core import fd
_nodata = fd._nodata
_pits = fd._pits 
_ds = fd._ds

@njit() #, fastmath=True)
def _nbs_us(idx_ds, flwdir_flat, shape):
    idx_lst = list()
    valid = list()
    nbs_us = np.ones((idx_ds.size, 8), dtype=np.int64)*-1
    for i in range(idx_ds.size):
        idxs_us = fd.us_indices(idx_ds[i], flwdir_flat, shape)
        if np.any(idxs_us):
            idx_lst.extend(idxs_us)
            nbs_us[i, :idxs_us.size] = idxs_us
            valid.append(i)
    return nbs_us, np.array(idx_lst, dtype=np.int64), np.array(valid, dtype=np.int64)

@njit
def setup_dd(idx_ds, flwdir_flat, shape):
    """set drainage direction network from downstream to upstream
    """
    nodes = list()              # list of arrays (n) with downstream indices
    nodes_up = list()           # list of arrays (n, m) with upstream indices; m <= 8
    idx_ds = np.asarray(idx_ds, np.int64)
    # move upstream
    j = 0
    while True:
        nbs_us, idx_next, idx_valid = _nbs_us(idx_ds, flwdir_flat, shape)
        if idx_next.size == 0:
            break
        elif j > 0 and idx_valid.size < idx_ds.size:
            idx_ds = idx_ds[idx_valid]
            nbs_us = nbs_us[idx_valid,:]
        nodes.append(idx_ds)
        nodes_up.append(nbs_us)
        # next iter
        j += 1
        idx_ds = idx_next
    return nodes[::-1], nodes_up[::-1]

# def setup_dd(flwdir_flat, idx_ds, shape):
#     """setup drainage direction network. returns lists of arrays with ds and us indices and basin no. for ds cells"""
#     nodes = list()              # list of arrays (n) with downstream indices
#     nodes_up = list()           # list of arrays (n, 1<>8) with upstream indices
#     basins = list()             # list of arrays (n) with basin no. for basin cells 
#     # NOTE: size n varies for each iteration
#     idx_ds = np.asarray(idx_ds, np.int64)
#     basins_ds = np.arange(idx_ds.size).astype(np.int64)
    
#     # create network moving upstream
#     j = 0
#     while True:
#         idx_next = list()       # flattened list of upstream indices
#         basins_next = list()    # flattened list of upstream basin indices
#         has_us = list()          # True if any upstream neighbor
#         # find upstream cells
#         nbs_up = np.ones((idx_ds.size, 8), dtype=np.int64)*-1
#         N=0
#         for i, idx in enumerate(idx_ds):
#             idxs_us = fd.us_indices(idx, flwdir_flat, shape)
#             ibas = basins_ds[i]
#             if idxs_us.size > 0:
#                 n_up = idxs_us.size
#                 nbs_up[i, :n_up] = idxs_us
#                 if n_up > N:
#                     N = n_up
#                 idx_next.extend(idxs_us)
#                 basins_next.extend(np.ones(idxs_us.size, dtype=np.int64)*ibas)
#                 has_us.append(i)
#         # append trimmed upstream nodes array
#         if len(has_us) > 0:
#             if j == 0: # include all in first iteration
#                 idx_valid = np.arange(idx_ds.size).astype(np.int64)
#             else:
#                 idx_valid = np.asarray(has_us)
#             nodes.append(np.atleast_1d(idx_ds[idx_valid]))
#             basins.append(basins_ds[idx_valid])
#             nodes_up.append(np.atleast_2d(nbs_up[idx_valid, :N]))
#         # break if no more upstream cells
#         if len(idx_next) == 0:
#             break
#         # next iteration
#         idx_ds = np.array(idx_next, dtype=np.int64)
#         basins_ds = np.array(basins_next, dtype=np.int64)
#         j += 1
            
#     return nodes[::-1], nodes_up[::-1], basins[::-1]

@njit
def delineate_basins(rnodes, rnodes_up, idx, values, shape):
    """"""
    basidx_flat = np.zeros(shape, dtype=values.dtype).ravel()
    basidx_flat[idx] = values
    for i in range(len(rnodes)):
        k = -i-1
        for j in range(len(rnodes[k])):
            idx_ds = rnodes[k][j]
            idxs_us = rnodes_up[k][j] # NOTE: has nodata (-1) values
            basidx_ds = basidx_flat[idx_ds]
            for idx_us in idxs_us:
                #NOTE: only flowwing block is different from flux.propagate_upstream
                if idx_us == -1: break
                if basidx_flat[idx_us] == 0: 
                    basidx_flat[idx_us] = basidx_ds
    return basidx_flat.reshape(shape)

@njit
def _main_upsteam(idxs_us, uparea_flat, upa_min):
    upa_max = upa_min
    idx_main_us = np.int64(-9999)
    for i in range(idxs_us.size):
        idx_us = idxs_us[i]
        if idx_us != -1: break
        upa = uparea_flat[idx_us]
        if upa > upa_max:
            upa_max = upa
            idx_main_us = idx_us
    return idx_main_us

@njit
def main_upstream(rnodes, rnodes_up, uparea, upa_min=np.float32(0.)):
    """return grid with main upstream cell index based on largest upstream area."""
    shape = uparea.shape
    uparea_flat = uparea.ravel()
    # output
    main_us = np.ones(uparea_flat.size, dtype=np.int64)*-9999
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j] # NOTE: has nodata (-1) values
            main_us[idx_ds] = _main_upsteam(idxs_us, uparea_flat, upa_min)
    return main_us.reshape(shape)

@njit
def _strahler_order(idxs_us, strord_flat):
    head_lst  = list()
    ord_max = np.int32(1)
    ord_cnt = 0
    for i in range(idxs_us.size):
        idx_us = idxs_us[i]
        if idx_us == -1: break
        ordi = strord_flat[idx_us]
        if ordi <= 0: # most upstream cells
            ordi = np.int32(1)
            head_lst.append(idx_us)
        if ordi >= ord_max:
            if ordi == ord_max:
                ord_cnt += 1
            else:
                ord_max = ordi
                ord_cnt = 1
    if ord_cnt >= 2: # where two channels of order i join, a channel of order i+1 results
        ord_max += 1
    return ord_max, np.array(head_lst, dtype=np.int64)

@njit
def stream_order(rnodes, rnodes_up, shape):
    strord_flat = np.ones(shape, dtype=np.int16).ravel()*-9999
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j] # NOTE: has nodata (-1) values
            ordi, idx_head = _strahler_order(idxs_us, strord_flat)
            strord_flat[idx_ds] = np.int16(ordi) # update stream order downstream cells
            if idx_head.size > 0: # update head cells
                strord_flat[idx_head] = np.int16(1)
    return strord_flat.reshape(shape)
                
