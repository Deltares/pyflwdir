# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, prange, guvectorize
from numba.errors import NumbaPendingDeprecationWarning
import numpy as np
import warnings
import dask
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# import flow direction definition
from .core import fd

# 1) VECTORIZED/PARALLEL
# This vectorized and parallel code used to work, but is slower
# also not sure if the guvectorize layout argument is correct
# @njit
# def _us8(idx, flwdir_flat, shape):
#     out = np.ones(8, dtype=np.uint32)*np.uint32(-1)
#     idxs_us = fd.us_indices(idx, flwdir_flat, (shape[0], shape[1]))
#     out[:idxs_us.size] = idxs_us
#     return out
# @guvectorize(
#     ['void(uint32[:], uint8[:], int64[:], uint32[:], uint32[:,:])'],
#     '(n),(k),(j),(m)->(n,m)', target='parallel'
#     )
# def _us8_vec(idx_ds, flwdir_flat, shape, empty8, out):
#     for i in range(idx_ds.size):
#         out[i,:] = _us8(idx_ds[i], flwdir_flat, shape)
# @njit(parallel=True)
# def _us8_vec(idx_ds, flwdir_flat, shape, emtpy8):
#     out = np.zeros((idx_ds.size, 8), dtype=np.uint32)
#     for i in prange(idx_ds.size):
#         out[i,] = _us8(idx_ds[i], flwdir_flat, shape)
#     return out
# @njit
# def setup_dd_vec(idx_ds, flwdir_flat, shape):
#     """set drainage direction network from downstream to upstream
#     """
#     nodes = list()              # list of arrays (n) with downstream indices
#     nodes_up = list()           # list of arrays (n, m) with upstream indices; m <= 8
#     empty8 = np.ones(8, dtype=np.uint32)
#     # move upstream
#     j = 0
#     while True:
#         nbs_us = _us8_vec(idx_ds, flwdir_flat, np.asarray(shape), empty8)
#         idx_valid = np.where(nbs_us[:,0] != np.uint32(-1))[0]
#         if idx_valid.size==0:
#             break
#         elif j > 0:
#             idx_ds = idx_ds[idx_valid]
#             nbs_us = nbs_us[idx_valid,:]
#         nodes.append(idx_ds)
#         nodes_up.append(nbs_us)
#         # next iter
#         j += 1
#         # NOTE 2d boolean indexing does not work currenlty in numba
#         idx_ds = nbs_us.reshape(-1)[nbs_us.reshape(-1) != np.uint32(-1)].astype(np.uint32)
#     return nodes[::-1], nodes_up[::-1]

# This dask distributed method is super slow. only with huge load (kin wave?) this might work.
# @dask.delayed
# def _us8_dask(idx, flwdir_flat, shape):
#     out = np.ones((1,8), dtype=np.uint32)*-1
#     idxs_us = fd.us_indices(idx, flwdir_flat, (shape[0], shape[1]))
#     out[0,:idxs_us.size] = idxs_us
#     return out
# def setup_dd_dask(idx_ds, flwdir_flat, shape):
#     """set drainage direction network from downstream to upstream
#     """
#     nodes = list()              # list of arrays (n) with downstream indices
#     nodes_up = list()           # list of arrays (n, m) with upstream indices; m <= 8
#     # move upstream
#     j = 0
#     while True:
#         tasks = []
#         for i in range(idx_ds.size):
#             tasks.append(_us8_dask(idx_ds[i], flwdir_flat, shape))
#         nbs_us = np.concatenate(dask.compute(*tasks, scheduler="processes"), axis=0)
#         idx_valid = np.where(nbs_us[:,0] != np.uint32(-1))[0]
#         if idx_valid.size==0:
#             break
#         elif j > 0:
#             idx_ds = idx_ds[idx_valid]
#             nbs_us = nbs_us[idx_valid,:]
#         nodes.append(idx_ds)
#         nodes_up.append(nbs_us)
#         # next iter
#         j += 1
#         # NOTE 2d boolean indexing does not work currenlty in numba
#         idx_ds = nbs_us[nbs_us != np.uint32(-1)].astype(np.uint32)
#     return nodes[::-1], nodes_up[::-1]

@njit
def _nbs_us(idx_ds, flwdir_flat, shape):
    nbs_us = np.ones((idx_ds.size, 8), dtype=np.uint32)*np.uint32(-1) # use uint32 to save memory as these networks get large!
    valid = np.zeros(idx_ds.size, dtype=np.int8)
    N = 1
    for i in range(idx_ds.size):
        idxs_us = fd.us_indices(idx_ds[i], flwdir_flat, shape) # int64, but always positive
        n = idxs_us.size
        if n > 0:
            nbs_us[i, :n] = idxs_us
            valid[i] += 1
            if n > N:
                N = n
    # the astype seems to be required, otherwise the following error is raised in a downstream function
    # TypeError: can't unbox heterogeneous list: array(uint32, 2d, C) != array(uint32, 2d, A)
    nbs_us = nbs_us[:,:N].astype(np.uint32) 
    return nbs_us, valid

@njit
def setup_dd(idx_ds, flwdir_flat, shape):
    """set drainage direction network from downstream to upstream
    """
    size = np.uint(shape[0]*shape[1])
    assert size < 2**32-2       # maximum size we can have with uint32 indices
    nodes = list()              # list of arrays (n) with downstream indices
    nodes_up = list()           # list of arrays (n, m) with upstream indices; m <= 8
    # move upstream
    j = 0
    while True:
        nbs_us, valid = _nbs_us(idx_ds, flwdir_flat, shape)
        idx_valid = np.where(valid == np.int8(1))[0]
        if idx_valid.size==0:
            break
        elif j > 0:
            idx_ds = idx_ds[idx_valid]
            nbs_us = nbs_us[idx_valid,:]
        nodes.append(idx_ds)
        nodes_up.append(nbs_us)
        # next iter
        j += 1
        # NOTE 2d boolean indexing does not work currenlty in numba; flatten first
        idx_ds = nbs_us.ravel()
        idx_ds = idx_ds[idx_ds < size]
    return nodes[::-1], nodes_up[::-1]

@njit
def basin_map(rnodes, rnodes_up, idx, values, shape):
    """"""
    size = shape[0]*shape[1]
    basidx_flat = np.zeros(size, dtype=values.dtype)
    basidx_flat[idx] = values
    for i in range(len(rnodes)):
        k = -i-1
        for j in range(len(rnodes[k])):
            idx_ds = rnodes[k][j]
            idxs_us = rnodes_up[k][j] # NOTE: has nodata np.uint32(-1) values
            basidx_ds = basidx_flat[idx_ds]
            for idx_us in idxs_us:
                #NOTE: only flowwing block is different from flux.propagate_upstream
                if idx_us > size: break
                if basidx_flat[idx_us] == 0: 
                    basidx_flat[idx_us] = basidx_ds
    return basidx_flat.reshape(shape)

@njit()
def upstream_area(rnodes, rnodes_up, cellare, shape):
    nrow, ncol = shape
    size = nrow*ncol
    assert cellare.size == nrow
    upa = np.ones(size, dtype=cellare.dtype)*-9999.
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j] # NOTE: has nodata np.uint32(-1) values
            upa_ds = cellare[idx_ds // ncol]
            for idx_us in idxs_us:
                if idx_us >= size: break
                upa_us = upa[idx_us]
                if upa_us <= 0:
                    upa_us = cellare[idx_us // ncol]
                    upa[idx_us] = upa_us
                upa_ds += upa_us
            upa[idx_ds] = upa_ds
    return upa.reshape(shape)

@njit
def _main_upsteam(idxs_us, uparea_flat, upa_min):
    size = uparea_flat.size
    upa_max = upa_min
    idx_main_us = np.uint32(-1)
    for i in range(idxs_us.size):
        idx_us = idxs_us[i]
        if idx_us >= size: break
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
    main_us = np.ones(uparea_flat.size, dtype=np.uint32)*np.uint32(-1)
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j] # NOTE: has nodata (-1) values
            main_us[idx_ds] = _main_upsteam(idxs_us, uparea_flat, upa_min)
    return main_us.reshape(shape)

@njit
def _strahler_order(idxs_us, strord_flat, size):
    head_lst  = list()
    ord_max = np.int8(1)
    ord_cnt = 0
    for i in range(idxs_us.size):
        idx_us = idxs_us[i]
        if idx_us >= size: break
        ordi = strord_flat[idx_us]
        if ordi <= 0: # most upstream cells
            ordi = np.int8(1)
            head_lst.append(idx_us)
        if ordi >= ord_max:
            if ordi == ord_max:
                ord_cnt += 1
            else:
                ord_max = ordi
                ord_cnt = 1
    if ord_cnt >= 2: # where two channels of order i join, a channel of order i+1 results
        ord_max += 1
    return ord_max, np.array(head_lst, dtype=np.uint32)

@njit
def stream_order(rnodes, rnodes_up, shape):
    size = np.uint32(shape[0]*shape[1])
    strord_flat = np.ones(size, dtype=np.int8)*np.int8(-1)
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j]           # NOTE: has nodata (-1) values
            ordi, idx_head = _strahler_order(idxs_us, strord_flat, size)
            strord_flat[idx_ds] = np.int8(ordi) # update stream order downstream cells
            if idx_head.size > 0:               # update head cells
                strord_flat[idx_head] = np.int8(1)
    return strord_flat.reshape(shape)
                
