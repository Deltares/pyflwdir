# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, prange
import numpy as np

_mv = np.uint32(-1)

@njit
def ds_index(idx0, idxs):
    """returns numpy array with the index donwstream of idx0"""
    return idxs[idxs]

@njit
def us_indices(idx0, idxs):
    """returns a numpy array with indices of upstream idx0"""
    idxs_us = np.where(idxs == idx0)[0]
    return idxs_us[idxs_us!=idx0].astype(np.uint32)

@njit
def us_index_main(idx0, idxs, uparea, upa_min):
    """returns the index of the upstream cell with the largest uparea"""
    idxs_us = us_indices(idx0, idxs)
    if idxs_us.size > 0:
        idx_us = idxs_us[np.argmax(idxs_us)]
    else:
        idx_us = _mv
    return idx_us

@njit
def pit_indices(idxs):
    """returns the indices of pits"""
    idx_lst = []
    for idx0 in range(idxs.size):
        if idx0 == idxs[idx0]:
            idx_lst.append(idx0)
    return np.array(idx_lst, dtype=np.uint32)

@njit
def error_indices(idxs):
    """returns the indices erroneous cells which do not have a pit at its downstream end"""
    # internal lists
    no_loop = idxs == _mv
    idxs_ds = pit_indices(idxs)
    no_loop[idxs_ds] = True 
    # loop over pits and mark upstream area
    while True:
        idxs_next = []
        for idx_ds in idxs_ds:
            idxs_us = us_indices(idx_ds, idxs)
            if idxs_us.size == 0: 
                continue
            no_loop[idxs_us] = ~no_loop[idxs_us]
            idxs_next.extend(idxs_us)
        if len(idxs_next) == 0:
            break
        idxs_ds = np.array(idxs_next) # next iter
    return np.where(~no_loop)[0]

@njit
def _ds_stream(idxs, idx0, stream_flat):
    """return index of nearest downstream stream"""
    at_stream = stream_flat[idx0]
    while not at_stream:
        idx_ds = ds_index(idx0, idxs)
        if idx_ds == idx0 or idx_ds == _mv: 
            break
        idx0 = idx_ds
        at_stream = stream_flat[idx0]
    return idx0

@njit
def ds_stream(idxs, idx_ds, stream):
    """return index of nearest downstream stream"""
    idx_out = np.zeros(idx_ds.size, idx_ds.dtype)
    stream_flat = stream.ravel()
    for i in range(idx_ds.size):
        idx_out[i] = _ds_stream(idxs, idx_ds[i], stream_flat)
    return idx_out


