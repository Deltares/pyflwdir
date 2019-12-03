# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit, prange
import numpy as np

_mv = np.uint32(-1)

@njit
def _interal_idx(idx0, idxs_valid, size):
    idxs_inv = np.ones(size, np.uint32)*_mv
    idxs_inv[idxs_valid] = np.array([i for i in range(idxs_valid.size)], dtype=np.uint32)
    return idxs_inv[idx0]

@njit
def us_indices(idx0, idxs_us):
    """returns a numpy array with indices of upstream idx0"""
    idxs_us0 = idxs_us[idx0, :].ravel()
    return idxs_us0[idxs_us0 != _mv]

# @njit
# def us_index_main(idx0, idxs, uparea, upa_min):
#     """returns the index of the upstream cell with the largest uparea"""
#     idxs_us = us_indices(idx0, idxs)
#     if idxs_us.size > 0:
#         idx_us = idxs_us[np.argmax(idxs_us)]
#     else:
#         idx_us = _mv
#     return idx_us

# @njit
# def pit_indices(idxs):
#     """returns the indices of pits"""
#     idx_lst = []
#     for idx0 in range(idxs.size):
#         if idx0 == idxs[idx0]:
#             idx_lst.append(idx0)
#     return np.array(idx_lst, dtype=np.uint32)

@njit
def error_indices(pits, idxs_ds, idxs_us):
    """returns the indices erroneous cells which do not have a pit at its downstream end"""
    # internal lists
    no_loop = idxs_ds == _mv
    idxs_ds0 = pits
    no_loop[idxs_ds0] = True 
    # loop over pits and mark upstream area
    while True:
        idxs_us0 = us_indices(idxs_ds0, idxs_us)
        if idxs_us0.size == 0: 
            break
        no_loop[idxs_us0] = ~no_loop[idxs_us0]
        idxs_ds0 = idxs_us0 # next iter
    return np.where(~no_loop)[0]

@njit
def _ds_stream(idx0, idxs_ds, stream_flat):
    """return index of nearest downstream stream"""
    at_stream = stream_flat[idx0]
    while not at_stream:
        idx_ds = idxs_ds[idx0]
        if idx_ds == idx0 or idx_ds == _mv: 
            break
        idx0 = idx_ds
        at_stream = stream_flat[idx0]
    return idx0

@njit
def ds_stream(idxs0, idxs_ds, idxs_valid, stream):
    """return index of nearest downstream stream"""
    idx_out = np.zeros(idxs0.size, dtype=np.uint32)
    stream_flat = stream.ravel()[idxs_valid]
    for i in range(idxs0.size):
        idx_out[i] = _ds_stream(np.uint32(idxs0[i]), idxs_ds, stream_flat)
    return idx_out


