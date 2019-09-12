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

@njit
def _streamsegment(idx_ds, stro_ds, flwdir_flat, stream_order_flat, shape):
    """returns main indices upstream from idx_ds with same stream order and indices of tributaries"""
    idx0 = fd.ds_index(idx_ds, flwdir_flat, shape)
    if idx0 != -1 and idx0 != idx_ds:
        nodes = list([idx0, idx_ds])
    else:
        nodes = list([idx_ds])
    tribs = list()
    while True:
        idxs_us = fd.us_indices(idx_ds, flwdir_flat, shape)
        for idx_us in idxs_us:
            if stream_order_flat[idx_us] == stro_ds: #main stream
                nodes.append(idx_us)
            elif stream_order_flat[idx_us] > 0 and stream_order_flat[idx_us] < 255: # works for int8 and uint8
                tribs.append(idx_us)
        if nodes[-1] == idx_ds:
            break
        idx_ds = nodes[-1]
    return np.array(nodes), np.array(tribs)

@njit
def river_nodes(idx_ds, flwdir_flat, stream_order_flat, shape):
    idx_ds = np.asarray(idx_ds) # make sure idx_ds is an array
    rivs = list()
    stro = list()
    while True:
        idx_next = list()
        for idx in idx_ds:
            stro0 = stream_order_flat[idx]
            riv, tribs = _streamsegment(idx, stro0, flwdir_flat, stream_order_flat, shape)
            if len(riv) > 1:
                rivs.append(riv)
                stro.append(stro0)
            idx_next.extend(tribs)
        if len(idx_next) == 0:
            break
        idx_ds = np.array(idx_next, idx_ds.dtype)
    return rivs, np.array(stro, dtype=stream_order_flat.dtype)