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


@njit
def smooth_river_slope(smooth_length, flwdir_flat, rivlen_flat, rivslp_flat, uparea_flat, shape, upa_min=0.5):
    """smooth river slope """
    rivslp_new = np.ones(shape[0]*shape[1], dtype=rivslp_flat.dtype)
    for idx in range(flwdir_flat):
        dd = flwdir_flat[idx]
        l = rivlen_flat[idx]
        if l <= 0 or dd == fd._nodata:
            rivslp_new[idx] = rivslp_flat[idx]
        else:
            len_lst = [l]
            slp_lst = [rivslp_flat[idx]]
            idx_lst = [idx]
            i = 0
            us, ds = 0, 0
            while True:
                idx = -1
                # add alternating up and downstream values 
                if us >= 0 and (ds < 0 or i // 2 == 0):
                    idxs_us, _ = fd.us_main_indices(idx_lst[us], flwdir_flat, uparea_flat=uparea_flat, shape=shape, upa_min=upa_min)
                    if idxs_us.size > 0:
                        idx = idxs_us[0]
                        us = i
                    else:
                        us = -1
                elif ds >= 0:
                    idx_ds = fd.ds_index(idx_lst[ds], flwdir_flat, shape)
                    if idx_ds != -1 and idx_ds != idx:
                        idx = idx_ds
                        ds = i
                    else:
                        ds = -1 
                else:
                    break
                # append index/ lenght / slope lists
                if idx >= 0:
                    idx_lst.append(idx)
                    len_lst.append(rivlen_flat[idx])
                    slp_lst.append(rivslp_flat[idx])
                    l += len_lst[-1]
                    if l > smooth_length: 
                        break
                i += 1
            # calculate new slope if new data                
            if len(idx_lst) > 1:
                lens = np.array(len_lst)
                slps = np.array(slp_lst)
                rivslp_new[idx] = (lens*slps)/np.sum(lens)
    return rivslp_new.reshape(shape)