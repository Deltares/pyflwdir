# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
from numba.typed import List
import numpy as np

# import flow direction definition
from pyflwdir import core
from pyflwdir import gis_utils

__all__ = ['fillnodata_upstream', 'accuflux', 'upstream_area', 'stream_order', 'setup_network']

@njit
def setup_network(idxs_pits, idxs_ds, idxs_us):
    """set drainage direction network tree from downstream to upstream"""
    tree = List()
    tree.append(idxs_pits)
    idxs = idxs_pits
    # move upstream
    while True:
        nbs_us = idxs_us[idxs,:] # has _mv values
        idxs = nbs_us.ravel()    # keep valid
        idxs = idxs[idxs != core._mv]
        if idxs.size == 0:       # break if no more upstream
            break
        tree.append(idxs)        # append next leave to tree
    return tree                  # down- to upstream

@njit
def accuflux(tree, idxs_us, idxs_valid, material, nodata):
    """accumulate 'material' in downstream direction using network tree"""
    material_flat = material.ravel()[idxs_valid]
    accu_flat = np.ones(idxs_valid.size, dtype=material.dtype)*nodata
    for i in range(len(tree)):
        idxs_ds0 = tree[-i-1] # from up- to downstream
        for idx_ds in idxs_ds0:
            idxs_us0 = idxs_us[idx_ds,:] # NOTE: contains _mv values
            accu_flat[idx_ds] = material_flat[idx_ds] # ds
            for idx_us in idxs_us0:
                if idx_us == core._mv:
                    break
                v0 = accu_flat[idx_us]
                if v0 == nodata:
                    v0 = material_flat[idx_us]
                    accu_flat[idx_us] = v0
                accu_flat[idx_ds] += v0
    accu_out = np.ones(material.size, dtype=material.dtype)*nodata
    accu_out[idxs_valid] = accu_flat
    return accu_out.reshape(material.shape)

@njit()
def upstream_area(tree, idxs_us, idxs_valid, shape, latlon=False, affine=gis_utils.IDENTITY):
    """accumulated upstream area using network tree and grid affine transform; nodata = -9999"""
    # NOTE same as accuflux but works with transform to calculate area
    nrow, ncol = np.uint32(shape[0]), np.uint32(shape[1])
    xres, yres, north = affine[0], affine[4], affine[5]
    area0 = abs(xres*yres)
    # intialize with correct dtypes
    upa_ds, upa_us = np.float64(0.), np.float64(0.)
    upa = np.ones(idxs_valid.size, dtype=np.float32)*-9999. 
    # loop over network from up- to downstream and calc upstream area
    for i in range(len(tree)):
        idxs_ds0 = tree[-i-1] # from up- to downstream
        for idx_ds in idxs_ds0:
            idxs_us0 = idxs_us[idx_ds,:] # NOTE: contains _mv values
            if latlon:
                r = np.uint32(idx_ds // ncol)
                lat = north + (r+0.5)*yres
                area0 = gis_utils.cellarea(lat, xres, yres)
            upa_ds = area0
            for idx_us in idxs_us0:
                if idx_us == core._mv:
                    break
                upa_us = upa[idx_us]
                if upa_us <= 0:
                    if latlon:
                        r = np.uint32(idx_us // ncol)
                        lat = north + (r+0.5)*yres
                        area0 = gis_utils.cellarea(lat, xres, yres)
                    upa_us = area0
                    upa[idx_us] = np.float32(upa_us)
                upa_ds += upa_us
            upa[idx_ds] = np.float32(upa_ds)
    upa_out = np.ones(nrow*ncol, dtype=upa.dtype)*-9999
    upa_out[idxs_valid] = upa
    return upa_out.reshape(shape)

@njit
def fillnodata_upstream(tree, idxs_us, idxs_valid, data, nodata):
    """label basins using network tree"""
    shape = data.shape
    data_flat = data.ravel()[idxs_valid]
    for i in range(len(tree)):
        idxs_ds0 = tree[i] # from down- to upstream
        for idx_ds in idxs_ds0:
            idxs_us0 = idxs_us[idx_ds,:] # NOTE: contains _mv values
            for idx_us in idxs_us0:
                if idx_us == core._mv:
                    break
                elif data_flat[idx_us] == nodata: 
                    data_flat[idx_us] = data_flat[idx_ds]
    data_out = np.ones(shape, dtype=data.dtype)*-9999
    data_out[idxs_valid] = data_flat
    return data_out.reshape(shape)

@njit
def basins(tree, idxs_us, idxs_valid, shape):
    """label basins using network tree"""
    idxs_pits = tree[0]
    basidx_flat = np.zeros(np.multiply(*shape), dtype=np.int32)
    basidx_flat[idxs_valid[idxs_pits]] = np.arange(idxs_pits.size).astype(np.int32) + 1
    return fillnodata_upstream(tree, idxs_us, idxs_valid, data=basidx_flat, nodata=np.int32(0)).reshape(shape)

@njit
def _strahler_order(idxs_us, strord_flat):
    """"""
    head_lst  = list()
    ord_max = np.int8(1)
    ord_cnt = 0
    for i in range(idxs_us.size):
        idx_us = idxs_us[i]
        if idx_us == core._mv: 
            break
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
def stream_order(tree, idxs_us, idxs_valid, shape):
    """"determine stream order using network tree; nodata = -1"""
    strord_flat = np.ones(idxs_valid.size, dtype=np.int8)*np.int8(-1)
    for i in range(len(tree)):
        idxs_ds0 = tree[-i-1] # from up- to downstream
        for idx_ds in idxs_ds0:
            idxs_us0 = idxs_us[idx_ds,:] # NOTE: contains _mv values
            ordi, idx_head = _strahler_order(idxs_us0, strord_flat)
            strord_flat[idx_ds] = np.int8(ordi) # update stream order downstream cells
            if idx_head.size > 0:               # update head cells
                strord_flat[idx_head] = np.int8(1)
    strord = np.ones(shape[0]*shape[1], dtype=strord_flat.dtype)**np.int8(-1)
    strord[idxs_valid] = strord_flat
    return strord.reshape(shape)