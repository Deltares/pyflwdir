# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
from numba.typed import List
import numpy as np

# import flow direction definition
from pyflwdir import core
from pyflwdir.core import _mv
from pyflwdir import gis_utils

__all__ = ['fillnodata_upstream', 'accuflux', 'upstream_area', 'stream_order', 'setup_network']

@njit
def _nbs_us(idxs_ds, idxs, _max_depth=8):
    """get n x 8 array filled with upstream neighbors of n downstream nodes"""
    nbs_us = np.ones((idxs_ds.size, _max_depth), dtype=np.uint32)*_mv # use uint32 to save memory as these networks get large!
    valid = np.zeros(idxs_ds.size, dtype=np.int8)
    N = 1
    for i in range(idxs_ds.size):
        idxs_us = core.us_indices(idxs_ds[i], idxs) # int64, but always positive
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
def setup_network(idxs, idxs_ds, _max_depth=8):
    """set drainage direction network tree from downstream to upstream"""
    size = idxs.size
    nodes = List()              # list of arrays (n) with downstream indices
    nodes_up = List()          # list of arrays (n, m) with upstream indices; m <= 8
    # move upstream
    j = 0
    while True:
        nbs_us, valid = _nbs_us(idxs_ds, idxs, _max_depth=_max_depth)
        idx_valid = np.where(valid == np.int8(1))[0]
        if idx_valid.size==0:
            break
        elif j > 0:
            idxs_ds = idxs_ds[idx_valid]
            nbs_us = nbs_us[idx_valid,:]
        nodes.append(idxs_ds)
        nodes_up.append(nbs_us)
        # next iter
        j += 1
        # NOTE 2d boolean indexing does not work currenlty in numba; flatten first
        idxs_ds = nbs_us.ravel()
        idxs_ds = idxs_ds[idxs_ds < size]
    return nodes[::-1], nodes_up[::-1]

@njit
def _strahler_order(idxs_us, strord_flat, size):
    """"""
    head_lst  = list()
    ord_max = np.int8(1)
    ord_cnt = 0
    for i in range(idxs_us.size):
        idx_us = idxs_us[i]
        if idx_us == _mv: break
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
    """"determine stream order using network tree; nodata = -1"""
    size = np.uint32(shape[0]*shape[1])
    strord_flat = np.ones(size, dtype=np.int8)*np.int8(-1)
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j]           # NOTE: contains _mv values
            ordi, idx_head = _strahler_order(idxs_us, strord_flat, size)
            strord_flat[idx_ds] = np.int8(ordi) # update stream order downstream cells
            if idx_head.size > 0:               # update head cells
                strord_flat[idx_head] = np.int8(1)
    return strord_flat.reshape(shape)

@njit
def accuflux(rnodes, rnodes_up, material, nodata):
    """accumulate 'material' in downstream direction using network tree"""
    shape = material.shape
    material_flat = material.ravel()
    accumulated = np.ones(material.size, dtype=material.dtype)*nodata
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j] # NOTE: contains _mv values
            accumulated[idx_ds] = material_flat[idx_ds] # ds
            for idx_us in idxs_us:
                if idx_us == _mv:
                    break
                v0 = accumulated[idx_us]
                if v0 == nodata:
                    v0 = material_flat[idx_us]
                    accumulated[idx_us] = v0
                accumulated[idx_ds] += v0
    return accumulated.reshape(shape)

@njit()
def upstream_area(rnodes, rnodes_up, shape, latlon=False, affine=gis_utils.IDENTITY):
    """accumulated upstream area using network tree and grid affine transform; nodata = -9999"""
    # NOTE same as accuflux but works with transform to calculate area
    nrow, ncol = np.uint32(shape[0]), np.uint32(shape[1])
    xres, yres, north = affine[0], affine[4], affine[5]
    area0 = abs(xres*yres)
    size = nrow*ncol
    # intialize with correct dtypes
    upa_ds, upa_us = np.float64(0.), np.float64(0.)
    upa = np.ones(size, dtype=np.float32)*-9999. 
    # loop over network from up- to downstream and calc upstream area
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j] # NOTE: contains _mv values
            if latlon:
                r = np.uint32(idx_ds // ncol)
                lat = north + (r+0.5)*yres
                area0 = gis_utils.cellarea(lat, xres, yres)
            upa_ds = area0
            for idx_us in idxs_us:
                if idx_us == _mv:
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
    return upa.reshape(shape)

@njit
def fillnodata_upstream(rnodes, rnodes_up, data, nodata):
    """label basins using network tree"""
    shape = data.shape
    data_flat = data.ravel()
    for i in range(len(rnodes)):
        k = -i-1
        for j in range(len(rnodes[k])):
            idx_ds = rnodes[k][j]
            idxs_us = rnodes_up[k][j] # NOTE: contains _mv values
            for idx_us in idxs_us:
                if idx_us == _mv:
                    break
                elif data_flat[idx_us] == nodata: 
                    data_flat[idx_us] = data_flat[idx_ds]
    return data_flat.reshape(shape)