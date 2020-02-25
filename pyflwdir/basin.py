# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
from numba.typed import List
import numpy as np

# import flow direction definition
from pyflwdir import core
from pyflwdir import gis_utils

_mv = core._mv


@njit
def accuflux(tree, idxs_us, material_flat, nodata):
    """accumulate 'material' in downstream direction using 
    the network tree"""
    size = idxs_us.shape[0]
    accu_flat = np.full(size, nodata, dtype=material_flat.dtype)
    for i in range(len(tree)):
        idxs = tree[-i - 1]  # from up- to downstream
        for idx0 in idxs:
            accu_flat[idx0] = material_flat[idx0]  # current cell
            for idx_us in idxs_us[idx0, :]:
                if idx_us == _mv:
                    break
                v0 = accu_flat[idx_us]
                if v0 == nodata:
                    v0 = material_flat[idx_us]
                    accu_flat[idx_us] = v0
                accu_flat[idx0] += v0  # accumulate material
    return accu_flat


@njit()
def upstream_area(
    tree, idxs_dense, idxs_us, ncol, affine=gis_utils.IDENTITY, latlon=False
):
    """accumulated upstream area using network tree and 
    grid affine transform; nodata = -9999"""
    # NOTE same as accuflux but works with transform to calculate area
    xres, yres, north = affine[0], affine[4], affine[5]
    area0 = abs(xres * yres)
    size = idxs_us.shape[0]
    # intialize with correct dtypes
    upa_ds, upa_us = np.float64(0.0), np.float64(0.0)
    upa = np.full(size, -9999.0, dtype=np.float64)
    # loop over network from up- to downstream and calc upstream area
    for i in range(len(tree)):
        idxs = tree[-i - 1]  # from up- to downstream
        for idx0 in idxs:
            if latlon:
                r = idxs_dense[idx0] // ncol
                lat = north + (r + 0.5) * yres
                area0 = gis_utils.cellarea(lat, xres, yres)
            upa_ds = area0
            for idx_us in idxs_us[idx0, :]:
                if idx_us == _mv:
                    break
                upa_us = upa[idx_us]
                if upa_us <= 0:
                    if latlon:
                        r = idxs_dense[idx_us] // ncol
                        lat = north + (r + 0.5) * yres
                        area0 = gis_utils.cellarea(lat, xres, yres)
                    upa_us = area0
                    upa[idx_us] = np.float64(upa_us)
                upa_ds += upa_us
            upa[idx0] = np.float64(upa_ds)
    return upa


@njit
def fillnodata_upstream(tree, idxs_us, data, nodata):
    """label basins using network tree"""
    for i in range(len(tree)):
        idxs = tree[i]  # from down- to upstream
        for idx0 in idxs:
            for idx_us in idxs_us[idx0, :]:
                if idx_us == _mv:
                    break
                if data[idx_us] == nodata:
                    data[idx_us] = data[idx0]
    return data


@njit
def basins(tree, idxs_us, idxs_pit):
    """label basins using network tree"""
    size = idxs_us.shape[0]
    basins = np.zeros(size, dtype=np.uint32)
    npit = idxs_pit.size
    basins[idxs_pit] = np.array([np.uint32(i + 1) for i in range(npit)])
    return fillnodata_upstream(tree, idxs_us, data=basins, nodata=np.uint32(0))


@njit
def _strahler_order(idxs_us, strord_flat):
    """"""
    head_lst = list()
    ord_max = np.int8(1)
    ord_cnt = 0
    for i in range(idxs_us.size):
        idx_us = idxs_us[i]
        if idx_us == _mv:
            break
        ordi = strord_flat[idx_us]
        if ordi <= 0:  # most upstream cells
            ordi = np.int8(1)
            head_lst.append(idx_us)
        if ordi >= ord_max:
            if ordi == ord_max:
                ord_cnt += 1
            else:
                ord_max = ordi
                ord_cnt = 1
    # where two channels of order i join, a channel of order i+1 results
    if ord_cnt >= 2:
        ord_max += 1
    return ord_max, np.array(head_lst, dtype=np.uint32)


@njit
def stream_order(tree, idxs_us):
    """"determine stream order using network tree; nodata = -1"""
    size = idxs_us.shape[0]
    strord_flat = np.full(size, -1, dtype=np.int8)
    for i in range(len(tree)):
        idxs = tree[-i - 1]  # from up- to downstream
        for idx0 in idxs:
            idxs_us0 = idxs_us[idx0, :]
            ordi, idx_head = _strahler_order(idxs_us0, strord_flat)
            if ordi > 127:
                raise TypeError("maximum stream order is 127")
            strord_flat[idx0] = np.int8(ordi)  # update stream order downstream cells
            if idx_head.size > 0:  # update head cells
                strord_flat[idx_head] = np.int8(1)
    return strord_flat
