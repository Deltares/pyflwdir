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
                accu_flat[idx0] += accu_flat[idx_us]  # accumulate material
    return accu_flat


@njit()
def upstream_area(
    tree, idxs_us, idxs_dense, ncol, affine=gis_utils.IDENTITY, latlon=False
):
    """accumulated upstream area using network tree and 
    grid affine transform; nodata = -9999"""
    # NOTE same as accuflux but works with transform to calculate area
    xres, yres, north = affine[0], affine[4], affine[5]
    area0 = abs(xres * yres)
    size = idxs_us.shape[0]
    # intialize with correct dtypes
    upa_ds = np.float64(0.0)
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
                upa_ds += upa[idx_us]
            upa[idx0] = np.float64(upa_ds)
    return upa


@njit
def fillnodata_upstream(tree, idxs_us, data_sparse, nodata):
    """label basins using network tree"""
    for i in range(len(tree)):
        idxs = tree[i]  # from down- to upstream
        for idx0 in idxs:
            for idx_us in idxs_us[idx0, :]:
                if idx_us == _mv:
                    break
                if data_sparse[idx_us] == nodata:
                    data_sparse[idx_us] = data_sparse[idx0]
    return data_sparse


@njit
def basins(tree, idxs_us, idxs_pit, ids):
    """label basins using network tree"""
    size = idxs_us.shape[0]
    basins = np.zeros(size, dtype=ids.dtype)
    basins[idxs_pit] = ids
    return fillnodata_upstream(tree, idxs_us, basins, 0)


@njit
def _strahler_order(strord_us):
    """Returns Strahler stream order based on array of upstream stream orders"""
    ord_max = np.int8(1)
    ord_cnt = 0
    for i in range(strord_us.size):
        ordi = strord_us[i]
        if ordi >= ord_max:
            if ordi == ord_max:
                ord_cnt += 1
            else:
                ord_max = ordi
                ord_cnt = 1
    # where two channels of order i join, a channel of order i+1 results
    if ord_cnt >= 2:
        ord_max += 1
    return ord_max


@njit
def stream_order(tree, idxs_us):
    """"determine stream order using network tree; nodata = -1"""
    size = idxs_us.shape[0]
    strord_flat = np.full(size, -1, dtype=np.int8)
    for i in range(len(tree)):
        idxs = tree[-i - 1]  # from up- to downstream
        for idx0 in idxs:
            idxs_us0 = core.upstream(idx0, idxs_us)
            ordi = _strahler_order(strord_flat[idxs_us0])
            if ordi > 127:
                raise TypeError("maximum stream order is 127")
            strord_flat[idx0] = np.int8(ordi)  # update stream order downstream cells
    return strord_flat


# TODO
def dist_to_mouth(
    tree, idxs_us, idxs_dense, ncol, affine=gis_utils.IDENTITY, latlon=False
):
    pass


def hand(tree, idxs_us):
    pass


# @njit # NOTE does not work atm with dicts (numba 0.48)
def pfafstetter(tree, idxs_us, uparea_sparse, min_upa, depth=1):
    """pfafstetter coding for single basin
    
    Verdin K . and Verdin J . 1999 A topological system for delineation 
    and codification of the Earth’s river basins J. Hydrol. 218 1–12 
    Online: https://linkinghub.elsevier.com/retrieve/pii/S0022169499000116
    """
    min_upa = np.atleast_1d(min_upa)
    pfaf = np.zeros(uparea_sparse.size, dtype=np.uint32)
    # initialize
    pfafs = np.array([0], dtype=pfaf.dtype)
    pfaf_dict = dict()
    idx1 = _mv
    for d in range(depth):
        pfaf_lst_next = []
        min_upa0 = min_upa[min(min_upa.size - 1, d)]
        for base in pfafs:
            if d > 0:
                i = base % 10 - 1
                pfafid = base * 10
                idxs0 = pfaf_dict[base // 10]
                idx0 = idxs0[i]
                idx1 = idxs0[i + 2] if i % 2 == 0 and i < idxs0.size - 1 else _mv
            else:
                idx0 = tree[0][0]
                pfafid = 0
            _, idxs1 = core.main_tibutaries(
                idx0, idxs_us, uparea_sparse, idx1, min_upa0, 4
            )
            N = idxs1[idxs1 != _mv].size
            if N >= 3:  # at least three subbasins (outlet, sub, top)
                if N < 9:
                    idxs1 = idxs1[:N]
                pfaf_lst = [pfafid + k for k in range(1, N + 1)]
                pfaf_lst_next.extend(pfaf_lst)
                pfaf[idxs1] = np.array(pfaf_lst, dtype=pfaf.dtype)
                pfaf_dict[base] = idxs1
        pfafs = np.array(pfaf_lst_next, dtype=pfaf.dtype)
    return fillnodata_upstream(tree, idxs_us, pfaf, np.uint32(0))
