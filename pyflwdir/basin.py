# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
from numba.typed import List
import numpy as np
import heapq

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
    tree, idxs_dense, idxs_us, ncol, affine=gis_utils.IDENTITY, latlon=False
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
def _strahler_order(idxs_us, strord_flat):
    """"""
    ord_max = np.int8(1)
    ord_cnt = 0
    for i in range(idxs_us.size):
        idx_us = idxs_us[i]
        if idx_us == _mv:
            break
        ordi = strord_flat[idx_us]
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
            idxs_us0 = idxs_us[idx0, :]
            ordi = _strahler_order(idxs_us0, strord_flat)
            if ordi > 127:
                raise TypeError("maximum stream order is 127")
            strord_flat[idx0] = np.int8(ordi)  # update stream order downstream cells
    return strord_flat


@njit
def main_tibutaries(idx0, idxs_us, uparea_sparse, idx_end=_mv, min_upa=0, n=4):
    """Return indices of n largest tributaries upstream from idx0 and downstream
    from idx_end"""
    # use heapq to keep list of n largest tributaries
    # initialize with n (zero, mv, mv, i) tuples
    upa_ini = np.zeros(n, dtype=uparea_sparse.dtype)
    ntrib = [(upa_ini[i], _mv, _mv, _mv) for i in range(n)]
    heapq.heapify(ntrib)
    upa0 = max(min_upa, heapq.nsmallest(1, ntrib)[0][0])
    upa_main = upa0
    upa_main0 = uparea_sparse[idx0]
    i = np.uint32(0)  # sequence of basins
    # ordered indices of sub- and interbasins from down to upstream
    idxs = np.zeros(n * 2 + 1, dtype=idxs_us.dtype)
    idxs[0] = idx0
    # stop when no more upstream cells or upstream area on main stream
    # smaller than nth largest tributary
    while idx0 != _mv and upa_main >= upa0 and idx0 != idx_end:
        idx_us_main = _mv
        upa_main = 0.0
        # find main upstream
        for idx_us in idxs_us[idx0, :]:
            if idx_us == _mv:
                break
            upa = uparea_sparse[idx_us]
            if upa > upa_main:
                idx_us_main = idx_us
                upa_main = upa
        # check min size interbasins
        # if upa_main0 - uparea_sparse[idx0] > upa0:
        # add tributaries
        for idx_us in idxs_us[idx0, :]:
            if idx_us == _mv:
                break
            upa = uparea_sparse[idx_us]
            if upa < upa_main and upa > upa0:
                i += 1
                heapq.heappushpop(ntrib, (upa, idx_us, idx_us_main, np.uint32(i)))
                upa0 = max(min_upa, heapq.nsmallest(1, ntrib)[0][0])
                upa_main0 = uparea_sparse[idx_us_main]
        # next iter
        idx0 = idx_us_main
    seq = np.argsort(np.array([ntrib[i][-1] for i in range(n)]))
    idxs_subbasin = np.array([ntrib[i][1] for i in seq])
    idxs_interbasin = np.array([ntrib[i][2] for i in seq])
    for i in range(1, n * 2 + 1):
        pfaf_id = i + 1
        if pfaf_id % 2 == 0:  # even pfafstetter code -> subbasins
            idx0 = idxs_subbasin[pfaf_id // 2 - 1]
        else:  # odd pfafstetter code -> inter-subbasin
            idx0 = idxs_interbasin[pfaf_id // 2 - 1]
        idxs[i] = idx0
    return idxs_subbasin, idxs


# @njit # NOTE does not work atm with dicts (numba 0.48)
def pfafstetter(idx0, tree, idxs_us, uparea_sparse, min_upa, depth=1):
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
                pfafid = 0
            idxs1 = main_tibutaries(idx0, idxs_us, uparea_sparse, idx1, min_upa0, 4)[1]
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
