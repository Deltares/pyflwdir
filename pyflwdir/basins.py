# -*- coding: utf-8 -*-
"""Methods to delineate (sub)basins."""
from numba import njit
import numpy as np

from pyflwdir import core

_mv = core._mv
all = []


@njit
def fillnodata_upstream(idxs_ds, seq, data, nodata):
    """Retuns a a copy of <data> where upstream cell with <nodata> values are filled
    based on the first downstream valid cell value.

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    data : 1D array
        original data with missing values
    nodata : float, integer
        nodata value

    Returns
    -------
    data_out: 1D array of data.dtype
        infilled data
    """
    data_out = data.copy()
    for idx0 in seq:  # down- to upstream
        if data_out[idx0] == nodata and data_out[idxs_ds[idx0]] != nodata:
            data_out[idx0] = data_out[idxs_ds[idx0]]
    return data_out


def basins(idxs_ds, idxs_pit, seq, ids=None):
    """Return basin map"""
    if ids is None:
        ids = np.arange(1, idxs_pit.size + 1, dtype=np.uint32)
    basins = np.zeros(idxs_ds.size, dtype=ids.dtype)
    basins[idxs_pit] = ids
    return fillnodata_upstream(idxs_ds, seq, basins, 0)


@njit
def contiguous_area_within_region(idxs_ds, seq, region_mask, stream_mask=None):
    """Returns most downstream contiguous area within region, i.e.: if a stream flows
    in and out of the region, only the most downstream contiguous area within region
    will be True in output mask. If a stream mask is provided the area is reduced to
    cells which drain to the stream.

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    region_mask : 1D array of bool
        mask of region
    stream_mask : 1D array of bool, optional
        mask of stream

    Returns
    -------
    mask: 1D array of bool
        Mask of most downstream contiguous area within region
    """
    # get area upstream of streams within region
    if stream_mask is not None:
        mask = stream_mask.copy()
    else:
        mask = np.array([np.bool(1) for _ in range(region_mask.size)])  # all True
    # keep only the most downstream contiguous area within region
    for idx in seq:  # down- to upstream
        idx_ds = idxs_ds[idx]
        mask[idx] = mask[idx_ds]  # propagate mask upstream
        if region_mask[idx] == False and region_mask[idx_ds]:  # leaving region
            mask[idx] = False
    return np.logical_and(mask, region_mask)


@njit
def subbasin_mask_within_region(idxs_ds, seq, region_mask, stream_mask=None):
    """Returns a mask of subbasins within a region, i.e. basins with upstream cells
    outside the region are excluded. If a stream mask is provided the area is reduced
    to cells which drain to the stream."""
    # get area upstream of streams within region
    if stream_mask is not None:
        mask = np.logical_and(region_mask, stream_mask)
        for idx in seq:  # down- to upstream
            mask[idx] = mask[idxs_ds[idx]]
    else:
        mask = region_mask.copy()
    # keep only subbasins (areas with no upstream cells outside region)
    for idx in seq[::-1]:  # up- to downstream
        idx_ds = idxs_ds[idx]
        if region_mask[idx_ds] == False:  # outside region
            mask[idx_ds] = False
        else:
            mask[idx_ds] = mask[idx]  # propagate mask downstream
    return mask


@njit
def subbasins(idxs_ds, seq, strord, min_sto=0):
    """Returns a subbasin map with unique IDs starting from one.
    Subbasins are defined based on a minimum stream order.

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    strord : 1D-array of uint8
        stream order
    min_sto : int, optional
        minimum stream order of subbasins, by default the stream order is set to
        two under the global maxmium stream order.

    Returns
    -------
    basins : 1D-arrays of uint32
        map with unique IDs for stream_order>=min_sto subbasins
    """
    if min_sto == 0:
        min_sto = strord.max() - 2
    _done = np.array([np.bool(0) for _ in range(strord.size)])  # initiate False array
    basins = np.full(idxs_ds.size, 0, dtype=np.uint32)
    i = np.uint32(1)
    for idx0 in seq[::-1]:  # up- to downstream
        if _done[idx0] or strord[idx0] < min_sto:
            continue
        sto = strord[idx0]
        while True:
            idx_ds = idxs_ds[idx0]
            sto_ds = strord[idx_ds]
            if idx0 == idx_ds or sto_ds > sto:  # pit or new stream
                basins[idx0] = i
                i += 1
                break
            else:
                _done[idx_ds] = True
                idx0 = idx_ds
    return fillnodata_upstream(idxs_ds, seq, basins, 0)


# TODO check
# @njit # NOTE does not work atm with dicts (numba 0.48)
def pfafstetter(idxs_pit, idxs_ds, seq, uparea, upa_min=0, depth=1, mv=_mv):
    """pfafstetter subbasins coding

    Verdin K . and Verdin J . 1999 A topological system for delineation
    and codification of the Earth’s river basins J. Hydrol. 218 1–12
    Online: https://linkinghub.elsevier.com/retrieve/pii/S0022169499000116
    """
    #
    idxs_us_main = core.main_upstream(idxs_ds, uparea, upa_min, mv=mv)
    idxs_us_trib = core.main_tributary(idxs_ds, idxs_us_main, uparea, upa_min, mv=mv)
    #
    upa_min = np.atleast_1d(upa_min)
    pfaf = np.zeros(uparea.size, dtype=np.uint32)
    # initialize
    for idx0 in idxs_pit:
        pfafs = np.array([0], dtype=pfaf.dtype)
        pfaf_dict = dict()
        idx1 = mv
        for d in range(depth):
            pfaf_lst_next = []
            min_upa0 = upa_min[min(upa_min.size - 1, d)]
            for base in pfafs:
                if d > 0:
                    i = base % 10 - 1
                    pfafid = base * 10
                    idxs0 = pfaf_dict[base // 10]
                    idx0 = idxs0[i]
                    idx1 = idxs0[i + 2] if i % 2 == 0 and i < idxs0.size - 1 else mv
                else:
                    pfafid = 0
                idxs_sub = core._tributaries(
                    idx0, idxs_us_main, idxs_us_trib, uparea, idx1, min_upa0, 4, mv=mv
                )
                idxs_sub = idxs_sub[idxs_sub != mv]
                if idxs_sub.size > 0:
                    idxs_inter = idxs_us_main[idxs_ds[idxs_sub]]
                    idxs1 = [idx0]
                    for i in range(idxs_sub.size):
                        idxs1.append(idxs_sub[i])
                        idxs1.append(idxs_inter[i])
                    idxs1 = np.array(idxs1)
                    pfaf_lst = [pfafid + k for k in range(1, idxs1.size + 1)]
                    pfaf_lst_next.extend(pfaf_lst)
                    pfaf[idxs1] = np.array(pfaf_lst, dtype=pfaf.dtype)
                    pfaf_dict[base] = idxs1
                elif d > 0:
                    pfaf[idx0] = pfafid + 1
            pfafs = np.array(pfaf_lst_next, dtype=pfaf.dtype)
    return fillnodata_upstream(idxs_ds, seq, pfaf, 0)
