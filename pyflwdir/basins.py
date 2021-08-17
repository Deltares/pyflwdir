# -*- coding: utf-8 -*-
"""Methods to delineate (sub)basins."""
from numba import njit
import numpy as np

from pyflwdir import core

_mv = core._mv
all = []


def basins(idxs_ds, idxs_pit, seq, ids=None):
    """Return basin map"""
    if ids is None:
        ids = np.arange(1, idxs_pit.size + 1, dtype=np.uint32)
    basins = np.zeros(idxs_ds.size, dtype=ids.dtype)
    basins[idxs_pit] = ids
    return core.fillnodata_upstream(idxs_ds, seq, basins, 0)


@njit
def interbasin_mask(idxs_ds, seq, region, stream=None):
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
    region : 1D array of bool
        mask of region
    stream : 1D array of bool, optional
        mask of stream

    Returns
    -------
    mask: 1D array of bool
        Mask of most downstream contiguous area within region
    """
    # get area upstream of streams within region
    if stream is not None:
        mask = stream.copy()
        # make sure all mask contains most downstream stream cells
        for idx0 in seq[::-1]:  # up- to downstream
            if mask[idx0]:
                mask[idxs_ds[idx0]] = True
    else:
        mask = np.array([bool(1) for _ in range(region.size)])  # all True
    # keep only the most downstream contiguous area within region
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        mask[idx0] = mask[idx_ds]
        if not region[idx0] and region[idx_ds]:
            # set mask to false in first cell(s) upstream of region
            mask[idx0] = False
        # propagate mask upstream
    return np.logical_and(mask, region)


# @njit
# def headwater_mask(idxs_ds, seq, region, stream=None):
#     """Returns a mask of headwater subbasins within a region, i.e. basins with upstream
#     cells outside the region are excluded. If a stream mask is provided the area is reduced
#     to cells which drain to the stream."""
#     # get area upstream of streams within region
#     if stream is not None:
#         mask = np.logical_and(region, stream)
#         for idx in seq:  # down- to upstream
#             mask[idx] = mask[idxs_ds[idx]]
#     else:
#         mask = region.copy()
#     # keep only subbasins (areas with no upstream cells outside region)
#     for idx in seq[::-1]:  # up- to downstream
#         idx_ds = idxs_ds[idx]
#         if region[idx_ds] == False:  # outside region
#             mask[idx_ds] = False
#         else:
#             mask[idx_ds] = mask[idx]  # propagate mask downstream
#     return mask


@njit
def subbasins_streamorder(idxs_ds, seq, strord, mask=None, min_sto=-2):
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
    mask : 1D array of bool, optional
        consider only True cells
    min_sto : int, optional
        minimum stream order of subbasins, by default the stream order is set to
        two under the global maxmium stream order.

    Returns
    -------
    basins : 1D-arrays of uint32
        map with unique IDs for stream_order>=min_sto subbasins
    """
    if min_sto < 0:
        min_sto = strord.max() + min_sto
    subbas = np.full(idxs_ds.shape, 0, dtype=np.int32)
    i = np.int32(1)
    for idx0 in seq[::-1]:  # up- to downstream
        if (mask is not None and mask[idx0] == False) or strord[idx0] < min_sto:
            continue
        idx_ds = idxs_ds[idx0]
        if strord[idx0] != strord[idx_ds] or idx_ds == idx0:
            subbas[idx0] = i
            i += 1
    return core.fillnodata_upstream(idxs_ds, seq, subbas, 0)


# TODO check
# @njit # NOTE does not work atm with dicts (numba 0.48)
def subbasins_pfafstetter(idxs_pit, idxs_ds, seq, uparea, upa_min=0, depth=1, mv=_mv):
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
    return core.fillnodata_upstream(idxs_ds, seq, pfaf, 0)
