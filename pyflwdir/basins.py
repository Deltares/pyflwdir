# -*- coding: utf-8 -*-
"""Methods to delineate (sub)basins."""
from numba import njit
import numpy as np

from pyflwdir import core, streams

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


@njit
def _tributaries(idxs_ds, seq, strord):
    idxs_trib = []
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        if strord[idx0] > 0 and strord[idx0] > strord[idx_ds]:
            idxs_trib.append(idx0)
    return np.array(idxs_trib, idxs_ds.dtype)


@njit
def subbasins_pfafstetter(
    idxs_pit, idxs_ds, seq, idxs_us_main, uparea, mask=None, depth=1, mv=_mv
):
    strord = streams.stream_order(idxs_ds, seq, idxs_us_main, mask=mask, mv=mv)
    strord = np.where(strord <= depth + 1, strord, 0).astype(strord.dtype)
    idxs_trib = _tributaries(idxs_ds, seq, strord)
    # initiate map with pfaf id at river branch based on classic stream order map
    pfaf_branch = np.zeros(idxs_ds.size, np.uint32)
    # keep basin label; depth; outlet index
    labs = [(int(0), int(0)) for _ in range(0)]  # set dtypes
    # propagate basin labels upstream its main stem
    pfaf0 = 1
    for d0 in range(1, depth):
        pfaf0 += 10 ** d0
    for i, idx in enumerate(idxs_pit):
        pfaf1 = pfaf0 + (i + 1) * 10 ** depth
        labs.append((pfaf1, 1))
        pfaf_branch[idx] = pfaf1
        while True:
            idx = idxs_us_main[idx]
            if idx == mv or strord[idx] == 0:
                break
            pfaf_branch[idx] = pfaf1
    while len(labs) > 0:
        pfaf0, d0 = labs.pop(0)
        # get tributaries to pfaf0
        idxs0 = np.array(
            [
                idx
                for idx in idxs_trib
                if pfaf_branch[idx] == 0 and pfaf_branch[idxs_ds[idx]] == pfaf0
            ],
            dtype=idxs_trib.dtype,
        )
        if idxs0.size == 0:
            continue
        # sort in descending order of subbasin uparea to get 4 largest subbasins
        idxs0s = idxs0[np.argsort(-uparea[idxs0])]
        idxs_trib0 = idxs0s[:4]
        # sort in down- to upstream order
        idxs_trib0s = idxs_trib0[np.argsort(-uparea[idxs_ds[idxs_trib0]])]
        # write label at sub- & interbasin outlets
        # pfaf_branch[idx0] = pfaf0
        pfaf_int_ds = pfaf0  # downstream interbasin
        for i, idx in enumerate(idxs_trib0s):
            idx1 = idxs_us_main[idxs_ds[idx]]  # interbasin outlet
            # propagate subbasin labels upstream its main stem
            pfaf_sub = pfaf0 + (i * 2 + 1) * 10 ** (depth - d0)  # subbasin
            pfaf_branch[idx] = pfaf_sub
            while True:
                idx = idxs_us_main[idx]
                if idx == mv or strord[idx] == 0:
                    break
                pfaf_branch[idx] = pfaf_sub
            if d0 < depth:  # next iter
                labs.append((pfaf_sub, d0 + 1))
            # propagate interbasin labels upstream main stem
            if pfaf_branch[idx1] == pfaf_int_ds:
                pfaf_int = pfaf0 + (i + 1) * 2 * 10 ** (depth - d0)  # interbasin
                pfaf_branch[idx1] = pfaf_int
                while True:
                    idx1 = idxs_us_main[idx1]
                    if idx1 == mv or pfaf_branch[idx1] != pfaf_int_ds:
                        break
                    pfaf_branch[idx1] = pfaf_int
                pfaf_int_ds = pfaf_int
                if d0 < depth:  # next iter
                    labs.append((pfaf_int, d0 + 1))
    return core.fillnodata_upstream(idxs_ds, seq, pfaf_branch, 0) % 10 ** depth
