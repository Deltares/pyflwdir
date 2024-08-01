# -*- coding: utf-8 -*-
"""Methods to delineate (sub)basins."""
from numba import njit
import numpy as np

from . import core, streams

_mv = core._mv
all = []


def basins(idxs_ds, idxs_pit, seq, ids=None):
    """Return basin map"""
    if ids is None:
        ids = np.arange(1, idxs_pit.size + 1, dtype=np.uint32)
    basins = np.zeros(idxs_ds.size, dtype=ids.dtype)
    basins[idxs_pit] = ids
    return core.fillnodata_upstream(idxs_ds, seq, basins, 0)


# NOTE not unit tested
# TODO: change this method to derive the interbasin for a single outflow as currently
# its results are ambiguous?!
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
    basins : 1D-arrays of int32
        map with unique IDs for stream_order>=min_sto subbasins
    """
    if min_sto < 0:
        min_sto = int(strord.max()) + min_sto
    subbas = np.full(idxs_ds.shape, 0, dtype=np.int32)
    idxs = []
    for idx0 in seq[::-1]:  # up- to downstream
        if (mask is not None and mask[idx0] is False) or strord[idx0] < min_sto:
            continue
        idx_ds = idxs_ds[idx0]
        if strord[idx0] != strord[idx_ds] or idx_ds == idx0:
            idxs.append(idx0)
            subbas[idx0] = len(idxs)
    idxs1 = np.array(idxs, dtype=idxs_ds.dtype)
    return core.fillnodata_upstream(idxs_ds, seq, subbas, 0), idxs1


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
    pfaf_branch = np.zeros(idxs_ds.size, np.int32)
    idxs = []
    # keep basin label; depth; outlet index
    labs = [(int(0), int(0)) for _ in range(0)]  # set dtypes
    # propagate basin labels upstream its main stem
    pfaf0 = 1
    for d0 in range(1, depth):
        pfaf0 += 10**d0
    for i, idx in enumerate(idxs_pit):
        idxs.append(idx)
        pfaf1 = pfaf0 + (i + 1) * 10**depth
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
            idxs.append(idx)
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
            if idx1 not in idxs:
                idxs.append(idx1)
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
    idxs1 = np.array(idxs, dtype=idxs_ds.dtype)
    pfafbas = core.fillnodata_upstream(idxs_ds, seq, pfaf_branch, 0) % 10**depth
    return pfafbas, idxs1


@njit
def subbasins_area(idxs_ds, seq, idxs_us_main, uparea, area_min):
    """Returns map with basin IDs, with a minimal area of `area_min`.
    Moving upstream from the basin outlets a new subbasin starts at tributaries
    with a contributing area larger than `area_min` and new interbasins when its area
    exceeds the `area_min`.

    Returns
    -------
    basins: 2D array of int32
        raster with basin IDs
    idxs1: 1D array of int
        linear indices of subbasin outlet cells
    """
    upa_out = uparea.copy()
    subbas = np.zeros(idxs_ds.size, dtype=np.uint32)
    idxs = []
    for idx in seq:  # down- to upstream
        idx_ds = idxs_ds[idx]
        if idx_ds == idx:
            idxs.append(idx)
            subbas[idx] = len(idxs)
            continue
        upa0 = upa_out[idx_ds]
        upa = uparea[idx]
        if (upa0 - upa) > area_min and upa > area_min:
            conf = (uparea[idx_ds] - upa) > area_min
            trib = idxs_us_main[idx_ds] != idx
            if not conf or trib:
                idxs.append(idx)
                subbas[idx] = len(idxs)
                upa_out[idx] = upa
            if trib:
                idx1 = idxs_us_main[idx_ds]  # main stem
                upa_out[idx_ds] -= upa
                upa_out[idx1] = upa_out[idx_ds]
        else:
            upa_out[idx] = upa0
    idxs1 = np.array(idxs, dtype=idxs_ds.dtype)
    return core.fillnodata_upstream(idxs_ds, seq, subbas, 0), idxs1
