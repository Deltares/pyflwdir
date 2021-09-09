# -*- coding: utf-8 -*-
"""Core flow direction functionality. All functions work based on the an array of 
next downstream indices (idxs_ds) and mostly return indices."""

from numba import njit, prange
from numba.typed import List
import numpy as np
import math
import heapq

from pyflwdir import gis_utils

__all__ = []
_mv = np.intp(-1)

# flwdir properties


@njit
def rank(idxs_ds, mv=_mv):
    """Returns the rank, i.e. the distance counted in number of cells from the outlet."""
    ranks = np.full(idxs_ds.size, -9999, dtype=np.int32)
    n = 0
    idxs_lst = []
    for idx0 in range(idxs_ds.size):
        idx_ds = idxs_ds[idx0]
        if idx_ds == mv or ranks[idx0] != -9999:
            continue
        idxs_lst.append(idx0)
        while True:
            rnk = ranks[idx_ds]
            if rnk >= 0:
                break
            elif idx_ds == idx0:  # pit
                rnk = np.int32(-1)
                break
            elif rnk == -1 or idx_ds in idxs_lst:  # loop -> mark with -1
                while len(idxs_lst) > 0:
                    ranks[idxs_lst.pop(-1)] = -1
                break
            # next iter
            idx0 = idx_ds
            idxs_lst.append(idx0)
            idx_ds = idxs_ds[idx0]
        while len(idxs_lst) > 0:
            rnk += 1
            n += 1
            ranks[idxs_lst.pop(-1)] = rnk
    return ranks, n


@njit
def upstream_count(idxs_ds, mv=_mv):
    """Returns array with number of upstream cells per cell."""
    n_up = np.full(idxs_ds.size, -9, dtype=np.int8)
    for idx0 in range(idxs_ds.size):
        idx_ds = idxs_ds[idx0]
        if idx_ds != mv:
            n_up[idx0] = max(n_up[idx0], 0)
            if idx0 != idx_ds:  # pit
                n_up[idx_ds] = max(n_up[idx_ds], 0) + 1
    return n_up


# returns 2D array (n, d) with indices


@njit
def upstream_matrix(idxs_ds, mv=_mv):
    """Returns a 2D array with upstream cell indices for each cell.
    The shape of the array is (idxs_ds.size, max number of upstream cells per cell).
    """
    n_up = upstream_count(idxs_ds, mv=mv)
    d = int(np.max(n_up))
    n = idxs_ds.size
    # 2D arrays of upstream index
    idxs_us = np.full((n, d), mv, dtype=idxs_ds.dtype)
    n_up[:] = 0
    for idx0 in range(n):
        idx_ds = idxs_ds[idx0]
        if idx_ds != idx0 and idx_ds != mv:
            i = n_up[idx_ds]
            idxs_us[idx_ds, i] = idx0
            n_up[idx_ds] += 1
    return idxs_us


@njit
def idxs_seq(idxs_ds, idxs_pit, shape, mv=_mv):
    """Returns indices ordered from down- to upstream.

    Parameters
    ----------
    idxs_ds, idxs_pit : 1D-array of int
        linear index of next downstream, pit cell

    Returns
    -------
    idxs_seq : ndarray of int, optional
        linear indices of valid cells ordered from down- to upstream
    """
    i, j = 0, 0
    idxs_us = upstream_matrix(idxs_ds, mv=mv)
    idxs_seq = np.full(idxs_ds.size, mv, idxs_ds.dtype)
    for idx in idxs_pit:
        idxs_seq[j] = idx
        j += 1
    while i < idxs_seq.size:
        idx0 = idxs_seq[i]
        if idx0 == mv:
            break
        for idx in idxs_us[idx0, :]:
            if idx == mv:
                break
            idxs_seq[j] = idx
            j += 1
        i += 1
    return idxs_seq[:i]


# returns 1D array (size == n) with indices at all locations


@njit
def main_upstream(idxs_ds, uparea, upa_min=0.0, mv=_mv):
    """Returns the index of the upstream cell with the largest uparea,
    -1 if no upstream cells (i.e. at headwater).

    Parameters
    ----------
    idxs_ds : 1D-array of int
        index of next downstream cell
    uparea : 1D-array
        upstream area
    upa_min : float, optional
        minimum upstream area threshold

    Returns
    -------
    1D-array of int
        main upstream indices
    """
    idxs_us_main = np.full(idxs_ds.size, mv, dtype=idxs_ds.dtype)
    upa_main = np.full(idxs_ds.size, upa_min, dtype=uparea.dtype)
    for idx0 in range(idxs_ds.size):
        idx_ds = idxs_ds[idx0]
        if idx_ds == idx0 or idx_ds == mv:  # pit or mv
            continue
        elif uparea[idx0] > upa_main[idx_ds]:
            idxs_us_main[idx_ds] = idx0
            upa_main[idx_ds] = uparea[idx0]
    return idxs_us_main


@njit
def main_tributary(idxs_ds, idxs_us_main, uparea, upa_min=0.0, mv=_mv):
    """Returns the index of the upstream cell with
    the second largest upstream area, i.e. the largest tributary.

    Parameters
    ----------
    idxs_ds : 1D-array of int
        index of next downstream cell
    idxs_us_main : 1D-array of int
        index of main upstream cell
    uparea : 1D-array
        upstream area values
    upa_min : float, optional
        minimum upstream area threshold

    Returns
    -------
    1D-array of int
        linear indices of tributaries
    """
    idxs_us_trib = np.full(idxs_ds.size, mv, dtype=idxs_ds.dtype)
    upa_main = np.full(idxs_ds.size, upa_min, dtype=uparea.dtype)
    for idx0 in range(idxs_ds.size):
        idx_ds = idxs_ds[idx0]
        # pit or mv or main upstream
        if idx_ds == idx0 or idx_ds == mv or idxs_us_main[idx_ds] == idx0:
            continue
        elif uparea[idx0] > upa_main[idx_ds]:
            idxs_us_trib[idx_ds] = idx0
            upa_main[idx_ds] = uparea[idx0]
    return idxs_us_trib


# returns 1D array (size < n) with indices of specific locations


@njit
def pit_indices(idxs_ds):
    """Returns pit indices, i.e. cells with no downstream cell"""
    idx_lst = []
    for idx0 in range(idxs_ds.size):
        if idx0 == idxs_ds[idx0]:
            idx_lst.append(idx0)
    return np.array(idx_lst, dtype=idxs_ds.dtype)


@njit
def loop_indices(idxs_ds, mv=_mv):
    """Returns indices loop cells, i.e. cells which do not have a pit at its most"""
    idxs = []
    ranks = rank(idxs_ds, mv)[0]
    for idx0 in range(idxs_ds.size):
        if ranks[idx0] == -1:
            idxs.append(idx0)
    return np.array(idxs, dtype=idxs_ds.dtype)


@njit
def headwater_indices(idxs_ds, mv=_mv):
    """Returns indices of headwater cells, i.e. cells with no upstream neighbors"""
    idxs = []
    counts = upstream_count(idxs_ds, mv)
    for idx0 in range(idxs_ds.size):
        if counts[idx0] == 0:
            idxs.append(idx0)
    return np.array(idxs, dtype=idxs_ds.dtype)


@njit
def flwdir_tuples(idxs_ds, mask=None, mv=_mv):
    """Returns list of up- and downstream linear index couples."""
    idxs = []
    for idx0 in range(idxs_ds.size):
        idx_ds = idxs_ds[idx0]
        if idx_ds == mv or (mask is not None and mask[idx0] != 1):
            continue
        idxs.append(np.array([idx0, idx_ds], dtype=idxs_ds.dtype))
    return idxs


# local functions


@njit
def _d8_idx(idx0, shape):
    """Returns linear indices of eight neighboring cells"""
    nrow, ncol = shape
    # assume c-style row-major
    r = idx0 // ncol
    c = idx0 % ncol
    idxs_lst = list()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:  # skip pit -> return empty array
                continue
            r_us, c_us = r + dr, c + dc
            if r_us >= 0 and r_us < nrow and c_us >= 0 and c_us < ncol:  # check bounds
                idx = r_us * ncol + c_us
                idxs_lst.append(idx)
    return np.array(idxs_lst)


@njit
def _upstream_d8_idx(idx0, idxs_ds, shape):
    """Returns a numpy array with linear indices of upstream neighbors.
    NOTE: This method only works for D8 type of flow direciton data. If upstream
    neighbours our outside the dirict 8 neighbors the returned array  will be
    incomplete."""
    idxs_lst = list()
    for idx in _d8_idx(idx0, shape):
        if idxs_ds[idx] == idx0:
            idxs_lst.append(idx)
    return np.array(idxs_lst, dtype=idxs_ds.dtype)


@njit
def _trace(
    idx0,
    idxs_nxt,
    ncol=None,
    mask=None,
    max_length=None,
    real_length=False,
    latlon=False,
    transform=gis_utils.IDENTITY,
    mv=_mv,
):
    """Returns indices of downstream cells, including the start cell, until:
    - a pit (downstream) / no upstream cell is found (upstream)
    - a True cell is found in mask OR
    - the distance from the start point is larger than max_length.

    Parameters
    ----------
    idx0 : int
        linear index of start cells
    idxs_nxt : 1D-array of int
        linear indices of downstream or main upstream cells
    ncol : int
        number of columns in raster
    mask : 1D-array of bool, optional
        True if stream cell
    max_length : float, optional
        maximum distance to move downstream, by default None
    real_length : bool, optional
        unit of length in meters if True, cells if False, by default False
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY

    Returns
    -------
    1D-array of int
        linear indices of trace
    float
        distance between start and end cell
    """
    idxs = []
    idxs.append(idx0)
    dist = 0.0
    d = 1.0
    while mask is None or (mask is not None and mask[idx0] == False):
        idx1 = idxs_nxt[idx0]
        if idx1 == idx0 or idx1 == mv:  # pit no more upstream cells
            break
        if real_length and ncol is not None:
            d = gis_utils.distance(idx0, idx1, ncol, latlon, transform)
        if max_length is not None and dist + d > max_length:
            break
        dist += d
        idx0 = idx1
        idxs.append(idx0)
    return np.array(idxs, dtype=idxs_nxt.dtype), dist


@njit
def _window(idx0, n, idxs_ds, idxs_us_main, mv=_mv):
    """Returns the indices of between the nth upstream to nth downstream cell from
    the current cell. Upstream cells are with based on the  _main_upstream method."""
    idxs = np.full(n * 2 + 1, mv, idxs_ds.dtype)
    idxs[n] = idx0
    # get n downstream cells
    for i in range(n):
        idx_ds = idxs_ds[idx0]
        if idx_ds == idx0:  # pit
            break
        idx0 = idx_ds
        idxs[n + i + 1] = idx0
    # get n upstreams cells
    idx0 = idxs[n]
    for i in range(n):
        idx_us = idxs_us_main[idx0]
        if idx_us == mv:  # at headwater / no upstream cells
            break
        idx0 = idx_us
        idxs[n - i - 1] = idx0
    return idxs


@njit
def _tributaries(
    idx0, idxs_us_main, idxs_us_trib, uparea, idx_end=_mv, upa_min=0.0, n=0, mv=_mv
):
    """Return indices of tributaries upstream from idx0 and downstream
    from idx_end.

    Parameters
    ----------
    idx0 : int
        linear index of start cell
    idxs_us_main, idxs_us_trib : 1D-array of int
        linear indices of main upstream and tributary cells
    uparea : 1D-array
        array with upstream area values
    idx_end : int, optional
        linear index of most upstream, by default missing value, i.e.: no fixed cell
    upa_min : float, optional
        minimum upstream area threshold for tributary
    n : int, optional
        number of tributaries, by default 0

    Returns
    -------
    1D array of int with size n
        linear indices of largest tributaries
    """
    if n > 0:
        # use heapq to keep n largest; initialize with correct dtypes
        i = int(0)
        ntrib = [(uparea[0], idxs_us_trib[0], i) for _ in range(n)]
    else:
        idxs = []
    # move upstream while checking tributaries
    while True:
        idx_main = idxs_us_main[idx0]
        if idx_main == idx_end or idx_main == mv:
            break
        elif uparea[idx_main] < upa_min:
            break
        idx_trib = idxs_us_trib[idx0]
        if idx_trib != mv:
            upa_trib = uparea[idx_trib]
            if upa_trib > upa_min:
                if n > 0:
                    i += 1
                    heapq.heappushpop(ntrib, (upa_trib, idx_trib, i))
                    upa_min = max(upa_min, heapq.nsmallest(1, ntrib)[0][0])
                else:
                    idxs.append(idx_trib)
        # next iter
        idx0 = idx_main
    # order from down to upstream
    if n > 0:
        seq = np.argsort(np.array([ntrib[i][-1] for i in range(n)]))
        idxs_out = np.array([ntrib[i][1] for i in seq], dtype=idxs_us_main.dtype)
    else:
        idxs_out = np.array(idxs, dtype=idxs_us_main.dtype)
    return idxs_out


@njit
def path(
    idxs0,
    idxs_nxt,
    ncol=None,
    mask=None,
    max_length=None,
    real_length=False,
    latlon=False,
    transform=gis_utils.IDENTITY,
    mv=_mv,
):
    """See _trace method, except this function works for a 1D-array linear indices.

    Returns
    -------
    list of 1D-array of int
        linear indices of path
    1D-array of float
        distance between start and end cell
    """
    paths = List()
    dists = np.zeros(idxs0.size, dtype=np.float64)
    for i in range(idxs0.size):
        path, d = _trace(
            idxs0[i],
            idxs_nxt,
            ncol=ncol,
            mask=mask,
            max_length=max_length,
            real_length=real_length,
            latlon=latlon,
            transform=transform,
            mv=mv,
        )
        paths.append(path)
        dists[i] = d
    return paths, dists


@njit
def snap(
    idxs0,
    idxs_nxt,
    ncol=None,
    mask=None,
    max_length=None,
    real_length=False,
    latlon=False,
    transform=gis_utils.IDENTITY,
    mv=_mv,
):
    """Returns indices the most down-/upstream cell where mask is True or is pit.

    See _trace method for parameters, except this function works based on a
    1D-array linear indices.

    Returns
    -------
    1D-array of int
        linear indices destination cells
    1D-array of float
        distance between start and end cell
    """
    idxs = np.full(idxs0.size, mv, dtype=idxs0.dtype)
    dists = np.zeros(idxs0.size, dtype=np.float32)
    for i in range(idxs0.size):
        path, d = _trace(
            idxs0[i],
            idxs_nxt,
            ncol=ncol,
            mask=mask,
            real_length=real_length,
            max_length=max_length,
            latlon=latlon,
            transform=transform,
            mv=mv,
        )
        idxs[i] = path[-1]
        dists[i] = d
    return idxs, dists


@njit
def inflow_idxs(idxs_ds, seq, region):
    """returns linear indices of most upstream pixels within region"""
    idxs = []
    mask = np.array([bool(1) for _ in range(idxs_ds.size)])  # all True
    for idx0 in seq[::-1]:  # up- to downstream
        idx_ds = idxs_ds[idx0]
        if idx0 != idx_ds:
            if mask[idx0] and region[idx_ds] and not region[idx0]:  # in
                idxs.append(idx0)
                mask[idx_ds] = False
            else:
                mask[idx_ds] = mask[idx0]
    return np.array(idxs, dtype=idxs_ds.dtype)


@njit
def outflow_idxs(idxs_ds, seq, region):
    """returns linear indices of most downstream pixels within region"""
    idxs = []
    mask = np.array([bool(1) for _ in range(idxs_ds.size)])  # all True
    for idx0 in seq:  # down- to upstream
        idx_ds = idxs_ds[idx0]
        # at mask and region and (pit or out)
        if mask[idx_ds] and region[idx0] and (idx_ds == idx0 or not region[idx_ds]):
            idxs.append(idx0)
            mask[idx0] = False
        else:
            mask[idx0] = mask[idx_ds]
    return np.array(idxs, dtype=idxs_ds.dtype)
